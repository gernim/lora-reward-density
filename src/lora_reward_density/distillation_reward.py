"""Distillation reward (regime c): on-policy distillation, per-token reverse KL.

The densest regime (O(N) bits/episode): every token carries a signal. Following
on-policy distillation, the per-token reward is the negative reverse-KL estimate

    r_t = log π_T(a_t | s_t) − log π_θ(a_t | s_t)

— high when the teacher likes a token more than the current student did, so
maximizing it pulls the student toward the teacher. Under the dense GRPO contract
(D11) every valid token is a *deposit*: `token_rewards[i,t] = r_t`, and
`step_reward_mask` is the completion mask. The loss group-normalizes and
reverse-cumsums these exactly as for outcome/process — no regime branch.

Two pieces are injected at construction so `score(batch)` stays signature-stable
(no Protocol / training_step change):

- **teacher**: a `TeacherClient` returning `log π_T` on the student's exact
  completion tokens. Decided (design §9.2/§9.3): the 8B teacher runs as a
  **separate Modal GPU function**, off the training device. Qwen3-8B shares the
  Qwen3 vocab with the student, so the student's `completion_token_ids` are valid
  teacher inputs (no re-tokenization).
- **policy_logprob_fn**: `log π_θ` for the *current* student. Bound by `train()`
  after `load_student` via `bind_policy(model)` — recomputed each rollout with a
  frozen-eval no-grad forward (the reward is a constant w.r.t. the gradient). This
  is the accepted reward↔policy coupling (the "least-risk" call), kept out of the
  user-owned `training_step`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from lora_reward_density.rewards import RewardOutput
from lora_reward_density.rollout import RolloutBatch

logger = logging.getLogger(__name__)

# Both return [N, T] logprobs of the completion tokens, aligned to
# `batch.completion_token_ids` / `batch.completion_mask`.
LogprobFn = Callable[[RolloutBatch], torch.Tensor]


class TeacherClient(Protocol):
    """Returns teacher logprobs `log π_T(a_t | s_t)` on the batch's completions."""

    def logprobs(self, batch: RolloutBatch) -> torch.Tensor: ...  # [N, T]


@dataclass(frozen=True)
class DistillationRewardConfig:
    teacher_model_id: str = "Qwen/Qwen3-8B"
    # Optional symmetric clip on |r_t| (nats). Per-token logprob gaps can spike on
    # rare tokens; clipping bounds the deposit magnitude before group-norm. None =
    # no clip. A decision-log knob — validate against advantage_std on a real run.
    reward_clip: float | None = None


def _distillation_token_rewards(
    teacher_logprobs: torch.Tensor,
    policy_logprobs: torch.Tensor,
    completion_mask: torch.Tensor,
    reward_clip: float | None = None,
) -> torch.Tensor:
    """Per-token `r_t = log π_T − log π_θ`, optionally clipped, zeroed at pad.

    All inputs are `[N, T]` and on the same device; returns `[N, T]` float.
    """
    r = teacher_logprobs - policy_logprobs
    if reward_clip is not None:
        r = r.clamp(-reward_clip, reward_clip)
    return r * completion_mask.float()


class DistillationRewardModule:
    """Dense per-token distillation reward (regime c). Implements `RewardModule`.

    Construct with a `teacher` client and optionally a `policy_logprob_fn`; in
    production `train()` calls `bind_policy(student)` to supply the latter. Tests
    inject both stubs and never load the 8B teacher or run a forward.
    """

    name = "distillation"

    def __init__(
        self,
        config: DistillationRewardConfig | None = None,
        teacher: TeacherClient | None = None,
        policy_logprob_fn: LogprobFn | None = None,
    ) -> None:
        self._config = config or DistillationRewardConfig()
        self._teacher = teacher
        self._policy_logprob_fn = policy_logprob_fn

    def bind_policy(self, model: object) -> None:
        """Bind the current student so `score()` can recompute `log π_θ`.

        Recomputes per rollout under frozen-eval + no-grad (like the D6 sampler
        recompute) — the reward is a constant w.r.t. the policy gradient.
        """
        from lora_reward_density.train import compute_completion_logprobs, frozen_eval

        def _policy_lp(batch: RolloutBatch) -> torch.Tensor:
            with frozen_eval(model):  # type: ignore[arg-type]  # nn.Module at runtime
                return compute_completion_logprobs(
                    model,  # type: ignore[arg-type]
                    batch.prompt_token_ids,
                    batch.completion_token_ids,
                    batch.completion_mask,
                    batch.prompt_attention_mask,
                )

        self._policy_logprob_fn = _policy_lp

    def score(self, batch: RolloutBatch) -> RewardOutput:
        if self._teacher is None:
            raise ValueError("DistillationRewardModule needs a teacher client")
        if self._policy_logprob_fn is None:
            raise ValueError(
                "DistillationRewardModule has no policy logprobs — call bind_policy(student) "
                "or pass policy_logprob_fn"
            )

        # Reward modules emit CPU tensors (design.md §6). teacher logprobs come
        # back from the Modal teacher (CPU); the policy forward runs on the
        # student's device — pull both to CPU before the subtraction.
        teacher_lp = self._teacher.logprobs(batch).detach().float().cpu()
        policy_lp = self._policy_logprob_fn(batch).detach().float().cpu()
        mask = batch.completion_mask.cpu()

        token_rewards = _distillation_token_rewards(
            teacher_lp, policy_lp, mask, self._config.reward_clip
        )
        valid = mask.sum().clamp_min(1)
        return RewardOutput(
            token_rewards=token_rewards,
            trajectory_rewards=token_rewards.sum(dim=1),  # masked sum (logging)
            step_reward_mask=mask.bool(),  # every valid token is a deposit
            metadata={
                "regime": self.name,
                "mean_token_reward": float(token_rewards.sum().item() / valid.item()),
            },
        )
