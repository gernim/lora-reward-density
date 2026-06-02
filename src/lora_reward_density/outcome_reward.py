"""Outcome reward (regime a): rule-based correctness via math-verify.

Each prompt's metadata must include a `gold_answer` (str). The verifier parses
both the gold and the model's completion via math-verify and emits a binary
correct/incorrect reward, broadcast across the trajectory's valid tokens.
"""

from __future__ import annotations

import contextlib
import io
import logging
from dataclasses import dataclass

import torch

from lora_reward_density.rewards import RewardOutput
from lora_reward_density.rollout import RolloutBatch

logger = logging.getLogger(__name__)

GOLD_ANSWER_KEY = "gold_answer"


@dataclass(frozen=True)
class OutcomeRewardConfig:
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    parse_timeout_seconds: int = 5
    verify_timeout_seconds: int = 5


class OutcomeRewardModule:
    """Binary correctness reward parsed via math-verify."""

    name = "outcome"

    def __init__(self, config: OutcomeRewardConfig | None = None) -> None:
        self._config = config or OutcomeRewardConfig()

    def score(self, batch: RolloutBatch) -> RewardOutput:
        # Imported here so the module loads without math-verify installed,
        # matching the lazy-import pattern in rollout.VLLMRolloutEngine.
        from math_verify import parse, verify

        cfg = self._config
        n = batch.num_completions
        traj = torch.full((n,), cfg.incorrect_reward, dtype=torch.float32)
        correctness = torch.zeros(n, dtype=torch.bool)
        parse_failures = 0

        def _parse_quiet(text: str) -> list:
            # math-verify's `parsing_timeout` returns an empty list silently
            # on timeout (rather than raising) and prints
            # "Timeout during parsing: <full text>" to stderr. That stderr
            # spew is huge on degenerate completions and dominates Modal
            # logs in long batches; redirecting it keeps logs readable.
            # Callers detect the timeout via the empty-result check below.
            with contextlib.redirect_stderr(io.StringIO()):
                return parse(text, parsing_timeout=cfg.parse_timeout_seconds)

        for i, completion in enumerate(batch.completions):
            prompt_idx = int(batch.group_index[i].item())
            md = batch.prompt_metadata[prompt_idx]
            if GOLD_ANSWER_KEY not in md:
                raise KeyError(f"prompt_metadata[{prompt_idx}] missing {GOLD_ANSWER_KEY!r}")
            gold_text = md[GOLD_ANSWER_KEY]

            try:
                gold = _parse_quiet(gold_text)
                ans = _parse_quiet(completion)
                if not ans:
                    # Empty parse result = silent timeout or unparseable
                    # output. The trajectory was never going to verify as
                    # correct, but we must explicitly count it — otherwise
                    # `parse_failures` only captures exception-raising
                    # failures and silently undercounts the rest.
                    parse_failures += 1
                    ok = False
                else:
                    ok = bool(verify(gold, ans, timeout_seconds=cfg.verify_timeout_seconds))
            except Exception as e:  # noqa: BLE001 — math-verify can raise sympy/timeout errors
                logger.warning("math-verify failed on completion %d: %s", i, e)
                ok = False
                parse_failures += 1

            correctness[i] = ok
            traj[i] = cfg.correct_reward if ok else cfg.incorrect_reward

        # Outcome is the one-step case of the dense contract (DeepSeekMath
        # §4.1.3): deposit each trajectory reward at its last valid token, zeros
        # elsewhere. The loss group-normalizes the deposits and reverse-cumsums,
        # which broadcasts the normalized scalar back across the trajectory's
        # tokens — identical to the old broadcast, but unified with the process
        # regime. Reward modules emit CPU tensors (design.md §6); completion_mask
        # may live on the rollout device (GPU), so pull it to CPU first.
        mask_cpu = batch.completion_mask.cpu()
        token_rewards = torch.zeros(mask_cpu.shape, dtype=torch.float32)
        step_reward_mask = torch.zeros(mask_cpu.shape, dtype=torch.bool)
        for i in range(n):
            valid = mask_cpu[i].nonzero(as_tuple=False)
            if valid.numel() == 0:
                continue  # degenerate empty completion: no token to deposit on
            last = int(valid[-1].item())
            token_rewards[i, last] = traj[i]
            step_reward_mask[i, last] = True
        return RewardOutput(
            token_rewards=token_rewards,
            trajectory_rewards=traj,
            step_reward_mask=step_reward_mask,
            metadata={
                "correctness": correctness,
                "parse_failures": parse_failures,
                "regime": self.name,
            },
        )
