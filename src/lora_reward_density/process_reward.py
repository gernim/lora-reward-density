"""Process reward (regime b): dense per-step credit assignment via a PRM.

Following DeepSeekMath's process-supervision GRPO (§4.1.3), this regime does
**not** collapse a trajectory to one scalar. Each reasoning step gets its own
PRM score, deposited at the step's final token position in ``token_rewards``
(zeros elsewhere); the ``RewardOutput.step_reward_mask`` field marks the deposit
positions so a legitimately-zero step is distinguishable from a non-deposit token.

The GRPO loss (user-owned) consumes these deposits: it group-normalizes them and
sets each token's advantage to the sum of normalized step rewards at or after
that token. That preserves the O(S) reward density that distinguishes process
from outcome — outcome is the degenerate one-step case (a single deposit at the
last token). Normalization is the loss's job, so this module emits **raw**
P(correct) scores.

Step boundaries are found in **token space** — the student's
``completion_token_ids`` are split on the separator token — so the
step→token-position map is exact by construction (no fragile char alignment).
Each step's text is the decoded token span, scored by the PRM (Math-Shepherd)
with its model-card convention: suffix each step with the ``step_tag`` (``ки``),
restrict the next-token logits to ``{+, -}``, softmax, take ``P(+)`` in [0, 1].
The PRM is a 7B Mistral with its own tokenizer, so it scores decoded *text*, not
the student token ids.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

import torch

from lora_reward_density.rewards import RewardOutput
from lora_reward_density.rollout import RolloutBatch

logger = logging.getLogger(__name__)

# A scorer maps (questions, step_lists) -> raw per-step P(correct) in [0, 1], one
# inner list per completion. The GPU-bound PRM forward implements this; tests
# inject a CPU stub. ``questions[i]`` and ``step_lists[i]`` belong to completion i.
StepScorer = Callable[[list[str], list[list[str]]], list[list[float]]]


class _Tokenizer(Protocol):
    """Minimal student-tokenizer surface used for token-space segmentation."""

    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@dataclass(frozen=True)
class ProcessRewardConfig:
    prm_model_id: str = "peiyi9979/math-shepherd-mistral-7b-prm"
    # Reasoning steps are split on this delimiter in token space. Single "\n"
    # (D11 follow-up): the Qwen3-1.7B-Base student mostly uses single newlines
    # between steps, so "\n\n" left most completions as one step. Caveat: "\n"
    # can over-segment LaTeX (a newline inside a $$...$$ block) — validate via
    # `debug_rollout --regime process` before a real run.
    step_separator: str = "\n"
    include_prompt: bool = True  # prepend the problem text to the PRM input
    empty_step_score: float = 0.0  # trajectory_rewards summary when no steps parse
    # Math-Shepherd model-card tokens.
    good_token: str = "+"
    bad_token: str = "-"
    step_tag: str = "ки"
    prm_batch_size: int = 4
    max_input_tokens: int = 2048
    device: str = "cuda"
    dtype: str = "bfloat16"


def _segment_token_spans(ids: list[int], sep_ids: list[int]) -> list[tuple[int, int]]:
    """Split a token-id sequence into per-step ``(start, end)`` inclusive spans.

    A step is terminated by an occurrence of the ``sep_ids`` subsequence; the
    separator tokens belong to the step they close, and that step's deposit
    position is the span's ``end`` index. Trailing tokens with no terminator
    form a final step (e.g. the answer line). Empty input → no steps.
    """
    n, m = len(ids), len(sep_ids)
    if n == 0 or m == 0:
        return [(0, n - 1)] if n else []
    spans: list[tuple[int, int]] = []
    start = 0
    i = 0
    while i <= n - m:
        if ids[i : i + m] == sep_ids:
            spans.append((start, i + m - 1))
            i += m
            start = i
        else:
            i += 1
    if start < n:  # trailing step without a closing separator
        spans.append((start, n - 1))
    return spans


class _MathShepherdScorer:
    """Math-Shepherd PRM forward: ``(questions, step_lists) -> per-step P(+)``.

    Loads the 7B PRM once. Heavy deps (``transformers``) are imported here, not
    at module top level, so the local suite (which injects a stub scorer) never
    pulls in the GPU stack — matching the lazy-import pattern in
    ``rollout.VLLMRolloutEngine`` and ``outcome_reward``.
    """

    def __init__(self, config: ProcessRewardConfig) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._cfg = config
        # transformers' from_pretrained return types are union-typed and trip
        # pyright on this GPU-only path; annotate as Any (the suite stubs the
        # scorer and never constructs this class).
        self._tok: Any = AutoTokenizer.from_pretrained(config.prm_model_id)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        model: Any = AutoModelForCausalLM.from_pretrained(
            config.prm_model_id, torch_dtype=getattr(torch, config.dtype)
        )
        self._model = model.to(config.device).eval()
        # [good_id, bad_id]; [1:] drops the BOS the tokenizer prepends.
        self._candidate_ids = self._tok.encode(f"{config.good_token} {config.bad_token}")[1:]
        self._step_tag_id = self._tok.encode(config.step_tag)[-1]

    def _format(self, question: str, steps: list[str]) -> str:
        tagged = "\n".join(f"{step} {self._cfg.step_tag}" for step in steps)
        q = question.strip()
        return f"{q} {tagged}" if q else tagged

    def __call__(self, questions: list[str], step_lists: list[list[str]]) -> list[list[float]]:
        cfg = self._cfg
        texts = [self._format(q, steps) for q, steps in zip(questions, step_lists, strict=True)]
        out: list[list[float]] = []
        for start in range(0, len(texts), cfg.prm_batch_size):
            chunk = texts[start : start + cfg.prm_batch_size]
            enc = self._tok(
                chunk,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.max_input_tokens,
            ).to(cfg.device)
            with torch.no_grad():
                logits = self._model(**enc).logits  # [b, L, V]
            # Restrict to {good, bad}, softmax, take P(good) at every position.
            good_prob = logits[:, :, self._candidate_ids].softmax(dim=-1)[:, :, 0]  # [b, L]
            tag_positions = enc["input_ids"] == self._step_tag_id
            for row in range(good_prob.shape[0]):
                out.append(good_prob[row][tag_positions[row]].float().cpu().tolist())
        return out


class ProcessRewardModule:
    """Dense step-level PRM reward (regime b). Implements the ``RewardModule`` Protocol.

    Needs the **student** tokenizer to segment completions in token space; the
    PRM-backed scorer (which owns the PRM's *own* tokenizer) is built lazily on
    first ``score()``. Tests inject both a stub ``tokenizer`` and a stub
    ``step_scorer`` and never touch the GPU path.
    """

    name = "process"

    def __init__(
        self,
        config: ProcessRewardConfig | None = None,
        step_scorer: StepScorer | None = None,
        tokenizer: _Tokenizer | None = None,
    ) -> None:
        self._config = config or ProcessRewardConfig()
        self._scorer = step_scorer  # None → lazily build the real PRM scorer
        self._tokenizer = tokenizer
        self._sep_ids: list[int] | None = None

    def _get_scorer(self) -> StepScorer:
        if self._scorer is None:
            self._scorer = _MathShepherdScorer(self._config)
        return self._scorer

    def _get_sep_ids(self) -> list[int]:
        if self._sep_ids is None:
            if self._tokenizer is None:
                raise ValueError("ProcessRewardModule needs a student tokenizer to segment steps")
            self._sep_ids = self._tokenizer.encode(
                self._config.step_separator, add_special_tokens=False
            )
        return self._sep_ids

    def score(self, batch: RolloutBatch) -> RewardOutput:
        cfg = self._config
        n, t_max = batch.completion_token_ids.shape
        sep_ids = self._get_sep_ids()
        assert self._tokenizer is not None  # _get_sep_ids enforces this

        # Per completion: token-space step spans, decoded step text, and the
        # deposit (last-token) index of each step. Valid tokens are right-padded
        # to the front, so an index into the valid prefix is also its index in
        # the full [N, T] tensor.
        questions: list[str] = []
        step_text_lists: list[list[str]] = []
        deposit_positions: list[list[int]] = []
        for i in range(n):
            valid_len = int(batch.completion_mask[i].sum().item())
            ids = batch.completion_token_ids[i, :valid_len].tolist()
            spans = _segment_token_spans(ids, sep_ids)
            step_text_lists.append(
                [self._tokenizer.decode(ids[s : e + 1]).strip() for s, e in spans]
            )
            deposit_positions.append([e for _s, e in spans])
            prompt_idx = int(batch.group_index[i].item())
            questions.append(batch.prompts[prompt_idx] if cfg.include_prompt else "")

        step_scores = self._get_scorer()(questions, step_text_lists)
        if len(step_scores) != n:
            raise ValueError(f"scorer returned {len(step_scores)} score lists for {n} completions")

        token_rewards = torch.zeros(n, t_max, dtype=torch.float32)
        step_reward_mask = torch.zeros(n, t_max, dtype=torch.bool)
        traj = torch.empty(n, dtype=torch.float32)
        for i in range(n):
            scores, positions = step_scores[i], deposit_positions[i]
            if len(scores) != len(positions):
                # PRM step-tag count can drift from token-space step count (e.g.
                # truncation); align on the shorter and keep going.
                logger.warning(
                    "completion %d: %d PRM scores vs %d token-space steps; truncating",
                    i,
                    len(scores),
                    len(positions),
                )
            for score, pos in zip(scores, positions, strict=False):
                token_rewards[i, pos] = score
                step_reward_mask[i, pos] = True
            traj[i] = float(sum(scores)) if scores else float(cfg.empty_step_score)

        return RewardOutput(
            token_rewards=token_rewards,
            trajectory_rewards=traj,
            step_reward_mask=step_reward_mask,
            metadata={
                "regime": self.name,
                "step_counts": torch.tensor([len(p) for p in deposit_positions], dtype=torch.int64),
                "num_empty_completions": sum(1 for p in deposit_positions if not p),
            },
        )
