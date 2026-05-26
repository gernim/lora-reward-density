"""Outcome reward (regime a): rule-based correctness via math-verify.

Each prompt's metadata must include a `gold_answer` (str). The verifier parses
both the gold and the model's completion via math-verify and emits a binary
correct/incorrect reward, broadcast across the trajectory's valid tokens.
"""

from __future__ import annotations

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

        for i, completion in enumerate(batch.completions):
            prompt_idx = int(batch.group_index[i].item())
            md = batch.prompt_metadata[prompt_idx]
            if GOLD_ANSWER_KEY not in md:
                raise KeyError(f"prompt_metadata[{prompt_idx}] missing {GOLD_ANSWER_KEY!r}")
            gold_text = md[GOLD_ANSWER_KEY]

            try:
                gold = parse(gold_text, parsing_timeout=cfg.parse_timeout_seconds)
                ans = parse(completion, parsing_timeout=cfg.parse_timeout_seconds)
                ok = bool(verify(gold, ans, timeout_seconds=cfg.verify_timeout_seconds))
            except Exception as e:  # noqa: BLE001 — math-verify can raise sympy/timeout errors
                logger.warning("math-verify failed on completion %d: %s", i, e)
                ok = False
                parse_failures += 1

            correctness[i] = ok
            traj[i] = cfg.correct_reward if ok else cfg.incorrect_reward

        token_rewards = traj.unsqueeze(1) * batch.completion_mask.float()
        return RewardOutput(
            token_rewards=token_rewards,
            trajectory_rewards=traj,
            metadata={
                "correctness": correctness,
                "parse_failures": parse_failures,
                "regime": self.name,
            },
        )
