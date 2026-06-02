"""Reward module protocol shared across the three regimes (outcome / process / distillation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from lora_reward_density.rollout import RolloutBatch


@dataclass(frozen=True)
class RewardOutput:
    """Per-token and per-trajectory rewards from a regime.

    `token_rewards` carries raw rewards *deposited* at specific token positions
    (zeros elsewhere); `step_reward_mask` marks those positions so the loss can
    group-normalize the deposits even when one is legitimately 0.0. The deposit
    pattern is the only thing that differs across regimes (DeepSeekMath §4.1.3):

    - outcome: a single deposit at each trajectory's last valid token.
    - process: one deposit per reasoning step, at the step's final token.
    - distillation: a deposit at every valid token (per-token reverse KL).

    `trajectory_rewards` is a scalar summary (e.g. the deposit sum) used for
    logging; the loss derives per-token advantages from `token_rewards` +
    `step_reward_mask`, not from `trajectory_rewards`.

    Group normalization to obtain GRPO advantages happens in the loss, not here.
    `metadata` is regime-specific diagnostics (wandb / ablation analysis) and is
    not consumed by the loss; anything the loss needs is a first-class field.

    `step_reward_mask` is optional for back-compat; when None the loss may fall
    back to treating every valid token as a deposit.
    """

    token_rewards: torch.Tensor  # [N, T] float, deposits at masked positions
    trajectory_rewards: torch.Tensor  # [N] float, scalar summary (logging)
    step_reward_mask: torch.Tensor | None = None  # [N, T] bool, deposit positions
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_rewards.dim() != 2:
            raise ValueError(
                f"token_rewards must be 2-D; got shape {tuple(self.token_rewards.shape)}"
            )
        if self.trajectory_rewards.dim() != 1:
            raise ValueError(
                f"trajectory_rewards must be 1-D; got shape {tuple(self.trajectory_rewards.shape)}"
            )
        if self.token_rewards.shape[0] != self.trajectory_rewards.shape[0]:
            raise ValueError(
                f"token_rewards.shape[0] ({self.token_rewards.shape[0]}) != "
                f"trajectory_rewards.shape[0] ({self.trajectory_rewards.shape[0]})"
            )
        if self.step_reward_mask is not None:
            if self.step_reward_mask.shape != self.token_rewards.shape:
                raise ValueError(
                    f"step_reward_mask {tuple(self.step_reward_mask.shape)} != "
                    f"token_rewards {tuple(self.token_rewards.shape)}"
                )
            if self.step_reward_mask.dtype != torch.bool:
                raise ValueError(
                    f"step_reward_mask must be bool; got {self.step_reward_mask.dtype}"
                )


class RewardModule(Protocol):
    """Each regime (outcome / process / distillation) implements this interface.

    The same RolloutBatch is fed to every regime so the GRPO loss never branches
    on regime — only the RewardModule changes.
    """

    name: str

    def score(self, batch: RolloutBatch) -> RewardOutput: ...
