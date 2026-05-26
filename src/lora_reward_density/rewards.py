"""Reward module protocol shared across the three regimes (outcome / process / distillation)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from lora_reward_density.rollout import RolloutBatch


@dataclass(frozen=True)
class RewardOutput:
    """Per-token and per-trajectory rewards from a regime.

    Both fields are populated regardless of regime so the GRPO loss can choose
    the granularity it consumes:

    - For outcome / process: `token_rewards` is the trajectory scalar broadcast
      across valid tokens (zeroed at pad positions); `trajectory_rewards` is
      the same scalar.
    - For distillation: `token_rewards` carries the per-token signal directly
      (e.g. negative reverse KL); `trajectory_rewards` is the masked sum.

    Group-mean normalization to obtain GRPO advantages happens in the loss, not
    here. `metadata` is for diagnostics (regime-specific) and is not consumed
    by the loss — wandb logging and ablation analysis only.
    """

    token_rewards: torch.Tensor  # [N, T] float
    trajectory_rewards: torch.Tensor  # [N] float
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


class RewardModule(Protocol):
    """Each regime (outcome / process / distillation) implements this interface.

    The same RolloutBatch is fed to every regime so the GRPO loss never branches
    on regime — only the RewardModule changes.
    """

    name: str

    def score(self, batch: RolloutBatch) -> RewardOutput: ...
