"""CPU tests for the distillation reward module (regime c).

The 8B teacher (separate Modal GPU) and the student forward are both injected,
so these tests stub them and exercise the pure logic the module owns: the
per-token reverse-KL reward, deposit-at-every-token, the step_reward_mask, and
the RewardOutput contract. No model is loaded.
"""

from __future__ import annotations

import pytest
import torch

from lora_reward_density.distillation_reward import (
    DistillationRewardConfig,
    DistillationRewardModule,
    _distillation_token_rewards,
)
from lora_reward_density.rollout import RolloutBatch


def _make_batch(n: int, t: int, valid_lens: list[int], pl: int = 3) -> RolloutBatch:
    comp_mask = torch.zeros(n, t, dtype=torch.bool)
    for i, length in enumerate(valid_lens):
        comp_mask[i, :length] = True
    return RolloutBatch(
        prompts=["q"] * n,
        prompt_metadata=[{} for _ in range(n)],
        prompt_token_ids=torch.zeros(n, pl, dtype=torch.long),
        prompt_attention_mask=torch.ones(n, pl, dtype=torch.bool),
        completions=["c"] * n,
        completion_token_ids=torch.zeros(n, t, dtype=torch.long),
        completion_mask=comp_mask,
        sampler_logprobs=torch.zeros(n, t),
        group_index=torch.zeros(n, dtype=torch.long),
        pad_token_id=0,
    )


class _StubTeacher:
    def __init__(self, logprobs: torch.Tensor) -> None:
        self._lp = logprobs

    def logprobs(self, batch):  # noqa: ARG002 — name matches the TeacherClient protocol
        return self._lp


# --- pure reward math -------------------------------------------------------


def test_token_rewards_is_teacher_minus_policy_masked():
    teacher = torch.tensor([[2.0, 2.0, 2.0]])
    policy = torch.tensor([[1.0, 0.0, 5.0]])
    mask = torch.tensor([[True, True, False]])
    r = _distillation_token_rewards(teacher, policy, mask)
    # r = [1, 2, -3], zeroed at the pad position → [1, 2, 0].
    assert torch.allclose(r, torch.tensor([[1.0, 2.0, 0.0]]))


def test_token_rewards_clip():
    teacher = torch.tensor([[10.0, -10.0]])
    policy = torch.tensor([[0.0, 0.0]])
    mask = torch.tensor([[True, True]])
    r = _distillation_token_rewards(teacher, policy, mask, reward_clip=3.0)
    assert torch.allclose(r, torch.tensor([[3.0, -3.0]]))


# --- score() with stub teacher + stub policy fn -----------------------------


def test_score_builds_per_token_deposits():
    batch = _make_batch(n=1, t=3, valid_lens=[2])
    teacher = _StubTeacher(torch.tensor([[2.0, 2.0, 2.0]]))
    policy_fn = lambda _b: torch.tensor([[1.0, 0.0, 5.0]])  # noqa: E731
    module = DistillationRewardModule(teacher=teacher, policy_logprob_fn=policy_fn)

    out = module.score(batch)

    assert torch.allclose(out.token_rewards, torch.tensor([[1.0, 2.0, 0.0]]))
    # Every valid token is a deposit → step_reward_mask == completion_mask.
    assert out.step_reward_mask is not None
    assert out.step_reward_mask[0].tolist() == [True, True, False]
    assert out.trajectory_rewards.item() == pytest.approx(3.0)  # masked sum
    assert out.token_rewards.device.type == "cpu"
    # Dense per-token reward → loss uses per-token advantage (no reverse-cumsum).
    assert out.per_token_advantage is True
    assert out.metadata["regime"] == "distillation"
    assert out.metadata["mean_token_reward"] == pytest.approx(1.5)  # (1+2)/2 valid


def test_score_respects_reward_clip():
    batch = _make_batch(n=1, t=2, valid_lens=[2])
    teacher = _StubTeacher(torch.tensor([[10.0, -10.0]]))
    policy_fn = lambda _b: torch.zeros(1, 2)  # noqa: E731
    module = DistillationRewardModule(
        DistillationRewardConfig(reward_clip=2.0), teacher=teacher, policy_logprob_fn=policy_fn
    )
    out = module.score(batch)
    assert torch.allclose(out.token_rewards, torch.tensor([[2.0, -2.0]]))


def test_score_without_teacher_raises():
    batch = _make_batch(n=1, t=2, valid_lens=[2])
    module = DistillationRewardModule(policy_logprob_fn=lambda _b: torch.zeros(1, 2))
    with pytest.raises(ValueError, match="teacher"):
        module.score(batch)


def test_score_without_policy_raises():
    batch = _make_batch(n=1, t=2, valid_lens=[2])
    module = DistillationRewardModule(teacher=_StubTeacher(torch.zeros(1, 2)))
    with pytest.raises(ValueError, match="policy"):
        module.score(batch)
