"""CPU end-to-end test of the user-owned ``training_step``.

`training_step` was previously only exercised on Modal (it needs a model, an
optimizer, and a rollout). That meant import/wiring/device bugs surfaced only
on a billed GPU run. This test drives the full inner sequence — rollout →
reward → student forward → reference forward → grpo_loss → backward → clip →
optimizer.step — on CPU using the mock rollout engine, a tiny stub LM, and a
stub reward module, so those bugs are caught in ~1s locally.

Out of scope (genuinely needs a GPU): CUDA/CPU device-placement bugs, since the
mock engine produces CPU tensors. This guards wiring, imports, gradient flow,
and the epochs contract — not device placement.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from lora_reward_density.grpo import grpo_loss
from lora_reward_density.rewards import RewardOutput
from lora_reward_density.rollout import DeterministicMockRolloutEngine, SamplingConfig
from lora_reward_density.train import (
    TrainRunConfig,
    compute_completion_logprobs,
    training_step,
)

# Few distinct words → all mock token IDs stay well under _VOCAB. Completions
# have ragged length (3 and 2 words) to exercise right-padding; prompts are
# multi-word to exercise the left-padded prompt block.
_CANNED = {
    "solve x plus": ["alpha beta gamma", "delta epsilon zeta", "alpha delta", "beta epsilon"],
    "compute y times": ["eta theta iota", "kappa lambda mu", "eta kappa", "theta lambda"],
}
_VOCAB = 64


class _TinyLM(nn.Module):
    """Minimal causal-LM stub: ``logits[i, t] = embed(input_ids[i, t])``.

    Enough for ``compute_completion_logprobs`` (it only reads ``.logits``) and
    trainable (the embedding weight), so an optimizer step produces an
    observable parameter change. No attention/dropout — we test wiring and
    gradient flow, not generation fidelity.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None):  # noqa: ARG002 — stub ignores mask/positions
        return SimpleNamespace(logits=self.embed(input_ids))


class _StubReward:
    """Reward module returning distinct per-trajectory rewards.

    Varied rewards make within-group advantages non-degenerate (non-zero std),
    so the surrogate term contributes a real gradient — not just the KL term.
    """

    name = "stub"

    def score(self, batch) -> RewardOutput:
        n = batch.num_completions
        traj = torch.arange(n, dtype=torch.float32)
        token_rewards = traj.unsqueeze(1) * batch.completion_mask.float()
        return RewardOutput(token_rewards=token_rewards, trajectory_rewards=traj)


def _make_step_kwargs(*, epochs: int = 1):
    torch.manual_seed(0)
    student = _TinyLM(_VOCAB)
    reference = _TinyLM(_VOCAB)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-2)
    return (
        student,
        reference,
        {
            "student": student,
            "reference": reference,
            "optimizer": optimizer,
            "rollout_engine": DeterministicMockRolloutEngine(_CANNED),
            "reward_module": _StubReward(),
            "grpo_loss": grpo_loss,
            "grpo_config": TrainRunConfig(),
            "prompts": list(_CANNED.keys()),
            "prompt_metadata": [{} for _ in _CANNED],
            "sampling_config": SamplingConfig(n=4, max_tokens=8),
            "epochs": epochs,
            "max_grad_norm": 1.0,
        },
    )


def test_training_step_runs_and_updates_student():
    student, reference, kwargs = _make_step_kwargs()
    before_student = student.embed.weight.detach().clone()
    before_reference = reference.embed.weight.detach().clone()

    diag = training_step(**kwargs)

    # Diagnostics flow through from the loss; loss is finite.
    assert {"loss", "mean_reward"} <= set(diag)
    assert math.isfinite(diag["loss"])
    # End-to-end gradient signal reached the student's parameters.
    assert not torch.allclose(before_student, student.embed.weight)
    # Reference is frozen (not in the optimizer, forwarded under no_grad).
    assert torch.allclose(before_reference, reference.embed.weight)


def test_training_step_multi_epoch_runs():
    """epochs>1 re-forwards the student each epoch and steps multiple times."""
    student, _reference, kwargs = _make_step_kwargs(epochs=3)
    before = student.embed.weight.detach().clone()
    diag = training_step(**kwargs)
    assert math.isfinite(diag["loss"])
    assert not torch.allclose(before, student.embed.weight)


def test_training_step_rejects_zero_epochs():
    _student, _reference, kwargs = _make_step_kwargs(epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        training_step(**kwargs)


class _PosLM(nn.Module):
    """Stub whose logits depend on position_ids as well as token ids.

    Position-sensitive so a wrong position assignment (e.g. not accounting for
    left-padding) changes the output — which lets us verify that
    compute_completion_logprobs passes left-pad-correct position_ids. A
    position-agnostic stub (like _TinyLM) would pass even with the bug.
    """

    def __init__(self, vocab_size: int, max_pos: int = 64) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, vocab_size)
        self.pos = nn.Embedding(max_pos, vocab_size)

    def forward(self, input_ids, attention_mask=None, position_ids=None):  # noqa: ARG002
        if position_ids is None:
            position_ids = torch.arange(input_ids.shape[1]).expand_as(input_ids)
        return SimpleNamespace(logits=self.tok(input_ids) + self.pos(position_ids))


def test_compute_completion_logprobs_is_left_pad_invariant():
    """Left-padding the prompt must not change the completion logprobs.

    With correct (mask-derived) position_ids, the real tokens get the same
    positions whether or not the prompt is left-padded, so the position-
    sensitive stub returns identical logprobs. Default arange position_ids would
    shift the real tokens under padding and break this — the step-0 ratio!=1
    symptom seen on Modal.
    """
    torch.manual_seed(0)
    model = _PosLM(_VOCAB)
    model.eval()

    completion = torch.tensor([[5, 6, 7]])
    comp_mask = torch.ones_like(completion, dtype=torch.bool)

    # Unpadded prompt [3, 4].
    prompt = torch.tensor([[3, 4]])
    prompt_mask = torch.ones_like(prompt, dtype=torch.bool)
    base = compute_completion_logprobs(model, prompt, completion, comp_mask, prompt_mask)

    # Same prompt, left-padded with two pad tokens (id 0).
    padded_prompt = torch.tensor([[0, 0, 3, 4]])
    padded_prompt_mask = torch.tensor([[False, False, True, True]])
    padded = compute_completion_logprobs(
        model, padded_prompt, completion, comp_mask, padded_prompt_mask
    )

    assert torch.allclose(base, padded, atol=1e-5)
