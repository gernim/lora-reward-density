from __future__ import annotations

import pytest
import torch

from lora_reward_density.rollout import (
    DeterministicMockRolloutEngine,
    RolloutBatch,
    SamplingConfig,
)


def _canned() -> dict[str, list[str]]:
    return {
        "p1": ["completion one alpha", "completion one beta"],
        "p2": ["completion two gamma", "completion two delta"],
    }


def test_mock_rollout_basic_shapes_align():
    engine = DeterministicMockRolloutEngine(_canned())
    batch = engine.rollout(
        prompts=["p1", "p2"],
        prompt_metadata=[{"id": 1}, {"id": 2}],
        config=SamplingConfig(n=2, max_tokens=8),
    )
    assert batch.num_prompts == 2
    assert batch.num_completions == 4
    assert batch.group_size == 2
    assert batch.completion_token_ids.shape == (4, 3)  # all "X X X" → 3 words
    assert batch.completion_mask.shape == (4, 3)
    assert batch.sampler_logprobs.shape == (4, 3)
    assert batch.group_index.tolist() == [0, 0, 1, 1]
    # Prompt fields align row-for-row with the P*G completions. "p1"/"p2" are
    # single tokens, so prompt_len == 1 and every position is real.
    assert batch.prompt_token_ids.shape == (4, 1)
    assert batch.prompt_attention_mask.shape == (4, 1)
    assert batch.prompt_attention_mask.all()


def test_mock_rollout_left_pads_ragged_prompts():
    """Prompts of differing length are LEFT-padded (real tokens flush right)."""
    engine = DeterministicMockRolloutEngine({"short": ["a a"], "a much longer prompt": ["b b"]})
    batch = engine.rollout(
        prompts=["short", "a much longer prompt"],
        prompt_metadata=[{}, {}],
        config=SamplingConfig(n=1, max_tokens=4),
    )
    # Lengths: "short" → 1 token, "a much longer prompt" → 4 tokens. prompt_len = 4.
    assert batch.prompt_token_ids.shape == (2, 4)
    # Row 0 ("short"): 3 pad positions on the LEFT, 1 real token on the right.
    assert batch.prompt_attention_mask[0].tolist() == [False, False, False, True]
    assert batch.prompt_token_ids[0, :3].tolist() == [0, 0, 0]  # pad_token_id == 0
    # Row 1: full-length, all real.
    assert batch.prompt_attention_mask[1].tolist() == [True, True, True, True]


def test_mock_rollout_pads_ragged_completions():
    engine = DeterministicMockRolloutEngine({"p": ["short", "this is much longer text"]})
    batch = engine.rollout(
        prompts=["p"],
        prompt_metadata=[{}],
        config=SamplingConfig(n=2, max_tokens=8),
    )
    # Lengths: 1 and 5. T_max = 5.
    assert batch.completion_token_ids.shape == (2, 5)
    assert batch.completion_mask[0].tolist() == [True, False, False, False, False]
    assert batch.completion_mask[1].tolist() == [True, True, True, True, True]
    # Padded positions hold pad_token_id (0).
    assert batch.completion_token_ids[0, 1:].tolist() == [0, 0, 0, 0]
    assert batch.pad_token_id == 0


def test_mock_rollout_logprobs_uniform_at_valid_positions():
    engine = DeterministicMockRolloutEngine(_canned())
    batch = engine.rollout(
        prompts=["p1"],
        prompt_metadata=[{}],
        config=SamplingConfig(n=1, max_tokens=4),
    )
    valid = batch.sampler_logprobs[batch.completion_mask]
    assert torch.allclose(valid, torch.full_like(valid, -2.0))


def test_mock_rollout_truncates_to_n():
    """If canned has more samples than config.n, only the first n are used."""
    engine = DeterministicMockRolloutEngine({"p": ["sample a", "sample b", "sample c"]})
    batch = engine.rollout(
        prompts=["p"],
        prompt_metadata=[{}],
        config=SamplingConfig(n=2, max_tokens=4),
    )
    assert batch.num_completions == 2
    assert batch.completions == ["sample a", "sample b"]


def test_mock_rollout_rejects_unknown_prompt():
    engine = DeterministicMockRolloutEngine({"p1": ["a"]})
    with pytest.raises(KeyError, match="not canned"):
        engine.rollout(
            prompts=["unknown"],
            prompt_metadata=[{}],
            config=SamplingConfig(n=1, max_tokens=4),
        )


def test_mock_rollout_rejects_insufficient_canned():
    engine = DeterministicMockRolloutEngine({"p1": ["a", "b"]})
    with pytest.raises(ValueError, match="canned samples"):
        engine.rollout(
            prompts=["p1"],
            prompt_metadata=[{}],
            config=SamplingConfig(n=4, max_tokens=4),
        )


def test_mock_rollout_rejects_misaligned_metadata():
    engine = DeterministicMockRolloutEngine({"p1": ["a"]})
    with pytest.raises(ValueError, match="must align"):
        engine.rollout(
            prompts=["p1"],
            prompt_metadata=[],
            config=SamplingConfig(n=1, max_tokens=4),
        )


def test_rollout_batch_validates_tensor_shape_mismatch():
    with pytest.raises(ValueError, match="completion_mask"):
        RolloutBatch(
            prompts=["p1"],
            prompt_metadata=[{}],
            prompt_token_ids=torch.zeros((1, 1), dtype=torch.long),
            prompt_attention_mask=torch.ones((1, 1), dtype=torch.bool),
            completions=["c1"],
            completion_token_ids=torch.zeros((1, 3), dtype=torch.long),
            completion_mask=torch.zeros((1, 4), dtype=torch.bool),
            sampler_logprobs=torch.zeros((1, 3), dtype=torch.float32),
            group_index=torch.zeros(1, dtype=torch.long),
            pad_token_id=0,
        )


def test_rollout_batch_validates_completion_count_not_multiple():
    with pytest.raises(ValueError, match="not a multiple"):
        RolloutBatch(
            prompts=["p1", "p2"],
            prompt_metadata=[{}, {}],
            prompt_token_ids=torch.zeros((3, 1), dtype=torch.long),
            prompt_attention_mask=torch.ones((3, 1), dtype=torch.bool),
            completions=["c1", "c2", "c3"],
            completion_token_ids=torch.zeros((3, 1), dtype=torch.long),
            completion_mask=torch.ones((3, 1), dtype=torch.bool),
            sampler_logprobs=torch.zeros((3, 1), dtype=torch.float32),
            group_index=torch.zeros(3, dtype=torch.long),
            pad_token_id=0,
        )


def test_rollout_batch_validates_metadata_length():
    with pytest.raises(ValueError, match="prompt_metadata"):
        RolloutBatch(
            prompts=["p1", "p2"],
            prompt_metadata=[{}],  # too short
            prompt_token_ids=torch.zeros((2, 1), dtype=torch.long),
            prompt_attention_mask=torch.ones((2, 1), dtype=torch.bool),
            completions=["c1", "c2"],
            completion_token_ids=torch.zeros((2, 1), dtype=torch.long),
            completion_mask=torch.ones((2, 1), dtype=torch.bool),
            sampler_logprobs=torch.zeros((2, 1), dtype=torch.float32),
            group_index=torch.zeros(2, dtype=torch.long),
            pad_token_id=0,
        )


def test_rollout_batch_validates_group_index_shape():
    with pytest.raises(ValueError, match="group_index"):
        RolloutBatch(
            prompts=["p1"],
            prompt_metadata=[{}],
            prompt_token_ids=torch.zeros((1, 1), dtype=torch.long),
            prompt_attention_mask=torch.ones((1, 1), dtype=torch.bool),
            completions=["c1"],
            completion_token_ids=torch.zeros((1, 1), dtype=torch.long),
            completion_mask=torch.ones((1, 1), dtype=torch.bool),
            sampler_logprobs=torch.zeros((1, 1), dtype=torch.float32),
            group_index=torch.zeros(2, dtype=torch.long),  # wrong length
            pad_token_id=0,
        )


def test_rollout_batch_validates_prompt_token_ids_count():
    with pytest.raises(ValueError, match="prompt_token_ids"):
        RolloutBatch(
            prompts=["p1"],
            prompt_metadata=[{}],
            prompt_token_ids=torch.zeros((2, 1), dtype=torch.long),  # shape[0] != n
            prompt_attention_mask=torch.ones((2, 1), dtype=torch.bool),
            completions=["c1"],
            completion_token_ids=torch.zeros((1, 1), dtype=torch.long),
            completion_mask=torch.ones((1, 1), dtype=torch.bool),
            sampler_logprobs=torch.zeros((1, 1), dtype=torch.float32),
            group_index=torch.zeros(1, dtype=torch.long),
            pad_token_id=0,
        )


def test_rollout_batch_validates_prompt_mask_shape_mismatch():
    with pytest.raises(ValueError, match="prompt_attention_mask"):
        RolloutBatch(
            prompts=["p1"],
            prompt_metadata=[{}],
            prompt_token_ids=torch.zeros((1, 3), dtype=torch.long),
            prompt_attention_mask=torch.ones((1, 2), dtype=torch.bool),  # width mismatch
            completions=["c1"],
            completion_token_ids=torch.zeros((1, 1), dtype=torch.long),
            completion_mask=torch.ones((1, 1), dtype=torch.bool),
            sampler_logprobs=torch.zeros((1, 1), dtype=torch.float32),
            group_index=torch.zeros(1, dtype=torch.long),
            pad_token_id=0,
        )
