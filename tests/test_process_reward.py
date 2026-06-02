"""CPU tests for the dense process reward module (regime b).

The 7B PRM forward (``_MathShepherdScorer``) is GPU-only and out of scope here —
these tests inject a stub ``step_scorer`` and a stub student ``tokenizer`` and
exercise the pure logic the module owns: token-space step segmentation, deposit
placement at step-boundary positions, the step_reward_mask, group_index→question
mapping, and the ``RewardOutput`` shape contract.
"""

from __future__ import annotations

import pytest
import torch

from lora_reward_density.process_reward import (
    ProcessRewardConfig,
    ProcessRewardModule,
    _segment_token_spans,
)
from lora_reward_density.rollout import RolloutBatch

# Token-id vocabulary for the stub tokenizer: SEP marks a step boundary; all
# other ids are "content". The separator is a single token here.
SEP = 99


class _FakeTokenizer:
    """Segments on the SEP token and decodes ids to a readable string."""

    def encode(self, text, add_special_tokens=False):  # noqa: ARG002 — stub
        return [SEP]

    def decode(self, ids):
        return " ".join("sep" if i == SEP else f"t{i}" for i in ids)


class _ConstScorer:
    """Every step gets the same score, so deposits and the trajectory sum are a
    known function of step count. Returns ``[]`` for completions with no steps."""

    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self, _questions, step_lists):
        return [[self.value] * len(steps) for steps in step_lists]


class _RecordingScorer:
    """Records inputs and returns preset per-completion step scores."""

    def __init__(self, scores: list[list[float]]) -> None:
        self.scores = scores
        self.questions: list[str] | None = None
        self.step_lists: list[list[str]] | None = None

    def __call__(self, questions, step_lists):
        self.questions = questions
        self.step_lists = step_lists
        return self.scores


def _make_batch(prompts: list[str], token_id_rows: list[list[int]], t: int, pl: int = 3):
    """Build a batch from per-completion token-id lists (right-padded to t).

    completion_mask is True over each row's real length. Completions are laid
    out in contiguous groups (prompt 0's G rows first, then prompt 1's, ...)."""
    p, n = len(prompts), len(token_id_rows)
    assert n % p == 0
    g = n // p
    group_index = torch.tensor([i // g for i in range(n)], dtype=torch.long)
    comp_ids = torch.zeros(n, t, dtype=torch.long)
    comp_mask = torch.zeros(n, t, dtype=torch.bool)
    for i, row in enumerate(token_id_rows):
        comp_ids[i, : len(row)] = torch.tensor(row, dtype=torch.long)
        comp_mask[i, : len(row)] = True
    completions = [" ".join(str(x) for x in row) for row in token_id_rows]
    return RolloutBatch(
        prompts=list(prompts),
        prompt_metadata=[{} for _ in prompts],
        prompt_token_ids=torch.zeros(n, pl, dtype=torch.long),
        prompt_attention_mask=torch.ones(n, pl, dtype=torch.bool),
        completions=completions,
        completion_token_ids=comp_ids,
        completion_mask=comp_mask,
        sampler_logprobs=torch.zeros(n, t),
        group_index=group_index,
        pad_token_id=0,
    )


# --- token-space segmentation ----------------------------------------------


def test_segment_token_spans_splits_on_separator():
    # ids: [1,2, SEP, 3, SEP, 4] → steps end at the SEP tokens and the trailing 4.
    spans = _segment_token_spans([1, 2, SEP, 3, SEP, 4], [SEP])
    assert spans == [(0, 2), (3, 4), (5, 5)]


def test_segment_token_spans_no_separator_is_single_step():
    assert _segment_token_spans([1, 2, 3], [SEP]) == [(0, 2)]


def test_segment_token_spans_empty_is_no_steps():
    assert _segment_token_spans([], [SEP]) == []


def test_segment_token_spans_multi_token_separator():
    # Separator is the 2-token subsequence [8, 9].
    spans = _segment_token_spans([1, 8, 9, 2, 3], [8, 9])
    assert spans == [(0, 2), (3, 4)]


# --- score() with injected scorer + tokenizer -------------------------------


def test_score_deposits_step_rewards_at_boundary_positions():
    # One completion, two steps: [1,2,SEP | 3] → boundaries at index 2 and 3.
    batch = _make_batch(prompts=["Q"], token_id_rows=[[1, 2, SEP, 3]], t=6)
    scorer = _RecordingScorer([[0.3, 0.7]])
    module = ProcessRewardModule(step_scorer=scorer, tokenizer=_FakeTokenizer())

    out = module.score(batch)

    assert out.token_rewards.shape == (1, 6)
    # Raw step scores deposited at the step-end indices, zeros elsewhere.
    assert torch.allclose(out.token_rewards[0], torch.tensor([0.0, 0.0, 0.3, 0.7, 0.0, 0.0]))
    assert out.step_reward_mask is not None
    assert out.step_reward_mask[0].tolist() == [False, False, True, True, False, False]
    # trajectory_rewards is the (unnormalized) sum, for logging only.
    assert out.trajectory_rewards[0].item() == pytest.approx(1.0)
    assert out.token_rewards.device.type == "cpu"
    assert out.metadata["regime"] == "process"
    assert torch.equal(out.metadata["step_counts"], torch.tensor([2]))


def test_score_zero_valued_deposit_is_marked_in_mask():
    # A legitimately-0.0 step must be a deposit (mask True), distinct from pad.
    batch = _make_batch(prompts=["Q"], token_id_rows=[[1, SEP, 2]], t=4)
    scorer = _RecordingScorer([[0.0, 0.5]])
    out = ProcessRewardModule(step_scorer=scorer, tokenizer=_FakeTokenizer()).score(batch)
    assert out.token_rewards[0].tolist() == [0.0, 0.0, 0.5, 0.0]
    assert out.step_reward_mask is not None
    assert out.step_reward_mask[0].tolist() == [False, True, True, False]


def test_score_maps_completions_to_their_prompt_and_decodes_steps():
    # 2 prompts, group_size 1. Each completion is a single step (no SEP).
    batch = _make_batch(prompts=["Q0", "Q1"], token_id_rows=[[1, 2], [3]], t=3)
    scorer = _RecordingScorer([[0.5], [0.5]])
    ProcessRewardModule(step_scorer=scorer, tokenizer=_FakeTokenizer()).score(batch)
    assert scorer.questions == ["Q0", "Q1"]
    # Step text is the decoded token span.
    assert scorer.step_lists == [["t1 t2"], ["t3"]]


def test_score_include_prompt_false_passes_empty_questions():
    batch = _make_batch(prompts=["Q0", "Q1"], token_id_rows=[[1], [2]], t=2)
    scorer = _RecordingScorer([[0.5], [0.5]])
    cfg = ProcessRewardConfig(include_prompt=False)
    ProcessRewardModule(cfg, step_scorer=scorer, tokenizer=_FakeTokenizer()).score(batch)
    assert scorer.questions == ["", ""]


def test_score_empty_completion_gets_empty_step_score():
    batch = _make_batch(prompts=["Q"], token_id_rows=[[1, SEP, 2], []], t=4)
    cfg = ProcessRewardConfig(empty_step_score=0.0)
    module = ProcessRewardModule(cfg, step_scorer=_ConstScorer(0.6), tokenizer=_FakeTokenizer())

    out = module.score(batch)

    # First completion: two steps × 0.6 summed = 1.2. Second: no tokens → 0.0.
    assert out.trajectory_rewards.tolist() == pytest.approx([1.2, 0.0])
    assert torch.equal(out.metadata["step_counts"], torch.tensor([2, 0]))
    assert out.metadata["num_empty_completions"] == 1
    assert torch.all(out.token_rewards[1] == 0.0)


def test_score_rejects_scorer_length_mismatch():
    batch = _make_batch(prompts=["Q"], token_id_rows=[[1], [2]], t=2)
    bad_scorer = _RecordingScorer([[0.5]])  # 1 list for 2 completions
    with pytest.raises(ValueError, match="score lists"):
        ProcessRewardModule(step_scorer=bad_scorer, tokenizer=_FakeTokenizer()).score(batch)


def test_score_without_tokenizer_raises():
    batch = _make_batch(prompts=["Q"], token_id_rows=[[1]], t=2)
    with pytest.raises(ValueError, match="tokenizer"):
        ProcessRewardModule(step_scorer=_ConstScorer(0.5)).score(batch)
