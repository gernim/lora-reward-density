from __future__ import annotations

import pytest
import torch

from lora_reward_density.outcome_reward import (
    GOLD_ANSWER_KEY,
    OutcomeRewardConfig,
    OutcomeRewardModule,
)
from lora_reward_density.rollout import (
    DeterministicMockRolloutEngine,
    RolloutBatch,
    SamplingConfig,
)


def _rollout(canned: dict[str, list[str]], golds: dict[str, str]) -> RolloutBatch:
    """Build a mock rollout where each prompt's metadata carries a gold_answer."""
    engine = DeterministicMockRolloutEngine(canned)
    prompts = list(canned.keys())
    metadata = [{GOLD_ANSWER_KEY: golds[p]} for p in prompts]
    g = len(next(iter(canned.values())))
    return engine.rollout(
        prompts=prompts,
        prompt_metadata=metadata,
        config=SamplingConfig(n=g, max_tokens=64),
    )


def test_outcome_reward_correct_completion_scores_one():
    canned = {"What is 1/2?": ["The answer is \\boxed{1/2}"]}
    golds = {"What is 1/2?": "1/2"}
    output = OutcomeRewardModule().score(_rollout(canned, golds))

    assert output.trajectory_rewards.tolist() == [1.0]
    assert output.metadata["correctness"].tolist() == [True]
    assert output.metadata["regime"] == "outcome"


def test_outcome_reward_incorrect_completion_scores_zero():
    canned = {"q": ["The answer is \\boxed{0.7}"]}
    golds = {"q": "1/2"}
    output = OutcomeRewardModule().score(_rollout(canned, golds))

    assert output.trajectory_rewards.tolist() == [0.0]
    assert output.metadata["correctness"].tolist() == [False]
    assert float(output.token_rewards.sum()) == 0.0


def test_outcome_reward_mixed_batch_aligns_to_group_index():
    canned = {
        "Q1": ["\\boxed{42}", "\\boxed{43}"],  # correct, wrong
        "Q2": ["\\boxed{0}", "\\boxed{0.0}"],  # both correct
    }
    golds = {"Q1": "42", "Q2": "0"}
    output = OutcomeRewardModule().score(_rollout(canned, golds))

    assert output.trajectory_rewards.tolist() == [1.0, 0.0, 1.0, 1.0]
    assert output.metadata["correctness"].tolist() == [True, False, True, True]


def test_outcome_reward_custom_reward_values():
    canned = {"q": ["\\boxed{1}"]}
    golds = {"q": "1"}
    cfg = OutcomeRewardConfig(correct_reward=10.0, incorrect_reward=-1.0)
    output = OutcomeRewardModule(cfg).score(_rollout(canned, golds))
    assert output.trajectory_rewards.tolist() == [10.0]


def test_outcome_reward_token_rewards_zero_at_pad():
    """trajectory_reward broadcasts onto valid tokens, zero on pad."""
    canned = {"q": ["\\boxed{1} more text", "\\boxed{1}"]}  # lengths 3, 1
    golds = {"q": "1"}
    output = OutcomeRewardModule().score(_rollout(canned, golds))

    assert output.trajectory_rewards.tolist() == [1.0, 1.0]
    # First completion: 3 valid tokens, all reward 1.0
    assert torch.allclose(output.token_rewards[0], torch.tensor([1.0, 1.0, 1.0]))
    # Second completion: 1 valid token, then pad
    assert torch.allclose(output.token_rewards[1], torch.tensor([1.0, 0.0, 0.0]))


def test_outcome_reward_missing_gold_answer_raises():
    canned = {"q": ["\\boxed{1}"]}
    engine = DeterministicMockRolloutEngine(canned)
    batch = engine.rollout(
        prompts=["q"],
        prompt_metadata=[{}],  # no gold_answer
        config=SamplingConfig(n=1, max_tokens=4),
    )
    with pytest.raises(KeyError, match="gold_answer"):
        OutcomeRewardModule().score(batch)


def test_outcome_reward_handles_unparseable_completion():
    """Completions that math-verify can't parse score as incorrect, not crash."""
    # Empty/garbage text — math-verify returns False rather than raising.
    canned = {"q": ["totally not math at all"]}
    golds = {"q": "42"}
    output = OutcomeRewardModule().score(_rollout(canned, golds))

    assert output.trajectory_rewards.tolist() == [0.0]
    assert output.metadata["correctness"].tolist() == [False]
