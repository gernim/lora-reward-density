from __future__ import annotations

from typing import Any

import pytest

from lora_reward_density.data import (
    DEFAULT_BASE_MODEL_TEMPLATE,
    _build_examples,
    load_math500,
)
from lora_reward_density.outcome_reward import GOLD_ANSWER_KEY


def _row(
    problem: str = "1+1=?",
    answer: str = "2",
    subject: str = "Algebra",
    level: int = 1,
    unique_id: str = "test/algebra/1.json",
) -> dict[str, Any]:
    return {
        "problem": problem,
        "answer": answer,
        "subject": subject,
        "level": level,
        "unique_id": unique_id,
        "solution": "",
    }


def test_build_examples_populates_prompt_and_gold_answer():
    rows = [_row(), _row(problem="2+2?", answer="4")]
    examples = _build_examples(rows, DEFAULT_BASE_MODEL_TEMPLATE)
    assert len(examples) == 2
    assert examples[0].prompt == "Problem: 1+1=?\n\nSolution: "
    assert examples[0].metadata[GOLD_ANSWER_KEY] == "2"
    assert examples[0].metadata["subject"] == "Algebra"
    assert examples[0].metadata["level"] == 1


def test_build_examples_truncates_to_num_examples():
    rows = [_row(problem=f"p{i}", answer=str(i)) for i in range(10)]
    examples = _build_examples(rows, DEFAULT_BASE_MODEL_TEMPLATE, num_examples=3)
    assert len(examples) == 3


def test_build_examples_shuffle_is_deterministic_by_seed():
    rows = [_row(problem=f"p{i}", answer=str(i)) for i in range(10)]
    a = _build_examples(rows, DEFAULT_BASE_MODEL_TEMPLATE, seed=42)
    b = _build_examples(rows, DEFAULT_BASE_MODEL_TEMPLATE, seed=42)
    assert [e.prompt for e in a] == [e.prompt for e in b]
    c = _build_examples(rows, DEFAULT_BASE_MODEL_TEMPLATE, seed=43)
    assert [e.prompt for e in a] != [e.prompt for e in c]


def test_build_examples_without_seed_preserves_order():
    rows = [_row(problem=f"p{i}", answer=str(i)) for i in range(5)]
    examples = _build_examples(rows, DEFAULT_BASE_MODEL_TEMPLATE)
    assert [e.prompt for e in examples] == [f"Problem: p{i}\n\nSolution: " for i in range(5)]


def test_build_examples_shuffle_then_truncate_gives_random_subset():
    """Truncation after shuffle yields a random subset, not the first N rows."""
    rows = [_row(problem=f"p{i}", answer=str(i)) for i in range(20)]
    examples = _build_examples(rows, DEFAULT_BASE_MODEL_TEMPLATE, seed=0, num_examples=5)
    assert len(examples) == 5
    # With a fixed seed and 20 rows, it's astronomically unlikely we get the
    # original order. If we do, the shuffle isn't running.
    assert [e.prompt for e in examples] != [f"Problem: p{i}\n\nSolution: " for i in range(5)]


def test_prompt_template_preserves_latex_braces():
    """LaTeX in the problem (\\frac{1}{2}) must survive substitution."""
    rows = [_row(problem=r"Find $\frac{1}{2} + \{3\}$.")]
    examples = _build_examples(rows, DEFAULT_BASE_MODEL_TEMPLATE)
    assert r"\frac{1}{2}" in examples[0].prompt
    assert r"\{3\}" in examples[0].prompt


def test_prompt_template_must_contain_problem_placeholder():
    with pytest.raises(ValueError, match=r"\{problem\}"):
        _build_examples([_row()], prompt_template="no placeholder here")


def test_custom_prompt_template_substitutes():
    rows = [_row(problem="hello")]
    template = "Q: {problem}\nA: "
    examples = _build_examples(rows, template)
    assert examples[0].prompt == "Q: hello\nA: "


def test_load_math500_integration():
    """Touches the real HuggingFaceH4/MATH-500 dataset (cached locally)."""
    pytest.importorskip("datasets")
    examples = load_math500(num_examples=3, seed=0)
    assert len(examples) == 3
    for ex in examples:
        assert ex.metadata[GOLD_ANSWER_KEY]
        assert ex.metadata["subject"]
        assert ex.metadata["level"] in {1, 2, 3, 4, 5}
        assert ex.prompt.startswith("Problem:")
        assert ex.prompt.endswith("Solution: ")
