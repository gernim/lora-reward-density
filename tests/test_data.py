from __future__ import annotations

from typing import Any

import pytest

from lora_reward_density.data import (
    DEFAULT_BASE_MODEL_TEMPLATE,
    _build_examples,
    _extract_boxed_answer,
    _normalize_hendrycks_rows,
    _parse_level,
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


def test_extract_boxed_answer_simple():
    assert _extract_boxed_answer(r"The answer is $\boxed{42}$.") == "42"


def test_extract_boxed_answer_nested_braces():
    """Nested braces (\\frac{1}{2}) must brace-match, not stop at the first }."""
    assert _extract_boxed_answer(r"So $\boxed{\frac{1}{2}}$") == r"\frac{1}{2}"


def test_extract_boxed_answer_uses_last_occurrence():
    """The final answer is the last \\boxed, not an earlier intermediate one."""
    sol = r"first \boxed{wrong}, then refine to \boxed{right}"
    assert _extract_boxed_answer(sol) == "right"


def test_extract_boxed_answer_tolerates_space_before_brace():
    assert _extract_boxed_answer(r"\boxed {7}") == "7"


def test_extract_boxed_answer_missing_returns_none():
    assert _extract_boxed_answer("no box here, answer is 5") is None


def test_extract_boxed_answer_unbalanced_returns_none():
    assert _extract_boxed_answer(r"\boxed{1 + ") is None


def _hrow(
    problem: str = "1+1?",
    solution: str = r"$\boxed{2}$",
    type_: str = "Algebra",
    level: str = "Level 1",
) -> dict[str, Any]:
    """A raw EleutherAI/hendrycks_math-shaped row (type/solution, no answer)."""
    return {"problem": problem, "solution": solution, "type": type_, "level": level}


def test_normalize_hendrycks_rows_maps_to_math500_schema():
    rows = _normalize_hendrycks_rows([_hrow()], source_config="algebra")
    assert len(rows) == 1
    row = rows[0]
    assert row["problem"] == "1+1?"
    assert row["answer"] == "2"  # extracted from \boxed{}
    assert row["subject"] == "Algebra"  # aliased from `type`
    assert row["level"] == "Level 1"
    assert row["unique_id"] == "hendrycks_math/algebra/0"


def test_normalize_hendrycks_rows_drops_unextractable():
    rows = _normalize_hendrycks_rows(
        [_hrow(solution="answer is 2, no box"), _hrow(solution=r"$\boxed{3}$")],
        source_config="geometry",
    )
    assert [r["answer"] for r in rows] == ["3"]


def test_parse_level_from_string():
    assert _parse_level("Level 1") == 1
    assert _parse_level("Level 5") == 5


def test_parse_level_from_int():
    assert _parse_level(3) == 3


def test_parse_level_unparseable_returns_none():
    assert _parse_level("Level ?") is None
    assert _parse_level(None) is None
    assert _parse_level("") is None


def test_level_filter_keeps_only_wanted_levels():
    """The difficulty-filter list comprehension load_math_train applies."""
    rows = [
        {"level": "Level 1"},
        {"level": "Level 3"},
        {"level": "Level 5"},
        {"level": "Level ?"},  # unparseable → dropped
    ]
    wanted = {1, 2, 3}
    kept = [r for r in rows if _parse_level(r["level"]) in wanted]
    assert [r["level"] for r in kept] == ["Level 1", "Level 3"]


def test_normalize_hendrycks_rows_feeds_build_examples():
    """End-to-end: normalized rows drop straight into the shared _build_examples."""
    rows = _normalize_hendrycks_rows(
        [_hrow(problem="2+2?", solution=r"$\boxed{4}$")], source_config="algebra"
    )
    examples = _build_examples(rows, DEFAULT_BASE_MODEL_TEMPLATE)
    assert examples[0].prompt == "Problem: 2+2?\n\nSolution: "
    assert examples[0].metadata[GOLD_ANSWER_KEY] == "4"


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
