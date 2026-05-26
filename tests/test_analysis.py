from __future__ import annotations

from pathlib import Path

import pytest

from lora_reward_density.analysis import (
    BaselineResults,
    mean_pass_at_k,
    parse_failure_rate,
    pass_at_k,
    pass_at_k_by_level,
    pass_at_k_by_subject,
    response_length_stats,
    summary_table,
)


def _results(
    *,
    num_prompts: int,
    n_samples: int,
    correctness: list[bool],
    response_lengths: list[int] | None = None,
    subjects: list[str] | None = None,
    levels: list[int] | None = None,
    model_id: str = "test-model",
    parse_failures: int = 0,
) -> BaselineResults:
    n = num_prompts * n_samples
    if len(correctness) != n:
        raise ValueError("correctness length mismatch in fixture")
    rewards = [1.0 if c else 0.0 for c in correctness]
    lengths = response_lengths if response_lengths is not None else [10] * n
    subjects = subjects if subjects is not None else ["Algebra"] * num_prompts
    levels = levels if levels is not None else [1] * num_prompts
    return BaselineResults(
        model_id=model_id,
        dataset="MATH-500",
        num_prompts=num_prompts,
        n_samples=n_samples,
        rewards=rewards,
        correctness=correctness,
        response_lengths=lengths,
        group_index=[i // n_samples for i in range(n)],
        parse_failures=parse_failures,
        prompt_metadata=[
            {"subject": subjects[i], "level": levels[i], "gold_answer": "x"}
            for i in range(num_prompts)
        ],
    )


# ----------- pass@k -----------


def test_pass_at_k_all_correct_is_one():
    assert pass_at_k(n=8, c=8, k=1) == 1.0
    assert pass_at_k(n=8, c=8, k=8) == 1.0


def test_pass_at_k_all_incorrect_is_zero():
    assert pass_at_k(n=8, c=0, k=1) == 0.0
    assert pass_at_k(n=8, c=0, k=8) == 0.0


def test_pass_at_k_one_of_eight_correct():
    # With 1/8 correct, pass@1 = 1/8, pass@8 = 1
    assert pass_at_k(n=8, c=1, k=1) == pytest.approx(0.125)
    assert pass_at_k(n=8, c=1, k=8) == 1.0


def test_pass_at_k_half_correct():
    # n=4, c=2, k=1 → 0.5
    assert pass_at_k(n=4, c=2, k=1) == pytest.approx(0.5)
    # n=4, c=2, k=2: P(at least one correct in 2 draws) = 1 - C(2,2)/C(4,2) = 1 - 1/6 = 5/6
    assert pass_at_k(n=4, c=2, k=2) == pytest.approx(5 / 6)


def test_pass_at_k_invalid_args():
    with pytest.raises(ValueError):
        pass_at_k(n=0, c=0, k=1)
    with pytest.raises(ValueError):
        pass_at_k(n=4, c=5, k=1)
    with pytest.raises(ValueError):
        pass_at_k(n=4, c=2, k=5)
    with pytest.raises(ValueError):
        pass_at_k(n=4, c=2, k=0)


# ----------- mean_pass_at_k -----------


def test_mean_pass_at_k_uniform_perfect():
    r = _results(num_prompts=3, n_samples=4, correctness=[True] * 12)
    assert mean_pass_at_k(r, k=1) == 1.0
    assert mean_pass_at_k(r, k=4) == 1.0


def test_mean_pass_at_k_mixed():
    # Prompt 0: 2/4 correct → pass@1=0.5
    # Prompt 1: 0/4 correct → pass@1=0
    # Prompt 2: 4/4 correct → pass@1=1
    # Mean = 0.5
    correctness = [True, True, False, False, False, False, False, False, True, True, True, True]
    r = _results(num_prompts=3, n_samples=4, correctness=correctness)
    assert mean_pass_at_k(r, k=1) == pytest.approx(0.5)


def test_mean_pass_at_k_rejects_k_larger_than_n_samples():
    r = _results(num_prompts=2, n_samples=4, correctness=[True] * 8)
    with pytest.raises(ValueError, match="pass@8"):
        mean_pass_at_k(r, k=8)


# ----------- response length -----------


def test_response_length_stats():
    r = _results(
        num_prompts=2,
        n_samples=2,
        correctness=[True] * 4,
        response_lengths=[10, 20, 30, 40],
    )
    stats = response_length_stats(r)
    assert stats["min"] == 10
    assert stats["max"] == 40
    assert stats["mean"] == 25
    assert stats["median"] == 25


# ----------- parse failure -----------


def test_parse_failure_rate():
    r = _results(num_prompts=2, n_samples=2, correctness=[True] * 4, parse_failures=1)
    assert parse_failure_rate(r) == 0.25


# ----------- by subject / level -----------


def test_pass_at_k_by_subject():
    r = _results(
        num_prompts=4,
        n_samples=2,
        correctness=[True, True, False, False, True, False, False, False],
        subjects=["Algebra", "Algebra", "Geometry", "Geometry"],
    )
    # Algebra: prompts 0 (2/2 → pass@1=1) and 1 (0/2 → pass@1=0); mean = 0.5
    # Geometry: prompts 2 (1/2 → 0.5) and 3 (0/2 → 0); mean = 0.25
    out = pass_at_k_by_subject(r, k=1)
    assert out == {"Algebra": pytest.approx(0.5), "Geometry": pytest.approx(0.25)}


def test_pass_at_k_by_level():
    r = _results(
        num_prompts=3,
        n_samples=2,
        correctness=[True, True, False, False, True, False],
        levels=[1, 1, 5],
    )
    # Level 1: prompts 0 (pass@1=1) and 1 (pass@1=0); mean = 0.5
    # Level 5: prompt 2 (pass@1=0.5); mean = 0.5
    out = pass_at_k_by_level(r, k=1)
    assert out == {1: pytest.approx(0.5), 5: pytest.approx(0.5)}


# ----------- reshape / serialization -----------


def test_reshape_per_prompt_grouping():
    r = _results(num_prompts=3, n_samples=2, correctness=[True] * 6)
    grouped = r.reshape_per_prompt(["a", "b", "c", "d", "e", "f"])
    assert grouped == [["a", "b"], ["c", "d"], ["e", "f"]]


def test_reshape_rejects_wrong_length():
    r = _results(num_prompts=3, n_samples=2, correctness=[True] * 6)
    with pytest.raises(ValueError):
        r.reshape_per_prompt(["a", "b", "c"])


def test_baseline_results_json_roundtrip(tmp_path: Path):
    r = _results(num_prompts=2, n_samples=3, correctness=[True, False, True] * 2)
    path = tmp_path / "results.json"
    r.to_json_path(path)
    loaded = BaselineResults.from_json_path(path)
    assert loaded.num_prompts == r.num_prompts
    assert loaded.correctness == r.correctness
    assert loaded.rewards == r.rewards


# ----------- summary table -----------


def test_summary_table_contains_pass_at_k_rows():
    r = _results(num_prompts=4, n_samples=8, correctness=[True] * 16 + [False] * 16)
    table = summary_table(r, ks=(1, 8))
    assert "pass@1" in table
    assert "pass@8" in table
    assert "test-model" in table
    assert "MATH-500" in table


def test_summary_table_skips_k_larger_than_n_samples():
    r = _results(num_prompts=2, n_samples=4, correctness=[True] * 8)
    table = summary_table(r, ks=(1, 4, 8))
    assert "pass@1" in table
    assert "pass@4" in table
    assert "pass@8" not in table


# ----------- plot smoke tests (just check they don't raise) -----------


def test_plot_factories_smoke(tmp_path: Path):
    from lora_reward_density.analysis import (
        plot_pass_at_k_comparison,
        plot_pass_by_level,
        plot_pass_by_subject,
        plot_response_length_hist,
    )

    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")  # non-interactive backend for headless test runs

    r = _results(
        num_prompts=4,
        n_samples=4,
        correctness=[True, False, True, False] * 4,
        response_lengths=[100, 200, 300, 400] * 4,
        subjects=["Algebra", "Algebra", "Geometry", "Number Theory"],
        levels=[1, 3, 5, 2],
    )

    plot_pass_at_k_comparison(
        {"base": r, "teacher": r}, ks=(1, 4), output_path=tmp_path / "pass_at_k.png"
    )
    plot_response_length_hist(r, output_path=tmp_path / "lengths.png")
    plot_pass_by_subject(r, k=1, output_path=tmp_path / "by_subject.png")
    plot_pass_by_level(r, k=1, output_path=tmp_path / "by_level.png")

    for name in ("pass_at_k.png", "lengths.png", "by_subject.png", "by_level.png"):
        assert (tmp_path / name).is_file()
