"""Analysis utilities for baseline_eval outputs.

Pure data crunching (pass@k, response-length stats, per-subject/level
breakdowns) plus a small set of matplotlib plot factories. Reads JSON dumps
produced by ``modal_app/baseline_eval.py`` and produces the figures and
numbers that go into the milestone PDF.

The stats functions have no plotting dependency; matplotlib is lazy-imported
inside ``plot_*`` functions so this module loads without it.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BaselineResults:
    """Output of one ``baseline_eval`` Modal run, serialized as JSON.

    Flat lists of length ``num_prompts * n_samples`` for per-completion fields
    (`rewards`, `correctness`, `response_lengths`, `group_index`); per-prompt
    list of length ``num_prompts`` for ``prompt_metadata``. Order matches
    ``RolloutBatch`` packing: completions for prompt 0 come first, then prompt
    1, etc.
    """

    model_id: str
    dataset: str
    num_prompts: int
    n_samples: int
    rewards: list[float]
    correctness: list[bool]
    response_lengths: list[int]
    group_index: list[int]
    parse_failures: int
    prompt_metadata: list[dict[str, Any]]
    sample_completions: list[str] = field(default_factory=list)
    sampling: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json_path(cls, path: Path | str) -> BaselineResults:
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def to_json_path(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2, sort_keys=True))

    @property
    def num_completions(self) -> int:
        return len(self.rewards)

    def _check_invariants(self) -> None:
        n = self.num_prompts * self.n_samples
        for name, values in (
            ("rewards", self.rewards),
            ("correctness", self.correctness),
            ("response_lengths", self.response_lengths),
            ("group_index", self.group_index),
        ):
            if len(values) != n:
                raise ValueError(
                    f"{name} has length {len(values)} but expected num_prompts*n_samples = {n}"
                )
        if len(self.prompt_metadata) != self.num_prompts:
            raise ValueError(
                f"prompt_metadata has length {len(self.prompt_metadata)} "
                f"but expected num_prompts = {self.num_prompts}"
            )

    def reshape_per_prompt(self, values: list[Any]) -> list[list[Any]]:
        """Group a flat ``[num_prompts*n_samples]`` list into ``[num_prompts][n_samples]``.

        Assumes the canonical packing order from ``rollout._pack_rollout``: all
        ``n_samples`` completions for prompt 0, then prompt 1, etc.
        """
        if len(values) != self.num_completions:
            raise ValueError(
                f"length mismatch: values={len(values)}, expected {self.num_completions}"
            )
        out: list[list[Any]] = []
        for i in range(self.num_prompts):
            start = i * self.n_samples
            out.append(list(values[start : start + self.n_samples]))
        return out


# ----------------------------------------------------------------------------
# Stats
# ----------------------------------------------------------------------------


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator from Chen et al. 2021 (Codex / HumanEval).

    Computes 1 - C(n-c, k) / C(n, k) in a numerically stable product form
    (no large factorials).

    Args:
        n: total samples per prompt.
        c: number of correct samples among the n.
        k: pass@k value to estimate.
    """
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n]; got c={c}, n={n}")
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, n]; got k={k}, n={n}")
    if n - c < k:
        return 1.0
    # 1 - prod_{i=n-c+1}^{n} (1 - k / i)
    return float(1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1)))


def mean_pass_at_k(results: BaselineResults, k: int) -> float:
    """Mean pass@k across all prompts in the results."""
    results._check_invariants()
    if k > results.n_samples:
        raise ValueError(
            f"pass@{k} requires at least {k} samples per prompt; "
            f"results have n_samples={results.n_samples}"
        )
    per_prompt = results.reshape_per_prompt(results.correctness)
    return float(np.mean([pass_at_k(len(c), sum(c), k) for c in per_prompt]))


def response_length_stats(results: BaselineResults) -> dict[str, float]:
    """min/mean/median/p90/p99/max of completion lengths in tokens."""
    results._check_invariants()
    arr = np.asarray(results.response_lengths)
    return {
        "min": float(arr.min()),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def parse_failure_rate(results: BaselineResults) -> float:
    return results.parse_failures / max(results.num_completions, 1)


def pass_at_k_by_subject(results: BaselineResults, k: int) -> dict[str, float]:
    """Per-subject mean pass@k. Useful for analysis splits."""
    results._check_invariants()
    per_prompt_correct = results.reshape_per_prompt(results.correctness)
    by_subject: dict[str, list[float]] = {}
    for i, samples in enumerate(per_prompt_correct):
        subject = str(results.prompt_metadata[i].get("subject", "unknown"))
        by_subject.setdefault(subject, []).append(pass_at_k(len(samples), sum(samples), k))
    return {s: float(statistics.mean(v)) for s, v in sorted(by_subject.items())}


def pass_at_k_by_level(results: BaselineResults, k: int) -> dict[int, float]:
    """Per-difficulty-level mean pass@k (MATH levels 1..5)."""
    results._check_invariants()
    per_prompt_correct = results.reshape_per_prompt(results.correctness)
    by_level: dict[int, list[float]] = {}
    for i, samples in enumerate(per_prompt_correct):
        level = int(results.prompt_metadata[i].get("level", 0))
        by_level.setdefault(level, []).append(pass_at_k(len(samples), sum(samples), k))
    return {lvl: float(statistics.mean(v)) for lvl, v in sorted(by_level.items())}


def summary_table(results: BaselineResults, ks: tuple[int, ...] = (1, 8)) -> str:
    """One-shot markdown summary table for the milestone."""
    lengths = response_length_stats(results)
    rows = [
        f"| Model | {results.model_id} |",
        f"| Dataset | {results.dataset} |",
        f"| Prompts | {results.num_prompts} |",
        f"| Samples/prompt | {results.n_samples} |",
    ]
    for k in ks:
        if k <= results.n_samples:
            rows.append(f"| pass@{k} | {mean_pass_at_k(results, k):.3f} |")
    rows.extend(
        [
            f"| Response length (mean) | {lengths['mean']:.0f} tokens |",
            f"| Response length (p90)  | {lengths['p90']:.0f} tokens |",
            f"| Parse failure rate | {parse_failure_rate(results):.3%} |",
        ]
    )
    return "| Metric | Value |\n|---|---|\n" + "\n".join(rows)


# ----------------------------------------------------------------------------
# Plot factories — matplotlib lazy-imported
# ----------------------------------------------------------------------------


def plot_pass_at_k_comparison(
    results_by_label: dict[str, BaselineResults],
    ks: tuple[int, ...] = (1, 4, 8),
    output_path: Path | str | None = None,
):
    """Grouped bar chart: pass@k for several k values, multiple models side-by-side."""
    import matplotlib.pyplot as plt

    labels = list(results_by_label.keys())
    n_models = len(labels)
    x = np.arange(len(ks))
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, label in enumerate(labels):
        results = results_by_label[label]
        valid_ks = [k for k in ks if k <= results.n_samples]
        values = [mean_pass_at_k(results, k) for k in valid_ks]
        positions = x[: len(valid_ks)] + (i - (n_models - 1) / 2) * width
        ax.bar(positions, values, width=width, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"pass@{k}" for k in ks])
    ax.set_ylabel("Mean")
    ax.set_ylim(0, 1)
    ax.set_title("Pass@k by model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_response_length_hist(
    results: BaselineResults,
    bins: int = 40,
    output_path: Path | str | None = None,
):
    """Histogram of completion lengths (tokens)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(results.response_lengths, bins=bins, alpha=0.8)
    stats = response_length_stats(results)
    ax.axvline(stats["mean"], linestyle="--", linewidth=1, label=f"mean={stats['mean']:.0f}")
    ax.axvline(stats["p90"], linestyle=":", linewidth=1, label=f"p90={stats['p90']:.0f}")
    ax.set_xlabel("Completion length (tokens)")
    ax.set_ylabel("Count")
    ax.set_title(f"Response length distribution ({results.model_id})")
    ax.legend()
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pass_by_subject(
    results: BaselineResults,
    k: int = 1,
    output_path: Path | str | None = None,
):
    """Horizontal bar chart of per-subject mean pass@k."""
    import matplotlib.pyplot as plt

    by_subject = pass_at_k_by_subject(results, k)
    subjects = list(by_subject.keys())
    values = [by_subject[s] for s in subjects]

    fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(subjects))))
    ax.barh(subjects, values)
    ax.set_xlim(0, 1)
    ax.set_xlabel(f"pass@{k}")
    ax.set_title(f"pass@{k} by subject ({results.model_id})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


def plot_pass_by_level(
    results: BaselineResults,
    k: int = 1,
    output_path: Path | str | None = None,
):
    """Bar chart of pass@k by MATH difficulty level (1..5)."""
    import matplotlib.pyplot as plt

    by_level = pass_at_k_by_level(results, k)
    levels = list(by_level.keys())
    values = [by_level[lvl] for lvl in levels]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar([str(lvl) for lvl in levels], values)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Difficulty level")
    ax.set_ylabel(f"pass@{k}")
    ax.set_title(f"pass@{k} by level ({results.model_id})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig
