"""Analyze baseline_eval run(s).

Two modes:

    # Single run — headline metrics, per-level / per-subject breakdowns,
    # 3 figures, spot-check completions. Default if no subcommand given.
    .venv/bin/python scripts/analyze_baseline.py [<run-dir>]

    # Comparison — side-by-side metrics table, pass@k bar chart figure
    # saved to `figs/comparisons/<runA>_vs_<runB>.png`.
    .venv/bin/python scripts/analyze_baseline.py compare <runA> <runB> [<runC> ...]

In single mode, omitting `<run-dir>` picks the most recent run under `runs/`
that contains a `baseline_eval.json`.

Labels for comparison figures are derived from each run's `model_id`, with
the HF org prefix stripped (e.g. `Qwen/Qwen3-8B` → `Qwen3-8B`). If two runs
have the same model_id, the run id is appended for disambiguation.
"""

from __future__ import annotations

import sys
from pathlib import Path

from lora_reward_density.analysis import (
    BaselineResults,
    mean_pass_at_k,
    parse_failure_rate,
    pass_at_k_by_level,
    pass_at_k_by_subject,
    plot_pass_at_k_comparison,
    plot_pass_by_level,
    plot_pass_by_subject,
    plot_response_length_hist,
    response_length_stats,
    summary_table,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = REPO_ROOT / "runs"
FIGS_DIR = REPO_ROOT / "figs"


def find_latest_run() -> Path:
    candidates = [d for d in RUNS_DIR.iterdir() if (d / "baseline_eval.json").is_file()]
    if not candidates:
        raise SystemExit(f"No run under {RUNS_DIR} contains baseline_eval.json")
    return max(candidates, key=lambda d: d.name)


def short_label(model_id: str) -> str:
    return model_id.split("/", 1)[-1]


def load_results(run_dir: Path) -> BaselineResults:
    json_path = run_dir / "baseline_eval.json"
    if not json_path.is_file():
        raise SystemExit(f"Missing {json_path}")
    return BaselineResults.from_json_path(json_path)


def analyze_single(run_dir: Path) -> None:
    r = load_results(run_dir)
    fig_dir = FIGS_DIR / run_dir.name
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"# Baseline analysis — {run_dir.name}\n")
    print(
        f"Model: `{r.model_id}`  ·  Dataset: `{r.dataset}`  ·  "
        f"{r.num_prompts} prompts × {r.n_samples} samples\n"
    )

    print("## Summary\n")
    print(summary_table(r, ks=(1, 4, 8)))
    print()

    print("## Headline numbers\n")
    print(f"- pass@1 = {mean_pass_at_k(r, 1):.3f}")
    print(f"- pass@4 = {mean_pass_at_k(r, 4):.3f}")
    print(f"- pass@8 = {mean_pass_at_k(r, 8):.3f}")
    pf_rate = parse_failure_rate(r)
    pf_flag = " ⚠ above 5% — verifier may be undercounting correctness" if pf_rate > 0.05 else ""
    print(f"- parse_failure_rate = {pf_rate:.3%}{pf_flag}")
    print()

    print("## Response length stats\n")
    stats = response_length_stats(r)
    for key, val in stats.items():
        print(f"- {key} = {val:.1f}")
    max_tokens = r.sampling.get("max_tokens")
    if max_tokens and stats["p99"] >= 0.95 * max_tokens:
        print(
            f"\n⚠ p99 length ({stats['p99']:.0f}) is within 5% of "
            f"max_tokens={max_tokens}; some completions are likely truncated."
        )
    print()

    print("## pass@1 by MATH level\n")
    for lvl, val in sorted(pass_at_k_by_level(r, k=1).items()):
        print(f"- level {lvl} = {val:.3f}")
    print()

    print("## pass@1 by subject\n")
    for subj, val in sorted(pass_at_k_by_subject(r, k=1).items(), key=lambda kv: -kv[1]):
        print(f"- {subj}: {val:.3f}")
    print()

    print("## Plots\n")
    plot_pass_by_level(r, k=1, output_path=fig_dir / "pass_by_level.png")
    plot_pass_by_subject(r, k=1, output_path=fig_dir / "pass_by_subject.png")
    plot_response_length_hist(r, output_path=fig_dir / "response_lengths.png")
    for name in ("pass_by_level.png", "pass_by_subject.png", "response_lengths.png"):
        print(f"- {(fig_dir / name).relative_to(REPO_ROOT)}")
    print()

    print("## Sample completions (first 5)\n")
    for i, c in enumerate(r.sample_completions[:5]):
        print(f"--- sample {i} ---")
        print(c)
        print()


def analyze_compare(run_dirs: list[Path], ks: tuple[int, ...] = (1, 4, 8)) -> None:
    if len(run_dirs) < 2:
        raise SystemExit("compare mode requires at least 2 run dirs")

    results: dict[str, BaselineResults] = {}
    for d in run_dirs:
        r = load_results(d)
        label = short_label(r.model_id)
        if label in results:
            label = f"{label} ({d.name})"
        results[label] = r

    labels = list(results.keys())
    n = len(labels)

    print(f"# Comparison: {' vs '.join(labels)}\n")
    for d, label in zip(run_dirs, labels, strict=True):
        r = results[label]
        print(
            f"- **{label}** ← `{d.relative_to(REPO_ROOT)}` "
            f"(max_tokens={r.sampling.get('max_tokens')}, "
            f"chat_template={r.sampling.get('chat_template')})"
        )
    print()

    # Headline pass@k table. Add a Δ column iff exactly two runs.
    show_delta = n == 2
    header_cells = ["Metric", *labels] + (["Δ"] if show_delta else [])
    print("## Headline\n")
    print("| " + " | ".join(header_cells) + " |")
    print("|" + "---|" * len(header_cells))

    def row(name: str, values: list[float], fmt: str = "{:.3f}") -> None:
        cells = [name, *(fmt.format(v) for v in values)]
        if show_delta:
            delta = values[1] - values[0]
            delta_str = fmt.format(delta)
            if delta >= 0:
                delta_str = "+" + delta_str
            cells.append(delta_str)
        print("| " + " | ".join(cells) + " |")

    for k in ks:
        row(f"pass@{k}", [mean_pass_at_k(r, k) for r in results.values()])
    row(
        "exploration gap (pass@8 − pass@1)",
        [mean_pass_at_k(r, 8) - mean_pass_at_k(r, 1) for r in results.values()],
    )
    row(
        "mean response length",
        [response_length_stats(r)["mean"] for r in results.values()],
        fmt="{:.0f}",
    )
    row(
        "parse_failure_rate (reported)",
        [parse_failure_rate(r) for r in results.values()],
        fmt="{:.3%}",
    )
    print()

    # Per-level pass@1.
    all_levels = sorted({lvl for r in results.values() for lvl in pass_at_k_by_level(r, k=1)})
    print("## pass@1 by MATH level\n")
    print("| Level | " + " | ".join(labels) + " |" + (" Δ |" if show_delta else ""))
    print("|" + "---|" * (1 + n + (1 if show_delta else 0)))
    for lvl in all_levels:
        per = [pass_at_k_by_level(r, k=1).get(lvl, 0.0) for r in results.values()]
        cells = [str(lvl), *(f"{v:.3f}" for v in per)]
        if show_delta:
            cells.append(f"{per[1] - per[0]:+.3f}")
        print("| " + " | ".join(cells) + " |")
    print()

    # Comparison figure.
    comparison_dir = FIGS_DIR / "comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    fig_name = "_vs_".join(d.name for d in run_dirs) + ".png"
    fig_path = comparison_dir / fig_name
    plot_pass_at_k_comparison(results, ks=ks, output_path=fig_path)
    print(f"## Figure\n\n- {fig_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "compare":
        run_dirs = [Path(a).resolve() for a in args[1:]]
        if len(run_dirs) < 2:
            raise SystemExit("Usage: analyze_baseline.py compare <runA> <runB> [<runC> ...]")
        analyze_compare(run_dirs)
    elif args:
        run_dir = Path(args[0]).resolve()
        analyze_single(run_dir)
    else:
        run_dir = find_latest_run()
        print(f"# Using latest run: {run_dir.relative_to(REPO_ROOT)}\n")
        analyze_single(run_dir)
