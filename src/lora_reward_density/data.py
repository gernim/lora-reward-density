"""Data loaders for math reasoning datasets.

Two MATH loaders:
- `load_math500` — the 500-problem *evaluation* subset (HuggingFaceH4/MATH-500).
- `load_math_train` — the ~7.5k-problem *train* split (EleutherAI/hendrycks_math).

Both emit identical `Example` objects via `_build_examples`. MATH-500 is drawn
from the MATH test split, which is disjoint from the train split, so the two
loaders keep training and evaluation cleanly separated. GSM8K can join later.
"""

from __future__ import annotations

import logging
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from lora_reward_density.outcome_reward import GOLD_ANSWER_KEY

logger = logging.getLogger(__name__)

DEFAULT_BASE_MODEL_TEMPLATE = "Problem: {problem}\n\nSolution: "
"""Prompt template for base models (Qwen3-1.7B-Base). Single literal `{problem}`
placeholder, substituted via str.replace so LaTeX braces in the problem don't
get interpreted as format-string fields."""


@dataclass(frozen=True)
class Example:
    """One (prompt, prompt_metadata) pair, ready to feed to a RolloutEngine.

    `metadata` carries `gold_answer` (consumed by OutcomeRewardModule) plus
    `subject` / `level` / `unique_id` for diagnostics and per-subject breakdowns.
    """

    prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _build_examples(
    rows: Sequence[Mapping[str, Any]],
    prompt_template: str,
    *,
    seed: int | None = None,
    num_examples: int | None = None,
) -> list[Example]:
    """Pure transform from raw HF rows to Examples. Factored out for testability.

    Order of operations: validate template, shuffle (if seed), truncate (if
    num_examples), then format. Shuffling-then-truncating is intentional so
    the truncated subset is a random sample, not just the first N rows.
    """
    if "{problem}" not in prompt_template:
        raise ValueError("prompt_template must contain the literal '{problem}' placeholder")

    rows = list(rows)  # materialize once so shuffle + truncate work uniformly
    if seed is not None:
        random.Random(seed).shuffle(rows)
    if num_examples is not None:
        rows = rows[:num_examples]

    examples: list[Example] = []
    for row in rows:
        prompt = prompt_template.replace("{problem}", row["problem"])
        metadata = {
            GOLD_ANSWER_KEY: row["answer"],
            "subject": row["subject"],
            "level": row["level"],
            "unique_id": row["unique_id"],
        }
        examples.append(Example(prompt=prompt, metadata=metadata))
    return examples


def load_math500(
    *,
    num_examples: int | None = None,
    prompt_template: str = DEFAULT_BASE_MODEL_TEMPLATE,
    seed: int | None = None,
    cache_dir: str | None = None,
) -> list[Example]:
    """Load HuggingFaceH4/MATH-500 (test split, 500 problems).

    Args:
        num_examples: If None, return all 500. Otherwise truncate to this many
            after shuffling (so `seed`+`num_examples` gives a deterministic
            random subset).
        prompt_template: Must contain literal `{problem}`. Default suits a base
            (non-chat) model. For chat-tuned teachers, build a chat-templated
            prompt with the tokenizer's `apply_chat_template` and pass the
            rendered string in here (or write a separate loader variant).
        seed: If given, shuffles deterministically before truncation. `None`
            preserves the dataset's native order.
        cache_dir: Override the HF `datasets` cache. On Modal, point this at a
            Volume mount so reruns don't redownload.
    """
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache_dir)
    # load_dataset's return type is broad (Dataset | DatasetDict | IterableDataset).
    # With split="test" we get a Dataset whose rows are dicts; cast for pyright.
    rows = cast(list[Mapping[str, Any]], list(ds))
    return _build_examples(
        rows,
        prompt_template=prompt_template,
        seed=seed,
        num_examples=num_examples,
    )


# The seven subject configs of EleutherAI/hendrycks_math. The dataset has no
# combined "all" config, so the train loader pulls each and concatenates.
HENDRYCKS_MATH_SUBJECTS: tuple[str, ...] = (
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
)

_BOXED_MARKER = r"\boxed"


def _extract_boxed_answer(solution: str) -> str | None:
    r"""Return the contents of the final ``\boxed{...}`` in ``solution``, or None.

    MATH solutions place the final answer in ``\boxed{}``. We scan from the
    last ``\boxed`` occurrence and brace-match so nested braces survive (e.g.
    ``\boxed{\frac{1}{2}}`` -> ``\frac{1}{2}``). Returns None when there is no
    well-formed ``\boxed{...}`` (missing, or unbalanced braces) so the caller
    can drop the row instead of feeding garbage to the verifier.

    Pure transform (no network) — unit-tested directly.
    """
    start = solution.rfind(_BOXED_MARKER)
    if start < 0:
        return None
    i = start + len(_BOXED_MARKER)
    while i < len(solution) and solution[i].isspace():  # tolerate "\boxed {...}"
        i += 1
    if i >= len(solution) or solution[i] != "{":
        return None
    depth = 0
    content_start = i + 1
    while i < len(solution):
        c = solution[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return solution[content_start:i].strip()
        i += 1
    return None  # unbalanced braces


def _normalize_hendrycks_rows(
    raw_rows: Sequence[Mapping[str, Any]],
    *,
    source_config: str,
) -> list[dict[str, Any]]:
    r"""Map EleutherAI/hendrycks_math rows to the MATH-500 row schema.

    hendrycks_math rows expose ``problem``/``level``/``type``/``solution`` but
    no bare ``answer``, ``subject``, or ``unique_id``. We extract the gold
    answer from the final ``\boxed{}`` in ``solution``, alias ``type`` ->
    ``subject`` (its values — "Algebra", "Counting & Probability", ... — match
    MATH-500's ``subject``), and synthesize a stable ``unique_id``. Rows with
    no extractable boxed answer are dropped; the caller infers the dropped
    count from the returned length.

    Pure transform (no network) so it's unit-testable on synthetic rows.
    """
    normalized: list[dict[str, Any]] = []
    for i, row in enumerate(raw_rows):
        answer = _extract_boxed_answer(row["solution"])
        if answer is None:
            continue
        normalized.append(
            {
                "problem": row["problem"],
                "answer": answer,
                "subject": row["type"],
                "level": row["level"],
                "unique_id": f"hendrycks_math/{source_config}/{i}",
            }
        )
    return normalized


def _parse_level(level: Any) -> int | None:
    """Parse the integer difficulty from a MATH ``level`` ('Level 3' -> 3).

    Returns None when the level is missing or unparseable (e.g. the occasional
    'Level ?'), so a difficulty filter drops those rows rather than guessing.
    """
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        m = re.search(r"\d+", level)
        if m is not None:
            return int(m.group())
    return None


def load_math_train(
    *,
    num_examples: int | None = None,
    prompt_template: str = DEFAULT_BASE_MODEL_TEMPLATE,
    seed: int | None = None,
    cache_dir: str | None = None,
    subjects: Sequence[str] = HENDRYCKS_MATH_SUBJECTS,
    levels: Sequence[int] | None = None,
) -> list[Example]:
    r"""Load the MATH *train* split (~7.5k problems) from EleutherAI/hendrycks_math.

    Training counterpart to `load_math500` (eval-only). The MATH train split is
    disjoint from the test split MATH-500 is drawn from, so train/eval stay
    cleanly separated.

    The source schema differs from MATH-500 (no bare ``answer``/``subject``/
    ``unique_id``); `_normalize_hendrycks_rows` reconciles it, then the same
    `_build_examples` path as `load_math500` produces an identical `Example`
    shape for downstream rollout/reward code.

    Args:
        num_examples: If None, return all rows. Otherwise truncate to this many
            after shuffling (so `seed`+`num_examples` gives a deterministic
            random subset).
        prompt_template: Must contain literal `{problem}`. Default suits a base
            (non-chat) model.
        seed: If given, shuffles deterministically before truncation. Strongly
            recommended for training — without it rows come out grouped by
            subject (all algebra first, ...), biasing any truncated subset and
            the rollout ordering.
        cache_dir: Override the HF `datasets` cache. On Modal, point this at a
            Volume mount so reruns don't redownload.
        subjects: Which subject configs to pull. Defaults to all seven.
        levels: If given, keep only problems whose MATH difficulty level (1-5)
            is in this set — a *difficulty filter* (see experiments.md D7). The
            point is GRPO learning signal: a base model gets too-hard problems
            wrong on every sample, giving uniform groups and zero advantage.
            Restricting to the model's learnable band (e.g. {1, 2, 3}) makes
            within-group reward variance — and thus a gradient — far likelier.
            None = all levels.
    """
    from datasets import load_dataset

    rows: list[dict[str, Any]] = []
    dropped = 0
    for subject in subjects:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train", cache_dir=cache_dir)
        raw = cast(list[Mapping[str, Any]], list(ds))
        normalized = _normalize_hendrycks_rows(raw, source_config=subject)
        dropped += len(raw) - len(normalized)
        rows.extend(normalized)

    if dropped:
        logger.warning(
            "load_math_train: dropped %d/%d rows with no extractable \\boxed{} answer",
            dropped,
            dropped + len(rows),
        )

    if levels is not None:
        wanted = set(levels)
        before = len(rows)
        rows = [r for r in rows if _parse_level(r["level"]) in wanted]
        logger.info(
            "load_math_train: kept %d/%d rows in levels %s",
            len(rows),
            before,
            sorted(wanted),
        )

    return _build_examples(
        rows,
        prompt_template=prompt_template,
        seed=seed,
        num_examples=num_examples,
    )
