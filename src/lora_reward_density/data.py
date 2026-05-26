"""Data loaders for math reasoning datasets.

For the milestone, we only need MATH-500 (Hendrycks et al.'s 500-problem
evaluation subset, packaged by HuggingFaceH4). GSM8K can join later.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from lora_reward_density.outcome_reward import GOLD_ANSWER_KEY

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
