"""Rollout engines: sample G completions per prompt with sampler logprobs.

Independent of reward regime — produces a uniform RolloutBatch shape that all
three regimes consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


@dataclass(frozen=True)
class SamplingConfig:
    """Sampling parameters for a rollout batch.

    `n` is GRPO's group size G — the number of completions sampled per prompt.
    Seed propagation depends on the engine: vLLM threads it into its sampler;
    the mock engine ignores it (canned data).
    """

    n: int
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    seed: int | None = None
    stop_token_ids: tuple[int, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class RolloutBatch:
    """A flat batch of P*G completions, where P prompts each yielded G samples.

    All tensor fields align along dim 0 = P*G. `group_index[i]` identifies which
    prompt produced row i — used by GRPO's group-mean baseline.

    Padding: token tensors are right-padded to T_max, with `completion_mask`
    True at valid positions. EOS counts as valid; positions beyond it are pad.
    """

    prompts: list[str]
    prompt_metadata: list[dict[str, Any]]
    completions: list[str]
    completion_token_ids: torch.Tensor  # [P*G, T_max] long
    completion_mask: torch.Tensor  # [P*G, T_max] bool
    sampler_logprobs: torch.Tensor  # [P*G, T_max] float, logprob under sampler
    group_index: torch.Tensor  # [P*G] long, in [0, P)
    pad_token_id: int

    @property
    def num_prompts(self) -> int:
        return len(self.prompts)

    @property
    def num_completions(self) -> int:
        return len(self.completions)

    @property
    def group_size(self) -> int:
        return self.num_completions // self.num_prompts

    def __post_init__(self) -> None:
        p = len(self.prompts)
        n = len(self.completions)
        if p == 0 or n == 0:
            raise ValueError("RolloutBatch must contain at least one prompt and completion")
        if n % p != 0:
            raise ValueError(f"completions ({n}) not a multiple of prompts ({p})")
        if len(self.prompt_metadata) != p:
            raise ValueError(
                f"prompt_metadata length ({len(self.prompt_metadata)}) != prompts length ({p})"
            )
        for tensor, name in (
            (self.completion_token_ids, "completion_token_ids"),
            (self.completion_mask, "completion_mask"),
            (self.sampler_logprobs, "sampler_logprobs"),
        ):
            if tensor.shape[0] != n:
                raise ValueError(f"{name}.shape[0] ({tensor.shape[0]}) != num_completions ({n})")
        if self.completion_token_ids.shape != self.completion_mask.shape:
            raise ValueError(
                f"completion_token_ids {tuple(self.completion_token_ids.shape)} "
                f"!= completion_mask {tuple(self.completion_mask.shape)}"
            )
        if self.completion_token_ids.shape != self.sampler_logprobs.shape:
            raise ValueError(
                f"completion_token_ids {tuple(self.completion_token_ids.shape)} "
                f"!= sampler_logprobs {tuple(self.sampler_logprobs.shape)}"
            )
        if self.group_index.shape != (n,):
            raise ValueError(f"group_index shape {tuple(self.group_index.shape)} != ({n},)")


class RolloutEngine(Protocol):
    """Protocol for rollout engines.

    Implementations must return a RolloutBatch whose shape invariants hold (see
    RolloutBatch.__post_init__). `prompts` and `prompt_metadata` must be the
    same length; the engine produces `config.n` completions per prompt.
    """

    def rollout(
        self,
        prompts: list[str],
        prompt_metadata: list[dict[str, Any]],
        config: SamplingConfig,
    ) -> RolloutBatch: ...


def _pack_rollout(
    *,
    prompts: list[str],
    prompt_metadata: list[dict[str, Any]],
    completions: list[str],
    token_id_lists: list[list[int]],
    logprob_lists: list[list[float]],
    group_index: list[int],
    pad_token_id: int,
) -> RolloutBatch:
    """Right-pad ragged completion sequences into rectangular tensors."""
    n = len(completions)
    if any(len(t) != len(lp) for t, lp in zip(token_id_lists, logprob_lists, strict=True)):
        raise ValueError("token_ids and logprobs must have equal length per completion")

    t_max = max((len(t) for t in token_id_lists), default=0)
    if t_max == 0:
        raise ValueError("all completions are empty; cannot build a rollout batch")

    token_ids = torch.full((n, t_max), pad_token_id, dtype=torch.long)
    mask = torch.zeros((n, t_max), dtype=torch.bool)
    logprobs = torch.zeros((n, t_max), dtype=torch.float32)
    for i, (toks, lps) in enumerate(zip(token_id_lists, logprob_lists, strict=True)):
        length = len(toks)
        if length == 0:
            continue
        token_ids[i, :length] = torch.tensor(toks, dtype=torch.long)
        mask[i, :length] = True
        logprobs[i, :length] = torch.tensor(lps, dtype=torch.float32)

    return RolloutBatch(
        prompts=prompts,
        prompt_metadata=prompt_metadata,
        completions=completions,
        completion_token_ids=token_ids,
        completion_mask=mask,
        sampler_logprobs=logprobs,
        group_index=torch.tensor(group_index, dtype=torch.long),
        pad_token_id=pad_token_id,
    )


class VLLMRolloutEngine:
    """Rollout engine backed by vLLM. Lazy-imports vllm so the rest of the
    package is usable without vllm installed (e.g. local CPU dev)."""

    def __init__(self, model_id: str, **vllm_kwargs: Any) -> None:
        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vllm is required for VLLMRolloutEngine; install with `pip install -e .[gpu]`"
            ) from e
        self._llm = LLM(model=model_id, **vllm_kwargs)
        tokenizer = self._llm.get_tokenizer()
        pad = tokenizer.pad_token_id
        if pad is None:
            pad = tokenizer.eos_token_id
        if pad is None:
            raise RuntimeError(
                f"tokenizer for {model_id!r} has neither pad_token_id nor eos_token_id"
            )
        self._pad_token_id = int(pad)

    def rollout(
        self,
        prompts: list[str],
        prompt_metadata: list[dict[str, Any]],
        config: SamplingConfig,
    ) -> RolloutBatch:
        from vllm import SamplingParams

        if len(prompts) != len(prompt_metadata):
            raise ValueError(
                f"prompts ({len(prompts)}) and prompt_metadata ({len(prompt_metadata)}) must align"
            )

        params = SamplingParams(
            n=config.n,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            seed=config.seed,
            logprobs=1,
            stop_token_ids=list(config.stop_token_ids) or None,
        )
        outputs = self._llm.generate(prompts, params)

        completions: list[str] = []
        token_id_lists: list[list[int]] = []
        logprob_lists: list[list[float]] = []
        group_index: list[int] = []

        for prompt_idx, output in enumerate(outputs):
            for completion in output.outputs:
                if completion.logprobs is None:
                    raise RuntimeError(
                        "vLLM returned no logprobs; SamplingParams.logprobs must be >= 1"
                    )
                tok_ids = list(completion.token_ids)
                lps: list[float] = []
                for t, tok_id in enumerate(tok_ids):
                    step = completion.logprobs[t]
                    if tok_id not in step:
                        raise RuntimeError(
                            f"chosen token {tok_id} missing from vllm logprob dict at step {t}"
                        )
                    lps.append(step[tok_id].logprob)
                completions.append(completion.text)
                token_id_lists.append(tok_ids)
                logprob_lists.append(lps)
                group_index.append(prompt_idx)

        return _pack_rollout(
            prompts=prompts,
            prompt_metadata=prompt_metadata,
            completions=completions,
            token_id_lists=token_id_lists,
            logprob_lists=logprob_lists,
            group_index=group_index,
            pad_token_id=self._pad_token_id,
        )


class DeterministicMockRolloutEngine:
    """Test-only engine that returns canned completions with synthetic token IDs.

    Build with a {prompt: [completion_1, ..., completion_G]} mapping. Token IDs
    are assigned by splitting on whitespace and looking up in a per-engine vocab.
    Sampler logprobs are uniformly -2.0 (distinguishable but not meaningful).

    Useful for end-to-end testing of reward modules and the future training loop
    without standing up a real model.
    """

    PAD_TOKEN_ID = 0

    def __init__(self, canned: dict[str, list[str]]) -> None:
        self._canned: dict[str, list[str]] = dict(canned)
        self._vocab: dict[str, int] = {}
        for samples in self._canned.values():
            for sample in samples:
                for word in sample.split():
                    if word not in self._vocab:
                        # Reserve 0 for pad; assign 1, 2, 3, ...
                        self._vocab[word] = len(self._vocab) + 1

    def _tokenize(self, text: str) -> list[int]:
        return [self._vocab[word] for word in text.split()]

    def rollout(
        self,
        prompts: list[str],
        prompt_metadata: list[dict[str, Any]],
        config: SamplingConfig,
    ) -> RolloutBatch:
        if len(prompts) != len(prompt_metadata):
            raise ValueError(
                f"prompts ({len(prompts)}) and prompt_metadata ({len(prompt_metadata)}) must align"
            )

        completions: list[str] = []
        token_id_lists: list[list[int]] = []
        logprob_lists: list[list[float]] = []
        group_index: list[int] = []

        for prompt_idx, prompt in enumerate(prompts):
            if prompt not in self._canned:
                raise KeyError(f"prompt not canned in mock: {prompt!r}")
            samples = self._canned[prompt]
            if len(samples) < config.n:
                raise ValueError(
                    f"mock has {len(samples)} canned samples for prompt {prompt!r} "
                    f"but config.n={config.n}"
                )
            for sample in samples[: config.n]:
                tok_ids = self._tokenize(sample)
                completions.append(sample)
                token_id_lists.append(tok_ids)
                logprob_lists.append([-2.0] * len(tok_ids))
                group_index.append(prompt_idx)

        return _pack_rollout(
            prompts=prompts,
            prompt_metadata=prompt_metadata,
            completions=completions,
            token_id_lists=token_id_lists,
            logprob_lists=logprob_lists,
            group_index=group_index,
            pad_token_id=self.PAD_TOKEN_ID,
        )
