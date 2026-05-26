"""Baseline-evaluation Modal entrypoint for the milestone (Tier A).

Loads MATH-500, samples G completions per prompt with vLLM, scores with
``OutcomeRewardModule``, and serializes a ``BaselineResults``-shaped JSON for
local analysis.

Two main use cases:

1. Student baseline:
   modal run modal_app/baseline_eval.py --model-id Qwen/Qwen3-1.7B-Base

2. Teacher reference (chat-templated):
   modal run modal_app/baseline_eval.py --model-id Qwen/Qwen3-8B --chat-template

The first run will build the image (~5 min) and pull model weights into a
Modal Volume. Subsequent runs reuse the cache.
"""

from __future__ import annotations

import modal

# Heavy stack lives in the image; only Modal SDK is needed locally.
# Base image must include the CUDA toolkit (nvcc + headers) — vLLM 0.21+
# JIT-compiles multiple kernels (FlashInfer sampling, DeepGEMM FP8 probe,
# Triton) on first use, all of which fail on debian_slim. The kernels are
# cached after the first run.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch>=2.4",
        "vllm>=0.6",
        "transformers>=4.50",
        "datasets>=2.20",
        "math-verify>=0.5",
        "numpy>=1.26",
        "pydantic>=2.7",
        "sentencepiece",
    )
    .add_local_python_source("lora_reward_density")
)

# Persistent HF cache so model weights don't re-download every run.
hf_cache = modal.Volume.from_name("lrd-hf-cache", create_if_missing=True)

app = modal.App("lora-reward-density-baseline-eval", image=image)


@app.function(
    gpu="H100",
    timeout=5400,  # 90 min — covers 8B teacher on full MATH-500
    volumes={"/hf-cache": hf_cache},
)
def run_baseline_eval(
    model_id: str,
    num_prompts: int,
    n_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    chat_template: bool,
    max_model_len: int | None,
) -> dict:
    import os
    import time

    # Route all HF caches into the Volume mount.
    os.environ["HF_HOME"] = "/hf-cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/hf-cache/huggingface/hub"

    from lora_reward_density.data import (
        DEFAULT_BASE_MODEL_TEMPLATE,
        load_math500,
    )
    from lora_reward_density.outcome_reward import OutcomeRewardModule
    from lora_reward_density.rollout import SamplingConfig, VLLMRolloutEngine

    # For chat-tuned models, load raw problems and wrap with the model's chat
    # template; for base models, use the standard "Problem: ... Solution: "
    # template directly.
    if chat_template:
        examples = load_math500(
            num_examples=num_prompts,
            prompt_template="{problem}",
            seed=seed,
            cache_dir="/hf-cache/datasets",
        )
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        instruction = (
            "Solve the following math problem step by step. "
            "Put your final answer inside \\boxed{}.\n\n"
        )
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": instruction + ex.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in examples
        ]
    else:
        examples = load_math500(
            num_examples=num_prompts,
            prompt_template=DEFAULT_BASE_MODEL_TEMPLATE,
            seed=seed,
            cache_dir="/hf-cache/datasets",
        )
        prompts = [ex.prompt for ex in examples]

    metadata = [ex.metadata for ex in examples]

    print(f"Loading vLLM for {model_id} ...")
    t0 = time.perf_counter()
    vllm_kwargs: dict = {"dtype": "bfloat16"}
    if max_model_len is not None:
        vllm_kwargs["max_model_len"] = max_model_len
    engine = VLLMRolloutEngine(model_id, **vllm_kwargs)
    load_seconds = time.perf_counter() - t0
    print(f"  vLLM loaded in {load_seconds:.1f}s")

    config = SamplingConfig(
        n=n_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )

    print(f"Sampling {n_samples} completions for {len(prompts)} prompts ...")
    t0 = time.perf_counter()
    batch = engine.rollout(prompts=prompts, prompt_metadata=metadata, config=config)
    rollout_seconds = time.perf_counter() - t0
    print(f"  Rollout done in {rollout_seconds:.1f}s ({batch.num_completions} completions)")

    print("Scoring with math-verify ...")
    t0 = time.perf_counter()
    reward_output = OutcomeRewardModule().score(batch)
    score_seconds = time.perf_counter() - t0
    print(f"  Scored in {score_seconds:.1f}s")

    response_lengths = batch.completion_mask.sum(dim=1).tolist()
    # First 50 completions for qualitative inspection in the report.
    sample_completions = batch.completions[: min(50, batch.num_completions)]

    return {
        "model_id": model_id,
        "dataset": "MATH-500",
        "num_prompts": len(examples),
        "n_samples": n_samples,
        "rewards": reward_output.trajectory_rewards.tolist(),
        "correctness": reward_output.metadata["correctness"].tolist(),
        "response_lengths": response_lengths,
        "group_index": batch.group_index.tolist(),
        "parse_failures": int(reward_output.metadata["parse_failures"]),
        "prompt_metadata": metadata,
        "sample_completions": sample_completions,
        "sampling": {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "seed": seed,
            "chat_template": chat_template,
            "load_seconds": round(load_seconds, 1),
            "rollout_seconds": round(rollout_seconds, 1),
            "score_seconds": round(score_seconds, 1),
        },
    }


@app.local_entrypoint()
def main(
    model_id: str = "Qwen/Qwen3-1.7B-Base",
    num_prompts: int = 500,
    n_samples: int = 8,
    # 4096 (not 1024) because Qwen3-8B with chat template enters a `<think>`
    # reasoning mode that routinely runs 2000+ tokens before producing a final
    # \boxed{}. At 1024, every hard-problem completion truncated mid-reasoning
    # and scored as incorrect. Student doesn't need this much headroom but the
    # extra budget removes a confound when comparing student vs. teacher.
    max_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
    seed: int = 0,
    chat_template: bool = False,
    max_model_len: int | None = None,
    output: str | None = None,
) -> None:
    """Run baseline eval on Modal and save results to a local run dir.

    Args:
        model_id: HF model ID. Default Qwen3-1.7B-Base (student).
        num_prompts: Number of MATH-500 problems (max 500).
        n_samples: G completions per prompt.
        chat_template: Apply the model's chat template (for instruct/teacher models).
        max_model_len: vLLM ``max_model_len`` cap. None = model default.
        output: Output JSON path. None = ``runs/<timestamp>/baseline_eval.json``.
    """
    import json
    from pathlib import Path

    from lora_reward_density.run_dir import create_run_dir

    config = {
        "model_id": model_id,
        "num_prompts": num_prompts,
        "n_samples": n_samples,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "chat_template": chat_template,
        "max_model_len": max_model_len,
    }

    if output is None:
        run = create_run_dir("runs", config=config)
        output_path = run.path / "baseline_eval.json"
        print(f"Run dir: {run.path}")
    else:
        output_path = Path(output)

    result = run_baseline_eval.remote(
        model_id=model_id,
        num_prompts=num_prompts,
        n_samples=n_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        chat_template=chat_template,
        max_model_len=max_model_len,
    )

    output_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"\nWrote {output_path}")

    correctness = result["correctness"]
    print(f"  num_completions = {len(correctness)}")
    print(f"  pass@1 (raw)    = {sum(correctness) / len(correctness):.3f}")
    print(f"  parse_failures  = {result['parse_failures']}")
    print(f"  timings (s)     = {result['sampling']}")
    print("\nNext: from lora_reward_density.analysis import BaselineResults, summary_table")
    print(f'      r = BaselineResults.from_json_path("{output_path}")')
    print("      print(summary_table(r))")
