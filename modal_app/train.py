"""Training Modal entrypoint for the 24-cell GRPO matrix.

Mirrors the shape of ``baseline_eval.py``. Loads MATH train + held-out
subset, instantiates the regime-specific reward module, calls
``lora_reward_density.train.train`` which dispatches to the user-owned
``training_step``.

Example invocations:

    # Outcome regime, rank 16, 200 GRPO steps
    modal run modal_app/train.py --regime outcome --rank 16 --num-rollouts 200

    # Distillation regime, FullFT, seed 2
    modal run modal_app/train.py --regime distillation --full-ft --seed 2

The Modal image uses the same CUDA-devel base as baseline_eval (D1) so
vLLM/FlashInfer/DeepGEMM JIT works. Adds peft/accelerate/trl and a
``lrd-training-runs`` Volume for checkpoint persistence.
"""

from __future__ import annotations

import modal

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch>=2.4",
        "transformers>=4.50",
        "peft>=0.13",
        "accelerate>=1.0",
        "trl>=0.11",
        "vllm>=0.6",
        "datasets>=2.20",
        "math-verify>=0.5",
        "numpy>=1.26",
        "pydantic>=2.7",
        "sentencepiece",
        "wandb>=0.18",
    )
    .add_local_python_source("lora_reward_density")
)

hf_cache = modal.Volume.from_name("lrd-hf-cache", create_if_missing=True)
runs_volume = modal.Volume.from_name("lrd-training-runs", create_if_missing=True)

app = modal.App("lora-reward-density-train", image=image)


# The `wandb` Modal secret (containing WANDB_API_KEY) is injected into the
# container so wandb.init can authenticate — `wandb login` on your laptop only
# configures the local machine, not the remote container. Logging itself
# activates only when `wandb_project` is set (defaults to "lora-reward-density");
# `_maybe_init_wandb` degrades gracefully if the key is absent, so a missing or
# misnamed key warns and continues rather than crashing the run.
@app.function(
    gpu="H100",
    timeout=14400,  # 4 hr — covers a FullFT distillation run with margin
    volumes={
        "/hf-cache": hf_cache,
        "/training-runs": runs_volume,
    },
    secrets=[modal.Secret.from_name("wandb", required_keys=["WANDB_API_KEY"])],
)
def run_training(
    *,
    regime: str,
    lora_rank: int | None,
    num_rollouts: int,
    batch_prompts: int,
    samples_per_prompt: int,
    max_tokens: int,
    temperature: float,
    learning_rate: float,
    clip_epsilon: float,
    kl_beta: float,
    advantage_eps: float,
    seed: int,
    train_levels: tuple[int, ...] | None,
    eval_num_prompts: int,
    eval_interval_rollouts: int,
    checkpoint_interval_rollouts: int,
    student_model_id: str,
    wandb_project: str | None,
    wandb_run_name: str | None,
    run_id: str,
) -> dict:
    """Heavy training pass on a single H100. Returns a summary dict."""
    import os
    from pathlib import Path

    os.environ["HF_HOME"] = "/hf-cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/hf-cache/huggingface/hub"

    # Lazy imports — keep Modal image build deterministic.
    from lora_reward_density.data import (
        DEFAULT_BASE_MODEL_TEMPLATE,
        load_math500,
        load_math_train,
    )
    from lora_reward_density.outcome_reward import OutcomeRewardModule
    from lora_reward_density.train import TrainRunConfig, train

    try:
        from lora_reward_density.grpo import grpo_loss
    except ImportError as e:
        raise RuntimeError(
            "Could not import grpo_loss from lora_reward_density.grpo. "
            "This is the user-owned implementation per design.md §10. "
            "Define `def grpo_loss(*, learner_logprobs, sampler_logprobs, "
            "ref_logprobs, completion_mask, group_index, reward_output, "
            "config) -> (loss, diagnostics)` in grpo.py and re-run."
        ) from e

    # Build the per-run config.
    config = TrainRunConfig(
        student_model_id=student_model_id,
        lora_rank=lora_rank,
        regime=regime,
        batch_prompts=batch_prompts,
        samples_per_prompt=samples_per_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        num_rollouts=num_rollouts,
        learning_rate=learning_rate,
        clip_epsilon=clip_epsilon,
        kl_beta=kl_beta,
        advantage_eps=advantage_eps,
        seed=seed,
        train_levels=train_levels,
        eval_interval_rollouts=eval_interval_rollouts,
        eval_num_prompts=eval_num_prompts,
        checkpoint_interval_rollouts=checkpoint_interval_rollouts,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    # Reward module — regime dispatch. Process / distillation modules don't
    # exist yet (Tier 2, see design.md §9.1 / §9.2); fall through with a
    # clear error for now.
    if regime == "outcome":
        reward_module = OutcomeRewardModule()
    elif regime == "process":
        raise NotImplementedError(
            "Process reward module not yet built (Tier 2). "
            "See design.md §9.1 for the planned interface."
        )
    elif regime == "distillation":
        raise NotImplementedError(
            "Distillation reward module not yet built (Tier 2). "
            "See design.md §9.2 + §9.3 for the planned interface."
        )
    else:
        raise ValueError(f"Unknown regime: {regime!r}")

    # Data: the MATH train split (~7.5k, EleutherAI/hendrycks_math) is the
    # sampling pool; eval is a fixed held-out subset of MATH-500 (last
    # `eval_num_prompts` after a seed=0 shuffle — same across all 24 cells for
    # comparability, see design.md §9.7). The MATH train split is disjoint from
    # the test split MATH-500 is drawn from, so train and eval don't overlap.
    eval_examples = load_math500(
        num_examples=500,
        prompt_template=DEFAULT_BASE_MODEL_TEMPLATE,
        seed=0,  # Eval split deterministic regardless of training seed.
        cache_dir="/hf-cache/datasets",
    )[-eval_num_prompts:]
    train_examples = load_math_train(
        prompt_template=DEFAULT_BASE_MODEL_TEMPLATE,
        seed=0,  # Deterministic pool order; per-rollout sampling varies by config.seed in train().
        cache_dir="/hf-cache/datasets",
        levels=train_levels,  # difficulty filter (D7); None = all levels
    )
    train_prompts = [ex.prompt for ex in train_examples]
    train_metadata = [ex.metadata for ex in train_examples]
    eval_prompts = [ex.prompt for ex in eval_examples]
    eval_metadata = [ex.metadata for ex in eval_examples]

    # Rollout engine factory — for now, transformers.generate via a thin
    # wrapper. Swap to vLLM-with-weight-sync when sampling becomes the
    # bottleneck (see design.md §3.7 / project-plan.md §5.2).
    def rollout_engine_factory(model, tokenizer):
        return _TrainerGenerateRollout(model, tokenizer)

    run_dir = Path("/training-runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        __import__("json").dumps(vars(config), indent=2, sort_keys=True, default=str)
    )

    summary = train(
        config,
        grpo_loss=grpo_loss,
        grpo_config=config,
        run_dir=run_dir,
        train_prompts=train_prompts,
        train_metadata=train_metadata,
        eval_prompts=eval_prompts,
        eval_metadata=eval_metadata,
        reward_module=reward_module,
        rollout_engine_factory=rollout_engine_factory,
    )

    # Persist volume changes (checkpoints + metrics.jsonl).
    runs_volume.commit()
    return summary


# ----------------------------------------------------------------------------
# In-process rollout via transformers.generate
# ----------------------------------------------------------------------------


class _TrainerGenerateRollout:
    """``RolloutEngine`` implementation using ``transformers.generate`` on
    the trainee model directly. Slower than vLLM but avoids the weight-sync
    headache during training.
    """

    def __init__(self, model, tokenizer):
        self._model = model
        self._tokenizer = tokenizer
        # Batched .generate() requires left-padding (right-padding wedges pads
        # between a short prompt and its generation, corrupting shorter rows),
        # and RolloutBatch's contract is that prompt_token_ids are left-padded.
        # Set on the instance (not a per-call kwarg) for transformers-version
        # robustness; every use of this engine's tokenizer is generation.
        self._tokenizer.padding_side = "left"

    def rollout(self, prompts, prompt_metadata, config):
        import torch

        from lora_reward_density.rollout import RolloutBatch

        device = next(self._model.parameters()).device
        # Tokenizer is left-padded (set in __init__); prompt_ids/prompt_mask are
        # therefore left-padded, satisfying RolloutBatch's prompt contract.
        encoded = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(
            device
        )
        prompt_ids = encoded.input_ids
        prompt_mask = encoded.attention_mask

        # Replicate each prompt G times so a single .generate produces P*G
        # completions, matching the canonical grouped-by-prompt order.
        g = config.n
        prompt_ids = prompt_ids.repeat_interleave(g, dim=0)
        prompt_mask = prompt_mask.repeat_interleave(g, dim=0)

        # temperature == 0 means greedy (used by run_periodic_eval). transformers
        # rejects temperature=0 with do_sample=True, so branch explicitly and omit
        # the sampling warpers entirely for greedy decoding.
        do_sample = config.temperature > 0.0
        gen_kwargs = {
            "input_ids": prompt_ids,
            "attention_mask": prompt_mask,
            "max_new_tokens": config.max_tokens,
            "do_sample": do_sample,
            "return_dict_in_generate": True,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = config.temperature
            gen_kwargs["top_p"] = config.top_p
        with torch.no_grad():
            gen = self._model.generate(**gen_kwargs)

        completion_ids = gen.sequences[:, prompt_ids.shape[1] :]  # [P*G, T]
        completion_mask = completion_ids != self._tokenizer.pad_token_id

        # sampler_logprobs is a ZERO PLACEHOLDER, not the generation logprobs.
        # Nothing consumes the generation-time logprobs anymore: training_step
        # recomputes the IS-ratio "old" logprobs with a full forward (D6), and
        # run_periodic_eval ignores them. Computing them here would require
        # output_logits/output_scores, which accumulate [P*G, T, V] (V≈152k →
        # tens of GB; OOMs at the experiment token budget). The full-vocab
        # accumulation, not the gather, was the cost — so we drop it entirely.
        sampler_logprobs = torch.zeros_like(completion_ids, dtype=torch.float32)

        completions = self._tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
        group_index = torch.arange(len(prompts), device=device).repeat_interleave(g)

        return RolloutBatch(
            prompts=list(prompts),
            prompt_metadata=list(prompt_metadata),
            # Already [P*G, prompt_len] (left-padded by the tokenizer, then
            # repeat_interleave'd above) — aligns row-for-row with completions.
            prompt_token_ids=prompt_ids,
            prompt_attention_mask=prompt_mask.bool(),
            completions=completions,
            completion_token_ids=completion_ids,
            completion_mask=completion_mask,
            sampler_logprobs=sampler_logprobs,
            group_index=group_index,
            pad_token_id=self._tokenizer.pad_token_id,
        )


# ----------------------------------------------------------------------------
# Local entrypoint
# ----------------------------------------------------------------------------


@app.local_entrypoint()
def main(
    regime: str = "outcome",
    rank: int = 16,
    full_ft: bool = False,
    num_rollouts: int = 200,
    batch_prompts: int = 8,
    samples_per_prompt: int = 8,
    max_tokens: int = 4096,
    temperature: float = 0.7,
    learning_rate: float = 1e-6,
    clip_epsilon: float = 0.2,
    kl_beta: float = 0.05,
    advantage_eps: float = 1e-8,
    seed: int = 0,
    train_levels: str | None = None,
    eval_num_prompts: int = 100,
    eval_interval_rollouts: int = 20,
    checkpoint_interval_rollouts: int = 50,
    student_model_id: str = "Qwen/Qwen3-1.7B-Base",
    wandb_project: str | None = "lora-reward-density",
    wandb_run_name: str | None = None,
) -> None:
    """Kick off a single matrix cell on Modal.

    Args:
        regime: one of {outcome, process, distillation}.
        rank: LoRA rank; ignored if ``--full-ft`` is set.
        full_ft: train all parameters (no adapter). For FullFT cells.
        seed: random seed. Use 0/1/2 for the three FullFT seeds.
        temperature: rollout sampling temperature (eval is always greedy).
        clip_epsilon: PPO-style IS ratio clip range [1-ε, 1+ε].
        kl_beta: coefficient on the KL-to-reference penalty.
        advantage_eps: added to group-std denominator for stability.
        train_levels: comma-separated MATH difficulty levels to train on, e.g.
            ``"1,2,3"`` (difficulty filter, experiments.md D7). Omit for all
            levels. Restricting to the model's learnable band gives GRPO
            within-group reward variance.
    """
    import json
    from pathlib import Path

    from lora_reward_density.run_dir import create_run_dir

    lora_rank = None if full_ft else rank
    parsed_levels = tuple(int(x.strip()) for x in train_levels.split(",")) if train_levels else None
    cell_label = f"{regime}_{'fullft' if full_ft else f'rank{rank}'}_seed{seed}"

    run = create_run_dir(
        "runs",
        config={
            "regime": regime,
            "lora_rank": lora_rank,
            "seed": seed,
            "num_rollouts": num_rollouts,
            "batch_prompts": batch_prompts,
            "samples_per_prompt": samples_per_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "learning_rate": learning_rate,
            "clip_epsilon": clip_epsilon,
            "kl_beta": kl_beta,
            "advantage_eps": advantage_eps,
            "train_levels": parsed_levels,
            "student_model_id": student_model_id,
            "cell_label": cell_label,
        },
    )
    print(f"Run dir: {run.path}  ·  cell: {cell_label}")

    summary = run_training.remote(
        regime=regime,
        lora_rank=lora_rank,
        num_rollouts=num_rollouts,
        batch_prompts=batch_prompts,
        samples_per_prompt=samples_per_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        learning_rate=learning_rate,
        clip_epsilon=clip_epsilon,
        kl_beta=kl_beta,
        advantage_eps=advantage_eps,
        seed=seed,
        train_levels=parsed_levels,
        eval_num_prompts=eval_num_prompts,
        eval_interval_rollouts=eval_interval_rollouts,
        checkpoint_interval_rollouts=checkpoint_interval_rollouts,
        student_model_id=student_model_id,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name or cell_label,
        run_id=run.run_id,
    )

    summary_path = run.path / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"\nWrote {summary_path}")
    print(f"  final_eval = {summary.get('final_eval')}")
    print(f"  total_seconds = {summary.get('total_seconds')}")
    print(
        f"  final_checkpoint = {summary.get('final_checkpoint')} "
        f"(in Modal volume `lrd-training-runs`)"
    )
    print("\nTo pull the checkpoint locally:")
    print(
        f"  .venv/bin/modal volume get lrd-training-runs "
        f"{run.run_id} {Path('runs') / run.run_id / 'modal_artifacts'}"
    )
