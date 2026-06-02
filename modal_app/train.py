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
    logprob_micro_batch_size: int | None,
    eval_num_prompts: int,
    eval_interval_rollouts: int,
    checkpoint_interval_rollouts: int,
    student_model_id: str,
    wandb_project: str | None,
    wandb_run_name: str | None,
    run_id: str,
) -> dict:
    """Heavy training pass on a single H100. Returns a summary dict."""
    import logging
    import os
    from pathlib import Path

    # Surface INFO logs in the Modal stream (root defaults to WARNING, which hid
    # the per-step progress line + "Loading student"/"Eval @ step" diagnostics).
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

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
        logprob_micro_batch_size=logprob_micro_batch_size,
        eval_interval_rollouts=eval_interval_rollouts,
        eval_num_prompts=eval_num_prompts,
        checkpoint_interval_rollouts=checkpoint_interval_rollouts,
        wandb_project=wandb_project,
        wandb_run_name=wandb_run_name,
    )

    # Reward module — regime dispatch. Distillation is still Tier 2.
    if regime == "outcome":
        reward_module = OutcomeRewardModule()
    elif regime == "process":
        from transformers import AutoTokenizer

        from lora_reward_density.process_reward import (
            ProcessRewardConfig,
            ProcessRewardModule,
        )

        # ProcessRewardModule segments completions in token space, so it needs
        # the student tokenizer (same model_id train() loads internally → ids
        # match). The 7B PRM loads lazily on first score(), on this GPU container
        # alongside the student + reference (watch H100 memory — see D11).
        student_tokenizer = AutoTokenizer.from_pretrained(student_model_id)
        reward_module = ProcessRewardModule(ProcessRewardConfig(), tokenizer=student_tokenizer)
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
        # Generate in EVAL mode. `load_student` enables gradient checkpointing,
        # which is gated on `self.training`; generating while the model is in
        # train() mode runs the checkpoint path during a no-grad forward and
        # produces GARBAGE completions (degenerate loops / token salad → 0
        # reward). The training loop calls rollout() with the student in train()
        # mode, so flip to eval() here and restore. (Eval already rolls out
        # under eval mode, which is why only training was affected.)
        was_training = self._model.training
        self._model.eval()
        try:
            with torch.no_grad():
                gen = self._model.generate(**gen_kwargs)
        finally:
            if was_training:
                self._model.train()

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
# Rollout+reward reproduction (debug) — run: modal run modal_app/train.py::debug_rollout
# ----------------------------------------------------------------------------


@app.function(
    gpu="A10G",
    timeout=1800,
    volumes={"/hf-cache": hf_cache, "/training-runs": runs_volume},
)
def debug_rollout(
    *,
    model_id: str = "Qwen/Qwen3-1.7B-Base",
    lora_rank: int = 16,
    num_prompts: int = 2,
    samples_per_prompt: int = 4,
    temperature: float = 1.0,
    max_tokens: int = 4096,
    levels: str = "1,2,3",
    regime: str = "outcome",
) -> None:
    """Reproduce ONE training-step rollout+reward EXACTLY — `load_student`
    (LoRA + gradient checkpointing) → the real `_TrainerGenerateRollout` engine
    → the regime's reward module on the resulting RolloutBatch — and print the
    per-completion reward detail.

    `--regime outcome` (default) prints gold / correctness / reward.
    `--regime process` prints the token-space step segmentation, the separator
    token, and each step's deposited PRM score — the check before trusting the
    configured step separator and the `ки` tag-count alignment on real Qwen3
    output (watch for LaTeX/equations split mid-expression by a single `\\n`).
    """
    import os

    os.environ["HF_HOME"] = "/hf-cache/huggingface"
    os.environ["HF_HUB_CACHE"] = "/hf-cache/huggingface/hub"

    from lora_reward_density.data import load_math_train
    from lora_reward_density.rollout import SamplingConfig
    from lora_reward_density.train import TrainRunConfig, load_student

    parsed_levels = [int(x.strip()) for x in levels.split(",")]
    student, tokenizer = load_student(
        model_id,
        lora_rank=lora_rank,
        lora_alpha=32,
        lora_dropout=0.0,
        lora_target_modules=TrainRunConfig().lora_target_modules,
    )
    engine = _TrainerGenerateRollout(student, tokenizer)

    examples = load_math_train(
        num_examples=num_prompts, levels=parsed_levels, seed=0, cache_dir="/hf-cache/datasets"
    )
    prompts = [ex.prompt for ex in examples]
    metadata = [ex.metadata for ex in examples]

    cfg = SamplingConfig(
        n=samples_per_prompt, max_tokens=max_tokens, temperature=temperature, top_p=0.95, seed=0
    )
    batch = engine.rollout(prompts, metadata, cfg)
    print(f"\nnum_completions={batch.num_completions} group_index={batch.group_index.tolist()}")

    if regime == "process":
        _debug_process(batch, tokenizer, metadata)
        return

    from lora_reward_density.outcome_reward import OutcomeRewardModule

    reward_output = OutcomeRewardModule().score(batch)
    correctness = reward_output.metadata.get("correctness")
    print(f"parse_failures={reward_output.metadata.get('parse_failures')}")
    for i in range(batch.num_completions):
        pi = int(batch.group_index[i].item())
        comp = batch.completions[i]
        is_correct = bool(correctness[i]) if correctness is not None else None
        has_box = "\\boxed" in comp
        print(f"\n--- completion {i} (prompt {pi}, level {metadata[pi].get('level')}) ---")
        print("  gold:", repr(metadata[pi]["gold_answer"]))
        print(f"  correct={is_correct} reward={float(reward_output.trajectory_rewards[i].item())}")
        print(f"  chars={len(comp)} has_boxed={has_box}")
        print("  tail:", repr(comp[-250:]))
    print(f"\nMEAN_REWARD = {float(reward_output.trajectory_rewards.mean().item())}")


def _debug_process(batch, tokenizer, metadata) -> None:
    """Print process-regime segmentation + per-step PRM deposits for one batch.

    Shows the separator's token ids, then for each completion the token-space
    step spans, each step's decoded text, and whether/what score was deposited
    at the step's final token. This is the validation that the configured
    separator segments Qwen3 output sensibly and the PRM produced one score per
    step (and that single `\\n` doesn't split LaTeX mid-equation).
    """
    from lora_reward_density.process_reward import (
        ProcessRewardConfig,
        ProcessRewardModule,
        _segment_token_spans,
    )

    cfg = ProcessRewardConfig()
    sep_ids = tokenizer.encode(cfg.step_separator, add_special_tokens=False)
    module = ProcessRewardModule(cfg, tokenizer=tokenizer)
    reward_output = module.score(batch)  # loads the 7B PRM lazily here
    token_rewards = reward_output.token_rewards
    step_mask = reward_output.step_reward_mask
    step_counts = reward_output.metadata["step_counts"]

    print(f"separator={cfg.step_separator!r} -> token ids {sep_ids}")
    for i in range(batch.num_completions):
        pi = int(batch.group_index[i].item())
        valid_len = int(batch.completion_mask[i].sum().item())
        ids = batch.completion_token_ids[i, :valid_len].tolist()
        spans = _segment_token_spans(ids, sep_ids)
        print(f"\n--- completion {i} (prompt {pi}, level {metadata[pi].get('level')}) ---")
        print(
            f"  valid_tokens={valid_len} steps={int(step_counts[i])} deposit_sum={float(reward_output.trajectory_rewards[i]):.3f}"
        )
        for j, (s, e) in enumerate(spans):
            step_txt = tokenizer.decode(ids[s : e + 1]).strip()
            deposited = bool(step_mask[i, e])
            score = float(token_rewards[i, e])
            print(
                f"    step {j} tok[{s}:{e}] deposited={deposited} score={score:.3f} text={step_txt[:120]!r}"
            )
    print(f"\nMEAN deposit_sum = {float(reward_output.trajectory_rewards.mean().item())}")


# ----------------------------------------------------------------------------
# Local entrypoint
# ----------------------------------------------------------------------------


def _pull_metrics_into(run_path, run_id: str) -> None:
    """Best-effort: copy the remote `metrics.jsonl` (written by the container to
    the `lrd-training-runs` volume) into the local run dir, which otherwise only
    has the launch-time snapshot + an empty placeholder. Never raises — a missing
    file (run died early) just prints and moves on."""
    try:
        data = b"".join(runs_volume.read_file(f"{run_id}/metrics.jsonl"))
    except Exception as e:  # noqa: BLE001 — best-effort; volume file may not exist yet
        print(f"  (metrics.jsonl not pulled for {run_id}: {e})")
        return
    (run_path / "metrics.jsonl").write_bytes(data)
    print(f"  pulled {data.count(chr(10).encode())} step(s) of metrics -> {run_path}/metrics.jsonl")


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
    logprob_micro_batch_size: int | None = None,
    eval_num_prompts: int = 100,
    eval_interval_rollouts: int = 20,
    checkpoint_interval_rollouts: int = 50,
    student_model_id: str = "Qwen/Qwen3-1.7B-Base",
    wandb_project: str | None = "lora-reward-density",
    wandb_run_name: str | None = None,
    matrix: bool = False,
    matrix_regimes: str = "outcome",
    matrix_ranks: str = "1,4,16,64,256",
    matrix_fullft_seeds: str = "0,1,2",
) -> None:
    """Kick off a single matrix cell on Modal — or the whole matrix with --matrix.

    Single cell (default): one `run_training` call, blocks for the result.
    Matrix (`--matrix`): fan out the cells across Modal containers (parallel up
    to your GPU quota), then gather. `regime`/`rank`/`full_ft`/`seed` are ignored
    in matrix mode; the cells come from `matrix_*`. All other args (num_rollouts,
    max_tokens, learning_rate, ...) are the *shared* per-cell config — pass the
    values your curve run validated. Use `modal run --detach` for long matrices
    so it survives a disconnect.

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
            levels — the re-baseline showed filtering isn't needed.
        matrix: launch the full matrix instead of one cell.
        matrix_regimes: comma-separated regimes for the matrix (only `outcome`
            is implemented; process/distillation cells will error until built).
        matrix_ranks: comma-separated LoRA ranks → one cell each (seed 0).
        matrix_fullft_seeds: comma-separated seeds → one FullFT cell each.
    """
    import json
    from pathlib import Path

    from lora_reward_density.run_dir import create_run_dir

    parsed_levels = tuple(int(x.strip()) for x in train_levels.split(",")) if train_levels else None

    # Config shared by every cell; only regime/lora_rank/seed/run_id/run_name vary.
    shared = {
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
        "logprob_micro_batch_size": logprob_micro_batch_size,
        "eval_num_prompts": eval_num_prompts,
        "eval_interval_rollouts": eval_interval_rollouts,
        "checkpoint_interval_rollouts": checkpoint_interval_rollouts,
        "student_model_id": student_model_id,
        "wandb_project": wandb_project,
    }

    def _make_run(regime_: str, lora_rank_: int | None, seed_: int):
        label = f"{regime_}_{'fullft' if lora_rank_ is None else f'rank{lora_rank_}'}_seed{seed_}"
        run = create_run_dir(
            "runs",
            suffix=label,  # unique, self-describing run_id per cell (same-second-safe)
            config={
                "regime": regime_,
                "lora_rank": lora_rank_,
                "seed": seed_,
                "cell_label": label,
                **shared,
            },
        )
        return label, run

    if matrix:
        regimes = [r.strip() for r in matrix_regimes.split(",") if r.strip()]
        ranks = [int(r) for r in matrix_ranks.split(",") if r.strip()]
        fullft_seeds = [int(s) for s in matrix_fullft_seeds.split(",") if s.strip()]
        cells: list[tuple[str, int | None, int]] = []
        for rg in regimes:
            cells += [(rg, rk, 0) for rk in ranks]  # LoRA cells, seed 0
            cells += [(rg, None, sd) for sd in fullft_seeds]  # FullFT cells

        print(f"Matrix: launching {len(cells)} cells in parallel (bounded by GPU quota)...")
        handles = []
        for rg, rk, sd in cells:
            label, run = _make_run(rg, rk, sd)
            call = run_training.spawn(
                regime=rg, lora_rank=rk, seed=sd, run_id=run.run_id, wandb_run_name=label, **shared
            )
            handles.append((label, run, call))
            print(f"  spawned {label} → {run.path}")

        print("\nGathering (blocking until all cells finish; safe to detach)...")
        results = {}
        for label, run, call in handles:
            try:
                summary = call.get()
                (run.path / "training_summary.json").write_text(
                    json.dumps(summary, indent=2, sort_keys=True, default=str)
                )
                _pull_metrics_into(run.path, run.run_id)
                results[label] = summary.get("final_eval")
                print(f"  DONE   {label}: final_eval={summary.get('final_eval')}")
            except Exception as e:  # noqa: BLE001 — one cell failing shouldn't abort the matrix
                results[label] = f"FAILED: {e}"
                print(f"  FAILED {label}: {e}")
        print(
            f"\nMatrix complete: {sum(1 for v in results.values() if not str(v).startswith('FAILED'))}"
            f"/{len(cells)} cells succeeded."
        )
        return

    lora_rank = None if full_ft else rank
    cell_label = f"{regime}_{'fullft' if full_ft else f'rank{rank}'}_seed{seed}"

    run = create_run_dir(
        "runs",
        suffix=cell_label,  # self-describing dir name, consistent with matrix mode
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
            "logprob_micro_batch_size": logprob_micro_batch_size,
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
        logprob_micro_batch_size=logprob_micro_batch_size,
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
    _pull_metrics_into(run.path, run.run_id)
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
