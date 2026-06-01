"""Training-loop orchestration for GRPO sweeps.

Per ``docs/design.md`` §10, the **inner training step** is the project's
user-owned algorithm: rollout → reward → GRPO loss → backward → optimizer
step. Everything *around* that inner step — model loading, PEFT/LoRA
wrapping, reference policy snapshotting, logprob gather utilities, W&B
plumbing, periodic eval, checkpointing — is scaffolding and lives here.

Boundary:

* ``training_step(...)`` holds the user-owned inner algorithm
  (rollout → reward → GRPO loss → backward → step).
* ``train(...)`` is the orchestrator that calls ``training_step`` once
  per rollout and handles all the bookkeeping.
* Helpers (``load_student``, ``gather_token_logprobs``, ...) are
  narrow and algorithm-agnostic.

Expected loss signature — the contract ``training_step`` builds against:

    def grpo_loss(
        *,
        learner_logprobs: torch.Tensor,   # [N, T] requires_grad
        sampler_logprobs: torch.Tensor,   # [N, T] detached
        ref_logprobs: torch.Tensor,       # [N, T] detached
        completion_mask: torch.Tensor,    # [N, T] bool
        group_index: torch.Tensor,        # [N] long
        reward_output: RewardOutput,
        config: GRPOConfig,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        ...

The body of ``grpo_loss`` lives in ``grpo.py`` and is user-owned. The
signature here is a system-design choice; change it if the user's design
diverges, but keep both files in sync.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.optim import Optimizer

    from lora_reward_density.rewards import RewardModule, RewardOutput
    from lora_reward_density.rollout import RolloutEngine, SamplingConfig

logger = logging.getLogger(__name__)


def _fmt_duration(seconds: float) -> str:
    """Compact h/m/s for progress lines, e.g. 754 -> '12m34s', 4830 -> '1h20m'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    return f"{s // 3600}h{(s % 3600) // 60:02d}m"


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainRunConfig:
    """Per-run training configuration. Serialized into ``RunDir.config``.

    One config per matrix cell. Regime-specific overrides go in
    ``regime_kwargs`` to avoid an inheritance hierarchy.
    """

    # Model / adaptation
    student_model_id: str = "Qwen/Qwen3-1.7B-Base"
    lora_rank: int | None = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    # Regime
    regime: str = "outcome"  # "outcome" / "process" / "distillation"
    regime_kwargs: dict[str, Any] = field(default_factory=dict)

    # Sampling
    batch_prompts: int = 8
    samples_per_prompt: int = 8
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95

    # Training-pool difficulty filter (MATH levels 1-5). None = all levels.
    # Restricting to the model's learnable band yields within-group reward
    # variance for GRPO (see experiments.md D7). Recorded here so config.json
    # captures which slice of the train pool a run actually saw.
    train_levels: tuple[int, ...] | None = None

    # Memory: chunk the per-token logprob forwards into micro-batches of this
    # many trajectories (None = single forward). Bounds peak logits memory to
    # micro_batch × T × V; needed at experiment scale where N × T × V OOMs.
    # The no-grad forwards (reference, sampler) use this directly; the learner
    # forward pairs it with gradient accumulation (see docs/microbatching.md).
    logprob_micro_batch_size: int | None = None

    # Optimization
    num_rollouts: int = 200
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0  # gradient-norm clip applied AFTER backward (torch.nn.utils.clip_grad_norm_); distinct from clip_epsilon below
    grad_accum_steps: int = 1
    epochs_per_rollout: int = 1

    # GRPO loss hyperparameters — consumed by the user's grpo_loss via the
    # `config` kwarg. Distinct from max_grad_norm: clip_epsilon clips the
    # importance-sampling ratio inside the surrogate objective; max_grad_norm
    # clips the L2 norm of the parameter gradient after backward.
    clip_epsilon: float = 0.2  # PPO-style IS ratio clip range [1-ε, 1+ε]
    kl_beta: float = 0.05  # coefficient on KL-to-reference penalty
    advantage_eps: float = 1e-8  # added to group-std denominator to avoid div by zero

    # Eval + checkpointing
    eval_interval_rollouts: int = 20
    eval_num_prompts: int = 100
    checkpoint_interval_rollouts: int = 50

    # Reproducibility
    seed: int = 0

    # W&B
    wandb_project: str | None = "lora-reward-density"
    wandb_run_name: str | None = None


# ----------------------------------------------------------------------------
# Loss contract
# ----------------------------------------------------------------------------


class GRPOLossFn(Protocol):
    """Callable interface the user's GRPO loss must satisfy.

    Implementation in ``src/lora_reward_density/grpo.py`` is user-owned
    per CS224R AI-tools policy.
    """

    def __call__(
        self,
        *,
        learner_logprobs: torch.Tensor,
        sampler_logprobs: torch.Tensor,
        ref_logprobs: torch.Tensor,
        completion_mask: torch.Tensor,
        group_index: torch.Tensor,
        reward_output: RewardOutput,
        config: Any,
    ) -> tuple[torch.Tensor, dict[str, float]]: ...


# ----------------------------------------------------------------------------
# Student + reference loading
# ----------------------------------------------------------------------------


def load_student(
    model_id: str,
    *,
    lora_rank: int | None,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: tuple[str, ...],
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> tuple[nn.Module, Any]:
    """Load the student. Returns (model, tokenizer).

    ``lora_rank is None`` → FullFT (all params trainable).
    ``lora_rank`` int → wrap with ``peft.get_peft_model`` for LoRA.

    Gradient checkpointing is enabled — memory pressure is tight
    (~45-60 GB on H100 80GB for FullFT 1.7B).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype or torch.bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    model.gradient_checkpointing_enable()
    model.train()

    if lora_rank is not None:
        from peft import LoraConfig, get_peft_model

        peft_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)
        model.print_trainable_parameters()

    return model, tokenizer


def load_reference(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> nn.Module:
    """Load a frozen reference copy of the student for KL-to-ref.

    Matches the student at initialization — base weights only, no LoRA
    adapter, no gradients, eval mode, immutable.
    """
    import torch
    from transformers import AutoModelForCausalLM

    dtype = dtype or torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


# ----------------------------------------------------------------------------
# Logprob utilities
# ----------------------------------------------------------------------------


def gather_token_logprobs(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    """Extract per-position logprobs of the chosen tokens.

    Args:
        logits: ``[N, T, V]`` — model output at each position.
        token_ids: ``[N, T]`` — the tokens we want logprobs for.

    Returns:
        ``[N, T]`` — ``log p(token_ids[i, t] | context)`` per position.

    Callers handle any left/right shift between logits and token_ids.

    Uses the identity ``log p(token) = logit[token] − logsumexp(logits)`` rather
    than ``log_softmax(...).gather(...)``. Both touch the ``[N, T, V]`` input,
    but log_softmax also *returns* an ``[N, T, V]`` tensor that autograd then
    retains for backward; this form keeps only ``[N, T]`` results, cutting the
    grad-path memory by a full vocab-sized tensor. (It does not shrink the
    no-grad peak — that's still bounded by the input logits; reduce N or
    micro-batch the forward for that.)
    """
    import torch

    chosen = logits.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
    return chosen - torch.logsumexp(logits, dim=-1)


def compute_completion_logprobs(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    attention_mask_prompt: torch.Tensor | None = None,
) -> torch.Tensor:
    """Forward through ``model`` on (prompt + completion); return per-token
    logprobs of completion tokens only.

    Returns ``[N, T_completion]``. Gradient flows iff ``model`` is in
    train mode and parameters are requires_grad.
    """
    import torch

    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    if attention_mask_prompt is None:
        attention_mask_prompt = torch.ones_like(prompt_ids, dtype=torch.bool)
    full_mask = torch.cat([attention_mask_prompt, completion_mask], dim=1)

    # Left-pad-correct position_ids. Prompts are left-padded (RolloutBatch
    # contract), so the default arange position_ids would assign positions to
    # pad tokens and shift every real token — corrupting RoPE and making these
    # logprobs disagree with the sampler's (ratio != 1 even at step 0). Derive
    # positions from the mask so the first real token is position 0.
    position_ids = full_mask.long().cumsum(dim=-1) - 1
    position_ids = position_ids.clamp(min=0)

    outputs = model(input_ids=full_ids, attention_mask=full_mask, position_ids=position_ids)
    logits = outputs.logits  # [N, prompt+T_completion, V]

    # Position k is predicted by logits at k-1; align so completion_logits[i, t]
    # predicts completion_ids[i, t].
    prompt_len = prompt_ids.shape[1]
    completion_logits = logits[:, prompt_len - 1 : -1, :]
    return gather_token_logprobs(completion_logits, completion_ids)


def microbatched_logprobs(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
    attention_mask_prompt: torch.Tensor | None = None,
    *,
    micro_batch_size: int | None = None,
) -> torch.Tensor:
    """Memory-bounded ``compute_completion_logprobs``: run the forward on chunks
    of ``micro_batch_size`` trajectories and concatenate the ``[N, T]`` result.

    Bounds peak logits memory to ``micro_batch_size × T × V`` instead of
    ``N × T × V`` (V≈152k → tens of GB; the wall that OOMs the reference /
    sampler forwards at experiment scale). ``None`` (or ``>= N``) runs a single
    forward, identical to ``compute_completion_logprobs``.

    **No-grad only.** Intended for the reference and sampler-recompute forwards.
    The result is correct under autograd, but concatenating grad-carrying chunks
    retains *every* chunk's ``[K, T, V]`` graph until backward, so it gives no
    memory benefit for the learner forward — that path needs per-chunk backward
    (gradient accumulation); see docs/microbatching.md.
    """
    import torch

    n = completion_ids.shape[0]
    if micro_batch_size is None or micro_batch_size >= n:
        return compute_completion_logprobs(
            model, prompt_ids, completion_ids, completion_mask, attention_mask_prompt
        )

    pmask = attention_mask_prompt
    chunks = []
    for start in range(0, n, micro_batch_size):
        sl = slice(start, start + micro_batch_size)
        chunks.append(
            compute_completion_logprobs(
                model,
                prompt_ids[sl],
                completion_ids[sl],
                completion_mask[sl],
                None if pmask is None else pmask[sl],
            )
        )
    return torch.cat(chunks, dim=0)


@contextmanager
def frozen_eval(model: nn.Module) -> Iterator[None]:
    """Context manager: temporarily eval mode + no_grad."""
    import torch

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        if was_training:
            model.train()


# ----------------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------------


def save_checkpoint(
    model: nn.Module,
    tokenizer: Any,
    out_dir: Path,
    *,
    is_lora: bool,
    rollout_step: int,
    metrics: dict[str, float] | None = None,
) -> Path:
    """Save a checkpoint. LoRA → adapter only (~10-100 MB).
    FullFT → full state (~3.4 GB for Qwen3-1.7B in bf16).
    """
    ckpt_dir = out_dir / f"step_{rollout_step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # save_pretrained exists on PreTrainedModel + PeftModel, not on nn.Module
    # base — pyright can't resolve it from the type annotation.
    model.save_pretrained(ckpt_dir, safe_serialization=True)  # type: ignore[reportCallIssue]
    tokenizer.save_pretrained(ckpt_dir)
    if metrics is not None:
        (ckpt_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    logger.info("Saved checkpoint to %s", ckpt_dir)
    return ckpt_dir


# ----------------------------------------------------------------------------
# Training-time eval
# ----------------------------------------------------------------------------


def run_periodic_eval(
    model: nn.Module,
    tokenizer: Any,
    reward_module: RewardModule,
    rollout_engine: RolloutEngine,
    *,
    eval_prompts: list[str],
    eval_metadata: list[dict[str, Any]],
    max_tokens: int,
    seed: int,
    rollout_step: int,
) -> dict[str, float]:
    """Run inference + scoring on a held-out subset; return headline metrics.

    Uses the same ``RolloutEngine`` interface as training rollouts, so the
    metric format matches what the loss sees. Eval is **greedy single-sample**
    (`n=1`): temperature=0 is deterministic, so multiple samples per prompt
    would be identical — pure wasted compute/memory (and the latter OOMs at
    the experiment token budget).

    Returns a dict suitable for W&B logging:
        {
            "eval/pass@1": ...,
            "eval/mean_reward": ...,
            "eval/parse_failure_rate": ...,
            "eval/mean_response_length": ...,
            "eval/rollout_step": rollout_step,
        }
    """
    from lora_reward_density.rollout import SamplingConfig

    cfg = SamplingConfig(
        n=1,  # greedy → deterministic; >1 sample would be identical
        max_tokens=max_tokens,
        temperature=0.0,  # greedy for eval — deterministic learning curves
        top_p=1.0,
        seed=seed,
    )
    with frozen_eval(model):
        batch = rollout_engine.rollout(eval_prompts, eval_metadata, cfg)
    reward_output = reward_module.score(batch)

    # n=1, so correctness is one greedy sample per prompt → pass@1 directly.
    # Regimes without `correctness` (process / distillation) report NaN here.
    correctness = reward_output.metadata.get("correctness")
    pass_at_1 = float("nan") if correctness is None else float(correctness.float().mean().item())

    return {
        "eval/pass@1": pass_at_1,
        "eval/mean_reward": float(reward_output.trajectory_rewards.mean().item()),
        "eval/parse_failure_rate": (
            reward_output.metadata.get("parse_failures", 0) / max(batch.num_completions, 1)
        ),
        "eval/mean_response_length": float(batch.completion_mask.sum(dim=1).float().mean().item()),
        "eval/rollout_step": float(rollout_step),
    }


# ----------------------------------------------------------------------------
# THE INNER STEP — USER-OWNED PER docs/design.md §10
# ----------------------------------------------------------------------------


def training_step(
    *,
    student: nn.Module,
    reference: nn.Module,
    optimizer: Optimizer,
    rollout_engine: RolloutEngine,
    reward_module: RewardModule,
    grpo_loss: GRPOLossFn,
    grpo_config: Any,
    prompts: list[str],
    prompt_metadata: list[dict[str, Any]],
    sampling_config: SamplingConfig,
    epochs: int = 1,
    max_grad_norm: float = 1.0,
    logprob_micro_batch_size: int | None = None,
) -> dict[str, float]:
    """Execute one outer-loop training step.

    **USER-OWNED PER docs/design.md §10.** The inner sequence

        rollout → reward → student forward (with grad) → reference forward
        → grpo_loss(...) → loss.backward() → optimizer.step()
        → return diagnostics

    is the GRPO algorithm in motion and must be implemented by the user.
    Claude can scaffold every helper this function calls
    (``compute_completion_logprobs``, ``frozen_eval``, the rollout engine,
    the reward module, the optimizer), but the assembly is yours.

    Expected return: a dict of float diagnostics for W&B / metrics.jsonl
    (``loss``, ``mean_reward``, ``ratio_clip_fraction``, ``kl_to_ref``,
    ``advantage_std``, ...). Exact contents flow from your loss's
    diagnostics dict; this function just forwards them.
    """

    import torch  # module-level torch is TYPE_CHECKING-only; import at runtime here

    batch = rollout_engine.rollout(prompts, prompt_metadata, sampling_config)
    reward_output = reward_module.score(batch)

    with frozen_eval(reference):
        reference_logprobs = microbatched_logprobs(
            reference,
            batch.prompt_token_ids,
            batch.completion_token_ids,
            batch.completion_mask,
            batch.prompt_attention_mask,
            micro_batch_size=logprob_micro_batch_size,
        )

    student.eval()
    with torch.no_grad():
        sampler_logprobs = microbatched_logprobs(
            student,
            batch.prompt_token_ids,
            batch.completion_token_ids,
            batch.completion_mask,
            batch.prompt_attention_mask,
            micro_batch_size=logprob_micro_batch_size,
        )

    if epochs < 1:
        raise ValueError("epochs must be >= 1")

    diagnostics: dict[str, float] = {}
    for _ in range(epochs):
        student.train()
        learner_logprobs = compute_completion_logprobs(
            student,
            batch.prompt_token_ids,
            batch.completion_token_ids,
            batch.completion_mask,
            batch.prompt_attention_mask,
        )

        loss, diagnostics = grpo_loss(
            learner_logprobs=learner_logprobs,
            sampler_logprobs=sampler_logprobs,
            ref_logprobs=reference_logprobs,
            completion_mask=batch.completion_mask,
            group_index=batch.group_index,
            reward_output=reward_output,
            config=grpo_config,
        )

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(), max_grad_norm, error_if_nonfinite=False
        )
        if torch.isfinite(grad_norm):
            optimizer.step()
            skipped_nonfinite = False
        else:
            logger.warning("non-finite grad norm (%.3g); skipping optimizer step", float(grad_norm))
            skipped_nonfinite = True

        optimizer.zero_grad()
        diagnostics["grad_norm"] = float(grad_norm)
        diagnostics["skipped_nonfinite"] = float(skipped_nonfinite)

    lengths = batch.completion_mask.sum(dim=1).float()
    diagnostics["response_len_mean"] = float(lengths.mean().item())
    diagnostics["response_len_max"] = float(lengths.max().item())
    diagnostics["frac_truncated"] = float(
        (lengths >= sampling_config.max_tokens).float().mean().item()
    )

    return diagnostics


# ----------------------------------------------------------------------------
# Outer orchestration — Claude-scaffolded
# ----------------------------------------------------------------------------


def train(
    config: TrainRunConfig,
    *,
    grpo_loss: GRPOLossFn,
    grpo_config: Any,
    run_dir: Path,
    train_prompts: list[str],
    train_metadata: list[dict[str, Any]],
    eval_prompts: list[str],
    eval_metadata: list[dict[str, Any]],
    reward_module: RewardModule,
    rollout_engine_factory: Any,
) -> dict[str, Any]:
    """Outer training loop. Calls ``training_step`` once per rollout.

    Args:
        config: hyperparameters serialized into the run.
        grpo_loss: the user's loss function (from ``grpo.py``).
        grpo_config: arbitrary config object forwarded to ``grpo_loss``.
        run_dir: where to write metrics.jsonl and checkpoints/.
        train_prompts / train_metadata: full training pool; we sample
            ``config.batch_prompts`` from this per rollout.
        eval_prompts / eval_metadata: held-out set for periodic eval
            (same across all matrix cells for comparability).
        reward_module: regime-specific reward (outcome / process /
            distillation).
        rollout_engine_factory: callable returning a fresh ``RolloutEngine``
            given the trainee model + tokenizer. Lets the caller choose
            between transformers.generate, vLLM-with-sync, or Tinker.

    Returns a summary dict (final eval metrics + total wall time + final
    checkpoint path).
    """
    import random

    import torch

    from lora_reward_density.rollout import SamplingConfig
    from lora_reward_density.seeding import set_seed

    set_seed(config.seed)
    metrics_path = run_dir / "metrics.jsonl"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading student (%s, lora_rank=%s)", config.student_model_id, config.lora_rank)
    student, tokenizer = load_student(
        config.student_model_id,
        lora_rank=config.lora_rank,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=config.lora_target_modules,
    )
    logger.info("Loading reference")
    reference = load_reference(config.student_model_id)

    # Optimizer: only on trainable params (handles PEFT automatically).
    trainable = [p for p in student.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    rollout_engine = rollout_engine_factory(student, tokenizer)

    # W&B init (optional — disabled if no wandb installed or no project).
    wandb_run = _maybe_init_wandb(config, run_dir)

    rng = random.Random(config.seed)
    sampling_config = SamplingConfig(
        n=config.samples_per_prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
        seed=config.seed,
    )

    t_start = time.perf_counter()
    last_eval: dict[str, float] = {}

    with metrics_path.open("w") as metrics_f:
        for rollout_step in range(config.num_rollouts):
            # Sample a batch of training prompts.
            idxs = rng.sample(range(len(train_prompts)), config.batch_prompts)
            prompts = [train_prompts[i] for i in idxs]
            metadata = [train_metadata[i] for i in idxs]

            t_step = time.perf_counter()
            diag = training_step(
                student=student,
                reference=reference,
                optimizer=optimizer,
                rollout_engine=rollout_engine,
                reward_module=reward_module,
                grpo_loss=grpo_loss,
                grpo_config=grpo_config,
                prompts=prompts,
                prompt_metadata=metadata,
                sampling_config=sampling_config,
                epochs=config.epochs_per_rollout,
                max_grad_norm=config.max_grad_norm,
                logprob_micro_batch_size=config.logprob_micro_batch_size,
            )
            step_seconds = time.perf_counter() - t_step

            record = {
                "rollout_step": rollout_step,
                "step_seconds": round(step_seconds, 2),
                **diag,
            }
            metrics_f.write(json.dumps(record) + "\n")
            metrics_f.flush()
            if wandb_run is not None:
                wandb_run.log(record, step=rollout_step)

            # Progress line for the Modal log stream: fraction done, ETA from the
            # running-average step time, and the assurance metrics. (W&B has the
            # live curves; this is for tailing `modal app logs`.)
            done = rollout_step + 1
            elapsed = time.perf_counter() - t_start
            eta = elapsed / done * (config.num_rollouts - done)
            logger.info(
                "[%d/%d %.0f%%] reward=%.3f adv_std=%.3f loss=%.2e kl=%.2e clip=%.2f "
                "| step %s elapsed %s eta %s",
                done,
                config.num_rollouts,
                100.0 * done / config.num_rollouts,
                diag.get("mean_reward", float("nan")),
                diag.get("advantage_std", float("nan")),
                diag.get("loss", float("nan")),
                diag.get("kl_to_ref", float("nan")),
                diag.get("ratio_clip_fraction", float("nan")),
                _fmt_duration(step_seconds),
                _fmt_duration(elapsed),
                _fmt_duration(eta),
            )

            if (rollout_step + 1) % config.eval_interval_rollouts == 0:
                last_eval = run_periodic_eval(
                    student,
                    tokenizer,
                    reward_module,
                    rollout_engine,
                    eval_prompts=eval_prompts,
                    eval_metadata=eval_metadata,
                    max_tokens=config.max_tokens,
                    seed=config.seed,
                    rollout_step=rollout_step,
                )
                # Persist eval to metrics.jsonl too (not just W&B) so the run dir
                # is a self-contained record. Eval lines carry `eval/*` keys;
                # training-step lines carry bare keys — analysis filters on that.
                metrics_f.write(json.dumps(last_eval) + "\n")
                metrics_f.flush()
                if wandb_run is not None:
                    wandb_run.log(last_eval, step=rollout_step)
                logger.info("Eval @ step %d: %s", rollout_step, last_eval)

            if (rollout_step + 1) % config.checkpoint_interval_rollouts == 0:
                save_checkpoint(
                    student,
                    tokenizer,
                    ckpt_dir,
                    is_lora=config.lora_rank is not None,
                    rollout_step=rollout_step,
                    metrics=last_eval or None,
                )

    # Final eval + checkpoint.
    final_eval = run_periodic_eval(
        student,
        tokenizer,
        reward_module,
        rollout_engine,
        eval_prompts=eval_prompts,
        eval_metadata=eval_metadata,
        max_tokens=config.max_tokens,
        seed=config.seed,
        rollout_step=config.num_rollouts - 1,
    )
    with metrics_path.open("a") as f:  # metrics_f is closed by now; append the final eval
        f.write(json.dumps(final_eval) + "\n")
    final_ckpt = save_checkpoint(
        student,
        tokenizer,
        ckpt_dir,
        is_lora=config.lora_rank is not None,
        rollout_step=config.num_rollouts - 1,
        metrics=final_eval,
    )

    total_seconds = time.perf_counter() - t_start
    summary = {
        "final_eval": final_eval,
        "final_checkpoint": str(final_ckpt),
        "last_periodic_eval": last_eval,
        "total_seconds": round(total_seconds, 1),
        "num_rollouts_completed": config.num_rollouts,
    }
    if wandb_run is not None:
        wandb_run.finish()
    return summary


def _maybe_init_wandb(config: TrainRunConfig, run_dir: Path) -> Any:
    """Initialize W&B if available and configured; otherwise return None."""
    if config.wandb_project is None:
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; skipping logging integration")
        return None
    try:
        return wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name or run_dir.name,
            config={k: v for k, v in vars(config).items() if not k.startswith("_")},
            dir=str(run_dir),
        )
    except Exception as e:  # noqa: BLE001 — auth/network failures must not kill training
        # e.g. no WANDB_API_KEY (secret not attached). Metrics still land in
        # metrics.jsonl; we just skip the live dashboard.
        logger.warning("wandb.init failed (%s); continuing without W&B logging", e)
        return None
