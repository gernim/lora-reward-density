
# Reward Density Doesn't Set LoRA Capacity: An Empirical Study of LoRA Rank Requirements in Policy-Gradient Post-Training

[![Final Report](https://img.shields.io/badge/Final%20Report-PDF-red)](paper/final-report.pdf) [![Project Poster](https://img.shields.io/badge/Poster-PDF-blue)](paper/project-poster.pdf) [![Stanford CS224R](https://img.shields.io/badge/Stanford-CS224R%20Spring%202026-cardinal)](https://cs224r.stanford.edu/) [![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

> **[Read the Full Report (PDF)](paper/final-report.pdf)** | **[View the Poster (PDF)](paper/project-poster.pdf)**

A Stanford CS224R final project that asks whether the LoRA rank needed to match full fine-tuning scales with the *information density* of an RL reward signal. Two recent results bracket the question: [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) (Thinking Machines Lab, 2025) shows a rank-1 adapter already matches full fine-tuning for sparse *outcome* RL (~`O(1)` bits/episode), while [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) reports a large speedup from a dense per-token reward (~`O(N)` bits/episode). Together they imply a scaling law: denser rewards should demand higher rank. We test that law directly by running **one GRPO algorithm across three reward densities** and sweeping LoRA rank against full fine-tuning in each.

The answer is a **clean null**: rank-1 LoRA matches full fine-tuning in every regime, and the required rank does *not* rise with reward density. Density's real effect is on **optimization stability**, and it turns on the reward's *structure*, not its raw information content.

## Overview

We train Qwen3-1.7B-Base on MATH with a from-scratch GRPO loss and change only the reward module across three points on the density spectrum:

- **Outcome** (`O(1)`) — one binary deposit on the last token, `math-verify` against the gold answer.
- **Process** (`O(S)`) — one deposit per reasoning step (segmented on `\n`), scored by the open [Math-Shepherd](https://arxiv.org/abs/2312.08935) PRM's per-step plausibility `P(+)`. No access to the gold answer.
- **Distillation** (`O(N)`) — a per-token reverse-KL reward against a Qwen3-8B teacher, `r_t = log π_T(o_t) − log π_θ(o_t)`. The teacher runs on a separate GPU and shares the student's vocabulary.

In each regime we sweep LoRA rank `{1, 4, 16, 64, 256}` and full fine-tuning (FullFT), three seeds each — a `3 × (5×3 LoRA + 3 FullFT) = 54`-cell matrix — and compare held-out greedy MATH accuracy (pass@1). A single advantage rule handles all three regimes by branching on *how densely the reward is deposited*, never on the regime name.

## Key Findings

### 1. Rank-1 LoRA matches full fine-tuning in every regime (a clean null)

LoRA eval accuracy sits inside FullFT's ±1 sd band at every rank, in every regime. The rank needed to match FullFT does not increase with reward density, falsifying the bits-per-episode capacity prediction.

| LoRA rank | Outcome (final) | Process (peak) | Distillation (final) |
|-----------|-----------------|----------------|----------------------|
| 1         | 0.550 ± 0.022   | 0.467 ± 0.031  | 0.427 ± 0.019        |
| 4         | 0.467 ± 0.071   | 0.500 ± 0.024  | 0.407 ± 0.033        |
| 16        | 0.527 ± 0.019   | 0.457 ± 0.034  | 0.450 ± 0.014        |
| 64        | 0.537 ± 0.031   | 0.483 ± 0.040  | 0.440 ± 0.022        |
| 256       | 0.547 ± 0.040   | 0.510 ± 0.008  | 0.430 ± 0.016        |
| **FullFT**| **0.553 ± 0.029** | **0.483 ± 0.060** | **0.420 ± 0.016** |

_Held-out `eval/pass@1`, mean ± sd over 3 seeds, 100 held-out MATH prompts. Process is shown at its pre-collapse peak (see finding 2); outcome/distillation are final. Base model (lr=0) anchor: pass@1 ≈ 0.375._

### 2. Density's real cost is optimization stability, not capacity

Only one regime collapses — and it's not the densest. Process pass@1 climbs to outcome-level accuracy (~0.48) by steps 20–40, then **cliffs to ~0** by steps 60–80 at *every* rank including FullFT. The per-step reward pays for emitting more steps, so completions lengthen until they saturate the 1536-token cap (`frac_truncated → 1.0`) and stop producing a `\boxed{}` answer. The signature is a blown-up advantage scale:

| Regime | Peak advantage std | Stable? |
|--------|--------------------|---------|
| Outcome (`O(1)`)      | 0.64 | ✅ |
| Distillation (`O(N)`) | 1.00 | ✅ |
| Process (`O(S)`)      | 9.05 | ❌ collapses |

_A ~9× advantage acts like a ~9× learning rate — process trains itself off a cliff._

### 3. It's reward *structure*, not raw density

The densest reward is stable; the middle one collapses. Both process and distillation drift their completions to the token cap, but only process's lengthening is harmful:

- **Process** sums a per-step reward into a return-to-go (reverse-cumsum). Over-segmenting on `\n` produces many steps, so the advantage scales like `√S` and there is a direct incentive to emit more steps — whether or not the model ever answers.
- **Distillation** uses each token's reward *directly* (no reverse-cumsum), so its advantage stays unit-scale, and it imitates a teacher whose long reasoning still ends in a boxed answer.

So the blanket claim "dense rewards are hard to optimize" is wrong. The gameable structure — length-correlated step rewards — is the problem, not the bit count.

### 4. The collapse mechanism: instability *precedes* the hack

Across all 18 process cells the advantage spike is an *early* event, not a symptom of long truncated outputs. `advantage_std` peaks at a mean of **step 28** (range 3–55) while completions are still short and not yet truncating — it precedes length saturation (mean step 70) and the pass@1 collapse (~step 79). The causal chain:

```
advantage-magnitude spike  →  length growth  →  truncation  →  no \boxed{} answer  →  pass@1 collapse
        (step ~28)              (step ~70)                                              (step ~79)
```

Once the policy has fully hacked, every group is uniform step-spam with near-zero within-group reward variance, so `advantage_std` falls back to ~0 — the cause is already gone by the time the collapse completes.

## The Bits-per-Episode Hypothesis

The project tests a specific, falsifiable scaling argument assembled from three sources:

| Ingredient | Source | Claim |
|------------|--------|-------|
| Outcome RL is information-sparse | [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/) | outcome RL carries ~`O(1)` bits/episode; rank-1 LoRA matches FullFT |
| Distillation is information-dense | [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/) | a per-token signal carries ~`O(N)` bits/episode (`N` = tokens) |
| Capacity is finite per parameter | [Physics of Language Models](https://arxiv.org/abs/2404.05405) | a trained network stores ~2 bits/parameter; a rank-`r` adapter holds ~`4rd` bits |

Chaining them predicts that the rank needed to match FullFT should grow with reward density. We add **process rewards** (`O(S)`, one signal per reasoning step) as a third, middle-density regime, and measure the rank requirement across all three under one algorithm. **The prediction fails**: the rank requirement is flat in density. We interpret RL as a low-rank nudge to a base model that already possesses the underlying skill — training is not installing new capacity, so the reward's information content does not translate into a demand for trainable parameters.

## One Algorithm, Three Rewards

The GRPO loss is regime-agnostic. The reward module is the only component that changes; everything downstream consumes the same `(token_rewards, step_reward_mask)` deposit contract.

```
   MATH prompts (+ gold answers)
            │
            ▼
   Rollout engine ── student π_θ = Qwen3-1.7B + LoRA-r
            │
            ├──────────────►  RewardModule           ◄── the ONLY regime-specific part
            │                  ├─ outcome      O(1)       ("one algorithm, three rewards")
            │                  ├─ process      O(S)
            │                  └─ distillation O(N)
            │                        │  deposits + step mask
            ▼                        ▼
   student forward pass ───►   GRPO loss
   → log π_θ                   clipped IS surrogate
                               + group-normalized advantage  (reverse-cumsum if sparse,
                               + KL-to-reference penalty       per-token if dense)
                                     │
                                     ▼
                               optimizer.step  →  LoRA-r (or FullFT) weight update
                                     │
                                     └──────── gradient update to the adapter ────────┐
                                                                                      │
        ┌─────────────────────────────────────────────────────────────────────────◄─┘
        ▼
   (next rollout)
```

Three architectural invariants keep the loss regime-agnostic:

- **One algorithm, three rewards.** Any `if regime == ...` branch inside the GRPO loss is a smell; the loss only ever branches on deposit *density*.
- **Rewards carry both granularities.** Every reward module exposes both `token_rewards: [N, T]` and `trajectory_rewards: [N]`, regardless of regime.
- **Sparse vs. dense advantage.** Sparse deposits (outcome, process) become a return-to-go via reverse-cumsum (DeepSeekMath §4.1.3); dense deposits (distillation) are used per-token to avoid a `√T` advantage blow-up.

## Project Structure

```
lora-reward-density/
├── src/lora_reward_density/
│   ├── grpo.py                  # GRPO loss: clipped IS surrogate + group-norm advantage + KL (hand-written)
│   ├── rewards.py               # RewardModule protocol + RewardOutput deposit contract
│   ├── outcome_reward.py        # O(1): math-verify vs gold, one deposit on the last token
│   ├── process_reward.py        # O(S): Math-Shepherd PRM, one deposit per \n-segmented step
│   ├── distillation_reward.py   # O(N): per-token reverse-KL to a Qwen3-8B teacher
│   ├── rollout.py               # RolloutEngine protocol + mock engine for GRPO correctness tests
│   ├── train.py                 # training step: rollout → reward → advantage → GRPO update → eval
│   ├── data.py                  # MATH (hendrycks_math) loading + fixed held-out eval split
│   ├── analysis.py              # run-metric aggregation for the results tables/figures
│   ├── run_dir.py               # reproducible run dirs (git SHA + pip freeze + env snapshot)
│   └── seeding.py               # seed control for the 3-seed error bars
├── modal_app/
│   ├── train.py                 # Modal entrypoint: single cell or the whole --matrix
│   ├── baseline_eval.py         # pre-training pass@k capability envelope (student vs teacher)
│   ├── reward_debug.py          # token-space segmentation + per-step PRM deposit inspector
│   └── smoke_test.py            # GPU sanity check
├── scripts/
│   ├── make_results_figures.py  # regenerates the four report figures from runs/
│   ├── grpo_correctness_harness.py  # validates the hand-written loss against a TRL reference (Δ=0)
│   └── sync_run_metrics.py      # pulls per-cell metrics out of runs/
├── tests/                       # 97 pytest unit tests (CPU-only; no GPU stack required)
├── docs/                        # design.md, runbook.md, experiments.md, results.md, GRPO notes
├── paper/
│   ├── final-report.pdf         # CS224R final report
│   └── project-poster.pdf       # project poster
└── pyproject.toml               # deps; [gpu] extra is deliberately excluded from local installs
```

## Installation

Local development runs the reward/rollout/loss logic on CPU — the heavy GPU stack (vLLM, CUDA, 8B+ models) is deliberately *not* installed locally and runs only on Modal.

```bash
git clone https://github.com/gernim/lora-reward-density.git
cd lora-reward-density

python3.11 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,modal]"
```

Always invoke project tools via the venv (`.venv/bin/<tool>`) rather than bare names.

## Usage

### Run the test suite (local, CPU-only, ~5s)

```bash
.venv/bin/pytest -q                       # full suite (97 tests)
.venv/bin/pytest tests/test_grpo.py       # one module
.venv/bin/ruff check src tests modal_app  # lint
.venv/bin/pyright src tests               # type-check
```

### GPU work on Modal

```bash
# GPU sanity check
.venv/bin/modal run modal_app/smoke_test.py

# Pre-training capability envelope (student vs teacher pass@k)
.venv/bin/modal run modal_app/baseline_eval.py

# A single matrix cell (outcome, rank 16, 100 GRPO steps)
.venv/bin/modal run modal_app/train.py \
    --regime outcome --rank 16 \
    --max-tokens 1536 --learning-rate 3e-5 --num-rollouts 100 \
    --eval-num-prompts 100 --eval-interval-rollouts 10
```

### Launch the full 54-cell matrix

LoRA and FullFT use different learning rates, so the matrix is launched as two `--matrix` invocations (an empty list disables the other arm):

```bash
# LoRA arm: ranks {1,4,16,64,256} × 3 seeds, lr 3e-5
.venv/bin/modal run --detach modal_app/train.py --matrix \
    --matrix-regimes "outcome,process,distillation" \
    --matrix-ranks "1,4,16,64,256" --matrix-lora-seeds "0,1,2" --matrix-fullft-seeds "" \
    --max-tokens 1536 --learning-rate 3e-5 --num-rollouts 100 \
    --eval-num-prompts 100 --eval-interval-rollouts 10

# FullFT arm: 3 seeds, lr 1e-5
.venv/bin/modal run --detach modal_app/train.py --matrix \
    --matrix-regimes "outcome,process,distillation" \
    --matrix-ranks "" --matrix-fullft-seeds "0,1,2" \
    --max-tokens 1536 --learning-rate 1e-5 --num-rollouts 100 \
    --eval-num-prompts 100 --eval-interval-rollouts 10
```

### Regenerate the report figures

```bash
.venv/bin/python scripts/make_results_figures.py    # writes fig1–fig4 from runs/
```

## Experimental Configuration

Every cell uses an identical configuration, so any difference in outcome is attributable to the regime and LoRA rank rather than hyperparameter tuning. Only the LoRA-vs-FullFT learning rate differs (both swept).

| Hyperparameter | Value | Hyperparameter | Value |
|----------------|-------|----------------|-------|
| Prompts per step (`P`)   | 4    | LR (LoRA / FullFT)   | 3×10⁻⁵ / 1×10⁻⁵ |
| Samples per prompt (`G`) | 4    | Weight decay         | 0 |
| Trajectories per step    | 16   | Max gradient norm    | 1.0 |
| Sampling temperature     | 0.7  | Clip ε               | 0.2 |
| Sampling top-`p`         | 0.95 | KL coefficient β     | 0.05 |
| Max completion length    | 1536 | LoRA rank (swept)    | {1, 4, 16, 64, 256} |
| Rollout steps            | 100  | LoRA α               | 32 |
| Optimizer                | AdamW| LoRA placement       | all-linear (attn + MLP) |

**Models.** Student: Qwen3-1.7B-Base. Distillation teacher: Qwen3-8B (separate GPU). **Task:** MATH (`hendrycks_math`), fixed held-out eval split identical across every cell. **Compute:** ~150 H200-hours for the full matrix (+ ~50 A100-hours for the distillation teacher). Every run snapshots its git SHA and pip freeze, and the from-scratch GRPO loss is validated against a TRL reference to Δ=0 on both loss and gradients.

## References

- **Thinking Machines Lab.** [LoRA Without Regret](https://thinkingmachines.ai/blog/lora/). 2025. *(Outcome RL ≈ `O(1)` bits/episode; rank-1 LoRA matches FullFT.)*
- **Thinking Machines Lab.** [On-Policy Distillation](https://thinkingmachines.ai/blog/on-policy-distillation/). 2025. *(Per-token distillation reward ≈ `O(N)` bits/episode.)*
- **Hu et al.** [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685). ICLR, 2022.
- **Shao et al.** [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300). 2024. *(GRPO; group-relative advantage; process supervision §4.1.3.)*
- **Schulman et al.** [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347). 2017. *(Clipped surrogate objective.)*
- **Wang et al.** [Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations](https://arxiv.org/abs/2312.08935). ACL, 2024. *(Process reward model.)*
- **Lightman et al.** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050). ICLR, 2024. *(Process supervision.)*
- **Biderman et al.** [LoRA Learns Less and Forgets Less](https://arxiv.org/abs/2405.09673). TMLR, 2024. *(LoRA–FullFT gap on SFT / continued pretraining.)*
- **Allen-Zhu & Li.** [Physics of Language Models: Part 3.3, Knowledge Capacity Scaling Laws](https://arxiv.org/abs/2404.05405). 2024. *(~2 bits stored per parameter.)*
- **DeepSeek-AI.** [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948). 2025.

## Author

**Mark Gernitis** — [gernitis@stanford.edu](mailto:gernitis@stanford.edu)

Stanford CS224R: Deep Reinforcement Learning, Spring 2026
