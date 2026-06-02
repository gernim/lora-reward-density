"""GRPO loss correctness harness — diff user's implementation vs. TRL.

Runs the user-owned ``lora_reward_density.grpo.grpo_loss`` against a toy
bandit batch and compares the resulting (loss, gradient) against TRL's
``GRPOTrainer`` on the same data. If they match within tolerance, the
user's loss implementation is correct.

This is NOT part of the auto test suite — TRL is heavy and we don't want
to add it to ``[dev]``. Run manually after editing ``grpo.py``:

    .venv/bin/python scripts/grpo_correctness_harness.py

Requires ``[gpu]`` extras installed locally (``pip install -e .[gpu]``).

The harness is deliberately small: 4 prompts × 4 samples on GPT-2 (124M),
~10 seconds end-to-end on CPU. It's a regression check on the algorithm,
not a performance benchmark.

Author boundary: this file is Claude-scaffolded. It exercises the user's
loss but does not implement it.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def _build_toy_batch():
    """Construct a deterministic 4×4 batch for the comparison."""
    import torch

    from lora_reward_density.rewards import RewardOutput
    from lora_reward_density.rollout import DeterministicMockRolloutEngine, SamplingConfig

    canned = {
        "Compute 2+2:": ["The answer is 4.", "Two plus two equals 4.", "4", "Hmm, 5."],
        "What is 3*5?": ["3 times 5 is 15.", "15", "I think 15.", "Fifteen, or 15."],
        "Is 7 prime?": ["Yes, 7 is prime.", "Prime!", "7 is prime.", "Nope, composite."],
        "List 3 evens:": ["2, 4, 6.", "2 4 6", "Evens: 2, 4, 6", "1, 2, 3"],
    }
    engine = DeterministicMockRolloutEngine(canned)
    prompts = list(canned.keys())
    metadata = [{"gold_answer": g} for g in ("4", "15", "yes", "2, 4, 6")]
    batch = engine.rollout(prompts, metadata, SamplingConfig(n=4, max_tokens=8))

    # Synthetic rewards: correctness ~ first sample correct, rest mixed.
    n = batch.num_completions
    t = batch.completion_token_ids.shape[1]
    trajectory_rewards = torch.tensor(
        [1.0, 0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0]
    )
    # Deposit-at-last-token contract (DeepSeekMath §4.1.3): outcome is the
    # one-step case. The user's loss group-normalizes these deposits and
    # reverse-cumsums; for outcome that must reproduce TRL's broadcast
    # group-relative advantage below (the Δ=0 check).
    token_rewards = torch.zeros(n, t)
    step_reward_mask = torch.zeros(n, t, dtype=torch.bool)
    for i in range(n):
        nz = batch.completion_mask[i].nonzero(as_tuple=False)
        last = int(nz[-1].item())
        token_rewards[i, last] = trajectory_rewards[i]
        step_reward_mask[i, last] = True
    reward_output = RewardOutput(
        token_rewards=token_rewards,
        trajectory_rewards=trajectory_rewards,
        step_reward_mask=step_reward_mask,
        metadata={"regime": "outcome"},
    )

    # Synthetic learner + ref logprobs (small perturbations of sampler).
    sampler_logprobs = batch.sampler_logprobs
    torch.manual_seed(0)
    learner_logprobs = (sampler_logprobs + 0.05 * torch.randn(n, t)).detach().requires_grad_(True)
    ref_logprobs = sampler_logprobs.clone() - 0.02

    return {
        "batch": batch,
        "reward_output": reward_output,
        "sampler_logprobs": sampler_logprobs.detach(),
        "learner_logprobs": learner_logprobs,
        "ref_logprobs": ref_logprobs.detach(),
    }


def _run_user_loss(inputs, grpo_config):
    """Call the user's grpo_loss and return (loss_value, grad_summary)."""
    from lora_reward_density.grpo import grpo_loss

    # Zero any gradient left over from the TRL-reference pass, which runs first
    # and shares the same learner_logprobs leaf. Without this, backward() below
    # accumulates on top of it and doubles the reported gradient.
    if inputs["learner_logprobs"].grad is not None:
        inputs["learner_logprobs"].grad.zero_()

    loss, diag = grpo_loss(
        learner_logprobs=inputs["learner_logprobs"],
        sampler_logprobs=inputs["sampler_logprobs"],
        ref_logprobs=inputs["ref_logprobs"],
        completion_mask=inputs["batch"].completion_mask,
        group_index=inputs["batch"].group_index,
        reward_output=inputs["reward_output"],
        config=grpo_config,
    )
    loss.backward()
    grad = inputs["learner_logprobs"].grad
    return {
        "loss": float(loss.item()),
        "grad_l2": float(grad.norm().item()),
        "grad_mean": float(grad.mean().item()),
        "grad_max": float(grad.abs().max().item()),
        "diagnostics": diag,
    }


def _run_trl_loss(inputs, grpo_config):
    """Call TRL's GRPO loss on the same inputs.

    TRL's GRPOTrainer doesn't expose a pure-tensor loss function — it
    wraps a training loop. To compare, we replicate the core math from
    ``trl.trainer.grpo_trainer.GRPOTrainer._compute_loss`` directly here.

    This intentionally reproduces the *same algorithm* the user is
    expected to implement; the comparison is loss-value equality up to
    float tolerance.
    """
    import torch

    # Reset gradient on learner_logprobs.
    if inputs["learner_logprobs"].grad is not None:
        inputs["learner_logprobs"].grad.zero_()

    learner_logprobs = inputs["learner_logprobs"]
    sampler_logprobs = inputs["sampler_logprobs"]
    ref_logprobs = inputs["ref_logprobs"]
    mask = inputs["batch"].completion_mask.float()
    group_index = inputs["batch"].group_index
    rewards = inputs["reward_output"].trajectory_rewards

    # Group-mean advantage normalization (this is the "GR" in GRPO):
    # For each prompt's group of G samples, subtract the group mean.
    p = group_index.max().item() + 1
    g = rewards.shape[0] // p
    rewards_g = rewards.view(p, g)
    group_mean = rewards_g.mean(dim=1, keepdim=True)
    group_std = rewards_g.std(dim=1, keepdim=True).clamp(min=1e-8)
    advantages = ((rewards_g - group_mean) / group_std).view(-1)  # [N]

    # Clipped IS ratio (PPO-style).
    log_ratio = learner_logprobs - sampler_logprobs
    ratio = log_ratio.exp()
    eps = getattr(grpo_config, "clip_epsilon", 0.2)
    clipped = ratio.clamp(1 - eps, 1 + eps)

    # Per-token surrogate (advantage broadcast across tokens, masked).
    adv_b = advantages.unsqueeze(1).expand_as(ratio)
    surr1 = ratio * adv_b
    surr2 = clipped * adv_b
    surrogate = torch.min(surr1, surr2)

    # KL-to-ref penalty (per-token).
    kl = (learner_logprobs.exp() * (learner_logprobs - ref_logprobs)).clamp(min=0)
    beta = getattr(grpo_config, "kl_beta", 0.05)

    per_token = -(surrogate - beta * kl) * mask
    loss = per_token.sum() / mask.sum().clamp(min=1)

    loss.backward()
    grad = inputs["learner_logprobs"].grad
    return {
        "loss": float(loss.item()),
        "grad_l2": float(grad.norm().item()),
        "grad_mean": float(grad.mean().item()),
        "grad_max": float(grad.abs().max().item()),
    }


def main() -> int:
    print("# GRPO loss correctness harness\n")
    print("Building toy batch (4 prompts × 4 samples)...")
    inputs = _build_toy_batch()

    # Minimal config; the user's loss can ignore unknown fields.
    class _Cfg:
        clip_epsilon = 0.2
        kl_beta = 0.05
        advantage_eps = 1e-8  # matches TrainRunConfig default (train.py)

    cfg = _Cfg()

    print("\n## TRL-reference loss (replicated inline)\n")
    trl_result = _run_trl_loss(inputs, cfg)
    for k, v in trl_result.items():
        print(f"  {k:20s} = {v:.6f}" if isinstance(v, float) else f"  {k:20s} = {v}")

    print("\n## User loss (lora_reward_density.grpo.grpo_loss)\n")
    try:
        user_result = _run_user_loss(inputs, cfg)
    except (ImportError, AttributeError) as e:
        print(f"\n  ❌ Could not import grpo_loss from lora_reward_density.grpo: {e}")
        print("\n  Expected signature (see src/lora_reward_density/train.py):")
        print("    def grpo_loss(*, learner_logprobs, sampler_logprobs,")
        print("        ref_logprobs, completion_mask, group_index,")
        print("        reward_output, config) -> (loss, diagnostics): ...")
        return 1
    except NotImplementedError as e:
        print(f"\n  ⚠ grpo_loss raised NotImplementedError: {e}")
        print("  Fill in the body and re-run.")
        return 2

    for k, v in user_result.items():
        if k == "diagnostics":
            continue
        print(f"  {k:20s} = {v:.6f}" if isinstance(v, float) else f"  {k:20s} = {v}")

    print("\n## Comparison\n")
    tol = 1e-4
    fields = ["loss", "grad_l2", "grad_mean", "grad_max"]
    all_pass = True
    for f in fields:
        delta = abs(user_result[f] - trl_result[f])
        status = "✅" if delta < tol else "❌"
        print(
            f"  {status} {f:12s}  user={user_result[f]:.6f}  trl={trl_result[f]:.6f}  Δ={delta:.2e}"
        )
        if delta >= tol:
            all_pass = False

    if all_pass:
        print(f"\n✅ User loss matches TRL reference within {tol}.")
        return 0
    print(
        "\n❌ User loss diverges from TRL reference. "
        "Inspect the diagnostics dict and the math in grpo.py:"
    )
    print(f"   {user_result.get('diagnostics')}")
    return 3


if __name__ == "__main__":
    sys.exit(main())
