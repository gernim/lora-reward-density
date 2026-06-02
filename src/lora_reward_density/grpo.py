import torch

from lora_reward_density.rewards import RewardOutput


def calculate_advantage(group_index, reward_output):
    return (
        reward_output.trajectory_rewards[group_index] - reward_output.trajectory_rewards.mean()
    ) / reward_output.trajectory_rewards.std(correction=0)


def calculate_importance_ratio(learner_logprobs, sampler_logprobs):
    return torch.exp(learner_logprobs - sampler_logprobs)


def grpo_loss(
    *,
    learner_logprobs,
    sampler_logprobs,
    ref_logprobs,
    completion_mask,
    group_index,
    reward_output: RewardOutput,
    config,
) -> tuple[torch.Tensor, dict[str, float]]:

    device = learner_logprobs.device
    dtype = torch.float32

    epsilon = config.clip_epsilon
    beta = config.kl_beta
    advantage_eps = config.advantage_eps

    #prompt_count = int(group_index.max().item()) + 1
    #group_size = group_index.numel() // prompt_count

    # do some error checking on sizes

    #rewards = reward_output.trajectory_rewards.detach().to(device=device, dtype=dtype)
    #rewards_grouped = rewards.view(prompt_count, group_size)

    # sum and mean per group of G (group_count)
    #mean = rewards_grouped.mean(dim=1, keepdim=True)
    #std = rewards_grouped.std(dim=1, unbiased=True, keepdim=True)

    #advantages = ((rewards_grouped - mean) / (std + advantage_eps)).reshape(-1)

    token_rewards = reward_output.token_rewards.detach().to(device=device, dtype=dtype)
    step_reward_mask = reward_output.step_reward_mask.to(device=device)
    advantages = torch.zeros_like(token_rewards)

    for prompt_group in group_index.unique():
        in_this_group = group_index == prompt_group
        group_deposits = token_rewards[in_this_group][step_reward_mask[in_this_group]]
        group_mean = group_deposits.mean()
        group_std = torch.nan_to_num(group_deposits.std())
        normalized_deposits = torch.where(step_reward_mask[in_this_group], (token_rewards[in_this_group] - group_mean)
                                          / (group_std + advantage_eps), 0.0
                                          )
        advantages[in_this_group] = torch.flip(torch.flip(normalized_deposits, [-1]).cumsum(-1), [-1])

    advantages = advantages * completion_mask

    """
    counts = torch.zeros(group_count, device=device).index_add_(0, group_index, torch.ones_like(rewards))
    sums = torch.zeros(group_count, device=device).index_add_(0, group_index, rewards)
    square_sums = torch.zeros(group_count, device=device).index_add_(0, group_index, rewards * rewards)

    group_mean = sums / counts
    group_var = (square_sums / counts - group_mean * group_mean).clamp_min(0)
    group_std = group_var.sqrt()

    advantage = (rewards - group_mean[group_index]) / (group_std[group_index] + advantage_eps)
    """

    importance_ratio = calculate_importance_ratio(learner_logprobs, sampler_logprobs)
    importance_ratio_clipped = importance_ratio.clamp(1.0 - epsilon, 1.0 + epsilon)

    objective = importance_ratio * advantages   #[N, T] * [N, T]
    clipped_objective = importance_ratio_clipped * advantages
    min_objective = torch.minimum(objective, clipped_objective)

    kl = (learner_logprobs.exp() * (learner_logprobs - ref_logprobs)).clamp(min=0)

    mask = completion_mask.to(device=device, dtype=dtype)
    token_count = mask.sum().clamp_min(1)

    token_loss = -(min_objective - beta * kl) * mask
    loss = token_loss.sum() / token_count

    clipped = (
        (importance_ratio < 1.0 - epsilon) | (importance_ratio > 1.0 + epsilon)
    ) & mask.bool()

    diagnostics = {
        "loss": float(loss.detach().item()),
        "mean_reward": float(reward_output.trajectory_rewards.mean().detach().item()),
        "mean_advantage": float(advantages.mean().detach().item()),
        "advantage_std": float(advantages.std(unbiased=False).detach().item()),
        "ratio_clip_fraction": float(
            clipped.to(dtype=dtype).sum().detach().item() / token_count.detach().item()
        ),
        "kl_to_ref": float((kl * mask).sum().detach().item() / token_count.detach().item()),
    }

    return loss, diagnostics
