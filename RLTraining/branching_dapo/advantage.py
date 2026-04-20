"""Recursive intra-branch advantage estimator for branching DAPO."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from branching_dapo.bootstrap import ensure_repo_paths
from branching_dapo.config_types import BranchAdvantageIndex, BranchAdvantageSettings
from branching_dapo.runtime_metrics import record_advantage_metrics
from branching_dapo.verl_compat import register_adv_est

ensure_repo_paths()


@dataclass(frozen=True)
class PromptGroupValueState:
    """Cached tree state for recursive node-value computation.

    Args:
        leaf_reward_by_node: Leaf reward by final node id.
        child_ids_by_node: Child node ids grouped by parent node id.

    Returns:
        Dataclass used for recursive value backups.
    """

    leaf_reward_by_node: dict[str, torch.Tensor]
    child_ids_by_node: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class PromptGroupAdvantageComponents:
    """Per-leaf inter, intra, and final advantage tensors for one prompt group.

    Args:
        inter_advantages: Prompt-level centered reward advantages.
        intra_advantages: Recursive intra-branch advantages.
        combined_advantages: Alpha-interpolated final advantages.

    Returns:
        Grouped advantage components aligned with the leaf order.
    """

    inter_advantages: torch.Tensor
    intra_advantages: torch.Tensor
    combined_advantages: torch.Tensor


def parse_branch_indices(index: np.ndarray) -> list[BranchAdvantageIndex]:
    """Parse serialized branch indices from the batch `uid` array.

    Args:
        index: Serialized `uid` array.

    Returns:
        Parsed branch-advantage indices.
    """

    return [BranchAdvantageIndex.from_json(str(raw_value)) for raw_value in index.tolist()]


def build_prompt_group_state(
    *, entries: list[BranchAdvantageIndex], rewards: torch.Tensor
) -> PromptGroupValueState:
    """Build recursive tree state for one prompt group.

    Args:
        entries: Branch-advantage entries for one prompt group.
        rewards: Scalar rewards aligned with the entries.

    Returns:
        Recursive tree state for node-value evaluation.
    """

    leaf_reward_by_node: dict[str, torch.Tensor] = {}
    child_ids_by_node: dict[str, set[str]] = defaultdict(set)
    for entry, reward in zip(entries, rewards):
        path_node_ids = entry.path_node_ids or ("node_root",)
        leaf_reward_by_node[entry.leaf_node_id] = reward
        for parent_node_id, child_node_id in zip(path_node_ids[:-1], path_node_ids[1:]):
            child_ids_by_node[parent_node_id].add(child_node_id)
    return PromptGroupValueState(
        leaf_reward_by_node=leaf_reward_by_node,
        child_ids_by_node={node_id: tuple(sorted(child_ids)) for node_id, child_ids in child_ids_by_node.items()},
    )


def compute_node_value(
    *,
    node_id: str,
    prompt_state: PromptGroupValueState,
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Recursively compute the average reward value for one tree node.

    Args:
        node_id: Tree node id to evaluate.
        prompt_state: Recursive prompt-group state.
        cache: Mutable node-value cache.

    Returns:
        Recursive average value for the node.
    """

    if node_id in cache:
        return cache[node_id]
    child_ids = prompt_state.child_ids_by_node.get(node_id, ())
    if not child_ids:
        cache[node_id] = prompt_state.leaf_reward_by_node[node_id]
        return cache[node_id]
    child_values = torch.stack(
        [compute_node_value(node_id=child_id, prompt_state=prompt_state, cache=cache) for child_id in child_ids]
    )
    cache[node_id] = torch.mean(child_values)
    return cache[node_id]


def compute_inter_prompt_advantages(
    *, rewards: torch.Tensor, settings: BranchAdvantageSettings
) -> torch.Tensor:
    """Compute prompt-level GRPO-style centered rewards.

    Args:
        rewards: Prompt-group scalar rewards.
        settings: Branch-advantage settings.

    Returns:
        Prompt-level inter-branch advantage scalars.
    """

    if rewards.numel() <= 1:
        return torch.zeros_like(rewards)
    centered_rewards = rewards - torch.mean(rewards)
    if not settings.normalize_inter_by_std:
        return centered_rewards
    reward_std = torch.std(rewards, correction=0)
    return centered_rewards / (reward_std + settings.epsilon)


def compute_leaf_intra_advantage(
    *,
    entry: BranchAdvantageIndex,
    prompt_state: PromptGroupValueState,
    cache: dict[str, torch.Tensor],
    settings: BranchAdvantageSettings,
) -> torch.Tensor:
    """Compute recursive intra-branch advantage for one leaf path.

    Args:
        entry: Branch-advantage index for one leaf.
        prompt_state: Recursive prompt-group state.
        cache: Mutable node-value cache.
        settings: Branch-advantage settings.

    Returns:
        Recursive intra-branch advantage scalar for the leaf.
    """

    local_deltas: list[torch.Tensor] = []
    for parent_node_id, chosen_child_id in zip(entry.path_node_ids[:-1], entry.path_node_ids[1:]):
        child_ids = prompt_state.child_ids_by_node.get(parent_node_id, ())
        chosen_value = compute_node_value(node_id=chosen_child_id, prompt_state=prompt_state, cache=cache)
        if len(child_ids) <= 1:
            local_deltas.append(torch.zeros_like(chosen_value))
            continue
        sibling_ids = tuple(child_id for child_id in child_ids if child_id != chosen_child_id)
        assert sibling_ids, f"Expected sibling nodes for branch point {parent_node_id}"
        sibling_values = torch.stack(
            [compute_node_value(node_id=child_id, prompt_state=prompt_state, cache=cache) for child_id in sibling_ids]
        )
        child_values = torch.cat([chosen_value.unsqueeze(0), sibling_values])
        local_delta = chosen_value - torch.mean(sibling_values)
        if settings.normalize_intra_by_std:
            child_std = torch.std(child_values, correction=0)
            local_delta = local_delta / (child_std + settings.epsilon)
        local_deltas.append(local_delta)
    if not local_deltas:
        sample_reward = next(iter(prompt_state.leaf_reward_by_node.values()))
        return rewards_zero(device=sample_reward.device, dtype=sample_reward.dtype)
    return torch.mean(torch.stack(local_deltas))


def rewards_zero(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Return a scalar zero tensor on the requested device.

    Args:
        device: Torch device for the zero scalar.
        dtype: Torch dtype for the zero scalar.

    Returns:
        Zero scalar tensor.
    """

    return torch.tensor(0.0, device=device, dtype=dtype)


def compute_prompt_group_components(
    *,
    entries: list[BranchAdvantageIndex],
    rewards: torch.Tensor,
    settings: BranchAdvantageSettings,
) -> PromptGroupAdvantageComponents:
    """Compute per-leaf inter, intra, and final advantages for one prompt.

    Args:
        entries: Branch indices for the prompt group.
        rewards: Scalar rewards aligned with `entries`.
        settings: Branch-advantage hyperparameters.

    Returns:
        Grouped advantage components aligned with `entries`.
    """

    prompt_state = build_prompt_group_state(entries=entries, rewards=rewards)
    cache: dict[str, torch.Tensor] = {}
    inter_advantages = compute_inter_prompt_advantages(rewards=rewards, settings=settings)
    intra_advantages = torch.stack(
        [
            compute_leaf_intra_advantage(
                entry=entry,
                prompt_state=prompt_state,
                cache=cache,
                settings=settings,
            )
            for entry in entries
        ]
    )
    combined_advantages = settings.alpha * intra_advantages + (1.0 - settings.alpha) * inter_advantages
    return PromptGroupAdvantageComponents(
        inter_advantages=inter_advantages,
        intra_advantages=intra_advantages,
        combined_advantages=combined_advantages,
    )


def summarize_branch_advantage_metrics(
    *,
    token_level_rewards: torch.Tensor,
    index: np.ndarray,
    config: Any,
) -> dict[str, float]:
    """Aggregate branching advantage metrics for one PPO batch.

    Args:
        token_level_rewards: Token-level rewards with scalar outcomes.
        index: Serialized branch-aware `uid` array.
        config: Algorithm config mapping with branching hyperparameters.

    Returns:
        Numeric batch metrics for rollout logging.
    """

    settings = BranchAdvantageSettings.from_algorithm_config(config=config)
    scalar_rewards = token_level_rewards.sum(dim=-1)
    parsed_entries = parse_branch_indices(index=index)
    prompt_indices: dict[str, list[int]] = defaultdict(list)
    for item_index, entry in enumerate(parsed_entries):
        prompt_indices[entry.prompt_uid].append(item_index)

    inter_tensors = []
    intra_tensors = []
    final_tensors = []
    for group_indices in prompt_indices.values():
        group_rewards = scalar_rewards[group_indices]
        group_entries = [parsed_entries[item_index] for item_index in group_indices]
        components = compute_prompt_group_components(
            entries=group_entries,
            rewards=group_rewards,
            settings=settings,
        )
        inter_tensors.append(components.inter_advantages)
        intra_tensors.append(components.intra_advantages)
        final_tensors.append(components.combined_advantages)

    if not inter_tensors:
        return {
            "branching/alpha": float(settings.alpha),
            "branching/adv/inter_mean": 0.0,
            "branching/adv/intra_mean": 0.0,
            "branching/adv/final_mean": 0.0,
            "branching/adv/inter_abs_mean": 0.0,
            "branching/adv/intra_abs_mean": 0.0,
            "branching/adv/final_abs_mean": 0.0,
        }
    inter_values = torch.cat(inter_tensors)
    intra_values = torch.cat(intra_tensors)
    final_values = torch.cat(final_tensors)
    return {
        "branching/alpha": float(settings.alpha),
        "branching/adv/inter_mean": float(torch.mean(inter_values).item()),
        "branching/adv/intra_mean": float(torch.mean(intra_values).item()),
        "branching/adv/final_mean": float(torch.mean(final_values).item()),
        "branching/adv/inter_abs_mean": float(torch.mean(torch.abs(inter_values)).item()),
        "branching/adv/intra_abs_mean": float(torch.mean(torch.abs(intra_values)).item()),
        "branching/adv/final_abs_mean": float(torch.mean(torch.abs(final_values)).item()),
    }


@register_adv_est("branch_interpolated_grpo")
def compute_branch_interpolated_grpo(
    *,
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    config: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute interpolated inter/intra branch advantages for branching DAPO.

    Args:
        token_level_rewards: Token-level rewards with scalar outcome at response end.
        response_mask: Response token mask.
        index: Serialized branch-aware `uid` array.
        config: Algorithm config mapping with branching hyperparameters.

    Returns:
        Token-broadcast advantages and returns.
    """

    settings = BranchAdvantageSettings.from_algorithm_config(config=config)
    scalar_rewards = token_level_rewards.sum(dim=-1)
    parsed_entries = parse_branch_indices(index=index)
    prompt_indices: dict[str, list[int]] = defaultdict(list)
    for item_index, entry in enumerate(parsed_entries):
        prompt_indices[entry.prompt_uid].append(item_index)

    final_scalars = torch.zeros_like(scalar_rewards)
    for group_indices in prompt_indices.values():
        group_rewards = scalar_rewards[group_indices]
        group_entries = [parsed_entries[item_index] for item_index in group_indices]
        components = compute_prompt_group_components(
            entries=group_entries,
            rewards=group_rewards,
            settings=settings,
        )
        final_scalars[group_indices] = components.combined_advantages

    record_advantage_metrics(
        metrics=summarize_branch_advantage_metrics(
            token_level_rewards=token_level_rewards,
            index=index,
            config=config,
        )
    )
    advantages = final_scalars.unsqueeze(-1) * response_mask
    returns = advantages
    return advantages, returns
