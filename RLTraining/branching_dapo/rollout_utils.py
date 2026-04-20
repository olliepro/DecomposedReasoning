"""Pure branching rollout helpers shared by the manager and unit tests."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from branching_dapo.config_types import BranchAdvantageIndex

from branching_dapo.bootstrap import ensure_repo_paths

ensure_repo_paths()

from branching_eval.tree_types import BranchTree, LeafRollout  # noqa: E402


@dataclass(frozen=True)
class PromptGroup:
    """One repeated-prompt group that should yield a shared branch tree.

    Args:
        prompt_uid: Original repeated-prompt uid.
        raw_prompt: Raw prompt messages for this group.
        data_source: Exported data source for this group.
        extra_info: Exported extra-info payload for this group.
        reward_model: Exported reward-model payload for this group.
        group_size: Number of repeated prompt copies in the group.

    Returns:
        Dataclass containing one grouped prompt batch.

    Example:
        >>> group = PromptGroup(
        ...     prompt_uid="u1",
        ...     raw_prompt=[{"role": "user", "content": "2+2?"}],
        ...     data_source="math",
        ...     extra_info={},
        ...     group_size=4,
        ... )
        >>> group.group_size
        4
    """

    prompt_uid: str
    raw_prompt: list[dict[str, str]]
    data_source: str
    extra_info: dict[str, object]
    reward_model: dict[str, object]
    group_size: int


@dataclass(frozen=True)
class LeafBatchRecord:
    """One rollout sample returned to `verl`.

    Args:
        prompt_ids: Unpadded prompt token ids.
        response_ids: Unpadded response token ids.
        response_logprobs: Optional response-token logprobs.
        reward_scores: Structured rollout metadata forwarded into reward computation.
        branch_index: Branch-aware uid payload used by the custom advantage estimator.

    Returns:
        Dataclass for one sample emitted by the branching rollout manager.
    """

    prompt_ids: list[int]
    response_ids: list[int]
    response_logprobs: list[float] | None
    reward_scores: dict[str, object]
    branch_index: BranchAdvantageIndex


def extract_prompt_text(raw_prompt: list[dict[str, str]]) -> str:
    """Extract the plain-text user prompt from RLHF chat-message format.

    Args:
        raw_prompt: Raw prompt message list from `RLHFDataset`.

    Returns:
        Plain-text user prompt string.
    """

    assert len(raw_prompt) == 1, "Branching rollout currently expects one user message per prompt."
    message = raw_prompt[0]
    assert message["role"] == "user", "Branching rollout currently expects a user-only prompt."
    content = message["content"]
    assert isinstance(content, str) and content.strip(), "Prompt content must be non-empty text."
    return content


def build_prompt_groups(
    *, non_tensor_batch: Mapping[str, np.ndarray], expected_group_size: int
) -> list[PromptGroup]:
    """Group repeated prompt rows by uid before shared branch expansion.

    Args:
        non_tensor_batch: Non-tensor batch fields from a `DataProto`.
        expected_group_size: Expected number of rows per repeated-prompt group.

    Returns:
        Ordered prompt groups for shared branch expansion.

    Example:
        >>> payload = {
        ...     "uid": np.array(["u1", "u1"], dtype=object),
        ...     "raw_prompt": np.array([[{"role": "user", "content": "hi"}]] * 2, dtype=object),
        ...     "data_source": np.array(["math", "math"], dtype=object),
        ...     "extra_info": np.array([{}, {}], dtype=object),
        ... }
        >>> groups = build_prompt_groups(non_tensor_batch=payload, expected_group_size=2)
        >>> [group.prompt_uid for group in groups]
        ['u1']
    """

    prompt_groups: list[PromptGroup] = []
    prompt_uid_values = non_tensor_batch["uid"]
    raw_prompts = non_tensor_batch["raw_prompt"]
    data_sources = non_tensor_batch["data_source"]
    extra_infos = non_tensor_batch["extra_info"]
    reward_models = non_tensor_batch["reward_model"]
    group_start = 0
    while group_start < len(prompt_uid_values):
        prompt_uid = str(prompt_uid_values[group_start])
        group_stop = group_start
        while group_stop < len(prompt_uid_values) and str(prompt_uid_values[group_stop]) == prompt_uid:
            group_stop += 1
        group_size = group_stop - group_start
        assert group_size == expected_group_size, (
            f"Expected repeated prompt group size {expected_group_size}, found {group_size} for uid={prompt_uid}"
        )
        prompt_groups.append(
            PromptGroup(
                prompt_uid=prompt_uid,
                raw_prompt=list(raw_prompts[group_start]),
                data_source=str(data_sources[group_start]),
                extra_info=dict(extra_infos[group_start]),
                reward_model=dict(reward_models[group_start]),
                group_size=group_size,
            )
        )
        group_start = group_stop
    return prompt_groups


def build_path_node_ids(tree: BranchTree, leaf: LeafRollout) -> tuple[str, ...]:
    """Build ordered node ids from root to one leaf.

    Args:
        tree: Branch tree containing nodes and edges.
        leaf: Leaf rollout row within the tree.

    Returns:
        Ordered node-id path from root to the leaf node.
    """

    parent_by_child = {edge.child_node_id: edge.parent_node_id for edge in tree.edges}
    path_node_ids = [leaf.node_id]
    while path_node_ids[-1] in parent_by_child:
        path_node_ids.append(parent_by_child[path_node_ids[-1]])
    return tuple(reversed(path_node_ids))


def resolve_leaf_branch_metadata(
    *, tree: BranchTree, leaf: LeafRollout, prompt_uid: str, selector_mode: str
) -> BranchAdvantageIndex:
    """Resolve one leaf's branch metadata for reward and advantage computation.

    Args:
        tree: Branch tree produced by the branching executor.
        leaf: One leaf rollout row.
        prompt_uid: Original repeated-prompt uid.
        selector_mode: Selector mode used for this branch tree.

    Returns:
        Branch-advantage index for the leaf.
    """

    path_node_ids = build_path_node_ids(tree=tree, leaf=leaf)
    parent_branch_id = path_node_ids[-2] if len(path_node_ids) > 1 else None
    selected_cluster_id = None
    cluster_name = None
    candidate_pool_key = None
    for branch_point in tree.branch_points:
        if branch_point.node_id != parent_branch_id:
            continue
        candidate_pool_key = branch_point.candidate_pool_key
        for selection in branch_point.selections:
            if selection.selector_mode != selector_mode:
                continue
            if not selection.selected_candidate_ids:
                continue
            selected_candidate_id = selection.selected_candidate_ids[0]
            if selection.cluster_by_candidate_id is not None:
                cluster_name = selection.cluster_by_candidate_id.get(selected_candidate_id)
                selected_cluster_id = cluster_name
    return BranchAdvantageIndex(
        prompt_uid=prompt_uid,
        branch_tree_id=f"{tree.run_id}:{tree.doc_id}:{tree.doc_attempt}",
        leaf_id=leaf.leaf_id,
        leaf_node_id=leaf.node_id,
        path_node_ids=path_node_ids,
        parent_branch_id=parent_branch_id,
        branch_depth=max(len(path_node_ids) - 1, 0),
        selected_cluster_id=selected_cluster_id,
        cluster_name=cluster_name,
        selector_mode=selector_mode,
        candidate_pool_key=candidate_pool_key,
    )


def build_reward_scores(branch_index: BranchAdvantageIndex) -> dict[str, object]:
    """Build rollout metadata payload forwarded through the stock DAPO manager.

    Args:
        branch_index: Branch metadata for one leaf.

    Returns:
        Structured reward-scores payload.
    """

    return {
        "branch_metadata": {
            "prompt_uid": branch_index.prompt_uid,
            "branch_tree_id": branch_index.branch_tree_id,
            "leaf_id": branch_index.leaf_id,
            "leaf_node_id": branch_index.leaf_node_id,
            "path_node_ids": list(branch_index.path_node_ids),
            "parent_branch_id": branch_index.parent_branch_id,
            "branch_depth": branch_index.branch_depth,
            "selected_cluster_id": branch_index.selected_cluster_id,
            "cluster_name": branch_index.cluster_name,
            "selector_mode": branch_index.selector_mode,
            "candidate_pool_key": branch_index.candidate_pool_key,
        }
    }


def summarize_rollout_records(records: list[LeafBatchRecord]) -> dict[str, float]:
    """Aggregate numeric rollout metrics for trainer logging.

    Args:
        records: One leaf record per repeated-prompt sample.

    Returns:
        Numeric metrics suitable for `DataProto.meta_info["timing"]`.
    """

    if not records:
        return {
            "branching/leaf_count": 0.0,
            "branching/branch_depth_mean": 0.0,
            "branching/branch_depth_max": 0.0,
            "branching/clustered_leaf_ratio": 0.0,
            "branching/cluster_count_mean": 0.0,
            "branching/unique_cluster_count_per_prompt_mean": 0.0,
            "branching/selector_mode_cluster_across_ratio": 0.0,
            "branching/selector_mode_random_ratio": 0.0,
        }
    branch_depths = [float(record.branch_index.branch_depth) for record in records]
    cluster_flags = [1.0 if record.branch_index.selected_cluster_id is not None else 0.0 for record in records]
    selector_modes = [record.branch_index.selector_mode for record in records]
    unique_clusters_by_prompt: dict[str, set[str]] = defaultdict(set)
    for record in records:
        cluster_id = record.branch_index.selected_cluster_id
        if cluster_id is not None:
            unique_clusters_by_prompt[record.branch_index.prompt_uid].add(cluster_id)
        else:
            unique_clusters_by_prompt.setdefault(record.branch_index.prompt_uid, set())
    unique_cluster_counts = [float(len(cluster_ids)) for cluster_ids in unique_clusters_by_prompt.values()]
    return {
        "branching/leaf_count": float(len(records)),
        "branching/branch_depth_mean": float(np.mean(branch_depths)),
        "branching/branch_depth_max": float(np.max(branch_depths)),
        "branching/clustered_leaf_ratio": float(np.mean(cluster_flags)),
        "branching/cluster_count_mean": float(np.mean(unique_cluster_counts)),
        "branching/unique_cluster_count_per_prompt_mean": float(np.mean(unique_cluster_counts)),
        "branching/selector_mode_cluster_across_ratio": float(
            np.mean([1.0 if mode == "cluster_across" else 0.0 for mode in selector_modes])
        ),
        "branching/selector_mode_random_ratio": float(
            np.mean([1.0 if mode == "random" else 0.0 for mode in selector_modes])
        ),
    }
