"""Tree/leaf helper utilities used by branch executor runtime."""

from __future__ import annotations

from branching_eval.runtime_types import DecodeOutcome, PathState
from branching_eval.tree_types import LeafRollout, TreeNode
from branching_eval.branch_decode_utils import length_tokens_exec, token_traces_from_choice
from vllm_client import GenerationChoice


def leaf_from_choice(*, choice: GenerationChoice, index: int) -> LeafRollout:
    """Build baseline leaf from one generation choice.

    Args:
        choice: Completion choice.
        index: Baseline rollout index.

    Returns:
        Baseline leaf rollout row.
    """

    token_ids = tuple(choice.token_ids) if choice.token_ids is not None else ()
    token_traces = token_traces_from_choice(choice=choice)
    return LeafRollout(
        leaf_id=f"leaf_baseline_{index}",
        node_id="node_root",
        text=str(choice.text),
        token_ids=token_ids,
        tokens=token_traces,
        verification=0,
        length_tokens_total=len(token_ids) if token_ids else len(token_traces),
        length_tokens_exec=length_tokens_exec(text=str(choice.text)),
        stop_reason=str(choice.finish_reason),
        task_metrics={},
    )


def leaf_from_outcome(*, outcome: DecodeOutcome, state: PathState) -> LeafRollout:
    """Build branch leaf from one decode outcome.

    Args:
        outcome: Final decode outcome.
        state: Path state that produced the leaf.

    Returns:
        Branch leaf rollout row.
    """

    final_text = outcome.assistant_prefix
    return LeafRollout(
        leaf_id=f"leaf_{state.node_id}_{len(outcome.token_ids)}",
        node_id=state.node_id,
        text=final_text,
        token_ids=tuple(outcome.token_ids),
        tokens=tuple(outcome.token_traces),
        verification=0,
        length_tokens_total=len(outcome.token_ids),
        length_tokens_exec=length_tokens_exec(text=final_text),
        stop_reason=outcome.stop_reason,
        task_metrics={},
    )


def node_event_payload(*, node: TreeNode) -> dict[str, object]:
    """Build node-created event payload for tree-events JSONL.

    Args:
        node: Tree node row.

    Returns:
        Event payload dictionary.
    """

    return {
        "node_id": node.node_id,
        "parent_node_id": node.parent_node_id,
        "branch_points_used": node.branch_points_used,
    }


def leaf_event_payload(*, leaf: LeafRollout) -> dict[str, object]:
    """Build leaf-completed event payload for tree-events JSONL.

    Args:
        leaf: Leaf rollout row.

    Returns:
        Event payload dictionary.
    """

    return {
        "leaf_id": leaf.leaf_id,
        "node_id": leaf.node_id,
        "length_tokens_total": leaf.length_tokens_total,
        "length_tokens_exec": leaf.length_tokens_exec,
        "stop_reason": leaf.stop_reason,
    }
