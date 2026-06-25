"""Tree/leaf helper utilities used by branch executor runtime."""

from __future__ import annotations

from branching_eval.runtime_types import DecodeOutcome, PathState
from branching_eval.tree_types import LeafRollout, TreeNode
from branching_eval.branch_decode_utils import (
    length_tokens_exec,
    token_traces_from_choice,
)
from vllm_client import GenerationChoice


def leaf_from_choice(
    *, choice: GenerationChoice, index: int, assistant_prefix: str = ""
) -> LeafRollout:
    """Build baseline leaf from one generation choice.

    Args:
        choice: Completion choice.
        index: Baseline rollout index.
        assistant_prefix: Optional assistant prefix included in the request.

    Returns:
        Baseline leaf rollout row.
    """

    token_ids = tuple(choice.token_ids) if choice.token_ids is not None else ()
    token_traces = token_traces_from_choice(choice=choice)
    text = f"{assistant_prefix}{choice.text}"
    return LeafRollout(
        leaf_id=f"leaf_baseline_{index}",
        node_id="node_root",
        text=text,
        token_ids=token_ids,
        tokens=token_traces,
        verification=0,
        length_tokens_total=len(token_ids) if token_ids else len(token_traces),
        length_tokens_exec=length_tokens_exec(text=text),
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
        repeat_stop_reason=outcome.repeat_stop_reason,
        repeat_block_kind=outcome.repeat_block_kind,
        repeat_block_count=outcome.repeat_block_count,
        repeat_last_similarity_ratio=outcome.repeat_last_similarity_ratio,
        steer_phase_token_spans=(
            outcome.steer_phase_token_spans or state.steer_phase_token_spans
        ),
        off_policy_token_spans=(
            outcome.off_policy_token_spans or state.off_policy_token_spans
        ),
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

    payload: dict[str, object] = {
        "leaf_id": leaf.leaf_id,
        "node_id": leaf.node_id,
        "text": leaf.text,
        "length_tokens_total": leaf.length_tokens_total,
        "length_tokens_exec": leaf.length_tokens_exec,
        "stop_reason": leaf.stop_reason,
        "steer_phase_token_spans": [
            [span_start, span_end]
            for span_start, span_end in leaf.steer_phase_token_spans
        ],
        "off_policy_token_spans": [
            [span_start, span_end]
            for span_start, span_end in leaf.off_policy_token_spans
        ],
    }
    if leaf.repeat_stop_reason is not None:
        payload["repeat_stop_reason"] = leaf.repeat_stop_reason
    if leaf.repeat_block_kind is not None:
        payload["repeat_block_kind"] = leaf.repeat_block_kind
    if leaf.repeat_block_count is not None:
        payload["repeat_block_count"] = leaf.repeat_block_count
    if leaf.repeat_last_similarity_ratio is not None:
        payload["repeat_last_similarity_ratio"] = leaf.repeat_last_similarity_ratio
    return payload
