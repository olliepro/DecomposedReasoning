"""Runtime dataclasses for branching decode state and outcomes."""

from __future__ import annotations

from dataclasses import dataclass

from branching_eval.tree_types import TokenTrace


@dataclass(frozen=True)
class PathState:
    """Mutable-free path state used while expanding one leaf branch.

    Args:
        node_id: Current tree node id.
        assistant_prefix: Assistant prefix text at this node.
        prompt_token_ids: Prompt token chain at this node.
        token_ids: Token id chain accumulated for this path.
        token_traces: Token traces accumulated for this path.
        branch_points_used: Number of chosen branch points on this path.

    Returns:
        Dataclass describing one active path.
    """

    node_id: str
    assistant_prefix: str
    prompt_token_ids: tuple[int, ...] | None
    token_ids: tuple[int, ...]
    token_traces: tuple[TokenTrace, ...]
    branch_points_used: int


@dataclass(frozen=True)
class DecodeOutcome:
    """Decode result until termination or branch trigger.

    Args:
        event_type: `trigger` or `terminated`.
        trigger_type: Trigger type when event is `trigger`.
        entropy_value: Entropy value when trigger is entropy-based.
        assistant_prefix: Updated assistant prefix.
        prompt_token_ids: Updated prompt token chain.
        token_ids: Updated token id chain.
        token_traces: Updated token traces.
        generated_tokens: Total generated tokens accumulated for this path state.
        stop_reason: Termination reason when event type is `terminated`.

    Returns:
        Decode outcome for branching expansion.
    """

    event_type: str
    trigger_type: str | None
    entropy_value: float | None
    assistant_prefix: str
    prompt_token_ids: tuple[int, ...] | None
    token_ids: tuple[int, ...]
    token_traces: tuple[TokenTrace, ...]
    generated_tokens: int
    stop_reason: str
