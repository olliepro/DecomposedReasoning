"""Tree-structured artifact dataclasses for branching experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from branching_eval.selector_types import SelectionOutcome


@dataclass(frozen=True)
class TokenTrace:
    """One generated token with probability and entropy metadata.

    Args:
        token_index: Token offset within the generated fragment.
        token_id: Generated token id when available.
        token_text: Decoded token text.
        logprob: Selected token logprob.
        probability: Selected token probability.
        entropy: Approximate next-token entropy.

    Returns:
        Dataclass containing per-token trace details.
    """

    token_index: int
    token_id: int | None
    token_text: str
    logprob: float
    probability: float
    entropy: float


@dataclass(frozen=True)
class CandidateRecord:
    """One candidate continuation in a branch candidate pool.

    Args:
        candidate_id: Stable candidate id within the pool.
        text: Candidate continuation text.
        token_ids: Candidate output token ids.
        tokens: Candidate per-token traces.
        finish_reason: Generation finish reason.
        stop_reason: Optional stop reason.

    Returns:
        Dataclass describing one branch candidate.
    """

    candidate_id: int
    text: str
    token_ids: tuple[int, ...]
    tokens: tuple[TokenTrace, ...]
    finish_reason: str
    stop_reason: str | int | None


@dataclass(frozen=True)
class TreeNode:
    """One node in the rollout tree.

    Args:
        node_id: Stable node id.
        parent_node_id: Parent node id, or None for root.
        prompt_text: Prompt text for this document.
        assistant_prefix: Assistant prefix at this node.
        prompt_token_ids: Prompt token chain used for continuation.
        branch_points_used: Number of chosen branch points on this path so far.

    Returns:
        Dataclass containing node state.
    """

    node_id: str
    parent_node_id: str | None
    prompt_text: str
    assistant_prefix: str
    prompt_token_ids: tuple[int, ...] | None
    branch_points_used: int


@dataclass(frozen=True)
class TreeEdge:
    """One selected edge from parent node to child node.

    Args:
        edge_id: Stable edge id.
        parent_node_id: Parent node id.
        child_node_id: Child node id.
        candidate_pool_id: Candidate pool id used at this branch point.
        candidate_id: Candidate id selected for this edge.
        selector_mode: Selector mode that chose this candidate.

    Returns:
        Dataclass representing one selected branch edge.
    """

    edge_id: str
    parent_node_id: str
    child_node_id: str
    candidate_pool_id: str
    candidate_id: int
    selector_mode: str


@dataclass(frozen=True)
class BranchPointRecord:
    """One branch-point event with candidate pool and selector outcomes.

    Args:
        branch_point_id: Stable branch point id.
        node_id: Node id where branching happened.
        trigger_type: Trigger type (`steer_boundary` or `high_entropy`).
        entropy_value: Entropy at trigger time when available.
        candidate_pool_key: Cache key used for candidate-pool reuse.
        candidate_pool_id: Candidate pool id.
        selections: Selection outcomes for all selector modes.

    Returns:
        Dataclass for branch-point artifact rows.
    """

    branch_point_id: str
    node_id: str
    trigger_type: str
    entropy_value: float | None
    candidate_pool_key: str
    candidate_pool_id: str
    selections: tuple[SelectionOutcome, ...]


@dataclass(frozen=True)
class CandidatePoolRecord:
    """Candidate pool generated once and shared across selectors.

    Args:
        candidate_pool_id: Stable pool id.
        cache_key: Stable cache key.
        branch_point_id: Branch point id that requested this pool.
        node_id: Node id where pool was generated.
        trigger_type: Trigger mode used for generation.
        entropy_value: Entropy value for entropy-trigger pools.
        candidates: Candidate rows.

    Returns:
        Dataclass describing one persisted candidate pool.
    """

    candidate_pool_id: str
    cache_key: str
    branch_point_id: str
    node_id: str
    trigger_type: str
    entropy_value: float | None
    candidates: tuple[CandidateRecord, ...]


@dataclass(frozen=True)
class LeafRollout:
    """One completed leaf rollout in the branch tree.

    Args:
        leaf_id: Stable leaf id.
        node_id: Final node id for this leaf.
        text: Final output text.
        token_ids: Leaf rollout token ids.
        tokens: Leaf rollout token traces.
        verification: Verification score in `{0, 1}`.
        length_tokens_total: Completion token count.
        length_tokens_exec: Optional tokens after `<exec>`.
        stop_reason: Termination reason.
        task_metrics: Raw task metric dictionary from lm_eval task scoring.

    Returns:
        Dataclass containing final rollout and metrics.
    """

    leaf_id: str
    node_id: str
    text: str
    token_ids: tuple[int, ...]
    tokens: tuple[TokenTrace, ...]
    verification: int
    length_tokens_total: int
    length_tokens_exec: int | None
    stop_reason: str
    task_metrics: dict[str, Any]


@dataclass
class BranchTree:
    """In-memory branch tree and associated artifacts for one document.

    Args:
        doc_id: Document id.
        doc_attempt: Attempt index for this document.
        run_id: Stable run identifier.
        task_name: Task name.
        model_id: Model id.
        selector_mode: Selector mode used to expand this tree.
        root_prompt: Prompt text.
        nodes: Node map.
        edges: Selected edges.
        branch_points: Branch points encountered while expanding leaves.
        candidate_pools: Candidate pools referenced by this tree.
        leaves: Completed leaf rollouts.

    Returns:
        Mutable tree container with convenience accessors.
    """

    doc_id: int
    task_name: str
    model_id: str
    selector_mode: str
    root_prompt: str
    doc_attempt: int = 0
    run_id: str = ""
    nodes: dict[str, TreeNode] = field(default_factory=dict)
    edges: list[TreeEdge] = field(default_factory=list)
    branch_points: list[BranchPointRecord] = field(default_factory=list)
    candidate_pools: list[CandidatePoolRecord] = field(default_factory=list)
    leaves: list[LeafRollout] = field(default_factory=list)

    def leaf_verifications(self) -> list[int]:
        """Return leaf verification values in list order.

        Args:
            None.

        Returns:
            List of verification labels.
        """

        return [leaf.verification for leaf in self.leaves]

    def leaf_lengths(self) -> list[int]:
        """Return leaf total completion lengths in list order.

        Args:
            None.

        Returns:
            List of total completion token lengths.
        """

        return [leaf.length_tokens_total for leaf in self.leaves]

    def add_node(self, *, node: TreeNode) -> None:
        """Insert one node into tree map.

        Args:
            node: Node dataclass.

        Returns:
            None.
        """

        self.nodes[node.node_id] = node


@dataclass(frozen=True)
class TreeEvent:
    """One append-only event row for live tree reconstruction.

    Args:
        timestamp_utc: UTC ISO-8601 timestamp for event emission.
        doc_id: Document id.
        task_name: Task name.
        model_id: Model label.
        selector_mode: Active selector mode.
        event_type: Event label (`node_created`, `leaf_completed`, etc.).
        payload: Event-specific payload mapping.

    Returns:
        Dataclass containing one JSONL event row.
    """

    timestamp_utc: str
    doc_id: int
    task_name: str
    model_id: str
    selector_mode: str
    event_type: str
    payload: dict[str, Any]

    def to_json_row(self) -> dict[str, Any]:
        """Convert event to canonical JSON row mapping.

        Args:
            None.

        Returns:
            JSON-serializable event mapping.

        Example:
            >>> event = TreeEvent(
            ...     timestamp_utc="2026-01-01T00:00:00+00:00",
            ...     doc_id=0,
            ...     task_name="aime24",
            ...     model_id="non_sft",
            ...     selector_mode="random",
            ...     event_type="node_created",
            ...     payload={"node_id": "node_root"},
            ... )
            >>> event.to_json_row()["event_type"]
            'node_created'
        """

        return {
            "timestamp_utc": self.timestamp_utc,
            "doc_id": self.doc_id,
            "task_name": self.task_name,
            "model_id": self.model_id,
            "selector_mode": self.selector_mode,
            "event_type": self.event_type,
            "payload": self.payload,
        }
