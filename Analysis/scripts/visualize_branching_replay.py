#!/usr/bin/env python3
"""Generate events-only HTML visualization for one branching_eval run."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from branching_eval.event_types import EventEnvelope, parse_event_row


@dataclass(frozen=True)
class AttemptKey:
    """Identity key for one replayed document attempt stream.

    Args:
        doc_id: Document id.
        doc_attempt: Attempt index.
        task_name: Task name label.
        model_id: Model id label.
        selector_mode: Selector mode label.

    Returns:
        Immutable key used for grouping and rendering.
    """

    doc_id: int
    doc_attempt: int
    task_name: str
    model_id: str
    selector_mode: str

    def slug(self) -> str:
        """Return filesystem-safe slug for this attempt key."""

        return (
            f"doc_{self.doc_id}_attempt_{self.doc_attempt}_"
            f"{self.task_name}_{self.model_id}_{self.selector_mode}"
        ).replace("/", "_")

    def label(self) -> str:
        """Return concise display label."""

        return f"doc {self.doc_id} · attempt {self.doc_attempt}"


@dataclass(frozen=True)
class NodeView:
    """One replayed tree node."""

    node_id: str
    parent_node_id: str | None
    branch_points_used: int


@dataclass(frozen=True)
class EdgeView:
    """One replayed selected edge."""

    parent_node_id: str
    child_node_id: str
    candidate_id: int | None
    selector_mode: str
    candidate_text: str
    candidate_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class DecodeChunkView:
    """One replayed decode chunk row."""

    node_id: str
    event_index: int
    timestamp_utc: str
    chunk_text: str
    chunk_token_ids: tuple[int, ...]
    finish_reason: str
    generated_tokens_after_chunk: int


@dataclass(frozen=True)
class TriggerView:
    """One replayed trigger event."""

    event_type: str
    node_id: str
    trigger_type: str
    entropy_value: float | None
    generated_tokens: int | None


@dataclass(frozen=True)
class LeafView:
    """One replayed scored leaf row."""

    leaf_id: str
    node_id: str
    text: str
    verification: int | None
    length_tokens_total: int | None
    stop_reason: str
    task_metrics: dict[str, Any]


@dataclass(frozen=True)
class VllmRequestView:
    """One replayed vLLM request envelope."""

    request_id: str
    request_stream_id: str
    request_kind: str
    event_index: int
    timestamp_utc: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class VllmResponseView:
    """One replayed vLLM response envelope."""

    request_id: str
    request_stream_id: str
    request_kind: str
    status: str
    event_index: int
    timestamp_utc: str
    payload: dict[str, Any]


@dataclass
class AttemptState:
    """Mutable replay state for one doc attempt stream."""

    key: AttemptKey
    nodes: dict[str, NodeView] = field(default_factory=dict)
    edges: list[EdgeView] = field(default_factory=list)
    decode_chunks: list[DecodeChunkView] = field(default_factory=list)
    triggers: list[TriggerView] = field(default_factory=list)
    leaves: dict[str, LeafView] = field(default_factory=dict)
    vllm_requests: dict[str, VllmRequestView] = field(default_factory=dict)
    vllm_responses: dict[str, VllmResponseView] = field(default_factory=dict)
    request_node_map: dict[str, str] = field(default_factory=dict)
    node_first_event_index: dict[str, int] = field(default_factory=dict)
    node_first_timestamp_utc: dict[str, str] = field(default_factory=dict)
    events: list[EventEnvelope] = field(default_factory=list)
    started: bool = False
    resumed_reason: str | None = None
    finished: bool = False
    finished_payload: dict[str, Any] = field(default_factory=dict)
    vllm_request_count: int = 0
    vllm_response_count: int = 0
    vllm_error_count: int = 0
    event_count: int = 0
    last_event_index: int = -1
    last_timestamp_utc: str = ""

    def status(self) -> str:
        """Return normalized attempt status."""

        if self.finished:
            return "completed"
        if self.started:
            return "incomplete"
        return "empty"

    def leaf_count(self) -> int:
        """Return number of scored leaves in this attempt."""

        return len(self.leaves)


@dataclass(frozen=True)
class RenderSummary:
    """Summary of one render pass."""

    event_count: int
    attempt_count: int
    selected_doc_count: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the visualization script."""

    parser = argparse.ArgumentParser(
        description=(
            "Visualize branching_eval run from tree_events.jsonl only. "
            "Supports snapshot and live follow modes."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--follow", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument(
        "--max-follow-iterations",
        type=int,
        default=None,
        help="Optional loop cap for automated tests.",
    )
    return parser.parse_args()


def read_events_lenient(*, path: Path) -> list[EventEnvelope]:
    """Read and parse canonical events, skipping malformed/incomplete rows.

    Args:
        path: Event log file path.

    Returns:
        Ordered list of parsed events.

    Example:
        >>> read_events_lenient(path=Path('/tmp/missing-tree-events.jsonl'))
        []
    """

    if not path.exists():
        return []
    events: list[EventEnvelope] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line_text = line.strip()
            if not line_text:
                continue
            try:
                payload = json.loads(line_text)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            try:
                events.append(parse_event_row(row=payload))
            except (AssertionError, KeyError, TypeError, ValueError):
                continue
    events.sort(key=lambda row: row.event_index)
    return events


def replay_attempts(*, events: list[EventEnvelope]) -> dict[AttemptKey, AttemptState]:
    """Replay canonical events into per-attempt state views.

    Args:
        events: Ordered canonical events.

    Returns:
        Mapping from attempt key to replay state.

    Example:
        >>> replay_attempts(events=[])
        {}
    """

    states: dict[AttemptKey, AttemptState] = {}
    for event in events:
        if event.doc_id is None or event.doc_attempt is None:
            continue
        key = AttemptKey(
            doc_id=event.doc_id,
            doc_attempt=event.doc_attempt,
            task_name=event.task_name,
            model_id=event.model_id,
            selector_mode=event.selector_mode,
        )
        state = states.get(key)
        if state is None:
            state = AttemptState(key=key)
            states[key] = state
        state.event_count += 1
        state.last_event_index = max(state.last_event_index, event.event_index)
        state.last_timestamp_utc = event.timestamp_utc
        state.events.append(event)
        apply_event_to_attempt(state=state, event=event)
    return states


def apply_event_to_attempt(*, state: AttemptState, event: EventEnvelope) -> None:
    """Apply one event envelope to mutable attempt replay state."""

    payload = event.payload
    event_type = event.event_type
    if event_type == "doc_started":
        state.started = True
        return
    if event_type == "doc_resumed":
        state.started = True
        state.resumed_reason = str(payload.get("reason", ""))
        return
    if event_type == "doc_finished":
        state.finished = True
        state.finished_payload = payload
        return
    if event_type == "node_created":
        _apply_node_created_event(state=state, event=event)
        return
    if event_type == "edge_selected":
        _apply_edge_selected_event(state=state, event=event)
        return
    if event_type == "decode_chunk":
        _apply_decode_chunk_event(state=state, event=event)
        return
    if event_type in {"trigger_fired", "trigger_skipped_max_branch_points"}:
        _apply_trigger_event(state=state, event=event)
        return
    if event_type in {"candidate_pool_resolved", "selector_applied", "leaf_completed"}:
        _apply_node_marker_event(state=state, event=event)
        return
    if event_type == "leaf_scored":
        _apply_leaf_scored_event(state=state, event=event)
        return
    if event_type == "vllm_request":
        _apply_vllm_request_event(state=state, event=event)
        return
    if event_type == "vllm_response":
        _apply_vllm_response_event(state=state, event=event)


def _apply_node_created_event(*, state: AttemptState, event: EventEnvelope) -> None:
    payload = event.payload
    node_id = str(payload.get("node_id", ""))
    if not node_id:
        return
    parent_node_id = payload.get("parent_node_id")
    state.nodes[node_id] = NodeView(
        node_id=node_id,
        parent_node_id=str(parent_node_id) if parent_node_id is not None else None,
        branch_points_used=int(payload.get("branch_points_used", 0)),
    )
    mark_node_event_touch(state=state, node_id=node_id, event=event)


def _apply_edge_selected_event(*, state: AttemptState, event: EventEnvelope) -> None:
    payload = event.payload
    parent_node_id = str(payload.get("parent_node_id", ""))
    child_node_id = str(payload.get("child_node_id", ""))
    if not parent_node_id or not child_node_id:
        return
    raw_ids = payload.get("candidate_token_ids_normalized", [])
    candidate_token_ids = (
        tuple(int(value) for value in raw_ids) if isinstance(raw_ids, list) else ()
    )
    state.edges.append(
        EdgeView(
            parent_node_id=parent_node_id,
            child_node_id=child_node_id,
            candidate_id=(
                int(payload["candidate_id"])
                if payload.get("candidate_id") is not None
                else None
            ),
            selector_mode=str(payload.get("selector_mode", "")),
            candidate_text=str(payload.get("candidate_text_normalized", "")),
            candidate_token_ids=candidate_token_ids,
        )
    )
    mark_node_event_touch(state=state, node_id=child_node_id, event=event)


def _apply_decode_chunk_event(*, state: AttemptState, event: EventEnvelope) -> None:
    payload = event.payload
    node_id = str(payload.get("node_id", ""))
    if not node_id:
        return
    raw_ids = payload.get("chunk_token_ids", [])
    chunk_token_ids = (
        tuple(int(value) for value in raw_ids) if isinstance(raw_ids, list) else ()
    )
    state.decode_chunks.append(
        DecodeChunkView(
            node_id=node_id,
            event_index=event.event_index,
            timestamp_utc=event.timestamp_utc,
            chunk_text=str(payload.get("chunk_text", "")),
            chunk_token_ids=chunk_token_ids,
            finish_reason=str(payload.get("finish_reason", "")),
            generated_tokens_after_chunk=int(
                payload.get("generated_tokens_after_chunk", 0)
            ),
        )
    )
    mark_node_event_touch(state=state, node_id=node_id, event=event)


def _apply_trigger_event(*, state: AttemptState, event: EventEnvelope) -> None:
    payload = event.payload
    node_id = str(payload.get("node_id", ""))
    if not node_id:
        return
    entropy_value = (
        float(payload["entropy_value"])
        if payload.get("entropy_value") is not None
        else None
    )
    generated_tokens = (
        int(payload["generated_tokens"])
        if payload.get("generated_tokens") is not None
        else None
    )
    state.triggers.append(
        TriggerView(
            event_type=event.event_type,
            node_id=node_id,
            trigger_type=str(payload.get("trigger_type", "")),
            entropy_value=entropy_value,
            generated_tokens=generated_tokens,
        )
    )
    mark_node_event_touch(state=state, node_id=node_id, event=event)


def _apply_node_marker_event(*, state: AttemptState, event: EventEnvelope) -> None:
    """Mark node activity for events that only carry identifiers in payload."""

    node_id = str(event.payload.get("node_id", ""))
    if not node_id:
        return
    mark_node_event_touch(state=state, node_id=node_id, event=event)


def _apply_leaf_scored_event(*, state: AttemptState, event: EventEnvelope) -> None:
    payload = event.payload
    leaf_id = str(payload.get("leaf_id", ""))
    if not leaf_id:
        return
    task_metrics = payload.get("task_metrics", {})
    task_metrics_mapping = task_metrics if isinstance(task_metrics, dict) else {}
    leaf = LeafView(
        leaf_id=leaf_id,
        node_id=str(payload.get("node_id", "")),
        text=str(payload.get("text", "")),
        verification=(
            int(payload["verification"])
            if payload.get("verification") is not None
            else None
        ),
        length_tokens_total=(
            int(payload["length_tokens_total"])
            if payload.get("length_tokens_total") is not None
            else None
        ),
        stop_reason=str(payload.get("stop_reason", "")),
        task_metrics=task_metrics_mapping,
    )
    state.leaves[leaf_id] = leaf
    if leaf.node_id:
        mark_node_event_touch(state=state, node_id=leaf.node_id, event=event)


def _apply_vllm_request_event(*, state: AttemptState, event: EventEnvelope) -> None:
    payload = event.payload
    request_id = str(payload.get("request_id", ""))
    request_stream_id = str(payload.get("request_stream_id", ""))
    if not request_id:
        return
    request = VllmRequestView(
        request_id=request_id,
        request_stream_id=request_stream_id,
        request_kind=str(payload.get("request_kind", "")),
        event_index=event.event_index,
        timestamp_utc=event.timestamp_utc,
        payload=payload,
    )
    state.vllm_requests[request_id] = request
    state.vllm_request_count += 1
    node_id = decode_node_id_from_stream_id(request_stream_id=request_stream_id)
    if node_id is None:
        return
    state.request_node_map[request_id] = node_id
    mark_node_event_touch(state=state, node_id=node_id, event=event)


def _apply_vllm_response_event(*, state: AttemptState, event: EventEnvelope) -> None:
    payload = event.payload
    request_id = str(payload.get("request_id", ""))
    request_stream_id = str(payload.get("request_stream_id", ""))
    if not request_id:
        return
    response = VllmResponseView(
        request_id=request_id,
        request_stream_id=request_stream_id,
        request_kind=str(payload.get("request_kind", "")),
        status=str(payload.get("status", "")),
        event_index=event.event_index,
        timestamp_utc=event.timestamp_utc,
        payload=payload,
    )
    state.vllm_responses[request_id] = response
    state.vllm_response_count += 1
    if response.status == "error":
        state.vllm_error_count += 1
    node_id = state.request_node_map.get(request_id)
    if node_id is None:
        node_id = decode_node_id_from_stream_id(request_stream_id=request_stream_id)
    if node_id is not None:
        mark_node_event_touch(state=state, node_id=node_id, event=event)


def decode_node_id_from_stream_id(*, request_stream_id: str) -> str | None:
    """Extract decode node id from request stream id.

    Args:
        request_stream_id: Logged request stream id.

    Returns:
        Node id for decode stream ids, else `None`.
    """

    if not request_stream_id.startswith("decode:"):
        return None
    node_id = request_stream_id.split(":", maxsplit=1)[1]
    return node_id if node_id else None


def mark_node_event_touch(
    *, state: AttemptState, node_id: str, event: EventEnvelope
) -> None:
    """Record earliest event reference for one node."""

    if not node_id:
        return
    prior_index = state.node_first_event_index.get(node_id)
    if prior_index is None or event.event_index < prior_index:
        state.node_first_event_index[node_id] = event.event_index
        state.node_first_timestamp_utc[node_id] = event.timestamp_utc


def selected_attempts_by_doc(
    *, states: dict[AttemptKey, AttemptState]
) -> list[AttemptState]:
    """Select one default attempt per doc (latest complete else latest partial)."""

    by_doc: dict[int, list[AttemptState]] = {}
    for state in states.values():
        by_doc.setdefault(state.key.doc_id, []).append(state)
    selected: list[AttemptState] = []
    for doc_id in sorted(by_doc):
        rows = sorted(
            by_doc[doc_id],
            key=lambda row: (row.key.doc_attempt, row.last_event_index),
        )
        completed = [row for row in rows if row.finished]
        if completed:
            selected.append(completed[-1])
            continue
        selected.append(rows[-1])
    return selected
