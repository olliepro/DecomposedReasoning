"""Compact graph and lazy node-detail payload builders for doc pages."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from branching_eval.event_types import EventEnvelope

try:
    from scripts.visualize_branching_replay import (
        AttemptState,
        EdgeView,
        NodeView,
        decode_node_id_from_stream_id,
    )
except ModuleNotFoundError:
    from visualize_branching_replay import (
        AttemptState,
        EdgeView,
        NodeView,
        decode_node_id_from_stream_id,
    )

VISIBLE_RUNTIME_EVENT_TYPES = {
    "trigger_fired",
    "trigger_skipped_max_branch_points",
    "candidate_pool_resolved",
    "selector_applied",
    "selector_continued_inline",
    "leaf_completed",
    "leaf_scored",
}


@dataclass(frozen=True)
class AttemptLayoutData:
    """Computed graph/event layout data reused across doc payload builders.

    Args:
        nodes: Node rows keyed by `node_id`.
        edges: Selected edge rows.
        graph_node_events: Node-scoped summary rows used for graph rendering.
        node_events: Node-scoped full event rows used for lazy inspector payloads.
        node_summaries: Compact node summary rows for graph rendering.
        leaf_rows_by_node: Full leaf-detail rows keyed by terminal node id.
        meta: Run-attempt metadata used by the doc page.

    Returns:
        Immutable layout payload for one replayed doc attempt.
    """

    nodes: dict[str, NodeView]
    edges: list[EdgeView]
    graph_node_events: dict[str, list[dict[str, Any]]]
    node_events: dict[str, list[dict[str, Any]]]
    node_summaries: list[dict[str, Any]]
    leaf_rows_by_node: dict[str, list[dict[str, Any]]]
    meta: dict[str, Any]


def tree_payload_for_attempt(
    *,
    state: AttemptState,
    detail_base_url: str = "",
) -> dict[str, Any]:
    """Build the compact graph payload consumed by one doc page.

    Args:
        state: Replayed attempt state.
        detail_base_url: Relative base URL for node detail JSON files.

    Returns:
        JSON-ready payload for the interactive tree workspace.
    """

    layout = build_attempt_layout(state=state, detail_base_url=detail_base_url)
    maxima = payload_maxima(
        node_rows=layout.node_summaries,
        node_events=layout.graph_node_events,
    )
    return {
        "nodes": layout.node_summaries,
        "edges": edge_payload_rows(edges=layout.edges),
        "branches": [],
        "node_events": layout.graph_node_events,
        "meta": {
            **layout.meta,
            "x_max": maxima,
            "branch_count": 0,
        },
    }


def node_detail_payloads_for_attempt(
    *,
    state: AttemptState,
    detail_base_url: str = "",
) -> dict[str, dict[str, Any]]:
    """Build lazy-loaded node detail payloads for one doc attempt.

    Args:
        state: Replayed attempt state.
        detail_base_url: Relative base URL for node detail JSON files.

    Returns:
        Mapping from `node_id` to node-detail payload.
    """

    layout = build_attempt_layout(state=state, detail_base_url=detail_base_url)
    payloads: dict[str, dict[str, Any]] = {}
    summary_by_node = {str(row["node_id"]): row for row in layout.node_summaries}
    for node_id, summary in summary_by_node.items():
        payloads[node_id] = {
            "node": summary,
            "events": layout.node_events.get(node_id, []),
            "leaves": layout.leaf_rows_by_node.get(node_id, []),
        }
    return payloads


def build_attempt_layout(
    *,
    state: AttemptState,
    detail_base_url: str,
) -> AttemptLayoutData:
    """Compute reusable compact graph and node-detail rows.

    Args:
        state: Replayed attempt state.
        detail_base_url: Relative base URL for node detail JSON files.

    Returns:
        Shared layout dataclass reused by graph/detail builders.
    """

    nodes, edges = normalized_graph(state=state)
    parent_by_child = {edge.child_node_id: edge.parent_node_id for edge in edges}
    incoming_tokens = {
        edge.child_node_id: len(edge.candidate_token_ids) for edge in edges
    }
    event_seconds = normalized_event_seconds_by_index(state=state)
    node_events = node_event_rows(state=state)
    base_steps, base_tokens, final_steps, final_tokens = node_metric_caches(
        node_ids=tuple(nodes.keys()),
        parent_by_child=parent_by_child,
        incoming_tokens=incoming_tokens,
        node_events=node_events,
    )
    annotate_node_event_metrics(
        node_events=node_events,
        base_steps=base_steps,
        base_tokens=base_tokens,
        event_seconds_by_index=event_seconds,
    )
    graph_node_events = graph_event_rows(node_events=node_events)
    leaf_rows_by_node = leaf_rows_by_node_id(state=state)
    depth_by_node = compute_depths(nodes=nodes)
    node_summaries = node_payload_rows(
        state=state,
        nodes=nodes,
        edges=edges,
        node_events=graph_node_events,
        leaf_rows_by_node=leaf_rows_by_node,
        depth_by_node=depth_by_node,
        final_steps=final_steps,
        final_tokens=final_tokens,
        event_seconds_by_index=event_seconds,
        detail_base_url=detail_base_url,
    )
    meta = {
        "status": state.status(),
        "started_at": state.events[0].timestamp_utc if state.events else "",
        "node_count": len(node_summaries),
        "edge_count": len(edges),
        "leaf_count": state.leaf_count(),
        "event_count": state.event_count,
        "vllm_request_count": state.vllm_request_count,
        "vllm_response_count": state.vllm_response_count,
        "vllm_error_count": state.vllm_error_count,
        "last_event_index": state.last_event_index,
        "resumed_reason": state.resumed_reason or "",
    }
    return AttemptLayoutData(
        nodes=nodes,
        edges=edges,
        graph_node_events=graph_node_events,
        node_events=node_events,
        node_summaries=node_summaries,
        leaf_rows_by_node=leaf_rows_by_node,
        meta=meta,
    )


def graph_event_rows(
    *,
    node_events: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    """Return summary-only event rows used by the initial graph payload."""

    graph_rows: dict[str, list[dict[str, Any]]] = {}
    keep_keys = (
        "event_id",
        "event_index",
        "timestamp_utc",
        "event_type",
        "node_id",
        "summary",
        "step_delta",
        "token_delta",
        "metrics",
    )
    for node_id, rows in node_events.items():
        graph_rows[node_id] = []
        for row in rows:
            graph_row = {key: row[key] for key in keep_keys if key in row}
            details = row.get("details")
            if row.get("event_type") == "leaf_scored" and isinstance(details, dict):
                graph_row["details"] = {
                    "leaf_id": details.get("leaf_id"),
                    "verification": details.get("verification"),
                    "stop_reason": details.get("stop_reason"),
                }
            graph_rows[node_id].append(graph_row)
    return graph_rows


def normalized_graph(
    *,
    state: AttemptState,
) -> tuple[dict[str, NodeView], list[EdgeView]]:
    """Return node/edge maps with implied nodes synthesized from edge rows."""

    nodes = dict(state.nodes)
    edges = [edge for edge in state.edges if edge.parent_node_id and edge.child_node_id]
    if "node_root" not in nodes:
        nodes["node_root"] = NodeView(
            node_id="node_root", parent_node_id=None, branch_points_used=0
        )
    for edge in edges:
        nodes.setdefault(
            edge.parent_node_id,
            NodeView(
                node_id=edge.parent_node_id, parent_node_id=None, branch_points_used=0
            ),
        )
        nodes.setdefault(
            edge.child_node_id,
            NodeView(
                node_id=edge.child_node_id,
                parent_node_id=edge.parent_node_id,
                branch_points_used=0,
            ),
        )
    return nodes, edges


def node_event_rows(*, state: AttemptState) -> dict[str, list[dict[str, Any]]]:
    """Build compact node-scoped event rows for the doc inspector."""

    rows: dict[str, list[dict[str, Any]]] = {}
    pending_requests: dict[str, EventEnvelope] = {}
    selected = selected_candidates_by_branch_point(events=state.events)
    shortlisted = shortlisted_candidates_by_branch_point(events=state.events)
    for event in state.events:
        request_id = str(event.payload.get("request_id", ""))
        if event.event_type == "vllm_request" and request_id:
            pending_requests[request_id] = event
            continue
        if event.event_type == "vllm_response":
            row = vllm_step_row(
                request_event=pending_requests.pop(request_id, None),
                response_event=event,
                state=state,
            )
        elif event.event_type in VISIBLE_RUNTIME_EVENT_TYPES:
            row = runtime_event_row(
                event=event,
                state=state,
                selected_by_branch_point=selected,
                shortlist_by_branch_point=shortlisted,
            )
        else:
            row = None
        if row is not None:
            rows.setdefault(str(row["node_id"]), []).append(row)
    for request_event in pending_requests.values():
        row = vllm_step_row(
            request_event=request_event, response_event=None, state=state
        )
        if row is not None:
            rows.setdefault(str(row["node_id"]), []).append(row)
    for node_rows in rows.values():
        node_rows.sort(key=lambda row: int(row["event_index"]))
    return rows


def selected_candidates_by_branch_point(
    *,
    events: list[EventEnvelope],
) -> dict[str, set[int]]:
    """Return selected candidate ids keyed by branch point id."""

    selected: dict[str, set[int]] = {}
    for event in events:
        if event.event_type != "selector_applied":
            continue
        branch_point_id = str(event.payload.get("branch_point_id", ""))
        raw_ids = event.payload.get("selected_candidate_ids", [])
        if branch_point_id and isinstance(raw_ids, list):
            selected[branch_point_id] = {int(value) for value in raw_ids}
    return selected


def shortlisted_candidates_by_branch_point(
    *,
    events: list[EventEnvelope],
) -> dict[str, set[int]]:
    """Return shortlist candidate ids keyed by branch point id."""

    shortlisted: dict[str, set[int]] = {}
    for event in events:
        if event.event_type != "selector_applied":
            continue
        branch_point_id = str(event.payload.get("branch_point_id", ""))
        shortlist_by_mode = event.payload.get("shortlist_by_mode", {})
        selector_mode = str(event.payload.get("active_selector_mode", ""))
        raw_ids = (
            shortlist_by_mode.get(selector_mode, [])
            if isinstance(shortlist_by_mode, dict)
            else []
        )
        if branch_point_id and isinstance(raw_ids, list):
            shortlisted[branch_point_id] = {int(value) for value in raw_ids}
    return shortlisted


def runtime_event_row(
    *,
    event: EventEnvelope,
    state: AttemptState,
    selected_by_branch_point: dict[str, set[int]],
    shortlist_by_branch_point: dict[str, set[int]],
) -> dict[str, Any] | None:
    """Build one compact runtime event row for the node inspector."""

    node_id = event_node_id(event=event, state=state)
    if node_id is None:
        return None
    return {
        "event_id": f"{event.event_index}:{node_id}:{event.event_type}",
        "event_index": event.event_index,
        "timestamp_utc": event.timestamp_utc,
        "event_type": event.event_type,
        "node_id": node_id,
        "summary": runtime_event_summary(event=event),
        "details": runtime_event_details(
            event=event,
            state=state,
            selected_by_branch_point=selected_by_branch_point,
            shortlist_by_branch_point=shortlist_by_branch_point,
        ),
        "step_delta": 1,
        "token_delta": 0,
    }


def runtime_event_summary(*, event: EventEnvelope) -> str:
    """Return concise summary text for one visible runtime event."""

    payload = event.payload
    if event.event_type == "trigger_fired":
        return f"trigger {str(payload.get('trigger_type', ''))}"
    if event.event_type == "trigger_skipped_max_branch_points":
        return "trigger skipped max branch points"
    if event.event_type == "candidate_pool_resolved":
        return f"candidate pool n={int(payload.get('num_candidates', 0))}"
    if event.event_type == "selector_applied":
        count = len(payload.get("selected_candidate_ids", []))
        return f"selector kept {count}"
    if event.event_type == "selector_continued_inline":
        return f"selector continued inline {payload.get('selected_candidate_id')}"
    if event.event_type == "leaf_completed":
        return f"leaf completed {payload.get('leaf_id')}"
    if event.event_type == "leaf_scored":
        return f"leaf scored verify={payload.get('verification')}"
    return event.event_type


def runtime_event_details(
    *,
    event: EventEnvelope,
    state: AttemptState,
    selected_by_branch_point: dict[str, set[int]],
    shortlist_by_branch_point: dict[str, set[int]],
) -> dict[str, Any]:
    """Return compact details for one runtime event inspector card."""

    payload = event.payload
    details = {
        "event_type": event.event_type,
        "event_index": event.event_index,
        "timestamp_utc": event.timestamp_utc,
    }
    if event.event_type == "candidate_pool_resolved":
        branch_point_id = str(payload.get("branch_point_id", ""))
        details.update(
            {
                "branch_point_id": branch_point_id,
                "candidate_pool_id": payload.get("candidate_pool_id"),
                "trigger_type": payload.get("trigger_type"),
                "num_candidates": payload.get("num_candidates"),
                "candidates": annotate_candidates(
                    candidates=payload.get("candidates"),
                    shortlist_ids=shortlist_by_branch_point.get(branch_point_id, set()),
                    selected_ids=selected_by_branch_point.get(branch_point_id, set()),
                ),
            }
        )
        return details
    if event.event_type == "leaf_scored":
        leaf_id = str(payload.get("leaf_id", ""))
        leaf = state.leaves.get(leaf_id)
        details.update(
            {
                "leaf_id": leaf_id,
                "verification": payload.get("verification"),
                "length_tokens_total": payload.get("length_tokens_total"),
                "stop_reason": payload.get("stop_reason"),
                "task_metrics": payload.get("task_metrics", {}),
                "text": leaf.text if leaf is not None else payload.get("text_preview", ""),
                "text_preview": payload.get("text_preview", ""),
            }
        )
        return details
    details.update(payload)
    return details


def annotate_candidates(
    *,
    candidates: Any,
    shortlist_ids: set[int],
    selected_ids: set[int],
) -> list[dict[str, Any]]:
    """Annotate compact candidate summaries with shortlist/selected flags."""

    if not isinstance(candidates, list):
        return []
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        candidate_id = int(candidate.get("candidate_id", -1))
        rows.append(
            {
                "candidate_id": candidate_id,
                "text": candidate.get("text_preview", ""),
                "text_preview": candidate.get("text_preview", ""),
                "output_token_count": candidate.get("output_token_count"),
                "finish_reason": candidate.get("finish_reason"),
                "stop_reason": candidate.get("stop_reason"),
                "tokens": candidate.get("tokens", []),
                "shortlisted": candidate_id in shortlist_ids,
                "selected": candidate_id in selected_ids,
            }
        )
    return rows


def vllm_step_row(
    *,
    request_event: EventEnvelope | None,
    response_event: EventEnvelope | None,
    state: AttemptState,
) -> dict[str, Any] | None:
    """Build one merged compact vLLM step row from request/response events."""

    anchor = response_event if response_event is not None else request_event
    if anchor is None:
        return None
    node_id = event_node_id(event=anchor, state=state)
    if node_id is None:
        return None
    request_payload = request_event.payload if request_event is not None else {}
    response_payload = response_event.payload if response_event is not None else {}
    event_index = (
        response_event.event_index if response_event is not None else anchor.event_index
    )
    timestamp_utc = (
        response_event.timestamp_utc
        if response_event is not None
        else anchor.timestamp_utc
    )
    return {
        "event_id": f"{event_index}:{node_id}:vllm_step",
        "event_index": event_index,
        "timestamp_utc": timestamp_utc,
        "event_type": "vllm_step",
        "node_id": node_id,
        "summary": vllm_step_summary(
            request_payload=request_payload,
            response_payload=response_payload,
        ),
        "details": vllm_step_details(
            request_payload=request_payload,
            response_payload=response_payload,
        ),
        "step_delta": 1,
        "token_delta": vllm_step_token_delta(
            request_payload=request_payload,
            response_payload=response_payload,
        ),
    }


def vllm_step_summary(
    *,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> str:
    """Return compact summary text for one merged vLLM step."""

    request_kind = str(
        request_payload.get("request_kind", response_payload.get("request_kind", ""))
    )
    delta = int(request_payload.get("delta_token_count", 0))
    status = str(response_payload.get("status", "pending"))
    if status != "ok":
        return f"{request_kind} Δ{delta} ({status})"
    output_tokens = first_choice_output_token_count(response_payload=response_payload)
    return f"{request_kind} Δ{delta} -> +{output_tokens} tok"


def first_choice_output_token_count(*, response_payload: dict[str, Any]) -> int:
    """Return output token count for the first response choice when present."""

    choices = response_payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return 0
    first = choices[0]
    if not isinstance(first, dict):
        return 0
    return int(first.get("output_token_count", 0))


def vllm_step_token_delta(
    *,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> int:
    """Return token delta contributed by one merged vLLM step."""

    stream_id = str(
        request_payload.get(
            "request_stream_id", response_payload.get("request_stream_id", "")
        )
    )
    if not stream_id.startswith("decode:"):
        return 0
    if str(response_payload.get("status", "pending")) != "ok":
        return 0
    return first_choice_output_token_count(response_payload=response_payload)


def vllm_step_details(
    *,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> dict[str, Any]:
    """Return compact request/response details for one vLLM step."""

    return {
        "request_id": request_payload.get(
            "request_id", response_payload.get("request_id")
        ),
        "request_stream_id": request_payload.get(
            "request_stream_id", response_payload.get("request_stream_id")
        ),
        "request_kind": request_payload.get(
            "request_kind", response_payload.get("request_kind")
        ),
        "current_input_token_count": request_payload.get("current_input_token_count"),
        "base_prefix_token_count": request_payload.get("base_prefix_token_count"),
        "delta_token_count": request_payload.get("delta_token_count"),
        "assistant_prefix_char_count": request_payload.get(
            "assistant_prefix_char_count"
        ),
        "status": response_payload.get("status", "pending"),
        "latency_seconds": response_payload.get("latency_seconds"),
        "error_message": response_payload.get("error_message"),
        "choice_count": response_payload.get("choice_count", 0),
        "choices": summarized_choices(choices=response_payload.get("choices")),
    }


def summarized_choices(*, choices: Any) -> list[dict[str, Any]]:
    """Return vLLM choice rows for token-strip rendering."""

    if not isinstance(choices, list):
        return []
    rows: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        rows.append(
            {
                "index": choice.get("index"),
                "text": choice.get("text", choice.get("text_preview", "")),
                "text_preview": choice.get("text_preview", ""),
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "output_token_count": choice.get("output_token_count"),
                "tokens": choice.get("tokens", []),
            }
        )
    return rows


def event_node_id(*, event: EventEnvelope, state: AttemptState) -> str | None:
    """Resolve canonical node id for one event row when available."""

    payload = event.payload
    if event.event_type in VISIBLE_RUNTIME_EVENT_TYPES:
        node_id = str(payload.get("node_id", ""))
        return node_id if node_id else None
    if event.event_type == "vllm_request":
        return decode_node_id_from_stream_id(
            request_stream_id=str(payload.get("request_stream_id", ""))
        )
    if event.event_type == "vllm_response":
        request_id = str(payload.get("request_id", ""))
        mapped = state.request_node_map.get(request_id)
        if mapped is not None:
            return mapped
        return decode_node_id_from_stream_id(
            request_stream_id=str(payload.get("request_stream_id", ""))
        )
    return None


def node_metric_caches(
    *,
    node_ids: tuple[str, ...],
    parent_by_child: dict[str, str],
    incoming_tokens: dict[str, int],
    node_events: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Compute base/final step and token totals for each node path."""

    event_steps = {node_id: len(rows) for node_id, rows in node_events.items()}
    event_tokens = {
        node_id: sum(int(row.get("token_delta", 0)) for row in rows)
        for node_id, rows in node_events.items()
    }
    base_steps: dict[str, int] = {}
    base_tokens: dict[str, int] = {}
    final_steps: dict[str, int] = {}
    final_tokens: dict[str, int] = {}

    def compute(node_id: str) -> tuple[int, int]:
        cached = final_steps.get(node_id), final_tokens.get(node_id)
        if cached[0] is not None and cached[1] is not None:
            return int(cached[0]), int(cached[1])
        parent_id = parent_by_child.get(node_id)
        parent_steps, parent_tokens = (
            (0, 0) if parent_id is None else compute(parent_id)
        )
        start_steps = parent_steps
        start_tokens = parent_tokens + incoming_tokens.get(node_id, 0)
        final_steps[node_id] = start_steps + event_steps.get(node_id, 0)
        final_tokens[node_id] = start_tokens + event_tokens.get(node_id, 0)
        base_steps[node_id] = start_steps
        base_tokens[node_id] = start_tokens
        return final_steps[node_id], final_tokens[node_id]

    for node_id in node_ids:
        compute(node_id)
    return base_steps, base_tokens, final_steps, final_tokens


def annotate_node_event_metrics(
    *,
    node_events: dict[str, list[dict[str, Any]]],
    base_steps: dict[str, int],
    base_tokens: dict[str, int],
    event_seconds_by_index: dict[int, float],
) -> None:
    """Attach running steps/tokens/time metrics to each node-local event row."""

    for node_id, rows in node_events.items():
        running_steps = base_steps.get(node_id, 0)
        running_tokens = base_tokens.get(node_id, 0)
        for row in rows:
            running_steps += int(row.get("step_delta", 0))
            running_tokens += int(row.get("token_delta", 0))
            event_index = int(row.get("event_index", -1))
            row["metrics"] = {
                "steps": running_steps,
                "tokens": running_tokens,
                "time_seconds": max(0.0, event_seconds_by_index.get(event_index, 0.0)),
            }


def leaf_rows_by_node_id(*, state: AttemptState) -> dict[str, list[dict[str, Any]]]:
    """Return full leaf-detail rows grouped by terminal node id."""

    rows: dict[str, list[dict[str, Any]]] = {}
    for leaf in sorted(state.leaves.values(), key=leaf_sort_key):
        rows.setdefault(leaf.node_id, []).append(
            {
                "leaf_id": leaf.leaf_id,
                "verification": leaf.verification,
                "length_tokens_total": leaf.length_tokens_total,
                "stop_reason": leaf.stop_reason,
                "task_metrics": leaf.task_metrics,
                "text": leaf.text,
                "text_preview": clean_candidate_preview(text=leaf.text, max_chars=160),
            }
        )
    return rows


def leaf_sort_key(leaf: Any) -> tuple[int, int, str]:
    """Return stable sort key for node-local leaf detail ordering."""

    verification_rank = 1 if getattr(leaf, "verification", None) == 1 else 0
    think_end_rank = (
        1 if str(getattr(leaf, "stop_reason", "")).strip().lower() == "think_end" else 0
    )
    return (-verification_rank, -think_end_rank, str(getattr(leaf, "leaf_id", "")))


def compute_depths(*, nodes: dict[str, NodeView]) -> dict[str, int]:
    """Compute root-relative node depths."""

    depth_by_node: dict[str, int] = {}

    def depth_for(node_id: str) -> int:
        if node_id in depth_by_node:
            return depth_by_node[node_id]
        node = nodes.get(node_id)
        if node is None:
            depth_by_node[node_id] = 0
            return 0
        parent_node_id = node.parent_node_id
        if not parent_node_id:
            depth_by_node[node_id] = 0
            return 0
        depth_by_node[node_id] = depth_for(parent_node_id) + 1
        return depth_by_node[node_id]

    for node_id in nodes:
        depth_for(node_id)
    return depth_by_node


def node_payload_rows(
    *,
    state: AttemptState,
    nodes: dict[str, NodeView],
    edges: list[EdgeView],
    node_events: dict[str, list[dict[str, Any]]],
    leaf_rows_by_node: dict[str, list[dict[str, Any]]],
    depth_by_node: dict[str, int],
    final_steps: dict[str, int],
    final_tokens: dict[str, int],
    event_seconds_by_index: dict[int, float],
    detail_base_url: str,
) -> list[dict[str, Any]]:
    """Build compact node summary rows for graph rendering."""

    preview_by_node = candidate_preview_by_node(edges=edges)
    rows: list[dict[str, Any]] = []
    for node_id, node in sorted(nodes.items()):
        first_event_index = state.node_first_event_index.get(node_id)
        detail_path = (
            f"{detail_base_url}/nodes/{node_id}.json" if detail_base_url else ""
        )
        rows.append(
            {
                "node_id": node.node_id,
                "parent_node_id": node.parent_node_id,
                "depth": depth_by_node.get(node_id, 0),
                "branch_points_used": node.branch_points_used,
                "candidate_preview": preview_by_node.get(node_id, "Root"),
                "event_count": len(node_events.get(node_id, [])),
                "leaf_count": len(leaf_rows_by_node.get(node_id, [])),
                "detail_path": detail_path,
                "metrics": {
                    "steps": final_steps.get(node_id, 0),
                    "tokens": final_tokens.get(node_id, 0),
                    "time_seconds": (
                        event_seconds_by_index.get(first_event_index, 0.0)
                        if first_event_index is not None
                        else 0.0
                    ),
                },
            }
        )
    return rows


def candidate_preview_by_node(*, edges: list[EdgeView]) -> dict[str, str]:
    """Return one-line selected-candidate previews keyed by child node id."""

    preview_by_node = {"node_root": "Root"}
    for edge in edges:
        preview_by_node[edge.child_node_id] = clean_candidate_preview(
            text=edge.candidate_text,
            max_chars=72,
        )
    return preview_by_node


def edge_payload_rows(*, edges: list[EdgeView]) -> list[dict[str, Any]]:
    """Return compact edge payload rows for the doc graph."""

    return [
        {
            "parent_node_id": edge.parent_node_id,
            "child_node_id": edge.child_node_id,
            "candidate_id": edge.candidate_id,
            "selector_mode": edge.selector_mode,
        }
        for edge in edges
    ]


def payload_maxima(
    *,
    node_rows: list[dict[str, Any]],
    node_events: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Return max metric values used to build the workspace axis scales."""

    maxima = {"steps": 0.0, "tokens": 0.0, "time_seconds": 0.0}
    for row in node_rows:
        metrics = row.get("metrics", {})
        maxima["steps"] = max(maxima["steps"], float(metrics.get("steps", 0.0)))
        maxima["tokens"] = max(maxima["tokens"], float(metrics.get("tokens", 0.0)))
        maxima["time_seconds"] = max(
            maxima["time_seconds"],
            float(metrics.get("time_seconds", 0.0)),
        )
    for events in node_events.values():
        for event in events:
            metrics = event.get("metrics", {})
            maxima["steps"] = max(maxima["steps"], float(metrics.get("steps", 0.0)))
            maxima["tokens"] = max(maxima["tokens"], float(metrics.get("tokens", 0.0)))
            maxima["time_seconds"] = max(
                maxima["time_seconds"],
                float(metrics.get("time_seconds", 0.0)),
            )
    return maxima


def clean_candidate_preview(*, text: str, max_chars: int) -> str:
    """Collapse multi-line text into a compact one-line preview."""

    collapsed = " ".join(str(text).replace("\t", " ").split())
    if not collapsed:
        return "(empty)"
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."


def normalized_event_seconds_by_index(*, state: AttemptState) -> dict[int, float]:
    """Return elapsed seconds keyed by event index with resume downtime removed."""

    if not state.events:
        return {}
    first_timestamp = _parse_timestamp(timestamp_utc=state.events[0].timestamp_utc)
    if first_timestamp is None:
        return {}
    downtime_seconds = 0.0
    prior_timestamp = first_timestamp
    event_seconds: dict[int, float] = {}
    for event in state.events:
        current_timestamp = _parse_timestamp(timestamp_utc=event.timestamp_utc)
        if current_timestamp is None:
            continue
        if event.event_type == "doc_resumed":
            downtime_seconds += max(
                0.0, (current_timestamp - prior_timestamp).total_seconds()
            )
        elapsed = max(
            0.0,
            (current_timestamp - first_timestamp).total_seconds() - downtime_seconds,
        )
        event_seconds[event.event_index] = elapsed
        prior_timestamp = current_timestamp
    return event_seconds


def _parse_timestamp(*, timestamp_utc: str) -> datetime | None:
    """Parse one ISO timestamp string into a datetime when possible."""

    if not timestamp_utc:
        return None
    try:
        return datetime.fromisoformat(timestamp_utc)
    except ValueError:
        return None
