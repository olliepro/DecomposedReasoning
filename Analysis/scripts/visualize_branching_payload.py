"""Tree payload builders for events-only branching replay visualization."""

from __future__ import annotations

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
    "leaf_completed",
    "leaf_scored",
}


def tree_payload_for_attempt(*, state: AttemptState) -> dict[str, Any]:
    """Build interactive tree payload for one replayed attempt.

    Args:
        state: Replayed attempt state.

    Returns:
        JSON-ready payload for tree rendering and node/event inspection.
    """

    graph_nodes, graph_edges = normalized_graph(state=state)
    parent_by_child, children_by_parent = graph_maps(edges=graph_edges)
    incoming_tokens = incoming_edge_tokens(edges=graph_edges)
    candidate_preview = candidate_preview_by_node(edges=graph_edges)
    event_seconds_by_index = normalized_event_seconds_by_index(state=state)
    event_rows = node_event_rows(state=state)
    base_steps, base_tokens, final_steps, final_tokens = node_metric_caches(
        node_ids=tuple(graph_nodes.keys()),
        parent_by_child=parent_by_child,
        incoming_tokens=incoming_tokens,
        node_events=event_rows,
    )
    annotate_node_event_metrics(
        node_events=event_rows,
        base_steps=base_steps,
        base_tokens=base_tokens,
        event_seconds_by_index=event_seconds_by_index,
    )
    depth_by_node = compute_depths(nodes=graph_nodes)
    node_rows = node_payload_rows(
        nodes=graph_nodes,
        candidate_preview=candidate_preview,
        node_events=event_rows,
        depth_by_node=depth_by_node,
        final_steps=final_steps,
        final_tokens=final_tokens,
        state=state,
        event_seconds_by_index=event_seconds_by_index,
    )
    edge_rows = edge_payload_rows(edges=graph_edges)
    maxima = payload_maxima(node_rows=node_rows, node_events=event_rows)
    return {
        "nodes": node_rows,
        "edges": edge_rows,
        "branches": [],
        "node_events": event_rows,
        "meta": {
            "x_max": maxima,
            "branch_count": 0,
            "node_count": len(node_rows),
            "event_count": state.event_count,
            "started_at": state.events[0].timestamp_utc if state.events else "",
        },
    }


def normalized_graph(
    *, state: AttemptState
) -> tuple[dict[str, NodeView], list[EdgeView]]:
    """Return graph nodes/edges with implied nodes synthesized from edge rows."""

    nodes = dict(state.nodes)
    edges = [edge for edge in state.edges if edge.parent_node_id and edge.child_node_id]
    if "node_root" not in nodes:
        nodes["node_root"] = NodeView(
            node_id="node_root",
            parent_node_id=None,
            branch_points_used=0,
        )
    for edge in edges:
        if edge.parent_node_id not in nodes:
            nodes[edge.parent_node_id] = NodeView(
                node_id=edge.parent_node_id,
                parent_node_id=None,
                branch_points_used=0,
            )
        if edge.child_node_id not in nodes:
            nodes[edge.child_node_id] = NodeView(
                node_id=edge.child_node_id,
                parent_node_id=edge.parent_node_id,
                branch_points_used=0,
            )
    return nodes, edges


def graph_maps(*, edges: list[EdgeView]) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Build parent/children maps from edge rows."""

    parent_by_child: dict[str, str] = {}
    children_by_parent: dict[str, list[str]] = {}
    for edge in edges:
        parent_by_child[edge.child_node_id] = edge.parent_node_id
        children_by_parent.setdefault(edge.parent_node_id, []).append(
            edge.child_node_id
        )
    for children in children_by_parent.values():
        children.sort()
    return parent_by_child, children_by_parent


def incoming_edge_tokens(*, edges: list[EdgeView]) -> dict[str, int]:
    """Return selected-edge token counts contributed to each child node."""

    counts: dict[str, int] = {}
    for edge in edges:
        counts[edge.child_node_id] = len(edge.candidate_token_ids)
    return counts


def candidate_preview_by_node(*, edges: list[EdgeView]) -> dict[str, str]:
    """Return compact selected-candidate preview text for each child node."""

    preview_by_node = {"node_root": "Root"}
    for edge in edges:
        preview_by_node[edge.child_node_id] = clean_candidate_preview(
            text=edge.candidate_text,
            max_chars=72,
        )
    return preview_by_node


def clean_candidate_preview(*, text: str, max_chars: int) -> str:
    """Return readable one-line candidate preview text for node titles."""

    collapsed = " ".join(str(text).replace("\n", " ").replace("\t", " ").split())
    if not collapsed:
        return "(empty)"
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."


def node_event_rows(*, state: AttemptState) -> dict[str, list[dict[str, Any]]]:
    """Build node-scoped rows for visible runtime events and merged vLLM steps."""
    rows: dict[str, list[dict[str, Any]]] = {}
    pending_requests: dict[str, EventEnvelope] = {}
    selected_by_branch_point = selected_candidates_by_branch_point(events=state.events)
    for event in state.events:
        if event.event_type == "vllm_request":
            request_id = str(event.payload.get("request_id", ""))
            if request_id:
                pending_requests[request_id] = event
            continue
        if event.event_type == "vllm_response":
            request_id = str(event.payload.get("request_id", ""))
            request_event = pending_requests.pop(request_id, None)
            row = vllm_step_row(
                request_event=request_event, response_event=event, state=state
            )
            if row is not None:
                append_node_row(node_rows=rows, row=row)
            continue
        if event.event_type not in VISIBLE_RUNTIME_EVENT_TYPES:
            continue
        row = runtime_event_row(
            event=event,
            state=state,
            selected_by_branch_point=selected_by_branch_point,
        )
        if row is None:
            continue
        append_node_row(node_rows=rows, row=row)
    for request_event in pending_requests.values():
        row = vllm_step_row(
            request_event=request_event, response_event=None, state=state
        )
        if row is None:
            continue
        append_node_row(node_rows=rows, row=row)
    for node_rows in rows.values():
        node_rows.sort(key=lambda item: int(item["event_index"]))
    return rows


def selected_candidates_by_branch_point(
    *, events: list[EventEnvelope]
) -> dict[str, set[int]]:
    """Return selected candidate ids keyed by branch point id from selector events."""

    selected: dict[str, set[int]] = {}
    for event in events:
        if event.event_type != "selector_applied":
            continue
        payload = event.payload
        branch_point_id = str(payload.get("branch_point_id", ""))
        raw_ids = payload.get("selected_candidate_ids", [])
        if not branch_point_id or not isinstance(raw_ids, list):
            continue
        selected[branch_point_id] = {int(value) for value in raw_ids}
    return selected


def runtime_event_row(
    *,
    event: EventEnvelope,
    state: AttemptState,
    selected_by_branch_point: dict[str, set[int]],
) -> dict[str, Any] | None:
    """Build one visible non-vLLM runtime event row for tree visualization."""

    node_id = event_node_id(event=event, state=state)
    if node_id is None:
        return None
    details = runtime_event_details(
        event=event,
        selected_by_branch_point=selected_by_branch_point,
    )
    return {
        "event_id": f"{event.event_index}:{node_id}:{event.event_type}",
        "node_id": node_id,
        "event_index": event.event_index,
        "timestamp_utc": event.timestamp_utc,
        "event_type": event.event_type,
        "summary": runtime_event_summary(event=event),
        "details": details,
        "step_delta": 1,
        "token_delta": 0,
    }


def runtime_event_summary(*, event: EventEnvelope) -> str:
    """Return concise summary text for visible runtime events."""

    payload = event.payload
    if event.event_type == "trigger_fired":
        return f"trigger {str(payload.get('trigger_type', ''))}"
    if event.event_type == "trigger_skipped_max_branch_points":
        return "trigger skipped max branch points"
    if event.event_type == "candidate_pool_resolved":
        return f"candidate pool n={int(payload.get('num_candidates', 0))}"
    if event.event_type == "selector_applied":
        selected_ids = payload.get("selected_candidate_ids", [])
        count = len(selected_ids) if isinstance(selected_ids, list) else 0
        return f"selector kept {count}"
    if event.event_type == "leaf_completed":
        return f"leaf completed {str(payload.get('leaf_id', ''))}"
    if event.event_type == "leaf_scored":
        return f"leaf scored verify={payload.get('verification')}"
    return event.event_type


def runtime_event_details(
    *,
    event: EventEnvelope,
    selected_by_branch_point: dict[str, set[int]],
) -> dict[str, Any]:
    """Return inspector details for visible runtime events."""

    payload = event.payload
    details = {
        "event_index": event.event_index,
        "timestamp_utc": event.timestamp_utc,
        "event_type": event.event_type,
    }
    if event.event_type == "candidate_pool_resolved":
        branch_point_id = str(payload.get("branch_point_id", ""))
        selected_ids = selected_by_branch_point.get(branch_point_id, set())
        details.update(
            {
                "branch_point_id": branch_point_id,
                "candidate_pool_id": payload.get("candidate_pool_id"),
                "node_id": payload.get("node_id"),
                "trigger_type": payload.get("trigger_type"),
                "loaded_from_cache": payload.get("loaded_from_cache"),
                "num_candidates": payload.get("num_candidates"),
                "selected_candidate_ids": sorted(selected_ids),
                "candidates": annotate_candidates(
                    candidates=payload.get("candidates"),
                    selected_ids=selected_ids,
                ),
            }
        )
        return details
    if event.event_type == "selector_applied":
        details.update(payload)
        return details
    if event.event_type == "leaf_scored":
        details.update(
            {
                "leaf_id": payload.get("leaf_id"),
                "node_id": payload.get("node_id"),
                "verification": payload.get("verification"),
                "length_tokens_total": payload.get("length_tokens_total"),
                "stop_reason": payload.get("stop_reason"),
                "task_metrics": payload.get("task_metrics"),
                "text": payload.get("text"),
            }
        )
        return details
    details.update(payload)
    return details


def annotate_candidates(
    *, candidates: Any, selected_ids: set[int]
) -> list[dict[str, Any]]:
    """Add `selected` flag and compact token stats to candidate payload rows."""

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
                "text": candidate.get("text"),
                "output_token_count": candidate.get("output_token_count"),
                "finish_reason": candidate.get("finish_reason"),
                "stop_reason": candidate.get("stop_reason"),
                "tokens": summarized_candidate_tokens(tokens=candidate.get("tokens")),
                "selected": candidate_id in selected_ids,
            }
        )
    return rows


def summarized_candidate_tokens(*, tokens: Any) -> list[dict[str, Any]]:
    """Return token stats rows for one candidate inspector payload."""

    if not isinstance(tokens, list):
        return []
    rows: list[dict[str, Any]] = []
    for token in tokens:
        if not isinstance(token, dict):
            continue
        rows.append(
            {
                "token_text": token.get("token_text"),
                "selected_logprob": token.get("selected_logprob"),
                "selected_probability": token.get("selected_probability"),
                "selected_entropy": token.get("selected_entropy"),
                "top_logprob_alternatives": token.get("top_logprob_alternatives", []),
            }
        )
    return rows


def vllm_step_row(
    *,
    request_event: EventEnvelope | None,
    response_event: EventEnvelope | None,
    state: AttemptState,
) -> dict[str, Any] | None:
    """Build one merged vLLM step row from request/response events."""

    if request_event is None and response_event is None:
        return None
    anchor = response_event if response_event is not None else request_event
    assert anchor is not None
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
    step_token_delta = vllm_step_token_delta(
        request_payload=request_payload,
        response_payload=response_payload,
    )
    summary = vllm_step_summary(
        request_payload=request_payload,
        response_payload=response_payload,
    )
    return {
        "event_id": f"{event_index}:{node_id}:vllm_step",
        "node_id": node_id,
        "event_index": event_index,
        "timestamp_utc": timestamp_utc,
        "event_type": "vllm_step",
        "summary": summary,
        "details": vllm_step_details(
            request_payload=request_payload,
            response_payload=response_payload,
        ),
        "step_delta": 1,
        "token_delta": step_token_delta,
    }


def vllm_step_summary(
    *, request_payload: dict[str, Any], response_payload: dict[str, Any]
) -> str:
    """Return compact summary for merged vLLM step rows."""

    request_kind = str(
        request_payload.get("request_kind", response_payload.get("request_kind", ""))
    )
    delta = int(request_payload.get("delta_token_count", 0))
    status = str(response_payload.get("status", "pending"))
    if status != "ok":
        return f"{request_kind} request Δ{delta} ({status})"
    choices = response_payload.get("choices", [])
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        output_tokens = int(first.get("output_token_count", 0))
    else:
        output_tokens = 0
    return f"{request_kind} Δ{delta} -> +{output_tokens} tok"


def vllm_step_token_delta(
    *, request_payload: dict[str, Any], response_payload: dict[str, Any]
) -> int:
    """Return token delta contributed by one merged vLLM step."""

    stream_id = str(
        request_payload.get(
            "request_stream_id", response_payload.get("request_stream_id", "")
        )
    )
    if not stream_id.startswith("decode:"):
        return 0
    status = str(response_payload.get("status", "pending"))
    if status != "ok":
        return 0
    choices = response_payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return 0
    first = choices[0]
    if not isinstance(first, dict):
        return 0
    return int(first.get("output_token_count", 0))


def vllm_step_details(
    *,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
) -> dict[str, Any]:
    """Build combined request/response details for one merged vLLM step."""

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
        "temperature": request_payload.get("temperature"),
        "top_p": request_payload.get("top_p"),
        "max_tokens": request_payload.get("max_tokens"),
        "n": request_payload.get("n"),
        "seed": request_payload.get("seed"),
        "stop": request_payload.get("stop"),
        "top_logprobs": request_payload.get("top_logprobs"),
        "current_input_token_count": request_payload.get("current_input_token_count"),
        "base_prefix_token_count": request_payload.get("base_prefix_token_count"),
        "delta_token_count": request_payload.get("delta_token_count"),
        "delta_input_token_ids": request_payload.get("delta_input_token_ids"),
        "assistant_prefix_tail": request_payload.get("assistant_prefix_tail"),
        "status": response_payload.get("status", "pending"),
        "latency_seconds": response_payload.get("latency_seconds"),
        "error_message": response_payload.get("error_message"),
        "choices": summarized_choices(choices=response_payload.get("choices")),
    }


def summarized_choices(*, choices: Any) -> list[dict[str, Any]]:
    """Return full vLLM choice rows with token-level payloads untruncated."""

    if not isinstance(choices, list):
        return []
    rows: list[dict[str, Any]] = []
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        token_rows = choice.get("tokens")
        rows.append(
            {
                "index": choice.get("index"),
                "text": choice.get("text"),
                "finish_reason": choice.get("finish_reason"),
                "stop_reason": choice.get("stop_reason"),
                "output_token_count": choice.get("output_token_count"),
                "tokens": token_rows if isinstance(token_rows, list) else [],
            }
        )
    return rows


def event_node_id(*, event: EventEnvelope, state: AttemptState) -> str | None:
    """Resolve canonical node id for one event row when available."""

    payload = event.payload
    if event.event_type in {
        "trigger_fired",
        "trigger_skipped_max_branch_points",
        "candidate_pool_resolved",
        "selector_applied",
        "leaf_completed",
        "leaf_scored",
    }:
        node_id = str(payload.get("node_id", ""))
        return node_id if node_id else None
    if event.event_type == "vllm_request":
        stream_id = str(payload.get("request_stream_id", ""))
        return decode_node_id_from_stream_id(request_stream_id=stream_id)
    if event.event_type == "vllm_response":
        request_id = str(payload.get("request_id", ""))
        mapped = state.request_node_map.get(request_id)
        if mapped is not None:
            return mapped
        stream_id = str(payload.get("request_stream_id", ""))
        return decode_node_id_from_stream_id(request_stream_id=stream_id)
    return None


def append_node_row(
    *,
    node_rows: dict[str, list[dict[str, Any]]],
    row: dict[str, Any],
) -> None:
    """Append one event row into node-keyed event mapping."""

    node_id = str(row["node_id"])
    node_rows.setdefault(node_id, []).append(row)


def node_metric_caches(
    *,
    node_ids: tuple[str, ...],
    parent_by_child: dict[str, str],
    incoming_tokens: dict[str, int],
    node_events: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    """Compute base/final step+token totals for each node path."""

    event_steps = {node_id: len(rows) for node_id, rows in node_events.items()}
    event_tokens = {
        node_id: sum(int(row.get("token_delta", 0)) for row in rows)
        for node_id, rows in node_events.items()
    }
    base_steps: dict[str, int] = {}
    base_tokens: dict[str, int] = {}
    final_steps: dict[str, int] = {}
    final_tokens: dict[str, int] = {}
    visiting: set[str] = set()

    def compute(node_id: str) -> tuple[int, int, int, int]:
        cached_steps = final_steps.get(node_id)
        cached_tokens = final_tokens.get(node_id)
        if cached_steps is not None and cached_tokens is not None:
            return (
                base_steps.get(node_id, 0),
                base_tokens.get(node_id, 0),
                cached_steps,
                cached_tokens,
            )
        assert node_id not in visiting, f"cycle detected in node graph at {node_id}"
        visiting.add(node_id)
        parent_id = parent_by_child.get(node_id)
        if parent_id is None:
            parent_final_steps = 0
            parent_final_tokens = 0
        else:
            _, _, parent_final_steps, parent_final_tokens = compute(parent_id)
        start_steps = parent_final_steps
        start_tokens = parent_final_tokens + incoming_tokens.get(node_id, 0)
        end_steps = start_steps + event_steps.get(node_id, 0)
        end_tokens = start_tokens + event_tokens.get(node_id, 0)
        base_steps[node_id] = start_steps
        base_tokens[node_id] = start_tokens
        final_steps[node_id] = end_steps
        final_tokens[node_id] = end_tokens
        visiting.remove(node_id)
        return start_steps, start_tokens, end_steps, end_tokens

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
    """Attach steps/tokens/time metrics to each node-local event row in place."""

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


def compute_depths(*, nodes: dict[str, NodeView]) -> dict[str, int]:
    """Compute root-relative depth for each replayed node."""

    depth_by_node: dict[str, int] = {}

    def depth_for(node_id: str) -> int:
        cached = depth_by_node.get(node_id)
        if cached is not None:
            return cached
        row = nodes.get(node_id)
        if row is None or row.parent_node_id is None or row.parent_node_id == "":
            depth_by_node[node_id] = 0
            return 0
        depth = depth_for(row.parent_node_id) + 1
        depth_by_node[node_id] = depth
        return depth

    for node_id in nodes:
        _ = depth_for(node_id)
    return depth_by_node


def node_payload_rows(
    *,
    nodes: dict[str, NodeView],
    candidate_preview: dict[str, str],
    node_events: dict[str, list[dict[str, Any]]],
    depth_by_node: dict[str, int],
    final_steps: dict[str, int],
    final_tokens: dict[str, int],
    state: AttemptState,
    event_seconds_by_index: dict[int, float],
) -> list[dict[str, Any]]:
    """Build client payload rows for rendered tree nodes."""

    rows: list[dict[str, Any]] = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        first_event_index = state.node_first_event_index.get(node_id)
        time_seconds = (
            event_seconds_by_index.get(first_event_index, 0.0)
            if first_event_index is not None
            else 0.0
        )
        rows.append(
            {
                "node_id": node.node_id,
                "parent_node_id": node.parent_node_id,
                "branch_points_used": node.branch_points_used,
                "depth": depth_by_node.get(node_id, 0),
                "event_count": len(node_events.get(node_id, [])),
                "candidate_preview": candidate_preview.get(node_id, "(unknown)"),
                "metrics": {
                    "steps": final_steps.get(node_id, 0),
                    "tokens": final_tokens.get(node_id, 0),
                    "time_seconds": time_seconds,
                },
            }
        )
    rows.sort(key=lambda row: (int(row["depth"]), str(row["node_id"])))
    return rows


def normalized_event_seconds_by_index(*, state: AttemptState) -> dict[int, float]:
    """Return resume-aware elapsed seconds keyed by event index.

    Args:
        state: Replayed attempt state for one doc attempt.

    Returns:
        Mapping from `event_index` to logical elapsed seconds where resume downtime
        between attempts is removed.

    Example:
        If a run pauses for one hour before `doc_resumed`, that one-hour gap is
        subtracted so time remains continuous in the tree axis.
    """

    if not state.events:
        return {}
    ordered_events = sorted(state.events, key=lambda row: row.event_index)
    start_seconds = timestamp_to_seconds(timestamp_utc=ordered_events[0].timestamp_utc)
    previous_seconds = start_seconds
    paused_seconds = 0.0
    elapsed_by_index: dict[int, float] = {}
    for event in ordered_events:
        current_seconds = timestamp_to_seconds(timestamp_utc=event.timestamp_utc)
        delta_seconds = max(0.0, current_seconds - previous_seconds)
        if event.event_type == "doc_resumed":
            paused_seconds += delta_seconds
        elapsed_seconds = max(0.0, current_seconds - start_seconds - paused_seconds)
        elapsed_by_index[event.event_index] = elapsed_seconds
        previous_seconds = current_seconds
    return elapsed_by_index


def edge_payload_rows(*, edges: list[EdgeView]) -> list[dict[str, Any]]:
    """Build edge payload rows for client rendering."""

    return [
        {
            "parent_node_id": edge.parent_node_id,
            "child_node_id": edge.child_node_id,
            "candidate_id": edge.candidate_id,
            "selector_mode": edge.selector_mode,
            "candidate_token_count": len(edge.candidate_token_ids),
            "candidate_preview": clean_candidate_preview(
                text=edge.candidate_text,
                max_chars=120,
            ),
        }
        for edge in edges
    ]


def payload_maxima(
    *,
    node_rows: list[dict[str, Any]],
    node_events: dict[str, list[dict[str, Any]]],
) -> dict[str, float]:
    """Return max metric values used to build axis scales in client code."""

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


def timestamp_to_seconds(*, timestamp_utc: str) -> float:
    """Convert ISO timestamp to epoch seconds."""

    normalized = timestamp_utc.replace("Z", "+00:00")
    return datetime.fromisoformat(normalized).timestamp()
