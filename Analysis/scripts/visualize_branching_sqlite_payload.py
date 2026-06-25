"""SQLite-typed payload builders for the dynamic branching viewer."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from branching_eval.event_db import EventDatabase

try:
    from scripts.visualize_branching_common import AttemptKey, clean_candidate_preview
except ModuleNotFoundError:
    from visualize_branching_common import AttemptKey, clean_candidate_preview

BASELINE_LEAF_NODE_PREFIX = "node_leaf_baseline_"
PROMPT_DETAIL_HEAD_CHARS = 4000
PROMPT_DETAIL_TAIL_CHARS = 2000
SELECTOR_DECISION_COMPANION_TYPES = {
    "candidate_pool_resolved",
    "selector_continued_inline",
}
CHUNK_DETAIL_EVENT_TYPES = {
    "repeat_forced_think_close",
}
VERBALIZED_SAMPLING_EVENT_TYPES = {"verbalized_sampling_applied"}


@dataclass(frozen=True)
class AttemptPageData:
    """Lightweight attempt page data loaded from typed SQLite tables."""

    key: AttemptKey
    status: str
    event_count: int
    node_count: int
    edge_count: int
    leaf_count: int
    last_event_index: int
    last_timestamp_utc: str


def tree_payload_from_sqlite(
    *, db: EventDatabase, key: AttemptKey, detail_base_url: str
) -> dict[str, Any]:
    """Build graph JSON from typed SQLite rows only."""

    node_rows = db.read_node_rows_for_attempt(**_key_kwargs(key=key))
    edge_rows = db.read_edge_rows_for_attempt(**_key_kwargs(key=key))
    baseline_leaf_rows = db.read_baseline_leaf_rows_for_attempt(**_key_kwargs(key=key))
    display_edge_rows = [
        *edge_rows,
        *_baseline_leaf_edge_rows(rows=baseline_leaf_rows),
    ]
    event_rows = _collapse_selector_decision_rows(
        rows=[
            *db.read_node_event_rows_for_attempt(**_key_kwargs(key=key)),
            *db.read_vllm_step_graph_rows_for_attempt(**_key_kwargs(key=key)),
        ]
    )
    summary_row = db.read_attempt_count_row(**_key_kwargs(key=key))
    diagnostics_row = db.read_doc_diagnostics_row(**_key_kwargs(key=key))
    nodes = _normalized_nodes(rows=node_rows, edges=display_edge_rows)
    _add_baseline_leaf_nodes(nodes=nodes, rows=baseline_leaf_rows)
    edge_payloads = _edge_payload_rows(rows=display_edge_rows)
    if not edge_payloads:
        edge_payloads = _synthetic_edge_payload_rows(nodes=nodes)
    parent_by_child = _parent_by_child(edges=edge_payloads)
    incoming_tokens = _incoming_token_counts(rows=display_edge_rows)
    depth_by_node = _depths(nodes=nodes)
    raw_events_by_node = _raw_event_rows_by_node(rows=event_rows)
    _move_baseline_leaf_events(rows_by_node=raw_events_by_node)
    event_seconds_by_index = _event_seconds_by_index(rows=event_rows)
    base_steps, base_tokens, final_steps, final_tokens = _path_metric_caches(
        nodes=nodes,
        parent_by_child=parent_by_child,
        incoming_tokens=incoming_tokens,
        event_rows_by_node=raw_events_by_node,
    )
    full_node_events = _event_rows_by_node(
        rows_by_node=raw_events_by_node,
        base_steps=base_steps,
        base_tokens=base_tokens,
        event_seconds_by_index=event_seconds_by_index,
    )
    event_count_by_node = {
        node_id: len(events) for node_id, events in full_node_events.items()
    }
    metrics_by_node = _final_metrics(
        nodes=nodes,
        final_steps=final_steps,
        final_tokens=final_tokens,
        event_seconds_by_index=event_seconds_by_index,
        node_events=full_node_events,
    )
    advantage_by_node = _advantage_by_child_node(
        rows=(
            db.read_node_advantage_rows_for_attempt(**_key_kwargs(key=key))
            if _is_rl_attempt_key(key=key)
            else []
        )
    )
    node_payloads = _node_payload_rows(
        nodes=nodes,
        depth_by_node=depth_by_node,
        metrics_by_node=metrics_by_node,
        node_events=full_node_events,
        event_count_by_node=event_count_by_node,
        advantage_by_node=advantage_by_node,
        detail_base_url=detail_base_url,
    )
    return {
        "nodes": node_payloads,
        "edges": edge_payloads,
        "branches": [],
        "node_events": full_node_events,
        "meta": _tree_meta(
            key=key,
            summary_row=summary_row,
            diagnostics_row=diagnostics_row,
            nodes=node_payloads,
            edges=edge_payloads,
            node_events=full_node_events,
        ),
    }


def node_payload_from_sqlite(
    *, db: EventDatabase, key: AttemptKey, node_id: str
) -> dict[str, Any] | None:
    """Build one clicked-node payload from typed SQLite rows."""

    baseline_leaf = _baseline_leaf_row_for_display_node(db=db, key=key, node_id=node_id)
    if baseline_leaf is not None:
        return _baseline_leaf_node_payload(db=db, key=key, leaf_row=baseline_leaf)

    node_row = db.read_node_row_for_attempt(
        node_id=node_id,
        **_key_kwargs(key=key),
    )
    detail_rows = db.read_node_detail_rows(
        node_id=node_id,
        **_key_kwargs(key=key),
    )
    detail_rows = _collapse_selector_decision_rows(
        rows=[
            *detail_rows,
            *db.read_vllm_step_rows_for_node(
                node_id=node_id,
                **_key_kwargs(key=key),
            ),
        ]
    )
    leaf_rows = db.read_leaf_rows_for_node(node_id=node_id, **_key_kwargs(key=key))
    path_rows = db.read_edge_path_rows_for_node(
        node_id=node_id,
        **_key_kwargs(key=key),
    )
    if node_row is None and not detail_rows and not leaf_rows:
        return None
    events = _light_event_rows(rows=detail_rows)
    node = _node_detail_row(node_id=node_id, row=node_row)
    node["event_count"] = len(events)
    node["token_total"] = sum(int(event.get("token_delta") or 0) for event in events)
    return {
        "node": node,
        "events": events,
        "leaves": _leaf_payload_rows(rows=leaf_rows),
        "trajectory": _trajectory_payload(
            edge_rows=path_rows,
            text_rows=[],
            leaf_rows=leaf_rows,
        ),
    }


def event_payload_from_sqlite(
    *, db: EventDatabase, event_index: int
) -> dict[str, Any] | None:
    """Build one clicked-event payload with heavy token details loaded lazily."""

    row = db.read_vllm_step_row_by_event(event_index=event_index)
    if row is None:
        row = db.read_node_event_row_by_index(event_index=event_index)
    if row is None:
        return None
    return _detail_event_rows(db=db, rows=[row])[0]


def token_trajectory_payload_from_sqlite(
    *, db: EventDatabase, key: AttemptKey, node_id: str
) -> dict[str, Any]:
    """Return stored final text for one leaf node without token reconstruction."""

    baseline_leaf = _baseline_leaf_row_for_display_node(db=db, key=key, node_id=node_id)
    if baseline_leaf is not None:
        return _baseline_leaf_text_payload(db=db, key=key, leaf_row=baseline_leaf)

    leaf_rows = db.read_leaf_rows_for_node(node_id=node_id, **_key_kwargs(key=key))
    leaf_row = max(
        leaf_rows,
        key=lambda row: int(row.get("event_index") or -1),
        default={},
    )
    return {
        "node_id": node_id,
        "token_count": int(leaf_row.get("length_tokens_total") or 0),
        "text": str(leaf_row.get("text") or ""),
        "tokens": [],
    }


def attempt_page_data_from_sqlite(
    *, db: EventDatabase, key: AttemptKey
) -> AttemptPageData:
    """Return lightweight counts for one doc-attempt page."""

    row = db.read_attempt_count_row(**_key_kwargs(key=key))
    return AttemptPageData(
        key=key,
        status=_status_from_row(row=row),
        event_count=int(row.get("event_count", 0)),
        node_count=int(row.get("node_count", 0)),
        edge_count=int(row.get("edge_count", 0)),
        leaf_count=int(row.get("leaf_count", 0)),
        last_event_index=int(row.get("last_event_index", -1)),
        last_timestamp_utc=str(row.get("last_timestamp_utc", "")),
    )


def _normalized_nodes(
    *, rows: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    nodes = {str(row["node_id"]): dict(row) for row in rows}
    nodes.setdefault("node_root", _synthetic_node(node_id="node_root", parent_id=None))
    for edge in edges:
        parent = str(edge["parent_node_id"])
        child = str(edge["child_node_id"])
        nodes.setdefault(parent, _synthetic_node(node_id=parent, parent_id=None))
        nodes.setdefault(child, _synthetic_node(node_id=child, parent_id=parent))
        nodes[child]["parent_node_id"] = nodes[child].get("parent_node_id") or parent
        nodes[child]["candidate_preview"] = edge.get("candidate_text") or "Root"
        nodes[child]["token_total"] = int(nodes[child].get("token_total") or 0)
    return nodes


def _baseline_rollout_index_for_row(*, row: dict[str, Any]) -> int | None:
    return _baseline_rollout_index(leaf_id=str(row.get("leaf_id") or ""))


def _baseline_leaf_node_id(*, leaf_id: str) -> str:
    return f"{BASELINE_LEAF_NODE_PREFIX}{leaf_id.removeprefix('leaf_baseline_')}"


def _baseline_leaf_edge_rows(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edge_rows = []
    for row in rows:
        parent_node_id = str(row.get("node_id") or "node_root")
        leaf_id = str(row.get("leaf_id") or "")
        edge_rows.append(
            {
                "event_index": int(row.get("event_index") or -1),
                "parent_node_id": parent_node_id,
                "child_node_id": _baseline_leaf_node_id(leaf_id=leaf_id),
                "candidate_id": _baseline_rollout_index_for_row(row=row),
                "selector_mode": "baseline_leaf",
                "candidate_text": _baseline_leaf_label(row=row),
                "candidate_token_count": int(row.get("length_tokens_total") or 0),
            }
        )
    return edge_rows


def _add_baseline_leaf_nodes(
    *, nodes: dict[str, dict[str, Any]], rows: list[dict[str, Any]]
) -> None:
    parent_leaf_counts: dict[str, int] = {}
    for row in rows:
        parent_node_id = str(row.get("node_id") or "node_root")
        leaf_id = str(row.get("leaf_id") or "")
        node_id = _baseline_leaf_node_id(leaf_id=leaf_id)
        parent_leaf_counts[parent_node_id] = (
            parent_leaf_counts.get(parent_node_id, 0) + 1
        )
        nodes[node_id] = {
            **_synthetic_node(node_id=node_id, parent_id=parent_node_id),
            "candidate_preview": _baseline_leaf_label(row=row),
            "leaf_count": 1,
            "token_total": int(row.get("length_tokens_total") or 0),
            "first_event_index": int(row.get("event_index") or -1),
        }
    for parent_node_id, leaf_count in parent_leaf_counts.items():
        if parent_node_id in nodes:
            current = int(nodes[parent_node_id].get("leaf_count") or 0)
            nodes[parent_node_id]["leaf_count"] = max(0, current - leaf_count)


def _baseline_leaf_label(*, row: dict[str, Any]) -> str:
    rollout_index = _baseline_rollout_index_for_row(row=row)
    prefix = (
        f"Baseline rollout {rollout_index}" if rollout_index is not None else "Leaf"
    )
    preview = clean_candidate_preview(
        text=str(row.get("text_preview") or row.get("text") or ""),
        max_chars=56,
    )
    return f"{prefix}: {preview}" if preview else prefix


def _synthetic_node(*, node_id: str, parent_id: str | None) -> dict[str, Any]:
    return {
        "node_id": node_id,
        "parent_node_id": parent_id,
        "branch_points_used": 0,
        "candidate_preview": "Root" if parent_id is None else "",
        "event_count": 0,
        "leaf_count": 0,
        "token_total": 0,
        "first_event_index": -1,
        "first_timestamp_utc": "",
    }


def _node_detail_row(*, node_id: str, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return _synthetic_node(node_id=node_id, parent_id=None)
    return {
        "node_id": str(row["node_id"]),
        "parent_node_id": row.get("parent_node_id"),
        "branch_points_used": int(row.get("branch_points_used") or 0),
        "candidate_preview": _preview(node=row),
        "event_count": int(row.get("event_count") or 0),
        "leaf_count": int(row.get("leaf_count") or 0),
        "token_total": int(row.get("token_total") or 0),
        "first_event_index": int(row.get("first_event_index") or -1),
        "first_timestamp_utc": str(row.get("first_timestamp_utc") or ""),
    }


def _parent_by_child(*, edges: list[dict[str, Any]]) -> dict[str, str]:
    return {
        str(edge["child_node_id"]): str(edge["parent_node_id"])
        for edge in edges
        if edge.get("child_node_id") and edge.get("parent_node_id")
    }


def _depths(*, nodes: dict[str, dict[str, Any]]) -> dict[str, int]:
    depths: dict[str, int] = {}

    def compute(node_id: str) -> int:
        if node_id in depths:
            return depths[node_id]
        parent = nodes.get(node_id, {}).get("parent_node_id")
        depths[node_id] = 0 if not parent else compute(str(parent)) + 1
        return depths[node_id]

    for node_id in nodes:
        compute(node_id)
    return depths


def _incoming_token_counts(*, rows: list[dict[str, Any]]) -> dict[str, int]:
    return {
        str(row["child_node_id"]): int(row.get("candidate_token_count") or 0)
        for row in rows
        if row.get("child_node_id")
    }


def _raw_event_rows_by_node(
    *, rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        node_id = str(row["node_id"])
        grouped.setdefault(node_id, []).append(row)
    return grouped


def _move_baseline_leaf_events(
    *, rows_by_node: dict[str, list[dict[str, Any]]]
) -> None:
    moved_rows: list[tuple[str, dict[str, Any]]] = []
    for node_id, rows in list(rows_by_node.items()):
        kept_rows = []
        for row in rows:
            if str(row.get("event_type") or "") not in {
                "leaf_completed",
                "leaf_scored",
            }:
                kept_rows.append(row)
                continue
            leaf_id = str(row.get("leaf_id") or "")
            rollout_index = _baseline_rollout_index(leaf_id=leaf_id)
            if rollout_index is None:
                kept_rows.append(row)
                continue
            display_node_id = _baseline_leaf_node_id(leaf_id=leaf_id)
            moved_rows.append((display_node_id, {**row, "node_id": display_node_id}))
        rows_by_node[node_id] = kept_rows
    for node_id, row in moved_rows:
        rows_by_node.setdefault(node_id, []).append(row)


def _path_metric_caches(
    *,
    nodes: dict[str, dict[str, Any]],
    parent_by_child: dict[str, str],
    incoming_tokens: dict[str, int],
    event_rows_by_node: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, int], dict[str, int], dict[str, int], dict[str, int]]:
    base_steps: dict[str, int] = {}
    base_tokens: dict[str, int] = {}
    final_steps: dict[str, int] = {}
    final_tokens: dict[str, int] = {}

    def compute(node_id: str) -> tuple[int, int]:
        if node_id in final_steps and node_id in final_tokens:
            return final_steps[node_id], final_tokens[node_id]
        node = nodes[node_id]
        parent = parent_by_child.get(node_id) or node.get("parent_node_id")
        parent_steps, parent_tokens = (
            compute(str(parent)) if parent and parent in nodes else (0, 0)
        )
        base_steps[node_id] = parent_steps
        base_tokens[node_id] = parent_tokens + incoming_tokens.get(node_id, 0)
        final_steps[node_id] = base_steps[node_id] + len(
            event_rows_by_node.get(node_id, [])
        )
        final_tokens[node_id] = base_tokens[node_id] + sum(
            _event_token_delta(row=row) for row in event_rows_by_node.get(node_id, [])
        )
        return final_steps[node_id], final_tokens[node_id]

    for node_id in nodes:
        compute(node_id)
    return base_steps, base_tokens, final_steps, final_tokens


def _final_metrics(
    *,
    nodes: dict[str, dict[str, Any]],
    final_steps: dict[str, int],
    final_tokens: dict[str, int],
    event_seconds_by_index: dict[int, float],
    node_events: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    return {
        node_id: {
            "steps": float(final_steps.get(node_id, 0)),
            "tokens": float(final_tokens.get(node_id, 0)),
            "time_seconds": _node_final_seconds(
                events=node_events.get(node_id, []),
                event_seconds_by_index=event_seconds_by_index,
            ),
        }
        for node_id in nodes
    }


def _event_rows_by_node(
    *,
    rows_by_node: dict[str, list[dict[str, Any]]],
    base_steps: dict[str, int],
    base_tokens: dict[str, int],
    event_seconds_by_index: dict[int, float],
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for node_id, rows in rows_by_node.items():
        running_steps = base_steps.get(node_id, 0)
        running_tokens = base_tokens.get(node_id, 0)
        grouped[node_id] = []
        for local_step, row in enumerate(rows, start=1):
            running_steps += int(row.get("step_delta") or 0)
            running_tokens += _event_token_delta(row=row)
            metrics = {
                "steps": float(running_steps),
                "tokens": float(running_tokens),
                "time_seconds": event_seconds_by_index.get(
                    int(row.get("event_index") or -1), 0.0
                ),
            }
            enriched = {**row, "local_step": local_step}
            grouped[node_id].append(_event_payload_row(row=enriched, metrics=metrics))
    return grouped


def _event_seconds_by_index(*, rows: list[dict[str, Any]]) -> dict[int, float]:
    timestamps = [
        _timestamp_seconds(value=str(row.get("timestamp_utc") or "")) for row in rows
    ]
    valid = [value for value in timestamps if value is not None]
    if not valid:
        return {}
    start = min(valid)
    elapsed: dict[int, float] = {}
    assert len(rows) == len(timestamps)
    for row, seconds in zip(rows, timestamps):
        if seconds is None:
            continue
        elapsed[int(row["event_index"])] = max(0.0, seconds - start)
    return elapsed


def _timestamp_seconds(*, value: str) -> float | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def _node_final_seconds(
    *, events: list[dict[str, Any]], event_seconds_by_index: dict[int, float]
) -> float:
    if not events:
        return 0.0
    event_index = int(events[-1].get("event_index") or -1)
    return event_seconds_by_index.get(event_index, 0.0)


def _light_event_rows(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for local_step, row in enumerate(rows, start=1):
        payload = _event_payload_row(
            row={**row, "local_step": local_step},
            metrics={},
        )
        if payload["event_type"] == "vllm_step":
            payload["details"] = _vllm_request_summary(row=row)
        payloads.append(payload)
    return payloads


def _event_payload_row(
    *, row: dict[str, Any], metrics: dict[str, float]
) -> dict[str, Any]:
    event = {
        "event_id": f"{row['event_index']}:{row['node_id']}:{row['event_type']}",
        "event_index": int(row["event_index"]),
        "timestamp_utc": str(row["timestamp_utc"]),
        "event_type": str(row["event_type"]),
        "node_id": str(row["node_id"]),
        "summary": _event_summary(row=row),
        "step_delta": int(row["step_delta"] or 0),
        "token_delta": _event_token_delta(row=row),
        "local_step": int(row.get("local_step") or 0),
        "metrics": metrics,
    }
    details = _basic_event_details(row=row)
    if details:
        event["details"] = details
    return event


def _event_summary(*, row: dict[str, Any]) -> str:
    if str(row["event_type"]) != "vllm_step":
        return str(row.get("summary") or row["event_type"])
    request_kind = str(row.get("request_kind") or "")
    delta = int(row.get("delta_token_count") or 0)
    status = str(row.get("status") or "pending")
    if status != "ok":
        return f"{request_kind} Δ{delta} ({status})"
    return f"{request_kind} Δ{delta} -> +{_event_token_delta(row=row)} tok"


def _event_token_delta(*, row: dict[str, Any]) -> int:
    if str(row["event_type"]) != "vllm_step":
        return int(row.get("token_delta") or 0)
    if str(row.get("status") or "pending") != "ok":
        return 0
    return int(row.get("output_token_count") or row.get("token_delta") or 0)


def _basic_event_details(*, row: dict[str, Any]) -> dict[str, Any]:
    event_type = str(row["event_type"])
    if event_type == "prompt_logged":
        return {
            "golden_answer": row.get("text_preview") or "",
            "detail_path": f"events/{int(row['event_index'])}.json",
        }
    if event_type in {"leaf_scored", "leaf_completed"}:
        return _leaf_details_from_event(row=row)
    if event_type in CHUNK_DETAIL_EVENT_TYPES:
        return {
            "text_preview": row.get("text_preview") or "",
            "detail_path": f"events/{int(row['event_index'])}.json",
        }
    if event_type == "candidate_pool_resolved":
        return {
            "branch_point_id": row.get("branch_point_id"),
            "num_candidates": row.get("summary", "").split("=")[-1],
            "candidates": [],
        }
    if event_type in {"selector_applied", "selector_continued_inline"}:
        return {"branch_point_id": row.get("branch_point_id")}
    if event_type in VERBALIZED_SAMPLING_EVENT_TYPES:
        return {
            "branch_point_id": row.get("branch_point_id"),
            "detail_path": f"events/{int(row['event_index'])}.json",
        }
    return {}


def _vllm_request_summary(*, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "request_id": row.get("request_id"),
        "request_stream_id": row.get("request_stream_id"),
        "request_kind": row.get("request_kind"),
        "current_input_token_count": row.get("current_input_token_count"),
        "base_prefix_token_count": row.get("base_prefix_token_count"),
        "delta_token_count": row.get("delta_token_count"),
        "assistant_prefix_char_count": row.get("assistant_prefix_char_count"),
        "status": row.get("status"),
        "latency_seconds": row.get("latency_seconds"),
        "error_message": row.get("error_message"),
        "choice_count": row.get("choice_count"),
        "choices": [],
        "detail_path": f"events/{int(row['event_index'])}.json",
    }


def _detail_event_rows(
    *, db: EventDatabase, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    event_indexes = _event_indexes(rows)
    selector_event_indexes = _selector_event_indexes(rows=rows)
    selector_pool_rows = _selector_pool_rows_by_event(
        rows=db.read_selector_pool_rows_for_events(event_indexes=selector_event_indexes)
    )
    pool_event_indexes = _pool_event_indexes(rows=selector_pool_rows.values())
    candidate_event_indexes = sorted({*event_indexes, *pool_event_indexes})
    candidate_rows = _rows_by_event(
        rows=db.read_candidate_rows_for_events(event_indexes=candidate_event_indexes)
    )
    candidate_tokens = _candidate_token_rows_by_event_candidate(
        rows=db.read_candidate_token_rows_for_events(
            event_indexes=candidate_event_indexes
        )
    )
    selector_flags = _rows_by_selector_event(
        rows=db.read_selector_flag_rows_for_events(event_indexes=selector_event_indexes)
    )
    selector_clusters = _rows_by_selector_event(
        rows=db.read_selector_cluster_rows_for_events(
            event_indexes=selector_event_indexes
        )
    )
    choice_rows = _rows_by_event(
        rows=db.read_vllm_choice_rows_for_events(event_indexes=event_indexes)
    )
    token_rows = _token_rows_by_event_choice(
        rows=db.read_vllm_choice_token_rows_for_events(event_indexes=event_indexes)
    )
    chunk_rows = _rows_by_event(
        rows=db.read_generated_chunk_rows_for_events(event_indexes=event_indexes)
    )
    verbalized_decisions = _rows_by_event(
        rows=db.read_verbalized_sampling_decision_rows_for_events(
            event_indexes=event_indexes
        )
    )
    verbalized_candidates = _rows_by_event(
        rows=db.read_verbalized_sampling_candidate_rows_for_events(
            event_indexes=event_indexes
        )
    )
    latest_choice_tokens_by_node: dict[str, list[dict[str, Any]]] = {}
    payloads = []
    for row in rows:
        event_index = int(row["event_index"])
        event_type = str(row["event_type"])
        node_id = str(row["node_id"])
        chunk = first_or_none(rows=chunk_rows.get(event_index, []))
        pool_row = selector_pool_rows.get(event_index)
        pool_event_index = _candidate_event_index(
            event_index=event_index,
            selector_pool_row=pool_row,
        )
        if event_type == "vllm_step" and str(row.get("status") or "") == "ok":
            latest_choice_tokens_by_node[node_id] = token_rows.get(event_index, {}).get(
                0, []
            )
        payloads.append(
            _detail_event_payload(
                row=row,
                candidates=candidate_rows.get(pool_event_index, []),
                candidate_tokens=candidate_tokens.get(pool_event_index, {}),
                selector_pool_row=pool_row,
                selector_flags=selector_flags.get(event_index, []),
                selector_clusters=selector_clusters.get(event_index, []),
                choices=choice_rows.get(event_index, []),
                choice_tokens=token_rows.get(event_index, {}),
                chunk=chunk,
                chunk_tokens=_chunk_token_rows(
                    row=row,
                    chunk=chunk,
                    latest_choice_tokens_by_node=latest_choice_tokens_by_node,
                ),
                verbalized_decision=first_or_none(
                    rows=verbalized_decisions.get(event_index, [])
                ),
                verbalized_candidates=verbalized_candidates.get(event_index, []),
            )
        )
    return payloads


def _detail_event_payload(
    *,
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    candidate_tokens: dict[int, list[dict[str, Any]]],
    selector_pool_row: dict[str, Any] | None,
    selector_flags: list[dict[str, Any]],
    selector_clusters: list[dict[str, Any]],
    choices: list[dict[str, Any]],
    choice_tokens: dict[int, list[dict[str, Any]]],
    chunk: dict[str, Any] | None,
    chunk_tokens: list[dict[str, Any]],
    verbalized_decision: dict[str, Any] | None,
    verbalized_candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    event = _event_payload_row(row=row, metrics={})
    event_type = str(row["event_type"])
    if event_type == "candidate_pool_resolved":
        event["details"] = _candidate_pool_details(
            row=row,
            candidates=candidates,
            candidate_tokens=candidate_tokens,
        )
    elif event_type == "prompt_logged":
        event["details"] = _prompt_logged_details(row=row)
    elif event_type in {"selector_applied", "selector_continued_inline"}:
        event["details"] = _selector_decision_details(
            row=row,
            pool_row=selector_pool_row,
            candidates=candidates,
            candidate_tokens=candidate_tokens,
            selector_flags=selector_flags,
            selector_clusters=selector_clusters,
        )
    elif event_type in VERBALIZED_SAMPLING_EVENT_TYPES:
        event["details"] = _verbalized_sampling_details(
            row=row,
            decision=verbalized_decision,
            candidates=verbalized_candidates,
        )
    elif event_type == "vllm_step":
        event["details"] = _vllm_details(
            row=row,
            choices=choices,
            choice_tokens=choice_tokens,
        )
    elif event_type in CHUNK_DETAIL_EVENT_TYPES:
        event["details"] = _generated_chunk_details(
            row=row,
            chunk=chunk,
            tokens=chunk_tokens,
        )
    elif event_type in {"leaf_scored", "leaf_completed"}:
        event["details"] = _leaf_details_from_event(
            row=row,
            hydrated=True,
        )
    return event


def _prompt_logged_details(*, row: dict[str, Any]) -> dict[str, Any]:
    prompt_text = str(row.get("prompt_text") or "")
    prompt_char_count = int(row.get("prompt_char_count") or len(prompt_text))
    inline_limit = PROMPT_DETAIL_HEAD_CHARS + PROMPT_DETAIL_TAIL_CHARS
    if len(prompt_text) <= inline_limit:
        prompt_head = prompt_text
        prompt_tail = ""
        omitted_chars = 0
    else:
        prompt_head = prompt_text[:PROMPT_DETAIL_HEAD_CHARS]
        prompt_tail = prompt_text[-PROMPT_DETAIL_TAIL_CHARS:]
        omitted_chars = len(prompt_text) - len(prompt_head) - len(prompt_tail)
    return {
        "prompt_detail_hydrated": True,
        "prompt_text": prompt_text if omitted_chars == 0 else "",
        "prompt_preview_head": prompt_head,
        "prompt_preview_tail": prompt_tail,
        "prompt_omitted_char_count": omitted_chars,
        "prompt_truncated": omitted_chars > 0,
        "prompt_char_count": prompt_char_count,
        "golden_answer": row.get("golden_answer") or row.get("text_preview") or "",
        "golden_answer_source": row.get("golden_answer_source") or "",
    }


def _selector_decision_details(
    *,
    row: dict[str, Any],
    pool_row: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
    candidate_tokens: dict[int, list[dict[str, Any]]],
    selector_flags: list[dict[str, Any]],
    selector_clusters: list[dict[str, Any]],
) -> dict[str, Any]:
    active_mode = _active_selector_mode(pool_row=pool_row, flags=selector_flags)
    selected_ids = _selected_candidate_ids(flags=selector_flags)
    shortlist_ids = _shortlist_candidate_ids(
        flags=selector_flags,
        active_mode=active_mode,
    )
    candidate_details = _flagged_candidate_details(
        candidates=candidates,
        candidate_tokens=candidate_tokens,
        selected_ids=set(selected_ids),
        shortlist_ids=set(shortlist_ids),
    )
    return {
        "branch_point_id": row.get("branch_point_id"),
        "node_id": row.get("node_id"),
        "active_selector_mode": active_mode,
        "candidate_pool_id": _pool_value(pool_row=pool_row, key="candidate_pool_id"),
        "trigger_type": _pool_value(pool_row=pool_row, key="trigger_type"),
        "num_candidates": len(candidate_details),
        "selected_candidate_ids": selected_ids,
        "shortlist_candidate_ids": shortlist_ids,
        "selected_candidates": _candidates_by_ids(
            candidates=candidate_details,
            candidate_ids=selected_ids,
        ),
        "shortlisted_candidates": [
            candidate
            for candidate in candidate_details
            if candidate["shortlisted"] and not candidate["selected"]
        ],
        "other_candidates": [
            candidate
            for candidate in candidate_details
            if not candidate["shortlisted"] and not candidate["selected"]
        ],
        "candidates": candidate_details,
        "cluster_groups_by_mode": _cluster_groups_by_mode(
            cluster_rows=selector_clusters,
            selector_flags=selector_flags,
        ),
    }


def _verbalized_sampling_details(
    *,
    row: dict[str, Any],
    decision: dict[str, Any] | None,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return click-detail payload for verbalized off-policy sampling."""

    source = decision or row
    candidate_details = [
        {
            "option_number": int(candidate.get("option_number") or 0),
            "text": str(candidate.get("candidate_text") or ""),
            "selected": bool(int(candidate.get("selected") or 0)),
        }
        for candidate in candidates
    ]
    selected = [candidate for candidate in candidate_details if candidate["selected"]]
    alternatives = [
        candidate for candidate in candidate_details if not candidate["selected"]
    ]
    return {
        "branch_point_id": source.get("branch_point_id"),
        "candidate_pool_id": source.get("candidate_pool_id"),
        "node_id": source.get("node_id"),
        "candidate_count": int(source.get("candidate_count") or 0),
        "branch_fanout": int(source.get("branch_fanout") or len(selected)),
        "sampled_option_numbers": _parse_number_list(
            text=str(source.get("sampled_option_numbers") or "")
        ),
        "parse_status": source.get("parse_status") or "",
        "selected_candidates": selected,
        "other_candidates": alternatives,
        "candidates": candidate_details,
        "enumeration_exec_text": str(source.get("enumeration_exec_text") or ""),
    }


def _flagged_candidate_details(
    *,
    candidates: list[dict[str, Any]],
    candidate_tokens: dict[int, list[dict[str, Any]]],
    selected_ids: set[int],
    shortlist_ids: set[int],
) -> list[dict[str, Any]]:
    details = []
    for candidate in candidates:
        candidate_id = int(candidate.get("candidate_id") or 0)
        detail = _candidate_detail(
            row=candidate,
            tokens=candidate_tokens.get(candidate_id, []),
        )
        detail["selected"] = candidate_id in selected_ids
        detail["shortlisted"] = candidate_id in shortlist_ids
        details.append(detail)
    return details


def _cluster_groups_by_mode(
    *, cluster_rows: list[dict[str, Any]], selector_flags: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    selected_by_mode = _selected_ids_by_mode(flags=selector_flags)
    grouped: dict[str, dict[str, list[int]]] = {}
    for row in cluster_rows:
        mode_name = str(row.get("mode_name") or "")
        cluster_name = str(row.get("cluster_name") or "")
        candidate_id = int(row.get("candidate_id") or 0)
        grouped.setdefault(mode_name, {}).setdefault(cluster_name, []).append(
            candidate_id
        )
    payload: dict[str, list[dict[str, Any]]] = {}
    active_selected_ids = selected_by_mode.get("active", set())
    for mode_name, clusters in sorted(grouped.items()):
        selected_ids = selected_by_mode.get(mode_name, active_selected_ids)
        payload[mode_name] = [
            _cluster_group_payload(
                cluster_name=cluster_name,
                candidate_ids=candidate_ids,
                selected_ids=selected_ids,
            )
            for cluster_name, candidate_ids in sorted(clusters.items())
        ]
    return payload


def _selected_ids_by_mode(*, flags: list[dict[str, Any]]) -> dict[str, set[int]]:
    grouped: dict[str, set[int]] = {}
    active_ids: set[int] = set()
    for row in flags:
        if int(row.get("selected") or 0) != 1:
            continue
        candidate_id = int(row.get("candidate_id") or 0)
        mode_name = str(row.get("mode_name") or "")
        grouped.setdefault(mode_name, set()).add(candidate_id)
        if mode_name == "active":
            active_ids.add(candidate_id)
    for mode_name in list(grouped):
        if mode_name != "active":
            grouped[mode_name].update(active_ids)
    return grouped


def _parse_number_list(*, text: str) -> list[int]:
    """Parse a comma-separated integer list stored in typed SQLite columns."""

    values: list[int] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if item:
            values.append(int(item))
    return values


def _cluster_group_payload(
    *, cluster_name: str, candidate_ids: list[int], selected_ids: set[int]
) -> dict[str, Any]:
    sorted_ids = sorted(candidate_ids)
    selected = [
        candidate_id for candidate_id in sorted_ids if candidate_id in selected_ids
    ]
    return {
        "cluster_name": cluster_name,
        "candidate_ids": sorted_ids,
        "candidate_count": len(sorted_ids),
        "selected_candidate_ids": selected,
        "selected_candidate_count": len(selected),
    }


def _candidate_pool_details(
    *,
    row: dict[str, Any],
    candidates: list[dict[str, Any]],
    candidate_tokens: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "branch_point_id": row.get("branch_point_id"),
        "candidate_pool_id": "",
        "trigger_type": "",
        "num_candidates": len(candidates),
        "candidates": [
            _candidate_detail(
                row=candidate,
                tokens=candidate_tokens.get(
                    int(candidate.get("candidate_id") or 0), []
                ),
            )
            for candidate in candidates
        ],
    }


def _candidate_detail(
    *, row: dict[str, Any], tokens: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "candidate_id": row.get("candidate_id"),
        "text": row.get("text") or row.get("text_preview") or "",
        "text_preview": row.get("text_preview") or "",
        "output_token_count": row.get("output_token_count"),
        "finish_reason": row.get("finish_reason") or "",
        "stop_reason": row.get("stop_reason") or "",
        "tokens": [_token_detail(row=token) for token in tokens],
        "shortlisted": False,
        "selected": False,
    }


def _active_selector_mode(
    *, pool_row: dict[str, Any] | None, flags: list[dict[str, Any]]
) -> str:
    if pool_row and pool_row.get("active_selector_mode"):
        return str(pool_row["active_selector_mode"])
    modes = [str(row.get("mode_name") or "") for row in flags]
    for mode in modes:
        if mode and mode != "active":
            return mode
    return ""


def _selected_candidate_ids(*, flags: list[dict[str, Any]]) -> list[int]:
    selected = {
        int(row["candidate_id"]) for row in flags if int(row.get("selected") or 0) == 1
    }
    return sorted(selected)


def _shortlist_candidate_ids(
    *, flags: list[dict[str, Any]], active_mode: str
) -> list[int]:
    shortlisted = {
        int(row["candidate_id"])
        for row in flags
        if int(row.get("shortlisted") or 0) == 1
        and (not active_mode or str(row.get("mode_name") or "") == active_mode)
    }
    return sorted(shortlisted)


def _candidates_by_ids(
    *, candidates: list[dict[str, Any]], candidate_ids: list[int]
) -> list[dict[str, Any]]:
    by_id = {
        int(candidate.get("candidate_id") or 0): candidate for candidate in candidates
    }
    return [
        by_id[candidate_id] for candidate_id in candidate_ids if candidate_id in by_id
    ]


def _pool_value(*, pool_row: dict[str, Any] | None, key: str) -> Any:
    if not pool_row:
        return ""
    return pool_row.get(key) or ""


def _vllm_details(
    *,
    row: dict[str, Any],
    choices: list[dict[str, Any]],
    choice_tokens: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    return {
        "request_id": row.get("request_id"),
        "request_stream_id": row.get("request_stream_id"),
        "request_kind": row.get("request_kind"),
        "assistant_prefix_tail": row.get("assistant_prefix_tail") or "",
        "status": row.get("status"),
        "latency_seconds": row.get("latency_seconds"),
        "error_message": row.get("error_message"),
        "choice_count": row.get("choice_count") or len(choices),
        "choices": [
            _choice_detail(
                row=choice,
                tokens=choice_tokens.get(int(choice.get("choice_index") or 0), []),
            )
            for choice in choices
        ],
    }


def _choice_detail(
    *, row: dict[str, Any], tokens: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "index": row.get("choice_index"),
        "text": row.get("text") or row.get("text_preview") or "",
        "text_preview": row.get("text_preview") or "",
        "finish_reason": row.get("finish_reason") or "",
        "stop_reason": row.get("stop_reason") or "",
        "output_token_count": row.get("output_token_count"),
        "tokens": [_token_detail(row=token) for token in tokens],
    }


def _token_detail(*, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "token_index": row.get("token_index"),
        "token_id": row.get("token_id"),
        "token_text": row.get("token_text") or "",
        "selected_logprob": row.get("selected_logprob"),
        "selected_probability": row.get("selected_probability"),
        "top_logprob_alternatives": [],
    }


def _generated_chunk_details(
    *,
    row: dict[str, Any],
    chunk: dict[str, Any] | None,
    tokens: list[dict[str, Any]],
) -> dict[str, Any]:
    chunk = chunk or {}
    return {
        "generated_chunk_detail_hydrated": True,
        "chunk_text": chunk.get("chunk_text") or row.get("text_preview") or "",
        "token_count": chunk.get("token_count") or row.get("token_delta") or 0,
        "generated_tokens_before_chunk": chunk.get("generated_tokens_before_chunk"),
        "generated_tokens_after_chunk": chunk.get("generated_tokens_after_chunk"),
        "chunk_was_normalized": bool(chunk.get("chunk_was_normalized") or False),
        "chunk_token_ids_source": chunk.get("chunk_token_ids_source") or "",
        "source": chunk.get("source") or "",
        "tokens": [_token_detail(row=token) for token in tokens],
    }


def _leaf_details_from_event(
    *,
    row: dict[str, Any],
    hydrated: bool = False,
) -> dict[str, Any]:
    text = str(row.get("text") or row.get("text_preview") or "")
    task_metrics = _leaf_status_metrics(row=row)
    return {
        "leaf_id": row.get("leaf_id"),
        "verification": row.get("verification"),
        "length_tokens_total": row.get("length_tokens_total"),
        "stop_reason": row.get("stop_reason") or "",
        "raw_answer_acc": task_metrics.get("raw_answer_acc"),
        "format_valid": task_metrics.get("format_valid"),
        "answer_acc": task_metrics.get("answer_acc"),
        "boxed_answer": task_metrics.get("boxed_answer"),
        "structure_issues": task_metrics.get("structure_issues"),
        "task_metrics": task_metrics,
        "text": text,
        "text_preview": row.get("text_preview") or "",
        "leaf_detail_hydrated": hydrated,
    }


def _leaf_payload_rows(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_leaf_details_from_event(row=row) for row in rows]


def _leaf_status_metrics(*, row: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in (
        "raw_answer_acc",
        "format_valid",
        "answer_acc",
        "boxed_answer",
        "structure_issues",
    ):
        value = row.get(key)
        if value is not None and str(value) != "":
            metrics[key] = value
    return metrics


def _baseline_leaf_row_for_display_node(
    *, db: EventDatabase, key: AttemptKey, node_id: str
) -> dict[str, Any] | None:
    leaf_id = _baseline_leaf_id_for_display_node(node_id=node_id)
    if leaf_id is None:
        return None
    for row in db.read_leaf_rows_for_attempt(**_key_kwargs(key=key)):
        if str(row.get("leaf_id") or "") == leaf_id:
            return _leaf_row_with_key(row=row, key=key)
    return None


def _baseline_leaf_id_for_display_node(*, node_id: str) -> str | None:
    if not node_id.startswith(BASELINE_LEAF_NODE_PREFIX):
        return None
    suffix = node_id.removeprefix(BASELINE_LEAF_NODE_PREFIX)
    if suffix.isdecimal():
        return f"leaf_baseline_{suffix}"
    if suffix.startswith("leaf_"):
        return suffix
    return None


def _leaf_row_with_key(*, row: dict[str, Any], key: AttemptKey) -> dict[str, Any]:
    return {
        **row,
        "doc_id": key.doc_id,
        "doc_attempt": key.doc_attempt,
        "task_name": key.task_name,
        "model_id": key.model_id,
        "selector_mode": key.selector_mode,
    }


def _baseline_leaf_node_payload(
    *, db: EventDatabase, key: AttemptKey, leaf_row: dict[str, Any]
) -> dict[str, Any]:
    leaf_id = str(leaf_row.get("leaf_id") or "")
    node_id = _baseline_leaf_node_id(leaf_id=leaf_id)
    parent_node_id = str(leaf_row.get("node_id") or "node_root")
    event_row = db.read_node_event_row_by_index(
        event_index=int(leaf_row.get("event_index") or -1)
    )
    event_rows = []
    if event_row is not None:
        event_rows = [{**event_row, "node_id": node_id}]
    node = {
        **_synthetic_node(node_id=node_id, parent_id=parent_node_id),
        "candidate_preview": _baseline_leaf_label(row=leaf_row),
        "leaf_count": 1,
        "token_total": int(leaf_row.get("length_tokens_total") or 0),
        "first_event_index": int(leaf_row.get("event_index") or -1),
    }
    events = _light_event_rows(rows=event_rows)
    node["event_count"] = len(events)
    keyed_leaf = _leaf_row_with_key(row=leaf_row, key=key)
    return {
        "node": node,
        "events": events,
        "leaves": [
            _leaf_details_from_event(
                row=keyed_leaf,
                hydrated=True,
            )
        ],
        "trajectory": {
            "text": "",
            "token_count": 0,
            "segment_count": 0,
            "segments": [],
        },
    }


def _baseline_leaf_text_payload(
    *, db: EventDatabase, key: AttemptKey, leaf_row: dict[str, Any]
) -> dict[str, Any]:
    _ = db, key
    leaf_id = str(leaf_row.get("leaf_id") or "")
    return {
        "node_id": _baseline_leaf_node_id(leaf_id=leaf_id),
        "token_count": int(leaf_row.get("length_tokens_total") or 0),
        "text": str(leaf_row.get("text") or ""),
        "tokens": [],
    }


def _baseline_rollout_index(*, leaf_id: str) -> int | None:
    prefix = "leaf_baseline_"
    if not leaf_id.startswith(prefix):
        return None
    raw_index = leaf_id.removeprefix(prefix)
    return int(raw_index) if raw_index.isdecimal() else None


def _trajectory_payload(
    *,
    edge_rows: list[dict[str, Any]],
    text_rows: list[dict[str, Any]],
    leaf_rows: list[dict[str, Any]],
    leaf_segments: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    segments = [_trajectory_segment(row=row) for row in edge_rows]
    if not segments:
        segments = _vllm_text_segments(rows=text_rows)
    if leaf_rows:
        segments.extend(
            leaf_segments
            if leaf_segments is not None
            else _leaf_trajectory_segments(rows=leaf_rows)
        )
    return {
        "text": "".join(str(segment["text"]) for segment in segments),
        "token_count": sum(int(segment["token_count"] or 0) for segment in segments),
        "segment_count": len(segments),
        "segments": segments,
    }


def _vllm_text_segments(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    return [
        {
            "event_index": int(rows[0].get("event_index") or -1),
            "parent_node_id": None,
            "child_node_id": None,
            "candidate_id": None,
            "selector_mode": "",
            "text": "".join(str(row.get("text") or "") for row in rows),
            "token_count": sum(int(row.get("output_token_count") or 0) for row in rows),
        }
    ]


def _trajectory_segment(*, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_index": int(row.get("event_index") or -1),
        "parent_node_id": row.get("parent_node_id"),
        "child_node_id": row.get("child_node_id"),
        "candidate_id": row.get("candidate_id"),
        "selector_mode": row.get("selector_mode") or "",
        "text": row.get("candidate_text") or "",
        "token_count": int(row.get("candidate_token_count") or 0),
    }


def _leaf_trajectory_segments(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    row = max(rows, key=lambda item: int(item.get("event_index") or -1))
    return [_leaf_trajectory_segment(row=row, text=str(row.get("text") or ""))]


def _leaf_trajectory_segment(
    *, row: dict[str, Any], text: str, token_count: Any | None = None
) -> dict[str, Any]:
    segment_text = text or str(row.get("text_preview") or "")
    return {
        "event_index": int(row.get("event_index") or -1),
        "parent_node_id": None,
        "child_node_id": row.get("node_id"),
        "candidate_id": None,
        "selector_mode": "",
        "text": segment_text,
        "token_count": int(token_count or row.get("length_tokens_total") or 0),
    }


def _node_payload_rows(
    *,
    nodes: dict[str, dict[str, Any]],
    depth_by_node: dict[str, int],
    metrics_by_node: dict[str, dict[str, float]],
    node_events: dict[str, list[dict[str, Any]]],
    event_count_by_node: dict[str, int],
    advantage_by_node: dict[str, dict[str, Any]],
    detail_base_url: str,
) -> list[dict[str, Any]]:
    payload_rows = []
    for node_id, node in sorted(nodes.items()):
        row = {
            "node_id": node_id,
            "parent_node_id": node.get("parent_node_id"),
            "depth": depth_by_node.get(node_id, 0),
            "branch_points_used": int(node.get("branch_points_used") or 0),
            "candidate_preview": _preview(node=node),
            "event_count": event_count_by_node.get(node_id, 0),
            "leaf_count": int(node.get("leaf_count") or 0),
            "detail_path": f"{detail_base_url}/nodes/{node_id}.json",
            "metrics": metrics_by_node.get(node_id, {}),
        }
        row.update(_node_advantage_payload(row=advantage_by_node.get(node_id)))
        payload_rows.append(row)
    return payload_rows


def _node_advantage_payload(*, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    advantage = float(row.get("mean_combined_advantage") or 0.0)
    return {
        "segment_advantage": advantage,
        "advantage_label": _format_advantage_label(value=advantage),
        "advantage_token_count": int(row.get("token_count") or 0),
        "advantage_leaf_count": int(row.get("leaf_count") or 0),
        "advantage_token_start": int(row.get("token_start") or 0),
        "advantage_token_end": int(row.get("token_end") or 0),
    }


def _format_advantage_label(*, value: float) -> str:
    if not math.isfinite(value):
        return ""
    return f"{value:+.1f}" if abs(value) >= 10.0 else f"{value:+.2f}"


def _preview(*, node: dict[str, Any]) -> str:
    text = str(node.get("candidate_preview") or "")
    return clean_candidate_preview(text=text, max_chars=72) if text else "Root"


def _edge_payload_rows(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "parent_node_id": row["parent_node_id"],
            "child_node_id": row["child_node_id"],
            "candidate_id": row["candidate_id"],
            "selector_mode": row["selector_mode"],
        }
        for row in rows
    ]


def _synthetic_edge_payload_rows(
    *, nodes: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    return [
        {
            "parent_node_id": str(node["parent_node_id"]),
            "child_node_id": node_id,
            "candidate_id": None,
            "selector_mode": "",
        }
        for node_id, node in sorted(nodes.items())
        if node.get("parent_node_id")
    ]


def _tree_meta(
    *,
    key: AttemptKey,
    summary_row: dict[str, Any],
    diagnostics_row: dict[str, Any] | None,
    nodes: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    node_events: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    x_max = {"steps": 0.0, "tokens": 0.0, "time_seconds": 0.0}
    for row in nodes:
        for metric, value in dict(row.get("metrics") or {}).items():
            x_max[metric] = max(float(x_max.get(metric, 0.0)), float(value or 0.0))
    return {
        "status": _status_from_row(row=summary_row),
        "started_at": "",
        "node_count": len(nodes),
        "edge_count": len(edges),
        "leaf_count": sum(int(row.get("leaf_count") or 0) for row in nodes),
        "event_count": int(summary_row.get("event_count", 0)),
        "last_event_index": int(summary_row.get("last_event_index", -1)),
        "resumed_reason": "",
        "branch_count": 0,
        "x_max": x_max,
        "attempt_slug": key.slug(),
        "is_rl_run": _is_rl_attempt_key(key=key),
        "diagnostics": _doc_diagnostics_payload(row=diagnostics_row),
    }


def _doc_diagnostics_payload(*, row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {}
    payload_json = row.get("payload_json")
    assert isinstance(payload_json, str), "doc diagnostics payload must be JSON text"
    payload = json.loads(payload_json)
    assert isinstance(payload, dict), "doc diagnostics payload must be an object"
    return {
        "event_index": int(row["event_index"]),
        "timestamp_utc": str(row["timestamp_utc"]),
        **payload,
    }


def _advantage_by_child_node(
    *, rows: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    return {str(row["child_node_id"]): row for row in rows if row.get("child_node_id")}


def _is_rl_attempt_key(*, key: AttemptKey) -> bool:
    return (
        key.task_name.startswith("branching_dapo") or key.model_id == "branching_dapo"
    )


def _summary_row(*, db: EventDatabase, key: AttemptKey) -> dict[str, Any]:
    return db.read_attempt_summary_row(**_key_kwargs(key=key)) or {}


def _selector_event_indexes(*, rows: list[dict[str, Any]]) -> list[int]:
    return [
        int(row["event_index"])
        for row in rows
        if str(row.get("event_type") or "")
        in {"selector_applied", "selector_continued_inline"}
    ]


def _selector_pool_rows_by_event(
    *, rows: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in rows:
        selector_event_index = int(row["selector_event_index"])
        grouped.setdefault(selector_event_index, row)
    return grouped


def _pool_event_indexes(*, rows: Iterable[dict[str, Any]]) -> list[int]:
    return [int(row["pool_event_index"]) for row in rows]


def _candidate_event_index(
    *, event_index: int, selector_pool_row: dict[str, Any] | None
) -> int:
    if selector_pool_row is None:
        return event_index
    return int(selector_pool_row["pool_event_index"])


def _rows_by_selector_event(
    *, rows: list[dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        event_index = int(row["selector_event_index"])
        grouped.setdefault(event_index, []).append(row)
    return grouped


def _rows_by_event(*, rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        raw_index = (
            row.get("pool_event_index")
            or row.get("response_event_index")
            or row.get("selector_event_index")
            or row.get("decision_event_index")
            or row.get("event_index")
        )
        assert raw_index is not None, "detail rows must include an event index"
        event_index = int(raw_index)
        grouped.setdefault(event_index, []).append(row)
    return grouped


def _token_rows_by_event_choice(
    *, rows: list[dict[str, Any]]
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        event_index = int(row["response_event_index"])
        choice_index = int(row["choice_index"])
        grouped.setdefault(event_index, {}).setdefault(choice_index, []).append(row)
    return grouped


def _candidate_token_rows_by_event_candidate(
    *, rows: list[dict[str, Any]]
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    grouped: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for row in rows:
        event_index = int(row["pool_event_index"])
        candidate_id = int(row["candidate_id"])
        grouped.setdefault(event_index, {}).setdefault(candidate_id, []).append(row)
    return grouped


def _chunk_token_rows(
    *,
    row: dict[str, Any],
    chunk: dict[str, Any] | None,
    latest_choice_tokens_by_node: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if str(row["event_type"]) not in CHUNK_DETAIL_EVENT_TYPES:
        return []
    token_count = int((chunk or {}).get("token_count") or row.get("token_delta") or 0)
    if token_count <= 0:
        return []
    node_id = str(row["node_id"])
    return latest_choice_tokens_by_node.get(node_id, [])[:token_count]


def first_or_none(*, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    return rows[0] if rows else None


def _sorted_event_rows(*, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (int(row.get("event_index") or -1), str(row["event_type"])),
    )


def _collapse_selector_decision_rows(
    *, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    sorted_rows = _sorted_event_rows(rows=rows)
    selected_branch_points = {
        str(row.get("branch_point_id") or "")
        for row in sorted_rows
        if str(row.get("event_type") or "") == "selector_applied"
    }
    if not selected_branch_points:
        return sorted_rows
    return [
        _selector_decision_row(row=row)
        for row in sorted_rows
        if not _is_selector_decision_companion(
            row=row, branch_points=selected_branch_points
        )
    ]


def _is_selector_decision_companion(
    *, row: dict[str, Any], branch_points: set[str]
) -> bool:
    return (
        str(row.get("event_type") or "") in SELECTOR_DECISION_COMPANION_TYPES
        and str(row.get("branch_point_id") or "") in branch_points
    )


def _selector_decision_row(*, row: dict[str, Any]) -> dict[str, Any]:
    if str(row.get("event_type") or "") != "selector_applied":
        return row
    return {**row, "summary": "selector decision"}


def _event_indexes(rows: list[dict[str, Any]]) -> list[int]:
    return [int(row["event_index"]) for row in rows]


def _key_kwargs(*, key: AttemptKey) -> dict[str, Any]:
    return {
        "doc_id": key.doc_id,
        "doc_attempt": key.doc_attempt,
        "task_name": key.task_name,
        "model_id": key.model_id,
        "selector_mode": key.selector_mode,
    }


def _status_from_row(*, row: dict[str, Any]) -> str:
    if int(row.get("finished_count", 0) or 0) > 0:
        return "completed"
    if int(row.get("started_count", 0) or 0) > 0:
        return "incomplete"
    return "empty"
