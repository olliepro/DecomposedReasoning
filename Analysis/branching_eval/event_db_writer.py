"""Typed-row writer helpers for the branching event SQLite store."""

from __future__ import annotations

import math
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from branching_eval import event_db_sql

GRAPH_EVENT_TYPES = {
    "prompt_logged",
    "trigger_fired",
    "trigger_skipped_max_branch_points",
    "candidate_pool_resolved",
    "selector_applied",
    "selector_continued_inline",
    "verbalized_sampling_applied",
    "malformed_steer_decision",
    "repeat_forced_think_close",
    "leaf_completed",
    "leaf_scored",
}


@dataclass
class NormalizedEventBatch:
    """Typed rows derived from canonical event payloads."""

    nodes: list[tuple[Any, ...]] = field(default_factory=list)
    edges: list[tuple[Any, ...]] = field(default_factory=list)
    leaves: list[tuple[Any, ...]] = field(default_factory=list)
    leaf_metrics: list[tuple[Any, ...]] = field(default_factory=list)
    node_events: list[tuple[Any, ...]] = field(default_factory=list)
    prompt_contexts: list[tuple[Any, ...]] = field(default_factory=list)
    candidate_pools: list[tuple[Any, ...]] = field(default_factory=list)
    candidates: list[tuple[Any, ...]] = field(default_factory=list)
    candidate_tokens: list[tuple[Any, ...]] = field(default_factory=list)
    selectors: list[tuple[Any, ...]] = field(default_factory=list)
    selector_flags: list[tuple[Any, ...]] = field(default_factory=list)
    selector_clusters: list[tuple[Any, ...]] = field(default_factory=list)
    verbalized_sampling_decisions: list[tuple[Any, ...]] = field(default_factory=list)
    verbalized_sampling_candidates: list[tuple[Any, ...]] = field(default_factory=list)
    vllm_requests: list[tuple[Any, ...]] = field(default_factory=list)
    vllm_responses: list[tuple[Any, ...]] = field(default_factory=list)
    vllm_choices: list[tuple[Any, ...]] = field(default_factory=list)
    vllm_choice_tokens: list[tuple[Any, ...]] = field(default_factory=list)
    generated_chunks: list[tuple[Any, ...]] = field(default_factory=list)


def append_normalized_rows(
    *, connection: sqlite3.Connection, rows: Sequence[dict[str, Any]]
) -> None:
    batch = _normalized_batch(rows=rows)
    _executemany(
        connection=connection, sql=event_db_sql.INSERT_NODE_SQL, rows=batch.nodes
    )
    _executemany(
        connection=connection, sql=event_db_sql.INSERT_EDGE_SQL, rows=batch.edges
    )
    _executemany(
        connection=connection, sql=event_db_sql.INSERT_LEAF_SQL, rows=batch.leaves
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_LEAF_METRIC_SQL,
        rows=batch.leaf_metrics,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_NODE_EVENT_SQL,
        rows=batch.node_events,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_PROMPT_CONTEXT_SQL,
        rows=batch.prompt_contexts,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_CANDIDATE_POOL_SQL,
        rows=batch.candidate_pools,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_CANDIDATE_SQL,
        rows=batch.candidates,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_CANDIDATE_TOKEN_SQL,
        rows=batch.candidate_tokens,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_SELECTOR_SQL,
        rows=batch.selectors,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_SELECTOR_FLAG_SQL,
        rows=batch.selector_flags,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_SELECTOR_CLUSTER_SQL,
        rows=batch.selector_clusters,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_VERBALIZED_SAMPLING_DECISION_SQL,
        rows=batch.verbalized_sampling_decisions,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_VERBALIZED_SAMPLING_CANDIDATE_SQL,
        rows=batch.verbalized_sampling_candidates,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_VLLM_REQUEST_SQL,
        rows=batch.vllm_requests,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_VLLM_RESPONSE_SQL,
        rows=batch.vllm_responses,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_VLLM_CHOICE_SQL,
        rows=batch.vllm_choices,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_VLLM_CHOICE_TOKEN_SQL,
        rows=batch.vllm_choice_tokens,
    )
    _executemany(
        connection=connection,
        sql=event_db_sql.INSERT_GENERATED_CHUNK_SQL,
        rows=batch.generated_chunks,
    )


def _normalized_batch(*, rows: Sequence[dict[str, Any]]) -> NormalizedEventBatch:
    batch = NormalizedEventBatch()
    for row in rows:
        _add_normalized_row(batch=batch, row=row)
    return batch


def _add_normalized_row(*, batch: NormalizedEventBatch, row: dict[str, Any]) -> None:
    payload = row.get("payload", {})
    assert isinstance(payload, dict), "event payload must be a mapping"
    base = _base_values(row=row)
    event_type = str(row["event_type"])
    _add_node_event(batch=batch, base=base, event_type=event_type, payload=payload)
    if event_type == "node_created":
        batch.nodes.append(_node_values(base=base, payload=payload))
    elif event_type == "edge_selected":
        batch.edges.append(_edge_values(base=base, payload=payload))
    elif event_type == "leaf_scored":
        _add_leaf_rows(batch=batch, base=base, payload=payload)
    elif event_type == "candidate_pool_resolved":
        _add_candidate_pool_rows(batch=batch, base=base, payload=payload)
    elif event_type == "prompt_logged":
        batch.prompt_contexts.append(_prompt_context_values(base=base, payload=payload))
    elif event_type in {"selector_applied", "selector_continued_inline"}:
        _add_selector_rows(batch=batch, base=base, payload=payload)
    elif event_type == "verbalized_sampling_applied":
        _add_verbalized_sampling_rows(batch=batch, base=base, payload=payload)
    elif event_type == "vllm_request":
        batch.vllm_requests.append(_vllm_request_values(base=base, payload=payload))
    elif event_type == "vllm_response":
        _add_vllm_response_rows(batch=batch, base=base, payload=payload)
    elif event_type in {
        "decode_chunk",
        "steer_block_generated",
        "repeat_forced_think_close",
    }:
        batch.generated_chunks.append(
            _generated_chunk_values(
                base=base,
                event_type=event_type,
                payload=payload,
            )
        )


def _base_values(*, row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["event_index"]),
        int(row["doc_id"]) if row.get("doc_id") is not None else None,
        int(row["doc_attempt"]) if row.get("doc_attempt") is not None else None,
        str(row["task_name"]),
        str(row["model_id"]),
        str(row["selector_mode"]),
        str(row["timestamp_utc"]),
    )


def _node_values(*, base: tuple[Any, ...], payload: dict[str, Any]) -> tuple[Any, ...]:
    return (
        *base,
        str(payload.get("node_id", "")),
        _optional_text(payload.get("parent_node_id")),
        int(payload.get("branch_points_used", 0) or 0),
    )


def _edge_values(*, base: tuple[Any, ...], payload: dict[str, Any]) -> tuple[Any, ...]:
    token_ids = payload.get("candidate_token_ids_normalized", [])
    return (
        *base,
        str(payload.get("parent_node_id", "")),
        str(payload.get("child_node_id", "")),
        _optional_int(payload.get("candidate_id")),
        str(payload.get("selector_mode", "")),
        str(payload.get("candidate_text_normalized", "")),
        len(token_ids) if isinstance(token_ids, list) else 0,
    )


def _add_leaf_rows(
    *, batch: NormalizedEventBatch, base: tuple[Any, ...], payload: dict[str, Any]
) -> None:
    batch.leaves.append(
        (
            *base,
            str(payload.get("leaf_id", "")),
            str(payload.get("node_id", "")),
            _optional_int(payload.get("verification")),
            _optional_int(payload.get("length_tokens_total")),
            _optional_int(payload.get("length_tokens_exec")),
            str(payload.get("stop_reason", "")),
            str(payload.get("text", payload.get("text_preview", ""))),
            str(payload.get("text_preview", payload.get("text", ""))),
        )
    )
    for name, value in _metric_items(value=payload.get("task_metrics", {})):
        batch.leaf_metrics.append((base[0], name, value[0], value[1]))


def _add_candidate_pool_rows(
    *, batch: NormalizedEventBatch, base: tuple[Any, ...], payload: dict[str, Any]
) -> None:
    batch.candidate_pools.append(
        (
            *base,
            str(payload.get("branch_point_id", "")),
            str(payload.get("candidate_pool_id", "")),
            str(payload.get("node_id", "")),
            str(payload.get("trigger_type", "")),
            int(payload.get("num_candidates", 0) or 0),
        )
    )
    for candidate in _dict_rows(value=payload.get("candidates", [])):
        event_index = int(base[0])
        candidate_id = int(candidate.get("candidate_id", -1))
        batch.candidates.append(
            _candidate_values(event_index=event_index, row=candidate)
        )
        for token in _dict_rows(value=candidate.get("tokens", [])):
            batch.candidate_tokens.append(
                _token_values(
                    owner_event_index=event_index,
                    owner_index=candidate_id,
                    row=token,
                )
            )


def _add_selector_rows(
    *, batch: NormalizedEventBatch, base: tuple[Any, ...], payload: dict[str, Any]
) -> None:
    event_type = _event_type_from_base(base=base, payload=payload)
    active_mode = str(payload.get("active_selector_mode", ""))
    selected_ids = _int_list(payload.get("selected_candidate_ids", []))
    if event_type == "selector_continued_inline":
        selected_ids = [_optional_int(payload.get("selected_candidate_id")) or -1]
    batch.selectors.append(
        (
            *base,
            event_type,
            str(payload.get("branch_point_id", "")),
            str(payload.get("node_id", "")),
            active_mode,
        )
    )
    _add_selector_flag_rows(
        batch=batch, base=base, payload=payload, selected_ids=selected_ids
    )
    _add_selector_cluster_rows(batch=batch, base=base, payload=payload)


def _event_type_from_base(*, base: tuple[Any, ...], payload: dict[str, Any]) -> str:
    _ = base, payload
    return (
        "selector_continued_inline"
        if "selected_candidate_id" in payload
        else "selector_applied"
    )


def _add_selector_flag_rows(
    *,
    batch: NormalizedEventBatch,
    base: tuple[Any, ...],
    payload: dict[str, Any],
    selected_ids: list[int],
) -> None:
    for candidate_id in selected_ids:
        if candidate_id >= 0:
            batch.selector_flags.append((base[0], "active", candidate_id, 1, 0))
    for mode, ids in _mode_ids(value=payload.get("selected_by_mode", {})):
        for candidate_id in ids:
            batch.selector_flags.append((base[0], mode, candidate_id, 1, 0))
    for mode, ids in _mode_ids(value=payload.get("shortlist_by_mode", {})):
        for candidate_id in ids:
            batch.selector_flags.append((base[0], mode, candidate_id, 0, 1))


def _add_selector_cluster_rows(
    *, batch: NormalizedEventBatch, base: tuple[Any, ...], payload: dict[str, Any]
) -> None:
    for mode, assignments in _cluster_assignment_items(
        value=payload.get("cluster_assignments_by_mode", {})
    ):
        for candidate_id, cluster_name in assignments:
            batch.selector_clusters.append((base[0], mode, candidate_id, cluster_name))


def _add_verbalized_sampling_rows(
    *, batch: NormalizedEventBatch, base: tuple[Any, ...], payload: dict[str, Any]
) -> None:
    sampled_numbers = ",".join(
        str(value) for value in _int_list(payload.get("sampled_option_numbers", []))
    )
    batch.verbalized_sampling_decisions.append(
        (
            *base,
            str(payload.get("branch_point_id", "")),
            str(payload.get("candidate_pool_id", "")),
            str(payload.get("node_id", "")),
            int(payload.get("candidate_count", 0) or 0),
            int(payload.get("branch_fanout", 0) or 0),
            sampled_numbers,
            str(payload.get("parse_status", "")),
            str(payload.get("enumeration_exec_text", "")),
        )
    )
    selected_numbers = set(_int_list(payload.get("sampled_option_numbers", [])))
    for candidate_rank, candidate in enumerate(
        _dict_rows(value=payload.get("candidates", []))
    ):
        option_number = int(candidate.get("option_number", 0) or 0)
        batch.verbalized_sampling_candidates.append(
            (
                base[0],
                option_number,
                candidate_rank,
                str(candidate.get("text", "")),
                1 if option_number in selected_numbers else 0,
            )
        )


def _vllm_request_values(
    *, base: tuple[Any, ...], payload: dict[str, Any]
) -> tuple[Any, ...]:
    return (
        *base,
        str(payload.get("request_id", "")),
        str(payload.get("request_stream_id", "")),
        str(payload.get("request_kind", "")),
        _optional_text(payload.get("prev_request_id")),
        _optional_int(payload.get("current_input_token_count")),
        _optional_int(payload.get("base_prefix_token_count")),
        _optional_int(payload.get("delta_token_count")),
        _optional_int(payload.get("assistant_prefix_char_count")),
        str(payload.get("assistant_prefix_tail", "")),
    )


def _add_vllm_response_rows(
    *, batch: NormalizedEventBatch, base: tuple[Any, ...], payload: dict[str, Any]
) -> None:
    batch.vllm_responses.append(
        (
            *base,
            str(payload.get("request_id", "")),
            str(payload.get("request_stream_id", "")),
            str(payload.get("request_kind", "")),
            str(payload.get("status", "")),
            _optional_float(payload.get("latency_seconds")),
            str(payload.get("error_message", "")),
            int(payload.get("choice_count", 0) or 0),
        )
    )
    for choice in _dict_rows(value=payload.get("choices", [])):
        event_index = int(base[0])
        choice_index = int(choice.get("index", 0) or 0)
        batch.vllm_choices.append(
            _vllm_choice_values(event_index=event_index, row=choice)
        )
        for token in _dict_rows(value=choice.get("tokens", [])):
            batch.vllm_choice_tokens.append(
                _token_values(
                    owner_event_index=event_index,
                    owner_index=choice_index,
                    row=token,
                )
            )


def _add_node_event(
    *,
    batch: NormalizedEventBatch,
    base: tuple[Any, ...],
    event_type: str,
    payload: dict[str, Any],
) -> None:
    node_id = _node_event_id(event_type=event_type, payload=payload)
    if node_id is None:
        return
    batch.node_events.append(
        (
            *base,
            _node_event_type(event_type=event_type),
            node_id,
            _event_summary(event_type=event_type, payload=payload),
            1,
            _token_delta(event_type=event_type, payload=payload),
            str(payload.get("branch_point_id", "")),
            str(payload.get("leaf_id", "")),
            _optional_int(payload.get("verification")),
            str(payload.get("stop_reason", "")),
            _optional_int(payload.get("length_tokens_total")),
            str(payload.get("request_id", "")),
            str(payload.get("request_stream_id", "")),
            str(payload.get("request_kind", "")),
            str(payload.get("status", "")),
            _optional_float(payload.get("latency_seconds")),
            str(payload.get("error_message", "")),
            _optional_int(payload.get("choice_count")),
            _first_choice_output_count(payload=payload),
            _node_event_text_preview(event_type=event_type, payload=payload),
            _node_event_text(event_type=event_type, payload=payload),
        )
    )


def _candidate_values(*, event_index: int, row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event_index,
        int(row.get("candidate_id", -1)),
        str(row.get("text", row.get("text_preview", ""))),
        str(row.get("text_preview", row.get("text", ""))),
        _optional_int(row.get("output_token_count")),
        str(row.get("finish_reason", "")),
        str(row.get("stop_reason", "")),
    )


def _prompt_context_values(
    *, base: tuple[Any, ...], payload: dict[str, Any]
) -> tuple[Any, ...]:
    return (
        *base,
        str(payload.get("node_id", "node_root")),
        str(payload.get("prompt_text", "")),
        _optional_int(payload.get("prompt_char_count")),
        str(payload.get("golden_answer", "")),
        str(payload.get("golden_answer_source", "")),
    )


def _vllm_choice_values(*, event_index: int, row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        event_index,
        int(row.get("index", 0) or 0),
        str(row.get("text", row.get("text_preview", ""))),
        str(row.get("text_preview", row.get("text", ""))),
        str(row.get("finish_reason", "")),
        str(row.get("stop_reason", "")),
        _optional_int(row.get("output_token_count")),
    )


def _token_values(
    *, owner_event_index: int, owner_index: int, row: dict[str, Any]
) -> tuple[Any, ...]:
    token_id = row.get("token_id")
    raw_logprob = row.get("selected_logprob")
    selected_logprob = None if raw_logprob is None else float(raw_logprob)
    raw_probability = row.get("selected_probability")
    if raw_probability is not None:
        selected_probability = float(raw_probability)
    elif selected_logprob is None:
        selected_probability = None
    else:
        selected_probability = math.exp(selected_logprob)
    return (
        owner_event_index,
        owner_index,
        int(row.get("token_index", 0) or 0),
        None if token_id is None else int(token_id),
        str(row.get("token_text", "")),
        selected_logprob,
        selected_probability,
    )


def _generated_chunk_values(
    *, base: tuple[Any, ...], event_type: str, payload: dict[str, Any]
) -> tuple[Any, ...]:
    chunk_text = str(payload.get("chunk_text", payload.get("forced_close_text", "")))
    token_ids = payload.get(
        "chunk_token_ids", payload.get("forced_close_token_ids", [])
    )
    return (
        *base,
        event_type,
        str(payload.get("node_id", "")),
        chunk_text,
        len(token_ids) if isinstance(token_ids, list) else 0,
        _optional_int(payload.get("generated_tokens_before_chunk")),
        _optional_int(payload.get("generated_tokens_after_chunk")),
        1 if bool(payload.get("chunk_was_normalized", False)) else 0,
        str(payload.get("chunk_token_ids_source", "")),
        str(payload.get("source", "")),
    )


def _node_event_id(*, event_type: str, payload: dict[str, Any]) -> str | None:
    if event_type in GRAPH_EVENT_TYPES:
        node_id = str(payload.get("node_id", ""))
        return node_id if node_id else None
    return None


def _node_event_type(*, event_type: str) -> str:
    return (
        "vllm_step" if event_type in {"vllm_request", "vllm_response"} else event_type
    )


def _event_summary(*, event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "prompt_logged":
        answer = str(payload.get("golden_answer", "")).strip()
        return f"input prompt · gold {answer}" if answer else "input prompt"
    if event_type == "trigger_fired":
        return f"trigger {payload.get('trigger_type', '')}"
    if event_type == "trigger_skipped_max_branch_points":
        return "trigger skipped max branch points"
    if event_type == "candidate_pool_resolved":
        return f"candidate pool n={int(payload.get('num_candidates', 0) or 0)}"
    if event_type == "selector_applied":
        return (
            f"selector kept {len(_int_list(payload.get('selected_candidate_ids', [])))}"
        )
    if event_type == "selector_continued_inline":
        return f"selector continued inline {payload.get('selected_candidate_id')}"
    if event_type == "verbalized_sampling_applied":
        selected = _int_list(payload.get("sampled_option_numbers", []))
        return (
            f"verbalized sampling selected {','.join(str(value) for value in selected)}"
        )
    if event_type == "doc_diagnostics_recorded":
        return "doc diagnostics recorded"
    if event_type == "malformed_steer_decision":
        return (
            "malformed steer decision"
            f" {payload.get('source', '')}"
            f" stop={payload.get('stop_reason', '')}"
        )
    if event_type == "repeat_forced_think_close":
        return (
            "repeat forced think close"
            f" {payload.get('repeat_block_kind', '')}"
            f" x{payload.get('repeat_block_count', '')}"
        )
    if event_type == "leaf_completed":
        return f"leaf completed {payload.get('leaf_id')}"
    if event_type == "leaf_scored":
        return f"leaf scored verify={payload.get('verification')}"
    if event_type in {
        "decode_chunk",
        "steer_block_generated",
        "repeat_forced_think_close",
    }:
        return (
            f"{event_type} +{_token_delta(event_type=event_type, payload=payload)} tok"
        )
    if event_type == "vllm_response":
        return f"{payload.get('request_kind', '')} ({payload.get('status', '')})"
    if event_type == "vllm_request":
        return f"{payload.get('request_kind', '')} request"
    return event_type


def _node_event_text_preview(*, event_type: str, payload: dict[str, Any]) -> str:
    if event_type == "prompt_logged":
        return str(payload.get("golden_answer", ""))
    if event_type in {
        "decode_chunk",
        "steer_block_generated",
        "repeat_forced_think_close",
    }:
        return str(payload.get("chunk_text", payload.get("forced_close_text", "")))
    if event_type == "malformed_steer_decision":
        return str(
            payload.get("candidate_text", payload.get("assistant_prefix_tail", ""))
        )
    if event_type == "verbalized_sampling_applied":
        return str(payload.get("enumeration_exec_text", ""))[:500]
    return str(payload.get("text_preview", ""))


def _node_event_text(*, event_type: str, payload: dict[str, Any]) -> str:
    if event_type in {"leaf_completed", "leaf_scored"}:
        return str(payload.get("text", payload.get("text_preview", "")))
    if event_type in {
        "decode_chunk",
        "steer_block_generated",
        "repeat_forced_think_close",
    }:
        return str(payload.get("chunk_text", payload.get("forced_close_text", "")))
    if event_type == "malformed_steer_decision":
        return str(
            payload.get("candidate_text", payload.get("assistant_prefix_tail", ""))
        )
    if event_type == "verbalized_sampling_applied":
        return str(payload.get("enumeration_exec_text", ""))
    return ""


def _token_delta(*, event_type: str, payload: dict[str, Any]) -> int:
    if event_type in {
        "decode_chunk",
        "steer_block_generated",
        "repeat_forced_think_close",
    }:
        token_ids = payload.get(
            "chunk_token_ids", payload.get("forced_close_token_ids", [])
        )
        return len(token_ids) if isinstance(token_ids, list) else 0
    if event_type == "vllm_response" and payload.get("status") == "ok":
        return _first_choice_output_count(payload=payload)
    return 0


def _first_choice_output_count(*, payload: dict[str, Any]) -> int:
    choices = payload.get("choices", [])
    if not isinstance(choices, list) or not choices:
        return 0
    first = choices[0]
    if not isinstance(first, dict):
        return 0
    return int(first.get("output_token_count", 0) or 0)


def _decode_node_id(request_stream_id: str) -> str | None:
    if not request_stream_id.startswith("decode:"):
        return None
    node_id = request_stream_id.split(":", maxsplit=1)[1]
    return node_id if node_id else None


def _dict_rows(*, value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [row for row in value if isinstance(row, dict)]


def _int_list(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    return [int(item) for item in value if isinstance(item, int)]


def _mode_ids(*, value: Any) -> list[tuple[str, list[int]]]:
    if not isinstance(value, dict):
        return []
    return [(str(mode), _int_list(ids)) for mode, ids in value.items()]


def _cluster_assignment_items(*, value: Any) -> list[tuple[str, list[tuple[int, str]]]]:
    if not isinstance(value, dict):
        return []
    rows: list[tuple[str, list[tuple[int, str]]]] = []
    for mode, assignments in value.items():
        mode_rows: list[tuple[int, str]] = []
        for assignment in _dict_rows(value=assignments):
            candidate_id = _optional_int(assignment.get("candidate_id"))
            if candidate_id is None:
                continue
            mode_rows.append((candidate_id, str(assignment.get("cluster_name", ""))))
        rows.append((str(mode), mode_rows))
    return rows


def _metric_items(*, value: Any) -> list[tuple[str, tuple[float | None, str]]]:
    if not isinstance(value, dict):
        return []
    return [(str(key), _metric_value(value=item)) for key, item in value.items()]


def _metric_value(*, value: Any) -> tuple[float | None, str]:
    if isinstance(value, bool):
        return (float(int(value)), str(value))
    if isinstance(value, (int, float)):
        return (float(value), str(value))
    return (None, str(value))


def _optional_int(value: Any) -> int | None:
    return (
        int(value) if isinstance(value, int) and not isinstance(value, bool) else None
    )


def _optional_float(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) else None


def _optional_text(value: Any) -> str | None:
    return None if value is None else str(value)


def _executemany(
    *, connection: sqlite3.Connection, sql: str, rows: Sequence[tuple[Any, ...]]
) -> None:
    if rows:
        connection.executemany(sql, rows)
