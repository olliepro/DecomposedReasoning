from __future__ import annotations

import json
from pathlib import Path

from branching_eval.event_db import EventDatabase
from branching_eval.steer_trajectory_extractor import (
    NodeKey,
    SuccessfulLeaf,
    build_success_path,
    extract_successful_trajectory,
    run_cli,
)


def write_tree_events(*, path: Path, rows: list[dict[str, object]]) -> None:
    """Write a canonical tree-events SQLite DB for tests.

    Args:
        path: Output tree-events path.
        rows: Event rows to serialize.

    Returns:
        None.
    """

    event_db = EventDatabase(path=path)
    with event_db.connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        event_db.append_event_rows(connection=connection, rows=rows)
        connection.commit()


def make_row(
    *,
    event_index: int,
    event_type: str,
    payload: dict[str, object],
    doc_id: int = 0,
    doc_attempt: int = 0,
) -> dict[str, object]:
    """Build one compact canonical event row for tests."""

    return {
        "event_index": event_index,
        "event_version": 2,
        "timestamp_utc": f"2026-04-13T00:00:{event_index:02d}+00:00",
        "run_id": "run_test",
        "doc_id": doc_id,
        "doc_attempt": doc_attempt,
        "task_name": "aime24",
        "model_id": "sft",
        "selector_mode": "random",
        "event_type": event_type,
        "payload": payload,
    }


def test_build_success_path_orders_root_to_leaf() -> None:
    root = NodeKey(doc_id=0, doc_attempt=0, node_id="root")
    child = NodeKey(doc_id=0, doc_attempt=0, node_id="child")
    grandchild = NodeKey(doc_id=0, doc_attempt=0, node_id="grandchild")
    parent_by_node = {root: None, child: "root", grandchild: "child"}
    path_nodes = build_success_path(
        parent_by_node=parent_by_node,
        selected_leaf=SuccessfulLeaf(
            leaf_id="leaf_grandchild",
            terminal_node=grandchild,
            source_event_type="leaf_scored",
            event_index=9,
        ),
    )
    assert path_nodes == (root, child, grandchild)


def test_extract_successful_trajectory_uses_earliest_scored_success(
    tmp_path: Path,
) -> None:
    tree_events_path = tmp_path / "run_a" / "tree_events.sqlite"
    rows = [
        make_row(
            event_index=0,
            event_type="node_created",
            payload={"node_id": "root", "parent_node_id": None},
        ),
        make_row(
            event_index=1,
            event_type="vllm_request",
            payload={
                "request_id": "req_root",
                "request_stream_id": "decode:root",
                "request_kind": "steer_single_candidate",
            },
        ),
        make_row(
            event_index=2,
            event_type="vllm_response",
            payload={
                "request_id": "req_root",
                "request_stream_id": "decode:root",
                "request_kind": "steer_single_candidate",
                "choices": [{"text": "Root steer"}],
            },
        ),
        make_row(
            event_index=3,
            event_type="node_created",
            payload={"node_id": "node_a", "parent_node_id": "root"},
        ),
        make_row(
            event_index=4,
            event_type="vllm_response",
            payload={
                "request_id": "req_a",
                "request_stream_id": "decode:node_a",
                "request_kind": "steer_single_candidate",
                "choices": [{"text": "Node A steer"}],
            },
        ),
        make_row(
            event_index=5,
            event_type="leaf_completed",
            payload={
                "leaf_id": "leaf_a",
                "node_id": "node_a",
                "verification": 1,
            },
        ),
        make_row(
            event_index=6,
            event_type="node_created",
            payload={"node_id": "node_b", "parent_node_id": "node_a"},
        ),
        make_row(
            event_index=7,
            event_type="vllm_request",
            payload={
                "request_id": "req_b",
                "request_stream_id": "decode:node_b",
                "request_kind": "steer_single_candidate",
            },
        ),
        make_row(
            event_index=8,
            event_type="vllm_response",
            payload={
                "request_id": "req_b",
                "request_stream_id": "decode:node_b",
                "request_kind": "steer_single_candidate",
                "choices": [{"text": "Node B steer"}],
            },
        ),
        make_row(
            event_index=9,
            event_type="leaf_scored",
            payload={
                "leaf_id": "leaf_b",
                "node_id": "node_b",
                "verification": 1,
                "stop_reason": "think_end",
            },
        ),
    ]
    write_tree_events(path=tree_events_path, rows=rows)

    extraction = extract_successful_trajectory(path=tree_events_path)

    assert extraction is not None
    assert extraction.selected_leaf.leaf_id == "leaf_b"
    assert [node.node_id for node in extraction.path_nodes] == [
        "root",
        "node_a",
        "node_b",
    ]
    assert [row["event_index"] for row in extraction.steer_rows] == [1, 2, 4, 7, 8]
    assert [item.steer_text for item in extraction.response_summaries] == [
        "Root steer",
        "Node A steer",
        "Node B steer",
    ]


def test_run_cli_writes_index_and_artifacts(tmp_path: Path) -> None:
    tree_events_path = tmp_path / "runs" / "demo_run" / "tree_events.sqlite"
    rows = [
        make_row(
            event_index=0,
            event_type="node_created",
            payload={"node_id": "root", "parent_node_id": None},
        ),
        make_row(
            event_index=1,
            event_type="leaf_scored",
            payload={"leaf_id": "leaf_root", "node_id": "root", "verification": 1},
        ),
    ]
    write_tree_events(path=tree_events_path, rows=rows)
    output_dir = tmp_path / "extracts"

    index_payload = run_cli(input_path=tmp_path / "runs", output_dir=output_dir)

    assert index_payload["skipped"] == []
    assert len(index_payload["written"]) == 1
    summary_path = (
        output_dir / "demo_run" / "successful_steer_single_candidate.summary.json"
    )
    jsonl_path = output_dir / "demo_run" / "successful_steer_single_candidate.jsonl"
    assert summary_path.exists()
    assert jsonl_path.exists()
    with summary_path.open("r", encoding="utf-8") as handle:
        summary_payload = json.load(handle)
    assert summary_payload["selected_leaf"]["leaf_id"] == "leaf_root"
    assert summary_payload["steer_row_count"] == 0
