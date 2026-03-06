"""Tests for events-only branching visualization replay and follow mode."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from branching_eval.artifact_store import ArtifactStore
from branching_eval.event_types import EventContext, parse_event_row
from scripts.visualize_branching_payload import tree_payload_for_attempt
from scripts.visualize_branching_replay import AttemptKey, replay_attempts
from scripts.visualize_branching_run import render_snapshot


def append_attempt_events(
    *,
    store: ArtifactStore,
    doc_id: int,
    doc_attempt: int,
    finished: bool,
) -> None:
    """Append minimal attempt events used by replay rendering tests."""

    context = EventContext(
        run_id=store.run_id,
        doc_id=doc_id,
        doc_attempt=doc_attempt,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    store.append_event(context=context, event_type="doc_started", payload={})
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": "node_root",
            "parent_node_id": None,
            "branch_points_used": 0,
        },
    )
    if not finished:
        return
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": f"leaf_{doc_id}_{doc_attempt}",
            "node_id": "node_root",
            "text": f"answer_{doc_id}_{doc_attempt}",
            "token_ids": [1, 2, 3],
            "tokens": [],
            "verification": 1,
            "length_tokens_total": 3,
            "length_tokens_exec": 3,
            "stop_reason": "length",
            "task_metrics": {"acc": 1.0},
        },
    )
    store.append_event(
        context=context,
        event_type="doc_finished",
        payload={
            "status": "completed",
            "leaf_count": 1,
            "leaf_lengths": [3],
            "doc_metrics": {"acc": 1.0},
            "diagnostics": {
                "doc_id": doc_id,
                "selector_mode": "random",
                "verification_variance_leaf": 0.0,
                "length_variance_leaf": 0.0,
                "breakpoint_variance": {
                    "bp1_verification_unweighted": 0.0,
                    "bp1_verification_weighted": 0.0,
                    "bp1_length_unweighted": 0.0,
                    "bp1_length_weighted": 0.0,
                    "bp2_verification_unweighted": 0.0,
                    "bp2_verification_weighted": 0.0,
                    "bp2_length_unweighted": 0.0,
                    "bp2_length_weighted": 0.0,
                },
                "length_summary": {
                    "count": 1,
                    "mean_value": 3.0,
                    "median_value": 3.0,
                    "std_value": 0.0,
                },
            },
        },
    )


def append_finished_attempt_with_custom_leaves(
    *,
    store: ArtifactStore,
    doc_id: int,
    doc_attempt: int,
) -> None:
    """Append one completed attempt with mixed verify/stop leaf outcomes."""

    context = EventContext(
        run_id=store.run_id,
        doc_id=doc_id,
        doc_attempt=doc_attempt,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    store.append_event(context=context, event_type="doc_started", payload={})
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": "node_root",
            "parent_node_id": None,
            "branch_points_used": 0,
        },
    )
    leaf_payloads = [
        {
            "leaf_id": "leaf_correct_length",
            "verification": 1,
            "stop_reason": "length",
        },
        {
            "leaf_id": "leaf_incorrect_length",
            "verification": 0,
            "stop_reason": "length",
        },
        {
            "leaf_id": "leaf_correct_think",
            "verification": 1,
            "stop_reason": "think_end",
        },
        {
            "leaf_id": "leaf_incorrect_repetition",
            "verification": 0,
            "stop_reason": "repeated_exec_block_loop",
        },
    ]
    for payload in leaf_payloads:
        store.append_event(
            context=context,
            event_type="leaf_scored",
            payload={
                "leaf_id": payload["leaf_id"],
                "node_id": "node_root",
                "text": payload["leaf_id"],
                "token_ids": [1, 2, 3],
                "tokens": [],
                "verification": payload["verification"],
                "length_tokens_total": 3,
                "length_tokens_exec": 3,
                "stop_reason": payload["stop_reason"],
                "task_metrics": {"acc": float(payload["verification"])},
            },
        )
    store.append_event(
        context=context,
        event_type="doc_finished",
        payload={
            "status": "completed",
            "leaf_count": 4,
            "leaf_lengths": [3, 3, 3, 3],
            "doc_metrics": {"acc": 2 / 3},
            "diagnostics": {"doc_id": doc_id, "selector_mode": "random"},
        },
    )


def test_event_replay_index_selects_latest_completed_else_partial(
    tmp_path: Path,
) -> None:
    """Default index should choose latest completed attempt per doc, else partial."""

    run_dir = tmp_path / "run"
    output_dir = run_dir / "viz"
    store = ArtifactStore(run_dir=run_dir, reuse_candidate_pools=False)
    append_attempt_events(store=store, doc_id=0, doc_attempt=0, finished=True)
    append_attempt_events(store=store, doc_id=1, doc_attempt=0, finished=False)
    append_attempt_events(store=store, doc_id=2, doc_attempt=0, finished=True)
    append_attempt_events(store=store, doc_id=2, doc_attempt=1, finished=False)
    summary = render_snapshot(run_dir=run_dir, output_dir=output_dir)
    assert summary.event_count > 0
    assert summary.selected_doc_count == 3
    summary_payload = json.loads((output_dir / "summary.json").read_text())
    selected = {
        int(row["doc_id"]): (int(row["doc_attempt"]), str(row["status"]))
        for row in summary_payload["selected_attempts"]
    }
    assert selected[0] == (0, "completed")
    assert selected[1] == (0, "incomplete")
    assert selected[2] == (0, "completed")
    assert (output_dir / "index.html").exists()
    assert (output_dir / "docs").exists()
    index_html = (output_dir / "index.html").read_text(encoding="utf-8")
    assert 'class="path-code"' in index_html
    assert 'class="muted path-row"' in index_html
    assert "<th>correct/incorrect</th>" in index_html
    assert ">1/0<" in index_html


def test_render_snapshot_tree_payload_is_valid_json_script(tmp_path: Path) -> None:
    """Tree payload script should contain raw JSON that `JSON.parse` can read."""

    run_dir = tmp_path / "run"
    output_dir = run_dir / "viz"
    store = ArtifactStore(run_dir=run_dir, reuse_candidate_pools=False)
    append_attempt_events(store=store, doc_id=0, doc_attempt=0, finished=True)

    render_snapshot(run_dir=run_dir, output_dir=output_dir)
    doc_paths = sorted((output_dir / "docs").glob("*.html"))
    assert doc_paths, "Expected at least one rendered doc page."
    html = doc_paths[0].read_text(encoding="utf-8")
    assert 'id="tree-data">{&quot;' not in html
    assert 'id="timeline-svg"' not in html
    assert 'data-mode="tokens"' in html
    assert 'data-mode="steps"' in html
    assert 'data-mode="time"' in html
    assert (
        'class="mode-btn active" type="button" data-mode="steps">Steps</button>' in html
    )
    assert (
        'class="mode-btn active" type="button" data-mode="tokens">Tokens</button>'
        not in html
    )
    assert "sortedNodeEvents" in html
    assert "treeEventMetricValue" in html
    assert "axisScaleByMode" in html
    assert "tickStep: 10" in html
    assert "tickStep: 512" in html
    assert "tickStep: 60" in html
    assert "pixelsPerUnit: 12.0" in html
    assert "pixelsPerUnit: 0.56" in html
    assert "pixelsPerUnit: 1.5" in html
    assert "truncateSvgTextToWidth" in html
    assert "nextPillStartByNode" in html
    assert "✅" in html
    assert "❌" in html
    assert "🛑" in html
    assert "🏁" in html
    assert "🔁" in html

    match = re.search(
        r'<script type="application/json" id="tree-data">(.*?)</script>',
        html,
        flags=re.DOTALL,
    )
    assert match is not None, "Expected tree-data script payload."
    payload = json.loads(match.group(1))
    assert isinstance(payload, dict)
    assert "nodes" in payload
    assert "branches" in payload
    assert "node_events" in payload
    assert payload["branches"] == []


def test_render_snapshot_scored_leaf_table_sorts_and_renders_heatmap(
    tmp_path: Path,
) -> None:
    """Scored leaves panel should sort rows and render verify-vs-stop heatmap."""

    run_dir = tmp_path / "run"
    output_dir = run_dir / "viz"
    store = ArtifactStore(run_dir=run_dir, reuse_candidate_pools=False)
    append_finished_attempt_with_custom_leaves(
        store=store,
        doc_id=0,
        doc_attempt=0,
    )

    render_snapshot(run_dir=run_dir, output_dir=output_dir)
    doc_paths = sorted((output_dir / "docs").glob("*.html"))
    assert doc_paths, "Expected at least one rendered doc page."
    html = doc_paths[0].read_text(encoding="utf-8")
    assert "<th>text</th>" not in html
    assert (
        '<h3 style="margin:0.28rem 0 0.55rem 0">Verification x Stop Reason</h3>' in html
    )
    assert (
        "<th>verify \\ stop</th><th>🏁 think_end</th><th>🛑 length</th>"
        "<th>🔁 repeated_exec_block_loop</th>" in html
    )
    assert html.index("<th scope='row'>correct</th>") < html.index(
        "<th scope='row'>incorrect</th>"
    )
    assert "🏁 think_end" in html
    assert "🛑 length" in html
    assert "🔁 repeated_exec_block_loop" in html
    leaf_ids = re.findall(r"<td><code>(leaf_[^<]+)</code></td>", html)
    assert leaf_ids[:3] == [
        "leaf_correct_think",
        "leaf_correct_length",
        "leaf_incorrect_length",
    ]


def test_follow_mode_rerenders_after_append(tmp_path: Path) -> None:
    """Follow mode should pick up appended events and refresh summary output."""

    run_dir = tmp_path / "run"
    output_dir = run_dir / "viz_follow"
    store = ArtifactStore(run_dir=run_dir, reuse_candidate_pools=False)
    append_attempt_events(store=store, doc_id=0, doc_attempt=0, finished=False)
    analysis_root = Path(__file__).resolve().parents[2]
    script_path = analysis_root / "scripts" / "visualize_branching_run.py"
    command = [
        sys.executable,
        str(script_path),
        "--run-dir",
        str(run_dir),
        "--output-dir",
        str(output_dir),
        "--follow",
        "--poll-seconds",
        "0.05",
        "--max-follow-iterations",
        "8",
    ]
    process = subprocess.Popen(
        command,
        cwd=str(analysis_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        time.sleep(0.15)
        append_attempt_events(store=store, doc_id=1, doc_attempt=0, finished=True)
        stdout, stderr = process.communicate(timeout=8)
    finally:
        if process.poll() is None:
            process.kill()
    assert process.returncode == 0, f"stdout={stdout}\nstderr={stderr}"
    summary_payload = json.loads((output_dir / "summary.json").read_text())
    assert summary_payload["event_count"] >= 5
    assert summary_payload["selected_doc_count"] == 2


def test_resume_gap_is_removed_from_time_axis() -> None:
    """Tree payload should remove downtime gaps before `doc_resumed`."""

    base = {
        "event_version": 2,
        "run_id": "run_resume_gap",
        "doc_id": 0,
        "doc_attempt": 0,
        "task_name": "aime24",
        "model_id": "sft",
        "selector_mode": "cluster_across",
    }
    events = [
        parse_event_row(
            row={
                **base,
                "event_index": 0,
                "timestamp_utc": "2026-02-23T00:00:00+00:00",
                "event_type": "doc_started",
                "payload": {},
            }
        ),
        parse_event_row(
            row={
                **base,
                "event_index": 1,
                "timestamp_utc": "2026-02-23T00:00:01+00:00",
                "event_type": "node_created",
                "payload": {
                    "node_id": "node_root",
                    "parent_node_id": None,
                    "branch_points_used": 0,
                },
            }
        ),
        parse_event_row(
            row={
                **base,
                "event_index": 2,
                "timestamp_utc": "2026-02-23T00:00:02+00:00",
                "event_type": "vllm_request",
                "payload": {
                    "request_id": "req_1",
                    "request_stream_id": "decode:node_root",
                    "request_kind": "decode_chunk",
                    "delta_token_count": 0,
                },
            }
        ),
        parse_event_row(
            row={
                **base,
                "event_index": 3,
                "timestamp_utc": "2026-02-23T00:00:03+00:00",
                "event_type": "vllm_response",
                "payload": {
                    "request_id": "req_1",
                    "request_stream_id": "decode:node_root",
                    "request_kind": "decode_chunk",
                    "status": "ok",
                    "choices": [
                        {
                            "index": 0,
                            "text": "alpha",
                            "finish_reason": "stop",
                            "stop_reason": "<steer",
                            "output_token_count": 2,
                            "tokens": [],
                        }
                    ],
                },
            }
        ),
        parse_event_row(
            row={
                **base,
                "event_index": 4,
                "timestamp_utc": "2026-02-23T01:00:03+00:00",
                "event_type": "doc_resumed",
                "payload": {"reason": "resume_from_partial_logs"},
            }
        ),
        parse_event_row(
            row={
                **base,
                "event_index": 5,
                "timestamp_utc": "2026-02-23T01:00:04+00:00",
                "event_type": "vllm_request",
                "payload": {
                    "request_id": "req_2",
                    "request_stream_id": "decode:node_root",
                    "request_kind": "decode_chunk",
                    "delta_token_count": 0,
                },
            }
        ),
        parse_event_row(
            row={
                **base,
                "event_index": 6,
                "timestamp_utc": "2026-02-23T01:00:05+00:00",
                "event_type": "vllm_response",
                "payload": {
                    "request_id": "req_2",
                    "request_stream_id": "decode:node_root",
                    "request_kind": "decode_chunk",
                    "status": "ok",
                    "choices": [
                        {
                            "index": 0,
                            "text": "beta",
                            "finish_reason": "stop",
                            "stop_reason": "<steer",
                            "output_token_count": 2,
                            "tokens": [],
                        }
                    ],
                },
            }
        ),
    ]
    states = replay_attempts(events=events)
    key = AttemptKey(
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="sft",
        selector_mode="cluster_across",
    )
    state = states[key]
    payload = tree_payload_for_attempt(state=state)
    root_rows = payload["node_events"]["node_root"]
    max_seconds = max(float(row["metrics"]["time_seconds"]) for row in root_rows)
    assert max_seconds < 10.0
    assert float(payload["meta"]["x_max"]["time_seconds"]) < 10.0
