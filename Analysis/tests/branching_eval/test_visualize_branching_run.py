"""Tests for events-only branching visualization replay and follow mode."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

from branching_eval.artifact_store import ArtifactStore
from branching_eval.doc_progress import DocProgressSnapshot
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


def write_progress_snapshot(
    *,
    store: ArtifactStore,
    doc_id: int,
    doc_attempt: int,
    status: str,
    passrate: float,
    avg_token_length: float,
    correct_count: int,
    incorrect_count: int,
    natural_count: int,
    max_count: int,
    repeating_count: int,
    unique_answer_count: int,
) -> None:
    """Write one compact per-doc progress snapshot for viz summary tests."""

    store.write_doc_progress(
        snapshot=DocProgressSnapshot(
            run_id=store.run_id,
            doc_id=doc_id,
            doc_attempt=doc_attempt,
            task_name="aime24",
            model_id="fake",
            selector_mode="random",
            rollout_mode="branching",
            status=status,
            leaf_count=correct_count + incorrect_count,
            passrate=passrate,
            avg_token_length=avg_token_length,
            correct_count=correct_count,
            incorrect_count=incorrect_count,
            natural_count=natural_count,
            max_count=max_count,
            repeating_count=repeating_count,
            other_count=0,
            unique_answer_count=unique_answer_count,
            last_update_timestamp="2026-04-18T12:00:00+00:00",
        )
    )


def test_event_replay_index_selects_latest_completed_else_partial(
    tmp_path: Path,
) -> None:
    """Default index should choose latest completed attempt per doc, else partial."""

    run_dir = tmp_path / "run"
    output_dir = run_dir / "viz"
    store = ArtifactStore(run_dir=run_dir)
    append_attempt_events(store=store, doc_id=0, doc_attempt=0, finished=True)
    append_attempt_events(store=store, doc_id=1, doc_attempt=0, finished=False)
    append_attempt_events(store=store, doc_id=2, doc_attempt=0, finished=True)
    append_attempt_events(store=store, doc_id=2, doc_attempt=1, finished=False)
    store.flush_events()
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


def test_render_snapshot_tree_payload_is_external_json(tmp_path: Path) -> None:
    """Tree payload should be written to a sidecar JSON file and fetched by the page."""

    run_dir = tmp_path / "run"
    output_dir = run_dir / "viz"
    store = ArtifactStore(run_dir=run_dir)
    append_attempt_events(store=store, doc_id=0, doc_attempt=0, finished=True)
    store.flush_events()

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

    assert 'id="tree-data"' not in html
    graph_match = re.search(r'data-graph-path="([^"]+)"', html)
    assert graph_match is not None, "Expected tree graph data path."
    graph_rel_path = graph_match.group(1)
    graph_path = doc_paths[0].parent / graph_rel_path
    assert graph_path.exists()
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    assert "nodes" in payload
    assert "branches" in payload
    assert "node_events" in payload
    assert payload["branches"] == []
    detail_path = payload["nodes"][0]["detail_path"]
    detail_file = doc_paths[0].parent / detail_path
    assert detail_file.exists()
    detail_payload = json.loads(detail_file.read_text(encoding="utf-8"))
    assert detail_payload["leaves"][0]["text"] == "answer_0_0"
    leaf_event = payload["node_events"]["node_root"][0]
    if "details" in leaf_event:
        assert "text" not in leaf_event["details"]


def test_render_snapshot_leaf_detail_reconstructs_steer_text(
    tmp_path: Path,
) -> None:
    """Leaf detail should reconstruct steer-block text without selector duplication."""

    run_dir = tmp_path / "run"
    output_dir = run_dir / "viz"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
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
    store.append_event(
        context=context,
        event_type="selector_continued_inline",
        payload={
            "node_id": "node_root",
            "selected_candidate_id": 81,
            "selected_candidate_text": "Verify provided values</steer",
        },
    )
    store.append_event(
        context=context,
        event_type="decode_chunk",
        payload={
            "node_id": "node_root",
            "chunk_text": "<think>\n<steer>",
            "chunk_token_ids": [1, 2],
            "generated_tokens_before_chunk": 0,
            "generated_tokens_after_chunk": 2,
            "finish_reason": "",
        },
    )
    store.append_event(
        context=context,
        event_type="steer_block_generated",
        payload={
            "node_id": "node_root",
            "source": "explicit_stop_nonbranch",
            "chunk_text": "Verify provided values</steer\n<exec>\n",
            "chunk_token_ids": [1, 2, 3],
            "generated_tokens_before_chunk": 0,
            "generated_tokens_after_chunk": 3,
            "branching_enabled": True,
        },
    )
    store.append_event(
        context=context,
        event_type="decode_chunk",
        payload={
            "node_id": "node_root",
            "chunk_text": "Count partitions",
            "chunk_token_ids": [4, 5],
            "generated_tokens_before_chunk": 3,
            "generated_tokens_after_chunk": 5,
            "finish_reason": "",
        },
    )
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_steer",
            "node_id": "node_root",
            "verification": 1,
            "length_tokens_total": 3,
            "length_tokens_exec": 3,
            "stop_reason": "think_end",
            "task_metrics": {"acc": 1.0},
            "text_preview": "Verify provided values</steer> <exec> Count partitions",
        },
    )
    store.append_event(
        context=context,
        event_type="doc_finished",
        payload={"status": "completed", "leaf_count": 1, "doc_metrics": {"acc": 1.0}},
    )
    store.flush_events()

    render_snapshot(run_dir=run_dir, output_dir=output_dir)
    doc_path = next((output_dir / "docs").glob("*.html"))
    html = doc_path.read_text(encoding="utf-8")
    graph_match = re.search(r'data-graph-path="([^"]+)"', html)
    assert graph_match is not None
    graph_rel_path = graph_match.group(1)
    graph_path = doc_path.parent / graph_rel_path
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    root_row = next(row for row in payload["nodes"] if row["node_id"] == "node_root")
    detail_file = doc_path.parent / root_row["detail_path"]
    detail_payload = json.loads(detail_file.read_text(encoding="utf-8"))
    leaf_text = detail_payload["leaves"][0]["text"]
    assert leaf_text.startswith("<think>\n<steer>Verify provided values</steer>\n")
    assert "<exec>\nCount partitions" in leaf_text
    assert leaf_text.count("Verify provided values") == 1


def test_render_snapshot_scored_leaf_table_sorts_and_renders_heatmap(
    tmp_path: Path,
) -> None:
    """Scored leaves panel should sort rows and render verify-vs-stop heatmap."""

    run_dir = tmp_path / "run"
    output_dir = run_dir / "viz"
    store = ArtifactStore(run_dir=run_dir)
    append_finished_attempt_with_custom_leaves(
        store=store,
        doc_id=0,
        doc_attempt=0,
    )
    store.flush_events()

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
    store = ArtifactStore(run_dir=run_dir)
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


def test_render_snapshot_prefers_progress_snapshots_for_gallery_summary(
    tmp_path: Path,
) -> None:
    """Gallery and summary JSON should use progress snapshots when present."""

    run_dir = tmp_path / "run"
    output_dir = run_dir / "viz"
    store = ArtifactStore(run_dir=run_dir)
    append_attempt_events(store=store, doc_id=0, doc_attempt=0, finished=True)
    write_progress_snapshot(
        store=store,
        doc_id=0,
        doc_attempt=0,
        status="complete",
        passrate=0.75,
        avg_token_length=17.5,
        correct_count=3,
        incorrect_count=1,
        natural_count=1,
        max_count=2,
        repeating_count=1,
        unique_answer_count=4,
    )

    render_snapshot(run_dir=run_dir, output_dir=output_dir)
    summary_payload = json.loads((output_dir / "summary.json").read_text())
    selected = summary_payload["selected_attempts"][0]
    assert selected["status"] == "complete"
    assert float(selected["passrate"]) == 0.75
    assert int(selected["unique_answer_count"]) == 4
    gallery_html = (output_dir / "gallery.html").read_text(encoding="utf-8")
    assert "passrate=0.75" in gallery_html
    assert "finish=mixed(natural=1, max=2, repeating=1)" in gallery_html
    assert "unique_answers=4" in gallery_html


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
