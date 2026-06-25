"""Tests for dynamic SQLite branching visualization."""

from __future__ import annotations

import json
import re
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

import pytest

from branching_eval.artifact_store import ArtifactStore
from branching_eval.doc_progress import DocProgressSnapshot
from branching_eval.event_db import EventDatabase
from branching_eval.event_types import EventContext
from scripts.visualize_branching_sqlite_payload import (
    event_payload_from_sqlite,
    node_payload_from_sqlite,
    token_trajectory_payload_from_sqlite,
    tree_payload_from_sqlite,
)
from scripts.visualize_branching_common import AttemptKey


def create_empty_event_db(*, path: Path) -> None:
    """Create a schema-only event DB for registry discovery tests."""

    EventDatabase(path=path)


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


def append_decode_chunk(
    *,
    store: ArtifactStore,
    context: EventContext,
    node_id: str,
    request_id: str,
    text: str,
) -> None:
    """Append a minimal request/response/chunk trio for trajectory tests."""

    token_ids = list(range(1, len(text.split()) + 1))
    store.append_event(
        context=context,
        event_type="vllm_request",
        payload={
            "request_id": request_id,
            "request_stream_id": f"decode:{node_id}",
            "request_kind": "decode_chunk",
        },
    )
    store.append_event(
        context=context,
        event_type="vllm_response",
        payload={
            "request_id": request_id,
            "request_stream_id": f"decode:{node_id}",
            "request_kind": "decode_chunk",
            "status": "ok",
            "choice_count": 1,
            "choices": [
                {
                    "index": 0,
                    "text": text,
                    "output_token_count": len(token_ids),
                    "tokens": [],
                }
            ],
        },
    )
    store.append_event(
        context=context,
        event_type="decode_chunk",
        payload={
            "node_id": node_id,
            "chunk_text": text,
            "chunk_token_ids": token_ids,
            "generated_tokens_before_chunk": 0,
            "generated_tokens_after_chunk": len(token_ids),
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


def test_sqlite_payload_disambiguates_answer_and_format_outcomes(
    tmp_path: Path,
) -> None:
    """A correct raw answer with bad structure should be visible as its own state."""

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
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_answer_correct_format_bad",
            "node_id": "node_root",
            "text": "same answer but malformed tags",
            "token_ids": [1, 2, 3],
            "tokens": [],
            "verification": 0,
            "length_tokens_total": 3,
            "length_tokens_exec": 3,
            "stop_reason": "think_end",
            "task_metrics": {
                "acc": False,
                "raw_answer_acc": True,
                "format_valid": False,
                "answer_acc": False,
                "boxed_answer": r"\boxed{\dfrac{1}{2}}",
                "structure_issues": "unpaired_steer_exec_tags",
            },
        },
    )
    store.append_event(
        context=context,
        event_type="doc_finished",
        payload={"status": "completed", "leaf_count": 1, "doc_metrics": {"acc": 0.0}},
    )
    store.flush_events()

    key = AttemptKey(
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    db = EventDatabase(path=run_dir / "tree_events.sqlite")
    graph = tree_payload_from_sqlite(db=db, key=key, detail_base_url="data/doc0")
    leaf_event = next(
        event
        for event in graph["node_events"]["node_root"]
        if event["event_type"] == "leaf_scored"
    )
    assert leaf_event["details"]["verification"] == 0
    assert leaf_event["details"]["raw_answer_acc"] == "True"
    assert leaf_event["details"]["format_valid"] == "False"
    assert leaf_event["details"]["boxed_answer"] == r"\boxed{\dfrac{1}{2}}"
    assert leaf_event["details"]["structure_issues"] == "unpaired_steer_exec_tags"

    node_detail = node_payload_from_sqlite(db=db, key=key, node_id="node_root")
    assert node_detail is not None
    leaf = node_detail["leaves"][0]
    assert leaf["verification"] == 0
    assert leaf["raw_answer_acc"] == "True"
    assert leaf["format_valid"] == "False"
    assert leaf["boxed_answer"] == r"\boxed{\dfrac{1}{2}}"
    assert leaf["task_metrics"]["structure_issues"] == "unpaired_steer_exec_tags"

    from scripts import serve_branching_viz
    from scripts.visualize_branching_run_registry import (
        RunRegistryEntry,
        summary_from_sqlite,
    )

    source = serve_branching_viz.DynamicVizSource(run_dir=run_dir)
    index_html = source.index_html()
    summary_payload = source.summary_payload()
    registry_summary = summary_from_sqlite(
        entry=RunRegistryEntry(run_id="run", run_dir=run_dir)
    )
    assert "avg_problem_answer_acc=1.0000 (n=1)" in index_html
    assert ">1/0<" in index_html
    assert summary_payload["avg_problem_answer_acc"] == 1.0
    assert summary_payload["selected_attempts"][0]["correct_count"] == 0
    assert summary_payload["selected_attempts"][0]["answer_correct_count"] == 1
    assert registry_summary.avg_problem_passrate == 1.0
    assert registry_summary.passrate_text() == "1.0000 (n=1)"


def test_sqlite_leaf_details_load_without_prompt_context_table(
    tmp_path: Path,
) -> None:
    """Older SQLite artifacts without prompt_context should still hydrate leaves."""

    run_dir = tmp_path / "run"
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
    leaf_event = store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_root",
            "node_id": "node_root",
            "text": "full final completion",
            "token_ids": [1, 2, 3],
            "tokens": [],
            "verification": 1,
            "length_tokens_total": 3,
            "length_tokens_exec": 3,
            "stop_reason": "stop",
            "task_metrics": {
                "raw_answer_acc": True,
                "format_valid": True,
                "answer_acc": True,
                "boxed_answer": r"\boxed{42}",
            },
        },
    )
    store.flush_events()
    with sqlite3.connect(run_dir / "tree_events.sqlite") as connection:
        connection.execute("DROP TABLE prompt_context")

    key = AttemptKey(
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    db = EventDatabase(path=run_dir / "tree_events.sqlite", initialize=False)
    node_detail = node_payload_from_sqlite(db=db, key=key, node_id="node_root")
    event_detail = event_payload_from_sqlite(
        db=db,
        event_index=leaf_event.event_index,
    )

    assert node_detail is not None
    assert node_detail["leaves"][0]["text"] == "full final completion"
    assert node_detail["leaves"][0]["raw_answer_acc"] == "True"
    assert event_detail is not None
    assert event_detail["details"]["boxed_answer"] == r"\boxed{42}"
    assert event_detail["details"]["raw_answer_acc"] == "True"
    assert event_detail["details"]["leaf_detail_hydrated"] is True


def test_dynamic_server_reload_reflects_sqlite_appends(tmp_path: Path) -> None:
    """Dynamic server should refresh data from SQLite on browser reload."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    append_attempt_events(store=store, doc_id=0, doc_attempt=0, finished=False)
    store.flush_events()
    analysis_root = Path(__file__).resolve().parents[2]
    script_path = analysis_root / "scripts" / "serve_branching_viz.py"
    process = subprocess.Popen(
        [sys.executable, str(script_path), "--run-dir", str(run_dir), "--port", "0"],
        cwd=str(analysis_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        _ = process.stdout.readline()
        base_url = process.stdout.readline().strip()
        with urllib.request.urlopen(f"{base_url}summary.json", timeout=8) as response:
            first_payload = json.loads(response.read().decode("utf-8"))
        append_attempt_events(store=store, doc_id=1, doc_attempt=0, finished=True)
        store.flush_events()
        with urllib.request.urlopen(f"{base_url}summary.json", timeout=8) as response:
            second_payload = json.loads(response.read().decode("utf-8"))
        slug_0 = "doc_0_attempt_0_aime24_fake_random"
        slug_1 = "doc_1_attempt_0_aime24_fake_random"
        with urllib.request.urlopen(
            f"{base_url}docs/data/{slug_0}.json", timeout=8
        ) as response:
            graph_0 = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(
            f"{base_url}docs/data/{slug_1}.json", timeout=8
        ) as response:
            graph_1 = json.loads(response.read().decode("utf-8"))
        with urllib.request.urlopen(
            f"{base_url}docs/data/{slug_1}/nodes/node_root.json", timeout=8
        ) as response:
            node_1 = json.loads(response.read().decode("utf-8"))
    finally:
        process.terminate()
        try:
            process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=8)
    assert first_payload["event_count"] == 2
    assert second_payload["event_count"] > first_payload["event_count"]
    assert graph_0["meta"]["event_count"] == 2
    assert graph_1["meta"]["event_count"] > graph_0["meta"]["event_count"]
    assert node_1["leaves"][0]["text"] == "answer_1_0"
    assert not (run_dir / "viz" / "summary.json").exists()
    assert not (run_dir / "viz" / "docs").exists()


def test_sqlite_node_advantage_rows_hydrate_rl_tree_labels(tmp_path: Path) -> None:
    """RL tree payloads should expose formatted segment advantage labels."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="structured_baseline",
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
        event_type="edge_selected",
        payload={
            "parent_node_id": "node_root",
            "child_node_id": "node_child",
            "candidate_id": 1,
            "selector_mode": "structured_baseline",
            "candidate_text_normalized": "Check the divisibility condition",
            "candidate_token_ids_normalized": [10, 11],
        },
    )
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": "node_child",
            "parent_node_id": "node_root",
            "branch_points_used": 1,
        },
    )
    store.flush_events()
    db = EventDatabase(path=run_dir / "tree_events.sqlite")
    db.upsert_node_advantage_rows(
        rows=[
            {
                "doc_id": 0,
                "doc_attempt": 0,
                "task_name": "branching_dapo_train",
                "model_id": "branching_dapo",
                "selector_mode": "structured_baseline",
                "prompt_uid": "prompt-1",
                "branch_tree_id": "run:0:0",
                "parent_node_id": "node_root",
                "child_node_id": "node_child",
                "branch_depth": 1,
                "token_start": 4,
                "token_end": 9,
                "mean_combined_advantage": 0.437,
                "token_count": 5,
                "leaf_count": 2,
                "updated_at_event_index": db.last_event_index(),
            }
        ]
    )

    key = AttemptKey(
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="structured_baseline",
    )
    rows = db.read_node_advantage_rows_for_attempt(
        doc_id=key.doc_id,
        doc_attempt=key.doc_attempt,
        task_name=key.task_name,
        model_id=key.model_id,
        selector_mode=key.selector_mode,
    )
    graph = tree_payload_from_sqlite(db=db, key=key, detail_base_url="data/doc0")
    node_child = next(
        node for node in graph["nodes"] if node["node_id"] == "node_child"
    )

    assert rows[0]["child_node_id"] == "node_child"
    assert rows[0]["mean_combined_advantage"] == 0.437
    assert graph["meta"]["is_rl_run"] is True
    assert node_child["advantage_label"] == "+0.44"
    assert node_child["segment_advantage"] == 0.437
    assert node_child["advantage_token_count"] == 5
    assert node_child["advantage_leaf_count"] == 2


def test_prompt_logged_event_is_first_graph_event_and_hydrates_detail(
    tmp_path: Path,
) -> None:
    """Prompt metadata should appear first while full prompt text hydrates lazily."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    store.append_event(
        context=context,
        event_type="doc_started",
        payload={"prompt_char_count": 31},
    )
    prompt_event = store.append_event(
        context=context,
        event_type="prompt_logged",
        payload={
            "node_id": "node_root",
            "prompt_text": "Question: compute 6 * 7\nAnswer:",
            "prompt_char_count": 31,
            "golden_answer": "42",
            "golden_answer_source": "Answer",
            "text_preview": "42",
        },
    )
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": "node_root",
            "parent_node_id": None,
            "branch_points_used": 0,
        },
    )
    store.flush_events()

    key = AttemptKey(
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    db = EventDatabase(path=run_dir / "tree_events.sqlite")
    graph = tree_payload_from_sqlite(db=db, key=key, detail_base_url="data/doc0")
    root_events = graph["node_events"]["node_root"]

    assert root_events[0]["event_type"] == "prompt_logged"
    assert root_events[0]["details"]["golden_answer"] == "42"
    assert root_events[0]["details"]["detail_path"] == (
        f"events/{prompt_event.event_index}.json"
    )
    assert "prompt_text" not in root_events[0]["details"]

    hydrated = event_payload_from_sqlite(
        db=db,
        event_index=prompt_event.event_index,
    )
    assert hydrated is not None
    assert hydrated["details"]["prompt_detail_hydrated"] is True
    assert hydrated["details"]["prompt_text"] == "Question: compute 6 * 7\nAnswer:"
    assert hydrated["details"]["golden_answer"] == "42"
    assert hydrated["details"]["golden_answer_source"] == "Answer"


def test_prompt_logged_event_truncates_detail_and_serves_full_prompt_text(
    tmp_path: Path,
) -> None:
    """Large prompts should not be inserted into clicked-event JSON."""

    from scripts import serve_branching_viz

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    prompt_text = ("head " * 900) + ("middle " * 900) + ("tail " * 900)
    prompt_event = store.append_event(
        context=context,
        event_type="prompt_logged",
        payload={
            "node_id": "node_root",
            "prompt_text": prompt_text,
            "prompt_char_count": len(prompt_text),
            "golden_answer": "42",
            "golden_answer_source": "Answer",
            "text_preview": "42",
        },
    )
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": "node_root",
            "parent_node_id": None,
            "branch_points_used": 0,
        },
    )
    store.flush_events()
    db = EventDatabase(path=run_dir / "tree_events.sqlite")

    hydrated = event_payload_from_sqlite(db=db, event_index=prompt_event.event_index)
    assert hydrated is not None
    details = hydrated["details"]
    assert details["prompt_detail_hydrated"] is True
    assert details["prompt_text"] == ""
    assert details["prompt_truncated"] is True
    assert details["prompt_preview_head"] == prompt_text[:4000]
    assert details["prompt_preview_tail"] == prompt_text[-2000:]
    assert details["prompt_omitted_char_count"] == len(prompt_text) - 6000

    source = serve_branching_viz.DynamicVizSource(run_dir=run_dir)
    response = serve_branching_viz.route_data_request(
        source=source,
        path=(
            "/docs/data/doc_0_attempt_0_aime24_fake_random"
            f"/events/{prompt_event.event_index}/prompt.txt"
        ),
    )
    assert response is not None
    assert response.content_type == "text/plain; charset=utf-8"
    assert response.body.decode("utf-8") == prompt_text


def test_dynamic_server_run_root_serves_multiple_runs(tmp_path: Path) -> None:
    """One dynamic server should route multiple SQLite-backed runs."""

    run_root = tmp_path / "runs"
    run_a = run_root / "run_a"
    run_b = run_root / "run_b"
    store_a = ArtifactStore(run_dir=run_a)
    store_b = ArtifactStore(run_dir=run_b)
    append_attempt_events(store=store_a, doc_id=0, doc_attempt=0, finished=True)
    append_attempt_events(store=store_b, doc_id=1, doc_attempt=0, finished=True)
    store_a.flush_events()
    store_b.flush_events()

    analysis_root = Path(__file__).resolve().parents[2]
    script_path = analysis_root / "scripts" / "serve_branching_viz.py"
    process = subprocess.Popen(
        [
            sys.executable,
            str(script_path),
            "--run-root",
            str(run_root),
            "--port",
            "0",
        ],
        cwd=str(analysis_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        _ = process.stdout.readline()
        base_url = process.stdout.readline().strip()
        with urllib.request.urlopen(base_url, timeout=8) as response:
            index_html = response.read().decode("utf-8")
        run_ids = re.findall(r"href='/runs/([^/]+)/'", index_html)
        assert len(set(run_ids)) == 2
        for run_id in set(run_ids):
            with urllib.request.urlopen(
                f"{base_url}runs/{run_id}/summary.json", timeout=8
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            assert payload["event_count"] == 4
            assert payload["selected_doc_count"] == 1
    finally:
        process.terminate()
        try:
            process.communicate(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=8)


def test_run_picker_hides_older_duplicate_runs(tmp_path: Path) -> None:
    """Duplicate run rows should keep the newest timestamp visible."""

    from scripts.visualize_branching_run_registry import (
        RunRegistryEntry,
        RunRegistrySummary,
        run_registry_page_html,
    )

    old_name = (
        "aime24_qwen35_4b_sft_structured_baseline_rerun_one_nl_"
        "structured_baseline_structured_baseline_seed20260513_20260527T221105Z"
    )
    new_name = (
        "aime24_qwen35_4b_sft_structured_baseline_rerun_one_nl_"
        "structured_baseline_structured_baseline_seed20260513_20260528T031836Z"
    )
    summaries = [
        RunRegistrySummary(
            run_id="old",
            run_name=old_name,
            run_dir=tmp_path / old_name,
            event_count=1,
            attempt_count=1,
            selected_doc_count=1,
            avg_problem_passrate=0.1,
            problem_passrate_count=1,
        ),
        RunRegistrySummary(
            run_id="new",
            run_name=new_name,
            run_dir=tmp_path / new_name,
            event_count=1,
            attempt_count=1,
            selected_doc_count=1,
            avg_problem_passrate=0.2,
            problem_passrate_count=1,
        ),
    ]

    html = run_registry_page_html(
        entries=[
            RunRegistryEntry(run_id=summary.run_id, run_dir=summary.run_dir)
            for summary in summaries
        ],
        summaries=summaries,
    )
    new_pos = html.index(new_name)
    old_pos = html.index(old_name)
    old_row = html[html.rfind("<tr", 0, old_pos) : html.index("</tr>", old_pos)]
    new_row = html[html.rfind("<tr", 0, new_pos) : html.index("</tr>", new_pos)]

    assert new_pos < old_pos
    assert "data-duplicate='0'" in new_row
    assert "data-duplicate='1'" in old_row


def test_run_registry_discovers_all_step_databases(tmp_path: Path) -> None:
    """Run-root discovery should not collapse an experiment to one step."""

    from scripts.visualize_branching_run_registry import RunRegistry

    run_root = tmp_path / "runs"
    experiment_dir = run_root / "qwen35_4b_branching_20260606T000000Z"
    step_names = ["batch_0000_step_000001", "batch_0001_step_000002"]
    for step_name in step_names:
        step_dir = experiment_dir / step_name
        step_dir.mkdir(parents=True)
        create_empty_event_db(path=step_dir / "tree_events.sqlite")

    registry = RunRegistry(run_dirs=[], run_roots=[run_root])
    registry.warm()
    entries = registry.entries()

    assert sorted(entry.run_dir.name for entry in entries) == step_names


def test_run_registry_discovers_latest_batch_by_prefix(tmp_path: Path) -> None:
    """Operational qwen35 view should expose one latest batch per matching root."""

    from scripts.visualize_branching_run_registry import RunRegistry

    run_root = tmp_path / "runs"
    qwen_dir = run_root / "qwen35_4b_branching_20260615T000000Z"
    other_dir = run_root / "olmo3_7b_branching_20260615T000000Z"
    for experiment_dir in [qwen_dir, other_dir]:
        for step_name in ["batch_0000_step_000001", "batch_0002_step_000003"]:
            step_dir = experiment_dir / step_name
            step_dir.mkdir(parents=True)
            create_empty_event_db(path=step_dir / "tree_events.sqlite")

    registry = RunRegistry(
        run_dirs=[],
        run_roots=[run_root],
        run_root_prefixes=["qwen35"],
        latest_batch_only=True,
    )
    registry.warm()
    entries = registry.entries()

    assert [entry.run_dir for entry in entries] == [qwen_dir / "batch_0002_step_000003"]


def test_run_registry_missing_id_does_not_force_refresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stale browser URL should not block on scratch run-root discovery."""

    import scripts.visualize_branching_run_registry as registry_module

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    create_empty_event_db(path=run_dir / "tree_events.sqlite")
    registry = registry_module.RunRegistry(
        run_dirs=[run_dir],
        run_roots=[tmp_path / "slow-root"],
    )
    registry.warm()
    assert registry.entries()

    def fail_discovery(**_: object) -> list[Path]:
        raise AssertionError("missing run id forced run-root discovery")

    monkeypatch.setattr(registry_module, "_discover_run_dirs", fail_discovery)

    assert registry.entry_for_id(run_id="stale-route") is None


def test_run_picker_sorts_by_model_size_task_selector(tmp_path: Path) -> None:
    """Run picker rows should follow the visible column hierarchy."""

    from scripts.visualize_branching_run_registry import (
        RunRegistryEntry,
        RunRegistrySummary,
        run_registry_page_html,
    )

    names = [
        "aime25_qwen35_4b_sft_eps33_embed_diverse_topk_random_seed1_20260528T010000Z",
        "aime24_qwen35_2b_hf_direct_baseline_seed1_20260528T010000Z",
        "aime25_qwen35_0p8b_hf_direct_baseline_seed1_20260528T010000Z",
    ]
    summaries = [
        RunRegistrySummary(
            run_id=f"run_{index}",
            run_name=name,
            run_dir=tmp_path / name,
            event_count=1,
            attempt_count=1,
            selected_doc_count=1,
            avg_problem_passrate=0.1,
            problem_passrate_count=1,
        )
        for index, name in enumerate(names)
    ]

    html = run_registry_page_html(
        entries=[
            RunRegistryEntry(run_id=summary.run_id, run_dir=summary.run_dir)
            for summary in summaries
        ],
        summaries=summaries,
    )
    assert "<tr><th>model</th><th>size</th><th>task</th><th>selector</th>" in html
    ordered_positions = [html.index(name) for name in [names[2], names[1], names[0]]]
    assert ordered_positions == sorted(ordered_positions)


def test_run_picker_marks_pivot_columns(tmp_path: Path) -> None:
    """Visible hierarchy columns should be mergeable by the run picker script."""

    from scripts.visualize_branching_run_registry import (
        RunRegistryEntry,
        RunRegistrySummary,
        run_registry_page_html,
    )

    run_name = (
        "aime25_qwen35_2b_sft_eps33_epsilon_greedy_embed_diverse_"
        "topk_random_seed20260513_20260528T185513Z"
    )
    summary = RunRegistrySummary(
        run_id="run",
        run_name=run_name,
        run_dir=tmp_path / run_name,
        event_count=1,
        attempt_count=1,
        selected_doc_count=1,
        avg_problem_passrate=0.25,
        problem_passrate_count=1,
    )

    html = run_registry_page_html(
        entries=[RunRegistryEntry(run_id=summary.run_id, run_dir=summary.run_dir)],
        summaries=[summary],
    )

    assert "data-pivot='model'" in html
    assert "data-pivot='size'" in html
    assert "data-pivot='task'" in html
    assert "data-pivot='selector'" in html
    assert "applyPivotRowspans();" in html


def test_dynamic_run_level_pages_use_sqlite_aggregates(
    tmp_path: Path, monkeypatch
) -> None:
    """Run-level dynamic pages should not replay every event row."""

    from scripts import serve_branching_viz

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    append_attempt_events(store=store, doc_id=0, doc_attempt=0, finished=True)
    append_attempt_events(store=store, doc_id=1, doc_attempt=0, finished=False)
    write_progress_snapshot(
        store=store,
        doc_id=0,
        doc_attempt=0,
        status="complete",
        passrate=1.0,
        avg_token_length=3.0,
        correct_count=1,
        incorrect_count=0,
        natural_count=1,
        max_count=0,
        repeating_count=0,
        unique_answer_count=1,
    )
    write_progress_snapshot(
        store=store,
        doc_id=1,
        doc_attempt=0,
        status="complete",
        passrate=0.5,
        avg_token_length=4.0,
        correct_count=1,
        incorrect_count=1,
        natural_count=1,
        max_count=1,
        repeating_count=0,
        unique_answer_count=2,
    )
    store.flush_events()

    def fail_full_event_read(self: object) -> list[dict[str, object]]:
        _ = self
        raise AssertionError("run-level pages should use aggregate SQLite queries")

    monkeypatch.setattr(
        serve_branching_viz.EventDatabase,
        "read_event_rows",
        fail_full_event_read,
    )
    source = serve_branching_viz.DynamicVizSource(run_dir=run_dir)

    index_html = source.index_html()
    summary_payload = source.summary_payload()

    assert "events=6" in index_html
    assert "attempts=2" in index_html
    assert "avg_problem_answer_acc=0.7500 (n=2)" in index_html
    assert ">1/0<" in index_html
    assert "gallery.html" not in index_html
    assert summary_payload["event_count"] == 6
    assert summary_payload["attempt_count"] == 2
    assert summary_payload["avg_problem_answer_acc"] == 0.75
    assert summary_payload["avg_problem_passrate"] == 0.75
    assert summary_payload["problem_passrate_count"] == 2


def test_dynamic_doc_pages_do_not_replay_attempt_rows(
    tmp_path: Path, monkeypatch
) -> None:
    """Doc endpoints should use typed SQLite tables, not replay event rows."""

    from scripts import serve_branching_viz

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    append_attempt_events(store=store, doc_id=0, doc_attempt=0, finished=True)
    store.flush_events()

    def fail_attempt_event_read(
        self: object, **kwargs: object
    ) -> list[dict[str, object]]:
        _ = self, kwargs
        raise AssertionError("dynamic doc endpoints should use typed SQLite queries")

    monkeypatch.setattr(
        serve_branching_viz.EventDatabase,
        "read_event_rows_for_attempt",
        fail_attempt_event_read,
    )
    source = serve_branching_viz.DynamicVizSource(run_dir=run_dir)
    slug = "doc_0_attempt_0_aime24_fake_random"

    html = source.attempt_html(slug=slug)
    assert html is not None
    assert "events=4" in html
    graph_payload = source.tree_payload(slug=slug)
    node_payload = source.node_payload(slug=slug, node_id="node_root")

    assert graph_payload is not None
    assert graph_payload["meta"]["event_count"] == 4
    assert node_payload is not None
    assert node_payload["leaves"][0]["text"] == "answer_0_0"


def test_token_trajectory_endpoint_returns_stored_leaf_text(tmp_path: Path) -> None:
    """Legacy trajectory endpoint should not reconstruct final text."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="cluster_across",
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
    append_decode_chunk(
        store=store,
        context=context,
        node_id="node_root",
        request_id="req_root",
        text="root ",
    )
    store.append_event(
        context=context,
        event_type="edge_selected",
        payload={
            "parent_node_id": "node_root",
            "child_node_id": "node_child",
            "candidate_id": 7,
            "selector_mode": "cluster_across",
            "candidate_text_normalized": "edge ",
            "candidate_token_ids_normalized": [10],
        },
    )
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": "node_child",
            "parent_node_id": "node_root",
            "branch_points_used": 1,
        },
    )
    append_decode_chunk(
        store=store,
        context=context,
        node_id="node_child",
        request_id="req_child",
        text="child ",
    )
    store.append_event(
        context=context,
        event_type="edge_selected",
        payload={
            "parent_node_id": "node_child",
            "child_node_id": "node_leaf",
            "candidate_id": 8,
            "selector_mode": "cluster_across",
            "candidate_text_normalized": "edge2 ",
            "candidate_token_ids_normalized": [11],
        },
    )
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": "node_leaf",
            "parent_node_id": "node_child",
            "branch_points_used": 2,
        },
    )
    append_decode_chunk(
        store=store,
        context=context,
        node_id="node_leaf",
        request_id="req_leaf",
        text="leaf",
    )
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_node_leaf_5",
            "node_id": "node_leaf",
            "verification": 1,
            "length_tokens_total": 5,
            "length_tokens_exec": 5,
            "stop_reason": "model_finished",
            "text": "saved final text",
            "text_preview": "saved final text",
            "task_metrics": {"answer_acc": 1.0},
        },
    )
    store.flush_events()

    db = EventDatabase(path=run_dir / "tree_events.sqlite")
    key = AttemptKey(
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="cluster_across",
    )
    payload = token_trajectory_payload_from_sqlite(
        db=db,
        key=key,
        node_id="node_leaf",
    )

    assert payload["text"] == "saved final text"
    assert payload["token_count"] == 5
    assert payload["tokens"] == []


def test_leaf_event_uses_saved_baseline_completion_text(tmp_path: Path) -> None:
    """Clicked HF-direct baseline leaves should use saved full text."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="aime25",
        model_id="fake",
        selector_mode="baseline",
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
        event_type="vllm_request",
        payload={
            "request_id": "req_rollout_3",
            "request_stream_id": "baseline:0:0:baseline_rollout_single:3",
            "request_kind": "baseline_rollout_single",
            "assistant_prefix_char_count": 8,
            "assistant_prefix_tail": "<think>\n",
        },
    )
    store.append_event(
        context=context,
        event_type="vllm_response",
        payload={
            "request_id": "req_rollout_3",
            "request_stream_id": "baseline:0:0:baseline_rollout_single:3",
            "request_kind": "baseline_rollout_single",
            "status": "ok",
            "choice_count": 1,
            "choices": [
                {
                    "index": 0,
                    "text_preview": "Full answer...",
                    "output_token_count": 3,
                    "tokens": [
                        {"token_index": 0, "token_text": "Full", "token_id": 11},
                        {"token_index": 1, "token_text": " answer", "token_id": 12},
                        {"token_index": 2, "token_text": " text", "token_id": 13},
                    ],
                }
            ],
        },
    )
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_baseline_3",
            "node_id": "node_root",
            "verification": 1,
            "length_tokens_total": 3,
            "length_tokens_exec": None,
            "stop_reason": "length",
            "task_metrics": {"exact_match": 1.0},
            "text": "<think>\nFull answer text",
            "text_preview": "<think> Full answer...",
        },
    )
    store.flush_events()

    leaf_event_index = next(
        int(row["event_index"])
        for row in store.read_event_rows()
        if row["event_type"] == "leaf_scored"
    )
    payload = event_payload_from_sqlite(
        db=EventDatabase(path=run_dir / "tree_events.sqlite"),
        event_index=leaf_event_index,
    )
    assert payload is not None
    details = payload["details"]
    assert details["text"] == "<think>\nFull answer text"
    assert details["leaf_detail_hydrated"] is True
    assert "token_trajectory" not in details


def test_leaf_event_uses_saved_structured_baseline_completion(
    tmp_path: Path,
) -> None:
    """Clicked structured-baseline leaves should use saved full text."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="structured_baseline",
    )
    node_id = "node_root_rollout_14"
    store.append_event(context=context, event_type="doc_started", payload={})
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": node_id,
            "parent_node_id": "node_root",
            "branch_points_used": 0,
        },
    )
    append_decode_chunk(
        store=store,
        context=context,
        node_id=node_id,
        request_id="req_chunk",
        text="<think><exec>partial",
    )
    store.append_event(
        context=context,
        event_type="vllm_request",
        payload={
            "request_id": "req_final",
            "request_stream_id": f"decode:{node_id}",
            "request_kind": "think_close_continuation",
        },
    )
    store.append_event(
        context=context,
        event_type="vllm_response",
        payload={
            "request_id": "req_final",
            "request_stream_id": f"decode:{node_id}",
            "request_kind": "think_close_continuation",
            "status": "ok",
            "choice_count": 1,
            "choices": [
                {
                    "index": 0,
                    "text": " final</exec></think>\n\\boxed{2}",
                    "output_token_count": 3,
                    "tokens": [
                        {"token_index": 0, "token_text": " final"},
                        {"token_index": 1, "token_text": "</exec></think>\n"},
                        {"token_index": 2, "token_text": "\\boxed{2}"},
                    ],
                }
            ],
        },
    )
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_node_root_rollout_14_6297",
            "node_id": node_id,
            "verification": 1,
            "length_tokens_total": 6,
            "length_tokens_exec": 6,
            "stop_reason": "model_finished",
            "text": "<think><exec>partial final</exec></think>\n\\boxed{2}",
            "text_preview": "<think><exec>partial...",
        },
    )
    store.flush_events()

    leaf_event_index = next(
        int(row["event_index"])
        for row in store.read_event_rows()
        if row["event_type"] == "leaf_scored"
    )
    payload = event_payload_from_sqlite(
        db=EventDatabase(path=run_dir / "tree_events.sqlite"),
        event_index=leaf_event_index,
    )

    assert payload is not None
    details = payload["details"]
    assert details["leaf_detail_hydrated"] is True
    assert details["text"].endswith(" final</exec></think>\n\\boxed{2}")
    assert "token_trajectory" not in details
    node_payload = node_payload_from_sqlite(
        db=EventDatabase(path=run_dir / "tree_events.sqlite"),
        key=AttemptKey(
            doc_id=0,
            doc_attempt=0,
            task_name="branching_dapo_train",
            model_id="branching_dapo",
            selector_mode="structured_baseline",
        ),
        node_id=node_id,
    )
    assert node_payload is not None
    assert node_payload["trajectory"]["text"].endswith(
        " final</exec></think>\n\\boxed{2}"
    )
    assert node_payload["trajectory"]["segment_count"] == 1


def test_leaf_event_preserves_saved_terminal_completion_order(
    tmp_path: Path,
) -> None:
    """Terminal think-close ordering comes from saved leaf text."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="structured_baseline",
    )
    node_id = "node_root_rollout_14"
    store.append_event(context=context, event_type="doc_started", payload={})
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": node_id,
            "parent_node_id": "node_root",
            "branch_points_used": 0,
        },
    )
    append_decode_chunk(
        store=store,
        context=context,
        node_id=node_id,
        request_id="req_before",
        text="before",
    )
    store.append_event(
        context=context,
        event_type="vllm_request",
        payload={
            "request_id": "req_terminal",
            "request_stream_id": f"decode:{node_id}",
            "request_kind": "think_close_continuation",
        },
    )
    store.append_event(
        context=context,
        event_type="vllm_response",
        payload={
            "request_id": "req_terminal",
            "request_stream_id": f"decode:{node_id}",
            "request_kind": "think_close_continuation",
            "status": "ok",
            "choice_count": 1,
            "choices": [
                {
                    "index": 0,
                    "text": " terminal",
                    "output_token_count": 1,
                    "tokens": [
                        {"token_index": 0, "token_text": " terminal"},
                    ],
                }
            ],
        },
    )
    append_decode_chunk(
        store=store,
        context=context,
        node_id=node_id,
        request_id="req_after",
        text="after",
    )
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_node_root_rollout_14_6297",
            "node_id": node_id,
            "verification": 1,
            "length_tokens_total": 3,
            "length_tokens_exec": 3,
            "stop_reason": "model_finished",
            "text": "beforeafter terminal",
            "text_preview": "before...",
        },
    )
    store.flush_events()

    leaf_event_index = next(
        int(row["event_index"])
        for row in store.read_event_rows()
        if row["event_type"] == "leaf_scored"
    )
    payload = event_payload_from_sqlite(
        db=EventDatabase(path=run_dir / "tree_events.sqlite"),
        event_index=leaf_event_index,
    )
    assert payload is not None
    details = payload["details"]
    assert details["text"].endswith("after terminal")
    assert details["text"].index("after") < details["text"].index("terminal")
    assert "token_trajectory" not in details


def test_repeat_forced_think_close_is_visible_generated_event(
    tmp_path: Path,
) -> None:
    """Future forced-close events should be logged as visible generated text."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="structured_baseline",
    )
    node_id = "node_root_rollout_11"
    forced = "</think>"
    store.append_event(context=context, event_type="doc_started", payload={})
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": node_id,
            "parent_node_id": "node_root",
            "branch_points_used": 0,
        },
    )
    store.append_event(
        context=context,
        event_type="repeat_forced_think_close",
        payload={
            "node_id": node_id,
            "repeat_stop_reason": "repeated_exec_block_loop",
            "repeat_block_kind": "exec",
            "repeat_block_count": 3,
            "forced_close_text": forced,
            "forced_close_token_ids": [1, 2, 3],
            "chunk_was_normalized": True,
            "chunk_token_ids_source": "synthetic_repeat_forced_close",
            "source": "repeat_forced_think_close",
            "generated_tokens_after_close": 10,
        },
    )
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_node_root_rollout_11_10818",
            "node_id": node_id,
            "verification": 1,
            "length_tokens_total": 3,
            "length_tokens_exec": 3,
            "stop_reason": "model_finished",
            "text_preview": forced,
        },
    )
    store.flush_events()

    key = AttemptKey(
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="structured_baseline",
    )
    node_payload = node_payload_from_sqlite(
        db=EventDatabase(path=run_dir / "tree_events.sqlite"),
        key=key,
        node_id=node_id,
    )
    assert node_payload is not None
    event = node_payload["events"][-1]
    forced_event = next(
        event
        for event in node_payload["events"]
        if event["event_type"] == "repeat_forced_think_close"
    )
    assert forced_event["details"]["text_preview"] == forced
    assert node_payload["trajectory"]["text"] == forced


def test_node_trajectory_includes_latest_leaf_completion(
    tmp_path: Path,
) -> None:
    """Node trajectories should append the final leaf completion, not the first one."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="structured_baseline",
    )
    node_id = "node_root_rollout_14"
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
        event_type="edge_selected",
        payload={
            "parent_node_id": "node_root",
            "child_node_id": node_id,
            "candidate_id": 14,
            "selector_mode": "structured_baseline",
            "candidate_text_normalized": "<think><exec>partial",
            "candidate_token_ids_normalized": [1, 2, 3],
        },
    )
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": node_id,
            "parent_node_id": "node_root",
            "branch_points_used": 0,
        },
    )
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_node_root_rollout_14_1",
            "node_id": node_id,
            "verification": 0,
            "length_tokens_total": 4,
            "length_tokens_exec": 4,
            "stop_reason": "model_finished",
            "text_preview": "first leaf",
        },
    )
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_node_root_rollout_14_6297",
            "node_id": node_id,
            "verification": 1,
            "length_tokens_total": 6,
            "length_tokens_exec": 6,
            "stop_reason": "model_finished",
            "text_preview": "second leaf",
        },
    )
    store.flush_events()

    db = EventDatabase(path=run_dir / "tree_events.sqlite")
    key = AttemptKey(
        doc_id=0,
        doc_attempt=0,
        task_name="branching_dapo_train",
        model_id="branching_dapo",
        selector_mode="structured_baseline",
    )
    node_detail = node_payload_from_sqlite(db=db, key=key, node_id=node_id)

    assert node_detail is not None
    assert node_detail["trajectory"]["text"].endswith("second leaf")
    assert node_detail["trajectory"]["segment_count"] == 2
    assert node_detail["trajectory"]["segments"][-1]["text"] == "second leaf"


def test_tree_payload_splits_baseline_leaves_into_display_nodes(
    tmp_path: Path,
) -> None:
    """Non-branching baseline rollouts should render as separate leaf nodes."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="aime25",
        model_id="fake",
        selector_mode="baseline",
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
    for leaf_index in range(2):
        store.append_event(
            context=context,
            event_type="leaf_scored",
            payload={
                "leaf_id": f"leaf_baseline_{leaf_index}",
                "node_id": "node_root",
                "verification": leaf_index,
                "length_tokens_total": 10 + leaf_index,
                "length_tokens_exec": None,
                "stop_reason": "length",
                "task_metrics": {"exact_match": float(leaf_index)},
                "text_preview": f"rollout {leaf_index} preview",
            },
        )
    store.flush_events()

    db = EventDatabase(path=run_dir / "tree_events.sqlite")
    key = AttemptKey(
        doc_id=0,
        doc_attempt=0,
        task_name="aime25",
        model_id="fake",
        selector_mode="baseline",
    )
    payload = tree_payload_from_sqlite(db=db, key=key, detail_base_url="data/doc0")
    node_by_id = {node["node_id"]: node for node in payload["nodes"]}

    assert "node_leaf_baseline_0" in node_by_id
    assert "node_leaf_baseline_1" in node_by_id
    assert node_by_id["node_root"]["leaf_count"] == 0
    assert node_by_id["node_leaf_baseline_0"]["parent_node_id"] == "node_root"
    assert node_by_id["node_leaf_baseline_0"]["leaf_count"] == 1
    assert node_by_id["node_leaf_baseline_0"]["metrics"]["tokens"] == 10.0
    assert {
        (edge["parent_node_id"], edge["child_node_id"]) for edge in payload["edges"]
    } >= {
        ("node_root", "node_leaf_baseline_0"),
        ("node_root", "node_leaf_baseline_1"),
    }
    assert (
        payload["node_events"]["node_leaf_baseline_0"][0]["event_type"] == "leaf_scored"
    )
    assert not payload["node_events"].get("node_root")

    detail = node_payload_from_sqlite(db=db, key=key, node_id="node_leaf_baseline_0")
    assert detail is not None
    assert detail["leaves"][0]["leaf_id"] == "leaf_baseline_0"


def test_selector_details_include_cluster_groups(tmp_path: Path) -> None:
    """Selector details should expose typed cluster groups for expandable UI."""

    run_dir = tmp_path / "run"
    store = ArtifactStore(run_dir=run_dir)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="cluster_across",
    )
    store.append_event(context=context, event_type="doc_started", payload={})
    store.append_event(
        context=context,
        event_type="candidate_pool_resolved",
        payload={
            "branch_point_id": "bp0",
            "candidate_pool_id": "pool0",
            "node_id": "node_root",
            "trigger_type": "entropy",
            "num_candidates": 3,
            "candidates": [
                {"candidate_id": 1, "text": "alpha", "tokens": []},
                {"candidate_id": 2, "text": "beta", "tokens": []},
                {"candidate_id": 3, "text": "gamma", "tokens": []},
            ],
        },
    )
    selector_event = store.append_event(
        context=context,
        event_type="selector_applied",
        payload={
            "branch_point_id": "bp0",
            "node_id": "node_root",
            "active_selector_mode": "cluster_across",
            "selected_candidate_ids": [2],
            "selected_by_mode": {"cluster_across": [2]},
            "shortlist_by_mode": {"cluster_across": [1, 2]},
            "cluster_assignments_by_mode": {
                "cluster_across": [
                    {"candidate_id": 1, "cluster_name": "algebra"},
                    {"candidate_id": 2, "cluster_name": "algebra"},
                    {"candidate_id": 3, "cluster_name": "geometry"},
                ]
            },
        },
    )
    store.flush_events()

    db = EventDatabase(path=run_dir / "tree_events.sqlite")
    payload = event_payload_from_sqlite(db=db, event_index=selector_event.event_index)

    assert payload is not None
    groups = payload["details"]["cluster_groups_by_mode"]["cluster_across"]
    assert groups[0]["cluster_name"] == "algebra"
    assert groups[0]["candidate_ids"] == [1, 2]
    assert groups[0]["selected_candidate_ids"] == [2]
    assert groups[1]["cluster_name"] == "geometry"


def test_tree_payload_hides_trigger_fired_events(tmp_path: Path) -> None:
    """Trigger events should stay logged but not clutter the graph timeline."""

    run_dir = tmp_path / "run"
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
    trigger_event = store.append_event(
        context=context,
        event_type="trigger_fired",
        payload={
            "node_id": "node_root",
            "trigger_type": "steer_boundary",
            "generated_tokens": 12,
        },
    )
    store.append_event(
        context=context,
        event_type="candidate_pool_resolved",
        payload={
            "branch_point_id": "bp0",
            "candidate_pool_id": "pool0",
            "node_id": "node_root",
            "trigger_type": "steer_boundary",
            "num_candidates": 1,
            "candidates": [{"candidate_id": 1, "text": "candidate"}],
        },
    )
    store.flush_events()

    db = EventDatabase(path=run_dir / "tree_events.sqlite")
    rows = db.read_event_rows()
    assert any(row["event_index"] == trigger_event.event_index for row in rows)
    payload = tree_payload_from_sqlite(
        db=db,
        key=AttemptKey(
            doc_id=0,
            doc_attempt=0,
            task_name="aime24",
            model_id="fake",
            selector_mode="random",
        ),
        detail_base_url="data/doc",
    )

    event_types = {
        event["event_type"]
        for events in payload["node_events"].values()
        for event in events
    }
    assert "trigger_fired" not in event_types
    assert "candidate_pool_resolved" in event_types


def test_doc_diagnostics_event_surfaces_in_meta_not_graph(tmp_path: Path) -> None:
    """Per-doc diagnostics should not create replay graph chips."""

    run_dir = tmp_path / "run"
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
    diagnostics_event = store.append_event(
        context=context,
        event_type="doc_diagnostics_recorded",
        payload={
            "doc_id": 0,
            "selector_mode": "random",
            "verification_variance_leaf": 0.25,
            "length_variance_leaf": 9.0,
        },
    )
    store.flush_events()

    payload = tree_payload_from_sqlite(
        db=EventDatabase(path=run_dir / "tree_events.sqlite"),
        key=AttemptKey(
            doc_id=0,
            doc_attempt=0,
            task_name="aime24",
            model_id="fake",
            selector_mode="random",
        ),
        detail_base_url="data/doc",
    )

    event_types = {
        event["event_type"]
        for events in payload["node_events"].values()
        for event in events
    }
    assert "doc_diagnostics_recorded" not in event_types
    assert (
        payload["meta"]["diagnostics"]["event_index"] == diagnostics_event.event_index
    )
    assert payload["meta"]["diagnostics"]["verification_variance_leaf"] == 0.25
