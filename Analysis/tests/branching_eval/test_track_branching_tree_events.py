from __future__ import annotations

from pathlib import Path

from branching_eval.event_types import EventEnvelope
from branching_eval.tree_event_stats import render_summary, summarize_events


def test_track_branching_tree_events_summary() -> None:
    """The tracker should summarize started, finished, and stop reasons."""

    events = (
        EventEnvelope(
            event_index=0,
            event_version=2,
            timestamp_utc="2026-04-01T00:00:00+00:00",
            run_id="run",
            doc_id=1,
            doc_attempt=0,
            task_name="task",
            model_id="model",
            selector_mode="cluster_across",
            event_type="rollout_started",
            payload={"leaf_limit": 16},
        ),
        EventEnvelope(
            event_index=1,
            event_version=2,
            timestamp_utc="2026-04-01T00:00:01+00:00",
            run_id="run",
            doc_id=1,
            doc_attempt=0,
            task_name="task",
            model_id="model",
            selector_mode="cluster_across",
            event_type="leaf_completed",
            payload={"stop_reason": "think_end"},
        ),
        EventEnvelope(
            event_index=2,
            event_version=2,
            timestamp_utc="2026-04-01T00:00:02+00:00",
            run_id="run",
            doc_id=1,
            doc_attempt=0,
            task_name="task",
            model_id="model",
            selector_mode="cluster_across",
            event_type="rollout_finished",
            payload={},
        ),
        EventEnvelope(
            event_index=3,
            event_version=2,
            timestamp_utc="2026-04-01T00:00:03+00:00",
            run_id="run",
            doc_id=2,
            doc_attempt=0,
            task_name="task",
            model_id="model",
            selector_mode="cluster_across",
            event_type="rollout_started",
            payload={"leaf_limit": 16},
        ),
        EventEnvelope(
            event_index=4,
            event_version=2,
            timestamp_utc="2026-04-01T00:00:04+00:00",
            run_id="run",
            doc_id=2,
            doc_attempt=0,
            task_name="task",
            model_id="model",
            selector_mode="cluster_across",
            event_type="leaf_completed",
            payload={"stop_reason": "max_gen_toks_reached"},
        ),
    )

    summary = summarize_events(path=Path("tree_events.sqlite"), events=events)
    text = render_summary(summary=summary)

    assert summary.started_docs() == 2
    assert summary.finished_docs() == 1
    assert summary.leaf_total() == 2
    assert summary.natural_total() == 1
    assert summary.unnatural_total() == 1
    assert "max_gen_toks_reached=1" in text
    assert "doc 2 | open | 1 | 0 | 1 | 16 | 6.2%" in text
