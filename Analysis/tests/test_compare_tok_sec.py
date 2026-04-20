"""Tests for token-throughput comparison helpers."""

from __future__ import annotations

import json
from pathlib import Path

from branching_eval.event_types import parse_event_row
from compare_tok_sec import (
    build_comparison_rows,
    select_latest_runs,
    summarize_request_kinds,
    summarize_run,
)


def test_summarize_run_computes_token_throughput(tmp_path: Path) -> None:
    """Run summary should aggregate token counts and both throughput modes."""

    run_dir = write_run_artifacts(
        root=tmp_path,
        run_name="aime24_sft_baseline_baseline_seed1234_20260306T000000.000000Z",
        mode="baseline",
        selector_mode=None,
        seed=1234,
        baseline_rollouts=16,
        events=[
            event_row(
                event_index=0,
                timestamp_utc="2026-03-06T00:00:00+00:00",
                event_type="vllm_request",
                payload={"request_kind": "baseline_rollout_pool"},
            ),
            event_row(
                event_index=1,
                timestamp_utc="2026-03-06T00:00:05+00:00",
                event_type="vllm_response",
                payload={
                    "status": "ok",
                    "request_kind": "baseline_rollout_pool",
                    "latency_seconds": 4.0,
                    "choices": [
                        {"output_token_count": 6},
                        {"output_token_count": 4},
                    ],
                },
            ),
        ],
    )

    summary = summarize_run(run_dir=run_dir)

    assert summary.mode == "baseline"
    assert summary.metrics.output_tokens == 10
    assert summary.metrics.request_latency_seconds == 4.0
    assert summary.metrics.wall_clock_seconds == 5.0
    assert summary.metrics.request_tok_per_sec() == 2.5
    assert summary.metrics.wall_tok_per_sec() == 2.0


def test_comparison_rows_match_latest_baseline(tmp_path: Path) -> None:
    """Branching runs should match the latest baseline for the same key."""

    older_baseline = write_run_artifacts(
        root=tmp_path,
        run_name="aime24_sft_baseline_baseline_seed1234_20260306T000000.000000Z",
        mode="baseline",
        selector_mode=None,
        seed=1234,
        baseline_rollouts=8,
        events=throughput_events(output_tokens=10, latency_seconds=5.0, end_second=5),
    )
    newer_baseline = write_run_artifacts(
        root=tmp_path,
        run_name="aime24_sft_baseline_baseline_seed1234_20260306T000100.000000Z",
        mode="baseline",
        selector_mode=None,
        seed=1234,
        baseline_rollouts=8,
        events=throughput_events(output_tokens=12, latency_seconds=6.0, end_second=6),
    )
    branching_run = write_run_artifacts(
        root=tmp_path,
        run_name="aime24_sft_branching_random_seed1234_20260306T000200.000000Z",
        mode="branching",
        selector_mode="random",
        seed=1234,
        baseline_rollouts=8,
        events=throughput_events(output_tokens=18, latency_seconds=9.0, end_second=9),
    )

    summaries = [
        summarize_run(run_dir=older_baseline),
        summarize_run(run_dir=newer_baseline),
        summarize_run(run_dir=branching_run),
    ]

    latest_summaries = select_latest_runs(summaries=summaries)
    comparisons = build_comparison_rows(summaries=latest_summaries)

    assert len(comparisons) == 1
    comparison = comparisons[0]
    assert comparison.baseline_run_dir == newer_baseline
    assert comparison.branching_run_dir == branching_run
    assert comparison.wall_tok_per_sec_ratio() == 1.0
    assert comparison.request_tok_per_sec_ratio() == 1.0


def test_request_kind_summary_keeps_wall_clock_span() -> None:
    """Per-request-kind summaries should retain request-response wall time."""

    request_kinds = summarize_request_kinds(
        events=tuple(
            parse_event_row(row=row)
            for row in throughput_events(
                output_tokens=10,
                latency_seconds=4.0,
                end_second=5,
            )
        )
    )

    assert len(request_kinds) == 1
    summary = request_kinds[0]
    assert summary.request_kind == "decode_chunk"
    assert summary.metrics.output_tokens == 10
    assert summary.metrics.request_tok_per_sec() == 2.5
    assert summary.metrics.wall_tok_per_sec() == 2.0


def write_run_artifacts(
    *,
    root: Path,
    run_name: str,
    mode: str,
    selector_mode: str | None,
    seed: int,
    baseline_rollouts: int,
    events: list[dict[str, object]],
) -> Path:
    """Write the minimal artifact set required for throughput tests.

    Args:
        root: Temporary test output root.
        run_name: Run directory name.
        mode: Experiment mode.
        selector_mode: Optional selector label.
        seed: Experiment seed.
        baseline_rollouts: Baseline rollout count.
        events: Event rows to persist.

    Returns:
        Created run directory path.
    """

    run_dir = root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config_snapshot = {
        "experiment": {
            "task_name": "aime24",
            "model_id": "sft",
            "mode": mode,
            "selector": selector_mode,
            "seed": seed,
        },
        "run_matrix": {"baseline_rollouts": baseline_rollouts},
    }
    (run_dir / "config_snapshot.json").write_text(
        json.dumps(config_snapshot),
        encoding="utf-8",
    )
    tree_events_text = "\n".join(json.dumps(event) for event in events) + "\n"
    (run_dir / "tree_events.jsonl").write_text(tree_events_text, encoding="utf-8")
    return run_dir


def throughput_events(
    *, output_tokens: int, latency_seconds: float, end_second: int
) -> list[dict[str, object]]:
    """Build a simple request-response event pair for throughput tests.

    Args:
        output_tokens: Tokens reported by the response.
        latency_seconds: Logged vLLM request latency.
        end_second: Response timestamp offset in seconds.

    Returns:
        Event row list with a single request-response pair.
    """

    return [
        event_row(
            event_index=0,
            timestamp_utc="2026-03-06T00:00:00+00:00",
            event_type="vllm_request",
            payload={"request_kind": "decode_chunk"},
        ),
        event_row(
            event_index=1,
            timestamp_utc=f"2026-03-06T00:00:{end_second:02d}+00:00",
            event_type="vllm_response",
            payload={
                "status": "ok",
                "request_kind": "decode_chunk",
                "latency_seconds": latency_seconds,
                "choices": [{"output_token_count": output_tokens}],
            },
        ),
    ]


def event_row(
    *,
    event_index: int,
    timestamp_utc: str,
    event_type: str,
    payload: dict[str, object],
) -> dict[str, object]:
    """Build one canonical event row for throughput tests.

    Args:
        event_index: Monotonic event index.
        timestamp_utc: Event timestamp string.
        event_type: Canonical event type.
        payload: Event payload mapping.

    Returns:
        JSON-ready event row.
    """

    return {
        "event_index": event_index,
        "event_version": 2,
        "timestamp_utc": timestamp_utc,
        "run_id": "run_test",
        "doc_id": 0,
        "doc_attempt": 0,
        "task_name": "aime24",
        "model_id": "sft",
        "selector_mode": "baseline",
        "event_type": event_type,
        "payload": payload,
    }
