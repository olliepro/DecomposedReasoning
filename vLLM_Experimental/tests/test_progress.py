from __future__ import annotations

import json
from pathlib import Path

from vllm_experimental.progress import (
    ProgressContext,
    ProgressLogger,
    complete_line_count,
    read_complete_native_metrics,
)


def read_jsonl(*, path: Path) -> list[dict[str, object]]:
    return [
        dict(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_progress_logger_records_salvageable_chunk_state(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.jsonl"
    sample_path = tmp_path / "sample_outputs.jsonl"
    event_path = tmp_path / "native_events.jsonl"
    frontier_path = tmp_path / "native_frontier.jsonl"
    sample_path.write_text('{"request": 0}\n{"partial"', encoding="utf-8")
    event_path.write_text(
        json.dumps(
            {
                "candidate_count": 2,
                "event": "branch_start",
                "fork_tokens": 100,
                "shared_blocks": 3,
            }
        )
        + "\n"
        + '{"event": "partial"',
        encoding="utf-8",
    )

    logger = ProgressLogger(
        context=ProgressContext(
            mode="eps_on_policy_diverse",
            prompt_concurrency=8,
            request_prompt_batch_size=4,
            request_count=8,
            progress_path=progress_path,
            sample_path=sample_path,
            native_event_path=event_path,
            frontier_trace_path=frontier_path,
        ),
        memory_sampler=lambda: 1234,
        interval_s=0,
    )
    logger.start()
    logger.chunk_started(chunk_index=0, request_offset=0, request_count=4)
    logger.chunk_completed(
        request_count=4,
        chunk_tokens=2048,
        chunk_latency_s=12.5,
        memory_high_mib=2345,
    )
    logger.finish(event="row_complete", reason="")

    rows = read_jsonl(path=progress_path)
    assert [row["event"] for row in rows] == [
        "row_start",
        "chunk_start",
        "chunk_complete",
        "row_complete",
    ]
    assert rows[-1]["requests_completed"] == 4
    assert rows[-1]["generated_tokens"] == 2048
    assert rows[-1]["sample_output_rows"] == 1
    assert rows[-1]["native_event_rows"] == 1
    assert rows[-1]["native_boundary_fire_count"] == 1
    assert rows[-1]["native_branch_count"] == 2
    assert rows[-1]["memory_high_mib"] == 2345


def test_progress_helpers_ignore_partial_tail(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        json.dumps(
            {
                "candidate_count": 1,
                "event": "branch_wave_start",
                "fork_tokens": 7,
                "shared_blocks": 2,
            }
        )
        + "\n"
        + "{",
        encoding="utf-8",
    )

    metrics = read_complete_native_metrics(path=path)

    assert complete_line_count(path=path) == 1
    assert metrics["branch_count"] == 1
    assert metrics["prefill_tokens_avoided"] == 7
