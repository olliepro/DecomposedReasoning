"""Durable progress logging for long native benchmark rows."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Callable, Literal

from vllm_experimental.native_event_metrics import NativeMetricAccumulator

ProgressEvent = Literal[
    "row_start",
    "heartbeat",
    "chunk_start",
    "chunk_complete",
    "row_complete",
    "row_failed",
]


@dataclass(frozen=True)
class ProgressContext:
    """Static fields identifying one benchmark row."""

    mode: str
    prompt_concurrency: int
    request_prompt_batch_size: int
    request_count: int
    progress_path: Path
    sample_path: Path
    native_event_path: Path
    frontier_trace_path: Path


@dataclass
class ProgressCounters:
    """Mutable counters for one benchmark row."""

    chunks_started: int = 0
    chunks_completed: int = 0
    requests_completed: int = 0
    generated_tokens: int = 0
    active_chunk_index: int = -1
    active_chunk_request_offset: int = 0
    active_chunk_request_count: int = 0
    last_chunk_latency_s: float = 0.0
    last_chunk_tokens: int = 0
    memory_high_mib: int = 0


def complete_line_count(*, path: Path) -> int:
    """Count complete JSONL rows without decoding a potentially large file."""

    if not path.exists():
        return 0
    count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            count += block.count(b"\n")
    return count


def complete_jsonl_lines(*, path: Path) -> list[str]:
    """Read complete JSONL lines, dropping a concurrently written tail."""

    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    if not text:
        return []
    lines = text.splitlines()
    if not text.endswith("\n"):
        return lines[:-1]
    return lines


def read_complete_native_metrics(*, path: Path) -> dict[str, object]:
    """Read native metrics from complete event rows only."""

    metrics = NativeMetricAccumulator()
    for line in complete_jsonl_lines(path=path):
        metrics.observe(event=dict(json.loads(line)))
    return metrics.payload()


class ProgressLogger:
    """Append progress JSONL and stdout heartbeats for one load row."""

    def __init__(
        self,
        *,
        context: ProgressContext,
        memory_sampler: Callable[[], int],
        interval_s: float,
    ) -> None:
        self.context = context
        self.memory_sampler = memory_sampler
        self.interval_s = interval_s
        self.started_unix = time.time()
        self.started_perf = time.perf_counter()
        self._counters = ProgressCounters()
        self._stop = Event()
        self._lock = Lock()
        self._emit_lock = Lock()
        self._thread: Thread | None = None
        self._finished = False

    def start(self) -> None:
        """Start logging the row and launch the heartbeat thread."""

        self.context.progress_path.unlink(missing_ok=True)
        self._set_memory_high(memory_mib=self.memory_sampler())
        self._emit(event="row_start")
        if self.interval_s > 0:
            self._thread = Thread(target=self._heartbeat_loop, daemon=True)
            self._thread.start()

    def chunk_started(
        self, *, chunk_index: int, request_offset: int, request_count: int
    ) -> None:
        """Record that a blocking generation chunk has started."""

        with self._lock:
            self._counters.chunks_started += 1
            self._counters.active_chunk_index = chunk_index
            self._counters.active_chunk_request_offset = request_offset
            self._counters.active_chunk_request_count = request_count
        self._emit(event="chunk_start")

    def chunk_completed(
        self,
        *,
        request_count: int,
        chunk_tokens: int,
        chunk_latency_s: float,
        memory_high_mib: int,
    ) -> None:
        """Record completed chunk work."""

        with self._lock:
            self._counters.chunks_completed += 1
            self._counters.requests_completed += request_count
            self._counters.generated_tokens += chunk_tokens
            self._counters.last_chunk_tokens = chunk_tokens
            self._counters.last_chunk_latency_s = chunk_latency_s
            self._counters.memory_high_mib = max(
                self._counters.memory_high_mib,
                memory_high_mib,
            )
        self._emit(event="chunk_complete")

    def finish(
        self, *, event: Literal["row_complete", "row_failed"], reason: str
    ) -> None:
        """Stop heartbeats and append the final row status."""

        if self._finished:
            return
        self._finished = True
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_s))
        self._set_memory_high(memory_mib=self.memory_sampler())
        self._emit(event=event, failure_reason=reason)

    def _heartbeat_loop(self) -> None:
        """Emit a periodic progress snapshot while the row is active."""

        while not self._stop.wait(self.interval_s):
            self._emit(event="heartbeat")

    def _set_memory_high(self, *, memory_mib: int) -> None:
        """Update the observed memory high watermark."""

        with self._lock:
            self._counters.memory_high_mib = max(
                self._counters.memory_high_mib,
                memory_mib,
            )

    def _counter_payload(self) -> dict[str, int | float]:
        """Return a lock-consistent copy of mutable counters."""

        with self._lock:
            return asdict(self._counters)

    def _context_payload(self) -> dict[str, object]:
        """Return JSON-safe context fields."""

        return {
            "mode": self.context.mode,
            "prompt_concurrency": self.context.prompt_concurrency,
            "request_prompt_batch_size": self.context.request_prompt_batch_size,
            "request_count": self.context.request_count,
            "progress_path": str(self.context.progress_path),
            "sample_path": str(self.context.sample_path),
            "native_event_path": str(self.context.native_event_path),
            "frontier_trace_path": str(self.context.frontier_trace_path),
        }

    def _emit(self, *, event: ProgressEvent, failure_reason: str = "") -> None:
        """Append and print one progress snapshot."""

        payload = self._payload(event=event, failure_reason=failure_reason)
        self.context.progress_path.parent.mkdir(parents=True, exist_ok=True)
        with self._emit_lock:
            with self.context.progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
            print(json.dumps({"native_progress": payload}, sort_keys=True), flush=True)

    def _payload(
        self, *, event: ProgressEvent, failure_reason: str
    ) -> dict[str, object]:
        """Build one progress payload."""

        counters = self._counter_payload()
        elapsed_s = max(time.perf_counter() - self.started_perf, 1e-9)
        native_metrics = read_complete_native_metrics(
            path=self.context.native_event_path
        )
        return {
            "event": event,
            "time_unix": time.time(),
            "started_unix": self.started_unix,
            "elapsed_s": elapsed_s,
            "generated_tokens_s": counters["generated_tokens"] / elapsed_s,
            "failure_reason": failure_reason,
            "sample_output_rows": complete_line_count(path=self.context.sample_path),
            "native_event_rows": complete_line_count(
                path=self.context.native_event_path
            ),
            "frontier_trace_rows": complete_line_count(
                path=self.context.frontier_trace_path
            ),
            "native_boundary_fire_count": native_metrics["boundary_fire_count"],
            "native_branch_count": native_metrics["branch_count"],
            "native_kv_blocks_freed": native_metrics["kv_blocks_freed"],
            "native_branch_pool_queued_count": native_metrics[
                "branch_pool_queued_count"
            ],
            "native_branch_pool_admitted_count": native_metrics[
                "branch_pool_admitted_count"
            ],
            "native_branch_pool_blocked_count": native_metrics[
                "branch_pool_blocked_count"
            ],
            "native_max_live_branch_pools": native_metrics["max_live_branch_pools"],
            "native_max_queued_branch_pools": native_metrics["max_queued_branch_pools"],
            "native_min_branch_free_blocks": native_metrics["min_branch_free_blocks"],
            "native_min_branch_seq_slots": native_metrics["min_branch_seq_slots"],
            **self._context_payload(),
            **counters,
        }
