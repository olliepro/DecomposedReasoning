"""Artifact logging for branching lm_eval runs."""

from __future__ import annotations

import fcntl
import json
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, TextIO

from branching_eval.doc_progress import DocProgressSnapshot
from branching_eval.event_types import (
    EVENT_SCHEMA_VERSION,
    EventContext,
    EventEnvelope,
    parse_event_row,
    quantize_floats,
    utc_now_iso,
)
from branching_eval.metrics_types import AggregateDiagnostics, DocDiagnostics
from branching_eval.tree_types import BranchTree
from io_utils import append_jsonl, read_jsonl, write_json


@dataclass(frozen=True)
class _QueuedEvent:
    """One buffered event accepted for background serialization and append."""

    event: EventEnvelope


@dataclass(frozen=True)
class _FlushRequest:
    """Barrier request that flushes buffered event bytes to disk visibility."""

    done_event: threading.Event


@dataclass(frozen=True)
class _StopRequest:
    """Sentinel request that flushes and terminates the writer thread."""

    done_event: threading.Event


class ArtifactStore:
    """Persist run artifacts and event logs.

    Args:
        run_dir: Root run directory.
        run_id: Optional explicit run id. Defaults to `run_dir.name`.

    Returns:
        Stateful artifact persistence helper.

    Example:
        >>> store = ArtifactStore(run_dir=Path("output/example"))
        >>> store.tree_events_path.name
        'tree_events.jsonl'
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        run_id: str | None = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.run_id = run_id or self.run_dir.name
        self.tree_events_path = self.run_dir / "tree_events.jsonl"
        self.doc_progress_dir = self.run_dir / "doc_progress"
        self._event_lock_path = self.run_dir / ".tree_events.lock"
        self._append_lock = threading.Lock()
        self._event_queue: Queue[_QueuedEvent | _FlushRequest | _StopRequest] = Queue()
        self._writer_error_lock = threading.Lock()
        self._writer_error: BaseException | None = None
        self._writer_thread: threading.Thread | None = None
        self._closed = False
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._next_event_index = self._read_last_event_index() + 1

    def write_config_snapshot(self, *, payload: dict[str, Any]) -> None:
        """Write run config snapshot.

        Args:
            payload: JSON-serializable run config payload.

        Returns:
            None.
        """

        write_json(path=self.run_dir / "config_snapshot.json", payload=payload)

    def load_config_snapshot(self) -> dict[str, Any] | None:
        """Load run config snapshot when present.

        Args:
            None.

        Returns:
            Snapshot mapping or `None`.
        """

        path = self.run_dir / "config_snapshot.json"
        if not path.exists():
            return None
        return self._load_json(path=path, fallback={})

    def write_run_manifest(self, *, payload: dict[str, Any]) -> None:
        """Write run-level manifest metadata.

        Args:
            payload: Manifest payload.

        Returns:
            None.
        """

        write_json(path=self.run_dir / "run_manifest.json", payload=payload)

    def load_run_manifest(self) -> dict[str, Any] | None:
        """Load run-level manifest metadata when present.

        Args:
            None.

        Returns:
            Manifest mapping or `None`.
        """

        path = self.run_dir / "run_manifest.json"
        if not path.exists():
            return None
        return self._load_json(path=path, fallback={})

    def append_event(
        self,
        *,
        context: EventContext,
        event_type: str,
        payload: dict[str, Any],
    ) -> EventEnvelope:
        """Schedule one canonical event row with monotonic event index.

        Args:
            context: Event attribution labels.
            event_type: Canonical event type.
            payload: Event-specific payload mapping.

        Returns:
            Logical event envelope accepted for buffered append. Disk visibility is
            guaranteed only after `flush_events()`, `close()`, or read helpers that
            flush automatically.
        """

        logical_payload = _as_mapping(value=dict(payload), context="event payload")
        with self._append_lock:
            self._raise_if_closed_locked()
            self._raise_if_writer_failed_locked()
            self._ensure_writer_thread_started_locked()
            event = EventEnvelope(
                event_index=self._allocate_event_index_locked(),
                event_version=EVENT_SCHEMA_VERSION,
                timestamp_utc=utc_now_iso(),
                run_id=context.run_id,
                doc_id=context.doc_id,
                doc_attempt=context.doc_attempt,
                task_name=context.task_name,
                model_id=context.model_id,
                selector_mode=context.selector_mode,
                event_type=event_type,
                payload=logical_payload,
            )
            self._event_queue.put(_QueuedEvent(event=event))
        return event

    def flush_events(self) -> None:
        """Flush all queued events to disk visibility.

        Args:
            None.

        Returns:
            None. All queued rows are visible to subsequent reads on return.
        """

        self._raise_if_writer_failed()
        writer_thread = self._writer_thread
        if writer_thread is None or not writer_thread.is_alive():
            self._raise_if_writer_failed()
            return
        flush_request = _FlushRequest(done_event=threading.Event())
        self._event_queue.put(flush_request)
        flush_request.done_event.wait()
        self._raise_if_writer_failed()

    def close(self) -> None:
        """Flush queued events, stop the writer thread, and surface deferred errors.

        Args:
            None.

        Returns:
            None. Disk visibility for scheduled events is guaranteed on success.
        """

        writer_thread = self._mark_closed_and_get_writer_thread()
        flush_error = _capture_exception(action=self.flush_events)
        stop_error = _capture_exception(action=self._stop_writer_thread)
        if writer_thread is not None:
            writer_thread.join()
        if flush_error is not None:
            raise flush_error
        if stop_error is not None:
            raise stop_error
        self._raise_if_writer_failed()

    def append_tree_event(
        self,
        *,
        tree: BranchTree,
        event_type: str,
        payload: dict[str, Any],
    ) -> EventEnvelope:
        """Append one event row from tree runtime context.

        Args:
            tree: Branch tree context labels.
            event_type: Event label.
            payload: Event payload mapping.

        Returns:
            Event envelope that was written.
        """

        return self.append_event(
            context=EventContext(
                run_id=tree.run_id if tree.run_id else self.run_id,
                doc_id=tree.doc_id,
                doc_attempt=tree.doc_attempt,
                task_name=tree.task_name,
                model_id=tree.model_id,
                selector_mode=tree.selector_mode,
            ),
            event_type=event_type,
            payload=payload,
        )

    def read_event_rows(self) -> list[dict[str, Any]]:
        """Read all event rows from `tree_events.jsonl`.

        Args:
            None.

        Returns:
            Ordered list of raw event row mappings.
        """

        self.flush_events()
        if not self.tree_events_path.exists():
            return []
        return read_jsonl(path=self.tree_events_path)

    def read_events(self) -> list[EventEnvelope]:
        """Read all typed event envelopes from `tree_events.jsonl`.

        Args:
            None.

        Returns:
            Ordered list of parsed event envelopes.
        """

        rows = self.read_event_rows()
        return [parse_event_row(row=row) for row in rows]

    def append_doc_diagnostics(self, *, diagnostics: DocDiagnostics) -> None:
        """Append one per-doc diagnostics row.

        Args:
            diagnostics: Per-doc diagnostic dataclass.

        Returns:
            None.
        """

        append_jsonl(
            path=self.run_dir / "doc_diagnostics.jsonl",
            payload=asdict(diagnostics),
        )

    def write_aggregate_diagnostics(self, *, diagnostics: AggregateDiagnostics) -> None:
        """Write aggregate variance diagnostics.

        Args:
            diagnostics: Aggregate diagnostics dataclass.

        Returns:
            None.
        """

        write_json(
            path=self.run_dir / "variance_diagnostics.json",
            payload=asdict(diagnostics),
        )

    def write_length_diagnostics(self, *, payload: dict[str, Any]) -> None:
        """Write aggregate length diagnostics.

        Args:
            payload: Length diagnostics payload.

        Returns:
            None.
        """

        write_json(path=self.run_dir / "length_diagnostics.json", payload=payload)

    def write_lm_eval_aggregates(self, *, payload: dict[str, Any]) -> None:
        """Write lm_eval aggregate metrics.

        Args:
            payload: Aggregated lm_eval metrics payload.

        Returns:
            None.
        """

        write_json(path=self.run_dir / "lm_eval_aggregates.json", payload=payload)

    def write_doc_progress(self, *, snapshot: DocProgressSnapshot) -> None:
        """Write the latest compact progress snapshot for one doc attempt.

        Args:
            snapshot: Immutable doc-progress snapshot to persist.

        Returns:
            None.
        """

        write_json(
            path=self.doc_progress_dir / snapshot.filename(),
            payload=snapshot.to_payload(),
        )

    def read_doc_progress_snapshots(self) -> list[dict[str, Any]]:
        """Read all latest per-attempt progress snapshots from disk.

        Args:
            None.

        Returns:
            Sorted list of snapshot payload mappings.
        """

        if not self.doc_progress_dir.exists():
            return []
        rows = [
            self._load_json(path=path, fallback={})
            for path in sorted(self.doc_progress_dir.glob("doc_*_attempt_*.json"))
        ]
        return [row for row in rows if row]

    def _allocate_event_index_locked(self) -> int:
        """Allocate next event index from in-memory counter.

        Args:
            None.

        Returns:
            Next monotonic event index for this process.

        Example:
            >>> store = ArtifactStore(run_dir=Path('/tmp/run_x'))
            >>> first = store._allocate_event_index_locked()
            >>> second = store._allocate_event_index_locked()
            >>> second == first + 1
            True

        Notes:
            This counter is initialized from disk once at startup and then updated
            in-memory. The runtime assumes only one writer process appends events
            to a run directory at a time.
        """

        event_index = self._next_event_index
        self._next_event_index += 1
        return event_index

    def _ensure_writer_thread_started_locked(self) -> None:
        """Start the background event writer thread on first append."""

        if self._writer_thread is not None:
            return
        writer_thread = threading.Thread(
            target=self._writer_loop,
            name=f"artifact-writer-{self.run_id}",
            daemon=True,
        )
        self._writer_thread = writer_thread
        writer_thread.start()

    def _mark_closed_and_get_writer_thread(self) -> threading.Thread | None:
        """Mark the store closed and return the current writer thread, if any."""

        with self._append_lock:
            self._closed = True
            return self._writer_thread

    def _stop_writer_thread(self) -> None:
        """Request clean writer shutdown when the background thread is still alive."""

        writer_thread = self._writer_thread
        if writer_thread is None or not writer_thread.is_alive():
            return
        self._raise_if_writer_failed()
        stop_request = _StopRequest(done_event=threading.Event())
        self._event_queue.put(stop_request)
        stop_request.done_event.wait()
        self._raise_if_writer_failed()

    def _writer_loop(self) -> None:
        """Run the buffered event writer until stop or fatal writer error."""

        try:
            with self._event_lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    with self.tree_events_path.open("a", encoding="utf-8") as handle:
                        self._process_writer_items(handle=handle)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except BaseException as error:
            self._record_writer_error(error=error)
        finally:
            self._release_pending_writer_requests()

    def _process_writer_items(self, *, handle: TextIO) -> None:
        """Process queued writer items in FIFO order until a stop request arrives."""

        while True:
            item = self._event_queue.get()
            try:
                if self._process_writer_item(item=item, handle=handle):
                    return
            finally:
                self._event_queue.task_done()

    def _process_writer_item(
        self,
        *,
        item: _QueuedEvent | _FlushRequest | _StopRequest,
        handle: TextIO,
    ) -> bool:
        """Handle one queued writer item and report whether the loop should stop."""

        if isinstance(item, _QueuedEvent):
            self._write_queued_event(event=item.event, handle=handle)
            return False
        if isinstance(item, _FlushRequest):
            try:
                handle.flush()
            finally:
                item.done_event.set()
            return False
        try:
            handle.flush()
        finally:
            item.done_event.set()
        return True

    def _write_queued_event(self, *, event: EventEnvelope, handle: TextIO) -> None:
        """Serialize one queued event and append it to the shared JSONL handle."""

        handle.write(self._serialize_event_line(event=event))
        handle.flush()

    def _serialize_event_line(self, *, event: EventEnvelope) -> str:
        """Serialize one logical event into one normalized JSONL line."""

        normalized_payload = _as_mapping(
            value=quantize_floats(value=event.payload),
            context="event payload",
        )
        event_row = {
            "event_index": event.event_index,
            "event_version": event.event_version,
            "timestamp_utc": event.timestamp_utc,
            "run_id": event.run_id,
            "doc_id": event.doc_id,
            "doc_attempt": event.doc_attempt,
            "task_name": event.task_name,
            "model_id": event.model_id,
            "selector_mode": event.selector_mode,
            "event_type": event.event_type,
            "payload": normalized_payload,
        }
        normalized_row = _as_mapping(
            value=quantize_floats(value=event_row),
            context="event row",
        )
        return json.dumps(normalized_row, ensure_ascii=False) + "\n"

    def _release_pending_writer_requests(self) -> None:
        """Unblock any waiters left in the queue after writer shutdown or failure."""

        while True:
            try:
                item = self._event_queue.get_nowait()
            except Empty:
                return
            try:
                if isinstance(item, (_FlushRequest, _StopRequest)):
                    item.done_event.set()
            finally:
                self._event_queue.task_done()

    def _record_writer_error(self, *, error: BaseException) -> None:
        """Record the first writer-thread failure for later surfacing."""

        with self._writer_error_lock:
            if self._writer_error is None:
                self._writer_error = error

    def _raise_if_writer_failed(self) -> None:
        """Raise the stored writer failure when the background writer has failed."""

        with self._writer_error_lock:
            error = self._writer_error
        if error is None:
            return
        raise RuntimeError("ArtifactStore event writer failed") from error

    def _raise_if_writer_failed_locked(self) -> None:
        """Locked variant used from append paths already holding `_append_lock`."""

        error = self._writer_error
        if error is None:
            return
        raise RuntimeError("ArtifactStore event writer failed") from error

    def _raise_if_closed_locked(self) -> None:
        """Reject new append attempts after the store has been closed."""

        if self._closed:
            raise RuntimeError("ArtifactStore is closed")

    def _read_last_event_index(self) -> int:
        if not self.tree_events_path.exists():
            return -1
        last_index = -1
        with self.tree_events_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line_text = line.strip()
                if not line_text:
                    continue
                payload = json.loads(line_text)
                if not isinstance(payload, dict):
                    continue
                raw_index = payload.get("event_index")
                if isinstance(raw_index, int):
                    last_index = max(last_index, raw_index)
        return last_index

    @staticmethod
    def _load_json(*, path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(fallback)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), f"Expected mapping JSON at {path}"
        return payload


def build_run_manifest_payload(
    *,
    run_id: str,
    task_name: str,
    model_id: str,
    selector_mode: str,
    config_snapshot: dict[str, Any],
    git_commit: str,
) -> dict[str, Any]:
    """Build canonical run manifest payload.

    Args:
        run_id: Stable run id.
        task_name: Task name.
        model_id: Model label.
        selector_mode: Selector mode.
        config_snapshot: Run config snapshot.
        git_commit: Git commit hash.

    Returns:
        Manifest payload mapping.
    """

    return {
        "run_id": run_id,
        "task_name": task_name,
        "model_id": model_id,
        "selector_mode": selector_mode,
        "event_version": EVENT_SCHEMA_VERSION,
        "git_commit": git_commit,
        "timestamp_utc": utc_now_iso(),
        "config": config_snapshot,
    }


def _as_mapping(*, value: Any, context: str) -> dict[str, Any]:
    assert isinstance(value, dict), f"{context} must be a mapping"
    return value


def _capture_exception(*, action: Any) -> BaseException | None:
    """Return the first exception raised by one cleanup action, if any."""

    try:
        action()
    except BaseException as error:
        return error
    return None
