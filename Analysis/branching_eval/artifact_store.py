"""Artifact logging for branching lm_eval runs."""

from __future__ import annotations

import fcntl
import json
import threading
from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from branching_eval.doc_progress import DocProgressSnapshot
from branching_eval.event_db import (
    EVENT_DB_FILENAME,
    EventDatabase,
)
from branching_eval.event_types import (
    EVENT_SCHEMA_VERSION,
    FLOAT_QUANTIZATION_DIGITS,
    EventContext,
    EventEnvelope,
    parse_event_row,
    quantize_floats,
    utc_now_iso,
)
from branching_eval.metrics_types import AggregateDiagnostics, DocDiagnostics
from branching_eval.tree_types import BranchTree
from io_utils import write_json


@dataclass(frozen=True)
class _QueuedEvent:
    """One buffered event accepted for background serialization and append."""

    event: EventEnvelope


@dataclass(frozen=True)
class _FlushRequest:
    """Barrier request that flushes buffered events to disk visibility."""

    done_event: threading.Event


@dataclass(frozen=True)
class _QueuedDocProgress:
    """One doc-progress snapshot accepted for serialized SQLite upsert."""

    payload: dict[str, Any]
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
        >>> store.tree_events_db_path.name
        'tree_events.sqlite'
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        run_id: str | None = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.run_id = run_id or self.run_dir.name
        self.tree_events_db_path = self.run_dir / EVENT_DB_FILENAME
        self._event_lock_path = self.run_dir / ".tree_events.sqlite.lock"
        self._append_lock = threading.Lock()
        self._event_queue: Queue[
            _QueuedEvent | _QueuedDocProgress | _FlushRequest | _StopRequest
        ] = Queue()
        self._writer_error_lock = threading.Lock()
        self._writer_error: BaseException | None = None
        self._writer_thread: threading.Thread | None = None
        self._closed = False
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._event_db = EventDatabase(path=self.tree_events_db_path)
        self._write_viz_launcher()
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
        """Read all event rows from `tree_events.sqlite`.

        Args:
            None.

        Returns:
            Ordered list of raw event row mappings.
        """

        self.flush_events()
        return self._event_db.read_event_rows()

    def read_events(self) -> list[EventEnvelope]:
        """Read all typed event envelopes from `tree_events.sqlite`.

        Args:
            None.

        Returns:
            Ordered list of parsed event envelopes.
        """

        rows = self.read_event_rows()
        return [parse_event_row(row=row) for row in rows]

    def append_doc_diagnostics(
        self, *, context: EventContext, diagnostics: DocDiagnostics
    ) -> EventEnvelope:
        """Append one per-doc diagnostics event row.

        Args:
            context: Event context for the document.
            diagnostics: Per-doc diagnostic dataclass.

        Returns:
            Event envelope that was written.
        """

        return self.append_event(
            context=context,
            event_type="doc_diagnostics_recorded",
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

        with self._append_lock:
            self._raise_if_closed_locked()
            self._raise_if_writer_failed_locked()
            self._ensure_writer_thread_started_locked()
            done_event = threading.Event()
            self._event_queue.put(
                _QueuedDocProgress(
                    payload=snapshot.to_payload(),
                    done_event=done_event,
                )
            )
        done_event.wait()
        self._raise_if_writer_failed()

    def read_doc_progress_snapshots(self) -> list[dict[str, Any]]:
        """Read all latest per-attempt progress snapshots from disk.

        Args:
            None.

        Returns:
            Sorted list of snapshot payload mappings.
        """

        return self._event_db.read_doc_progress_rows()

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
                    with closing(self._event_db.connect()) as connection:
                        self._process_writer_items(connection=connection)
                finally:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        except BaseException as error:
            self._record_writer_error(error=error)
        finally:
            self._release_pending_writer_requests()

    def _process_writer_items(self, *, connection: Any) -> None:
        """Process queued writer items in FIFO order until a stop request arrives."""

        while True:
            item = self._event_queue.get()
            try:
                if self._process_writer_item(item=item, connection=connection):
                    return
            finally:
                self._event_queue.task_done()

    def _process_writer_item(
        self,
        *,
        item: _QueuedEvent | _QueuedDocProgress | _FlushRequest | _StopRequest,
        connection: Any,
    ) -> bool:
        """Handle one queued writer item and report whether the loop should stop."""

        if isinstance(item, _QueuedEvent):
            batch = [self._event_row(event=item.event)]
            progress_rows, extra_rows, done_events, should_stop = (
                self._drain_writer_batch()
            )
            batch.extend(extra_rows)
            self._write_sqlite_batch(
                connection=connection,
                event_rows=batch,
                progress_rows=progress_rows,
            )
            for done_event in done_events:
                done_event.set()
            return should_stop
        if isinstance(item, _QueuedDocProgress):
            progress_rows = [item.payload]
            extra_progress, event_rows, done_events, should_stop = (
                self._drain_writer_batch()
            )
            progress_rows.extend(extra_progress)
            self._write_sqlite_batch(
                connection=connection,
                event_rows=event_rows,
                progress_rows=progress_rows,
            )
            item.done_event.set()
            for done_event in done_events:
                done_event.set()
            return should_stop
        if isinstance(item, _FlushRequest):
            try:
                connection.commit()
            finally:
                item.done_event.set()
            return False
        try:
            connection.commit()
        finally:
            item.done_event.set()
        return True

    def _drain_writer_batch(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[threading.Event], bool]:
        """Drain immediately available writer items after one queued event."""

        progress_rows: list[dict[str, Any]] = []
        event_rows: list[dict[str, Any]] = []
        done_events: list[threading.Event] = []
        while True:
            try:
                item = self._event_queue.get_nowait()
            except Empty:
                return progress_rows, event_rows, done_events, False
            try:
                if isinstance(item, _QueuedEvent):
                    event_rows.append(self._event_row(event=item.event))
                elif isinstance(item, _QueuedDocProgress):
                    progress_rows.append(item.payload)
                    done_events.append(item.done_event)
                elif isinstance(item, (_FlushRequest, _StopRequest)):
                    done_events.append(item.done_event)
                    return (
                        progress_rows,
                        event_rows,
                        done_events,
                        isinstance(item, _StopRequest),
                    )
            finally:
                self._event_queue.task_done()

    def _write_sqlite_batch(
        self,
        *,
        connection: Any,
        event_rows: list[dict[str, Any]],
        progress_rows: list[dict[str, Any]],
    ) -> None:
        """Write one mixed batch to SQLite through the single writer connection."""

        if not event_rows and not progress_rows:
            return
        connection.execute("BEGIN IMMEDIATE")
        self._event_db.append_event_rows(connection=connection, rows=event_rows)
        for payload in progress_rows:
            self._event_db.upsert_doc_progress(
                connection=connection,
                payload=payload,
            )
        connection.commit()

    def _event_row(self, *, event: EventEnvelope) -> dict[str, Any]:
        """Serialize one logical event into one normalized row mapping."""

        normalized_payload = _as_mapping(
            value=_quantized_event_payload(event=event),
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
        return normalized_row

    def _release_pending_writer_requests(self) -> None:
        """Unblock any waiters left in the queue after writer shutdown or failure."""

        while True:
            try:
                item = self._event_queue.get_nowait()
            except Empty:
                return
            try:
                if isinstance(item, (_QueuedDocProgress, _FlushRequest, _StopRequest)):
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
        return self._event_db.last_event_index()

    def _write_viz_launcher(self) -> None:
        """Create default dynamic visualization launcher artifacts for this run."""

        viz_dir = self.run_dir / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        analysis_root = Path(__file__).resolve().parents[1]
        launcher_path = viz_dir / "start_viz.sh"
        launcher_text = _viz_launcher_text(
            analysis_root=analysis_root,
            run_dir=self.run_dir,
        )
        _write_text_if_changed(path=launcher_path, text=launcher_text)
        launcher_path.chmod(0o755)
        _write_text_if_changed(
            path=viz_dir / "README.md",
            text=_viz_readme_text(run_dir=self.run_dir),
        )

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


def _quantized_event_payload(*, event: EventEnvelope) -> Any:
    """Return an event payload with stable float quantization."""

    if event.event_type != "vllm_response":
        return quantize_floats(value=event.payload)
    return _quantized_vllm_response_payload(payload=event.payload)


def _quantized_vllm_response_payload(*, payload: dict[str, Any]) -> dict[str, Any]:
    """Quantize the large vLLM response payload without generic recursion."""

    normalized = dict(payload)
    normalized["latency_seconds"] = _quantized_float(
        value=payload.get("latency_seconds")
    )
    choices = payload.get("choices")
    if isinstance(choices, list):
        normalized["choices"] = [
            (
                _quantized_vllm_choice(choice=choice)
                if isinstance(choice, dict)
                else quantize_floats(value=choice)
            )
            for choice in choices
        ]
    return normalized


def _quantized_vllm_choice(*, choice: dict[str, Any]) -> dict[str, Any]:
    """Quantize one serialized vLLM choice row."""

    normalized = dict(choice)
    tokens = choice.get("tokens")
    if isinstance(tokens, list):
        normalized["tokens"] = [
            (
                _quantized_vllm_token(token=token)
                if isinstance(token, dict)
                else quantize_floats(value=token)
            )
            for token in tokens
        ]
    return normalized


def _quantized_vllm_token(*, token: dict[str, Any]) -> dict[str, Any]:
    """Quantize one compact vLLM token row."""

    normalized = dict(token)
    normalized["selected_logprob"] = _quantized_float(
        value=token.get("selected_logprob")
    )
    if "selected_probability" in normalized:
        normalized["selected_probability"] = _quantized_float(
            value=token.get("selected_probability")
        )
    return normalized


def _quantized_float(*, value: Any) -> Any:
    """Round floats with the same policy as generic event quantization."""

    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, FLOAT_QUANTIZATION_DIGITS)
    return value


def _write_text_if_changed(*, path: Path, text: str) -> None:
    """Write text only when the target content differs."""

    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.write_text(text, encoding="utf-8")


def _viz_launcher_text(*, analysis_root: Path, run_dir: Path) -> str:
    """Return the per-run dynamic visualization launcher shell script."""

    return f"""#!/usr/bin/env bash
set -euo pipefail
cd {json.dumps(str(analysis_root))}
exec uv run python scripts/serve_branching_viz.py --run-dir {json.dumps(str(run_dir))} "$@"
"""


def _viz_readme_text(*, run_dir: Path) -> str:
    """Return concise per-run dynamic visualization instructions."""

    return (
        "# Dynamic Branching Visualization\n\n"
        "Run `./start_viz.sh` from this directory, then open the printed local URL. "
        "Reload the browser page to read the latest `tree_events.sqlite` data.\n\n"
        f"Run directory: `{run_dir}`\n"
    )
