"""Artifact logging and cache persistence for branching lm_eval runs."""

from __future__ import annotations

import fcntl
import json
import threading
from dataclasses import asdict
from pathlib import Path
from typing import Any

from branching_eval.event_types import (
    EVENT_SCHEMA_VERSION,
    EventContext,
    EventEnvelope,
    parse_event_row,
    quantize_floats,
    utc_now_iso,
)
from branching_eval.metrics_types import AggregateDiagnostics, DocDiagnostics
from branching_eval.selector_types import SelectionOutcome
from branching_eval.tree_types import BranchTree, CandidatePoolRecord
from io_utils import append_jsonl, read_jsonl, write_json


class ArtifactStore:
    """Persist run artifacts, event logs, and candidate-pool caches.

    Args:
        run_dir: Root run directory.
        reuse_candidate_pools: Enables candidate-pool cache reuse.
        run_id: Optional explicit run id. Defaults to `run_dir.name`.

    Returns:
        Stateful artifact persistence helper.

    Example:
        >>> store = ArtifactStore(run_dir=Path("output/example"), reuse_candidate_pools=True)
        >>> store.tree_events_path.name
        'tree_events.jsonl'
    """

    def __init__(
        self,
        *,
        run_dir: Path,
        reuse_candidate_pools: bool,
        run_id: str | None = None,
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.run_id = run_id or self.run_dir.name
        self.reuse_candidate_pools = reuse_candidate_pools
        self.cache_dir = self.run_dir / "cache"
        self.tree_events_path = self.run_dir / "tree_events.jsonl"
        self._event_lock_path = self.run_dir / ".tree_events.lock"
        self._append_lock = threading.Lock()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pool_index_path = self.cache_dir / "candidate_pool_index.json"
        self.pool_payload_path = self.cache_dir / "candidate_pools.jsonl"
        self.selection_cache_path = self.cache_dir / "selection_cache.json"
        self._pool_index = self._load_json(path=self.pool_index_path, fallback={})
        self._selection_cache = self._load_json(
            path=self.selection_cache_path,
            fallback={},
        )
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
        """Append one canonical event row with monotonic event index.

        Args:
            context: Event attribution labels.
            event_type: Canonical event type.
            payload: Event-specific payload mapping.

        Returns:
            Event envelope that was written.
        """

        normalized_payload = _as_mapping(
            value=quantize_floats(value=payload),
            context="event payload",
        )
        with self._append_lock:
            with self._event_lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                event_index = self._allocate_event_index_locked()
                event = EventEnvelope(
                    event_index=event_index,
                    event_version=EVENT_SCHEMA_VERSION,
                    timestamp_utc=utc_now_iso(),
                    run_id=context.run_id,
                    doc_id=context.doc_id,
                    doc_attempt=context.doc_attempt,
                    task_name=context.task_name,
                    model_id=context.model_id,
                    selector_mode=context.selector_mode,
                    event_type=event_type,
                    payload=normalized_payload,
                )
                event_row = _as_mapping(
                    value=quantize_floats(value=event.to_json_row()),
                    context="event row",
                )
                append_jsonl(path=self.tree_events_path, payload=event_row)
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return event

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

    def load_candidate_pool_id(self, *, cache_key: str) -> str | None:
        """Load cached candidate pool id for one key.

        Args:
            cache_key: Candidate pool cache key.

        Returns:
            Cached pool id or `None`.
        """

        if not self.reuse_candidate_pools:
            return None
        value = self._pool_index.get(cache_key)
        return str(value) if isinstance(value, str) else None

    def persist_candidate_pool(
        self,
        *,
        cache_key: str,
        pool: CandidatePoolRecord,
    ) -> None:
        """Persist one candidate pool and update key index.

        Args:
            cache_key: Candidate pool cache key.
            pool: Candidate pool payload.

        Returns:
            None.
        """

        append_jsonl(path=self.pool_payload_path, payload=asdict(pool))
        self._pool_index[cache_key] = pool.candidate_pool_id
        write_json(path=self.pool_index_path, payload=self._pool_index)

    def load_selection_cache(self, *, candidate_pool_id: str) -> dict[str, Any] | None:
        """Load cached selection outputs for one candidate pool.

        Args:
            candidate_pool_id: Candidate pool id.

        Returns:
            Selection cache payload or `None`.
        """

        value = self._selection_cache.get(candidate_pool_id)
        return value if isinstance(value, dict) else None

    def persist_selection_cache(
        self,
        *,
        candidate_pool_id: str,
        selections: tuple[SelectionOutcome, ...],
    ) -> None:
        """Persist selector outputs for one candidate pool.

        Args:
            candidate_pool_id: Candidate pool id.
            selections: Selector outcomes for all modes.

        Returns:
            None.
        """

        self._selection_cache[candidate_pool_id] = {
            outcome.selector_mode: _serialize_selection(outcome=outcome)
            for outcome in selections
        }
        write_json(path=self.selection_cache_path, payload=self._selection_cache)

    def load_candidate_pool(
        self,
        *,
        candidate_pool_id: str,
    ) -> CandidatePoolRecord | None:
        """Load one candidate pool by id from JSONL cache.

        Args:
            candidate_pool_id: Candidate pool id.

        Returns:
            Parsed candidate pool record or `None`.
        """

        if not self.pool_payload_path.exists():
            return None
        with self.pool_payload_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                payload = json.loads(line)
                if payload.get("candidate_pool_id") != candidate_pool_id:
                    continue
                assert isinstance(payload, dict), "candidate row must be a mapping"
                return _parse_candidate_pool(payload=payload)
        return None

    def _allocate_event_index_locked(self) -> int:
        """Allocate next event index from in-memory counter.

        Args:
            None.

        Returns:
            Next monotonic event index for this process.

        Example:
            >>> store = ArtifactStore(run_dir=Path('/tmp/run_x'), reuse_candidate_pools=False)
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


def _serialize_selection(*, outcome: SelectionOutcome) -> dict[str, Any]:
    return {
        "selector_mode": outcome.selector_mode,
        "selected_candidate_ids": list(outcome.selected_candidate_ids),
        "cluster_by_candidate_id": outcome.cluster_by_candidate_id,
        "embedding_by_candidate_id": outcome.embedding_by_candidate_id,
    }


def _parse_candidate_pool(*, payload: dict[str, Any]) -> CandidatePoolRecord:
    from branching_eval.tree_types import CandidateRecord, TokenTrace

    raw_candidates = payload.get("candidates", [])
    assert isinstance(raw_candidates, list), "candidates must be a list"
    candidates = tuple(
        _parse_candidate(candidate=row)
        for row in raw_candidates
        if isinstance(row, dict)
    )
    return CandidatePoolRecord(
        candidate_pool_id=str(payload.get("candidate_pool_id", "")),
        cache_key=str(payload.get("cache_key", "")),
        branch_point_id=str(payload.get("branch_point_id", "")),
        node_id=str(payload.get("node_id", "")),
        trigger_type=str(payload.get("trigger_type", "")),
        entropy_value=(
            float(payload["entropy_value"])
            if payload.get("entropy_value") is not None
            else None
        ),
        candidates=tuple(candidates),
    )


def _parse_candidate(*, candidate: dict[str, Any]) -> Any:
    from branching_eval.tree_types import CandidateRecord

    token_rows = candidate.get("tokens", [])
    assert isinstance(token_rows, list), "token rows must be a list"
    tokens = tuple(
        _parse_token_trace(token_row=token_row)
        for token_row in token_rows
        if isinstance(token_row, dict)
    )
    token_ids = tuple(int(item) for item in candidate.get("token_ids", []))
    return CandidateRecord(
        candidate_id=int(candidate.get("candidate_id", 0)),
        text=str(candidate.get("text", "")),
        token_ids=token_ids,
        tokens=tokens,
        finish_reason=str(candidate.get("finish_reason", "unknown")),
        stop_reason=candidate.get("stop_reason"),
    )


def _parse_token_trace(*, token_row: dict[str, Any]) -> Any:
    from branching_eval.tree_types import TokenTrace

    token_id = token_row.get("token_id")
    return TokenTrace(
        token_index=int(token_row.get("token_index", 0)),
        token_id=int(token_id) if token_id is not None else None,
        token_text=str(token_row.get("token_text", "")),
        logprob=float(token_row.get("logprob", 0.0)),
        probability=float(token_row.get("probability", 0.0)),
        entropy=float(token_row.get("entropy", 0.0)),
    )
