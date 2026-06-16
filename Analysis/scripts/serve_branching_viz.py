#!/usr/bin/env python3
"""Serve the branching visualization dynamically from a run SQLite DB."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import threading
import time
from dataclasses import dataclass, field
from html import escape
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from branching_eval.event_db import (  # noqa: E402
    EVENT_DB_FILENAME,
    EventDatabase,
)
from scripts.visualize_branching_interactive import (  # noqa: E402
    render_tree_workspace,
    tree_workspace_script,
)
from scripts.visualize_branching_replay import (  # noqa: E402
    AttemptKey,
)
from scripts.visualize_branching_run import (  # noqa: E402
    ProgressAttemptView,
    selected_progress_attempts_by_doc,
)
from scripts.visualize_branching_run_registry import (  # noqa: E402
    RunRegistry,
    RunRegistryEntry,
    RunRegistrySummary,
    run_registry_page_html,
    split_run_route,
    summary_from_path,
    summary_from_sqlite,
)
from scripts.visualize_branching_sqlite_payload import (  # noqa: E402
    AttemptPageData,
    attempt_page_data_from_sqlite,
    event_payload_from_sqlite,
    node_payload_from_sqlite,
    token_trajectory_payload_from_sqlite,
    tree_payload_from_sqlite,
)
from scripts.visualize_branching_web import theme_css, wrap_page  # noqa: E402


@dataclass(frozen=True)
class ResponsePayload:
    """HTTP response body and content type."""

    body: bytes
    content_type: str


@dataclass
class CacheEntry:
    """One stale-while-refresh payload cache entry."""

    payload: Any
    loaded_at: float
    refreshing: bool = False


class ReusableThreadingHTTPServer(ThreadingHTTPServer):
    """HTTP server variant that supports immediate visualizer restarts."""

    allow_reuse_address = True


class RegistrySummaryCache:
    """Small cache for expensive run-root summary rows."""

    def __init__(self, *, refresh_after_seconds: float = 120.0) -> None:
        self._refresh_after_seconds = refresh_after_seconds
        self._lock = threading.Lock()
        self._cache: dict[str, CacheEntry] = {}

    def summaries_for_entries(
        self, *, entries: list[RunRegistryEntry], sync_missing_limit: int = 4
    ) -> list[RunRegistrySummary]:
        """Return summaries without scanning every historical DB on the request path."""

        now = time.monotonic()
        summaries: list[RunRegistrySummary] = []
        sync_remaining = sync_missing_limit
        for entry in entries:
            cache_key = str(entry.run_dir)
            cached = self._cached_entry(cache_key=cache_key)
            if cached is not None and not self._is_stale(entry=cached, now=now):
                summaries.append(cached.payload)
                continue
            if sync_remaining > 0:
                summary = summary_from_sqlite(entry=entry)
                self._store(cache_key=cache_key, payload=summary)
                summaries.append(summary)
                sync_remaining -= 1
            else:
                summary = (
                    cached.payload
                    if cached is not None
                    else summary_from_path(entry=entry)
                )
                summaries.append(summary)
        return summaries

    def _cached_entry(self, *, cache_key: str) -> CacheEntry | None:
        with self._lock:
            return self._cache.get(cache_key)

    def _is_stale(self, *, entry: CacheEntry, now: float) -> bool:
        return now - entry.loaded_at >= self._refresh_after_seconds

    def _store(self, *, cache_key: str, payload: Any) -> None:
        with self._lock:
            self._cache[cache_key] = CacheEntry(
                payload=payload,
                loaded_at=time.monotonic(),
            )


@dataclass(frozen=True)
class AttemptSummaryView:
    """Run-index summary sourced from SQLite aggregate queries."""

    key: AttemptKey
    status: str
    correct_count: int
    incorrect_count: int
    answer_correct_count: int
    answer_count: int
    leaf_count: int
    avg_token_length: float
    natural_count: int
    max_count: int
    repeating_count: int
    other_count: int
    unique_answer_count: int
    event_count: int
    last_event_index: int
    last_timestamp_utc: str

    def verification_counts_text(self) -> str:
        """Return `correct/incorrect` display text."""

        return f"{self.correct_count}/{self.incorrect_count}"

    def answer_counts_text(self) -> str:
        """Return answer-correct/answer-incorrect display text."""

        if self.answer_count > 0:
            return f"{self.answer_correct_count}/{self.answer_count - self.answer_correct_count}"
        return self.verification_counts_text()

    def answer_accuracy(self) -> float | None:
        """Return raw-answer accuracy, falling back to reward accuracy."""

        if self.answer_count > 0:
            return self.answer_correct_count / self.answer_count
        if self.leaf_count > 0:
            return self.correct_count / self.leaf_count
        return None


@dataclass
class LeafStats:
    """Aggregated leaf-score stats for one attempt."""

    leaf_count_total: int = 0
    correct_count: int = 0
    incorrect_count: int = 0
    answer_correct_count: int = 0
    answer_count: int = 0
    natural_count: int = 0
    max_count: int = 0
    repeating_count: int = 0
    other_count: int = 0
    length_token_total: int = 0
    length_count: int = 0
    unique_answer_count: int = 0
    unique_texts: set[str] = field(default_factory=set)

    def leaf_count(self) -> int:
        """Return known scored-leaf count."""

        return self.leaf_count_total

    def avg_token_length(self) -> float:
        """Return mean leaf token length when available."""

        if self.length_count == 0:
            return 0.0
        return self.length_token_total / self.length_count


@dataclass(frozen=True)
class ProblemPassrateSummary:
    """Equal-weight aggregate over selected problem answer accuracy."""

    average: float | None
    problem_count: int

    def pill_text(self) -> str:
        """Return compact index-page display text."""

        if self.average is None:
            return "avg_problem_answer_acc=n/a"
        return f"avg_problem_answer_acc={self.average:.4f} (n={self.problem_count})"


class DynamicVizSource:
    """Read current branching visualization data directly from SQLite."""

    def __init__(self, *, run_dir: Path) -> None:
        self.run_dir = run_dir.resolve()
        self.db = EventDatabase(path=self.run_dir / EVENT_DB_FILENAME, initialize=False)
        self._cache: dict[str, CacheEntry] = {}
        self._cache_lock = threading.Lock()
        self._ensure_fast_indexes_later()

    def _ensure_fast_indexes_later(self) -> None:
        """Create current-DB read indexes without blocking the first request."""

        thread = threading.Thread(target=self._ensure_fast_indexes, daemon=True)
        thread.start()

    def _ensure_fast_indexes(self) -> None:
        try:
            self.db.ensure_fast_read_indexes()
        except sqlite3.Error:
            return

    def index_html(self) -> str:
        """Return the current run index page."""

        return self._cached_index_html()

    def _cached_index_html(self) -> str:
        """Return cached run index HTML, building slow DB summaries off-thread."""

        if self._should_build_index_synchronously():
            return self._index_html_uncached()
        cache_key = "run_index_html"
        with self._cache_lock:
            entry = self._cache.get(cache_key)
            if entry is not None:
                stale = time.monotonic() - entry.loaded_at >= 5.0
                if stale and not entry.refreshing:
                    entry.refreshing = True
                    self._refresh_later(
                        cache_key=cache_key,
                        builder=self._index_html_uncached,
                    )
                return str(entry.payload)
            self._cache[cache_key] = CacheEntry(
                payload=loading_index_page_html(run_dir=self.run_dir),
                loaded_at=time.monotonic(),
                refreshing=True,
            )
        thread = threading.Thread(
            target=self._refresh_cache_entry,
            kwargs={"cache_key": cache_key, "builder": self._index_html_uncached},
            daemon=True,
        )
        thread.start()
        thread.join(timeout=0.75)
        with self._cache_lock:
            return str(self._cache[cache_key].payload)

    def _index_html_uncached(self) -> str:
        """Build the run index from lightweight typed SQLite rows."""

        if not self._should_use_fast_index():
            return self._full_index_html_uncached()
        progress_attempts = self.progress_attempts()
        selected_progress = selected_progress_attempts_by_doc(
            progress_attempts=progress_attempts
        )
        return index_page_html(
            run_dir=self.run_dir,
            event_count=None,
            attempt_summaries=[],
            selected_summaries=[],
            selected_progress_attempts=selected_progress,
            all_progress_attempts=progress_attempts,
        )

    def _full_index_html_uncached(self) -> str:
        """Build the full run index for small SQLite DBs."""

        summaries = self.attempt_summaries()
        selected_summaries = selected_attempt_summaries_by_doc(summaries=summaries)
        progress_attempts = self.progress_attempts()
        selected_progress = selected_progress_attempts_by_doc(
            progress_attempts=progress_attempts
        )
        return index_page_html(
            run_dir=self.run_dir,
            event_count=self.db.event_count(),
            attempt_summaries=summaries,
            selected_summaries=selected_summaries,
            selected_progress_attempts=selected_progress,
            all_progress_attempts=progress_attempts,
        )

    def _should_build_index_synchronously(self) -> bool:
        """Return whether the run DB is small enough for a blocking index."""

        return not self._should_use_fast_index()

    def _should_use_fast_index(self) -> bool:
        """Return whether the run index should avoid full-DB aggregates."""

        return (self.run_dir / EVENT_DB_FILENAME).stat().st_size > 128 * 1024 * 1024

    def summary_payload(self) -> dict[str, Any]:
        """Return the current machine-readable run summary."""

        summaries = self.attempt_summaries()
        selected_summaries = selected_attempt_summaries_by_doc(summaries=summaries)
        progress_attempts = self.progress_attempts()
        selected_progress = selected_progress_attempts_by_doc(
            progress_attempts=progress_attempts
        )
        if selected_progress:
            return progress_summary_payload(
                run_dir=self.run_dir,
                event_count=self.db.event_count(),
                attempt_count=len(progress_attempts),
                selected_progress_attempts=selected_progress,
                selected_summaries=selected_summaries,
            )
        return summary_payload_from_summaries(
            run_dir=self.run_dir,
            event_count=self.db.event_count(),
            summaries=summaries,
            selected_summaries=selected_summaries,
        )

    def attempt_html(self, *, slug: str) -> str | None:
        """Return the current doc-attempt page for one slug."""

        key = self.attempt_key_for_slug(slug=slug)
        if key is None:
            return None
        page_data = attempt_page_data_from_sqlite(db=self.db, key=key)
        return attempt_page_html(run_dir=self.run_dir, page_data=page_data)

    def tree_payload(self, *, slug: str) -> dict[str, Any] | None:
        """Return compact graph data for one attempt slug."""

        key = self.attempt_key_for_slug(slug=slug)
        if key is None:
            return None
        return self._cached(
            cache_key=f"tree:{slug}",
            builder=lambda: tree_payload_from_sqlite(
                db=self.db,
                key=key,
                detail_base_url=f"data/{key.slug()}",
            ),
        )

    def node_payload(self, *, slug: str, node_id: str) -> dict[str, Any] | None:
        """Return lazy inspector data for one node in one attempt slug."""

        key = self.attempt_key_for_slug(slug=slug)
        if key is None:
            return None
        return self._cached(
            cache_key=f"node:{slug}:{node_id}",
            builder=lambda: node_payload_from_sqlite(
                db=self.db,
                key=key,
                node_id=node_id,
            ),
        )

    def event_payload(self, *, event_index: int) -> dict[str, Any] | None:
        """Return heavy details for one clicked event."""

        return self._cached(
            cache_key=f"event:{event_index}",
            builder=lambda: event_payload_from_sqlite(
                db=self.db,
                event_index=event_index,
            ),
        )

    def prompt_text(self, *, event_index: int) -> str | None:
        """Return the full logged prompt text for one prompt event."""

        return self.db.read_prompt_text_by_event_index(event_index=event_index)

    def token_trajectory_payload(
        self, *, slug: str, node_id: str
    ) -> dict[str, Any] | None:
        """Return token-level final trajectory data for one decode node."""

        key = self.attempt_key_for_slug(slug=slug)
        if key is None:
            return None
        return self._cached(
            cache_key=f"token-trajectory:{slug}:{node_id}",
            builder=lambda: token_trajectory_payload_from_sqlite(
                db=self.db,
                key=key,
                node_id=node_id,
            ),
        )

    def _cached(
        self,
        *,
        cache_key: str,
        builder: Callable[[], Any],
        refresh_after_seconds: float = 1.0,
    ) -> Any:
        """Return cached payload immediately and refresh stale entries off-thread."""

        now = time.monotonic()
        with self._cache_lock:
            entry = self._cache.get(cache_key)
        if entry is None:
            payload = builder()
            with self._cache_lock:
                self._cache[cache_key] = CacheEntry(
                    payload=payload,
                    loaded_at=time.monotonic(),
                )
            return payload
        should_refresh = now - entry.loaded_at >= refresh_after_seconds
        if should_refresh and not entry.refreshing:
            with self._cache_lock:
                current = self._cache.get(cache_key)
                if current is not None and not current.refreshing:
                    current.refreshing = True
                    self._refresh_later(cache_key=cache_key, builder=builder)
        return entry.payload

    def _refresh_later(self, *, cache_key: str, builder: Callable[[], Any]) -> None:
        thread = threading.Thread(
            target=self._refresh_cache_entry,
            kwargs={"cache_key": cache_key, "builder": builder},
            daemon=True,
        )
        thread.start()

    def _refresh_cache_entry(
        self, *, cache_key: str, builder: Callable[[], Any]
    ) -> None:
        try:
            payload = builder()
            loaded_at = time.monotonic()
            with self._cache_lock:
                self._cache[cache_key] = CacheEntry(
                    payload=payload,
                    loaded_at=loaded_at,
                )
        except Exception:
            with self._cache_lock:
                entry = self._cache.get(cache_key)
                if entry is not None:
                    entry.refreshing = False

    def attempt_key_for_slug(self, *, slug: str) -> AttemptKey | None:
        """Resolve a page slug using SQLite attempt metadata."""

        for progress in self.progress_attempts():
            if progress.slug() == slug:
                return AttemptKey(
                    doc_id=progress.doc_id,
                    doc_attempt=progress.doc_attempt,
                    task_name=progress.task_name,
                    model_id=progress.model_id,
                    selector_mode=progress.selector_mode,
                )
        for key in self.attempt_keys():
            if key.slug() == slug:
                return key
        return None

    def attempt_keys(self) -> list[AttemptKey]:
        """Return cached attempt keys from SQLite."""

        try:
            return self._cached(
                cache_key="attempt_keys",
                builder=self._attempt_keys_uncached,
                refresh_after_seconds=2.0,
            )
        except sqlite3.Error:
            return self._cached_payload(cache_key="attempt_keys") or []

    def _attempt_keys_uncached(self) -> list[AttemptKey]:
        """Return uncached attempt keys from SQLite."""

        return [
            attempt_key_from_row(row=row) for row in self.db.read_attempt_key_rows()
        ]

    def progress_attempts(self) -> list[ProgressAttemptView]:
        """Return progress snapshots from SQLite."""

        try:
            return self._cached(
                cache_key="progress_attempts",
                builder=self._progress_attempts_uncached,
                refresh_after_seconds=2.0,
            )
        except sqlite3.Error:
            return self._cached_payload(cache_key="progress_attempts") or []

    def _progress_attempts_uncached(self) -> list[ProgressAttemptView]:
        """Return uncached progress snapshots from SQLite."""

        rows = self.db.read_doc_progress_rows()
        if not rows:
            rows = self._node_attempt_progress_rows()
        return [progress_view_from_payload(payload=payload) for payload in rows]

    def _node_attempt_progress_rows(self) -> list[dict[str, Any]]:
        """Return fallback progress rows from typed run tables."""

        return self.db.read_node_attempt_progress_rows()

    def selected_progress_attempts(self) -> list[ProgressAttemptView]:
        """Return preferred progress attempts by document."""

        return selected_progress_attempts_by_doc(
            progress_attempts=self.progress_attempts()
        )

    def attempt_summaries(self) -> list[AttemptSummaryView]:
        """Return attempt summaries without replaying full event rows."""

        try:
            return self._cached(
                cache_key="attempt_summaries",
                builder=self._attempt_summaries_uncached,
                refresh_after_seconds=2.0,
            )
        except sqlite3.Error:
            return self._cached_payload(cache_key="attempt_summaries") or []

    def _attempt_summaries_uncached(self) -> list[AttemptSummaryView]:
        """Return uncached attempt summaries without replaying full event rows."""

        leaf_stats = leaf_stats_by_key(rows=self.db.read_leaf_summary_rows())
        return [
            attempt_summary_from_row(
                row=row,
                stats=leaf_stats.get(attempt_tuple_from_row(row=row), LeafStats()),
            )
            for row in self.db.read_attempt_summary_rows()
        ]

    def _cached_payload(self, *, cache_key: str) -> Any | None:
        with self._cache_lock:
            entry = self._cache.get(cache_key)
        return None if entry is None else entry.payload


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the dynamic visualization server."""

    parser = argparse.ArgumentParser(
        description="Serve reloadable branching visualizations from SQLite run dirs."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Run directory containing tree_events.sqlite. May be provided multiple times.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=[],
        help="Directory whose direct children are run directories.",
    )
    parser.add_argument(
        "--run-root-prefix",
        action="append",
        default=[],
        help="Only include child run roots whose names start with this prefix.",
    )
    parser.add_argument(
        "--latest-batch-only",
        action="store_true",
        help="For experiment roots, serve only the latest batch SQLite DB.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    if not args.run_dir and not args.run_root:
        parser.error("at least one --run-dir or --run-root is required")
    return args


def make_handler(*, registry: RunRegistry) -> type[BaseHTTPRequestHandler]:
    """Build a request handler bound to one or more run directories."""

    sources: dict[str, DynamicVizSource] = {}
    sources_lock = threading.Lock()
    summary_cache = RegistrySummaryCache()

    def source_for(entry: RunRegistryEntry) -> DynamicVizSource:
        cache_key = str(entry.run_dir)
        with sources_lock:
            source = sources.get(cache_key)
            if source is None:
                source = DynamicVizSource(run_dir=entry.run_dir)
                sources[cache_key] = source
            return source

    class BranchingVizHandler(BaseHTTPRequestHandler):
        """HTTP handler that reads each requested payload from SQLite."""

        def do_GET(self) -> None:
            try:
                response = route_registry_request(
                    registry=registry,
                    summary_cache=summary_cache,
                    source_for=source_for,
                    request_path=self.path,
                )
                if response is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_payload(response=response)
            except BrokenPipeError:
                return
            except sqlite3.Error as exc:
                self.send_error(
                    HTTPStatus.SERVICE_UNAVAILABLE,
                    explain=str(exc),
                )

        def _send_payload(self, *, response: ResponsePayload) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", response.content_type)
            self.send_header("Content-Length", str(len(response.body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(response.body)

    return BranchingVizHandler


def route_registry_request(
    *,
    registry: RunRegistry,
    summary_cache: RegistrySummaryCache,
    source_for: Callable[[RunRegistryEntry], DynamicVizSource],
    request_path: str,
) -> ResponsePayload | None:
    """Route a request through the run registry before run-local dispatch."""

    path = normalized_request_path(request_path=request_path)
    asset_response = route_asset_request(path=path)
    if asset_response is not None:
        return asset_response
    single_entry = registry.single_entry()
    if single_entry is not None:
        return route_request(source=source_for(single_entry), request_path=request_path)
    if path in {"", "/", "/index.html", "/runs", "/runs/"}:
        entries = registry.entries()
        summaries = summary_cache.summaries_for_entries(
            entries=entries,
            sync_missing_limit=1,
        )
        return html_response(
            html=run_registry_page_html(entries=entries, summaries=summaries)
        )
    entry, inner_path = split_run_route(registry=registry, path=path)
    if entry is None or inner_path is None:
        return None
    return route_request(source=source_for(entry), request_path=inner_path)


def route_request(
    *, source: DynamicVizSource, request_path: str
) -> ResponsePayload | None:
    """Route one HTTP GET to a dynamic SQLite-backed payload."""

    path = normalized_request_path(request_path=request_path)
    asset_response = route_asset_request(path=path)
    if asset_response is not None:
        return asset_response
    if path in {"", "/", "/index.html"}:
        return html_response(html=source.index_html())
    if path == "/summary.json":
        return json_response(payload=source.summary_payload())
    if path.startswith("/docs/data/"):
        return route_data_request(source=source, path=path)
    if path.startswith("/docs/") and path.endswith(".html"):
        html = source.attempt_html(
            slug=path.removeprefix("/docs/").removesuffix(".html")
        )
        return None if html is None else html_response(html=html)
    return None


def route_asset_request(*, path: str) -> ResponsePayload | None:
    """Return shared local viewer assets."""

    if path == "/assets/theme.css":
        return ResponsePayload(
            body=theme_css().encode("utf-8"),
            content_type="text/css; charset=utf-8",
        )
    if path == "/assets/tree_workspace.js":
        return ResponsePayload(
            body=tree_workspace_script().encode("utf-8"),
            content_type="application/javascript; charset=utf-8",
        )
    return None


def route_data_request(
    *, source: DynamicVizSource, path: str
) -> ResponsePayload | None:
    """Route one graph or node-detail JSON request."""

    relative = path.removeprefix("/docs/data/")
    event_marker = "/events/"
    trajectory_marker = "/trajectory.json"
    prompt_text_suffix = "/prompt.txt"
    if event_marker in relative and relative.endswith(prompt_text_suffix):
        raw_event_path = relative.rsplit(event_marker, maxsplit=1)[1]
        raw_event_index = raw_event_path.removesuffix(prompt_text_suffix)
        prompt_text = source.prompt_text(event_index=int(raw_event_index))
        return None if prompt_text is None else text_response(text=prompt_text)
    if event_marker in relative and relative.endswith(".json"):
        raw_event_index = relative.rsplit(event_marker, maxsplit=1)[1]
        event_index = int(raw_event_index.removesuffix(".json"))
        payload = source.event_payload(event_index=event_index)
        return None if payload is None else json_response(payload=payload)
    if relative.endswith(trajectory_marker):
        slug_node_path = relative.removesuffix(trajectory_marker)
        marker = "/nodes/"
        if marker not in slug_node_path:
            return None
        slug, node_id = slug_node_path.split(marker, maxsplit=1)
        payload = source.token_trajectory_payload(slug=slug, node_id=node_id)
        return None if payload is None else json_response(payload=payload)
    if relative.endswith(".json") and "/nodes/" not in relative:
        slug = relative.removesuffix(".json")
        payload = source.tree_payload(slug=slug)
        return None if payload is None else json_response(payload=payload)
    marker = "/nodes/"
    if marker not in relative or not relative.endswith(".json"):
        return None
    slug, raw_node_id = relative.split(marker, maxsplit=1)
    node_id = raw_node_id.removesuffix(".json")
    payload = source.node_payload(slug=slug, node_id=node_id)
    return None if payload is None else json_response(payload=payload)


def normalized_request_path(*, request_path: str) -> str:
    """Return a decoded URL path without query or fragment text."""

    parsed = urlparse(request_path)
    return unquote(parsed.path)


def html_response(*, html: str) -> ResponsePayload:
    """Build a UTF-8 HTML response."""

    return ResponsePayload(
        body=html.encode("utf-8"),
        content_type="text/html; charset=utf-8",
    )


def json_response(*, payload: dict[str, Any]) -> ResponsePayload:
    """Build a UTF-8 JSON response."""

    return ResponsePayload(
        body=(json.dumps(payload, indent=2) + "\n").encode("utf-8"),
        content_type="application/json; charset=utf-8",
    )


def text_response(*, text: str) -> ResponsePayload:
    """Build a UTF-8 plain-text response."""

    return ResponsePayload(
        body=text.encode("utf-8"),
        content_type="text/plain; charset=utf-8",
    )


def attempt_key_from_row(*, row: dict[str, Any]) -> AttemptKey:
    """Build an attempt key from one SQLite metadata row."""

    return AttemptKey(
        doc_id=int(row["doc_id"]),
        doc_attempt=int(row["doc_attempt"]),
        task_name=str(row["task_name"]),
        model_id=str(row["model_id"]),
        selector_mode=str(row["selector_mode"]),
    )


def progress_view_from_payload(*, payload: dict[str, Any]) -> ProgressAttemptView:
    """Build a progress view row from one compact snapshot payload."""

    return ProgressAttemptView(
        doc_id=int(payload.get("doc_id", 0)),
        doc_attempt=int(payload.get("doc_attempt", 0)),
        task_name=str(payload.get("task_name", "")),
        model_id=str(payload.get("model_id", "")),
        selector_mode=str(payload.get("selector_mode", "")),
        rollout_mode=str(payload.get("rollout_mode", "")),
        status=str(payload.get("status", "incomplete")),
        leaf_count=int(payload.get("leaf_count", 0)),
        passrate=float(payload.get("passrate", 0.0)),
        avg_token_length=float(payload.get("avg_token_length", 0.0)),
        correct_count=int(payload.get("correct_count", 0)),
        incorrect_count=int(payload.get("incorrect_count", 0)),
        natural_count=int(payload.get("natural_count", 0)),
        max_count=int(payload.get("max_count", 0)),
        repeating_count=int(payload.get("repeating_count", 0)),
        other_count=int(payload.get("other_count", 0)),
        unique_answer_count=int(payload.get("unique_answer_count", 0)),
        last_update_timestamp=str(payload.get("last_update_timestamp", "")),
    )


def attempt_tuple_from_row(*, row: dict[str, Any]) -> tuple[int, int, str, str, str]:
    """Return a hashable attempt key tuple from one SQLite row."""

    return (
        int(row["doc_id"]),
        int(row["doc_attempt"]),
        str(row["task_name"]),
        str(row["model_id"]),
        str(row["selector_mode"]),
    )


def leaf_stats_by_key(
    *, rows: list[dict[str, Any]]
) -> dict[tuple[int, int, str, str, str], LeafStats]:
    """Convert SQL leaf aggregate rows by attempt key."""

    stats_by_key: dict[tuple[int, int, str, str, str], LeafStats] = {}
    for row in rows:
        key = attempt_tuple_from_row(row=row)
        stats_by_key[key] = LeafStats(
            leaf_count_total=int(row.get("leaf_count", 0)),
            correct_count=int(row.get("correct_count", 0)),
            incorrect_count=int(row.get("incorrect_count", 0)),
            answer_correct_count=int(row.get("answer_correct_count", 0)),
            answer_count=int(row.get("answer_count", 0)),
            natural_count=int(row.get("natural_count", 0)),
            max_count=int(row.get("max_count", 0)),
            repeating_count=int(row.get("repeating_count", 0)),
            other_count=int(row.get("other_count", 0)),
            length_token_total=0,
            length_count=0,
            unique_answer_count=int(row.get("unique_answer_count", 0)),
        )
        stats_by_key[key].length_token_total = int(
            float(row.get("avg_token_length", 0.0))
            * max(1, int(row.get("leaf_count", 0)))
        )
        stats_by_key[key].length_count = int(row.get("leaf_count", 0))
    return stats_by_key


def update_leaf_stats(*, stats: LeafStats, payload: dict[str, Any]) -> None:
    """Update aggregate leaf stats from one `leaf_scored` payload."""

    stats.leaf_count_total += 1
    verification = payload.get("verification")
    if verification == 1:
        stats.correct_count += 1
    if verification == 0:
        stats.incorrect_count += 1
    raw_answer_acc = _bool_metric(
        value=_task_metric_value(payload=payload, key="raw_answer_acc")
    )
    if raw_answer_acc is not None:
        stats.answer_count += 1
        if raw_answer_acc:
            stats.answer_correct_count += 1
    update_stop_reason_counts(
        stats=stats, stop_reason=str(payload.get("stop_reason", ""))
    )
    length_value = payload.get("length_tokens_total")
    if isinstance(length_value, int):
        stats.length_token_total += length_value
        stats.length_count += 1
    text = str(payload.get("text", payload.get("text_preview", "")))
    if text:
        stats.unique_texts.add(text)


def _task_metric_value(*, payload: dict[str, Any], key: str) -> Any:
    metrics = payload.get("task_metrics", {})
    if not isinstance(metrics, dict):
        return None
    return metrics.get(key)


def update_stop_reason_counts(*, stats: LeafStats, stop_reason: str) -> None:
    """Update finish-reason buckets from one stop-reason label."""

    if stop_reason == "think_end":
        stats.natural_count += 1
    elif stop_reason == "length":
        stats.max_count += 1
    elif stop_reason == "repeated_exec_block_loop":
        stats.repeating_count += 1
    else:
        stats.other_count += 1


def _bool_metric(*, value: Any) -> bool | None:
    """Parse a metric value that may have been logged as text or number."""

    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0", "yes", "y"}:
        return True
    if normalized in {"false", "0", "0.0", "no", "n"}:
        return False
    return None


def attempt_summary_from_row(
    *, row: dict[str, Any], stats: LeafStats
) -> AttemptSummaryView:
    """Build an attempt summary from SQL aggregates and leaf stats."""

    return AttemptSummaryView(
        key=attempt_key_from_row(row=row),
        status=status_from_counts(row=row),
        correct_count=stats.correct_count,
        incorrect_count=stats.incorrect_count,
        answer_correct_count=stats.answer_correct_count,
        answer_count=stats.answer_count,
        leaf_count=stats.leaf_count(),
        avg_token_length=stats.avg_token_length(),
        natural_count=stats.natural_count,
        max_count=stats.max_count,
        repeating_count=stats.repeating_count,
        other_count=stats.other_count,
        unique_answer_count=stats.unique_answer_count or len(stats.unique_texts),
        event_count=int(row["event_count"]),
        last_event_index=int(row["last_event_index"]),
        last_timestamp_utc=str(row.get("last_timestamp_utc", "")),
    )


def status_from_counts(*, row: dict[str, Any]) -> str:
    """Return replay-compatible status from event-type aggregate counts."""

    if int(row.get("finished_count", 0)) > 0:
        return "completed"
    if int(row.get("started_count", 0)) > 0:
        return "incomplete"
    return "empty"


def selected_attempt_summaries_by_doc(
    *, summaries: list[AttemptSummaryView]
) -> list[AttemptSummaryView]:
    """Select one preferred aggregate attempt per doc."""

    selected_by_doc: dict[int, AttemptSummaryView] = {}
    for summary in summaries:
        existing = selected_by_doc.get(summary.key.doc_id)
        if existing is None or attempt_summary_rank(summary) > attempt_summary_rank(
            existing
        ):
            selected_by_doc[summary.key.doc_id] = summary
    return [selected_by_doc[doc_id] for doc_id in sorted(selected_by_doc)]


def attempt_summary_rank(summary: AttemptSummaryView) -> tuple[int, int, int]:
    """Return selection rank matching replay-selected attempt behavior."""

    return (
        1 if summary.status in {"complete", "completed"} else 0,
        summary.key.doc_attempt,
        summary.last_event_index,
    )


def attempt_page_html(*, run_dir: Path, page_data: AttemptPageData) -> str:
    """Return one per-attempt HTML page without writing sidecar files."""

    title = (
        f"{page_data.key.label()} · {page_data.key.task_name} · "
        f"{page_data.key.model_id} · {page_data.key.selector_mode}"
    )
    body_html = attempt_body_html(page_data=page_data, title=title)
    footer = f"Last event timestamp: {page_data.last_timestamp_utc or 'n/a'}"
    return wrap_page(
        title=title,
        subtitle=f"{run_dir.name} · SQLite typed event view",
        body_html=body_html,
        footer_text=footer,
        script_urls=("/assets/tree_workspace.js",),
        style_url="/assets/theme.css",
    )


def attempt_body_html(*, page_data: AttemptPageData, title: str) -> str:
    """Return the body markup shared by dynamic and static doc pages."""

    tree_html = render_tree_workspace(graph_path=f"data/{page_data.key.slug()}.json")
    status_class = status_badge_class(status=page_data.status)
    return f"""
<section class="panel">
  <h1>{escape(title)}</h1>
  <p class="muted" style="margin:0.45rem 0 0.9rem 0">
    Attempt view is loaded from typed <code>tree_events.sqlite</code> tables.
  </p>
  <div class="pill-row">
    <span class="pill {status_class}">{escape(page_data.status)}</span>
    <span class="pill">events={page_data.event_count}</span>
    <span class="pill">nodes={page_data.node_count}</span>
    <span class="pill">edges={page_data.edge_count}</span>
    <span class="pill">leaves={page_data.leaf_count}</span>
    <span class="pill">last_event_index={page_data.last_event_index}</span>
  </div>
</section>
{tree_html}
<section class="panel">
  <a href="../index.html">Back to run index</a>
</section>
"""


def index_page_html(
    *,
    run_dir: Path,
    event_count: int | None,
    attempt_summaries: list[AttemptSummaryView],
    selected_summaries: list[AttemptSummaryView],
    selected_progress_attempts: list[ProgressAttemptView],
    all_progress_attempts: list[ProgressAttemptView],
) -> str:
    """Return run-level index HTML."""

    body_html = index_body_html(
        run_dir=run_dir,
        event_count=event_count,
        attempt_summaries=attempt_summaries,
        selected_summaries=selected_summaries,
        selected_progress_attempts=selected_progress_attempts,
        all_progress_attempts=all_progress_attempts,
    )
    return wrap_page(
        title=f"Branching Replay · {run_dir.name}",
        subtitle="events-only replay",
        body_html=body_html,
        footer_text="Regenerated from tree_events.sqlite without tree snapshots.",
        style_url="/assets/theme.css",
    )


def loading_index_page_html(*, run_dir: Path) -> str:
    """Return a fast placeholder while a large live DB warms in the background."""

    body_html = f"""
<section class="panel">
  <h1>Run Replay: {escape(run_dir.name)}</h1>
  <p class="muted">
    Building the current run index from SQLite. Reload this page in a few seconds.
  </p>
  <div class="pill-row">
    <span class="pill">loading</span>
  </div>
</section>
"""
    return wrap_page(
        title=f"Branching Replay · {run_dir.name}",
        subtitle="events-only replay",
        body_html=body_html,
        footer_text="Regenerated from tree_events.sqlite without tree snapshots.",
        style_url="/assets/theme.css",
    )


def index_body_html(
    *,
    run_dir: Path,
    event_count: int | None,
    attempt_summaries: list[AttemptSummaryView],
    selected_summaries: list[AttemptSummaryView],
    selected_progress_attempts: list[ProgressAttemptView],
    all_progress_attempts: list[ProgressAttemptView],
) -> str:
    """Return index page body markup."""

    selected_rows = selected_index_rows_html(
        selected_summaries=selected_summaries,
        selected_progress_attempts=selected_progress_attempts,
    )
    all_rows = (
        all_progress_rows_html(
            progress_attempts=all_progress_attempts,
            summaries=attempt_summaries,
        )
        if all_progress_attempts
        else all_attempt_rows_html(summaries=attempt_summaries)
    )
    selected_count = len(selected_progress_attempts) or len(selected_summaries)
    attempt_count = len(all_progress_attempts) or len(attempt_summaries)
    passrate_summary = selected_problem_passrate_summary(
        selected_summaries=selected_summaries,
        selected_progress_attempts=selected_progress_attempts,
    )
    source_path = str(run_dir / EVENT_DB_FILENAME)
    event_count_text = "n/a" if event_count is None else str(event_count)
    return f"""
<section class="panel">
  <h1>Run Replay: {escape(run_dir.name)}</h1>
  <p class="muted path-row">
    <span class="path-label">Source of truth:</span>
    <code class="path-code" title="{escape(source_path)}">{escape(source_path)}</code>
  </p>
  <div class="pill-row">
    <span class="pill">events={event_count_text}</span>
    <span class="pill">attempts={attempt_count}</span>
    <span class="pill">selected_docs={selected_count}</span>
    <span class="pill">{escape(passrate_summary.pill_text())}</span>
  </div>
</section>
<section class="panel">
  <h2>Default Doc View (Latest Completed Attempt)</h2>
  <table>
    <thead>
      <tr><th>doc</th><th>attempt</th><th>status</th><th>mode</th><th>selector</th><th>passrate</th><th>avg_tok</th><th>correct/incorrect</th><th>unique_answers</th><th>view</th></tr>
    </thead>
    <tbody>{selected_rows}</tbody>
  </table>
</section>
<section class="panel">
  <h2>All Attempts</h2>
  <table>
    <thead>
      <tr><th>doc</th><th>attempt</th><th>status</th><th>task</th><th>model</th><th>selector</th><th>correct/incorrect</th><th>last_event</th><th>view</th></tr>
    </thead>
    <tbody>{all_rows}</tbody>
  </table>
</section>
"""


def selected_problem_passrate_summary(
    *,
    selected_summaries: list[AttemptSummaryView],
    selected_progress_attempts: list[ProgressAttemptView],
) -> ProblemPassrateSummary:
    """Average selected problem answer accuracy with equal weight per problem."""

    passrate_by_doc: dict[int, float] = {}
    for summary in selected_summaries:
        answer_accuracy = summary.answer_accuracy()
        if answer_accuracy is not None:
            passrate_by_doc[summary.key.doc_id] = answer_accuracy
    for progress in selected_progress_attempts:
        if progress.doc_id in passrate_by_doc or progress.leaf_count <= 0:
            continue
        passrate_by_doc[progress.doc_id] = progress.passrate
    passrates = list(passrate_by_doc.values())
    if not passrates:
        return ProblemPassrateSummary(average=None, problem_count=0)
    return ProblemPassrateSummary(
        average=sum(passrates) / len(passrates),
        problem_count=len(passrates),
    )


def selected_index_rows_html(
    *,
    selected_summaries: list[AttemptSummaryView],
    selected_progress_attempts: list[ProgressAttemptView],
) -> str:
    """Return selected-attempt index table rows."""

    summaries_by_key = attempt_summaries_by_key(summaries=selected_summaries)
    rows = "".join(
        index_selected_progress_row_html(
            progress=progress,
            summary=summaries_by_key.get(progress_attempt_tuple(progress=progress)),
        )
        for progress in selected_progress_attempts
    )
    if rows:
        return rows
    rows = "".join(
        index_selected_summary_row_html(summary=summary)
        for summary in selected_summaries
    )
    return (
        rows
        or "<tr><td colspan='9'>No completed or partial doc attempts found.</td></tr>"
    )


def index_selected_progress_row_html(
    *, progress: ProgressAttemptView, summary: AttemptSummaryView | None = None
) -> str:
    """Build one selected-doc row from progress data plus answer aggregates."""

    rel_link = f"docs/{progress.slug()}.html"
    status = progress.status
    counts_text = progress_answer_counts_text(progress=progress, summary=summary)
    return (
        "<tr>"
        f"<td>{progress.doc_id}</td>"
        f"<td>{progress.doc_attempt}</td>"
        f"<td><span class='pill {status_badge_class(status=status)}'>{escape(status)}</span></td>"
        f"<td><code>{escape(progress.rollout_mode)}</code></td>"
        f"<td><code>{escape(progress.selector_mode)}</code></td>"
        f"<td>{escape(_format_metric(progress.passrate))}</td>"
        f"<td>{escape(_format_metric(progress.avg_token_length))}</td>"
        f"<td>{counts_text}</td>"
        f"<td>{progress.unique_answer_count}</td>"
        f"<td><a href='{escape(rel_link)}'>open</a></td>"
        "</tr>"
    )


def all_attempt_rows_html(*, summaries: list[AttemptSummaryView]) -> str:
    """Return all-attempt index table rows."""

    rows = "".join(
        index_attempt_summary_row_html(summary=summary)
        for summary in sorted(summaries, key=attempt_summary_sort_key)
    )
    return rows or "<tr><td colspan='9'>No attempt events found.</td></tr>"


def all_progress_rows_html(
    *,
    progress_attempts: list[ProgressAttemptView],
    summaries: list[AttemptSummaryView],
) -> str:
    """Return all-attempt rows from typed progress snapshots."""

    summaries_by_key = attempt_summaries_by_key(summaries=summaries)
    rows = "".join(
        index_attempt_progress_row_html(
            progress=progress,
            summary=summaries_by_key.get(progress_attempt_tuple(progress=progress)),
        )
        for progress in sorted(progress_attempts, key=progress_sort_key)
    )
    return rows or "<tr><td colspan='9'>No attempt events found.</td></tr>"


def attempt_summaries_by_key(
    *, summaries: list[AttemptSummaryView]
) -> dict[tuple[int, int, str, str, str], AttemptSummaryView]:
    """Return attempt summaries keyed by attempt identity."""

    return {summary_attempt_tuple(summary=summary): summary for summary in summaries}


def summary_attempt_tuple(
    *, summary: AttemptSummaryView
) -> tuple[int, int, str, str, str]:
    """Return the hashable attempt identity for an aggregate summary."""

    return (
        summary.key.doc_id,
        summary.key.doc_attempt,
        summary.key.task_name,
        summary.key.model_id,
        summary.key.selector_mode,
    )


def progress_attempt_tuple(
    *, progress: ProgressAttemptView
) -> tuple[int, int, str, str, str]:
    """Return the hashable attempt identity for a progress row."""

    return (
        progress.doc_id,
        progress.doc_attempt,
        progress.task_name,
        progress.model_id,
        progress.selector_mode,
    )


def progress_sort_key(progress: ProgressAttemptView) -> tuple[int, int, str, str, str]:
    """Return stable all-attempt sort key for progress rows."""

    return (
        progress.doc_id,
        progress.doc_attempt,
        progress.task_name,
        progress.model_id,
        progress.selector_mode,
    )


def attempt_summary_sort_key(summary: AttemptSummaryView) -> tuple[int, int, int]:
    """Return stable all-attempt table sort key."""

    return (
        summary.key.doc_id,
        summary.key.doc_attempt,
        summary.last_event_index,
    )


def index_selected_summary_row_html(*, summary: AttemptSummaryView) -> str:
    """Build one selected-doc row from aggregate attempt data."""

    rel_link = f"docs/{summary.key.slug()}.html"
    status = summary.status
    return (
        "<tr>"
        f"<td>{summary.key.doc_id}</td>"
        f"<td>{summary.key.doc_attempt}</td>"
        f"<td><span class='pill {status_badge_class(status=status)}'>{escape(status)}</span></td>"
        f"<td><code>unknown</code></td>"
        f"<td><code>{escape(summary.key.selector_mode)}</code></td>"
        f"<td>-</td>"
        f"<td>-</td>"
        f"<td>{summary.answer_counts_text()}</td>"
        f"<td>-</td>"
        f"<td><a href='{escape(rel_link)}'>open</a></td>"
        "</tr>"
    )


def index_attempt_summary_row_html(*, summary: AttemptSummaryView) -> str:
    """Build one all-attempt row from aggregate attempt data."""

    rel_link = f"docs/{summary.key.slug()}.html"
    status = summary.status
    return (
        "<tr>"
        f"<td>{summary.key.doc_id}</td>"
        f"<td>{summary.key.doc_attempt}</td>"
        f"<td><span class='pill {status_badge_class(status=status)}'>{escape(status)}</span></td>"
        f"<td><code>{escape(summary.key.task_name)}</code></td>"
        f"<td><code>{escape(summary.key.model_id)}</code></td>"
        f"<td><code>{escape(summary.key.selector_mode)}</code></td>"
        f"<td>{summary.answer_counts_text()}</td>"
        f"<td>{summary.last_event_index}</td>"
        f"<td><a href='{escape(rel_link)}'>open</a></td>"
        "</tr>"
    )


def index_attempt_progress_row_html(
    *, progress: ProgressAttemptView, summary: AttemptSummaryView | None = None
) -> str:
    """Build one all-attempt row from typed progress data."""

    rel_link = f"docs/{progress.slug()}.html"
    status = progress.status
    counts_text = progress_answer_counts_text(progress=progress, summary=summary)
    return (
        "<tr>"
        f"<td>{progress.doc_id}</td>"
        f"<td>{progress.doc_attempt}</td>"
        f"<td><span class='pill {status_badge_class(status=status)}'>{escape(status)}</span></td>"
        f"<td><code>{escape(progress.task_name)}</code></td>"
        f"<td><code>{escape(progress.model_id)}</code></td>"
        f"<td><code>{escape(progress.selector_mode)}</code></td>"
        f"<td>{counts_text}</td>"
        f"<td>{escape(progress.last_update_timestamp)}</td>"
        f"<td><a href='{escape(rel_link)}'>open</a></td>"
        "</tr>"
    )


def progress_answer_counts_text(
    *, progress: ProgressAttemptView, summary: AttemptSummaryView | None
) -> str:
    """Return answer correctness counts for a progress row."""

    if summary is not None:
        return summary.answer_counts_text()
    return f"{progress.correct_count}/{progress.incorrect_count}"


def summary_payload_from_summaries(
    *,
    run_dir: Path,
    event_count: int,
    summaries: list[AttemptSummaryView],
    selected_summaries: list[AttemptSummaryView],
) -> dict[str, Any]:
    """Return machine-readable summary JSON from aggregate attempt data."""

    passrate_summary = selected_problem_passrate_summary(
        selected_summaries=selected_summaries,
        selected_progress_attempts=[],
    )
    return {
        "run_dir": str(run_dir),
        "event_count": event_count,
        "attempt_count": len(summaries),
        "selected_doc_count": len(selected_summaries),
        "avg_problem_answer_acc": passrate_summary.average,
        "problem_answer_acc_count": passrate_summary.problem_count,
        "avg_problem_passrate": passrate_summary.average,
        "problem_passrate_count": passrate_summary.problem_count,
        "selected_attempts": summary_rows_from_attempt_summaries(
            summaries=selected_summaries
        ),
    }


def progress_summary_payload(
    *,
    run_dir: Path,
    event_count: int,
    attempt_count: int,
    selected_progress_attempts: list[ProgressAttemptView],
    selected_summaries: list[AttemptSummaryView],
) -> dict[str, Any]:
    """Return summary JSON without replaying full event rows."""

    passrate_summary = selected_problem_passrate_summary(
        selected_summaries=selected_summaries,
        selected_progress_attempts=selected_progress_attempts,
    )
    return {
        "run_dir": str(run_dir),
        "event_count": event_count,
        "attempt_count": attempt_count,
        "selected_doc_count": len(selected_progress_attempts),
        "avg_problem_answer_acc": passrate_summary.average,
        "problem_answer_acc_count": passrate_summary.problem_count,
        "avg_problem_passrate": passrate_summary.average,
        "problem_passrate_count": passrate_summary.problem_count,
        "selected_attempts": progress_summary_rows(
            selected_progress_attempts=selected_progress_attempts,
            selected_summaries=selected_summaries,
        ),
    }


def progress_summary_rows(
    *,
    selected_progress_attempts: list[ProgressAttemptView],
    selected_summaries: list[AttemptSummaryView],
) -> list[dict[str, Any]]:
    """Return summary rows backed by doc-progress snapshots."""

    summaries_by_key = attempt_summaries_by_key(summaries=selected_summaries)
    return [
        {
            "doc_id": progress.doc_id,
            "doc_attempt": progress.doc_attempt,
            "task_name": progress.task_name,
            "model_id": progress.model_id,
            "selector_mode": progress.selector_mode,
            "rollout_mode": progress.rollout_mode,
            "status": progress.status,
            "leaf_count": progress.leaf_count,
            "passrate": progress.passrate,
            "avg_token_length": progress.avg_token_length,
            "correct_count": progress.correct_count,
            "incorrect_count": progress.incorrect_count,
            "answer_correct_count": progress_answer_correct_count(
                progress=progress,
                summary=summaries_by_key.get(progress_attempt_tuple(progress=progress)),
            ),
            "answer_count": progress_answer_count(
                progress=progress,
                summary=summaries_by_key.get(progress_attempt_tuple(progress=progress)),
            ),
            "natural_count": progress.natural_count,
            "max_count": progress.max_count,
            "repeating_count": progress.repeating_count,
            "other_count": progress.other_count,
            "unique_answer_count": progress.unique_answer_count,
            "last_update_timestamp": progress.last_update_timestamp,
        }
        for progress in selected_progress_attempts
    ]


def progress_answer_correct_count(
    *, progress: ProgressAttemptView, summary: AttemptSummaryView | None
) -> int:
    """Return raw-answer correct count for one progress row."""

    if summary is not None and summary.answer_count > 0:
        return summary.answer_correct_count
    return progress.correct_count


def progress_answer_count(
    *, progress: ProgressAttemptView, summary: AttemptSummaryView | None
) -> int:
    """Return raw-answer scored count for one progress row."""

    if summary is not None and summary.answer_count > 0:
        return summary.answer_count
    return progress.leaf_count


def summary_rows_from_attempt_summaries(
    *, summaries: list[AttemptSummaryView]
) -> list[dict[str, Any]]:
    """Return summary rows backed by aggregate attempt summaries."""

    return [
        {
            "doc_id": summary.key.doc_id,
            "doc_attempt": summary.key.doc_attempt,
            "task_name": summary.key.task_name,
            "model_id": summary.key.model_id,
            "selector_mode": summary.key.selector_mode,
            "status": summary.status,
            "leaf_count": summary.leaf_count,
            "correct_count": summary.correct_count,
            "incorrect_count": summary.incorrect_count,
            "answer_correct_count": summary.answer_correct_count,
            "answer_count": summary.answer_count,
            "avg_token_length": summary.avg_token_length,
            "unique_answer_count": summary.unique_answer_count,
            "event_count": summary.event_count,
            "last_event_index": summary.last_event_index,
        }
        for summary in summaries
    ]


def status_badge_class(*, status: str) -> str:
    """Return CSS class for a status pill."""

    if status in {"complete", "completed"}:
        return "good"
    if status in {"failed", "error"}:
        return "bad"
    return "warn"


def _format_metric(value: object) -> str:
    """Return compact metric text for index tables."""

    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    """Run the dynamic visualization server."""

    args = parse_args()
    registry = RunRegistry(
        run_dirs=args.run_dir,
        run_roots=args.run_root,
        run_root_prefixes=args.run_root_prefix,
        latest_batch_only=args.latest_batch_only,
    )
    if args.run_root:
        registry.warm_async()
    else:
        registry.warm()
    handler = make_handler(registry=registry)
    server = ReusableThreadingHTTPServer((args.host, args.port), handler)
    host, port = server.server_address[:2]
    single_entry = registry.single_entry()
    if single_entry is not None:
        print(f"Serving branching viz for {single_entry.run_dir}", flush=True)
    else:
        print("Serving branching viz run registry", flush=True)
    print(f"http://{host}:{port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping branching viz server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
