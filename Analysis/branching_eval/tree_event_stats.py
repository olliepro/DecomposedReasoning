"""High-level statistics for batch-scoped branching tree event logs."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from branching_eval.event_db import EVENT_DB_FILENAME, EventDatabase
from branching_eval.event_types import EventEnvelope, parse_event_row


@dataclass(frozen=True)
class DocSummary:
    """Summary for one document in a batch.

    Args:
        doc_id: Document identifier.
        started: Whether a `rollout_started` event was observed.
        finished: Whether a `rollout_finished` event was observed.
        leaves: Completed leaves observed so far.
        natural_leaves: Leaves that ended with `think_end`.
        unnatural_leaves: Leaves that ended for another reason.
        leaf_limit: Per-doc leaf cap if present in the events.

    Returns:
        Compact per-document progress summary.
    """

    doc_id: int
    started: bool
    finished: bool
    leaves: int
    natural_leaves: int
    unnatural_leaves: int
    leaf_limit: int | None

    def status(self) -> str:
        """Return the human-readable document status."""

        return "complete" if self.finished else "open"

    def fill_display(self) -> str:
        """Return the leaf-fill ratio formatted as a percent string."""

        if self.leaf_limit in (None, 0):
            return "n/a"
        return f"{100.0 * self.leaves / self.leaf_limit:.1f}%"


@dataclass(frozen=True)
class BatchSummary:
    """High-level stats for one batch event file.

    Args:
        path: Tree-event file path.
        event_count: Number of event rows read.
        doc_summaries: Per-document summaries.
        stop_reasons: Leaf stop-reason histogram.
        latest_timestamp_utc: Most recent event timestamp.
        latest_event_index: Most recent global event index.

    Returns:
        Compact batch summary for console rendering.
    """

    path: Path
    event_count: int
    doc_summaries: tuple[DocSummary, ...]
    stop_reasons: Counter[str]
    latest_timestamp_utc: str | None
    latest_event_index: int | None

    def started_docs(self) -> int:
        """Return count of docs that have started."""

        return sum(1 for item in self.doc_summaries if item.started)

    def finished_docs(self) -> int:
        """Return count of docs that have finished."""

        return sum(1 for item in self.doc_summaries if item.finished)

    def leaf_total(self) -> int:
        """Return total completed leaves."""

        return sum(item.leaves for item in self.doc_summaries)

    def natural_total(self) -> int:
        """Return total leaves that terminated naturally."""

        return sum(item.natural_leaves for item in self.doc_summaries)

    def unnatural_total(self) -> int:
        """Return total leaves that terminated unnaturally."""

        return sum(item.unnatural_leaves for item in self.doc_summaries)

    def leaf_capacity(self) -> int | None:
        """Return total leaf capacity if all started docs expose a cap."""

        limits = [item.leaf_limit for item in self.doc_summaries if item.started]
        if not limits or any(limit is None for limit in limits):
            return None
        return sum(limit for limit in limits if limit is not None)

    def open_docs(self) -> tuple[DocSummary, ...]:
        """Return the docs that have not yet finished."""

        return tuple(item for item in self.doc_summaries if not item.finished)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Track high-level stats for branching SQLite events."
    )
    parser.add_argument(
        "--tree-events-path",
        type=Path,
        default=None,
        help="Path to tree_events.sqlite.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory containing tree_events.sqlite.",
    )
    parser.add_argument(
        "--follow",
        action="store_true",
        help="Refresh the summary until interrupted.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=5.0,
        help="Polling interval when --follow is enabled.",
    )
    return parser.parse_args()


def load_events(*, path: Path) -> list[EventEnvelope]:
    """Load typed event envelopes from a run dir or SQLite DB."""

    raw_rows = load_event_rows(path=path)
    return [parse_event_row(row=row) for row in raw_rows]


def load_event_rows(*, path: Path) -> list[dict[str, object]]:
    """Load raw event rows from a run dir or SQLite DB."""

    if path.is_dir():
        return EventDatabase(path=path / EVENT_DB_FILENAME).read_event_rows()
    if path.name == EVENT_DB_FILENAME:
        return EventDatabase(path=path).read_event_rows()
    raise AssertionError(f"expected run dir or {EVENT_DB_FILENAME}: {path}")


def summarize_events(*, path: Path, events: Sequence[EventEnvelope]) -> BatchSummary:
    """Build high-level batch stats from one event stream."""

    started_docs: set[int] = set()
    finished_docs: set[int] = set()
    leaf_counts: Counter[int] = Counter()
    natural_counts: Counter[int] = Counter()
    unnatural_counts: Counter[int] = Counter()
    leaf_limits: dict[int, int] = {}
    stop_reasons: Counter[str] = Counter()
    latest_timestamp_utc: str | None = None
    latest_event_index: int | None = None
    for event in events:
        latest_timestamp_utc = event.timestamp_utc
        latest_event_index = event.event_index
        if event.doc_id is None:
            continue
        if event.event_type == "rollout_started":
            started_docs.add(event.doc_id)
            leaf_limit = event.payload.get("leaf_limit")
            if isinstance(leaf_limit, int):
                leaf_limits[event.doc_id] = leaf_limit
        elif event.event_type == "rollout_finished":
            finished_docs.add(event.doc_id)
        elif event.event_type == "leaf_completed":
            leaf_counts[event.doc_id] += 1
            stop_reason = str(event.payload.get("stop_reason", "unknown"))
            stop_reasons[stop_reason] += 1
            if stop_reason == "think_end":
                natural_counts[event.doc_id] += 1
            else:
                unnatural_counts[event.doc_id] += 1
    doc_ids = sorted(started_docs | finished_docs | set(leaf_counts) | set(leaf_limits))
    doc_summaries = tuple(
        DocSummary(
            doc_id=doc_id,
            started=doc_id in started_docs,
            finished=doc_id in finished_docs,
            leaves=leaf_counts[doc_id],
            natural_leaves=natural_counts[doc_id],
            unnatural_leaves=unnatural_counts[doc_id],
            leaf_limit=leaf_limits.get(doc_id),
        )
        for doc_id in doc_ids
    )
    return BatchSummary(
        path=path,
        event_count=len(events),
        doc_summaries=doc_summaries,
        stop_reasons=stop_reasons,
        latest_timestamp_utc=latest_timestamp_utc,
        latest_event_index=latest_event_index,
    )


def render_summary(*, summary: BatchSummary) -> str:
    """Render a batch summary as compact console text."""

    capacity = summary.leaf_capacity()
    fill = (
        "n/a"
        if capacity in (None, 0)
        else f"{100.0 * summary.leaf_total() / capacity:.1f}%"
    )
    lines = [
        f"file={summary.path}",
        (
            "events={events} started={started} finished={finished} open={open_docs} "
            "leaves={leaves} natural={natural} unnatural={unnatural} fill={fill}"
        ).format(
            events=summary.event_count,
            started=summary.started_docs(),
            finished=summary.finished_docs(),
            open_docs=len(summary.open_docs()),
            leaves=summary.leaf_total(),
            natural=summary.natural_total(),
            unnatural=summary.unnatural_total(),
            fill=fill,
        ),
        f"latest_event_index={summary.latest_event_index} latest_timestamp_utc={summary.latest_timestamp_utc}",
        "stop_reasons="
        + ", ".join(
            f"{name}={count}" for name, count in summary.stop_reasons.most_common()
        ),
        "",
        "doc | state | leaves | natural | unnatural | leaf_limit | fill",
        "---:|---|---:|---:|---:|---:|---:",
    ]
    for item in summary.doc_summaries:
        lines.append(
            "doc {doc_id} | {state} | {leaves} | {natural} | {unnatural} | {leaf_limit} | {fill}".format(
                doc_id=item.doc_id,
                state=item.status(),
                leaves=item.leaves,
                natural=item.natural_leaves,
                unnatural=item.unnatural_leaves,
                leaf_limit=item.leaf_limit if item.leaf_limit is not None else "n/a",
                fill=item.fill_display(),
            )
        )
    return "\n".join(lines)


def follow(*, path: Path, interval_seconds: float) -> None:
    """Print the batch summary repeatedly until interrupted."""

    while True:
        events = tuple(load_events(path=path))
        summary = summarize_events(path=path, events=events)
        os.system("clear")
        print(render_summary(summary=summary))
        time.sleep(interval_seconds)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    source_path = resolve_source_path(args=args)
    if args.follow:
        follow(path=source_path, interval_seconds=args.interval_seconds)
        return
    events = tuple(load_events(path=source_path))
    summary = summarize_events(path=source_path, events=events)
    print(render_summary(summary=summary))


def resolve_source_path(*, args: argparse.Namespace) -> Path:
    """Return the event source path selected by CLI args."""

    if args.run_dir is not None:
        return Path(args.run_dir)
    if args.tree_events_path is not None:
        return Path(args.tree_events_path)
    raise SystemExit("Provide --run-dir or --tree-events-path.")
