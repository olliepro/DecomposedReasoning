"""Typed event-schema utilities for canonical branching run logs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

EVENT_SCHEMA_VERSION = 2
FLOAT_QUANTIZATION_DIGITS = 4


@dataclass(frozen=True)
class EventContext:
    """Canonical labels attached to one event row.

    Args:
        run_id: Stable run identifier.
        doc_id: Document id for doc-scoped events, else `None`.
        doc_attempt: Attempt index for doc-scoped events, else `None`.
        task_name: Task name for attribution.
        model_id: Model id label.
        selector_mode: Selector mode label.

    Returns:
        Dataclass containing canonical event labels.
    """

    run_id: str
    doc_id: int | None
    doc_attempt: int | None
    task_name: str
    model_id: str
    selector_mode: str


@dataclass(frozen=True)
class EventEnvelope:
    """One versioned event row in `tree_events.jsonl`.

    Args:
        event_index: Monotonic global event index in this run log.
        event_version: Event schema version number.
        timestamp_utc: UTC timestamp for append time.
        run_id: Stable run identifier.
        doc_id: Document id for doc-scoped events, else `None`.
        doc_attempt: Attempt index for doc-scoped events, else `None`.
        task_name: Task name for attribution.
        model_id: Model id label.
        selector_mode: Selector mode label.
        event_type: Canonical event type label.
        payload: Event payload mapping.

    Returns:
        Dataclass containing one append-only event row.

    Example:
        >>> row = EventEnvelope(
        ...     event_index=0,
        ...     event_version=2,
        ...     timestamp_utc="2026-01-01T00:00:00+00:00",
        ...     run_id="run_abc",
        ...     doc_id=3,
        ...     doc_attempt=0,
        ...     task_name="aime24",
        ...     model_id="sft",
        ...     selector_mode="random",
        ...     event_type="doc_started",
        ...     payload={"node_count": 1},
        ... )
        >>> row.to_json_row()["event_type"]
        'doc_started'
    """

    event_index: int
    event_version: int
    timestamp_utc: str
    run_id: str
    doc_id: int | None
    doc_attempt: int | None
    task_name: str
    model_id: str
    selector_mode: str
    event_type: str
    payload: dict[str, Any]

    def to_json_row(self) -> dict[str, Any]:
        """Return JSON-serializable event mapping.

        Args:
            None.

        Returns:
            Event row mapping for JSONL append.
        """

        return {
            "event_index": self.event_index,
            "event_version": self.event_version,
            "timestamp_utc": self.timestamp_utc,
            "run_id": self.run_id,
            "doc_id": self.doc_id,
            "doc_attempt": self.doc_attempt,
            "task_name": self.task_name,
            "model_id": self.model_id,
            "selector_mode": self.selector_mode,
            "event_type": self.event_type,
            "payload": self.payload,
        }


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format.

    Args:
        None.

    Returns:
        UTC timestamp string.
    """

    return datetime.now(tz=timezone.utc).isoformat()


def quantize_floats(*, value: Any, digits: int = FLOAT_QUANTIZATION_DIGITS) -> Any:
    """Recursively round all float fields in a JSON-like object.

    Args:
        value: JSON-like object to normalize.
        digits: Decimal places to keep for all floats.

    Returns:
        Normalized object with rounded float leaves.
    """

    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {
            str(key): quantize_floats(value=item, digits=digits)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [quantize_floats(value=item, digits=digits) for item in value]
    if isinstance(value, tuple):
        return tuple(quantize_floats(value=item, digits=digits) for item in value)
    return value


def parse_event_row(*, row: dict[str, Any]) -> EventEnvelope:
    """Parse one event row mapping into typed event envelope.

    Args:
        row: Raw JSONL row mapping.

    Returns:
        Parsed event envelope.
    """

    raw_payload = row.get("payload", {})
    assert isinstance(raw_payload, dict), "event payload must be a mapping"
    return EventEnvelope(
        event_index=int(row["event_index"]),
        event_version=int(row["event_version"]),
        timestamp_utc=str(row["timestamp_utc"]),
        run_id=str(row["run_id"]),
        doc_id=int(row["doc_id"]) if row.get("doc_id") is not None else None,
        doc_attempt=(
            int(row["doc_attempt"]) if row.get("doc_attempt") is not None else None
        ),
        task_name=str(row["task_name"]),
        model_id=str(row["model_id"]),
        selector_mode=str(row["selector_mode"]),
        event_type=str(row["event_type"]),
        payload=raw_payload,
    )
