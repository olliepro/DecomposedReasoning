from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Protocol


class SourceAuditPaths(Protocol):
    """Path subset needed for source-audit construction."""

    @property
    def raw_sample_path(self) -> Path:
        """Path to raw sampled rows."""
        ...

    @property
    def filtered_path(self) -> Path:
        """Path to filtered candidate rows."""
        ...

    @property
    def stratified_path(self) -> Path:
        """Path to stratified sampled rows."""
        ...


@dataclass(frozen=True)
class SourceAuditSummary:
    """Source-count summary for one pipeline stage."""

    row_count: int
    counts_by_dataset_label: dict[str, int]
    counts_by_dataset_source: dict[str, int]
    counts_by_original_dataset: dict[str, int]
    counts_by_source_family: dict[str, int]


def normalize_dataset_labels(value: object) -> tuple[str, ...]:
    """Normalize dataset labels into a canonical string tuple."""

    if isinstance(value, str):
        labels = [value]
    elif isinstance(value, list):
        labels = [str(item) for item in value]
    else:
        labels = []
    return tuple(sorted(label.strip().lower() for label in labels if label))


def normalize_source_family(row: dict[str, object]) -> str:
    """Resolve a stable source-family label from a raw dataset row."""

    raw_value = str(row.get("original_dataset") or row.get("dataset_source") or "unknown")
    name = raw_value.split("/")[-1]
    name = re.sub(pattern=r"(_filtered|_cleaned)$", repl="", string=name, flags=re.IGNORECASE)
    return name or "unknown"


def audit_rows(rows: Iterable[dict[str, object]]) -> SourceAuditSummary:
    """Summarize source composition across an iterable of rows."""

    row_count = 0
    counts_by_dataset_label: Counter[str] = Counter()
    counts_by_dataset_source: Counter[str] = Counter()
    counts_by_original_dataset: Counter[str] = Counter()
    counts_by_source_family: Counter[str] = Counter()
    for row in rows:
        row_count += 1
        dataset_labels = normalize_dataset_labels(value=row.get("dataset"))
        counts_by_dataset_label["|".join(dataset_labels) or "unknown"] += 1
        counts_by_dataset_source[str(row.get("dataset_source", "unknown"))] += 1
        counts_by_original_dataset[str(row.get("original_dataset", "unknown"))] += 1
        counts_by_source_family[normalize_source_family(row=row)] += 1
    return SourceAuditSummary(
        row_count=row_count,
        counts_by_dataset_label=dict(counts_by_dataset_label),
        counts_by_dataset_source=dict(counts_by_dataset_source),
        counts_by_original_dataset=dict(counts_by_original_dataset),
        counts_by_source_family=dict(counts_by_source_family),
    )


def build_source_audit_payload(
    *,
    paths: SourceAuditPaths,
    iter_jsonl_fn: Callable[[Path], Iterable[dict[str, object]]],
) -> dict[str, object]:
    """Build source-audit payload across pipeline stages."""

    audit_payload: dict[str, object] = {}
    path_by_stage = {
        "sample": paths.raw_sample_path,
        "filter": paths.filtered_path,
        "stratify": paths.stratified_path,
    }
    for stage_name, stage_path in path_by_stage.items():
        if not stage_path.exists():
            continue
        audit_payload[stage_name] = asdict(audit_rows(rows=iter_jsonl_fn(stage_path)))
    return audit_payload
