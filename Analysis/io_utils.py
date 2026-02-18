"""I/O helpers for analysis artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from analysis_types import RunArtifactsIndex


def make_run_id(*, prefix: str = "run") -> str:
    """Create a UTC timestamp-based run identifier.

    Args:
        prefix: Prefix prepended to timestamp.

    Returns:
        Run identifier string.
    """
    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return f"{prefix}_{timestamp}"


def build_artifacts_index(*, output_root: Path, run_id: str) -> RunArtifactsIndex:
    """Build canonical artifact paths for one run.

    Args:
        output_root: Root output directory.
        run_id: Stable run identifier.

    Returns:
        `RunArtifactsIndex` for all run outputs.
    """
    run_dir = output_root / run_id
    return RunArtifactsIndex(
        run_id=run_id,
        run_dir=run_dir,
        config_path=run_dir / "config.json",
        steps_path=run_dir / "steps.jsonl",
        candidates_path=run_dir / "steer_candidates.jsonl",
        token_stats_path=run_dir / "token_stats.jsonl",
        chosen_path_log_path=run_dir / "chosen_path.log",
        report_path=run_dir / "report.html",
    )


def ensure_parent_dir(*, path: Path) -> None:
    """Create parent directory for given file path.

    Args:
        path: File path.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(*, path: Path, payload: dict[str, Any]) -> None:
    """Write one JSON mapping file.

    Args:
        path: Output path.
        payload: JSON payload mapping.

    Returns:
        None.
    """
    ensure_parent_dir(path=path)
    content = json.dumps(payload, ensure_ascii=False, indent=2)
    path.write_text(content + "\n", encoding="utf-8")


def append_jsonl(*, path: Path, payload: dict[str, Any]) -> None:
    """Append one row to JSONL file.

    Args:
        path: Output JSONL path.
        payload: Row payload.

    Returns:
        None.
    """
    ensure_parent_dir(path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def append_text(*, path: Path, content: str) -> None:
    """Append plain text content to a file.

    Args:
        path: Output text path.
        content: Text content to append.

    Returns:
        None.
    """
    ensure_parent_dir(path=path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)


def read_json(*, path: Path) -> dict[str, Any]:
    """Read JSON mapping file.

    Args:
        path: Input JSON path.

    Returns:
        Parsed JSON mapping.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "json root must be a mapping"
    return payload


def read_jsonl(*, path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows into memory.

    Args:
        path: Input JSONL path.

    Returns:
        List of parsed row mappings.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if isinstance(row, dict):
                rows.append(row)
    return rows
