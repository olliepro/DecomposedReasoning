"""Tests for report rebuilding from artifact files."""

from __future__ import annotations

import json
from pathlib import Path

from build_report import build_report, default_env_paths, project_root_dir


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON mapping for test fixture setup.

    Args:
        path: Output path.
        payload: JSON mapping payload.

    Returns:
        None.
    """
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    """Write JSONL rows for test fixture setup.

    Args:
        path: Output path.
        rows: Row payloads.

    Returns:
        None.
    """
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_build_report_from_run_dir(tmp_path: Path) -> None:
    """Report builder should produce `report.html` from artifact files."""
    write_json(
        path=tmp_path / "config.json",
        payload={
            "model": "m",
            "api_mode_config": {"default_mode": "completions"},
            "branch_factor": 100,
            "seed": 0,
        },
    )
    write_jsonl(path=tmp_path / "steps.jsonl", rows=[])
    write_jsonl(path=tmp_path / "steer_candidates.jsonl", rows=[])
    write_jsonl(path=tmp_path / "token_stats.jsonl", rows=[])
    write_json(path=tmp_path / "final_text.json", payload={"assistant_text": "ok"})
    output_path = build_report(run_dir=tmp_path, output_path=None)
    assert output_path.exists()
    report_html = output_path.read_text(encoding="utf-8")
    assert "id='report-data'" in report_html
    assert "JSON.parse" in report_html


def test_build_report_with_missing_optional_jsonl(tmp_path: Path) -> None:
    """Report build should succeed when optional candidate/token artifacts are absent."""
    write_json(
        path=tmp_path / "config.json",
        payload={
            "model": "m",
            "api_mode_config": {"default_mode": "completions"},
            "branch_factor": 100,
            "seed": 0,
        },
    )
    write_jsonl(path=tmp_path / "steps.jsonl", rows=[])
    write_json(path=tmp_path / "final_text.json", payload={"assistant_text": "ok"})
    output_path = build_report(run_dir=tmp_path, output_path=None)
    assert output_path.exists()
    report_html = output_path.read_text(encoding="utf-8")
    assert "id='report-data'" in report_html


def test_default_env_paths_prioritize_project_root_env(tmp_path: Path) -> None:
    """Default dotenv lookup should start at repo-root `.env`."""
    project_root = project_root_dir()
    paths = default_env_paths(run_dir=tmp_path)
    assert paths[0] == project_root / ".env"
