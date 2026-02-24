#!/usr/bin/env python3
"""Cleanup utility for branching-eval SLURM/launcher/nohup logs."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_PATTERNS = (
    "branching_eval/output/slurm_logs/*",
    "branching_eval/output/**/slurm_logs/*",
    "branching_eval/output/**/launcher_logs/slurm-*.out",
    "branching_eval/output/**/launcher_logs/slurm-*.err",
    "branching_eval/output/nohup_branch_*.log",
)


def parse_args() -> argparse.Namespace:
    """Parse CLI args for branching-eval log cleanup.

    Args:
        None.

    Returns:
        Parsed CLI namespace.

    Example:
        >>> args = parse_args()  # doctest: +SKIP
    """

    parser = argparse.ArgumentParser(
        description=(
            "Find/delete branching-eval log files only. " "Default mode is dry-run."
        )
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Analysis directory root that contains branching_eval/output.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Delete matched files (default: dry-run listing only).",
    )
    return parser.parse_args()


def find_log_files(*, analysis_root: Path) -> list[Path]:
    """Collect matching branching-eval log files from default glob patterns.

    Args:
        analysis_root: Root Analysis path.

    Returns:
        Sorted list of matched file paths.
    """

    matches: set[Path] = set()
    for pattern in DEFAULT_PATTERNS:
        for path in analysis_root.glob(pattern):
            if not path.is_file():
                continue
            matches.add(path.resolve())
    return sorted(matches)


def format_bytes(*, byte_count: int) -> str:
    """Format byte count in human-readable units."""

    if byte_count < 1024:
        return f"{byte_count} B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f} KiB"
    return f"{byte_count / (1024 * 1024):.2f} MiB"


def main() -> None:
    """Run cleanup workflow."""

    args = parse_args()
    analysis_root = args.analysis_root.resolve()
    files = find_log_files(analysis_root=analysis_root)
    total_bytes = sum(path.stat().st_size for path in files if path.exists())
    mode = "apply" if args.apply else "dry-run"
    print(f"[cleanup] mode={mode} matched_files={len(files)} bytes={total_bytes}")
    for path in files:
        print(str(path))
    if not args.apply:
        print("[cleanup] dry-run only; pass --apply to delete these files.")
        return
    deleted = 0
    for path in files:
        if not path.exists():
            continue
        path.unlink()
        deleted += 1
    print(
        "[cleanup] deleted "
        f"files={deleted} freed={format_bytes(byte_count=total_bytes)}"
    )


if __name__ == "__main__":
    main()
