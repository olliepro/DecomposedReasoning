"""Shared path bootstrap for repo-local branching DAPO extensions."""

from __future__ import annotations

import sys
from pathlib import Path


def project_root() -> Path:
    """Return the repository root for local extension imports.

    Args:
        None.

    Returns:
        Absolute repository root path.
    """

    return Path(__file__).resolve().parents[2]


def ensure_repo_paths() -> None:
    """Add repo-local module roots for `verl` and `Analysis` imports.

    Args:
        None.

    Returns:
        None.
    """

    root = project_root()
    candidate_paths = (
        root / "RLTraining",
        root / "RLTraining" / "verl",
        root / "Analysis",
    )
    for candidate_path in reversed(candidate_paths):
        candidate_text = str(candidate_path)
        if candidate_text not in sys.path:
            sys.path.insert(0, candidate_text)
