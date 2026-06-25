#!/usr/bin/env python3
"""CLI wrapper for batch tree-events high-level stats."""

from __future__ import annotations

import sys
from pathlib import Path

ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from branching_eval.tree_event_stats import main

if __name__ == "__main__":
    main()
