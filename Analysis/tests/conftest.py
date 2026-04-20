"""Pytest path setup for Analysis tests.

This keeps `scripts.*` imports working when the test runner does not
automatically place the `Analysis` directory on `sys.path`.
"""

from __future__ import annotations

import sys
from pathlib import Path


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))
