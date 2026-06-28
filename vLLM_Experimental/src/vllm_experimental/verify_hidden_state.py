"""Verify native hidden-state diversity benchmark artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from vllm_experimental.types import DEFAULT_SCRATCH_ROOT


@dataclass(frozen=True)
class HiddenStateVerification:
    """Evidence that native hidden-state diversity was used."""

    metric_rows: int
    promote_events: int
    hidden_vector_child_count: int
    max_pairwise_diversity: float
    max_selected_diversity: float


def jsonl_rows(*, path: Path) -> Iterable[dict[str, object]]:
    """Yield JSON objects from a JSONL file."""

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            assert isinstance(row, dict), f"JSONL row is not an object: {path}"
            yield row


def row_int(row: dict[str, object], key: str, default: int = 0) -> int:
    """Read an integer from a parsed JSON row."""

    return int(row.get(key, default))  # type: ignore[arg-type]


def row_float(row: dict[str, object], key: str, default: float = 0.0) -> float:
    """Read a float from a parsed JSON row."""

    return float(row.get(key, default))  # type: ignore[arg-type]


def verify_run(*, run_dir: Path) -> HiddenStateVerification:
    """Assert a run used native model-hidden-state diversity."""

    metrics_path = run_dir / "metrics.jsonl"
    assert metrics_path.exists(), f"missing metrics: {metrics_path}"
    metric_rows = list(jsonl_rows(path=metrics_path))
    assert metric_rows, f"empty metrics: {metrics_path}"
    hidden_rows = [
        row
        for row in metric_rows
        if row.get("diversity_vector_source") == "model_hidden_state"
        and row_int(row, "hidden_vector_child_count") > 0
        and row_float(row, "pool_hidden_pairwise_diversity") > 0.0
    ]
    assert hidden_rows, "metrics do not prove model-hidden-state diversity"

    promote_events = 0
    hidden_vector_child_count = 0
    pairwise_diversities: list[float] = []
    selected_diversities: list[float] = []
    for event_path in sorted(run_dir.glob("native_events_*.jsonl")):
        for event in jsonl_rows(path=event_path):
            if event.get("event") not in {"branch_promote", "branch_return"}:
                continue
            if event.get("diversity_vector_source") != "model_hidden_state":
                continue
            count = row_int(event, "hidden_vector_child_count")
            assert count > 0, f"hidden vector count missing in {event_path}"
            pairwise = row_float(event, "pool_hidden_pairwise_diversity")
            selected = row_float(event, "selected_hidden_diversity")
            assert pairwise > 0.0, f"pairwise diversity missing in {event_path}"
            promote_events += 1
            hidden_vector_child_count += count
            pairwise_diversities.append(pairwise)
            selected_diversities.append(selected)
    assert promote_events > 0, "native events do not prove hidden-state promotion"
    return HiddenStateVerification(
        metric_rows=len(metric_rows),
        promote_events=promote_events,
        hidden_vector_child_count=hidden_vector_child_count,
        max_pairwise_diversity=max(pairwise_diversities),
        max_selected_diversity=max(selected_diversities or [0.0]),
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--mode", default="eps_on_policy_diverse")
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_SCRATCH_ROOT / "benchmarks"
    )
    args = parser.parse_args(argv)
    result = verify_run(run_dir=args.root / args.run_name / args.mode)
    print(json.dumps(result.__dict__, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
