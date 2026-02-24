"""One-time entropy threshold calibration from existing token-stat artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        None.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(
        description="Calibrate entropy threshold from token stats."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("Analysis/output"),
        help="Root directory containing branch_* run folders.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Analysis/branching_eval/calibration/entropy_thresholds.json"),
        help="Output JSON path for calibrated thresholds.",
    )
    parser.add_argument(
        "--profile-name",
        type=str,
        default="aime24_default",
        help="Calibration profile name key.",
    )
    return parser.parse_args()


def main() -> None:
    """Run calibration and write JSON output.

    Args:
        None.

    Returns:
        None.

    Example:
        >>> main()  # doctest: +SKIP
    """

    args = parse_args()
    entropies = collect_rollout_entropies(input_root=args.input_root)
    assert entropies, f"No entropy values found under {args.input_root}"
    threshold = percentile(values=entropies, percentile_value=95.0)
    payload = {
        args.profile_name: {
            "entropy_threshold": threshold,
            "percentile": 95.0,
            "source": str(args.input_root.resolve()),
            "sample_count": len(entropies),
        }
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


def collect_rollout_entropies(*, input_root: Path) -> list[float]:
    """Collect rollout-token entropies from branch run token-stat files.

    Args:
        input_root: Root directory containing `branch_*` outputs.

    Returns:
        List of entropy values.
    """

    values: list[float] = []
    for run_dir in sorted(input_root.glob("branch_*")):
        token_stats_path = run_dir / "token_stats.jsonl"
        if not token_stats_path.exists():
            continue
        values.extend(read_rollout_entropies(path=token_stats_path))
    return values


def read_rollout_entropies(*, path: Path) -> list[float]:
    """Read rollout token entropies from one JSONL file.

    Args:
        path: JSONL file path.

    Returns:
        Parsed entropy values.
    """

    entropies: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line_text = line.strip()
            if not line_text:
                continue
            row = json.loads(line_text)
            if str(row.get("source", "")) != "rollout":
                continue
            entropy = row.get("entropy")
            if isinstance(entropy, (int, float)):
                entropies.append(float(entropy))
    return entropies


def percentile(*, values: list[float], percentile_value: float) -> float:
    """Compute percentile using linear interpolation.

    Args:
        values: Numeric values.
        percentile_value: Percentile in `[0, 100]`.

    Returns:
        Percentile value.
    """

    assert values, "values must be non-empty"
    assert 0.0 <= percentile_value <= 100.0, "percentile must be in [0, 100]"
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (percentile_value / 100.0) * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    lower = ordered[lower_index]
    upper = ordered[upper_index]
    return float(lower + (upper - lower) * fraction)


if __name__ == "__main__":
    main()
