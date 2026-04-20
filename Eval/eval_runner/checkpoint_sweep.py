"""Helpers for checkpoint eval sweeps and AIME-first ranking."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from typing import Sequence

REQUIRED_TASK_NAMES = ("aime24", "aime25", "minerva_math500")
AIME_TASK_NAMES = ("aime24", "aime25")
AVG_AT_K_PATTERN = re.compile(pattern=r"^avg_at_(\d+)$")


@dataclass(frozen=True)
class CheckpointTarget:
    """One checkpoint directory scheduled for eval.

    Args:
        path: Absolute or relative checkpoint directory path.
        label: Stable directory name used in output filenames.
        step: Numeric checkpoint step, or `None` for `final_model`.
        is_final_model: Whether this target is the final merged model dir.

    Returns:
        Dataclass describing one eval target.

    Example:
        >>> target = CheckpointTarget(
        ...     path=Path("checkpoint-56"),
        ...     label="checkpoint-56",
        ...     step=56,
        ...     is_final_model=False,
        ... )
        >>> target.sort_step()
        56
    """

    path: Path
    label: str
    step: int | None
    is_final_model: bool

    def sort_step(self) -> int:
        """Return a numeric step used for stable ordering.

        Args:
            None.

        Returns:
            Numeric step for sorting, with `final_model` placed last.
        """

        return self.step if self.step is not None else 10**18


@dataclass(frozen=True)
class CheckpointRankingEntry:
    """Resolved metrics and ranking keys for one checkpoint.

    Args:
        checkpoint_path: Checkpoint directory path.
        checkpoint_label: Checkpoint directory name.
        checkpoint_step: Numeric checkpoint step, or `None`.
        aime24_avg_at_k: Flattened AIME24 metric value.
        aime25_avg_at_k: Flattened AIME25 metric value.
        math500_math_verify: Flattened Math500 verify score.
        aime_mean: Mean of the two AIME metrics.
        metrics: Raw flattened metric mapping used for ranking.

    Returns:
        Dataclass describing one ranked checkpoint row.
    """

    checkpoint_path: str
    checkpoint_label: str
    checkpoint_step: int | None
    aime24_avg_at_k: float
    aime25_avg_at_k: float
    math500_math_verify: float
    aime_mean: float
    metrics: dict[str, float]


@dataclass(frozen=True)
class IncompleteCheckpoint:
    """One checkpoint missing eval outputs required for ranking.

    Args:
        checkpoint_label: Checkpoint directory name.
        checkpoint_path: Checkpoint directory path.
        missing_tasks: Eval task names that still have no result JSON.

    Returns:
        Dataclass describing one incomplete checkpoint sweep row.
    """

    checkpoint_label: str
    checkpoint_path: str
    missing_tasks: tuple[str, ...]


def extract_metric(task_payload: dict[str, Any], metric_names: Sequence[str]) -> float:
    """Extract a benchmark metric by exact or prefix match.

    Args:
        task_payload: Task payload from `results[task_name]`.
        metric_names: Metric names accepted for this benchmark.

    Returns:
        Extracted float metric value.
    """

    for metric_name in metric_names:
        for key, value in task_payload.items():
            if key == metric_name or key.startswith(f"{metric_name},"):
                return float(value)
    raise KeyError(f"None of metric candidates found: {tuple(metric_names)}")


def extract_avg_at_k_metric_name(task_payload: dict[str, Any]) -> str:
    """Extract the unique `avg_at_k` key from one AIME result payload.

    Args:
        task_payload: Task payload from `results[aime_task]`.

    Returns:
        Canonical `avg_at_<k>` metric name.
    """

    metric_names = {
        key.split(",")[0]
        for key in task_payload
        if AVG_AT_K_PATTERN.fullmatch(key.split(",")[0])
    }
    assert metric_names, "AIME payload is missing an `avg_at_<k>` metric."
    assert len(metric_names) == 1, "Expected exactly one `avg_at_<k>` metric in AIME payload."
    return next(iter(metric_names))


def flatten_benchmark_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Flatten raw `lm-eval` JSON into canonical benchmark metric keys.

    Args:
        payload: Parsed `lm-eval` aggregated result payload.

    Returns:
        Flat metric mapping used for ranking.
    """

    results = payload.get("results", {})
    assert isinstance(results, dict), "lm-eval payload missing `results` mapping."
    flattened_metrics: dict[str, float] = {}
    minerva_payload = results.get("minerva_math500")
    if isinstance(minerva_payload, dict):
        flattened_metrics["bench/math500/math_verify"] = extract_metric(
            task_payload=minerva_payload,
            metric_names=("math_verify",),
        )
    for task_name in AIME_TASK_NAMES:
        task_payload = results.get(task_name)
        if not isinstance(task_payload, dict):
            continue
        metric_name = extract_avg_at_k_metric_name(task_payload=task_payload)
        flattened_metrics[f"bench/{task_name}/{metric_name}"] = extract_metric(
            task_payload=task_payload,
            metric_names=(metric_name,),
        )
    assert flattened_metrics, "No recognized benchmark metrics found."
    return flattened_metrics


def load_and_flatten_metrics(result_json_path: Path) -> dict[str, float]:
    """Load and flatten benchmark metrics from one result JSON file.

    Args:
        result_json_path: Aggregated result JSON emitted by `lm-eval`.

    Returns:
        Flat benchmark metric mapping.
    """

    payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "Expected dict payload from lm-eval JSON."
    return flatten_benchmark_metrics(payload=payload)


def checkpoint_step_from_label(checkpoint_label: str) -> int | None:
    """Parse `checkpoint-<step>` names into integer steps.

    Args:
        checkpoint_label: Directory label to parse.

    Returns:
        Parsed integer step, or `None` when not applicable.
    """

    if not checkpoint_label.startswith("checkpoint-"):
        return None
    raw_step = checkpoint_label.removeprefix("checkpoint-")
    return int(raw_step) if raw_step.isdigit() else None


def discover_checkpoint_targets(run_dir: Path) -> list[CheckpointTarget]:
    """Discover raw HF checkpoint directories in stable numeric order.

    Args:
        run_dir: SFT output directory containing `checkpoint-*` folders.

    Returns:
        Ordered checkpoint targets including `final_model` when present.
    """

    targets: list[CheckpointTarget] = []
    for candidate in sorted(run_dir.iterdir(), key=lambda path: path.name):
        if not candidate.is_dir():
            continue
        if candidate.name == "final_model":
            targets.append(
                CheckpointTarget(
                    path=candidate,
                    label=candidate.name,
                    step=None,
                    is_final_model=True,
                )
            )
            continue
        step = checkpoint_step_from_label(checkpoint_label=candidate.name)
        if step is None:
            continue
        targets.append(
            CheckpointTarget(
                path=candidate,
                label=candidate.name,
                step=step,
                is_final_model=False,
            )
        )
    return sorted(targets, key=lambda target: (target.is_final_model, target.sort_step()))


def find_aime_metric(metrics: dict[str, float], task_name: str) -> float:
    """Return the single flattened AIME `avg_at_k` metric for one task.

    Args:
        metrics: Flattened benchmark metric mapping.
        task_name: Target AIME task name.

    Returns:
        Float metric value for the unique AIME `avg_at_k` metric.
    """

    aime_metrics = {
        key: value
        for key, value in metrics.items()
        if key.startswith(f"bench/{task_name}/avg_at_")
    }
    assert len(aime_metrics) == 1, f"Expected one {task_name} avg_at_k metric, found {sorted(aime_metrics)}"
    return next(iter(aime_metrics.values()))


def load_checkpoint_metrics(
    *,
    benchmark_dir: Path,
    checkpoint: CheckpointTarget,
    required_tasks: Sequence[str] = REQUIRED_TASK_NAMES,
) -> tuple[dict[str, float] | None, IncompleteCheckpoint | None]:
    """Load flattened metrics for one checkpoint when all required tasks exist.

    Args:
        benchmark_dir: Directory containing `*_task.json` outputs.
        checkpoint: Checkpoint target being ranked.
        required_tasks: Eval tasks required for ranking.

    Returns:
        Tuple of flattened metrics or incomplete metadata.
    """

    missing_tasks: list[str] = []
    merged_metrics: dict[str, float] = {}
    for task_name in required_tasks:
        result_path = benchmark_dir / f"{checkpoint.label}_{task_name}.json"
        if not result_path.exists():
            missing_tasks.append(task_name)
            continue
        merged_metrics.update(load_and_flatten_metrics(result_json_path=result_path))
    if missing_tasks:
        return None, IncompleteCheckpoint(
            checkpoint_label=checkpoint.label,
            checkpoint_path=str(checkpoint.path),
            missing_tasks=tuple(missing_tasks),
        )
    return merged_metrics, None


def build_ranking_entry(
    *, checkpoint: CheckpointTarget, metrics: dict[str, float]
) -> CheckpointRankingEntry:
    """Build the ranking row for one fully evaluated checkpoint.

    Args:
        checkpoint: Ranked checkpoint target.
        metrics: Flattened benchmark metric mapping.

    Returns:
        Ranking entry with precomputed AIME mean.
    """

    aime24_score = find_aime_metric(metrics=metrics, task_name="aime24")
    aime25_score = find_aime_metric(metrics=metrics, task_name="aime25")
    math500_score = float(metrics["bench/math500/math_verify"])
    aime_mean = (aime24_score + aime25_score) / 2.0
    return CheckpointRankingEntry(
        checkpoint_path=str(checkpoint.path),
        checkpoint_label=checkpoint.label,
        checkpoint_step=checkpoint.step,
        aime24_avg_at_k=aime24_score,
        aime25_avg_at_k=aime25_score,
        math500_math_verify=math500_score,
        aime_mean=aime_mean,
        metrics=metrics,
    )


def rank_checkpoint_entries(
    entries: Sequence[CheckpointRankingEntry],
) -> list[CheckpointRankingEntry]:
    """Sort checkpoint entries by AIME-first policy.

    Args:
        entries: Ranking entries for fully evaluated checkpoints.

    Returns:
        Entries sorted from best to worst.
    """

    return sorted(
        entries,
        key=lambda entry: (
            entry.aime_mean,
            entry.math500_math_verify,
            entry.checkpoint_step if entry.checkpoint_step is not None else 10**18,
        ),
        reverse=True,
    )


def collect_checkpoint_rankings(
    *,
    run_dir: Path,
    benchmark_dir_name: str = "benchmark_evals",
    required_tasks: Sequence[str] = REQUIRED_TASK_NAMES,
) -> tuple[list[CheckpointRankingEntry], list[IncompleteCheckpoint]]:
    """Collect ranked and incomplete checkpoint sweep rows for one run.

    Args:
        run_dir: SFT output directory containing checkpoint folders.
        benchmark_dir_name: Child directory containing eval JSON outputs.
        required_tasks: Eval tasks required for ranking.

    Returns:
        Tuple of ranked entries and incomplete checkpoint rows.
    """

    benchmark_dir = run_dir / benchmark_dir_name
    targets = discover_checkpoint_targets(run_dir=run_dir)
    ranking_entries: list[CheckpointRankingEntry] = []
    incomplete_entries: list[IncompleteCheckpoint] = []
    for checkpoint in targets:
        metrics, incomplete = load_checkpoint_metrics(
            benchmark_dir=benchmark_dir,
            checkpoint=checkpoint,
            required_tasks=required_tasks,
        )
        if incomplete is not None:
            incomplete_entries.append(incomplete)
            continue
        assert metrics is not None
        ranking_entries.append(build_ranking_entry(checkpoint=checkpoint, metrics=metrics))
    return rank_checkpoint_entries(entries=ranking_entries), incomplete_entries


def write_ranking_artifacts(
    *,
    run_dir: Path,
    benchmark_dir_name: str = "benchmark_evals",
    required_tasks: Sequence[str] = REQUIRED_TASK_NAMES,
) -> tuple[list[CheckpointRankingEntry], list[IncompleteCheckpoint]]:
    """Write JSON, CSV, and best-checkpoint artifacts for one sweep.

    Args:
        run_dir: SFT output directory containing checkpoints and eval outputs.
        benchmark_dir_name: Child directory containing eval JSON outputs.
        required_tasks: Eval tasks required for ranking.

    Returns:
        Ranked entries and incomplete rows that were written to disk.
    """

    ranking_entries, incomplete_entries = collect_checkpoint_rankings(
        run_dir=run_dir,
        benchmark_dir_name=benchmark_dir_name,
        required_tasks=required_tasks,
    )
    benchmark_dir = run_dir / benchmark_dir_name
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    ranking_json_path = benchmark_dir / "checkpoint_ranking.json"
    ranking_csv_path = benchmark_dir / "checkpoint_ranking.csv"
    incomplete_json_path = benchmark_dir / "checkpoint_ranking_incomplete.json"
    best_checkpoint_path = benchmark_dir / "best_checkpoint.txt"

    ranking_payload = [asdict(entry) for entry in ranking_entries]
    ranking_json_path.write_text(json.dumps(ranking_payload, indent=2), encoding="utf-8")
    incomplete_json_path.write_text(
        json.dumps([asdict(entry) for entry in incomplete_entries], indent=2),
        encoding="utf-8",
    )

    with ranking_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "rank",
                "checkpoint_label",
                "checkpoint_path",
                "checkpoint_step",
                "aime_mean",
                "aime24_avg_at_k",
                "aime25_avg_at_k",
                "math500_math_verify",
            ),
        )
        writer.writeheader()
        for rank, entry in enumerate(ranking_entries, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "checkpoint_label": entry.checkpoint_label,
                    "checkpoint_path": entry.checkpoint_path,
                    "checkpoint_step": entry.checkpoint_step,
                    "aime_mean": entry.aime_mean,
                    "aime24_avg_at_k": entry.aime24_avg_at_k,
                    "aime25_avg_at_k": entry.aime25_avg_at_k,
                    "math500_math_verify": entry.math500_math_verify,
                }
            )

    if ranking_entries:
        best_checkpoint_path.write_text(ranking_entries[0].checkpoint_path, encoding="utf-8")
    elif best_checkpoint_path.exists():
        best_checkpoint_path.unlink()

    return ranking_entries, incomplete_entries


def build_parser() -> argparse.ArgumentParser:
    """Build the checkpoint sweep CLI parser.

    Args:
        None.

    Returns:
        Argument parser for discovery and ranking subcommands.
    """

    parser = argparse.ArgumentParser(description="Checkpoint sweep helpers.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    discover_parser = subparsers.add_parser("discover", help="List raw HF checkpoints in sweep order.")
    discover_parser.add_argument("--run-dir", required=True, type=Path)

    rank_parser = subparsers.add_parser("rank", help="Rank evaluated checkpoints and emit artifacts.")
    rank_parser.add_argument("--run-dir", required=True, type=Path)
    rank_parser.add_argument("--benchmark-dir-name", default="benchmark_evals")

    return parser


def main() -> None:
    """Run the checkpoint sweep helper CLI.

    Args:
        None.

    Returns:
        None.
    """

    args = build_parser().parse_args()
    if args.command == "discover":
        for checkpoint in discover_checkpoint_targets(run_dir=args.run_dir):
            print(checkpoint.path)
        return

    ranking_entries, incomplete_entries = write_ranking_artifacts(
        run_dir=args.run_dir,
        benchmark_dir_name=args.benchmark_dir_name,
    )
    if ranking_entries:
        best_entry = ranking_entries[0]
        print(
            json.dumps(
                {
                    "best_checkpoint": best_entry.checkpoint_path,
                    "checkpoint_label": best_entry.checkpoint_label,
                    "aime_mean": best_entry.aime_mean,
                    "math500_math_verify": best_entry.math500_math_verify,
                    "incomplete_checkpoint_count": len(incomplete_entries),
                }
            )
        )
        return
    print(json.dumps({"best_checkpoint": None, "incomplete_checkpoint_count": len(incomplete_entries)}))


if __name__ == "__main__":
    main()
