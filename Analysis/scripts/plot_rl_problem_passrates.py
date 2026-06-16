"""Plot prompt-weighted answer pass rates from RL branching SQLite step DBs."""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt

STEP_RE = re.compile(r"batch_(?P<batch>\d+)_step_(?P<step>\d+)$")


@dataclass(frozen=True)
class StepPassrate:
    """Per-step pass-rate summary read from one SQLite DB."""

    run_label: str
    step: int
    db_path: Path
    problem_count: int
    leaf_count: int
    problem_passrate_mean: float
    leaf_answer_reward_mean: float
    centered_problem_passrate_mean: float | None = None
    centered_leaf_answer_reward_mean: float | None = None

    def csv_row(self) -> dict[str, object]:
        """Return a CSV-serializable row."""

        return {
            "run_label": self.run_label,
            "step": self.step,
            "problem_count": self.problem_count,
            "leaf_count": self.leaf_count,
            "problem_passrate_mean": self.problem_passrate_mean,
            "leaf_answer_reward_mean": self.leaf_answer_reward_mean,
            "centered_problem_passrate_mean": self.centered_problem_passrate_mean,
            "centered_leaf_answer_reward_mean": self.centered_leaf_answer_reward_mean,
            "db_path": str(self.db_path),
        }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        action="append",
        type=Path,
        required=True,
        help="Artifact run root containing batch_*_step_* child DBs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the CSV and PNG outputs.",
    )
    parser.add_argument(
        "--name",
        default="problem_passrates",
        help="Output filename stem.",
    )
    parser.add_argument(
        "--center-window",
        type=int,
        default=1,
        help="Centered rolling window size. Use 1 for unsmoothed values.",
    )
    return parser.parse_args()


def step_from_db_path(*, db_path: Path) -> int:
    """Extract trainer step number from a step DB path."""

    match = STEP_RE.fullmatch(db_path.parent.name)
    assert match is not None, f"unexpected step DB parent: {db_path.parent}"
    return int(match.group("step"))


def run_label_from_root(*, run_root: Path) -> str:
    """Build a compact label from a verbose RL artifact root."""

    name = run_root.name
    size = "4B" if "qwen35_4b" in name else "2B" if "qwen35_2b" in name else "?"
    mode = "branching" if "_branching_" in name else "structured"
    return f"{size} {mode} {extract_job_id(name=name)}"


def extract_job_id(*, name: str) -> str:
    """Return the Slurm job id embedded in an artifact root name."""

    match = re.search(r"_(\d{7})_", name)
    return match.group(1) if match else "unknown"


def read_step_passrate(*, db_path: Path, run_label: str) -> StepPassrate | None:
    """Read prompt-weighted and leaf-weighted answer rates from one step DB."""

    query = """
        WITH leaf_rewards AS (
            SELECT
                leaf_score.doc_id,
                leaf_score.doc_attempt,
                CASE
                    WHEN answer_reward.metric_value IS NOT NULL
                        THEN answer_reward.metric_value
                    WHEN raw_answer.metric_value IS NOT NULL
                        THEN raw_answer.metric_value
                    WHEN LOWER(raw_answer.metric_text) IN ('true', '1', '1.0', 'yes', 'y')
                        THEN 1.0
                    ELSE 0.0
                END AS answer_reward
            FROM leaf_score
            LEFT JOIN leaf_metric AS answer_reward
              ON answer_reward.leaf_event_index = leaf_score.event_index
             AND answer_reward.metric_name = 'answer_reward'
            LEFT JOIN leaf_metric AS raw_answer
              ON raw_answer.leaf_event_index = leaf_score.event_index
             AND raw_answer.metric_name = 'raw_answer_acc'
        ),
        problem_rates AS (
            SELECT
                doc_id,
                doc_attempt,
                COUNT(*) AS leaf_count,
                AVG(answer_reward) AS passrate
            FROM leaf_rewards
            GROUP BY doc_id, doc_attempt
        )
        SELECT
            COUNT(*) AS problem_count,
            COALESCE(SUM(leaf_count), 0) AS leaf_count,
            AVG(passrate) AS problem_passrate_mean,
            (SELECT AVG(answer_reward) FROM leaf_rewards) AS leaf_answer_reward_mean
        FROM problem_rates
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        row = connection.execute(query).fetchone()
    if row is None or row[0] == 0 or row[2] is None or row[3] is None:
        return None
    return StepPassrate(
        run_label=run_label,
        step=step_from_db_path(db_path=db_path),
        db_path=db_path,
        problem_count=int(row[0]),
        leaf_count=int(row[1]),
        problem_passrate_mean=float(row[2]),
        leaf_answer_reward_mean=float(row[3]),
    )


def collect_run_points(*, run_root: Path) -> list[StepPassrate]:
    """Collect per-step pass-rate rows for one run root."""

    run_label = run_label_from_root(run_root=run_root)
    points: list[StepPassrate] = []
    for db_path in sorted(run_root.glob("batch_*_step_*/tree_events.sqlite")):
        point = read_step_passrate(db_path=db_path, run_label=run_label)
        if point is not None:
            points.append(point)
    return sorted(points, key=lambda point: point.step)


def centered_average(*, values: list[float], index: int, window: int) -> float:
    """Return centered rolling average using partial windows at series edges."""

    assert window > 0
    radius = window // 2
    start = max(index - radius, 0)
    stop = min(index + radius + 1, len(values))
    return sum(values[start:stop]) / (stop - start)


def with_centered_rolling(
    *, points: list[StepPassrate], window: int
) -> list[StepPassrate]:
    """Attach centered rolling means to each step row."""

    if window <= 1:
        return points
    smoothed: list[StepPassrate] = []
    for run_label in sorted({point.run_label for point in points}):
        run_points = [point for point in points if point.run_label == run_label]
        problem_values = [point.problem_passrate_mean for point in run_points]
        leaf_values = [point.leaf_answer_reward_mean for point in run_points]
        for index, point in enumerate(run_points):
            smoothed.append(
                StepPassrate(
                    run_label=point.run_label,
                    step=point.step,
                    db_path=point.db_path,
                    problem_count=point.problem_count,
                    leaf_count=point.leaf_count,
                    problem_passrate_mean=point.problem_passrate_mean,
                    leaf_answer_reward_mean=point.leaf_answer_reward_mean,
                    centered_problem_passrate_mean=centered_average(
                        values=problem_values, index=index, window=window
                    ),
                    centered_leaf_answer_reward_mean=centered_average(
                        values=leaf_values, index=index, window=window
                    ),
                )
            )
    return sorted(smoothed, key=lambda point: (point.run_label, point.step))


def write_csv(*, points: list[StepPassrate], output_path: Path) -> None:
    """Write pass-rate rows to CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(points[0].csv_row().keys()))
        writer.writeheader()
        writer.writerows(point.csv_row() for point in points)


def plot_points(
    *, points: list[StepPassrate], output_path: Path, center_window: int
) -> None:
    """Plot prompt-weighted pass rates and leaf-weighted reward means."""

    fig, ax = plt.subplots(figsize=(12, 7))
    for run_label in sorted({point.run_label for point in points}):
        run_points = [point for point in points if point.run_label == run_label]
        steps = [point.step for point in run_points]
        problem_rates = [
            (
                point.centered_problem_passrate_mean
                if point.centered_problem_passrate_mean is not None
                else point.problem_passrate_mean
            )
            for point in run_points
        ]
        leaf_rates = [
            (
                point.centered_leaf_answer_reward_mean
                if point.centered_leaf_answer_reward_mean is not None
                else point.leaf_answer_reward_mean
            )
            for point in run_points
        ]
        ax.plot(steps, problem_rates, marker="o", linewidth=2, label=run_label)
        ax.plot(steps, leaf_rates, linestyle="--", linewidth=1, alpha=0.55)
    title_suffix = f"Centered {center_window}-Step" if center_window > 1 else "Per-Step"
    ax.set_title(f"Running Qwen3.5 RL: {title_suffix} Mean of Problem Pass Rates")
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Answer pass rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    points = [
        point
        for run_root in args.run_root
        for point in collect_run_points(run_root=run_root)
    ]
    assert points, "no scored step DBs found"
    points = with_centered_rolling(points=points, window=args.center_window)
    csv_path = args.output_dir / f"{args.name}.csv"
    png_path = args.output_dir / f"{args.name}.png"
    write_csv(points=points, output_path=csv_path)
    plot_points(points=points, output_path=png_path, center_window=args.center_window)
    print(f"wrote {csv_path}")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
