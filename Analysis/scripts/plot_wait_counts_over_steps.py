"""Plot literal wait counts over saved RL step DBs."""

from __future__ import annotations

import argparse
import math
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

STEP_RE = re.compile(r"batch_(?P<batch>\d+)_step_(?P<step>\d+)$")
WAIT_RE = re.compile(r"\bwait\b", flags=re.IGNORECASE)
EXEC_RE = re.compile(
    r"<exec\b[^>]*>(?P<exec>.*?)</exec>", flags=re.IGNORECASE | re.DOTALL
)

RUN_ROOT = Path(
    "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/"
    "qwen35_branching_dapo/artifacts/"
    "qwen35_4b_branch_gs50_structured_baseline_gs50_struct_all_lr3e6_"
    "steer30_retry_5611096_20260616T222538Z"
)
OUTPUT_BASE = Path(
    "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/"
    "qwen35_branching_dapo/wait_counts"
)


@dataclass(frozen=True)
class StepDb:
    """One saved step DB with scored leaves."""

    step: int
    db_path: Path
    leaf_count: int


@dataclass(frozen=True)
class StepWaitStats:
    """Aggregated wait and exec-block metrics for one trainer step."""

    step: int
    leaf_count: int
    total_chars: int
    wait_count: int
    exec_block_count: int
    exec_block_chars: int

    @property
    def wait_per_char(self) -> float:
        return self.wait_count / self.total_chars

    @property
    def wait_per_million_chars(self) -> float:
        return self.wait_per_char * 1_000_000.0

    @property
    def mean_exec_block_chars(self) -> float:
        if self.exec_block_count == 0:
            return 0.0
        return self.exec_block_chars / self.exec_block_count

    @property
    def exec_chars_per_leaf(self) -> float:
        return self.exec_block_chars / self.leaf_count


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=RUN_ROOT)
    parser.add_argument("--output-base", type=Path, default=OUTPUT_BASE)
    parser.add_argument("--date-stamp", default=datetime.now().strftime("%Y%m%d"))
    return parser.parse_args()


def step_from_db_path(*, db_path: Path) -> int:
    """Extract trainer step from a batch-step DB path."""

    match = STEP_RE.fullmatch(db_path.parent.name)
    assert match is not None, f"unexpected step DB path: {db_path}"
    return int(match.group("step"))


def count_scored_leaves(*, db_path: Path) -> int:
    """Return leaf_score row count for one DB."""

    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        return int(connection.execute("SELECT COUNT(*) FROM leaf_score").fetchone()[0])


def collect_step_dbs(*, run_root: Path) -> list[StepDb]:
    """Collect saved, non-empty scored step DBs."""

    step_dbs: list[StepDb] = []
    for db_path in sorted(run_root.glob("batch_*_step_*/tree_events.sqlite")):
        if db_path.stat().st_size <= 0:
            continue
        leaf_count = count_scored_leaves(db_path=db_path)
        if leaf_count <= 0:
            continue
        step_dbs.append(
            StepDb(
                step=step_from_db_path(db_path=db_path),
                db_path=db_path,
                leaf_count=leaf_count,
            )
        )
    return sorted(step_dbs, key=lambda step_db: step_db.step)


def count_wait_instances(*, text: str) -> int:
    """Count whole-word wait instances with punctuation boundaries."""

    return len(WAIT_RE.findall(text))


def exec_block_char_count(*, text: str) -> tuple[int, int]:
    """Return exec block count and total inside-block character count."""

    matches = list(EXEC_RE.finditer(text))
    return len(matches), sum(len(match.group("exec")) for match in matches)


def summarize_texts(*, step: int, texts: Sequence[str]) -> StepWaitStats:
    """Aggregate wait and exec-block stats from leaf texts."""

    total_chars = 0
    wait_count = 0
    exec_block_count = 0
    exec_block_chars = 0
    for text in texts:
        total_chars += len(text)
        wait_count += count_wait_instances(text=text)
        block_count, block_chars = exec_block_char_count(text=text)
        exec_block_count += block_count
        exec_block_chars += block_chars
    assert total_chars > 0, f"step {step} had no text characters"
    return StepWaitStats(
        step=step,
        leaf_count=len(texts),
        total_chars=total_chars,
        wait_count=wait_count,
        exec_block_count=exec_block_count,
        exec_block_chars=exec_block_chars,
    )


def summarize_step_db(*, step_db: StepDb) -> StepWaitStats:
    """Read one DB and aggregate its leaf_score text."""

    with sqlite3.connect(f"file:{step_db.db_path}?mode=ro", uri=True) as connection:
        rows = connection.execute("SELECT text FROM leaf_score ORDER BY event_index")
        texts = [str(row[0] or "") for row in rows]
    assert len(texts) == step_db.leaf_count
    return summarize_texts(step=step_db.step, texts=texts)


def pearson_correlation(*, xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Return Pearson correlation, or None when undefined."""

    if len(xs) < 2 or len(ys) < 2:
        return None
    x_values = np.asarray(xs, dtype=np.float64)
    y_values = np.asarray(ys, dtype=np.float64)
    if float(np.std(x_values)) == 0.0 or float(np.std(y_values)) == 0.0:
        return None
    return float(np.corrcoef(x_values, y_values)[0, 1])


def regression_line(
    *, xs: Sequence[float], ys: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    """Return a simple least-squares line for plotting."""

    x_values = np.asarray(xs, dtype=np.float64)
    y_values = np.asarray(ys, dtype=np.float64)
    slope, intercept = np.polyfit(x_values, y_values, deg=1)
    line_xs = np.linspace(float(np.min(x_values)), float(np.max(x_values)), num=100)
    return line_xs, slope * line_xs + intercept


def plot_raw_wait_counts(*, stats: Sequence[StepWaitStats], output_path: Path) -> None:
    """Plot raw wait counts over trainer steps."""

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(
        [row.step for row in stats],
        [row.wait_count for row in stats],
        marker="o",
        linewidth=2,
    )
    ax.set_title('Raw Whole-Word "wait" Count Over Saved Steps')
    ax.set_xlabel("Trainer step")
    ax.set_ylabel('Total "wait" count across scored rollouts')
    ax.grid(True, alpha=0.25)
    save_figure(fig=fig, output_path=output_path)


def plot_normalized_wait_counts(
    *, stats: Sequence[StepWaitStats], output_path: Path
) -> None:
    """Plot normalized wait counts and mean exec-block characters over time."""

    fig, ax = plt.subplots(figsize=(13, 7))
    wait_line = ax.plot(
        [row.step for row in stats],
        [row.wait_per_million_chars for row in stats],
        marker="o",
        linewidth=2,
        label='"wait" per million chars',
    )
    exec_ax = ax.twinx()
    exec_line = exec_ax.plot(
        [row.step for row in stats],
        [row.mean_exec_block_chars for row in stats],
        marker="s",
        linewidth=2,
        color="tab:orange",
        label="mean exec-block chars",
    )
    ax.set_title('Whole-Word "wait" Rate and Mean Exec-Block Length')
    ax.set_xlabel("Trainer step")
    ax.set_ylabel('"wait" instances per million characters')
    exec_ax.set_ylabel("Mean characters per exec block")
    ax.grid(True, alpha=0.25)
    lines = wait_line + exec_line
    labels = [str(line.get_label()) for line in lines]
    ax.legend(lines, labels, loc="best")
    save_figure(fig=fig, output_path=output_path)


def plot_wait_vs_exec_length(
    *, stats: Sequence[StepWaitStats], output_path: Path
) -> None:
    """Plot normalized wait count against mean exec-block length."""

    xs = [row.mean_exec_block_chars for row in stats]
    ys = [row.wait_per_million_chars for row in stats]
    corr = pearson_correlation(xs=xs, ys=ys)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xs, ys, s=48, alpha=0.82)
    if corr is not None and len(set(xs)) > 1:
        line_xs, line_ys = regression_line(xs=xs, ys=ys)
        ax.plot(line_xs, line_ys, linestyle="--", linewidth=1.5)
    ax.set_title(f'"wait" Rate vs Mean Exec-Block Length (r={format_corr(corr=corr)})')
    ax.set_xlabel("Mean characters per exec block")
    ax.set_ylabel('"wait" instances per million characters')
    ax.grid(True, alpha=0.25)
    save_figure(fig=fig, output_path=output_path)


def save_figure(*, fig: Figure, output_path: Path) -> None:
    """Save and close a matplotlib figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def format_corr(*, corr: float | None) -> str:
    """Return printable correlation text."""

    return "n/a" if corr is None or math.isnan(corr) else f"{corr:.3f}"


def print_summary(
    *, run_root: Path, stats: Sequence[StepWaitStats], output_dir: Path
) -> None:
    """Print sampled-step and correlation summary."""

    total_wait = sum(row.wait_count for row in stats)
    total_chars = sum(row.total_chars for row in stats)
    total_exec_chars = sum(row.exec_block_chars for row in stats)
    total_exec_blocks = sum(row.exec_block_count for row in stats)
    raw_exec_corr = pearson_correlation(
        xs=[float(row.exec_block_chars) for row in stats],
        ys=[float(row.wait_count) for row in stats],
    )
    rate_mean_exec_corr = pearson_correlation(
        xs=[row.mean_exec_block_chars for row in stats],
        ys=[row.wait_per_million_chars for row in stats],
    )
    print(f"run_root={run_root}")
    print(f"steps={len(stats)} first={stats[0].step} last={stats[-1].step}")
    print(f"total_wait_count={total_wait}")
    print(f"total_chars={total_chars}")
    print(f"wait_per_million_chars={(total_wait / total_chars) * 1_000_000.0:.4f}")
    print(f"total_exec_blocks={total_exec_blocks}")
    print(f"total_exec_chars={total_exec_chars}")
    print(f"mean_exec_block_chars={total_exec_chars / total_exec_blocks:.2f}")
    print(f"corr_raw_wait_vs_total_exec_chars={format_corr(corr=raw_exec_corr)}")
    print(
        "corr_wait_rate_vs_mean_exec_block_chars="
        f"{format_corr(corr=rate_mean_exec_corr)}"
    )
    print(f"output_dir={output_dir}")


def main() -> None:
    """CLI entrypoint."""

    args = parse_args()
    step_dbs = collect_step_dbs(run_root=args.run_root)
    assert step_dbs, f"no scored step DBs found under {args.run_root}"
    stats = [summarize_step_db(step_db=step_db) for step_db in step_dbs]
    output_dir = args.output_base / args.date_stamp / args.run_root.name
    plot_raw_wait_counts(
        stats=stats,
        output_path=output_dir / "wait_raw_count_over_steps.png",
    )
    plot_normalized_wait_counts(
        stats=stats,
        output_path=output_dir / "wait_per_million_chars_over_steps.png",
    )
    plot_wait_vs_exec_length(
        stats=stats,
        output_path=output_dir / "wait_rate_vs_exec_block_length.png",
    )
    print_summary(run_root=args.run_root, stats=stats, output_dir=output_dir)


if __name__ == "__main__":
    main()
