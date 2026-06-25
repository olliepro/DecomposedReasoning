"""Plot RL response lengths and curated failure types over saved step DBs."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

STEP_RE = re.compile(r"batch_(?P<batch>\d+)_step_(?P<step>\d+)$")

DEFAULT_RUN_PATH = Path(
    "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/"
    "qwen35_branching_dapo/artifacts/"
    "qwen35_4b_branch_gs50_structured_baseline_gs50_struct_all_lr3e6_"
    "steer30_repcheck_on_retry_5611096_20260618T215954Z/"
    "batch_0399_step_000400"
)
DEFAULT_OUTPUT_BASE = Path(
    "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/"
    "qwen35_branching_dapo/response_failure_analysis"
)

TRUNCATION_REASONS = {
    "max_gen_toks_reached",
    "post_think_toks_reached",
    "max_context_length_reached",
}
TAG_STRUCTURE_ISSUES = {
    "steer_exec_outside_think_block",
    "non_whitespace_outside_steer_exec",
    "unequal_steer_exec_block_counts",
    "unpaired_steer_exec_tags",
    "empty_steer_exec_block",
    "missing_steer_exec_blocks",
    "steer_exec_not_interleaved",
}
THINK_ENVELOPE_ISSUES = {
    "expected_single_complete_think_block",
    "empty_think_block",
}

FAILURE_ORDER = [
    "Wrong boxed answer",
    "Missing boxed answer",
    "Malformed steer/think boundary",
    "Length limit / truncation",
    "Tag structure violation",
    "Other incorrect",
]
SIGNAL_ORDER = [
    "Wrong boxed answer",
    "Missing boxed answer",
    "Malformed steer/think boundary",
    "Length limit / truncation",
    "Think envelope malformed",
    "Steer/exec tag structure",
]


@dataclass(frozen=True)
class LeafRecord:
    """One scored rollout leaf."""

    step: int
    verification: bool
    length_tokens_total: int
    stop_reason: str
    issues: tuple[str, ...]
    boxed_present: bool
    answer_acc: bool
    format_valid: bool


@dataclass(frozen=True)
class StepSummary:
    """Per-step aggregate plotted to CSV and figures."""

    step: int
    leaf_count: int
    correct_count: int
    incorrect_count: int
    avg_correct_tokens: float
    avg_incorrect_tokens: float
    primary_failures: dict[str, int]
    failure_signals: dict[str, int]

    @property
    def passrate(self) -> float:
        return self.correct_count / self.leaf_count


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-path", type=Path, default=DEFAULT_RUN_PATH)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--date-stamp", default=datetime.now().strftime("%Y%m%d"))
    return parser.parse_args()


def resolve_run_root(*, run_path: Path) -> Path:
    """Accept a run root, batch directory, or tree_events.sqlite path."""

    if run_path.name == "tree_events.sqlite":
        run_path = run_path.parent
    if STEP_RE.fullmatch(run_path.name):
        return run_path.parent
    return run_path


def step_from_db_path(*, db_path: Path) -> int:
    """Extract trainer step from a tree_events.sqlite path."""

    match = STEP_RE.fullmatch(db_path.parent.name)
    assert match is not None, f"unexpected step DB path: {db_path}"
    return int(match.group("step"))


def parse_bool_text(value: object) -> bool:
    """Parse metric text booleans."""

    return str(value).strip().lower() == "true"


def parse_issues(value: object) -> tuple[str, ...]:
    """Parse the structure_issues metric into individual issue names."""

    if value is None or value == "":
        return ()
    try:
        parsed = ast.literal_eval(str(value))
    except (SyntaxError, ValueError):
        return (str(value),)
    if isinstance(parsed, list):
        return tuple(str(item) for item in parsed)
    return (str(parsed),)


def connect_readonly(*, db_path: Path) -> sqlite3.Connection:
    """Open a retained DB snapshot without taking locks."""

    return sqlite3.connect(f"file:{db_path}?mode=ro&immutable=1", uri=True)


def iter_step_dbs(*, run_root: Path) -> list[Path]:
    """Return saved step DBs that have scored leaves."""

    db_paths: list[Path] = []
    for db_path in sorted(run_root.glob("batch_*_step_*/tree_events.sqlite")):
        if db_path.stat().st_size <= 0:
            continue
        with connect_readonly(db_path=db_path) as connection:
            leaf_count = int(
                connection.execute("SELECT COUNT(*) FROM leaf_score").fetchone()[0]
            )
        if leaf_count > 0:
            db_paths.append(db_path)
    return sorted(db_paths, key=lambda path: step_from_db_path(db_path=path))


def read_leaves(*, db_path: Path) -> list[LeafRecord]:
    """Read scored leaves and the relevant reward/format metrics from one DB."""

    step = step_from_db_path(db_path=db_path)
    query = """
        SELECT
            ls.verification,
            ls.length_tokens_total,
            ls.stop_reason,
            MAX(CASE WHEN lm.metric_name = 'structure_issues'
                THEN lm.metric_text END) AS structure_issues,
            MAX(CASE WHEN lm.metric_name = 'boxed_present'
                THEN lm.metric_text END) AS boxed_present,
            MAX(CASE WHEN lm.metric_name = 'answer_acc'
                THEN lm.metric_text END) AS answer_acc,
            MAX(CASE WHEN lm.metric_name = 'format_valid'
                THEN lm.metric_text END) AS format_valid
        FROM leaf_score AS ls
        LEFT JOIN leaf_metric AS lm ON lm.leaf_event_index = ls.event_index
        GROUP BY ls.event_index
        ORDER BY ls.event_index
    """
    with connect_readonly(db_path=db_path) as connection:
        rows = connection.execute(query).fetchall()
    return [
        LeafRecord(
            step=step,
            verification=bool(row[0]),
            length_tokens_total=int(row[1] or 0),
            stop_reason=str(row[2] or ""),
            issues=parse_issues(row[3]),
            boxed_present=parse_bool_text(row[4]),
            answer_acc=parse_bool_text(row[5]),
            format_valid=parse_bool_text(row[6]),
        )
        for row in rows
    ]


def primary_failure_category(*, leaf: LeafRecord) -> str:
    """Return one curated primary failure category for an incorrect leaf."""

    if leaf.verification:
        return "Correct"
    issue_set = set(leaf.issues)
    if leaf.stop_reason in TRUNCATION_REASONS:
        return "Length limit / truncation"
    if leaf.stop_reason == "missing_steer_or_think_close":
        return "Malformed steer/think boundary"
    if "missing_boxed_answer" in issue_set or not leaf.boxed_present:
        return "Missing boxed answer"
    if leaf.boxed_present and not leaf.answer_acc:
        return "Wrong boxed answer"
    if (
        issue_set & (TAG_STRUCTURE_ISSUES | THINK_ENVELOPE_ISSUES)
        or not leaf.format_valid
    ):
        return "Tag structure violation"
    return "Other incorrect"


def failure_signals_for_leaf(*, leaf: LeafRecord) -> set[str]:
    """Return curated failure signals; signals may overlap."""

    issue_set = set(leaf.issues)
    signals: set[str] = set()
    if not leaf.verification and leaf.boxed_present and not leaf.answer_acc:
        signals.add("Wrong boxed answer")
    if "missing_boxed_answer" in issue_set or not leaf.boxed_present:
        signals.add("Missing boxed answer")
    if leaf.stop_reason == "missing_steer_or_think_close":
        signals.add("Malformed steer/think boundary")
    if leaf.stop_reason in TRUNCATION_REASONS:
        signals.add("Length limit / truncation")
    if issue_set & THINK_ENVELOPE_ISSUES:
        signals.add("Think envelope malformed")
    if issue_set & TAG_STRUCTURE_ISSUES:
        signals.add("Steer/exec tag structure")
    return signals


def mean_or_nan(values: Sequence[int]) -> float:
    """Return the mean length or NaN when there are no values."""

    if len(values) == 0:
        return math.nan
    return sum(values) / len(values)


def summarize_step(*, leaves: Sequence[LeafRecord]) -> StepSummary:
    """Aggregate one trainer step."""

    assert len(leaves) > 0
    step = leaves[0].step
    correct_lengths = [leaf.length_tokens_total for leaf in leaves if leaf.verification]
    incorrect_lengths = [
        leaf.length_tokens_total for leaf in leaves if not leaf.verification
    ]
    primary = Counter(
        primary_failure_category(leaf=leaf) for leaf in leaves if not leaf.verification
    )
    signals: Counter[str] = Counter()
    for leaf in leaves:
        signals.update(failure_signals_for_leaf(leaf=leaf))
    return StepSummary(
        step=step,
        leaf_count=len(leaves),
        correct_count=len(correct_lengths),
        incorrect_count=len(incorrect_lengths),
        avg_correct_tokens=mean_or_nan(correct_lengths),
        avg_incorrect_tokens=mean_or_nan(incorrect_lengths),
        primary_failures={name: int(primary.get(name, 0)) for name in FAILURE_ORDER},
        failure_signals={name: int(signals.get(name, 0)) for name in SIGNAL_ORDER},
    )


def save_figure(*, fig: Figure, output_path: Path) -> None:
    """Save a figure with consistent settings."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_response_lengths(
    *, summaries: Sequence[StepSummary], output_path: Path
) -> None:
    """Plot average response-token lengths for correct and incorrect leaves."""

    fig, ax_object = plt.subplots(figsize=(13, 7))
    ax = cast(Axes, ax_object)
    steps = [row.step for row in summaries]
    ax.plot(
        steps,
        [row.avg_correct_tokens for row in summaries],
        marker="o",
        linewidth=2,
        label="Correct responses",
    )
    ax.plot(
        steps,
        [row.avg_incorrect_tokens for row in summaries],
        marker="o",
        linewidth=2,
        label="Incorrect responses",
    )
    ax.set_title("Average Response Length by Correctness")
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Mean response length (tokens)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    save_figure(fig=fig, output_path=output_path)


def plot_primary_failures(
    *, summaries: Sequence[StepSummary], output_path: Path
) -> None:
    """Plot mutually exclusive primary failure counts over time."""

    fig, ax_object = plt.subplots(figsize=(13, 7))
    ax = cast(Axes, ax_object)
    steps = [row.step for row in summaries]
    stacks = [
        [row.primary_failures.get(category, 0) for row in summaries]
        for category in FAILURE_ORDER
    ]
    ax.stackplot(steps, stacks, labels=FAILURE_ORDER, alpha=0.85)
    ax.set_title("Primary Failure Type Counts Over Time")
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Incorrect rollout count")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2)
    save_figure(fig=fig, output_path=output_path)


def plot_failure_signals(
    *, summaries: Sequence[StepSummary], output_path: Path
) -> None:
    """Plot curated overlapping failure-signal counts over time."""

    fig, ax_object = plt.subplots(figsize=(13, 7))
    ax = cast(Axes, ax_object)
    steps = [row.step for row in summaries]
    for category in SIGNAL_ORDER:
        ax.plot(
            steps,
            [row.failure_signals.get(category, 0) for row in summaries],
            marker="o",
            linewidth=2,
            label=category,
        )
    ax.set_title("Curated Failure Signals Over Time")
    ax.set_xlabel("Trainer step")
    ax.set_ylabel("Rollout count; signals may overlap")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncol=2)
    save_figure(fig=fig, output_path=output_path)


def write_step_summary_csv(
    *, summaries: Sequence[StepSummary], output_path: Path
) -> None:
    """Write per-step summary data."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "step",
        "leaf_count",
        "correct_count",
        "incorrect_count",
        "passrate",
        "avg_correct_tokens",
        "avg_incorrect_tokens",
        *[f"primary_{name}" for name in FAILURE_ORDER],
        *[f"signal_{name}" for name in SIGNAL_ORDER],
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            record = {
                "step": row.step,
                "leaf_count": row.leaf_count,
                "correct_count": row.correct_count,
                "incorrect_count": row.incorrect_count,
                "passrate": f"{row.passrate:.6f}",
                "avg_correct_tokens": f"{row.avg_correct_tokens:.6f}",
                "avg_incorrect_tokens": f"{row.avg_incorrect_tokens:.6f}",
            }
            record.update(
                {
                    f"primary_{name}": row.primary_failures.get(name, 0)
                    for name in FAILURE_ORDER
                }
            )
            record.update(
                {
                    f"signal_{name}": row.failure_signals.get(name, 0)
                    for name in SIGNAL_ORDER
                }
            )
            writer.writerow(record)


def write_summary_md(
    *, run_root: Path, summaries: Sequence[StepSummary], output_path: Path
) -> None:
    """Write a compact Markdown interpretation of the generated data."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first = summaries[0]
    last = summaries[-1]
    total_leaves = sum(row.leaf_count for row in summaries)
    total_correct = sum(row.correct_count for row in summaries)
    primary_totals = Counter[str]()
    signal_totals = Counter[str]()
    for row in summaries:
        primary_totals.update(row.primary_failures)
        signal_totals.update(row.failure_signals)
    lines = [
        f"# Response Length and Failure Analysis",
        "",
        f"- Run root: `{run_root}`",
        f"- Saved steps: `{len(summaries)}` from `{first.step}` to `{last.step}`",
        f"- Total scored leaves: `{total_leaves}`",
        f"- Overall answer passrate: `{total_correct / total_leaves:.4f}`",
        "",
        "## Length Snapshot",
        "",
        "| Step | Correct n | Incorrect n | Avg correct tokens | Avg incorrect tokens |",
        "| ---: | ---: | ---: | ---: | ---: |",
        *_length_rows(first=first, last=last),
        "",
        "## Primary Failure Totals",
        "",
        "| Category | Count | Share of incorrect |",
        "| --- | ---: | ---: |",
        *_counter_rows(
            counter=primary_totals, denominator=total_leaves - total_correct
        ),
        "",
        "## Curated Failure Signal Totals",
        "",
        "Signals may overlap, so shares do not sum to 1.",
        "",
        "| Signal | Count | Share of leaves |",
        "| --- | ---: | ---: |",
        *_counter_rows(counter=signal_totals, denominator=total_leaves),
        "",
    ]
    output_path.write_text("\n".join(lines))


def _length_rows(*, first: StepSummary, last: StepSummary) -> list[str]:
    """Format first/last length rows for Markdown."""

    return [
        (
            f"| {row.step} | {row.correct_count} | {row.incorrect_count} | "
            f"{row.avg_correct_tokens:.1f} | {row.avg_incorrect_tokens:.1f} |"
        )
        for row in (first, last)
    ]


def _counter_rows(*, counter: Counter[str], denominator: int) -> list[str]:
    """Format known counter rows in stable order."""

    order = FAILURE_ORDER if set(counter).issubset(set(FAILURE_ORDER)) else SIGNAL_ORDER
    rows: list[str] = []
    for name in order:
        count = counter.get(name, 0)
        if count == 0:
            continue
        rows.append(f"| {name} | {count} | {count / denominator:.4f} |")
    return rows


def output_directory(*, output_base: Path, date_stamp: str, run_root: Path) -> Path:
    """Return the run-specific output directory."""

    return output_base / date_stamp / run_root.name


def main() -> None:
    """Run the analysis."""

    args = parse_args()
    run_root = resolve_run_root(run_path=args.run_path)
    assert run_root.is_dir(), f"missing run root: {run_root}"
    step_dbs = iter_step_dbs(run_root=run_root)
    assert step_dbs, f"no scored step DBs found under {run_root}"
    summaries = [summarize_step(leaves=read_leaves(db_path=db)) for db in step_dbs]
    output_dir = output_directory(
        output_base=args.output_base,
        date_stamp=str(args.date_stamp),
        run_root=run_root,
    )
    plot_response_lengths(
        summaries=summaries,
        output_path=output_dir / "avg_response_tokens_by_correctness.png",
    )
    plot_primary_failures(
        summaries=summaries,
        output_path=output_dir / "primary_failure_types_over_time.png",
    )
    plot_failure_signals(
        summaries=summaries,
        output_path=output_dir / "curated_failure_signals_over_time.png",
    )
    write_step_summary_csv(
        summaries=summaries, output_path=output_dir / "step_summary.csv"
    )
    write_summary_md(
        run_root=run_root, summaries=summaries, output_path=output_dir / "summary.md"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
