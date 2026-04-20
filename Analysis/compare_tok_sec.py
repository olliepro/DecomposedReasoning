"""Compare baseline and branching token throughput from branching-eval logs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from branching_eval.event_types import EventEnvelope, parse_event_row
from io_utils import read_jsonl


@dataclass(frozen=True)
class ThroughputMetrics:
    """Aggregate token-throughput metrics for one request slice.

    Args:
        output_tokens: Total generated output tokens across successful responses.
        request_count: Count of successful vLLM responses.
        request_latency_seconds: Sum of per-request latency values.
        wall_clock_seconds: Elapsed wall time from first request to last response.

    Returns:
        Dataclass with derived request and wall throughput metrics.
    """

    output_tokens: int
    request_count: int
    request_latency_seconds: float
    wall_clock_seconds: float

    def request_tok_per_sec(self) -> float:
        """Return tokens per second using summed request latency.

        Args:
            None.

        Returns:
            Throughput based on `output_tokens / request_latency_seconds`.
        """

        if self.request_latency_seconds <= 0.0:
            return 0.0
        return self.output_tokens / self.request_latency_seconds

    def wall_tok_per_sec(self) -> float:
        """Return tokens per second using wall-clock runtime span.

        Args:
            None.

        Returns:
            Throughput based on `output_tokens / wall_clock_seconds`.
        """

        if self.wall_clock_seconds <= 0.0:
            return 0.0
        return self.output_tokens / self.wall_clock_seconds


@dataclass(frozen=True)
class RequestKindSummary:
    """Throughput summary for one `request_kind`.

    Args:
        request_kind: Logged request-kind label.
        metrics: Aggregate throughput metrics for that request kind.

    Returns:
        Dataclass describing one request-kind slice.
    """

    request_kind: str
    metrics: ThroughputMetrics


@dataclass(frozen=True)
class RunSummary:
    """Resolved throughput summary for one branching-eval run directory.

    Args:
        run_dir: Run artifact directory.
        task_name: lm_eval task name.
        model_id: Experiment model id.
        mode: Experiment mode (`baseline` or `branching`).
        selector_mode: Active selector label.
        seed: Experiment seed.
        baseline_rollouts: Baseline rollout count configured for this run.
        metrics: Aggregate throughput metrics across all successful requests.
        request_kinds: Per-request-kind breakdown for the run.
        event_count: Count of parsed events.

    Returns:
        Dataclass used for run-level and pairwise throughput comparisons.

    Example:
        >>> summary = RunSummary(
        ...     run_dir=Path("output/example"),
        ...     task_name="aime24",
        ...     model_id="sft",
        ...     mode="baseline",
        ...     selector_mode="baseline",
        ...     seed=1234,
        ...     baseline_rollouts=16,
        ...     metrics=ThroughputMetrics(
        ...         output_tokens=32,
        ...         request_count=1,
        ...         request_latency_seconds=4.0,
        ...         wall_clock_seconds=5.0,
        ...     ),
        ...     request_kinds=(),
        ...     event_count=2,
        ... )
        >>> round(summary.metrics.request_tok_per_sec(), 2)
        8.0
    """

    run_dir: Path
    task_name: str
    model_id: str
    mode: str
    selector_mode: str
    seed: int
    baseline_rollouts: int
    metrics: ThroughputMetrics
    request_kinds: tuple[RequestKindSummary, ...]
    event_count: int


@dataclass(frozen=True)
class ComparisonRow:
    """Matched baseline-vs-branching throughput comparison row.

    Args:
        task_name: Shared lm_eval task.
        model_id: Shared model id.
        seed: Shared experiment seed.
        selector_mode: Branching selector mode.
        baseline_run_dir: Matched baseline run directory.
        branching_run_dir: Matched branching run directory.
        baseline_metrics: Baseline throughput metrics.
        branching_metrics: Branching throughput metrics.

    Returns:
        Dataclass with aligned run-pair comparison metrics.
    """

    task_name: str
    model_id: str
    seed: int
    selector_mode: str
    baseline_run_dir: Path
    branching_run_dir: Path
    baseline_metrics: ThroughputMetrics
    branching_metrics: ThroughputMetrics

    def wall_tok_per_sec_ratio(self) -> float:
        """Return branching-to-baseline wall throughput ratio.

        Args:
            None.

        Returns:
            `branching / baseline` wall token-per-second ratio.
        """

        baseline_value = self.baseline_metrics.wall_tok_per_sec()
        if baseline_value <= 0.0:
            return 0.0
        return self.branching_metrics.wall_tok_per_sec() / baseline_value

    def request_tok_per_sec_ratio(self) -> float:
        """Return branching-to-baseline request throughput ratio.

        Args:
            None.

        Returns:
            `branching / baseline` request token-per-second ratio.
        """

        baseline_value = self.baseline_metrics.request_tok_per_sec()
        if baseline_value <= 0.0:
            return 0.0
        return self.branching_metrics.request_tok_per_sec() / baseline_value


@dataclass(frozen=True)
class CliArgs:
    """Parsed CLI arguments for throughput comparison.

    Args:
        run_dirs: Explicit run directories to compare.
        output_root: Optional output root used for run discovery.
        latest_only: Whether to keep only the latest run per experiment key.
        task_name: Optional task filter.
        model_id: Optional model filter.
        seed: Optional seed filter.
        selector_mode: Optional selector filter for branching runs.
        json_output: Whether to emit JSON instead of text.
        show_request_kinds: Whether to include per-request-kind text output.

    Returns:
        Dataclass containing normalized CLI inputs.
    """

    run_dirs: tuple[Path, ...]
    output_root: Path | None
    latest_only: bool
    task_name: str | None
    model_id: str | None
    seed: int | None
    selector_mode: str | None
    json_output: bool
    show_request_kinds: bool


def parse_args() -> CliArgs:
    """Parse CLI arguments for token-throughput comparison.

    Args:
        None.

    Returns:
        Normalized CLI argument dataclass.
    """

    parser = argparse.ArgumentParser(
        description="Compare baseline and branching tokens/sec from run logs."
    )
    parser.add_argument("--run-dir", dest="run_dirs", type=Path, action="append")
    parser.add_argument(
        "--output-root", type=Path, default=Path("output/branching_eval")
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Use every discovered run instead of keeping the latest per key.",
    )
    parser.add_argument("--task", dest="task_name", type=str, default=None)
    parser.add_argument("--model", dest="model_id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--selector", dest="selector_mode", type=str, default=None)
    parser.add_argument("--json", dest="json_output", action="store_true")
    parser.add_argument("--show-request-kinds", action="store_true")
    raw_args = parser.parse_args()
    return CliArgs(
        run_dirs=tuple(raw_args.run_dirs or ()),
        output_root=raw_args.output_root,
        latest_only=not raw_args.all_runs,
        task_name=raw_args.task_name,
        model_id=raw_args.model_id,
        seed=raw_args.seed,
        selector_mode=raw_args.selector_mode,
        json_output=bool(raw_args.json_output),
        show_request_kinds=bool(raw_args.show_request_kinds),
    )


def main() -> None:
    """Run CLI comparison workflow.

    Args:
        None.

    Returns:
        None.
    """

    args = parse_args()
    run_dirs = resolve_run_dirs(args=args)
    summaries = [summarize_run(run_dir=run_dir) for run_dir in run_dirs]
    summaries = filter_summaries(
        summaries=summaries,
        task_name=args.task_name,
        model_id=args.model_id,
        seed=args.seed,
        selector_mode=args.selector_mode,
    )
    if args.latest_only:
        summaries = select_latest_runs(summaries=summaries)
    comparisons = build_comparison_rows(summaries=summaries)
    if args.json_output:
        print(
            json.dumps(
                build_json_payload(summaries=summaries, comparisons=comparisons),
                indent=2,
            )
        )
        return
    print(
        render_text_report(
            summaries=summaries,
            comparisons=comparisons,
            show_request_kinds=args.show_request_kinds,
        )
    )


def resolve_run_dirs(*, args: CliArgs) -> tuple[Path, ...]:
    """Resolve explicit or discovered run directories.

    Args:
        args: Parsed CLI argument dataclass.

    Returns:
        Ordered tuple of run directories to summarize.
    """

    if args.run_dirs:
        run_dirs = tuple(run_dir.resolve() for run_dir in args.run_dirs)
        assert run_dirs, "at least one --run-dir is required"
        return run_dirs
    assert (
        args.output_root is not None
    ), "--output-root is required when --run-dir is unset"
    return discover_run_dirs(output_root=args.output_root.resolve())


def discover_run_dirs(*, output_root: Path) -> tuple[Path, ...]:
    """Discover branching-eval run directories under one output root.

    Args:
        output_root: Root directory containing run subdirectories.

    Returns:
        Ordered tuple of valid run directories.

    Example:
        >>> discover_run_dirs(output_root=Path("missing_root"))
        ()
    """

    if not output_root.exists():
        return ()
    run_dirs = [
        child.resolve()
        for child in output_root.iterdir()
        if child.is_dir() and (child / "tree_events.jsonl").exists()
    ]
    run_dirs.sort()
    return tuple(run_dirs)


def summarize_run(*, run_dir: Path) -> RunSummary:
    """Load one run directory and compute throughput aggregates.

    Args:
        run_dir: Branching-eval run directory containing config and events.

    Returns:
        Run summary with aggregate and per-request-kind throughput.
    """

    config_snapshot = load_json(path=run_dir / "config_snapshot.json")
    events = load_events(run_dir=run_dir)
    experiment = extract_experiment(config_snapshot=config_snapshot)
    metrics = summarize_events(events=events)
    request_kinds = summarize_request_kinds(events=events)
    return RunSummary(
        run_dir=run_dir.resolve(),
        task_name=experiment["task_name"],
        model_id=experiment["model_id"],
        mode=experiment["mode"],
        selector_mode=experiment["selector_mode"],
        seed=experiment["seed"],
        baseline_rollouts=experiment["baseline_rollouts"],
        metrics=metrics,
        request_kinds=request_kinds,
        event_count=len(events),
    )


def load_json(*, path: Path) -> dict[str, Any]:
    """Load one JSON mapping from disk.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON mapping.
    """

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"expected mapping JSON at {path}"
    return payload


def load_events(*, run_dir: Path) -> tuple[EventEnvelope, ...]:
    """Load typed event envelopes from one run directory.

    Args:
        run_dir: Branching-eval run directory.

    Returns:
        Ordered tuple of parsed event envelopes.
    """

    rows = read_jsonl(path=run_dir / "tree_events.jsonl")
    return tuple(parse_event_row(row=row) for row in rows)


def extract_experiment(*, config_snapshot: dict[str, Any]) -> dict[str, Any]:
    """Extract comparison metadata from config snapshot payload.

    Args:
        config_snapshot: Parsed `config_snapshot.json` payload.

    Returns:
        Flat mapping containing experiment comparison labels.
    """

    experiment_payload = config_snapshot.get("experiment", {})
    run_matrix_payload = config_snapshot.get("run_matrix", {})
    assert isinstance(
        experiment_payload, dict
    ), "config snapshot missing experiment payload"
    assert isinstance(
        run_matrix_payload, dict
    ), "config snapshot missing run_matrix payload"
    selector_mode = experiment_payload.get("selector")
    if selector_mode is None:
        selector_mode = "baseline"
    return {
        "task_name": str(experiment_payload["task_name"]),
        "model_id": str(experiment_payload["model_id"]),
        "mode": str(experiment_payload["mode"]),
        "selector_mode": str(selector_mode),
        "seed": int(experiment_payload["seed"]),
        "baseline_rollouts": int(run_matrix_payload.get("baseline_rollouts", 0)),
    }


def summarize_events(*, events: Iterable[EventEnvelope]) -> ThroughputMetrics:
    """Aggregate token and timing metrics from successful vLLM responses.

    Args:
        events: Event sequence from one run log.

    Returns:
        Aggregate throughput metrics for the full run.
    """

    relevant_events = tuple(events)
    response_events = successful_vllm_responses(events=relevant_events)
    output_tokens = sum(
        output_tokens_from_response(event=event) for event in response_events
    )
    request_latency_seconds = sum(
        response_latency_seconds(event=event) for event in response_events
    )
    wall_clock_seconds = request_wall_clock_seconds(events=relevant_events)
    return ThroughputMetrics(
        output_tokens=output_tokens,
        request_count=len(response_events),
        request_latency_seconds=request_latency_seconds,
        wall_clock_seconds=wall_clock_seconds,
    )


def summarize_request_kinds(
    *, events: Iterable[EventEnvelope]
) -> tuple[RequestKindSummary, ...]:
    """Build per-request-kind throughput slices for one run.

    Args:
        events: Event sequence from one run log.

    Returns:
        Ordered tuple of per-request-kind throughput summaries.
    """

    request_kind_to_events: dict[str, list[EventEnvelope]] = {}
    for event in events:
        if event.event_type not in {"vllm_request", "vllm_response"}:
            continue
        payload = event.payload
        request_kind = str(payload.get("request_kind", "unknown"))
        if event.event_type == "vllm_response" and payload.get("status") != "ok":
            continue
        request_kind_to_events.setdefault(request_kind, []).append(event)
    summaries = [
        RequestKindSummary(
            request_kind=request_kind,
            metrics=summarize_events(events=request_events),
        )
        for request_kind, request_events in sorted(request_kind_to_events.items())
    ]
    return tuple(summaries)


def successful_vllm_responses(
    *, events: Iterable[EventEnvelope]
) -> tuple[EventEnvelope, ...]:
    """Return successful `vllm_response` events.

    Args:
        events: Event sequence from one run log.

    Returns:
        Ordered tuple of successful response events.
    """

    rows = []
    for event in events:
        if event.event_type != "vllm_response":
            continue
        if event.payload.get("status") != "ok":
            continue
        rows.append(event)
    return tuple(rows)


def output_tokens_from_response(*, event: EventEnvelope) -> int:
    """Sum generated output tokens from one response event.

    Args:
        event: Successful `vllm_response` event.

    Returns:
        Total output tokens across all returned choices.
    """

    choices = event.payload.get("choices", [])
    assert isinstance(choices, list), "response choices must be a list"
    return sum(
        choice_output_tokens(choice=choice)
        for choice in choices
        if isinstance(choice, dict)
    )


def choice_output_tokens(*, choice: dict[str, Any]) -> int:
    """Read one choice's output-token count from event payload.

    Args:
        choice: Serialized choice payload from `vllm_response`.

    Returns:
        Output token count for the choice.
    """

    raw_value = choice.get("output_token_count", 0)
    return int(raw_value)


def response_latency_seconds(*, event: EventEnvelope) -> float:
    """Return latency seconds from one response event.

    Args:
        event: Successful `vllm_response` event.

    Returns:
        Parsed latency seconds.
    """

    return float(event.payload.get("latency_seconds", 0.0))


def request_wall_clock_seconds(*, events: Iterable[EventEnvelope]) -> float:
    """Compute request wall-clock span from first request to last response.

    Args:
        events: Event sequence from one run log.

    Returns:
        Non-negative request wall-clock seconds.

    Example:
        >>> request_wall_clock_seconds(events=())
        0.0
    """

    request_times = timestamps_for_event_type(events=events, event_type="vllm_request")
    response_times = timestamps_for_event_type(
        events=events, event_type="vllm_response"
    )
    if not request_times or not response_times:
        return 0.0
    return max(0.0, (max(response_times) - min(request_times)).total_seconds())


def timestamps_for_event_type(
    *, events: Iterable[EventEnvelope], event_type: str
) -> tuple[datetime, ...]:
    """Return parsed timestamps for one event type.

    Args:
        events: Event sequence from one run log.
        event_type: Canonical event type label.

    Returns:
        Ordered tuple of matching event timestamps.
    """

    timestamps = []
    for event in events:
        if event.event_type != event_type:
            continue
        timestamps.append(datetime.fromisoformat(event.timestamp_utc))
    return tuple(timestamps)


def filter_summaries(
    *,
    summaries: list[RunSummary],
    task_name: str | None,
    model_id: str | None,
    seed: int | None,
    selector_mode: str | None,
) -> list[RunSummary]:
    """Filter run summaries by optional comparison labels.

    Args:
        summaries: Run summaries to filter.
        task_name: Optional task-name filter.
        model_id: Optional model-id filter.
        seed: Optional seed filter.
        selector_mode: Optional selector filter for branching summaries.

    Returns:
        Filtered run summary list.
    """

    filtered = []
    for summary in summaries:
        if task_name is not None and summary.task_name != task_name:
            continue
        if model_id is not None and summary.model_id != model_id:
            continue
        if seed is not None and summary.seed != seed:
            continue
        if selector_mode is not None and summary.selector_mode != selector_mode:
            continue
        filtered.append(summary)
    return filtered


def select_latest_runs(*, summaries: list[RunSummary]) -> list[RunSummary]:
    """Keep only the latest run for each experiment identity.

    Args:
        summaries: Loaded run summaries, potentially with duplicates.

    Returns:
        One summary per `(task, model, mode, selector, seed, baseline_rollouts)`.
    """

    latest_by_key: dict[tuple[str, str, str, str, int, int], RunSummary] = {}
    for summary in summaries:
        key = run_identity_key(summary=summary)
        previous = latest_by_key.get(key)
        if previous is None or summary.run_dir.name > previous.run_dir.name:
            latest_by_key[key] = summary
    return sorted(latest_by_key.values(), key=run_sort_key)


def run_identity_key(*, summary: RunSummary) -> tuple[str, str, str, str, int, int]:
    """Return grouping key used for latest-run deduplication.

    Args:
        summary: Run summary row.

    Returns:
        Tuple uniquely identifying a repeated experiment setting.
    """

    return (
        summary.task_name,
        summary.model_id,
        summary.mode,
        summary.selector_mode,
        summary.seed,
        summary.baseline_rollouts,
    )


def run_sort_key(summary: RunSummary) -> tuple[str, str, int, str, str]:
    """Return stable display ordering for run summaries.

    Args:
        summary: Run summary row.

    Returns:
        Sorting tuple for reports.
    """

    return (
        summary.task_name,
        summary.model_id,
        summary.seed,
        summary.mode,
        summary.selector_mode,
    )


def build_comparison_rows(*, summaries: list[RunSummary]) -> list[ComparisonRow]:
    """Match baseline runs to branching runs for throughput comparison.

    Args:
        summaries: Run summaries to pair.

    Returns:
        Ordered baseline-vs-branching comparison rows.
    """

    baseline_by_key: dict[tuple[str, str, int, int], RunSummary] = {}
    for summary in summaries:
        if summary.mode != "baseline":
            continue
        baseline_by_key[comparison_key(summary=summary)] = summary
    rows = []
    for summary in summaries:
        if summary.mode != "branching":
            continue
        baseline = baseline_by_key.get(comparison_key(summary=summary))
        if baseline is None:
            continue
        rows.append(
            ComparisonRow(
                task_name=summary.task_name,
                model_id=summary.model_id,
                seed=summary.seed,
                selector_mode=summary.selector_mode,
                baseline_run_dir=baseline.run_dir,
                branching_run_dir=summary.run_dir,
                baseline_metrics=baseline.metrics,
                branching_metrics=summary.metrics,
            )
        )
    rows.sort(
        key=lambda row: (row.task_name, row.model_id, row.seed, row.selector_mode)
    )
    return rows


def comparison_key(*, summary: RunSummary) -> tuple[str, str, int, int]:
    """Return baseline-branching pairing key.

    Args:
        summary: Run summary row.

    Returns:
        Tuple aligning baseline and branching runs for comparison.
    """

    return (
        summary.task_name,
        summary.model_id,
        summary.seed,
        summary.baseline_rollouts,
    )


def build_json_payload(
    *, summaries: list[RunSummary], comparisons: list[ComparisonRow]
) -> dict[str, Any]:
    """Build JSON payload for CLI output.

    Args:
        summaries: Run summaries to serialize.
        comparisons: Comparison rows to serialize.

    Returns:
        JSON-ready payload containing runs and matched comparisons.
    """

    return {
        "runs": [run_summary_json(summary=summary) for summary in summaries],
        "comparisons": [comparison_row_json(row=row) for row in comparisons],
    }


def run_summary_json(*, summary: RunSummary) -> dict[str, Any]:
    """Serialize one run summary to JSON.

    Args:
        summary: Run summary row.

    Returns:
        JSON-ready mapping for one run.
    """

    return {
        "run_dir": str(summary.run_dir),
        "task_name": summary.task_name,
        "model_id": summary.model_id,
        "mode": summary.mode,
        "selector_mode": summary.selector_mode,
        "seed": summary.seed,
        "baseline_rollouts": summary.baseline_rollouts,
        "metrics": metrics_json(metrics=summary.metrics),
        "request_kinds": [
            {
                "request_kind": request_kind.request_kind,
                "metrics": metrics_json(metrics=request_kind.metrics),
            }
            for request_kind in summary.request_kinds
        ],
        "event_count": summary.event_count,
    }


def comparison_row_json(*, row: ComparisonRow) -> dict[str, Any]:
    """Serialize one matched comparison row to JSON.

    Args:
        row: Comparison row.

    Returns:
        JSON-ready mapping for one baseline-vs-branching pair.
    """

    return {
        "task_name": row.task_name,
        "model_id": row.model_id,
        "seed": row.seed,
        "selector_mode": row.selector_mode,
        "baseline_run_dir": str(row.baseline_run_dir),
        "branching_run_dir": str(row.branching_run_dir),
        "baseline_metrics": metrics_json(metrics=row.baseline_metrics),
        "branching_metrics": metrics_json(metrics=row.branching_metrics),
        "wall_tok_per_sec_ratio": row.wall_tok_per_sec_ratio(),
        "request_tok_per_sec_ratio": row.request_tok_per_sec_ratio(),
    }


def metrics_json(*, metrics: ThroughputMetrics) -> dict[str, Any]:
    """Serialize throughput metrics to JSON.

    Args:
        metrics: Throughput metrics to serialize.

    Returns:
        JSON-ready mapping with derived throughput values.
    """

    return {
        "output_tokens": metrics.output_tokens,
        "request_count": metrics.request_count,
        "request_latency_seconds": metrics.request_latency_seconds,
        "wall_clock_seconds": metrics.wall_clock_seconds,
        "request_tok_per_sec": metrics.request_tok_per_sec(),
        "wall_tok_per_sec": metrics.wall_tok_per_sec(),
    }


def render_text_report(
    *,
    summaries: list[RunSummary],
    comparisons: list[ComparisonRow],
    show_request_kinds: bool,
) -> str:
    """Render a text report for terminal use.

    Args:
        summaries: Run summaries to display.
        comparisons: Matched comparison rows to display.
        show_request_kinds: Whether to include per-request-kind sections.

    Returns:
        Multi-line plain-text report.
    """

    lines = [
        "Runs",
        render_runs_table(summaries=summaries),
        "",
        "Comparisons",
        render_comparisons_table(comparisons=comparisons),
    ]
    if show_request_kinds:
        lines.extend(render_request_kind_sections(summaries=summaries))
    return "\n".join(lines)


def render_runs_table(*, summaries: list[RunSummary]) -> str:
    """Render the run summary table.

    Args:
        summaries: Run summaries to display.

    Returns:
        Multi-line text table.
    """

    headers = (
        "mode",
        "selector",
        "task",
        "model",
        "seed",
        "tokens",
        "req_tok/s",
        "wall_tok/s",
        "run_dir",
    )
    rows = [
        (
            summary.mode,
            summary.selector_mode,
            summary.task_name,
            summary.model_id,
            str(summary.seed),
            str(summary.metrics.output_tokens),
            format_float(summary.metrics.request_tok_per_sec()),
            format_float(summary.metrics.wall_tok_per_sec()),
            summary.run_dir.name,
        )
        for summary in summaries
    ]
    return render_table(headers=headers, rows=rows)


def render_comparisons_table(*, comparisons: list[ComparisonRow]) -> str:
    """Render the matched comparison table.

    Args:
        comparisons: Matched comparison rows to display.

    Returns:
        Multi-line text table.
    """

    headers = (
        "task",
        "model",
        "seed",
        "selector",
        "baseline_wall",
        "branch_wall",
        "wall_ratio",
        "req_ratio",
    )
    rows = [
        (
            row.task_name,
            row.model_id,
            str(row.seed),
            row.selector_mode,
            format_float(row.baseline_metrics.wall_tok_per_sec()),
            format_float(row.branching_metrics.wall_tok_per_sec()),
            format_ratio(row.wall_tok_per_sec_ratio()),
            format_ratio(row.request_tok_per_sec_ratio()),
        )
        for row in comparisons
    ]
    return render_table(headers=headers, rows=rows)


def render_request_kind_sections(*, summaries: list[RunSummary]) -> list[str]:
    """Render optional per-request-kind sections.

    Args:
        summaries: Run summaries to display.

    Returns:
        List of text sections appended to the main report.
    """

    sections: list[str] = ["", "Request Kinds"]
    for summary in summaries:
        sections.append(f"{summary.run_dir.name}")
        headers = ("request_kind", "tokens", "req_tok/s", "wall_tok/s")
        rows = [
            (
                request_kind.request_kind,
                str(request_kind.metrics.output_tokens),
                format_float(request_kind.metrics.request_tok_per_sec()),
                format_float(request_kind.metrics.wall_tok_per_sec()),
            )
            for request_kind in summary.request_kinds
        ]
        sections.append(render_table(headers=headers, rows=rows))
        sections.append("")
    if sections[-1] == "":
        sections.pop()
    return sections


def render_table(*, headers: tuple[str, ...], rows: list[tuple[str, ...]]) -> str:
    """Render a simple left-aligned text table.

    Args:
        headers: Column header row.
        rows: String row tuples matching the header width.

    Returns:
        Multi-line plain-text table.
    """

    if not rows:
        return "(none)"
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    line_rows = [
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    line_rows.extend(
        "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        for row in rows
    )
    return "\n".join(line_rows)


def format_float(value: float) -> str:
    """Format one float for compact table output.

    Args:
        value: Float value to format.

    Returns:
        Compact fixed-point string.
    """

    return f"{value:.2f}"


def format_ratio(value: float) -> str:
    """Format one throughput ratio for compact table output.

    Args:
        value: Ratio value to format.

    Returns:
        Compact ratio string with `x` suffix.
    """

    return f"{value:.2f}x"


if __name__ == "__main__":
    main()
