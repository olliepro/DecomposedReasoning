from __future__ import annotations

import json
import statistics
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import tiktoken


THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass(frozen=True)
class TokenComparison:
    """Token counts for one source/transformed row pair.

    Args:
        row_id: Stable row identifier.
        source: Source dataset name.
        before_think_tokens: Think tokens before transform.
        after_think_tokens: Think tokens after transform.
        before_full_tokens: Full prompt tokens before transform.
        after_full_tokens: Full prompt tokens after transform.
        think_ratio: After/before ratio for think tokens.
        full_ratio: After/before ratio for full prompt tokens.
    """

    row_id: str
    source: str
    before_think_tokens: int
    after_think_tokens: int
    before_full_tokens: int
    after_full_tokens: int
    think_ratio: float
    full_ratio: float


def think_text(row: dict[str, object]) -> str:
    """Return concatenated assistant think blocks for one row.

    Args:
        row: JSONL row containing `messages`.

    Returns:
        Concatenated think-block text.

    Example:
        text = think_text(row=row)
    """

    blocks: list[str] = []
    messages = row.get("messages", [])
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            blocks.extend(THINK_PATTERN.findall(content))
    return "\n\n".join(blocks)


def full_text(row: dict[str, object]) -> str:
    """Return full transcript text for one row.

    Args:
        row: JSONL row containing `messages`.

    Returns:
        Concatenated `role: content` transcript.

    Example:
        text = full_text(row=row)
    """

    parts: list[str] = []
    messages = row.get("messages", [])
    for message in messages if isinstance(messages, list) else []:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "")
        content = message.get("content")
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
    return "\n".join(parts)


def percentile(values: list[float], quantile: float) -> float:
    """Compute a linear-interpolated percentile.

    Args:
        values: Numeric sample.
        quantile: Quantile in `[0, 1]`.

    Returns:
        Interpolated percentile value.
    """

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = (len(sorted_values) - 1) * quantile
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = index - lower_index
    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    return lower_value * (1 - fraction) + upper_value * fraction


def build_summary(values: list[float]) -> dict[str, float]:
    """Summarize a numeric series.

    Args:
        values: Numeric sample.

    Returns:
        Summary statistics dictionary.
    """

    return {
        "total": float(sum(values)),
        "min": float(min(values)),
        "p50": float(percentile(values=values, quantile=0.5)),
        "p90": float(percentile(values=values, quantile=0.9)),
        "p95": float(percentile(values=values, quantile=0.95)),
        "p99": float(percentile(values=values, quantile=0.99)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
    }


def load_rows(path: Path) -> dict[str, dict[str, object]]:
    """Load a JSONL file into an id-keyed row mapping.

    Args:
        path: Source JSONL path.

    Returns:
        Mapping from row id to parsed row.
    """

    rows_by_id: dict[str, dict[str, object]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_id[row["id"]] = row
    return rows_by_id


def save_histogram(
    values: list[float],
    output_path: Path,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    """Save a histogram plot.

    Args:
        values: Numeric series to plot.
        output_path: Destination image path.
        title: Plot title.
        xlabel: X-axis label.
        color: Histogram color.
    """

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=40, color=color)
    plt.xlabel(xlabel)
    plt.ylabel("rows")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def save_zoomed_histogram(
    values: list[float],
    output_path: Path,
    title: str,
    xlabel: str,
    color: str,
    x_limits: tuple[float, float],
) -> None:
    """Save a histogram focused on a bounded x-range.

    Args:
        values: Numeric series to plot.
        output_path: Destination image path.
        title: Plot title.
        xlabel: X-axis label.
        color: Histogram color.
        x_limits: Visible x-axis range.
    """

    filtered_values = [
        value for value in values if x_limits[0] <= value <= x_limits[1]
    ]
    plt.figure(figsize=(10, 6))
    plt.hist(filtered_values, bins=40, color=color)
    plt.xlim(*x_limits)
    plt.xlabel(xlabel)
    plt.ylabel("rows")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def main() -> None:
    """Generate token-count plots for the cleaned transformed subset.

    Example:
        uv run python analyze_transform_subset.py
    """

    base_dir = Path("output_transform_async_16384")
    plot_dir = base_dir / "transformed_subset_analysis"
    plot_dir.mkdir(parents=True, exist_ok=True)

    source_rows = load_rows(path=base_dir / "stratified_sample.jsonl")
    encoding = tiktoken.get_encoding("cl100k_base")
    paired_rows: list[TokenComparison] = []
    source_counts: Counter[str] = Counter()

    with (base_dir / "transformed_output.jsonl").open() as handle:
        for line in handle:
            if not line.strip():
                continue
            after_row = json.loads(line)
            row_id = after_row["id"]
            before_row = source_rows[row_id]
            before_think_tokens = len(encoding.encode(think_text(row=before_row)))
            after_think_tokens = len(encoding.encode(think_text(row=after_row)))
            before_full_tokens = len(encoding.encode(full_text(row=before_row)))
            after_full_tokens = len(encoding.encode(full_text(row=after_row)))
            source_name = after_row.get("dataset_source", "<missing>")
            assert isinstance(source_name, str)
            source_counts[source_name] += 1
            paired_rows.append(
                TokenComparison(
                    row_id=row_id,
                    source=source_name,
                    before_think_tokens=before_think_tokens,
                    after_think_tokens=after_think_tokens,
                    before_full_tokens=before_full_tokens,
                    after_full_tokens=after_full_tokens,
                    think_ratio=after_think_tokens / before_think_tokens,
                    full_ratio=after_full_tokens / before_full_tokens,
                )
            )

    before_think = [float(row.before_think_tokens) for row in paired_rows]
    after_think = [float(row.after_think_tokens) for row in paired_rows]
    before_full = [float(row.before_full_tokens) for row in paired_rows]
    after_full = [float(row.after_full_tokens) for row in paired_rows]
    think_ratio = [float(row.think_ratio) for row in paired_rows]
    full_ratio = [float(row.full_ratio) for row in paired_rows]

    summary = {
        "rows": len(paired_rows),
        "sources": dict(source_counts.most_common()),
        "before_think_tokens": build_summary(values=before_think),
        "after_think_tokens": build_summary(values=after_think),
        "before_full_tokens": build_summary(values=before_full),
        "after_full_tokens": build_summary(values=after_full),
        "think_ratio": build_summary(values=think_ratio),
        "full_ratio": build_summary(values=full_ratio),
    }
    (plot_dir / "token_ratio_stats.json").write_text(json.dumps(summary, indent=2))
    (plot_dir / "token_ratio_rows.json").write_text(
        json.dumps([asdict(row) for row in paired_rows], indent=2)
    )

    save_histogram(
        values=think_ratio,
        output_path=plot_dir / "think_token_ratio_hist.png",
        title="Distribution of <think> Token Ratios",
        xlabel="after / before <think> token ratio",
        color="#d95f02",
    )
    save_histogram(
        values=full_ratio,
        output_path=plot_dir / "full_token_ratio_hist.png",
        title="Distribution of Full Prompt Token Ratios",
        xlabel="after / before full prompt token ratio",
        color="#1b9e77",
    )
    save_zoomed_histogram(
        values=think_ratio,
        output_path=plot_dir / "think_token_ratio_hist_zoom_0.5_1.5.png",
        title="Distribution of <think> Token Ratios (0.5 to 1.5)",
        xlabel="after / before <think> token ratio",
        color="#d95f02",
        x_limits=(0.5, 1.5),
    )
    save_zoomed_histogram(
        values=full_ratio,
        output_path=plot_dir / "full_token_ratio_hist_zoom_0.5_1.5.png",
        title="Distribution of Full Prompt Token Ratios (0.5 to 1.5)",
        xlabel="after / before full prompt token ratio",
        color="#1b9e77",
        x_limits=(0.5, 1.5),
    )

    plt.figure(figsize=(10, 6))
    plt.hist(before_think, bins=40, alpha=0.6, label="before", color="#7570b3")
    plt.hist(after_think, bins=40, alpha=0.6, label="after", color="#e7298a")
    plt.xlabel("<think> tokens")
    plt.ylabel("rows")
    plt.title("<think> Token Distribution: Before vs After")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "think_tokens_before_after_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.hist(before_full, bins=40, alpha=0.6, label="before", color="#66a61e")
    plt.hist(after_full, bins=40, alpha=0.6, label="after", color="#1f78b4")
    plt.xlabel("full prompt tokens")
    plt.ylabel("rows")
    plt.title("Full Prompt Token Distribution: Before vs After")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / "full_tokens_before_after_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(before_think, after_think, s=10, alpha=0.45, color="#a6761d")
    plot_limit = max(max(before_think), max(after_think))
    plt.plot([0, plot_limit], [0, plot_limit], linestyle="--", color="black")
    plt.xlabel("before <think> tokens")
    plt.ylabel("after <think> tokens")
    plt.title("Before vs After <think> Tokens")
    plt.tight_layout()
    plt.savefig(plot_dir / "think_tokens_before_after_scatter.png", dpi=160)
    plt.close()

    source_names = [name for name, _ in source_counts.most_common()]
    source_ratios = [
        [float(row.think_ratio) for row in paired_rows if row.source == name]
        for name in source_names
    ]
    plt.figure(figsize=(12, 6))
    plt.boxplot(source_ratios, tick_labels=source_names, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("after / before <think> token ratio")
    plt.title("<think> Token Ratio by Source")
    plt.tight_layout()
    plt.savefig(plot_dir / "think_token_ratio_by_source_boxplot.png", dpi=160)
    plt.close()

    valid_band_path = (
        plot_dir / "transformed_output_think_ratio_0.8_1.3_valid.jsonl"
    )
    valid_band_ids: set[str] = set()
    if valid_band_path.exists():
        with valid_band_path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                row_id = row.get("id")
                if isinstance(row_id, str):
                    valid_band_ids.add(row_id)

    if valid_band_ids:
        overlay_before_think = [
            float(row.before_think_tokens)
            for row in paired_rows
            if row.row_id in valid_band_ids
        ]
        overlay_after_think = [
            float(row.after_think_tokens)
            for row in paired_rows
            if row.row_id in valid_band_ids
        ]
        plt.figure(figsize=(10, 6))
        plt.scatter(
            before_think,
            after_think,
            s=10,
            alpha=0.2,
            color="#999999",
            label="all transformed (1022)",
        )
        plt.scatter(
            overlay_before_think,
            overlay_after_think,
            s=18,
            alpha=0.8,
            color="#d95f02",
            label="valid ratio-band subset (538)",
        )
        plot_limit = max(max(before_think), max(after_think))
        plt.plot([0, plot_limit], [0, plot_limit], linestyle="--", color="black")
        plt.xlabel("before <think> tokens")
        plt.ylabel("after <think> tokens")
        plt.title("Before vs After <think> Tokens with Valid Subset Overlay")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            plot_dir / "think_tokens_before_after_scatter_with_valid_overlay.png",
            dpi=160,
        )
        plt.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
