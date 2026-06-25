#!/usr/bin/env python3
"""Finalize and validate the verbalized off-policy SFT warm-up dataset."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, cast

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
if str(ANALYSIS_ROOT) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_ROOT))

from scripts.merge_sft_warmup_cleaned_outputs import merged_rows, write_outputs

DEFAULT_SOURCE_JSONL = (
    ANALYSIS_ROOT.parent
    / "BuildSFTDataset/output_transform_async_16384/transformed_subset_analysis/"
    "merged_with_output_transformed_output_aug390_t1p0_pruned_top10_max_exec_"
    "truncated_seed42_plus_non_sequitur500_1to2_v2.jsonl"
)
CONTROL_TAG_PATTERN = re.compile(r"</?(?:think|steer|exec)>")
STEER_BLOCK_RE = re.compile(r"<steer>(.*?)</steer>", flags=re.DOTALL)
EXEC_BLOCK_RE = re.compile(r"<exec>(.*?)</exec>", flags=re.DOTALL)
WAIT_RE = re.compile(r"\bwait\b", flags=re.IGNORECASE)
ALT_RE = re.compile(r"\balternatively\b", flags=re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--topup-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--target-count", type=int, default=2000)
    parser.add_argument("--spotcheck-count", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    """Merge rows, validate invariants, and write inspection artifacts."""

    args = parse_args()
    rows = merged_rows(
        base_run_dir=args.base_run_dir,
        topup_run_dirs=args.topup_run_dir,
        target_count=args.target_count,
    )
    write_outputs(output_dir=args.output_dir, rows=rows, args=args)
    summary = validation_summary(rows=rows, base_run_dir=args.base_run_dir)
    write_json(path=args.output_dir / "validation_summary.json", payload=summary)
    write_text(
        path=args.output_dir / "summary_stats.md", text=summary_markdown(summary)
    )
    write_text(
        path=args.output_dir / "spotcheck_50.md",
        text=spotcheck_markdown(rows=rows, count=args.spotcheck_count),
    )


def validation_summary(
    *, rows: list[dict[str, Any]], base_run_dir: Path
) -> dict[str, Any]:
    """Build validation and summary stats for final rows."""

    source_indices = [int(row["source_index"]) for row in rows]
    completions = [str(row["completion"]) for row in rows]
    candidate_counts = verbalized_candidate_counts(
        db_path=base_run_dir / "tree_events.sqlite"
    )
    summary = {
        "row_count": len(rows),
        "unique_source_indices": len(set(source_indices)),
        "source_index_min": min(source_indices),
        "source_index_max": max(source_indices),
        "system_prompt_rows": count_system_prompt_rows(rows=rows),
        "exec_steer_spacing_violations": sum(
            "</exec><steer>" in text for text in completions
        ),
        "exec_steer_double_newline_violations": sum(
            "</exec>\n\n<steer>" in text for text in completions
        ),
        "final_answer_control_tag_rows": count_final_answer_control_tags(rows=rows),
        "wait_count": sum(len(WAIT_RE.findall(text)) for text in completions),
        "alternatively_count": sum(len(ALT_RE.findall(text)) for text in completions),
        "block_stats": block_stats(completions=completions),
        "verbalized_candidate_count_histogram": candidate_counts,
    }
    assert summary["row_count"] == 2000, summary
    assert summary["unique_source_indices"] == 2000, summary
    assert summary["system_prompt_rows"] == 0, summary
    assert summary["exec_steer_spacing_violations"] == 0, summary
    assert summary["exec_steer_double_newline_violations"] == 0, summary
    assert summary["final_answer_control_tag_rows"] == 0, summary
    assert candidate_counts, "no verbalized sampling decisions found"
    assert all(3 <= int(key) <= 10 for key in candidate_counts), candidate_counts
    return summary


def count_system_prompt_rows(*, rows: list[dict[str, Any]]) -> int:
    """Count rows whose dataset-visible prompt/messages still contain system rows."""

    count = 0
    for row in rows:
        prompt = cast(list[dict[str, Any]], row["prompt"])
        messages = cast(list[dict[str, Any]], row["messages"])
        if any(message.get("role") == "system" for message in [*prompt, *messages]):
            count += 1
    return count


def count_final_answer_control_tags(*, rows: list[dict[str, Any]]) -> int:
    """Count rows whose final answer contains internal control tags."""

    count = 0
    for row in rows:
        final_answer = str(
            row.get("final_answer") or final_answer_from_completion(row=row)
        )
        if CONTROL_TAG_PATTERN.search(final_answer):
            count += 1
    return count


def final_answer_from_completion(*, row: dict[str, Any]) -> str:
    """Extract final-answer text from a row completion."""

    completion = str(row["completion"])
    think_end = completion.find("</think>")
    if think_end < 0:
        return completion
    return completion[think_end + len("</think>") :].strip()


def block_stats(*, completions: list[str]) -> dict[str, float | int]:
    """Return simple steer/exec block length stats."""

    steer_lengths = block_token_lengths(pattern=STEER_BLOCK_RE, texts=completions)
    exec_lengths = block_token_lengths(pattern=EXEC_BLOCK_RE, texts=completions)
    return {
        "steer_block_count": len(steer_lengths),
        "exec_block_count": len(exec_lengths),
        "mean_steer_block_tokens": mean(values=steer_lengths),
        "mean_exec_block_tokens": mean(values=exec_lengths),
        "max_steer_block_tokens": max(steer_lengths) if steer_lengths else 0,
        "max_exec_block_tokens": max(exec_lengths) if exec_lengths else 0,
    }


def block_token_lengths(*, pattern: re.Pattern[str], texts: list[str]) -> list[int]:
    """Return whitespace token lengths for all matching control blocks."""

    return [
        len(match.group(1).split())
        for text in texts
        for match in pattern.finditer(text)
    ]


def mean(*, values: list[int]) -> float:
    """Return arithmetic mean or zero."""

    return sum(values) / len(values) if values else 0.0


def verbalized_candidate_counts(*, db_path: Path) -> dict[str, int]:
    """Read candidate-count histogram from the typed SQLite decision table."""

    if not db_path.exists():
        return {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as connection:
        table_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' "
            "AND name='verbalized_sampling_decision'"
        ).fetchone()
        if table_exists is None:
            return {}
        rows = connection.execute(
            "SELECT candidate_count, COUNT(*) "
            "FROM verbalized_sampling_decision GROUP BY candidate_count"
        ).fetchall()
    return {str(int(candidate_count)): int(count) for candidate_count, count in rows}


def summary_markdown(summary: dict[str, Any]) -> str:
    """Render concise summary stats as Markdown."""

    block = cast(dict[str, Any], summary["block_stats"])
    return "\n".join(
        [
            "# SFT Warm-up Verbalized Off-Policy Summary",
            "",
            f"- rows: {summary['row_count']}",
            f"- unique source indices: {summary['unique_source_indices']}",
            f"- source index range: {summary['source_index_min']}..{summary['source_index_max']}",
            f"- system prompt rows: {summary['system_prompt_rows']}",
            f"- `</exec><steer>` spacing violations: {summary['exec_steer_spacing_violations']}",
            f"- `</exec>\\n\\n<steer>` spacing violations: {summary['exec_steer_double_newline_violations']}",
            f"- final-answer control-tag rows: {summary['final_answer_control_tag_rows']}",
            f"- wait count: {summary['wait_count']}",
            f"- alternatively count: {summary['alternatively_count']}",
            f"- mean steer block tokens: {block['mean_steer_block_tokens']:.2f}",
            f"- mean exec block tokens: {block['mean_exec_block_tokens']:.2f}",
            f"- candidate count histogram: {summary['verbalized_candidate_count_histogram']}",
            "",
        ]
    )


def spotcheck_markdown(*, rows: list[dict[str, Any]], count: int) -> str:
    """Render evenly spaced rows for chat/browser inspection."""

    selected = evenly_spaced_rows(rows=rows, count=count)
    sections = ["# SFT Warm-up Verbalized Off-Policy Spot Check", ""]
    for index, row in enumerate(selected, start=1):
        sections.append(f"## {index}. source_index={row['source_index']}")
        sections.append("")
        sections.append("<details><summary>Prompt</summary>\n")
        sections.append("```text\n" + prompt_text(row=row) + "\n```")
        sections.append("</details>\n")
        sections.append("<details><summary>Completion</summary>\n")
        sections.append("```text\n" + str(row["completion"]) + "\n```")
        sections.append("</details>\n")
    return "\n".join(sections)


def evenly_spaced_rows(
    *, rows: list[dict[str, Any]], count: int
) -> list[dict[str, Any]]:
    """Return up to count rows spread across the dataset."""

    if count >= len(rows):
        return rows
    return [rows[round(i * (len(rows) - 1) / (count - 1))] for i in range(count)]


def prompt_text(*, row: dict[str, Any]) -> str:
    """Return readable prompt messages for Markdown spot checks."""

    messages = cast(list[dict[str, Any]], row["prompt"])
    return "\n\n".join(
        f"{message.get('role', '')}:\n{message.get('content', '')}"
        for message in messages
    )


def write_json(*, path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_text(*, path: Path, text: str) -> None:
    """Write UTF-8 text."""

    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
