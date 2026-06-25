#!/usr/bin/env python3
"""Merge SFT warm-up cleaned rows into a target-sized final dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, cast

from datasets import Dataset

CONTROL_TAG_PATTERN = re.compile(r"</?(?:think|steer|exec)>")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-run-dir", type=Path, required=True)
    parser.add_argument("--topup-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-count", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    """Merge cleaned rows and write final artifacts."""

    args = parse_args()
    rows = merged_rows(
        base_run_dir=args.base_run_dir,
        topup_run_dirs=args.topup_run_dir,
        target_count=args.target_count,
    )
    write_outputs(output_dir=args.output_dir, rows=rows, args=args)


def merged_rows(
    *, base_run_dir: Path, topup_run_dirs: list[Path], target_count: int
) -> list[dict[str, Any]]:
    """Return accepted rows from base first, then top-up rows."""

    assert target_count > 0, "target count must be positive"
    merged: list[dict[str, Any]] = []
    seen: set[int] = set()
    for row in read_jsonl(path=base_run_dir / "cleaned_generations.jsonl"):
        if row_has_control_tagged_final_answer(row=row):
            continue
        append_if_new(rows=merged, seen=seen, row=normalized_cleaned_payload(row=row))
    for topup_run_dir in topup_run_dirs:
        topup_rows = accepted_rows_from_run(run_dir=topup_run_dir)
        for row in sorted(topup_rows, key=lambda payload: int(payload["source_index"])):
            append_if_new(rows=merged, seen=seen, row=row)
            if len(merged) >= target_count:
                return merged[:target_count]
    assert (
        len(merged) >= target_count
    ), f"only {len(merged)} valid rows available for target {target_count}"
    return merged[:target_count]


def accepted_rows_from_run(*, run_dir: Path) -> list[dict[str, Any]]:
    """Read accepted rows from final outputs, or raw rows for stopped top-ups."""

    cleaned_path = run_dir / "cleaned_generations.jsonl"
    if cleaned_path.exists():
        return [
            normalized_cleaned_payload(row=row)
            for row in read_jsonl(path=cleaned_path)
            if not row_has_control_tagged_final_answer(row=row)
        ]
    raw_rows = read_jsonl(path=run_dir / "raw_generations.jsonl")
    return [
        cleaned_payload_from_raw(row=row)
        for row in raw_rows
        if bool(row.get("structure_valid"))
        and not row_has_control_tagged_final_answer(row=row)
    ]


def cleaned_payload_from_raw(*, row: dict[str, Any]) -> dict[str, Any]:
    """Convert one raw generation row into the cleaned dataset schema."""

    prompt_messages = without_system_messages(
        messages=cast(list[dict[str, str]], row["prompt"])
    )
    messages = [
        *prompt_messages,
        {"role": "assistant", "content": str(row["completion"])},
    ]
    return {
        "source_index": int(row["source_index"]),
        "custom_id": str(row["custom_id"]),
        "ground_truth": str(row.get("ground_truth", "")),
        "prompt": prompt_messages,
        "messages": messages,
        "completion": str(row["completion"]),
        "final_answer": str(row["final_answer"]),
        "steer_count": int(row["steer_count"]),
        "exec_count": int(row["exec_count"]),
        "length_tokens_total": int(row["length_tokens_total"]),
    }


def normalized_cleaned_payload(*, row: dict[str, Any]) -> dict[str, Any]:
    """Return one cleaned row with dataset-visible messages only."""

    prompt_messages = without_system_messages(
        messages=cast(list[dict[str, str]], row["prompt"])
    )
    assistant = {"role": "assistant", "content": str(row["completion"])}
    return {
        **row,
        "prompt": prompt_messages,
        "messages": [*prompt_messages, assistant],
    }


def without_system_messages(*, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return dataset-visible messages with generation-only system rows removed."""

    return [message for message in messages if message["role"] != "system"]


def row_has_control_tagged_final_answer(*, row: dict[str, Any]) -> bool:
    """Return true when the post-think answer contains control tags."""

    return bool(
        CONTROL_TAG_PATTERN.search(
            final_answer_from_completion(completion=str(row["completion"]))
        )
    )


def final_answer_from_completion(*, completion: str) -> str:
    """Return text after the first closing think tag."""

    think_end = completion.find("</think>")
    if think_end < 0:
        return str(completion)
    return completion[think_end + len("</think>") :].strip()


def append_if_new(
    *, rows: list[dict[str, Any]], seen: set[int], row: dict[str, Any]
) -> None:
    """Append a row if its source index has not already been accepted."""

    source_index = int(row["source_index"])
    if source_index in seen:
        return
    rows.append(row)
    seen.add(source_index)


def write_outputs(
    *, output_dir: Path, rows: list[dict[str, Any]], args: argparse.Namespace
) -> None:
    """Write merged JSONL, parquet, and summary artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(path=output_dir / "cleaned_generations.jsonl", rows=rows)
    Dataset.from_list(rows).to_parquet(str(output_dir / "cleaned_generations.parquet"))
    write_json(
        path=output_dir / "merge_summary.json",
        payload={
            "target_count": args.target_count,
            "cleaned_count": len(rows),
            "base_run_dir": str(args.base_run_dir),
            "topup_run_dirs": [str(path) for path in args.topup_run_dir],
            "source_index_min": min(int(row["source_index"]) for row in rows),
            "source_index_max": max(int(row["source_index"]) for row in rows),
            "reject_final_answer_control_tags": True,
        },
    )


def read_jsonl(*, path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows."""

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(cast(dict[str, Any], json.loads(stripped)))
    return rows


def write_jsonl(*, path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(*, path: Path, payload: dict[str, Any]) -> None:
    """Write formatted JSON."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
