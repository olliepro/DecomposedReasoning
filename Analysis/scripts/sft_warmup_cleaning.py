"""Cleaning and JSON helpers for SFT warm-up generation scripts."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from datasets import Dataset

CONTROL_TAG_PATTERN = re.compile(r"</?(?:think|steer|exec)>")


@dataclass(frozen=True)
class CleanedCompletion:
    """Structure-cleaning result for one assistant completion."""

    is_valid: bool
    reason: str
    final_answer: str
    steer_count: int
    exec_count: int


@dataclass(frozen=True)
class WarmupResultRow:
    """One generated source prompt and completion row."""

    source_index: int
    custom_id: str
    ground_truth: str
    prompt: list[dict[str, str]]
    completion: str
    stop_reason: str
    length_tokens_total: int
    structure_valid: bool
    cleaning_reason: str
    final_answer: str
    steer_count: int
    exec_count: int

    def raw_payload(self) -> dict[str, Any]:
        """Return the JSONL payload for raw generations."""

        return asdict(self)

    def cleaned_payload(self) -> dict[str, Any]:
        """Return the JSONL/parquet payload for accepted SFT rows."""

        prompt_messages = without_system_messages(messages=self.prompt)
        messages = [
            *prompt_messages,
            {"role": "assistant", "content": self.completion},
        ]
        return {
            "source_index": self.source_index,
            "custom_id": self.custom_id,
            "ground_truth": self.ground_truth,
            "prompt": prompt_messages,
            "messages": messages,
            "completion": self.completion,
            "final_answer": self.final_answer,
            "steer_count": self.steer_count,
            "exec_count": self.exec_count,
            "length_tokens_total": self.length_tokens_total,
        }


def without_system_messages(*, messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return dataset-visible messages with generation-only system rows removed."""

    return [message for message in messages if message["role"] != "system"]


def clean_completion(*, completion: str) -> CleanedCompletion:
    """Validate the strict steer/exec warm-up structure."""

    text = completion.replace("<|im_end|>", "").replace("<|endoftext|>", "")
    stripped = text.strip()
    if not stripped.startswith("<think>"):
        return invalid_clean(reason="missing_open_think")
    think_end = stripped.find("</think>")
    if think_end < 0:
        return invalid_clean(reason="missing_close_think")
    inner = stripped[len("<think>") : think_end]
    final_answer = stripped[think_end + len("</think>") :].strip()
    if not final_answer:
        return invalid_clean(reason="empty_final_answer")
    if CONTROL_TAG_PATTERN.search(final_answer):
        return invalid_clean(reason="final_answer_control_tag")
    block_result = parse_think_blocks(inner=inner)
    if not block_result.is_valid:
        return replace(block_result, final_answer=final_answer)
    return replace(block_result, final_answer=final_answer)


def parse_think_blocks(*, inner: str) -> CleanedCompletion:
    """Parse alternating non-empty steer/exec blocks inside think text."""

    position = 0
    expected = "steer"
    steer_count = 0
    exec_count = 0
    while position < len(inner):
        while position < len(inner) and inner[position].isspace():
            position += 1
        if position >= len(inner):
            break
        open_tag = f"<{expected}>"
        close_tag = f"</{expected}>"
        if not inner.startswith(open_tag, position):
            return invalid_clean(reason=f"expected_{expected}_open")
        content_start = position + len(open_tag)
        content_end = inner.find(close_tag, content_start)
        if content_end < 0:
            return invalid_clean(reason=f"missing_{expected}_close")
        if not inner[content_start:content_end].strip():
            return invalid_clean(reason=f"empty_{expected}_block")
        position = content_end + len(close_tag)
        if expected == "steer":
            steer_count += 1
            expected = "exec"
        else:
            exec_count += 1
            expected = "steer"
    if steer_count == 0 or exec_count == 0:
        return invalid_clean(reason="missing_steer_exec_pair")
    if steer_count != exec_count:
        return invalid_clean(reason="unbalanced_steer_exec_blocks")
    return CleanedCompletion(
        is_valid=True,
        reason="ok",
        final_answer="",
        steer_count=steer_count,
        exec_count=exec_count,
    )


def invalid_clean(*, reason: str) -> CleanedCompletion:
    """Return a failed cleaning result."""

    return CleanedCompletion(
        is_valid=False,
        reason=reason,
        final_answer="",
        steer_count=0,
        exec_count=0,
    )


def write_cleaned_outputs(*, run_dir: Path) -> dict[str, Any]:
    """Rewrite cleaned accepted/rejected files from raw generations."""

    raw_rows = read_jsonl(path=run_dir / "raw_generations.jsonl")
    raw_rows.sort(key=lambda row: int(row["source_index"]))
    cleaned = [row for row in raw_rows if bool(row["structure_valid"])]
    rejected = [row for row in raw_rows if not bool(row["structure_valid"])]
    cleaned_payloads = [
        WarmupResultRow(**row).cleaned_payload()
        for row in cast(list[dict[str, Any]], cleaned)
    ]
    write_jsonl(path=run_dir / "cleaned_generations.jsonl", rows=cleaned_payloads)
    write_jsonl(path=run_dir / "rejected_generations.jsonl", rows=rejected)
    if cleaned_payloads:
        Dataset.from_list(cleaned_payloads).to_parquet(
            str(run_dir / "cleaned_generations.parquet")
        )
    summary = {
        "raw_count": len(raw_rows),
        "cleaned_count": len(cleaned),
        "rejected_count": len(rejected),
        "cleaned_rate": len(cleaned) / len(raw_rows) if raw_rows else 0.0,
        "rejection_reasons": count_rejection_reasons(rows=rejected),
    }
    write_json(path=run_dir / "cleaning_summary.json", payload=summary)
    return summary


def count_rejection_reasons(*, rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count cleaning rejection reasons."""

    counts: dict[str, int] = {}
    for row in rows:
        reason = str(row.get("cleaning_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def append_jsonl(*, path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON payload to a JSONL file."""

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_jsonl(*, path: Path) -> list[dict[str, Any]]:
    """Read JSONL rows if the file exists."""

    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(cast(dict[str, Any], json.loads(stripped)))
    return rows


def write_jsonl(*, path: Path, rows: list[dict[str, Any]]) -> None:
    """Write JSONL rows atomically enough for final cleaning outputs."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(*, path: Path, payload: dict[str, Any]) -> None:
    """Write one formatted JSON file."""

    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
