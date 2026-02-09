from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_steering_clusters import (  # noqa: E402
    extract_single_think_block,
    parse_steering_execution_pairs,
    residual_text_outside_segments,
)

DEFAULT_DATASET_PATH = Path(
    "/Users/olliepro/Code/School/DecomposedReasoning/BuildSFTDataset/output/transformed_output.jsonl"
)


@dataclass(frozen=True)
class ThinkStructureIssue:
    """Validation issue for one dataset row.

    Args:
        row_index: Zero-based row index in dataset file.
        row_id: Row identifier.
        dataset_source: Dataset source string.
        issue_type: Stable issue category name.
        details: Human-readable details for this issue.

    Example:
        >>> issue = ThinkStructureIssue(
        ...     row_index=0,
        ...     row_id="row-1",
        ...     dataset_source="sample",
        ...     issue_type="missing_assistant_content",
        ...     details="No assistant message content found.",
        ... )
        >>> issue.issue_type
        'missing_assistant_content'
    """

    row_index: int
    row_id: str
    dataset_source: str
    issue_type: str
    details: str

    def to_json(self) -> dict[str, Any]:
        """Convert issue to JSON-serializable dictionary.

        Returns:
            Dictionary payload suitable for JSON output.
        """

        return {
            "row_index": self.row_index,
            "row_id": self.row_id,
            "dataset_source": self.dataset_source,
            "issue_type": self.issue_type,
            "details": self.details,
        }


@dataclass(frozen=True)
class ThinkStructureSummary:
    """Aggregate validation summary for one dataset.

    Args:
        dataset_path: Validated dataset path.
        rows_scanned: Number of rows scanned.
        rows_valid: Number of rows with no issues.
        rows_invalid: Number of rows with at least one issue.
        issue_counts: Counts by issue type.
        issue_examples: Sample issues for quick inspection.
    """

    dataset_path: Path
    rows_scanned: int
    rows_valid: int
    rows_invalid: int
    issue_counts: dict[str, int]
    issue_examples: list[ThinkStructureIssue]

    def to_json(self) -> dict[str, Any]:
        """Convert summary to JSON-serializable dictionary.

        Returns:
            JSON-compatible summary dictionary.
        """

        return {
            "dataset_path": str(self.dataset_path),
            "rows_scanned": self.rows_scanned,
            "rows_valid": self.rows_valid,
            "rows_invalid": self.rows_invalid,
            "issue_counts": self.issue_counts,
            "issue_examples": [issue.to_json() for issue in self.issue_examples],
        }


def load_jsonl_rows(dataset_path: Path) -> list[dict[str, Any]]:
    """Load JSONL rows from disk.

    Args:
        dataset_path: Path to JSONL file.

    Returns:
        Parsed row dictionaries.
    """

    rows: list[dict[str, Any]] = []
    with dataset_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            assert isinstance(row, dict), "Each JSONL line must decode to object"
            rows.append(row)
    return rows


def extract_last_assistant_content(messages: list[dict[str, Any]]) -> str | None:
    """Return content from the last assistant message.

    Args:
        messages: Row messages list.

    Returns:
        Last assistant content string or None.

    Example:
        >>> extract_last_assistant_content(
        ...     messages=[
        ...         {"role": "assistant", "content": "first"},
        ...         {"role": "assistant", "content": "last"},
        ...     ]
        ... )
        'last'
    """

    for message in reversed(messages):
        if message.get("role") == "assistant" and isinstance(
            message.get("content"), str
        ):
            return str(message["content"])
    return None


def validate_row_think_structure(
    row: dict[str, Any], row_index: int
) -> list[ThinkStructureIssue]:
    """Validate `<think>` structure for one row.

    Args:
        row: One dataset row.
        row_index: Zero-based row index.

    Returns:
        List of structure issues found for the row.
    """

    row_id = str(row.get("id", f"row-{row_index}"))
    dataset_source = str(row.get("dataset_source", "unknown"))
    issues: list[ThinkStructureIssue] = []

    messages = row.get("messages")
    if not isinstance(messages, list):
        issues.append(
            ThinkStructureIssue(
                row_index=row_index,
                row_id=row_id,
                dataset_source=dataset_source,
                issue_type="invalid_messages",
                details=f"Expected list for messages, got {type(messages).__name__}.",
            )
        )
        return issues

    assistant_content = extract_last_assistant_content(messages=messages)
    if assistant_content is None:
        issues.append(
            ThinkStructureIssue(
                row_index=row_index,
                row_id=row_id,
                dataset_source=dataset_source,
                issue_type="missing_assistant_content",
                details="No assistant content string found.",
            )
        )
        return issues

    think_text, anomalies = extract_single_think_block(
        content=assistant_content,
        row_id=row_id,
        source_name=dataset_source,
    )
    if anomalies:
        anomaly_count = int(anomalies[0].get("count", -1))
        issues.append(
            ThinkStructureIssue(
                row_index=row_index,
                row_id=row_id,
                dataset_source=dataset_source,
                issue_type="invalid_think_block_count",
                details=f"Expected exactly 1 think block, found {anomaly_count}.",
            )
        )
        return issues
    assert think_text is not None, "think_text must be present when anomalies are empty"

    pairs, parse_meta = parse_steering_execution_pairs(think_text=think_text)
    if parse_meta["paired_count"] == 0:
        issues.append(
            ThinkStructureIssue(
                row_index=row_index,
                row_id=row_id,
                dataset_source=dataset_source,
                issue_type="no_paired_sections",
                details="No paired steering/execution sections found.",
            )
        )
    if parse_meta["unmatched_steering"] != 0 or parse_meta["unmatched_execution"] != 0:
        issues.append(
            ThinkStructureIssue(
                row_index=row_index,
                row_id=row_id,
                dataset_source=dataset_source,
                issue_type="unmatched_sections",
                details=(
                    f"unmatched_steering={parse_meta['unmatched_steering']} "
                    f"unmatched_execution={parse_meta['unmatched_execution']}"
                ),
            )
        )
    if len(pairs) != int(parse_meta["paired_count"]):
        issues.append(
            ThinkStructureIssue(
                row_index=row_index,
                row_id=row_id,
                dataset_source=dataset_source,
                issue_type="paired_count_mismatch",
                details=(
                    f"len(pairs)={len(pairs)} "
                    f"paired_count={int(parse_meta['paired_count'])}"
                ),
            )
        )

    residual_text = residual_text_outside_segments(think_text=think_text)
    if residual_text:
        issues.append(
            ThinkStructureIssue(
                row_index=row_index,
                row_id=row_id,
                dataset_source=dataset_source,
                issue_type="residual_text_outside_sections",
                details=residual_text.replace("\n", " ")[:220],
            )
        )
    return issues


def validate_dataset_think_structure(
    dataset_path: Path, max_issue_examples: int
) -> ThinkStructureSummary:
    """Validate think-block structure across an entire JSONL dataset.

    Args:
        dataset_path: Path to transformed dataset JSONL.
        max_issue_examples: Maximum issue examples retained in summary.

    Returns:
        Dataset-level think-structure summary.

    Example:
        >>> summary = validate_dataset_think_structure(
        ...     dataset_path=DEFAULT_DATASET_PATH,
        ...     max_issue_examples=5,
        ... )
        >>> summary.rows_scanned >= summary.rows_valid
        True
    """

    rows = load_jsonl_rows(dataset_path=dataset_path)
    issues: list[ThinkStructureIssue] = []
    invalid_row_ids: set[str] = set()
    issue_counter: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        row_issues = validate_row_think_structure(row=row, row_index=row_index)
        if not row_issues:
            continue
        invalid_row_ids.add(row_issues[0].row_id)
        for issue in row_issues:
            issue_counter[issue.issue_type] += 1
            if len(issues) < max_issue_examples:
                issues.append(issue)

    rows_scanned = len(rows)
    rows_invalid = len(invalid_row_ids)
    rows_valid = rows_scanned - rows_invalid
    return ThinkStructureSummary(
        dataset_path=dataset_path,
        rows_scanned=rows_scanned,
        rows_valid=rows_valid,
        rows_invalid=rows_invalid,
        issue_counts=dict(issue_counter),
        issue_examples=issues,
    )


def test_clean_dataset_think_blocks_are_structured_and_parseable() -> None:
    """Assert clean transformed dataset has parseable think-block structure."""

    summary = validate_dataset_think_structure(
        dataset_path=DEFAULT_DATASET_PATH,
        max_issue_examples=12,
    )
    assert summary.rows_invalid == 0, json.dumps(summary.to_json(), indent=2)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for dataset-wide think validation.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(
        description="Validate <think> block structure and parseability for a clean dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Path to transformed_output.jsonl dataset.",
    )
    parser.add_argument(
        "--max-issue-examples",
        type=int,
        default=20,
        help="Maximum issue examples emitted in summary JSON.",
    )
    return parser.parse_args()


def main() -> None:
    """Run dataset think-structure validation and exit non-zero on failures."""

    args = parse_args()
    summary = validate_dataset_think_structure(
        dataset_path=args.dataset_path,
        max_issue_examples=int(args.max_issue_examples),
    )
    print(json.dumps(summary.to_json(), indent=2))
    assert (
        summary.rows_invalid == 0
    ), "Dataset contains rows with invalid think structure."
    print("ok: think-block structure validation passed")


if __name__ == "__main__":
    main()
