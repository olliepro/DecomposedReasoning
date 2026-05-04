from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import tiktoken

from .non_sequitur_builder import row_complete_prompt_token_count
from .trace_augmentor import TraceBlock, extract_trace_text, parse_trace_text


@dataclass(frozen=True)
class PreparedRow:
    """One source row that passed non-sequitur sampling filters.

    Args:
        row_index: Zero-based row index in the source dataset.
        row: Original dataset row.
        parsed_blocks: Parsed `<steer>/<exec>` blocks from the assistant think trace.
        complete_prompt_token_count: Full-row message token count using `cl100k_base`.

    Example:
        >>> PreparedRow(
        ...     row_index=0,
        ...     row={"id": "row-1"},
        ...     parsed_blocks=(),
        ...     complete_prompt_token_count=42,
        ... ).row_index
        0
    """

    row_index: int
    row: dict[str, Any]
    parsed_blocks: tuple[TraceBlock, ...]
    complete_prompt_token_count: int


@dataclass(frozen=True)
class SourcePrepSummary:
    """Summary of source-row filtering for non-sequitur augmentation.

    Args:
        total_rows: Total input rows scanned.
        eligible_rows: Rows that passed all filters.
        skipped_over_token_limit: Rows with full-row tokens `>= max_source_tokens`.
        skipped_invalid_trace: Rows whose assistant trace could not be parsed.
        skipped_short_trace: Rows with fewer than two steer/exec pairs.
        skipped_invalid_blocks: Rows that failed interleaving or exec-limit checks.
    """

    total_rows: int
    eligible_rows: int
    skipped_over_token_limit: int
    skipped_invalid_trace: int
    skipped_short_trace: int
    skipped_invalid_blocks: int

    def to_json(self) -> dict[str, int]:
        """Return a JSON-friendly summary dictionary.

        Returns:
            Flat count mapping for summary files.
        """

        return {
            "total_rows": self.total_rows,
            "eligible_rows": self.eligible_rows,
            "skipped_over_token_limit": self.skipped_over_token_limit,
            "skipped_invalid_trace": self.skipped_invalid_trace,
            "skipped_short_trace": self.skipped_short_trace,
            "skipped_invalid_blocks": self.skipped_invalid_blocks,
        }


def prepare_source_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    encoding: tiktoken.Encoding,
    max_source_tokens: int,
) -> tuple[list[PreparedRow], SourcePrepSummary]:
    """Filter source rows for non-sequitur augmentation.

    Args:
        rows: Candidate dataset rows.
        encoding: `cl100k_base` tokenizer for full-row counts.
        max_source_tokens: Strict full-row source limit.

    Returns:
        Prepared rows and a filtering summary.
    """

    prepared_rows: list[PreparedRow] = []
    skipped_over_token_limit = 0
    skipped_invalid_trace = 0
    skipped_short_trace = 0
    skipped_invalid_blocks = 0

    for row_index, raw_row in enumerate(rows):
        full_row_tokens = row_complete_prompt_token_count(
            row=raw_row,
            encoding=encoding,
        )
        if full_row_tokens >= max_source_tokens:
            skipped_over_token_limit += 1
            continue
        try:
            parsed_blocks = tuple(parse_trace_text(extract_trace_text(dict(raw_row))))
        except Exception:  # noqa: BLE001
            skipped_invalid_trace += 1
            continue
        if len(parsed_blocks) < 4:
            skipped_short_trace += 1
            continue
        block_errors = validate_source_blocks(parsed_blocks=parsed_blocks)
        if block_errors:
            skipped_invalid_blocks += 1
            continue
        prepared_rows.append(
            PreparedRow(
                row_index=row_index,
                row=dict(raw_row),
                parsed_blocks=parsed_blocks,
                complete_prompt_token_count=full_row_tokens,
            )
        )

    summary = SourcePrepSummary(
        total_rows=len(rows),
        eligible_rows=len(prepared_rows),
        skipped_over_token_limit=skipped_over_token_limit,
        skipped_invalid_trace=skipped_invalid_trace,
        skipped_short_trace=skipped_short_trace,
        skipped_invalid_blocks=skipped_invalid_blocks,
    )
    return prepared_rows, summary


def validate_source_blocks(parsed_blocks: Sequence[TraceBlock]) -> list[str]:
    """Check structural validity for source blocks without length filtering.

    Args:
        parsed_blocks: Parsed interleaved source blocks.

    Returns:
        List of structural validation errors.
    """

    errors: list[str] = []
    if not parsed_blocks:
        return ["no steer/exec blocks found"]
    if len(parsed_blocks) % 2 != 0:
        errors.append(f"block count must be even, got {len(parsed_blocks)}")
    if parsed_blocks[0].type != "steer":
        errors.append(f"first block must be steer, got {parsed_blocks[0].type}")
    for index, block in enumerate(parsed_blocks):
        expected_type = "steer" if index % 2 == 0 else "exec"
        if block.type != expected_type:
            errors.append(
                f"block index {index} must be {expected_type}, got {block.type}"
            )
        if not block.text.strip():
            errors.append(f"block index {index} is empty")
    return errors
