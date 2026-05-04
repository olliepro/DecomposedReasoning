from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import tiktoken
from tqdm import tqdm

from pipeline_common import compute_think_token_count, iter_jsonl

EXEC_PATTERN = re.compile(r"(<exec>\n?)(.*?)(\n?</exec>)", re.DOTALL)


@dataclass(frozen=True)
class CurationConfig:
    """Parameters for seeded exec-block truncation.

    Args:
        input_path: Source JSONL file.
        output_path: Destination JSONL file.
        summary_path: Destination summary JSON file.
        encoding_name: Tokenizer encoding name.
        seed: Random seed for block selection.
        over_1024_probability: Fraction of `>1024` exec blocks to truncate.
        over_512_probability: Fraction of `(512, 1024]` exec blocks to truncate.

    Example:
        config = CurationConfig(
            input_path=Path("input.jsonl"),
            output_path=Path("output.jsonl"),
            summary_path=Path("output.summary.json"),
            encoding_name="cl100k_base",
            seed=42,
            over_1024_probability=0.5,
            over_512_probability=0.25,
        )
    """

    input_path: Path
    output_path: Path
    summary_path: Path
    encoding_name: str
    seed: int
    over_1024_probability: float
    over_512_probability: float


@dataclass(frozen=True)
class BlockCandidate:
    """One eligible exec block address within the dataset.

    Args:
        row_index: Zero-based row index.
        message_index: Zero-based message index within the row.
        block_index: Zero-based exec index within the message.
        token_count: Original exec token count.
        bucket: Eligibility bucket.
    """

    row_index: int
    message_index: int
    block_index: int
    token_count: int
    bucket: str

    def key(self) -> tuple[int, int, int]:
        """Return the stable block address tuple.

        Returns:
            `(row_index, message_index, block_index)`.
        """

        return (self.row_index, self.message_index, self.block_index)


@dataclass(frozen=True)
class BlockStats:
    """Token counts for one exec block before and after curation.

    Args:
        before_tokens: Original token count.
        after_tokens: Final token count.
        bucket: Eligibility bucket or `unchanged`.
        selected: Whether the block was sampled for truncation.
    """

    before_tokens: int
    after_tokens: int
    bucket: str
    selected: bool


@dataclass(frozen=True)
class RowStats:
    """Per-row curation summary.

    Args:
        row_id: Stable row identifier.
        max_exec_before: Largest exec block before truncation.
        max_exec_after: Largest exec block after truncation.
        changed_blocks: Number of blocks changed in the row.
        think_token_count: Recomputed think token count.
    """

    row_id: str
    max_exec_before: int
    max_exec_after: int
    changed_blocks: int
    think_token_count: int


@dataclass(frozen=True)
class RunSummary:
    """Aggregate curation metrics written to disk.

    Args:
        input_path: Source dataset path.
        output_path: Curated dataset path.
        rows_input: Number of source rows.
        rows_output: Number of written rows.
        rows_changed: Rows with at least one changed exec block.
        eligible_blocks_over_1024: Count of blocks above 1024 tokens.
        eligible_blocks_over_512: Count of blocks in `(512, 1024]`.
        truncated_blocks_over_1024: Count truncated to 1024.
        truncated_blocks_over_512: Count truncated to 512.
        max_exec_before: Global max exec token count before truncation.
        max_exec_after: Global max exec token count after truncation.
    """

    input_path: str
    output_path: str
    rows_input: int
    rows_output: int
    rows_changed: int
    eligible_blocks_over_1024: int
    eligible_blocks_over_512: int
    truncated_blocks_over_1024: int
    truncated_blocks_over_512: int
    max_exec_before: int
    max_exec_after: int
    seed: int
    over_1024_probability: float
    over_512_probability: float


def parse_args() -> CurationConfig:
    """Parse CLI arguments into a typed config.

    Returns:
        Parsed curation config.
    """

    parser = argparse.ArgumentParser(
        description="Seeded truncation pass for long <exec> blocks."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path)
    parser.add_argument("--encoding-name", default="cl100k_base")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--over-1024-probability", type=float, default=0.5)
    parser.add_argument("--over-512-probability", type=float, default=0.25)
    args = parser.parse_args()
    summary_path = args.summary_path or args.output_path.with_suffix(".summary.json")
    return CurationConfig(
        input_path=args.input_path,
        output_path=args.output_path,
        summary_path=summary_path,
        encoding_name=args.encoding_name,
        seed=args.seed,
        over_1024_probability=args.over_1024_probability,
        over_512_probability=args.over_512_probability,
    )


def classify_bucket(token_count: int) -> str:
    """Classify one exec block into a truncation bucket.

    Args:
        token_count: Original exec token count.

    Returns:
        Bucket label.
    """

    if token_count > 1024:
        return "over_1024"
    if token_count > 512:
        return "over_512"
    return "unchanged"


def collect_block_candidates(
    *,
    rows: list[dict[str, object]],
    encoding: tiktoken.Encoding,
) -> list[BlockCandidate]:
    """Collect all eligible exec blocks with stable dataset addresses.

    Args:
        rows: Parsed dataset rows.
        encoding: Tokenizer encoding.

    Returns:
        Eligible block candidates in dataset order.
    """

    candidates: list[BlockCandidate] = []
    for row_index, row in enumerate(rows):
        messages = row["messages"]
        assert isinstance(messages, list)
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict) or message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            for block_index, (_, body, _) in enumerate(EXEC_PATTERN.findall(content)):
                token_count = len(encoding.encode(body))
                bucket = classify_bucket(token_count)
                if bucket == "unchanged":
                    continue
                candidates.append(
                    BlockCandidate(
                        row_index=row_index,
                        message_index=message_index,
                        block_index=block_index,
                        token_count=token_count,
                        bucket=bucket,
                    )
                )
    return candidates


def sample_block_keys(
    *,
    candidates: list[BlockCandidate],
    probability: float,
    rng: random.Random,
) -> set[tuple[int, int, int]]:
    """Sample an exact fraction of eligible blocks with a seeded RNG.

    Args:
        candidates: Eligible candidates within one bucket.
        probability: Fraction to sample.
        rng: Seeded random generator.

    Returns:
        Selected block-address keys.

    Example:
        keys = sample_block_keys(
            candidates=[candidate],
            probability=0.5,
            rng=random.Random(42),
        )
    """

    sample_size = round(len(candidates) * probability)
    if sample_size <= 0:
        return set()
    return {candidate.key() for candidate in rng.sample(candidates, sample_size)}


def build_selected_keys(
    *,
    candidates: list[BlockCandidate],
    config: CurationConfig,
    rng: random.Random,
) -> set[tuple[int, int, int]]:
    """Build the final set of selected block addresses.

    Args:
        candidates: All eligible block candidates.
        config: Curation policy.
        rng: Seeded random generator.

    Returns:
        Selected block-address keys across both buckets.
    """

    high_candidates = [item for item in candidates if item.bucket == "over_1024"]
    medium_candidates = [item for item in candidates if item.bucket == "over_512"]
    selected_high = sample_block_keys(
        candidates=high_candidates,
        probability=config.over_1024_probability,
        rng=rng,
    )
    selected_medium = sample_block_keys(
        candidates=medium_candidates,
        probability=config.over_512_probability,
        rng=rng,
    )
    return selected_high | selected_medium


def truncate_text(
    *,
    text: str,
    encoding: tiktoken.Encoding,
    limit: int | None,
) -> tuple[str, int]:
    """Truncate text to a token limit when requested.

    Args:
        text: Exec block body.
        encoding: Tokenizer encoding.
        limit: Target token cap or None.

    Returns:
        Tuple of `(possibly_truncated_text, final_token_count)`.
    """

    token_ids = encoding.encode(text)
    if limit is None or len(token_ids) <= limit:
        return text, len(token_ids)
    truncated_text = encoding.decode(token_ids[:limit]).rstrip()
    return truncated_text, len(encoding.encode(truncated_text))


def rewrite_assistant_content(
    *,
    row_index: int,
    message_index: int,
    content: str,
    encoding: tiktoken.Encoding,
    selected_keys: set[tuple[int, int, int]],
) -> tuple[str, list[BlockStats]]:
    """Apply truncation policy to all exec blocks in one assistant message.

    Args:
        row_index: Zero-based row index.
        message_index: Zero-based message index.
        content: Assistant message content.
        encoding: Tokenizer encoding.
        selected_keys: Selected block-address keys.

    Returns:
        Updated content and per-block stats.
    """

    block_stats: list[BlockStats] = []
    current_block_index = -1

    def replace(match: re.Match[str]) -> str:
        nonlocal current_block_index
        current_block_index += 1
        prefix, body, suffix = match.groups()
        before_tokens = len(encoding.encode(body))
        bucket = classify_bucket(before_tokens)
        selected = (row_index, message_index, current_block_index) in selected_keys
        limit = None
        if selected and bucket == "over_1024":
            limit = 1024
        if selected and bucket == "over_512":
            limit = 512
        updated_body, after_tokens = truncate_text(
            text=body,
            encoding=encoding,
            limit=limit,
        )
        block_stats.append(
            BlockStats(
                before_tokens=before_tokens,
                after_tokens=after_tokens,
                bucket=bucket,
                selected=selected,
            )
        )
        return f"{prefix}{updated_body}{suffix}"

    return EXEC_PATTERN.sub(replace, content), block_stats


def update_row(
    *,
    row_index: int,
    row: dict[str, object],
    encoding: tiktoken.Encoding,
    selected_keys: set[tuple[int, int, int]],
) -> tuple[dict[str, object], RowStats, list[BlockStats]]:
    """Return a row with truncated exec blocks and refreshed token counts.

    Args:
        row_index: Zero-based row index.
        row: Parsed JSONL row.
        encoding: Tokenizer encoding.
        selected_keys: Selected block-address keys.

    Returns:
        Tuple of updated row, row stats, and all block stats.
    """

    updated_row = dict(row)
    raw_messages = row["messages"]
    assert isinstance(raw_messages, list)
    messages = [dict(message) for message in raw_messages if isinstance(message, dict)]
    block_stats: list[BlockStats] = []
    for message_index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        updated_content, message_stats = rewrite_assistant_content(
            row_index=row_index,
            message_index=message_index,
            content=content,
            encoding=encoding,
            selected_keys=selected_keys,
        )
        message["content"] = updated_content
        block_stats.extend(message_stats)
    updated_row["messages"] = messages
    think_token_count = compute_think_token_count(encoding=encoding, messages=messages)
    updated_row["think_token_count"] = think_token_count
    max_exec_before = max((item.before_tokens for item in block_stats), default=0)
    max_exec_after = max((item.after_tokens for item in block_stats), default=0)
    changed_blocks = sum(item.before_tokens != item.after_tokens for item in block_stats)
    row_stats = RowStats(
        row_id=str(row.get("id", "")),
        max_exec_before=max_exec_before,
        max_exec_after=max_exec_after,
        changed_blocks=changed_blocks,
        think_token_count=think_token_count,
    )
    return updated_row, row_stats, block_stats


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    """Write rows to a JSONL file.

    Args:
        path: Destination path.
        rows: Parsed row objects.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def summarize_run(
    *,
    config: CurationConfig,
    row_stats: list[RowStats],
    block_stats: list[BlockStats],
) -> RunSummary:
    """Build aggregate stats for the truncation run.

    Args:
        config: Curation policy.
        row_stats: Per-row stats.
        block_stats: Per-block stats.

    Returns:
        Aggregate summary object.
    """

    return RunSummary(
        input_path=str(config.input_path),
        output_path=str(config.output_path),
        rows_input=len(row_stats),
        rows_output=len(row_stats),
        rows_changed=sum(item.changed_blocks > 0 for item in row_stats),
        eligible_blocks_over_1024=sum(item.bucket == "over_1024" for item in block_stats),
        eligible_blocks_over_512=sum(item.bucket == "over_512" for item in block_stats),
        truncated_blocks_over_1024=sum(
            item.bucket == "over_1024" and item.selected for item in block_stats
        ),
        truncated_blocks_over_512=sum(
            item.bucket == "over_512" and item.selected for item in block_stats
        ),
        max_exec_before=max((item.max_exec_before for item in row_stats), default=0),
        max_exec_after=max((item.max_exec_after for item in row_stats), default=0),
        seed=config.seed,
        over_1024_probability=config.over_1024_probability,
        over_512_probability=config.over_512_probability,
    )


def main() -> None:
    """Run seeded truncation over one JSONL dataset.

    Example:
        python curate_exec_blocks.py \\
            --input-path input.jsonl \\
            --output-path output.jsonl
    """

    config = parse_args()
    assert config.input_path.exists(), f"Missing input file: {config.input_path}"
    encoding = tiktoken.get_encoding(config.encoding_name)
    rng = random.Random(config.seed)
    rows = list(iter_jsonl(config.input_path))
    candidates = collect_block_candidates(rows=rows, encoding=encoding)
    selected_keys = build_selected_keys(
        candidates=candidates,
        config=config,
        rng=rng,
    )
    updated_rows: list[dict[str, object]] = []
    row_stats: list[RowStats] = []
    block_stats: list[BlockStats] = []

    for row_index, row in enumerate(
        tqdm(rows, total=len(rows), desc="Curating exec blocks", unit="row")
    ):
        updated_row, current_row_stats, current_block_stats = update_row(
            row_index=row_index,
            row=row,
            encoding=encoding,
            selected_keys=selected_keys,
        )
        updated_rows.append(updated_row)
        row_stats.append(current_row_stats)
        block_stats.extend(current_block_stats)

    write_jsonl(path=config.output_path, rows=updated_rows)
    config.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = summarize_run(
        config=config,
        row_stats=row_stats,
        block_stats=block_stats,
    )
    config.summary_path.write_text(
        json.dumps(asdict(summary), indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
