from __future__ import annotations

import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import tiktoken
from datasets import load_dataset, load_dataset_builder
from tqdm import tqdm

from pipeline_common import (
    compute_think_token_count,
    count_jsonl_rows,
    decode_json_line,
    iter_jsonl,
    write_jsonl,
)
from pipeline_types import FilterConfig, SamplingConfig, StratifyConfig


def list_shard_files(dataset_name: str, split: str) -> list[str]:
    """List data shard files for a split.

    Args:
        dataset_name: Dataset name.
        split: Split name.

    Returns:
        Shard file list.
    """
    builder = load_dataset_builder(dataset_name)
    data_files = getattr(builder.config, "data_files", None)
    if not data_files or split not in data_files:
        return []
    return list(data_files[split])


def select_shards(shard_files: list[str], num_shards: int, seed: int) -> list[str]:
    """Randomly select shards.

    Args:
        shard_files: All shard file paths.
        num_shards: Number of shards to select.
        seed: Random seed.

    Returns:
        Selected shard paths.
    """
    if num_shards >= len(shard_files):
        return shard_files
    rng = random.Random(seed)
    indexes = sorted(rng.sample(range(len(shard_files)), k=num_shards))
    return [shard_files[index] for index in indexes]


def iter_rows_from_shard(
    dataset_name: str,
    split: str,
    shard_file: str,
    seed: int,
    shuffle_buffer: int,
) -> Iterable[dict[str, object]]:
    """Stream rows from one shard.

    Args:
        dataset_name: Dataset name.
        split: Split name.
        shard_file: Shard file path.
        seed: Shuffle seed.
        shuffle_buffer: Shuffle buffer size.

    Returns:
        Row iterator.
    """
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=True,
        data_files={split: [shard_file]},
    )
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
    return dataset


def run_sample_stage(
    sampling: SamplingConfig,
    output_path: Path,
    resume: bool,
    dry_run: bool,
) -> dict[str, object]:
    """Run sample stage.

    Args:
        sampling: Sampling config.
        output_path: Raw sample output path.
        resume: Resume from existing output.
        dry_run: Skip writes when true.

    Returns:
        Stage metadata.
    """
    shard_files = list_shard_files(dataset_name=sampling.dataset_name, split=sampling.split)
    selected_shards = select_shards(
        shard_files=shard_files,
        num_shards=sampling.num_shards,
        seed=sampling.seed,
    )
    target_rows = len(selected_shards) * sampling.rows_per_shard
    existing_rows = count_jsonl_rows(output_path) if resume else 0
    rows_left = max(target_rows - existing_rows, 0)

    if dry_run:
        return {
            "target_rows": target_rows,
            "existing_rows": existing_rows,
            "rows_left": rows_left,
            "selected_shards": len(selected_shards),
        }
    if rows_left == 0:
        return {
            "target_rows": target_rows,
            "existing_rows": existing_rows,
            "rows_left": 0,
            "selected_shards": len(selected_shards),
            "skipped": True,
        }

    if not resume and output_path.exists():
        output_path.unlink()
    if not output_path.exists():
        existing_rows = 0

    current_index = 0
    written_rows = 0
    progress = tqdm(total=target_rows, desc="Sampling rows", unit="row")
    progress.update(existing_rows)

    for shard_index, shard_file in enumerate(selected_shards):
        shard_seed = sampling.seed + shard_index
        rows = iter_rows_from_shard(
            dataset_name=sampling.dataset_name,
            split=sampling.split,
            shard_file=shard_file,
            seed=shard_seed,
            shuffle_buffer=sampling.shuffle_buffer,
        )
        for row_index, row in enumerate(rows):
            if row_index >= sampling.rows_per_shard:
                break
            if current_index < existing_rows:
                current_index += 1
                continue
            output_row = dict(row)
            output_row["sample_meta"] = {
                "shard_index": shard_index,
                "shard_file": shard_file,
                "seed": shard_seed,
            }
            write_jsonl(output_path=output_path, row=output_row)
            current_index += 1
            written_rows += 1
            progress.update(1)
            if current_index >= target_rows:
                break
        if current_index >= target_rows:
            break
    progress.close()

    return {
        "target_rows": target_rows,
        "existing_rows": existing_rows,
        "written_rows": written_rows,
        "selected_shards": len(selected_shards),
        "rows_left": max(target_rows - (existing_rows + written_rows), 0),
    }


def run_filter_stage(
    filter_config: FilterConfig,
    input_path: Path,
    output_path: Path,
    dry_run: bool,
) -> dict[str, object]:
    """Run filter stage.

    Args:
        filter_config: Filter config.
        input_path: Raw sample path.
        output_path: Filtered output path.
        dry_run: Skip writes when true.

    Returns:
        Stage metadata.
    """
    if not input_path.exists():
        raise SystemExit(f"Missing raw sample: {input_path}")

    total_input = count_jsonl_rows(input_path)
    if dry_run:
        return {
            "input_rows": total_input,
            "min_tokens": filter_config.min_tokens,
            "max_tokens": filter_config.max_tokens,
        }

    if output_path.exists():
        output_path.unlink()

    encoding = tiktoken.get_encoding(filter_config.encoding_name)
    written_rows = 0
    multi_json_lines = 0
    counts_by_source: dict[str, int] = defaultdict(int)

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            tqdm(handle, desc="Filtering rows", unit="line"),
            start=1,
        ):
            if not line.strip():
                continue
            rows = decode_json_line(line=line, line_number=line_number)
            if len(rows) > 1:
                multi_json_lines += 1
            for row in rows:
                messages = row.get("messages", [])
                if not isinstance(messages, list):
                    continue
                token_count = compute_think_token_count(
                    encoding=encoding,
                    messages=messages,
                )
                if token_count < filter_config.min_tokens:
                    continue
                if token_count > filter_config.max_tokens:
                    continue
                source = str(row.get("dataset_source", "unknown"))
                row["think_token_count"] = token_count
                counts_by_source[source] += 1
                write_jsonl(output_path=output_path, row=row)
                written_rows += 1

    return {
        "input_rows": total_input,
        "written_rows": written_rows,
        "multi_json_lines": multi_json_lines,
        "counts_by_source": dict(counts_by_source),
    }


def compute_balanced_targets(counts_by_source: dict[str, int], target_rows: int) -> dict[str, int]:
    """Compute balanced source targets with source caps.

    Args:
        counts_by_source: Available rows by source.
        target_rows: Target final rows.

    Returns:
        Target rows by source.

    Example:
        >>> compute_balanced_targets({"a": 10, "b": 2}, target_rows=6)
        {'a': 4, 'b': 2}
    """
    total_available = sum(counts_by_source.values())
    if total_available < target_rows:
        raise SystemExit(
            f"Not enough rows after filtering ({total_available}) for target {target_rows}."
        )

    sources = sorted(counts_by_source.keys())
    targets = {source: 0 for source in sources}
    remaining = target_rows
    eligible = set(sources)

    while remaining > 0 and eligible:
        per_source = max(1, remaining // len(eligible))
        progressed = False
        for source in list(eligible):
            capacity = counts_by_source[source] - targets[source]
            if capacity <= 0:
                eligible.remove(source)
                continue
            add = min(per_source, capacity, remaining)
            targets[source] += add
            remaining -= add
            progressed = True
            if targets[source] >= counts_by_source[source]:
                eligible.remove(source)
            if remaining == 0:
                break
        if not progressed:
            break

    if remaining > 0:
        sources_with_capacity = [
            source for source in sources if targets[source] < counts_by_source[source]
        ]
        index = 0
        while remaining > 0 and sources_with_capacity:
            source = sources_with_capacity[index % len(sources_with_capacity)]
            if targets[source] < counts_by_source[source]:
                targets[source] += 1
                remaining -= 1
            index += 1

    if sum(targets.values()) != target_rows:
        raise SystemExit("Failed to allocate balanced targets.")
    return targets


def run_stratify_stage(
    stratify: StratifyConfig,
    input_path: Path,
    output_path: Path,
    dry_run: bool,
) -> dict[str, object]:
    """Run stratify stage.

    Args:
        stratify: Stratify config.
        input_path: Filtered input path.
        output_path: Stratified output path.
        dry_run: Skip writes when true.

    Returns:
        Stage metadata.
    """
    if not input_path.exists():
        raise SystemExit(f"Missing filtered rows: {input_path}")

    counts_by_source = Counter()
    total_rows = 0
    for row in iter_jsonl(path=input_path):
        source = str(row.get("dataset_source", "unknown"))
        counts_by_source[source] += 1
        total_rows += 1

    targets = compute_balanced_targets(
        counts_by_source=dict(counts_by_source),
        target_rows=stratify.target_rows,
    )
    if dry_run:
        return {
            "input_rows": total_rows,
            "targets": targets,
            "counts_by_source": dict(counts_by_source),
        }

    if output_path.exists():
        output_path.unlink()

    rng = random.Random(stratify.seed)
    reservoirs: dict[str, list[dict[str, object]]] = {
        source: [] for source, target in targets.items() if target > 0
    }
    seen: dict[str, int] = defaultdict(int)

    with input_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(
            tqdm(handle, desc="Stratified sampling", unit="line"),
            start=1,
        ):
            if not line.strip():
                continue
            rows = decode_json_line(line=line, line_number=line_number)
            for row in rows:
                source = str(row.get("dataset_source", "unknown"))
                target = targets.get(source, 0)
                if target == 0:
                    continue
                seen[source] += 1
                reservoir = reservoirs[source]
                if len(reservoir) < target:
                    reservoir.append(row)
                    continue
                index = rng.randint(1, seen[source])
                if index <= target:
                    reservoir[index - 1] = row

    sampled_rows: list[dict[str, object]] = []
    for source in sorted(reservoirs.keys()):
        sampled_rows.extend(reservoirs[source])
    rng.shuffle(sampled_rows)
    for row in sampled_rows:
        write_jsonl(output_path=output_path, row=row)

    return {
        "input_rows": total_rows,
        "output_rows": len(sampled_rows),
        "targets": targets,
        "counts_by_source": dict(counts_by_source),
    }
