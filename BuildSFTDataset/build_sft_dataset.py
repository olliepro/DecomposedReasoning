from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, cast

import tiktoken
from datasets import load_dataset, load_dataset_builder
from google import genai
from google.genai import types
from tqdm import tqdm

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
STAGE_ORDER = ("sample", "filter", "stratify", "transform")

DEFAULT_DATASET = "allenai/Dolci-Think-SFT-7B"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_STATE_FILE = "pipeline_state.json"
DEFAULT_ENV_PATH = Path(".env")

DEFAULT_ROWS_PER_SHARD = 3000
DEFAULT_NUM_SHARDS = 20
DEFAULT_SEED = 42
DEFAULT_SHUFFLE_BUFFER = 10000
DEFAULT_MIN_TOKENS = 500
DEFAULT_MAX_TOKENS = 3000
DEFAULT_ENCODING = "cl100k_base"
DEFAULT_TARGET_ROWS = 2000

DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_MODE = "gemini"
DEFAULT_MAX_OUTPUT_TOKENS = 20000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_THINKING_LEVEL = "low"
DEFAULT_BATCH = True
DEFAULT_MAX_ROWS = 2000
DEFAULT_CONFIRM_THRESHOLD = 100
DEFAULT_RETRY_LIMIT = 3
DEFAULT_RETRY_SLEEP = 2.0
DEFAULT_BATCH_POLL_SECONDS = 5.0

DEFAULT_SYSTEM_PROMPT = Path("system_prompt.md")
DEFAULT_USER_PROMPT = Path("user_prompt.md")

Mode = Literal["gemini", "express", "vertex"]
Stage = Literal["sample", "filter", "stratify", "transform"]


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for sample stage.

    Args:
        dataset_name: Hugging Face dataset name.
        split: Dataset split.
        rows_per_shard: Number of rows sampled per shard.
        num_shards: Number of shards to include.
        seed: Random seed.
        shuffle_buffer: Streaming shuffle buffer size.
    """

    dataset_name: str
    split: str
    rows_per_shard: int
    num_shards: int
    seed: int
    shuffle_buffer: int


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for filter stage.

    Args:
        min_tokens: Minimum think token count.
        max_tokens: Maximum think token count.
        encoding_name: Tiktoken encoding name.
    """

    min_tokens: int
    max_tokens: int
    encoding_name: str


@dataclass(frozen=True)
class StratifyConfig:
    """Configuration for stratify stage.

    Args:
        target_rows: Final row count.
        seed: Random seed.
    """

    target_rows: int
    seed: int


@dataclass(frozen=True)
class TransformConfig:
    """Configuration for transform stage.

    Args:
        mode: API mode.
        model_id: Model ID.
        api_key: API key.
        project_id: GCP project for vertex mode.
        location: GCP location for vertex mode.
        max_output_tokens: Max output tokens.
        temperature: Sampling temperature.
        thinking_level: Gemini reasoning level.
        batch: Whether to use batch API.
        retry_limit: Retry count for non-batch calls.
        retry_sleep_seconds: Backoff base.
        batch_poll_seconds: Batch polling interval.
        max_rows: Target number of transformed rows in output.
        dry_run: If true, do not call APIs.
    """

    mode: Mode
    model_id: str
    api_key: str
    project_id: str | None
    location: str | None
    max_output_tokens: int
    temperature: float
    thinking_level: str | None
    batch: bool
    retry_limit: int
    retry_sleep_seconds: float
    batch_poll_seconds: float
    max_rows: int
    dry_run: bool


@dataclass(frozen=True)
class PromptConfig:
    """Prompt file paths for transform stage.

    Args:
        system_prompt_path: System prompt markdown path.
        user_prompt_path: User prompt markdown path.
    """

    system_prompt_path: Path
    user_prompt_path: Path


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options shared across stages.

    Args:
        output_dir: Output directory.
        resume: Resume outputs when possible.
        auto_yes: Skip interactive confirmation.
        dry_run: If true, avoid writes/API requests.
        stage: Requested stage or special mode.
        confirm_threshold: Confirmation threshold for rows left.
    """

    output_dir: Path
    resume: bool
    auto_yes: bool
    dry_run: bool
    stage: str | None
    confirm_threshold: int


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved pipeline output paths.

    Args:
        output_dir: Output directory.
        raw_sample_path: Raw sampled JSONL.
        filtered_path: Token-filtered JSONL.
        stratified_path: Stratified final sample JSONL.
        transformed_path: Transformed output JSONL.
        state_path: Pipeline state JSON path.
    """

    output_dir: Path
    raw_sample_path: Path
    filtered_path: Path
    stratified_path: Path
    transformed_path: Path
    state_path: Path


@dataclass(frozen=True)
class StageStatus:
    """Persisted stage metadata.

    Args:
        completed: Whether stage completed.
        config_hash: Hash of config used.
        updated_at: UTC timestamp when marked.
        metadata: Arbitrary stage metadata.
    """

    completed: bool
    config_hash: str
    updated_at: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class ThinkTask:
    """Mapping of one transformed think block to source row/message.

    Args:
        row_index: Index in buffered rows.
        message_index: Message index in a row.
        block_index: Think block index in a message.
    """

    row_index: int
    message_index: int
    block_index: int


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a simple dotenv file into key/value pairs.

    Args:
        path: Dotenv file path.

    Returns:
        Mapping of environment variables.
    """
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip("\"").strip("'")
    return values


def resolve_paths(runtime: RuntimeConfig) -> PipelinePaths:
    """Build output paths from runtime config.

    Args:
        runtime: Runtime options.

    Returns:
        Resolved output paths.
    """
    output_dir = runtime.output_dir
    return PipelinePaths(
        output_dir=output_dir,
        raw_sample_path=output_dir / "raw_sample.jsonl",
        filtered_path=output_dir / "filtered_candidates.jsonl",
        stratified_path=output_dir / "stratified_sample.jsonl",
        transformed_path=output_dir / "transformed_output.jsonl",
        state_path=output_dir / DEFAULT_STATE_FILE,
    )


def utc_now() -> str:
    """Return current UTC timestamp string."""
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(output_path: Path, row: dict[str, object]) -> None:
    """Append one object to JSONL.

    Args:
        output_path: Target JSONL path.
        row: JSON-serializable row.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def decode_json_line(line: str, line_number: int) -> list[dict[str, object]]:
    """Decode one line into one or more JSON objects.

    Args:
        line: JSONL line.
        line_number: 1-based line number.

    Returns:
        List of decoded objects.

    Example:
        >>> decode_json_line('{"a":1}{"b":2}', line_number=1)
        [{'a': 1}, {'b': 2}]
    """
    decoder = json.JSONDecoder()
    index = 0
    objects: list[dict[str, object]] = []
    while index < len(line):
        while index < len(line) and line[index].isspace():
            index += 1
        if index >= len(line):
            break
        obj, end = decoder.raw_decode(line, index)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected object at line {line_number} offset {index}")
        objects.append(obj)
        index = end
    return objects


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty lines in a JSONL file.

    Args:
        path: JSONL path.

    Returns:
        Number of non-empty rows.
    """
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Yield JSON objects from a JSONL file.

    Args:
        path: JSONL path.

    Yields:
        Decoded rows.
    """
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            for row in decode_json_line(line=line, line_number=line_number):
                yield row


def extract_think_blocks(text: str) -> list[str]:
    """Extract think blocks from assistant content.

    Args:
        text: Assistant message content.

    Returns:
        Extracted think blocks.
    """
    return [match.strip() for match in THINK_PATTERN.findall(text)]


def compute_think_token_count(encoding: tiktoken.Encoding, messages: list[dict[str, object]]) -> int:
    """Compute total tokens in all assistant think blocks.

    Args:
        encoding: Tiktoken encoding.
        messages: Conversation messages.

    Returns:
        Total token count for think blocks.
    """
    total = 0
    for message in messages:
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        for block in extract_think_blocks(text=content):
            total += len(encoding.encode(block))
    return total


def hash_config(value: dict[str, object]) -> str:
    """Hash a stage config dictionary.

    Args:
        value: Serializable configuration map.

    Returns:
        SHA256 hash.
    """
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_state(path: Path) -> dict[str, object]:
    """Load pipeline state or create a default state object.

    Args:
        path: State JSON file.

    Returns:
        Pipeline state dictionary.
    """
    if not path.exists():
        return {"version": 1, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, object]) -> None:
    """Persist pipeline state.

    Args:
        path: State JSON file.
        state: State payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_stage_status(state: dict[str, object], stage: Stage) -> StageStatus | None:
    """Fetch stage status from state.

    Args:
        state: Pipeline state object.
        stage: Stage name.

    Returns:
        Stage status or None.
    """
    stages = state.get("stages", {})
    payload = stages.get(stage)
    if not isinstance(payload, dict):
        return None
    return StageStatus(
        completed=bool(payload.get("completed", False)),
        config_hash=str(payload.get("config_hash", "")),
        updated_at=str(payload.get("updated_at", "")),
        metadata=dict(payload.get("metadata", {})),
    )


def set_stage_status(
    state: dict[str, object],
    stage: Stage,
    completed: bool,
    config_hash: str,
    metadata: dict[str, object],
) -> None:
    """Upsert one stage status into pipeline state.

    Args:
        state: Mutable pipeline state.
        stage: Stage name.
        completed: Completion flag.
        config_hash: Config hash for this run.
        metadata: Stage metadata.
    """
    stages = state.setdefault("stages", {})
    assert isinstance(stages, dict)
    stages[stage] = {
        "completed": completed,
        "config_hash": config_hash,
        "updated_at": utc_now(),
        "metadata": metadata,
    }


def is_stage_complete(
    state: dict[str, object],
    stage: Stage,
    expected_hash: str,
    required_path: Path,
) -> bool:
    """Check whether a stage is validly completed.

    Args:
        state: Pipeline state object.
        stage: Stage name.
        expected_hash: Current config hash for stage.
        required_path: Stage output that must exist.

    Returns:
        True if stage can be treated as complete.
    """
    status = get_stage_status(state=state, stage=stage)
    if status is None:
        return False
    return status.completed and status.config_hash == expected_hash and required_path.exists()


def choose_auto_stage(completed: dict[Stage, bool]) -> Stage | None:
    """Pick the next incomplete stage in order.

    Args:
        completed: Completion flags by stage.

    Returns:
        Next stage or None if all done.
    """
    for stage in STAGE_ORDER:
        if not completed[stage]:
            return stage
    return None


def confirm_large_work(stage: Stage, rows_left: int, runtime: RuntimeConfig) -> None:
    """Confirm long-running work before execution.

    Args:
        stage: Stage name.
        rows_left: Number of rows left to process.
        runtime: Runtime options.
    """
    if runtime.dry_run or runtime.auto_yes:
        return
    if rows_left <= runtime.confirm_threshold:
        return
    prompt = (
        f"Stage '{stage}' has {rows_left} rows left to process. "
        "Continue? [y/N]: "
    )
    answer = input(prompt).strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted by user.")


def list_shard_files(dataset_name: str, split: str) -> list[str]:
    """List split shard files for a dataset.

    Args:
        dataset_name: Hugging Face dataset name.
        split: Dataset split.

    Returns:
        Ordered shard paths.
    """
    builder = load_dataset_builder(dataset_name)
    data_files = getattr(builder.config, "data_files", None)
    if not data_files or split not in data_files:
        return []
    return list(data_files[split])


def select_shards(shard_files: list[str], num_shards: int, seed: int) -> list[str]:
    """Randomly select shards.

    Args:
        shard_files: Full shard file list.
        num_shards: Number to select.
        seed: Random seed.

    Returns:
        Selected shard list.
    """
    if num_shards >= len(shard_files):
        return shard_files
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(shard_files)), k=num_shards))
    return [shard_files[index] for index in indices]


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
    paths: PipelinePaths,
    runtime: RuntimeConfig,
) -> dict[str, object]:
    """Run sample stage with deterministic resume support.

    Args:
        sampling: Sampling config.
        paths: Pipeline paths.
        runtime: Runtime options.

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

    existing_rows = count_jsonl_rows(paths.raw_sample_path) if runtime.resume else 0
    rows_left = max(target_rows - existing_rows, 0)
    confirm_large_work(stage="sample", rows_left=rows_left, runtime=runtime)

    if runtime.dry_run:
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

    if not runtime.resume and paths.raw_sample_path.exists():
        paths.raw_sample_path.unlink()
    if not paths.raw_sample_path.exists():
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
            if current_index >= target_rows:
                break
            output_row = dict(row)
            output_row["sample_meta"] = {
                "shard_index": shard_index,
                "shard_file": shard_file,
                "seed": shard_seed,
            }
            write_jsonl(output_path=paths.raw_sample_path, row=output_row)
            current_index += 1
            written_rows += 1
            progress.update(1)
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
    paths: PipelinePaths,
    runtime: RuntimeConfig,
) -> dict[str, object]:
    """Run filter stage.

    Args:
        filter_config: Filter config.
        paths: Pipeline paths.
        runtime: Runtime options.

    Returns:
        Stage metadata.
    """
    if not paths.raw_sample_path.exists():
        raise SystemExit(f"Missing raw sample: {paths.raw_sample_path}")

    total_input = count_jsonl_rows(paths.raw_sample_path)
    if runtime.dry_run:
        return {
            "input_rows": total_input,
            "min_tokens": filter_config.min_tokens,
            "max_tokens": filter_config.max_tokens,
        }

    encoding = tiktoken.get_encoding(filter_config.encoding_name)
    counts_by_source: dict[str, int] = defaultdict(int)
    written_rows = 0
    multi_json_lines = 0

    if paths.filtered_path.exists():
        paths.filtered_path.unlink()

    with paths.raw_sample_path.open("r", encoding="utf-8") as handle:
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
                write_jsonl(output_path=paths.filtered_path, row=row)
                written_rows += 1

    return {
        "input_rows": total_input,
        "written_rows": written_rows,
        "multi_json_lines": multi_json_lines,
        "counts_by_source": dict(counts_by_source),
    }


def compute_balanced_targets(
    counts_by_source: dict[str, int],
    target_rows: int,
) -> dict[str, int]:
    """Allocate balanced targets across sources with caps.

    Args:
        counts_by_source: Available rows by source.
        target_rows: Desired final row count.

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
        available_sources = [s for s in sources if targets[s] < counts_by_source[s]]
        idx = 0
        while remaining > 0 and available_sources:
            source = available_sources[idx % len(available_sources)]
            if targets[source] < counts_by_source[source]:
                targets[source] += 1
                remaining -= 1
            idx += 1

    if sum(targets.values()) != target_rows:
        raise SystemExit("Failed to allocate balanced targets.")
    return targets


def run_stratify_stage(
    stratify: StratifyConfig,
    paths: PipelinePaths,
    runtime: RuntimeConfig,
) -> dict[str, object]:
    """Run stratify stage using balanced targets and reservoir sampling.

    Args:
        stratify: Stratify config.
        paths: Pipeline paths.
        runtime: Runtime options.

    Returns:
        Stage metadata.
    """
    if not paths.filtered_path.exists():
        raise SystemExit(f"Missing filtered rows: {paths.filtered_path}")

    counts_by_source: Counter[str] = Counter()
    total_rows = 0
    with paths.filtered_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            rows = decode_json_line(line=line, line_number=line_number)
            for row in rows:
                source = str(row.get("dataset_source", "unknown"))
                counts_by_source[source] += 1
                total_rows += 1

    targets = compute_balanced_targets(
        counts_by_source=dict(counts_by_source),
        target_rows=stratify.target_rows,
    )

    if runtime.dry_run:
        return {
            "input_rows": total_rows,
            "targets": targets,
            "counts_by_source": dict(counts_by_source),
        }

    rng = random.Random(stratify.seed)
    reservoirs: dict[str, list[dict[str, object]]] = {
        source: [] for source, target in targets.items() if target > 0
    }
    seen: dict[str, int] = defaultdict(int)

    with paths.filtered_path.open("r", encoding="utf-8") as handle:
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

    if paths.stratified_path.exists():
        paths.stratified_path.unlink()
    for row in sampled_rows:
        write_jsonl(output_path=paths.stratified_path, row=row)

    return {
        "input_rows": total_rows,
        "output_rows": len(sampled_rows),
        "targets": targets,
        "counts_by_source": dict(counts_by_source),
    }


def clean_model_output(text: str) -> str:
    """Normalize model output by removing code fences and padding.

    Args:
        text: Raw model output.

    Returns:
        Cleaned text.

    Example:
        >>> clean_model_output("```xml\\n<x/>\\n```")
        '<x/>'
    """
    cleaned = text.lstrip("\ufeff").strip()
    if "```" in cleaned:
        cleaned = cleaned[cleaned.find("```") :]
    cleaned = re.sub(r"^```(?:[a-zA-Z0-9_-]+)?\s*", "", cleaned)
    if "```" in cleaned:
        cleaned = cleaned[: cleaned.rfind("```")]
    return cleaned.strip()


def extract_response_text(response: object) -> str:
    """Extract text content from response object.

    Args:
        response: API response object.

    Returns:
        Extracted text.
    """
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        return text
    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""
    first = candidates[0]
    content = getattr(first, "content", None)
    parts = getattr(content, "parts", None) or []
    texts = [getattr(part, "text", "") for part in parts if isinstance(getattr(part, "text", None), str)]
    return "".join(texts)


def normalize_thinking_level(level: str | None) -> str | None:
    """Normalize thinking level from CLI.

    Args:
        level: Raw thinking level.

    Returns:
        Uppercased level, or None if disabled.
    """
    if level is None:
        return None
    if level.lower() == "none":
        return None
    return level.strip().upper()


def validate_thinking_level(model_id: str, thinking_level: str | None) -> str | None:
    """Validate thinking level compatibility.

    Args:
        model_id: Target model ID.
        thinking_level: Normalized thinking level.

    Returns:
        Validated thinking level.
    """
    if thinking_level is None:
        return None
    if "gemini-3" not in model_id:
        raise SystemExit("--thinking-level is only supported for Gemini 3 models.")
    if thinking_level in {"MINIMAL", "MEDIUM"} and "flash" not in model_id:
        raise SystemExit("MINIMAL/MEDIUM thinking levels require a Gemini 3 Flash model.")
    return thinking_level


def resolve_api_key(raw_key: str | None, dotenv_values: dict[str, str], dry_run: bool) -> str:
    """Resolve API key from CLI/env/dotenv.

    Args:
        raw_key: CLI key.
        dotenv_values: Parsed dotenv map.
        dry_run: Dry-run toggle.

    Returns:
        API key value.
    """
    if raw_key:
        return raw_key
    candidates = [
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("VERTEX_KEY"),
        os.getenv("VERTEX_API_KEY"),
        os.getenv("GOOGLE_CLOUD_API_KEY"),
        dotenv_values.get("GEMINI_API_KEY"),
        dotenv_values.get("GOOGLE_API_KEY"),
        dotenv_values.get("VERTEX_KEY"),
        dotenv_values.get("VERTEX_API_KEY"),
        dotenv_values.get("GOOGLE_CLOUD_API_KEY"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    if dry_run:
        return "DRY_RUN_KEY"
    raise SystemExit("API key is required. Use --api-key or set VERTEX_KEY/GEMINI_API_KEY.")


def load_prompt_text(path: Path) -> str:
    """Load prompt markdown from disk.

    Args:
        path: Prompt path.

    Returns:
        Prompt text.
    """
    if not path.exists():
        raise SystemExit(f"Missing prompt file: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"Prompt file is empty: {path}")
    return text


def render_user_prompt(template: str, think_text: str) -> str:
    """Render user prompt by injecting think text.

    Args:
        template: User prompt template with `{think_text}`.
        think_text: Raw think text.

    Returns:
        Rendered prompt.
    """
    if "{think_text}" not in template:
        raise SystemExit("user_prompt.md must include `{think_text}` placeholder.")
    return template.replace("{think_text}", think_text)


def build_client(model: TransformConfig) -> genai.Client:
    """Build GenAI client for configured mode.

    Args:
        model: Transform configuration.

    Returns:
        Initialized client.
    """
    if model.mode == "gemini":
        return genai.Client(api_key=model.api_key)

    http_options = types.HttpOptions(apiVersion="v1")
    if model.mode == "vertex":
        if not model.project_id or not model.location:
            raise SystemExit("--project and --location are required in vertex mode.")
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
        os.environ["GOOGLE_CLOUD_PROJECT"] = model.project_id
        os.environ["GOOGLE_CLOUD_LOCATION"] = model.location

    return genai.Client(vertexai=True, api_key=model.api_key, http_options=http_options)


def call_model(
    client: genai.Client,
    model: TransformConfig,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call non-batch model API once.

    Args:
        client: GenAI client.
        model: Transform config.
        system_prompt: System prompt text.
        user_prompt: User prompt text.

    Returns:
        Cleaned response text.
    """
    thinking_config = (
        types.ThinkingConfig(thinking_level=model.thinking_level)
        if model.thinking_level
        else None
    )
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        temperature=model.temperature,
        max_output_tokens=model.max_output_tokens,
        thinking_config=thinking_config,
    )
    response = client.models.generate_content(
        model=model.model_id,
        contents=user_prompt,
        config=config,
    )
    return clean_model_output(response.text or "")


def transform_think_block(
    client: genai.Client,
    model: TransformConfig,
    system_prompt: str,
    user_template: str,
    think_text: str,
) -> str:
    """Transform one think block with retries.

    Args:
        client: GenAI client.
        model: Transform config.
        system_prompt: System prompt.
        user_template: User prompt template.
        think_text: Think block text.

    Returns:
        Transformed think content.
    """
    user_prompt = render_user_prompt(template=user_template, think_text=think_text)
    last_error: Exception | None = None
    for attempt in range(1, model.retry_limit + 1):
        try:
            return call_model(
                client=client,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            time.sleep(model.retry_sleep_seconds * attempt)
    raise RuntimeError("Model call failed after retries") from last_error


def replace_think_blocks(content: str, replacements: Iterator[str]) -> str:
    """Replace think blocks using sequential replacement values.

    Args:
        content: Assistant content text.
        replacements: Iterator of replacement strings.

    Returns:
        Updated content.
    """

    def _replace(match: re.Match[str]) -> str:
        block = match.group(1)
        if not block.strip():
            return "<think>\n\n</think>"
        try:
            transformed = clean_model_output(next(replacements))
        except StopIteration as exc:
            raise RuntimeError("Ran out of transformed outputs.") from exc
        return f"<think>\n{transformed}\n</think>"

    return THINK_PATTERN.sub(_replace, content)


def resolve_batch_model_id(model_id: str) -> str:
    """Normalize model id for batch API.

    Args:
        model_id: Input model ID.

    Returns:
        Batch-compatible model ID.
    """
    if model_id.startswith("models/") or model_id.startswith("projects/"):
        return model_id
    return f"models/{model_id}"


def collect_transform_rows(
    source_path: Path,
    seen_ids: set[str],
    max_needed: int,
) -> list[dict[str, object]]:
    """Collect rows to transform, excluding already emitted IDs.

    Args:
        source_path: Stratified sample path.
        seen_ids: IDs already in output.
        max_needed: Number of rows to collect.

    Returns:
        Rows to transform.
    """
    rows: list[dict[str, object]] = []
    for row in iter_jsonl(path=source_path):
        row_id = row.get("id")
        if isinstance(row_id, str) and row_id in seen_ids:
            continue
        rows.append(row)
        if len(rows) >= max_needed:
            break
    return rows


def load_seen_ids(path: Path) -> set[str]:
    """Load seen row IDs from transformed output.

    Args:
        path: Transformed output path.

    Returns:
        Set of seen IDs.
    """
    if not path.exists():
        return set()
    seen: set[str] = set()
    for row in iter_jsonl(path=path):
        row_id = row.get("id")
        if isinstance(row_id, str):
            seen.add(row_id)
    return seen


def build_inline_batch_requests(
    rows: list[dict[str, object]],
    system_prompt: str,
    user_template: str,
    model: TransformConfig,
) -> tuple[list[dict[str, object]], list[ThinkTask]]:
    """Build inline batch requests and think-task mapping.

    Args:
        rows: Rows to transform.
        system_prompt: System prompt text.
        user_template: User prompt template.
        model: Transform config.

    Returns:
        Tuple of requests and task mapping.
    """
    requests: list[dict[str, object]] = []
    tasks: list[ThinkTask] = []

    for row_index, row in enumerate(rows):
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        for message_index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            blocks = extract_think_blocks(text=content)
            for block_index, block in enumerate(blocks):
                request_index = len(tasks)
                user_prompt = render_user_prompt(template=user_template, think_text=block)
                request_config: dict[str, object] = {
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "temperature": model.temperature,
                    "max_output_tokens": model.max_output_tokens,
                }
                if model.thinking_level:
                    request_config["thinking_config"] = {
                        "thinking_level": model.thinking_level,
                    }
                requests.append(
                    {
                        "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                        "metadata": {"request_index": str(request_index)},
                        "config": request_config,
                    }
                )
                tasks.append(
                    ThinkTask(
                        row_index=row_index,
                        message_index=message_index,
                        block_index=block_index,
                    )
                )

    return requests, tasks


def run_batch_requests(
    client: genai.Client,
    model: TransformConfig,
    inline_requests: list[dict[str, object]],
) -> list[str]:
    """Run batch API and return cleaned outputs.

    Args:
        client: GenAI client.
        model: Transform config.
        inline_requests: Batch request payloads.

    Returns:
        Cleaned outputs aligned with request order.
    """
    if not inline_requests:
        return []

    batch_job = client.batches.create(
        model=resolve_batch_model_id(model_id=model.model_id),
        src=inline_requests,
        config={"display_name": f"build-sft-{int(time.time())}"},
    )
    assert batch_job.name is not None
    print(f"Batch submitted: name={batch_job.name}", flush=True)

    poll_count = 0
    while True:
        batch_job = client.batches.get(name=batch_job.name)
        state_obj = getattr(batch_job, "state", None)
        state_name = getattr(state_obj, "name", str(state_obj))
        poll_count += 1
        if poll_count == 1 or poll_count % 12 == 0:
            print(f"Batch status: state={state_name} polls={poll_count}", flush=True)
        if state_name in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED"}:
            break
        time.sleep(model.batch_poll_seconds)

    if state_name != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job ended with {state_name}")

    destination = getattr(batch_job, "dest", None)
    responses = getattr(destination, "inlined_responses", None)
    if not responses:
        raise RuntimeError("Batch job succeeded without inline responses.")

    outputs: list[str] = [""] * len(inline_requests)
    seen_indexes: set[int] = set()
    failed_indexes: set[int] = set()
    failed_reasons: dict[int, str] = {}

    for item in responses:
        metadata = getattr(item, "metadata", None)
        request_index_raw = metadata.get("request_index") if isinstance(metadata, dict) else None
        if request_index_raw is None:
            raise RuntimeError("Batch response item is missing request_index metadata.")
        try:
            request_index = int(request_index_raw)
        except ValueError as exc:
            raise RuntimeError(f"Invalid request_index metadata: {request_index_raw}") from exc
        if request_index < 0 or request_index >= len(outputs):
            raise RuntimeError(f"Out-of-range request_index metadata: {request_index}")
        if request_index in seen_indexes:
            raise RuntimeError(f"Duplicate request_index metadata: {request_index}")

        response_error = getattr(item, "error", None)
        if response_error is not None:
            failed_indexes.add(request_index)
            failed_reasons[request_index] = str(response_error)
            continue
        response = getattr(item, "response", None)
        outputs[request_index] = clean_model_output(
            extract_response_text(response=response)
        )
        seen_indexes.add(request_index)

    all_indexes = set(range(len(outputs)))
    missing_indexes = sorted(all_indexes - seen_indexes - failed_indexes)
    if missing_indexes:
        for missing_index in missing_indexes:
            failed_indexes.add(missing_index)
            failed_reasons[missing_index] = "Missing inlined response for request index."

    if failed_indexes:
        print(
            f"Retrying {len(failed_indexes)} failed batch item(s) with direct calls.",
            flush=True,
        )
        for failed_index in sorted(failed_indexes):
            inline_request = inline_requests[failed_index]
            inline_contents = cast(Any, inline_request.get("contents"))
            inline_config = cast(Any, inline_request.get("config"))
            last_error: Exception | None = None
            for attempt in range(1, model.retry_limit + 1):
                try:
                    direct_response = client.models.generate_content(
                        model=model.model_id,
                        contents=inline_contents,
                        config=inline_config,
                    )
                    outputs[failed_index] = clean_model_output(
                        extract_response_text(response=direct_response)
                    )
                    seen_indexes.add(failed_index)
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    time.sleep(model.retry_sleep_seconds * attempt)
            if failed_index not in seen_indexes:
                reason = failed_reasons.get(failed_index, "Unknown batch response error.")
                raise RuntimeError(
                    "Failed to recover batch item after retries. "
                    f"request_index={failed_index} batch_error={reason}"
                ) from last_error

    if len(seen_indexes) != len(outputs):
        unresolved_indexes = sorted(all_indexes - seen_indexes)
        raise RuntimeError(
            f"Unresolved batch outputs after retry. Missing indexes: {unresolved_indexes[:10]}"
        )

    return outputs


def apply_transforms_to_rows(
    rows: list[dict[str, object]],
    outputs: list[str],
) -> list[dict[str, object]]:
    """Apply transformed think-block outputs to row list.

    Args:
        rows: Input rows.
        outputs: Replacement outputs in order.

    Returns:
        Rows with replaced think blocks.
    """
    output_iter = iter(outputs)
    updated_rows: list[dict[str, object]] = []

    for row in rows:
        messages = row.get("messages", [])
        updated_messages: list[dict[str, object]] = []
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict):
                updated_messages.append(message)
                continue
            if message.get("role") != "assistant":
                updated_messages.append(message)
                continue
            content = message.get("content")
            if not isinstance(content, str):
                updated_messages.append(message)
                continue
            updated_content = replace_think_blocks(content=content, replacements=output_iter)
            updated_message = dict(message)
            updated_message["content"] = updated_content
            updated_messages.append(updated_message)

        output_row = dict(row)
        output_row["messages"] = updated_messages
        updated_rows.append(output_row)

    try:
        next(output_iter)
        raise RuntimeError("Received extra outputs not mapped to think blocks.")
    except StopIteration:
        pass

    return updated_rows


def run_transform_stage(
    model: TransformConfig,
    prompts: PromptConfig,
    paths: PipelinePaths,
    runtime: RuntimeConfig,
) -> dict[str, object]:
    """Run transform stage from stratified sample to transformed output.

    Args:
        model: Transform config.
        prompts: Prompt paths.
        paths: Pipeline paths.
        runtime: Runtime options.

    Returns:
        Stage metadata.
    """
    if not paths.stratified_path.exists():
        raise SystemExit(f"Missing stratified sample: {paths.stratified_path}")

    existing_rows = count_jsonl_rows(path=paths.transformed_path) if runtime.resume else 0
    rows_left = max(model.max_rows - existing_rows, 0)
    confirm_large_work(stage="transform", rows_left=rows_left, runtime=runtime)

    if rows_left == 0:
        return {
            "target_rows": model.max_rows,
            "existing_rows": existing_rows,
            "rows_left": 0,
            "skipped": True,
        }

    seen_ids = load_seen_ids(path=paths.transformed_path) if runtime.resume else set()
    rows_to_transform = collect_transform_rows(
        source_path=paths.stratified_path,
        seen_ids=seen_ids,
        max_needed=rows_left,
    )
    total_think_blocks = 0
    for row in rows_to_transform:
        messages = row.get("messages", [])
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "assistant":
                continue
            content = message.get("content")
            if isinstance(content, str):
                total_think_blocks += len(extract_think_blocks(text=content))

    if runtime.dry_run:
        return {
            "target_rows": model.max_rows,
            "existing_rows": existing_rows,
            "rows_left": rows_left,
            "rows_selected": len(rows_to_transform),
            "think_blocks": total_think_blocks,
            "batch": model.batch,
        }

    if not runtime.resume and paths.transformed_path.exists():
        paths.transformed_path.unlink()

    system_prompt = load_prompt_text(path=prompts.system_prompt_path)
    user_template = load_prompt_text(path=prompts.user_prompt_path)
    client = build_client(model=model)

    if model.batch:
        requests, _ = build_inline_batch_requests(
            rows=rows_to_transform,
            system_prompt=system_prompt,
            user_template=user_template,
            model=model,
        )
        outputs = run_batch_requests(
            client=client,
            model=model,
            inline_requests=requests,
        )
        updated_rows = apply_transforms_to_rows(rows=rows_to_transform, outputs=outputs)
    else:
        updated_rows = []
        progress = tqdm(rows_to_transform, desc="Transforming rows", unit="row")
        for row in progress:
            messages = row.get("messages", [])
            updated_messages: list[dict[str, object]] = []
            for message in messages if isinstance(messages, list) else []:
                if not isinstance(message, dict):
                    updated_messages.append(message)
                    continue
                if message.get("role") != "assistant":
                    updated_messages.append(message)
                    continue
                content = message.get("content")
                if not isinstance(content, str):
                    updated_messages.append(message)
                    continue
                blocks = extract_think_blocks(text=content)
                transformed = [
                    transform_think_block(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        user_template=user_template,
                        think_text=block,
                    )
                    for block in blocks
                ]
                updated_content = replace_think_blocks(content=content, replacements=iter(transformed))
                updated_message = dict(message)
                updated_message["content"] = updated_content
                updated_messages.append(updated_message)
            output_row = dict(row)
            output_row["messages"] = updated_messages
            updated_rows.append(output_row)

    for row in updated_rows:
        output_row = dict(row)
        output_row["transform_meta"] = {
            "model": model.model_id,
            "mode": model.mode,
            "batch": model.batch,
            "max_output_tokens": model.max_output_tokens,
            "thinking_level": model.thinking_level,
            "system_prompt_path": str(prompts.system_prompt_path),
            "user_prompt_path": str(prompts.user_prompt_path),
        }
        write_jsonl(output_path=paths.transformed_path, row=output_row)

    return {
        "target_rows": model.max_rows,
        "existing_rows": existing_rows,
        "rows_selected": len(rows_to_transform),
        "rows_emitted": len(updated_rows),
        "think_blocks": total_think_blocks,
        "batch": model.batch,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed argparse namespace.
    """
    parser = argparse.ArgumentParser(description="Build staged SFT dataset with optional transform.")

    parser.add_argument("--stage", choices=["sample", "filter", "stratify", "transform", "all"], default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-threshold", type=int, default=DEFAULT_CONFIRM_THRESHOLD)

    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument("--rows-per-shard", type=int, default=DEFAULT_ROWS_PER_SHARD)
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER)

    parser.add_argument("--min-tokens", type=int, default=DEFAULT_MIN_TOKENS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--encoding", default=DEFAULT_ENCODING)

    parser.add_argument("--target-rows", type=int, default=DEFAULT_TARGET_ROWS)

    parser.add_argument("--mode", choices=["gemini", "express", "vertex"], default=DEFAULT_MODE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default="global")
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--thinking-level",
        choices=["minimal", "low", "medium", "high", "none"],
        default=DEFAULT_THINKING_LEVEL,
    )
    parser.add_argument("--retry-limit", type=int, default=DEFAULT_RETRY_LIMIT)
    parser.add_argument("--retry-sleep", type=float, default=DEFAULT_RETRY_SLEEP)
    parser.add_argument("--batch-poll", type=float, default=DEFAULT_BATCH_POLL_SECONDS)
    parser.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    parser.add_argument("--batch", action="store_true", default=DEFAULT_BATCH)
    parser.add_argument("--no-batch", dest="batch", action="store_false")

    parser.add_argument("--system-prompt", type=Path, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--user-prompt", type=Path, default=DEFAULT_USER_PROMPT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_PATH)

    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> tuple[RuntimeConfig, SamplingConfig, FilterConfig, StratifyConfig, TransformConfig, PromptConfig]:
    """Build typed config objects from parsed args.

    Args:
        args: CLI arguments.

    Returns:
        Tuple of runtime, stage, model, and prompt configs.
    """
    runtime = RuntimeConfig(
        output_dir=args.output_dir,
        resume=args.resume,
        auto_yes=args.yes,
        dry_run=args.dry_run,
        stage=args.stage,
        confirm_threshold=args.confirm_threshold,
    )
    sampling = SamplingConfig(
        dataset_name=args.dataset,
        split=args.split,
        rows_per_shard=args.rows_per_shard,
        num_shards=args.num_shards,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    filter_config = FilterConfig(
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        encoding_name=args.encoding,
    )
    stratify = StratifyConfig(target_rows=args.target_rows, seed=args.seed)

    dotenv_values = parse_dotenv(path=args.env_file)
    api_key = resolve_api_key(raw_key=args.api_key, dotenv_values=dotenv_values, dry_run=args.dry_run)
    thinking_level = validate_thinking_level(
        model_id=args.model,
        thinking_level=normalize_thinking_level(level=args.thinking_level),
    )
    transform = TransformConfig(
        mode=args.mode,
        model_id=args.model,
        api_key=api_key,
        project_id=args.project,
        location=args.location,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        thinking_level=thinking_level,
        batch=args.batch,
        retry_limit=args.retry_limit,
        retry_sleep_seconds=args.retry_sleep,
        batch_poll_seconds=args.batch_poll,
        max_rows=args.max_rows,
        dry_run=args.dry_run,
    )
    prompts = PromptConfig(
        system_prompt_path=args.system_prompt,
        user_prompt_path=args.user_prompt,
    )
    return runtime, sampling, filter_config, stratify, transform, prompts


def stage_config_hashes(
    sampling: SamplingConfig,
    filter_config: FilterConfig,
    stratify: StratifyConfig,
    transform: TransformConfig,
    prompts: PromptConfig,
) -> dict[Stage, str]:
    """Build config hashes for all stages.

    Args:
        sampling: Sampling config.
        filter_config: Filter config.
        stratify: Stratify config.
        transform: Transform config.
        prompts: Prompt config.

    Returns:
        Hashes by stage.
    """
    transform_hash_payload = asdict(transform)
    transform_hash_payload["api_key"] = "***"
    return {
        "sample": hash_config(asdict(sampling)),
        "filter": hash_config(asdict(filter_config)),
        "stratify": hash_config(asdict(stratify)),
        "transform": hash_config(
            {
                **transform_hash_payload,
                "system_prompt": str(prompts.system_prompt_path),
                "user_prompt": str(prompts.user_prompt_path),
            }
        ),
    }


def main() -> None:
    """CLI entrypoint for staged dataset generation and transformation."""
    args = parse_args()
    runtime, sampling, filter_config, stratify, transform, prompts = build_configs(args=args)
    paths = resolve_paths(runtime=runtime)

    state = load_state(path=paths.state_path)
    hashes = stage_config_hashes(
        sampling=sampling,
        filter_config=filter_config,
        stratify=stratify,
        transform=transform,
        prompts=prompts,
    )

    completed: dict[Stage, bool] = {
        "sample": is_stage_complete(state=state, stage="sample", expected_hash=hashes["sample"], required_path=paths.raw_sample_path),
        "filter": is_stage_complete(state=state, stage="filter", expected_hash=hashes["filter"], required_path=paths.filtered_path),
        "stratify": is_stage_complete(state=state, stage="stratify", expected_hash=hashes["stratify"], required_path=paths.stratified_path),
        "transform": is_stage_complete(state=state, stage="transform", expected_hash=hashes["transform"], required_path=paths.transformed_path),
    }

    if runtime.stage == "all":
        stages_to_run: list[Stage] = [stage for stage in STAGE_ORDER if not completed[stage]]
    elif runtime.stage is None:
        next_stage = choose_auto_stage(completed=completed)
        stages_to_run = [next_stage] if next_stage else []
    else:
        stages_to_run = [runtime.stage]

    if not stages_to_run:
        print("All stages are already complete for current configuration.")
        return

    for stage in stages_to_run:
        print(f"Running stage: {stage}")
        if stage == "sample":
            metadata = run_sample_stage(sampling=sampling, paths=paths, runtime=runtime)
        elif stage == "filter":
            metadata = run_filter_stage(filter_config=filter_config, paths=paths, runtime=runtime)
        elif stage == "stratify":
            metadata = run_stratify_stage(stratify=stratify, paths=paths, runtime=runtime)
        elif stage == "transform":
            metadata = run_transform_stage(model=transform, prompts=prompts, paths=paths, runtime=runtime)
        else:
            raise SystemExit(f"Unsupported stage: {stage}")

        print(json.dumps({"stage": stage, "metadata": metadata}, indent=2))
        set_stage_status(
            state=state,
            stage=stage,
            completed=True,
            config_hash=hashes[stage],
            metadata=metadata,
        )
        if not runtime.dry_run:
            save_state(path=paths.state_path, state=state)


if __name__ == "__main__":
    main()
