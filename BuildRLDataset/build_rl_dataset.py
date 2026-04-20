from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, cast

from datasets import Dataset, load_dataset
from tqdm import tqdm

STAGE_ORDER = ("sample", "filter", "stratify", "export")

DEFAULT_DATASET = "allenai/Dolci-Think-RL-7B"
DEFAULT_SPLIT = "train"
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_STATE_FILE = "pipeline_state.json"
DEFAULT_SAMPLE_ROWS = 50_000
DEFAULT_TARGET_TRAIN_ROWS = 10_000
DEFAULT_SEED = 42
DEFAULT_SHUFFLE_BUFFER = 20_000
DEFAULT_CONFIRM_THRESHOLD = 1_000

Stage = Literal["sample", "filter", "stratify", "export"]


@dataclass(frozen=True)
class SamplingConfig:
    """Configuration for the raw sample stage.

    Args:
        dataset_name: Hugging Face dataset name.
        split: Dataset split name.
        sample_rows: Number of rows to sample into `raw_sample.jsonl`.
        seed: Shuffle seed for streaming sample order.
        shuffle_buffer: Streaming shuffle buffer size.

    Returns:
        Sampling configuration for the `sample` stage.
    """

    dataset_name: str
    split: str
    sample_rows: int
    seed: int
    shuffle_buffer: int


@dataclass(frozen=True)
class FilterConfig:
    """Configuration for the math-only filter stage.

    Args:
        required_dataset_label: Dataset label that must be present.

    Returns:
        Filter configuration for the `filter` stage.
    """

    required_dataset_label: str


@dataclass(frozen=True)
class StratifyConfig:
    """Configuration for balanced train sampling.

    Args:
        target_train_rows: Final train row count.
        seed: Reservoir-sampling seed.

    Returns:
        Stratify configuration for the `stratify` stage.
    """

    target_train_rows: int
    seed: int


@dataclass(frozen=True)
class ExportConfig:
    """Configuration for RLHF parquet export.

    Args:
        prompt_role: Chat role used for the exported prompt message.

    Returns:
        Export configuration for the `export` stage.
    """

    prompt_role: str = "user"


@dataclass(frozen=True)
class RuntimeConfig:
    """Runtime options shared by all stages.

    Args:
        output_dir: Output directory root.
        resume: Whether to reuse stage outputs when valid.
        auto_yes: Skip confirmation prompts.
        dry_run: Avoid writes when `True`.
        stage: Explicit requested stage or `all`.
        confirm_threshold: Row count threshold for confirmation prompts.

    Returns:
        Runtime configuration used by the staged CLI.
    """

    output_dir: Path
    resume: bool
    auto_yes: bool
    dry_run: bool
    stage: str | None
    confirm_threshold: int


@dataclass(frozen=True)
class PipelinePaths:
    """Resolved output paths for the RL dataset pipeline.

    Args:
        output_dir: Output root.
        raw_sample_path: JSONL path for sampled rows.
        filtered_path: JSONL path for math-only rows.
        stratified_path: JSONL path for balanced train rows.
        train_parquet_path: Final parquet path.
        manifest_path: Export manifest path.
        source_audit_path: Source-audit JSON path.
        state_path: Pipeline state JSON path.

    Returns:
        Dataclass containing the pipeline filesystem layout.
    """

    output_dir: Path
    raw_sample_path: Path
    filtered_path: Path
    stratified_path: Path
    train_parquet_path: Path
    manifest_path: Path
    source_audit_path: Path
    state_path: Path


@dataclass(frozen=True)
class StageStatus:
    """Persisted stage metadata.

    Args:
        completed: Whether the stage completed successfully.
        config_hash: Stage config hash used for cache validation.
        updated_at: UTC timestamp for the latest update.
        metadata: Arbitrary stage metadata payload.

    Returns:
        Stage status entry loaded from `pipeline_state.json`.
    """

    completed: bool
    config_hash: str
    updated_at: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class SourceAuditSummary:
    """Source-count summary for one pipeline stage.

    Args:
        row_count: Number of rows examined.
        counts_by_dataset_label: Counts keyed by joined `dataset` labels.
        counts_by_dataset_source: Counts keyed by `dataset_source`.
        counts_by_original_dataset: Counts keyed by `original_dataset`.
        counts_by_source_family: Counts keyed by normalized source family.

    Returns:
        Dataclass describing stage-level source composition.
    """

    row_count: int
    counts_by_dataset_label: dict[str, int]
    counts_by_dataset_source: dict[str, int]
    counts_by_original_dataset: dict[str, int]
    counts_by_source_family: dict[str, int]


def utc_now() -> str:
    """Return the current UTC timestamp.

    Args:
        None.

    Returns:
        ISO-8601 UTC timestamp string.
    """

    return datetime.now(timezone.utc).isoformat()


def hash_config(value: dict[str, object]) -> str:
    """Hash a stage configuration payload.

    Args:
        value: JSON-serializable configuration mapping.

    Returns:
        SHA256 hash string.
    """

    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_paths(runtime: RuntimeConfig) -> PipelinePaths:
    """Resolve output paths for the staged pipeline.

    Args:
        runtime: Runtime configuration.

    Returns:
        Fully resolved pipeline paths.
    """

    output_dir = runtime.output_dir
    return PipelinePaths(
        output_dir=output_dir,
        raw_sample_path=output_dir / "raw_sample.jsonl",
        filtered_path=output_dir / "filtered_candidates.jsonl",
        stratified_path=output_dir / "stratified_sample.jsonl",
        train_parquet_path=output_dir / "train.parquet",
        manifest_path=output_dir / "manifest.json",
        source_audit_path=output_dir / "source_audit.json",
        state_path=output_dir / DEFAULT_STATE_FILE,
    )


def load_state(path: Path) -> dict[str, object]:
    """Load pipeline state or return an empty state payload.

    Args:
        path: Pipeline state path.

    Returns:
        Mutable pipeline state mapping.
    """

    if not path.exists():
        return {"version": 1, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, object]) -> None:
    """Persist pipeline state to disk.

    Args:
        path: Pipeline state path.
        state: Mutable pipeline state payload.

    Returns:
        None.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_stage_status(state: dict[str, object], stage: Stage) -> StageStatus | None:
    """Read one stage status from the pipeline state.

    Args:
        state: Pipeline state mapping.
        stage: Stage name to fetch.

    Returns:
        Parsed stage status or `None`.
    """

    stages = state.get("stages", {})
    if not isinstance(stages, dict):
        return None
    payload = stages.get(stage)
    if not isinstance(payload, dict):
        return None
    metadata_payload = payload.get("metadata", {})
    metadata = dict(metadata_payload) if isinstance(metadata_payload, dict) else {}
    return StageStatus(
        completed=bool(payload.get("completed", False)),
        config_hash=str(payload.get("config_hash", "")),
        updated_at=str(payload.get("updated_at", "")),
        metadata=metadata,
    )


def set_stage_status(
    *,
    state: dict[str, object],
    stage: Stage,
    completed: bool,
    config_hash: str,
    metadata: dict[str, object],
) -> None:
    """Upsert one stage status into the pipeline state.

    Args:
        state: Mutable pipeline state mapping.
        stage: Stage name being updated.
        completed: Completion flag.
        config_hash: Config hash for cache validation.
        metadata: Arbitrary stage metadata.

    Returns:
        None.
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
    *,
    state: dict[str, object],
    stage: Stage,
    expected_hash: str,
    required_path: Path,
) -> bool:
    """Check whether a stage can be reused under the current configuration.

    Args:
        state: Pipeline state mapping.
        stage: Stage name to validate.
        expected_hash: Config hash expected for the stage.
        required_path: Output path that must exist.

    Returns:
        `True` when the stage output is reusable.
    """

    status = get_stage_status(state=state, stage=stage)
    if status is None:
        return False
    return status.completed and status.config_hash == expected_hash and required_path.exists()


def choose_auto_stage(completed: dict[Stage, bool]) -> Stage | None:
    """Return the next incomplete stage in order.

    Args:
        completed: Completion flags keyed by stage.

    Returns:
        The next stage to run, or `None` when all are complete.
    """

    for stage in STAGE_ORDER:
        if not completed[cast(Stage, stage)]:
            return cast(Stage, stage)
    return None


def confirm_large_work(stage: Stage, rows_left: int, runtime: RuntimeConfig) -> None:
    """Confirm long-running work before execution.

    Args:
        stage: Stage name about to run.
        rows_left: Remaining row count to process.
        runtime: Runtime configuration.

    Returns:
        None.
    """

    if runtime.dry_run or runtime.auto_yes or rows_left <= runtime.confirm_threshold:
        return
    answer = input(f"Stage '{stage}' has {rows_left} rows left. Continue? [y/N]: ").strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted by user.")


def decode_json_line(line: str, line_number: int) -> list[dict[str, object]]:
    """Decode one JSONL line into one or more JSON objects.

    Args:
        line: JSONL line text.
        line_number: One-based line number for error context.

    Returns:
        Decoded object list.

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


def write_jsonl_row(output_path: Path, row: dict[str, object]) -> None:
    """Append one JSON object to a JSONL file.

    Args:
        output_path: Target JSONL path.
        row: JSON-serializable row payload.

    Returns:
        None.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def count_jsonl_rows(path: Path) -> int:
    """Count non-empty rows in a JSONL file.

    Args:
        path: JSONL path to count.

    Returns:
        Number of non-empty lines.
    """

    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Yield JSON objects from a JSONL file.

    Args:
        path: JSONL path to read.

    Returns:
        Iterator over decoded row mappings.
    """

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            for row in decode_json_line(line=line, line_number=line_number):
                yield row


def normalize_dataset_labels(value: object) -> tuple[str, ...]:
    """Normalize dataset labels into a canonical string tuple.

    Args:
        value: Raw `dataset` field from the source row.

    Returns:
        Sorted label tuple.
    """

    if isinstance(value, str):
        labels = [value]
    elif isinstance(value, list):
        labels = [str(item) for item in value]
    else:
        labels = []
    return tuple(sorted(label.strip().lower() for label in labels if label))


def normalize_source_family(row: dict[str, object]) -> str:
    """Resolve a stable source-family label from a raw dataset row.

    Args:
        row: Source dataset row.

    Returns:
        Normalized source-family label.
    """

    raw_value = str(row.get("original_dataset") or row.get("dataset_source") or "unknown")
    name = raw_value.split("/")[-1]
    name = re.sub(pattern=r"(_filtered|_cleaned)$", repl="", string=name, flags=re.IGNORECASE)
    return name or "unknown"


def extract_ground_truth_values(row: dict[str, object]) -> list[str]:
    """Normalize `ground_truth` into a string list.

    Args:
        row: Source dataset row.

    Returns:
        Non-empty ground-truth string list.
    """

    value = row.get("ground_truth")
    if isinstance(value, str):
        return [value] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def is_math_training_row(row: dict[str, object], filter_config: FilterConfig) -> bool:
    """Return whether a raw row is usable for math RL training.

    Args:
        row: Source dataset row.
        filter_config: Filter-stage configuration.

    Returns:
        `True` when the row is math-only and contains required fields.
    """

    dataset_labels = normalize_dataset_labels(value=row.get("dataset"))
    if filter_config.required_dataset_label.lower() not in dataset_labels:
        return False
    prompt_value = row.get("prompt")
    if not isinstance(prompt_value, str) or not prompt_value.strip():
        return False
    if not extract_ground_truth_values(row=row):
        return False
    return True


def compute_balanced_targets(counts_by_source: dict[str, int], target_rows: int) -> dict[str, int]:
    """Allocate balanced per-source targets with capacity caps.

    Args:
        counts_by_source: Available row counts by source family.
        target_rows: Desired final train row count.

    Returns:
        Per-source sample targets.

    Example:
        >>> compute_balanced_targets({"a": 10, "b": 2}, target_rows=6)
        {'a': 4, 'b': 2}
    """

    total_available = sum(counts_by_source.values())
    if total_available < target_rows:
        raise SystemExit(f"Not enough filtered rows ({total_available}) for target {target_rows}.")

    sources = sorted(counts_by_source)
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

    available_sources = [source for source in sources if targets[source] < counts_by_source[source]]
    while remaining > 0 and available_sources:
        for source in available_sources:
            if remaining == 0:
                break
            if targets[source] >= counts_by_source[source]:
                continue
            targets[source] += 1
            remaining -= 1

    if sum(targets.values()) != target_rows:
        raise SystemExit("Failed to allocate balanced source targets.")
    return targets


def iter_streaming_rows(sampling: SamplingConfig) -> Iterable[dict[str, object]]:
    """Create a deterministic streaming iterator over source dataset rows.

    Args:
        sampling: Sampling-stage configuration.

    Returns:
        Iterable of raw dataset row mappings.
    """

    dataset = load_dataset(sampling.dataset_name, split=sampling.split, streaming=True)
    if sampling.shuffle_buffer > 0:
        dataset = dataset.shuffle(seed=sampling.seed, buffer_size=sampling.shuffle_buffer)
    return cast(Iterable[dict[str, object]], dataset)


def run_sample_stage(
    *, sampling: SamplingConfig, paths: PipelinePaths, runtime: RuntimeConfig
) -> dict[str, object]:
    """Run the raw streaming sample stage.

    Args:
        sampling: Sampling-stage configuration.
        paths: Pipeline output paths.
        runtime: Runtime configuration.

    Returns:
        Metadata describing sample-stage work.
    """

    existing_rows = count_jsonl_rows(paths.raw_sample_path) if runtime.resume else 0
    rows_left = max(sampling.sample_rows - existing_rows, 0)
    confirm_large_work(stage="sample", rows_left=rows_left, runtime=runtime)
    if runtime.dry_run:
        return {"target_rows": sampling.sample_rows, "existing_rows": existing_rows, "rows_left": rows_left}
    if rows_left == 0:
        return {"target_rows": sampling.sample_rows, "existing_rows": existing_rows, "rows_left": 0, "skipped": True}

    if not runtime.resume and paths.raw_sample_path.exists():
        paths.raw_sample_path.unlink()
        existing_rows = 0

    written_rows = 0
    progress = tqdm(total=sampling.sample_rows, desc="Sampling rows", unit="row")
    progress.update(existing_rows)
    for row_index, row in enumerate(iter_streaming_rows(sampling=sampling)):
        if row_index < existing_rows:
            continue
        if row_index >= sampling.sample_rows:
            break
        output_row = dict(row)
        output_row["sample_meta"] = {"seed": sampling.seed, "row_index": row_index}
        write_jsonl_row(output_path=paths.raw_sample_path, row=output_row)
        written_rows += 1
        progress.update(1)
    progress.close()
    return {
        "target_rows": sampling.sample_rows,
        "existing_rows": existing_rows,
        "written_rows": written_rows,
        "rows_left": max(sampling.sample_rows - (existing_rows + written_rows), 0),
    }


def run_filter_stage(
    *, filter_config: FilterConfig, paths: PipelinePaths, runtime: RuntimeConfig
) -> dict[str, object]:
    """Filter sampled rows down to usable math-only training examples.

    Args:
        filter_config: Filter-stage configuration.
        paths: Pipeline output paths.
        runtime: Runtime configuration.

    Returns:
        Metadata describing filtered-row composition.
    """

    if not paths.raw_sample_path.exists():
        raise SystemExit(f"Missing raw sample: {paths.raw_sample_path}")
    total_input = count_jsonl_rows(paths.raw_sample_path)
    if runtime.dry_run:
        return {"input_rows": total_input, "required_dataset_label": filter_config.required_dataset_label}

    counts_by_source: Counter[str] = Counter()
    written_rows = 0
    if paths.filtered_path.exists():
        paths.filtered_path.unlink()

    for row in tqdm(iter_jsonl(paths.raw_sample_path), total=total_input, desc="Filtering rows", unit="row"):
        if not is_math_training_row(row=row, filter_config=filter_config):
            continue
        source_family = normalize_source_family(row=row)
        output_row = dict(row)
        output_row["source_family"] = source_family
        write_jsonl_row(output_path=paths.filtered_path, row=output_row)
        counts_by_source[source_family] += 1
        written_rows += 1

    return {
        "input_rows": total_input,
        "written_rows": written_rows,
        "counts_by_source_family": dict(counts_by_source),
    }


def run_stratify_stage(
    *, stratify: StratifyConfig, paths: PipelinePaths, runtime: RuntimeConfig
) -> dict[str, object]:
    """Run balanced source-family train sampling.

    Args:
        stratify: Stratify-stage configuration.
        paths: Pipeline output paths.
        runtime: Runtime configuration.

    Returns:
        Metadata describing target allocation and output size.
    """

    if not paths.filtered_path.exists():
        raise SystemExit(f"Missing filtered rows: {paths.filtered_path}")

    counts_by_source: Counter[str] = Counter()
    total_rows = 0
    for row in iter_jsonl(paths.filtered_path):
        source_family = str(row.get("source_family", "unknown"))
        counts_by_source[source_family] += 1
        total_rows += 1

    targets = compute_balanced_targets(
        counts_by_source=dict(counts_by_source),
        target_rows=stratify.target_train_rows,
    )
    if runtime.dry_run:
        return {"input_rows": total_rows, "targets": targets, "counts_by_source_family": dict(counts_by_source)}

    rng = random.Random(stratify.seed)
    reservoirs: dict[str, list[dict[str, object]]] = {source: [] for source, target in targets.items() if target > 0}
    seen_counts: dict[str, int] = defaultdict(int)
    for row in tqdm(iter_jsonl(paths.filtered_path), total=total_rows, desc="Stratified sampling", unit="row"):
        source_family = str(row.get("source_family", "unknown"))
        target = targets.get(source_family, 0)
        if target <= 0:
            continue
        seen_counts[source_family] += 1
        reservoir = reservoirs[source_family]
        if len(reservoir) < target:
            reservoir.append(row)
            continue
        replacement_index = rng.randint(1, seen_counts[source_family])
        if replacement_index <= target:
            reservoir[replacement_index - 1] = row

    sampled_rows: list[dict[str, object]] = []
    for source_family in sorted(reservoirs):
        sampled_rows.extend(reservoirs[source_family])
    rng.shuffle(sampled_rows)
    if paths.stratified_path.exists():
        paths.stratified_path.unlink()
    for row in sampled_rows:
        write_jsonl_row(output_path=paths.stratified_path, row=row)

    return {
        "input_rows": total_rows,
        "output_rows": len(sampled_rows),
        "targets": targets,
        "counts_by_source_family": dict(counts_by_source),
    }


def build_prompt_messages(prompt_text: str, export_config: ExportConfig) -> list[dict[str, str]]:
    """Convert a plain-text prompt into RLHFDataset chat-message format.

    Args:
        prompt_text: Source prompt string.
        export_config: Export-stage configuration.

    Returns:
        Single-message chat prompt list.
    """

    return [{"role": export_config.prompt_role, "content": prompt_text}]


def build_export_row(
    *, row: dict[str, object], index: int, export_config: ExportConfig
) -> dict[str, object]:
    """Convert one stratified source row into RLHF parquet format.

    Args:
        row: Stratified source row.
        index: Zero-based exported row index.
        export_config: Export-stage configuration.

    Returns:
        RLHFDataset-compatible exported row.
    """

    source_prompt_text = str(row["prompt"])
    ground_truth_values = extract_ground_truth_values(row=row)
    source_family = str(row.get("source_family", normalize_source_family(row=row)))
    extra_info = dict(cast(dict[str, object], row.get("extra_info") or {}))
    extra_info.update(
        {
            "index": index,
            "source_family": source_family,
            "dataset_source": str(row.get("dataset_source", "")),
            "original_dataset": str(row.get("original_dataset", "")),
            "source_row_id": str(row.get("id", row.get("key", ""))),
            "passrate": row.get("passrate"),
            "total_rollouts": row.get("total_rollouts"),
            "total_correct_rollouts": row.get("total_correct_rollouts"),
        }
    )
    export_row = dict(row)
    export_row["source_prompt_text"] = source_prompt_text
    export_row["prompt"] = build_prompt_messages(prompt_text=source_prompt_text, export_config=export_config)
    export_row["data_source"] = source_family
    export_row["reward_model"] = {"ground_truth": ground_truth_values}
    export_row["extra_info"] = extra_info
    return export_row


def audit_rows(rows: Iterable[dict[str, object]]) -> SourceAuditSummary:
    """Summarize source composition across an iterable of rows.

    Args:
        rows: Row iterable to audit.

    Returns:
        Stage-level source audit summary.
    """

    row_count = 0
    counts_by_dataset_label: Counter[str] = Counter()
    counts_by_dataset_source: Counter[str] = Counter()
    counts_by_original_dataset: Counter[str] = Counter()
    counts_by_source_family: Counter[str] = Counter()
    for row in rows:
        row_count += 1
        dataset_labels = normalize_dataset_labels(value=row.get("dataset"))
        counts_by_dataset_label["|".join(dataset_labels) or "unknown"] += 1
        counts_by_dataset_source[str(row.get("dataset_source", "unknown"))] += 1
        counts_by_original_dataset[str(row.get("original_dataset", "unknown"))] += 1
        counts_by_source_family[normalize_source_family(row=row)] += 1
    return SourceAuditSummary(
        row_count=row_count,
        counts_by_dataset_label=dict(counts_by_dataset_label),
        counts_by_dataset_source=dict(counts_by_dataset_source),
        counts_by_original_dataset=dict(counts_by_original_dataset),
        counts_by_source_family=dict(counts_by_source_family),
    )


def build_source_audit(paths: PipelinePaths) -> dict[str, object]:
    """Build source-audit payload across pipeline stages.

    Args:
        paths: Pipeline output paths.

    Returns:
        JSON-serializable source-audit payload.
    """

    audit_payload: dict[str, object] = {}
    path_by_stage = {
        "sample": paths.raw_sample_path,
        "filter": paths.filtered_path,
        "stratify": paths.stratified_path,
    }
    for stage_name, stage_path in path_by_stage.items():
        if not stage_path.exists():
            continue
        audit_payload[stage_name] = asdict(audit_rows(rows=iter_jsonl(stage_path)))
    return audit_payload


def run_export_stage(
    *, export_config: ExportConfig, paths: PipelinePaths, runtime: RuntimeConfig, sampling: SamplingConfig
) -> dict[str, object]:
    """Export the final stratified train split and audit artifacts.

    Args:
        export_config: Export-stage configuration.
        paths: Pipeline output paths.
        runtime: Runtime configuration.
        sampling: Sampling configuration used for manifest metadata.

    Returns:
        Metadata describing exported files and train size.
    """

    if not paths.stratified_path.exists():
        raise SystemExit(f"Missing stratified rows: {paths.stratified_path}")
    stratified_rows = list(iter_jsonl(paths.stratified_path))
    if runtime.dry_run:
        return {"train_rows": len(stratified_rows), "train_parquet_path": str(paths.train_parquet_path)}

    export_rows = [
        build_export_row(row=row, index=index, export_config=export_config)
        for index, row in enumerate(stratified_rows)
    ]
    dataset = Dataset.from_list(export_rows)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(paths.train_parquet_path))

    source_audit = build_source_audit(paths=paths)
    paths.source_audit_path.write_text(json.dumps(source_audit, indent=2), encoding="utf-8")
    manifest = {
        "dataset_name": sampling.dataset_name,
        "split": sampling.split,
        "train_rows": len(export_rows),
        "files": {
            "train_parquet": str(paths.train_parquet_path),
            "source_audit": str(paths.source_audit_path),
        },
        "prompt_format": "chat_messages",
        "filter_rule": "dataset contains math and required training fields are present",
    }
    paths.manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "train_rows": len(export_rows),
        "train_parquet_path": str(paths.train_parquet_path),
        "manifest_path": str(paths.manifest_path),
        "source_audit_path": str(paths.source_audit_path),
    }


def build_configs(
    args: argparse.Namespace,
) -> tuple[RuntimeConfig, SamplingConfig, FilterConfig, StratifyConfig, ExportConfig]:
    """Build typed configuration objects from parsed CLI arguments.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Runtime, sampling, filter, stratify, and export configs.
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
        dataset_name=args.dataset_name,
        split=args.split,
        sample_rows=args.sample_rows,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
    )
    filter_config = FilterConfig(required_dataset_label="math")
    stratify = StratifyConfig(target_train_rows=args.target_train_rows, seed=args.seed)
    export_config = ExportConfig(prompt_role=args.prompt_role)
    return runtime, sampling, filter_config, stratify, export_config


def stage_config_hashes(
    *,
    sampling: SamplingConfig,
    filter_config: FilterConfig,
    stratify: StratifyConfig,
    export_config: ExportConfig,
) -> dict[Stage, str]:
    """Compute config hashes for all pipeline stages.

    Args:
        sampling: Sampling configuration.
        filter_config: Filter configuration.
        stratify: Stratify configuration.
        export_config: Export configuration.

    Returns:
        Config hashes keyed by stage.
    """

    return {
        "sample": hash_config(asdict(sampling)),
        "filter": hash_config(asdict(filter_config)),
        "stratify": hash_config(asdict(stratify)),
        "export": hash_config(asdict(export_config)),
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the staged RL dataset builder.

    Args:
        None.

    Returns:
        Parsed CLI namespace.
    """

    parser = argparse.ArgumentParser(description="Build a staged math-only RL dataset from Dolci-Think-RL-7B.")
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument("--split", default=DEFAULT_SPLIT)
    parser.add_argument(
        "--stage",
        choices=("sample", "filter", "stratify", "export", "all"),
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    parser.add_argument("--target-train-rows", type=int, default=DEFAULT_TARGET_TRAIN_ROWS)
    parser.add_argument("--shuffle-buffer", type=int, default=DEFAULT_SHUFFLE_BUFFER)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prompt-role", default="user")
    parser.add_argument("--confirm-threshold", type=int, default=DEFAULT_CONFIRM_THRESHOLD)
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint for the staged RL dataset builder.

    Args:
        None.

    Returns:
        None.
    """

    args = parse_args()
    runtime, sampling, filter_config, stratify, export_config = build_configs(args=args)
    paths = resolve_paths(runtime=runtime)
    state = load_state(path=paths.state_path)
    hashes = stage_config_hashes(
        sampling=sampling,
        filter_config=filter_config,
        stratify=stratify,
        export_config=export_config,
    )
    completed: dict[Stage, bool] = {
        "sample": is_stage_complete(
            state=state,
            stage="sample",
            expected_hash=hashes["sample"],
            required_path=paths.raw_sample_path,
        ),
        "filter": is_stage_complete(
            state=state,
            stage="filter",
            expected_hash=hashes["filter"],
            required_path=paths.filtered_path,
        ),
        "stratify": is_stage_complete(
            state=state,
            stage="stratify",
            expected_hash=hashes["stratify"],
            required_path=paths.stratified_path,
        ),
        "export": is_stage_complete(
            state=state,
            stage="export",
            expected_hash=hashes["export"],
            required_path=paths.train_parquet_path,
        ),
    }

    if runtime.stage == "all":
        stages_to_run = [cast(Stage, stage) for stage in STAGE_ORDER if not completed[cast(Stage, stage)]]
    elif runtime.stage is None:
        next_stage = choose_auto_stage(completed=completed)
        stages_to_run = [next_stage] if next_stage else []
    else:
        stages_to_run = [cast(Stage, runtime.stage)]

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
        elif stage == "export":
            metadata = run_export_stage(
                export_config=export_config,
                paths=paths,
                runtime=runtime,
                sampling=sampling,
            )
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
