from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import tiktoken

from pipeline_types import PipelinePaths, STAGE_ORDER, Stage, StageStatus

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse a dotenv file.

    Args:
        path: Dotenv file path.

    Returns:
        Environment value mapping.
    """
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_paths(output_dir: Path, state_filename: str) -> PipelinePaths:
    """Resolve pipeline output paths.

    Args:
        output_dir: Output directory.
        state_filename: State file name.

    Returns:
        Resolved path bundle.
    """
    return PipelinePaths(
        output_dir=output_dir,
        raw_sample_path=output_dir / "raw_sample.jsonl",
        filtered_path=output_dir / "filtered_candidates.jsonl",
        stratified_path=output_dir / "stratified_sample.jsonl",
        transformed_path=output_dir / "transformed_output.jsonl",
        state_path=output_dir / state_filename,
    )


def utc_now() -> str:
    """Get current UTC timestamp string.

    Returns:
        ISO-8601 timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def write_jsonl(output_path: Path, row: dict[str, object]) -> None:
    """Append one row to JSONL.

    Args:
        output_path: Target JSONL path.
        row: JSON-serializable object.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def decode_json_line(line: str, line_number: int) -> list[dict[str, object]]:
    """Decode one line that may contain multiple JSON objects.

    Args:
        line: Raw line.
        line_number: 1-based line number.

    Returns:
        Decoded JSON objects.

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
    """Count non-empty JSONL lines.

    Args:
        path: JSONL path.

    Returns:
        Non-empty line count.
    """
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def iter_jsonl(path: Path) -> Iterator[dict[str, object]]:
    """Iterate decoded JSON rows from JSONL.

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
    """Extract `<think>` blocks from text.

    Args:
        text: Assistant content.

    Returns:
        Think block contents.
    """
    return [match.strip() for match in THINK_PATTERN.findall(text)]


def compute_think_token_count(encoding: tiktoken.Encoding, messages: list[dict[str, object]]) -> int:
    """Count think-block tokens in assistant messages.

    Args:
        encoding: Tokenizer encoding.
        messages: Conversation messages.

    Returns:
        Total token count.
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
    """Hash config dictionary.

    Args:
        value: Config dict.

    Returns:
        SHA256 hash.
    """
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_state(path: Path) -> dict[str, object]:
    """Load pipeline state.

    Args:
        path: State path.

    Returns:
        State payload.
    """
    if not path.exists():
        return {"version": 1, "stages": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: dict[str, object]) -> None:
    """Save pipeline state.

    Args:
        path: State path.
        state: State payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def get_stage_status(state: dict[str, object], stage: Stage) -> StageStatus | None:
    """Get one stage status object.

    Args:
        state: State payload.
        stage: Stage name.

    Returns:
        Stage status or None.
    """
    stages = state.get("stages", {})
    payload = stages.get(stage) if isinstance(stages, dict) else None
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
    """Set stage status in state.

    Args:
        state: Mutable state object.
        stage: Stage name.
        completed: Completion state.
        config_hash: Stage config hash.
        metadata: Stage metadata payload.
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
    """Check whether stage completion is valid.

    Args:
        state: State payload.
        stage: Stage name.
        expected_hash: Current config hash.
        required_path: Expected output path.

    Returns:
        True if stage is complete and compatible.
    """
    status = get_stage_status(state=state, stage=stage)
    if status is None:
        return False
    return status.completed and status.config_hash == expected_hash and required_path.exists()


def choose_auto_stage(completed: dict[Stage, bool]) -> Stage | None:
    """Pick next incomplete stage.

    Args:
        completed: Completion map by stage.

    Returns:
        Next stage or None.
    """
    for stage in STAGE_ORDER:
        if not completed[stage]:
            return stage  # type: ignore[return-value]
    return None


def confirm_large_work(
    stage: Stage,
    rows_left: int,
    confirm_threshold: int,
    auto_yes: bool,
    dry_run: bool,
) -> None:
    """Request confirmation when work is large.

    Args:
        stage: Stage name.
        rows_left: Rows remaining.
        confirm_threshold: Threshold requiring confirmation.
        auto_yes: Skip confirmation when true.
        dry_run: Skip confirmation in dry-run mode.
    """
    if dry_run or auto_yes or rows_left <= confirm_threshold:
        return
    answer = input(
        f"Stage '{stage}' has {rows_left} rows left. Continue? [y/N]: "
    ).strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Aborted by user.")


def count_sources(path: Path, source_key: str = "dataset_source") -> dict[str, int]:
    """Count rows by source in a JSONL file.

    Args:
        path: JSONL path.
        source_key: Source field name.

    Returns:
        Count map by source.
    """
    counter: Counter[str] = Counter()
    for row in iter_jsonl(path=path):
        source = str(row.get(source_key, "unknown"))
        counter[source] += 1
    return dict(counter)
