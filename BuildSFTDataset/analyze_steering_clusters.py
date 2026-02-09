from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import time
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.manifold import TSNE
from tqdm import tqdm

DEFAULT_BASE_DIR = Path(
    "/Users/olliepro/Code/School/DecomposedReasoning/BuildSFTDataset"
)
DEFAULT_TRANSFORMED_PATH = DEFAULT_BASE_DIR / "output" / "transformed_output.jsonl"
DEFAULT_OG_PATH = DEFAULT_BASE_DIR / "output" / "stratified_sample.jsonl"
DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "output" / "cluster_analysis"
DEFAULT_ENV_FILE = DEFAULT_BASE_DIR / ".env"
DEFAULT_NAMING_MODEL = "gemini-3-flash-preview"
DEFAULT_EMBEDDING_MODEL = "gemini-embedding-001"
DEFAULT_MIN_CLUSTER_SIZE = 5
DEFAULT_MAX_CLUSTER_SIZE = 2000
DEFAULT_NAMING_CONCURRENCY = 100
DEFAULT_EMBED_CONCURRENCY = 20
DEFAULT_TOKEN_CONCURRENCY = 20
DEFAULT_EMBED_BATCH_SIZE = 100
DEFAULT_NAMING_REQUESTS_PER_MINUTE = 1200
DEFAULT_EMBED_REQUESTS_PER_MINUTE = 2000
DEFAULT_TOKEN_REQUESTS_PER_MINUTE = 3500
DEFAULT_TSNE_SAMPLE_SIZE = 10_000
DEFAULT_SEED = 42
DEFAULT_API_TIMEOUT_SECONDS = 90
DEFAULT_MAX_RETRIES = 12
DEFAULT_PLOT_DPI = 260
LIGHT_RED_BAR_COLOR = "#f4a6a6"
LIGHT_RED_BAR_COLOR_ALT = "#f8c3c3"
DATASET_SOURCE_LABEL_ALIASES: dict[str, str] = {
    "saumyamalik/openthoughts3-full-filtered-science-decontam-v2": (
        "OpenThoughts 3 Science (Saumya Malik)"
    ),
    "saumyamalik/openthoughts3-full-filtered-math-decontam-v2": (
        "OpenThoughts 3 Math (Saumya Malik)"
    ),
    "saumyamalik/correct-python-sft-187k-x16-thoughts-filtered-decontam-v2": (
        "Correct Python SFT 187K x16 (Saumya Malik)"
    ),
    "allenai/oasst1-r1-format-filtered-keyword-filtered-filter-datecutoff-chinese-filtered": (
        "OASST1 R1 (AllenAI, Chinese Filtered)"
    ),
    "allenai/synthetic-2-sft-cn-fltrd-final-ngram-filtered-chinese-filtered": (
        "Synthetic 2 SFT (AllenAI, Chinese Filtered)"
    ),
}
DATASET_OWNER_LABEL_ALIASES: dict[str, str] = {
    "saumyamalik": "Saumya Malik",
    "allenai": "AllenAI",
}
STAGE_ORDER = (
    "extract",
    "tokens",
    "cluster1",
    "name1",
    "cluster2",
    "assign_noise",
    "name2",
    "report",
)

THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
STEER_PATTERN = re.compile(
    r"<(?:steer|steering)>(.*?)</(?:steer|steering)>", flags=re.IGNORECASE | re.DOTALL
)
EXEC_PATTERN = re.compile(
    r"<(?:exec|execute|execution)>(.*?)</(?:exec|execute|execution)>",
    flags=re.IGNORECASE | re.DOTALL,
)
SEGMENT_BLOCK_PATTERN = re.compile(
    r"<(?:steer|steering|exec|execute|execution)\b[^>]*>.*?</(?:steer|steering|exec|execute|execution)\s*>",
    flags=re.IGNORECASE | re.DOTALL,
)
SEGMENT_TAG_PATTERN = re.compile(
    r"</?(?:steer|steering|exec|execute|execution)\b[^>]*>", flags=re.IGNORECASE
)
CODE_FENCE_PATTERN = re.compile(
    r"^```(?:json|xml)?\s*|```$", flags=re.IGNORECASE | re.MULTILINE
)


class ModelsAPIProtocol(Protocol):
    """Protocol for async Gemini model methods used by this pipeline.

    Args:
        model: Model name.
        contents: Input text payload.
        config: Optional model config object.

    Returns:
        Provider response object with method-specific fields.
    """

    async def count_tokens(
        self, *, model: str, contents: str, config: Any | None = None
    ) -> Any: ...

    async def embed_content(
        self, *, model: str, contents: str, config: Any | None = None
    ) -> Any: ...

    async def generate_content(
        self, *, model: str, contents: str, config: Any | None = None
    ) -> Any: ...


class AioAPIProtocol(Protocol):
    """Protocol for async Gemini aio wrapper.

    Args:
        models: Async models endpoint object.
    """

    models: ModelsAPIProtocol


class GeminiClientProtocol(Protocol):
    """Protocol for Gemini clients used by this script.

    Args:
        aio: Async API wrapper.
    """

    aio: AioAPIProtocol


@dataclass(frozen=True)
class SteeringRecord:
    """Section-level extracted steering/execution record.

    Args:
        section_id: Unique section identifier.
        row_id: Dataset row id.
        dataset_source: Source dataset label.
        steering_text: Steering text content.
        execution_text: Execution text content.
        original_think_text: Original think block from OG sample.
        new_think_text: Reformatted think block from transformed output.

    Example:
        >>> SteeringRecord(
        ...     section_id="row1-sec0",
        ...     row_id="row1",
        ...     dataset_source="src",
        ...     steering_text="plan",
        ...     execution_text="do",
        ...     original_think_text="old",
        ...     new_think_text="new",
        ... )
    """

    section_id: str
    row_id: str
    dataset_source: str
    steering_text: str
    execution_text: str
    original_think_text: str
    new_think_text: str


@dataclass(frozen=True)
class ClusterLabelRecord:
    """Cluster labels and names for one section.

    Args:
        section_id: Unique section identifier.
        cluster_pass1: HDBSCAN label from pass1.
        cluster_name_pass1: Name returned for pass1 cluster.
        cluster_pass2: Final cluster label after noise assignment.
        cluster_name_pass2: Final pass2 cluster name.
        noise_assigned: True if this section was pass2 noise and reassigned.
    """

    section_id: str
    cluster_pass1: int
    cluster_name_pass1: str
    cluster_pass2: int
    cluster_name_pass2: str
    noise_assigned: bool


@dataclass(frozen=True)
class TokenStatsRecord:
    """Token metrics for one section.

    Args:
        section_id: Unique section identifier.
        row_id: Dataset row id.
        steering_tokens: Gemini token count for steering string.
        execution_tokens: Gemini token count for execution string.
        original_think_tokens: Gemini token count for OG think block.
        new_think_tokens: Gemini token count for transformed think block.
        think_token_delta: New minus original think token count.
    """

    section_id: str
    row_id: str
    steering_tokens: int
    execution_tokens: int
    original_think_tokens: int
    new_think_tokens: int
    think_token_delta: int


@dataclass(frozen=True)
class RunConfig:
    """Runtime configuration for clustering pipeline.

    Args:
        transformed_path: Transformed JSONL input path.
        og_path: OG stratified sample JSONL path.
        output_dir: Analysis output directory.
        env_file: Dotenv path for API key lookup.
        naming_model: Gemini model for naming and count_tokens.
        embedding_model: Gemini embedding model name.
        min_cluster_size: HDBSCAN minimum cluster size.
        max_cluster_size: HDBSCAN maximum cluster size.
        naming_concurrency: Async concurrency for naming calls.
        embed_concurrency: Async concurrency for embedding calls.
        token_concurrency: Async concurrency for count_tokens calls.
        embed_batch_size: Number of texts per embedding request.
        naming_requests_per_minute: Request cap for naming API calls.
        embed_requests_per_minute: Request cap for embedding API calls.
        token_requests_per_minute: Request cap for token API calls.
        tsne_sample_size: Sample size for t-SNE plots.
        seed: Random seed.
        resume: Resume from existing stage outputs.
        stage: Stage to run.
        api_timeout_seconds: Timeout per API call.
        max_retries: Retry limit for transient API failures.
    """

    transformed_path: Path
    og_path: Path
    output_dir: Path
    env_file: Path
    naming_model: str
    embedding_model: str
    min_cluster_size: int
    max_cluster_size: int
    naming_concurrency: int
    embed_concurrency: int
    token_concurrency: int
    embed_batch_size: int
    naming_requests_per_minute: int
    embed_requests_per_minute: int
    token_requests_per_minute: int
    tsne_sample_size: int
    seed: int
    resume: bool
    stage: str
    api_timeout_seconds: int
    max_retries: int


@dataclass(frozen=True)
class RetryConfig:
    """Retry and timeout settings for async API calls.

    Args:
        timeout_seconds: Timeout per request.
        max_retries: Maximum attempts.
        base_sleep_seconds: Base backoff in seconds.
        max_sleep_seconds: Maximum backoff in seconds.
    """

    timeout_seconds: int
    max_retries: int
    base_sleep_seconds: float = 1.5
    max_sleep_seconds: float = 120.0


@dataclass(frozen=True)
class StageStatus:
    """Persisted stage completion metadata.

    Args:
        completed: True if stage completed.
        updated_at: ISO timestamp.
        metadata: Stage metadata payload.
    """

    completed: bool
    updated_at: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class CacheStats:
    """Cache hit and miss counts.

    Args:
        hit_count: Number of cache hits.
        miss_count: Number of cache misses.
    """

    hit_count: int
    miss_count: int


@dataclass(frozen=True)
class UniqueTextClusteringResult:
    """Cluster outputs mapped from unique text inputs back to all rows.

    Args:
        row_labels: Cluster labels aligned to original input rows.
        row_embeddings: Embeddings aligned to original input rows.
        unique_text_count: Number of unique text strings clustered.
        duplicate_row_count: Number of rows mapped from duplicate strings.
        cache_stats: Embedding cache hit/miss stats from the unique embed call.

    Example:
        >>> labels = np.asarray([0, 1, 0], dtype=np.int64)
        >>> vectors = np.asarray([[1.0], [2.0], [1.0]], dtype=np.float32)
        >>> result = UniqueTextClusteringResult(
        ...     row_labels=labels,
        ...     row_embeddings=vectors,
        ...     unique_text_count=2,
        ...     duplicate_row_count=1,
        ...     cache_stats=CacheStats(hit_count=0, miss_count=2),
        ... )
        >>> int(result.unique_text_count)
        2
    """

    row_labels: np.ndarray
    row_embeddings: np.ndarray
    unique_text_count: int
    duplicate_row_count: int
    cache_stats: CacheStats


@dataclass(frozen=True)
class ThinkResidualRecord:
    """Non-whitespace think-text content found outside steer/execute segments.

    Args:
        row_id: Source row identifier.
        dataset_source: Dataset source name.
        steer_segment_count: Number of `<steer>/<steering>` segments in think text.
        residual_char_count: Character count of residual text outside segment blocks.
        residual_preview: Preview text for quick inspection.

    Example:
        >>> ThinkResidualRecord(
        ...     row_id="row-1",
        ...     dataset_source="sample",
        ...     steer_segment_count=2,
        ...     residual_char_count=18,
        ...     residual_preview="leftover text",
        ... )
    """

    row_id: str
    dataset_source: str
    steer_segment_count: int
    residual_char_count: int
    residual_preview: str


class JsonlCache:
    """Simple persistent JSONL key-value cache.

    Each line is a JSON object with keys `key` and `value`.

    Args:
        cache_path: JSONL file path.

    Example:
        >>> cache = JsonlCache(cache_path=Path('/tmp/example_cache.jsonl'))
        >>> cache.set(key='a', value=1)
        >>> cache.get(key='a')
        1
    """

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._store: dict[str, Any] = {}
        self._write_lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                key = row.get("key")
                if isinstance(key, str):
                    self._store[key] = row.get("value")

    def get(self, key: str) -> Any | None:
        """Return cached value for key if present.

        Args:
            key: Cache key.

        Returns:
            Cached value or None.
        """

        return self._store.get(key)

    async def set(self, key: str, value: Any) -> None:
        """Persist one cache value.

        Args:
            key: Cache key.
            value: JSON-serializable value.
        """

        if key in self._store:
            return
        self._store[key] = value
        payload = {"key": key, "value": value}
        async with self._write_lock:
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False))
                handle.write("\n")


def parse_dotenv(path: Path) -> dict[str, str]:
    """Parse `.env` file into a key-value mapping.

    Args:
        path: Dotenv path.

    Returns:
        Parsed key-value map.
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


def resolve_api_key(env_file: Path) -> str:
    """Resolve Gemini API key from environment and `.env` values.

    Args:
        env_file: Dotenv path.

    Returns:
        API key string.

    Raises:
        SystemExit: If key is missing.
    """

    dotenv_values = parse_dotenv(path=env_file)
    candidates = [
        os.getenv("VERTEX_KEY"),
        os.getenv("GEMINI_API_KEY"),
        os.getenv("GOOGLE_API_KEY"),
        os.getenv("GOOGLE_CLOUD_API_KEY"),
        dotenv_values.get("VERTEX_KEY"),
        dotenv_values.get("GEMINI_API_KEY"),
        dotenv_values.get("GOOGLE_API_KEY"),
        dotenv_values.get("GOOGLE_CLOUD_API_KEY"),
    ]
    for candidate in candidates:
        if candidate:
            return candidate
    raise SystemExit(
        "Missing API key. Set VERTEX_KEY or GEMINI_API_KEY in env or .env."
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file into a list of dictionaries.

    Args:
        path: JSONL path.

    Returns:
        Parsed rows.
    """

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            assert isinstance(row, dict), f"Expected dict row in {path}"
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write dictionaries to JSONL file.

    Args:
        path: Destination path.
        rows: Row iterable.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def utc_now() -> str:
    """Return current UTC timestamp as ISO string."""

    return datetime.now(timezone.utc).isoformat()


def debug_print(message: str) -> None:
    """Print a timestamped debug line and flush stdout.

    Args:
        message: Debug message payload.
    """

    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[debug {timestamp}] {message}", flush=True)


def hash_text(kind: str, model: str, text: str) -> str:
    """Build SHA256 cache key from kind, model, and text.

    Args:
        kind: Value category label.
        model: Model name.
        text: Input text payload.

    Returns:
        Hex SHA256 digest.
    """

    payload = f"{kind}\n{model}\n{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def extract_first_assistant_content(messages: list[dict[str, Any]]) -> str | None:
    """Return first assistant content string from message list.

    Args:
        messages: Conversation messages.

    Returns:
        Assistant content or None.
    """

    for message in messages:
        if message.get("role") == "assistant" and isinstance(
            message.get("content"), str
        ):
            return str(message["content"])
    return None


def extract_single_think_block(
    content: str, row_id: str, source_name: str
) -> tuple[str | None, list[dict[str, Any]]]:
    """Extract exactly one think block and return anomalies.

    Args:
        content: Assistant message content.
        row_id: Row identifier.
        source_name: Dataset source string.

    Returns:
        Tuple of think block text or None and anomaly list.
    """

    matches = THINK_PATTERN.findall(content)
    if len(matches) == 1:
        return matches[0].strip(), []
    anomaly = {
        "row_id": row_id,
        "dataset_source": source_name,
        "anomaly": "invalid_think_block_count",
        "count": len(matches),
    }
    return None, [anomaly]


def parse_steering_execution_pairs(
    think_text: str,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Parse steering/execution pairs from think text.

    Args:
        think_text: Think block text.

    Returns:
        Tuple of paired section list and parse metadata.

    Example:
        >>> parse_steering_execution_pairs('<steer>A</steer><exec>B</exec>')[0]
        [('A', 'B')]
    """

    steering_items = [item.strip() for item in STEER_PATTERN.findall(think_text)]
    execution_items = [item.strip() for item in EXEC_PATTERN.findall(think_text)]
    pair_count = min(len(steering_items), len(execution_items))
    pairs = [
        (steering_items[index], execution_items[index]) for index in range(pair_count)
    ]
    parse_meta = {
        "steering_count": len(steering_items),
        "execution_count": len(execution_items),
        "paired_count": pair_count,
        "unmatched_steering": max(len(steering_items) - pair_count, 0),
        "unmatched_execution": max(len(execution_items) - pair_count, 0),
    }
    return pairs, parse_meta


def count_steering_segments(think_text: str) -> int:
    """Count steering segments in a think block.

    Args:
        think_text: Think block text.

    Returns:
        Number of `<steer>/<steering>` segments.

    Example:
        >>> count_steering_segments("<steer>a</steer><steering>b</steering>")
        2
    """

    return len(STEER_PATTERN.findall(think_text))


def residual_text_outside_segments(think_text: str) -> str:
    """Return non-whitespace text outside steer/execute blocks.

    Args:
        think_text: Think block text.

    Returns:
        Residual text after removing known segment blocks/tags.
        Empty string indicates no non-whitespace residual text.

    Example:
        >>> residual_text_outside_segments("<steer>a</steer> \\n ")
        ''
    """

    without_blocks = SEGMENT_BLOCK_PATTERN.sub("", think_text)
    without_tags = SEGMENT_TAG_PATTERN.sub("", without_blocks)
    return without_tags.strip()


def collect_think_residual_records(
    think_df: pd.DataFrame,
    *,
    think_column: str,
    max_preview_chars: int = 220,
) -> list[ThinkResidualRecord]:
    """Collect rows whose think text has non-whitespace text outside segments.

    Args:
        think_df: DataFrame containing row_id, dataset_source, and think text column.
        think_column: Name of the think text column to inspect.
        max_preview_chars: Maximum characters stored in residual previews.

    Returns:
        List of residual records for rows with non-empty residual text.
    """

    assert think_column in think_df.columns, f"Missing think column: {think_column}"
    records: list[ThinkResidualRecord] = []
    for row in think_df.itertuples(index=False):
        think_text = getattr(row, think_column, "")
        if not isinstance(think_text, str):
            continue
        residual_text = residual_text_outside_segments(think_text=think_text)
        if residual_text == "":
            continue
        row_id = str(getattr(row, "row_id", ""))
        dataset_source = str(getattr(row, "dataset_source", ""))
        preview = residual_text.replace("\n", " ")[:max_preview_chars]
        records.append(
            ThinkResidualRecord(
                row_id=row_id,
                dataset_source=dataset_source,
                steer_segment_count=count_steering_segments(think_text=think_text),
                residual_char_count=len(residual_text),
                residual_preview=preview,
            )
        )
    return records


def load_state(path: Path) -> dict[str, Any]:
    """Load run state JSON.

    Args:
        path: State file path.

    Returns:
        State dict.
    """

    if not path.exists():
        return {"version": 1, "stages": {}, "updated_at": utc_now()}
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "State payload must be dict"
    return payload


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Persist run state JSON.

    Args:
        path: State file path.
        state: State payload.
    """

    state["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def mark_stage_complete(
    state: dict[str, Any], stage: str, metadata: dict[str, Any]
) -> None:
    """Mark stage complete with metadata.

    Args:
        state: Mutable state dict.
        stage: Stage name.
        metadata: Stage metadata.
    """

    stages = state.setdefault("stages", {})
    assert isinstance(stages, dict), "State stages must be dict"
    stages[stage] = asdict(
        StageStatus(completed=True, updated_at=utc_now(), metadata=metadata)
    )


def stage_is_complete(state: dict[str, Any], stage: str) -> bool:
    """Return True if stage is marked complete.

    Args:
        state: State dict.
        stage: Stage name.

    Returns:
        Completion status.
    """

    stages = state.get("stages", {})
    if not isinstance(stages, dict):
        return False
    stage_payload = stages.get(stage)
    return bool(
        isinstance(stage_payload, dict) and stage_payload.get("completed") is True
    )


def strip_code_fences(text: str) -> str:
    """Strip top-level markdown code fences from text.

    Args:
        text: Input text.

    Returns:
        Text with simple code-fence wrappers removed.
    """

    cleaned = text.strip()
    cleaned = CODE_FENCE_PATTERN.sub("", cleaned).strip()
    return cleaned


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model output text.

    Args:
        text: Model output.

    Returns:
        Parsed dict or None.
    """

    cleaned = strip_code_fences(text=text)
    decoder = json.JSONDecoder()
    for start_index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[start_index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def required_stage_outputs(config: RunConfig) -> dict[str, list[Path]]:
    """Map stage names to required output files.

    Args:
        config: Runtime configuration.

    Returns:
        Mapping of stage to required file list.
    """

    base = config.output_dir
    return {
        "extract": [base / "sections.parquet", base / "think_rows.parquet"],
        "tokens": [
            base / "token_stats.parquet",
            base / "think_token_stats.parquet",
            base / "token_summary.json",
        ],
        "cluster1": [
            base / "clusters_pass1.parquet",
            base / "embeddings_pass1.npy",
            base / "plots" / "tsne_pass1.png",
        ],
        "name1": [
            base / "naming_prompts_pass1.jsonl",
            base / "naming_responses_pass1.jsonl",
            base / "clusters_pass1.parquet",
        ],
        "cluster2": [
            base / "clusters_pass2.parquet",
            base / "embeddings_pass2.npy",
            base / "plots" / "tsne_pass2.png",
        ],
        "assign_noise": [
            base / "clusters_final.parquet",
            base / "plots" / "tsne_pass2_noise_assigned.png",
        ],
        "name2": [
            base / "naming_prompts_pass2.jsonl",
            base / "naming_responses_pass2.jsonl",
            base / "clusters_final.parquet",
        ],
        "report": [
            base / "cluster_report.md",
            base / "cluster_report.json",
            base / "plots" / "final_cluster_sizes.png",
            base / "plots" / "steering_token_hist.png",
            base / "plots" / "execution_token_hist.png",
            base / "plots" / "execution_to_steering_ratio_hist.png",
            base / "plots" / "think_tokens_og_vs_new.png",
            base / "plots" / "think_token_diff_hist.png",
            base / "plots" / "think_tokens_vs_steer_segments_scatter.png",
            base / "think_text_outside_segments.jsonl",
        ],
    }


def token_outputs_look_valid(config: RunConfig) -> bool:
    """Return True when token stage outputs are internally consistent.

    Args:
        config: Runtime configuration.

    Returns:
        True when expected files exist and contain non-empty tables.
    """

    token_stats_path = config.output_dir / "token_stats.parquet"
    think_stats_path = config.output_dir / "think_token_stats.parquet"
    summary_path = config.output_dir / "token_summary.json"
    if (
        not token_stats_path.exists()
        or not think_stats_path.exists()
        or not summary_path.exists()
    ):
        return False
    try:
        token_stats_df = pd.read_parquet(token_stats_path)
        think_stats_df = pd.read_parquet(think_stats_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    if token_stats_df.empty or think_stats_df.empty:
        return False
    expected_sections = (
        int(summary.get("sections", 0)) if isinstance(summary, dict) else 0
    )
    expected_rows = int(summary.get("rows", 0)) if isinstance(summary, dict) else 0
    if expected_sections <= 0 or expected_rows <= 0:
        return False
    return (
        token_stats_df.shape[0] == expected_sections
        and think_stats_df.shape[0] == expected_rows
    )


def stage_should_skip(config: RunConfig, state: dict[str, Any], stage: str) -> bool:
    """Return True when stage should be skipped due to resume.

    Args:
        config: Runtime configuration.
        state: Pipeline state.
        stage: Stage name.

    Returns:
        Skip decision.
    """

    if not config.resume:
        return False
    if not stage_is_complete(state=state, stage=stage):
        return False
    required_outputs = required_stage_outputs(config=config).get(stage, [])
    outputs_exist = all(path.exists() for path in required_outputs)
    if not outputs_exist:
        return False
    if stage == "tokens":
        return token_outputs_look_valid(config=config)
    return True


def to_numeric_series(values: Any, *, label: str) -> pd.Series:
    """Normalize arbitrary values into a numeric pandas Series.

    Args:
        values: Series-like values or array-like values.
        label: Diagnostic label used in assertion messages.

    Returns:
        Numeric pandas series with NaN for non-numeric values.
    """

    if isinstance(values, pd.Series):
        series = values
    elif isinstance(values, pd.DataFrame):
        assert (
            values.shape[1] == 1
        ), f"Expected one column for {label}, got {values.shape[1]}"
        series = values.iloc[:, 0]
    else:
        series = pd.Series(values)
    return cast(pd.Series, pd.to_numeric(series, errors="coerce"))


def map_cached_numeric_values(
    keys: pd.Series, cache_values: dict[str, Any], *, label: str
) -> pd.Series:
    """Map cache key series to integer values from a cache dictionary.

    Args:
        keys: Series of cache keys.
        cache_values: Cache mapping from key to numeric value.
        label: Diagnostic label.

    Returns:
        Integer pandas series.
    """

    mapped = keys.astype(str).apply(lambda key: cache_values[str(key)])
    numeric = to_numeric_series(values=mapped, label=label)
    assert numeric.notna().all(), f"Missing cache value for {label}"
    return numeric.astype(int)


def coerce_cluster_label(value: Any) -> int:
    """Convert a cluster label value into int.

    Args:
        value: Label from pandas index/group key.

    Returns:
        Integer cluster label.
    """

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        assert not math.isnan(value), "Cluster label cannot be NaN"
        return int(value)
    if isinstance(value, str):
        return int(value.strip())
    raise TypeError(f"Unsupported cluster label type: {type(value)!r}")


def summarize_series(values: Any) -> dict[str, float]:
    """Build numeric summary for a pandas series.

    Args:
        values: Numeric series-like values.

    Returns:
        Summary dict with percentiles.
    """

    cleaned = (
        to_numeric_series(values=values, label="summary_values").dropna().astype(float)
    )
    if cleaned.empty:
        return {}
    return {
        "count": float(cleaned.shape[0]),
        "mean": float(cleaned.mean()),
        "std": float(cleaned.std(ddof=0)),
        "min": float(cleaned.min()),
        "p25": float(cleaned.quantile(0.25)),
        "p50": float(cleaned.quantile(0.50)),
        "p75": float(cleaned.quantile(0.75)),
        "max": float(cleaned.max()),
    }


def build_client(api_key: str) -> genai.Client:
    """Create Gemini client using API key.

    Args:
        api_key: API key.

    Returns:
        Initialized client.
    """

    return genai.Client(api_key=api_key)


class AsyncMinuteRateLimiter:
    """Async sliding-window rate limiter for requests per minute.

    Args:
        requests_per_minute: Maximum requests allowed in a rolling 60s window.
    """

    def __init__(self, requests_per_minute: int) -> None:
        assert requests_per_minute > 0, "Rate limit must be positive"
        self.requests_per_minute = requests_per_minute
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        """Wait until one request slot is available."""

        window_seconds = 60.0
        while True:
            async with self._lock:
                now = time.monotonic()
                cutoff = now - window_seconds
                while self._timestamps and self._timestamps[0] <= cutoff:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return
                wait_seconds = window_seconds - (now - self._timestamps[0]) + 0.01
            await asyncio.sleep(min(max(wait_seconds, 0.05), 2.0))


def build_rate_limiter(requests_per_minute: int) -> AsyncMinuteRateLimiter | None:
    """Build optional request-per-minute limiter.

    Args:
        requests_per_minute: Rolling minute cap. Non-positive disables throttling.

    Returns:
        Rate limiter instance or None.
    """

    if requests_per_minute <= 0:
        return None
    return AsyncMinuteRateLimiter(requests_per_minute=requests_per_minute)


def is_rate_limit_exception(exc: Exception) -> bool:
    """Return True when exception indicates rate limiting."""

    message = str(exc).lower()
    rate_limit_signatures = [
        "429",
        "rate_limit_exceeded",
        "resource_exhausted",
        "quota exceeded",
        "quota",
    ]
    return any(signature in message for signature in rate_limit_signatures)


def is_retryable_exception(exc: Exception) -> bool:
    """Return True for transient API exceptions.

    Args:
        exc: Exception instance.

    Returns:
        Retryability decision.
    """

    message = str(exc).lower()
    retry_signatures = [
        "429",
        "rate",
        "timeout",
        "deadline",
        "temporarily unavailable",
        "connection",
        "503",
        "502",
        "504",
    ]
    return any(signature in message for signature in retry_signatures)


def compute_backoff_seconds(
    attempt: int, retry_config: RetryConfig, rate_limited: bool
) -> float:
    """Compute retry backoff duration.

    Args:
        attempt: 1-indexed retry attempt number.
        retry_config: Retry configuration.
        rate_limited: True for explicit quota/rate-limit failures.

    Returns:
        Sleep duration in seconds.
    """

    exponential = retry_config.base_sleep_seconds * (2 ** (attempt - 1))
    if rate_limited:
        base_sleep = max(exponential, 8.0)
        jitter = random.random() * 2.0
    else:
        base_sleep = exponential
        jitter = random.random() * 0.5
    return min(base_sleep, retry_config.max_sleep_seconds) + jitter


async def call_with_retry(
    call_factory: Callable[[], Any],
    retry_config: RetryConfig,
    request_label: str,
) -> Any:
    """Call an async API operation with timeout and retries.

    Args:
        call_factory: Async no-arg callable.
        retry_config: Retry settings.
        request_label: Label used in error messages.

    Returns:
        API response object.

    Raises:
        RuntimeError: If retries are exhausted.
    """

    last_error: Exception | None = None
    for attempt in range(1, retry_config.max_retries + 1):
        try:
            return await asyncio.wait_for(
                call_factory(), timeout=retry_config.timeout_seconds
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            retryable = is_retryable_exception(exc=exc)
            if attempt >= retry_config.max_retries or not retryable:
                error_text = str(exc).replace("\n", " ")[:220]
                debug_print(
                    "retry abort "
                    f"request={request_label} attempt={attempt}/{retry_config.max_retries} "
                    f"retryable={retryable} error={error_text}"
                )
                break
            rate_limited = is_rate_limit_exception(exc=exc)
            sleep_seconds = compute_backoff_seconds(
                attempt=attempt,
                retry_config=retry_config,
                rate_limited=rate_limited,
            )
            error_text = str(exc).replace("\n", " ")[:220]
            debug_print(
                "retry scheduled "
                f"request={request_label} attempt={attempt}/{retry_config.max_retries} "
                f"sleep={sleep_seconds:.2f}s rate_limited={rate_limited} error={error_text}"
            )
            await asyncio.sleep(sleep_seconds)
    raise RuntimeError(f"API call failed for {request_label}") from last_error


async def run_async_text_jobs(
    job_payloads: dict[str, str],
    cache: JsonlCache,
    concurrency: int,
    async_fn: Callable[[str], Any],
    retry_config: RetryConfig,
    requests_per_minute: int,
    progress_desc: str,
) -> tuple[dict[str, Any], CacheStats]:
    """Run async text-keyed jobs with cache and retries.

    Args:
        job_payloads: Mapping of cache key to text payload.
        cache: Persistent JSONL cache.
        concurrency: Max async concurrency.
        async_fn: Async function called with text payload.
        retry_config: Retry settings.
        requests_per_minute: Rolling one-minute request cap.
        progress_desc: Progress bar description label.

    Returns:
        Tuple of results by key and cache stats.
    """

    results: dict[str, Any] = {}
    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = build_rate_limiter(requests_per_minute=requests_per_minute)
    hit_count = 0
    miss_count = 0
    total_jobs = len(job_payloads)
    debug_print(
        f"{progress_desc}: start total_jobs={total_jobs} concurrency={concurrency} "
        f"requests_per_minute={requests_per_minute}"
    )

    async def _worker(cache_key: str, text: str) -> tuple[str, Any, bool]:
        cached_value = cache.get(key=cache_key)
        if cached_value is not None:
            return cache_key, cached_value, True

        async with semaphore:
            if rate_limiter is not None:
                await rate_limiter.acquire()

            async def _call() -> Any:
                return await async_fn(text)

            response_value = await call_with_retry(
                call_factory=_call,
                retry_config=retry_config,
                request_label=cache_key,
            )
            await cache.set(key=cache_key, value=response_value)
            return cache_key, response_value, False

    tasks = [
        asyncio.create_task(_worker(cache_key=key, text=text))
        for key, text in job_payloads.items()
    ]
    progress_bar = tqdm(
        total=len(tasks), desc=progress_desc, unit="req", dynamic_ncols=True
    )
    progress_interval = max(1, min(500, max(25, len(tasks) // 20)))
    try:
        for task in asyncio.as_completed(tasks):
            cache_key, value, cached = await task
            results[cache_key] = value
            if cached:
                hit_count += 1
            else:
                miss_count += 1
            progress_bar.update(1)
            completed = hit_count + miss_count
            if completed % progress_interval == 0 or completed == len(tasks):
                debug_print(
                    f"{progress_desc}: completed={completed}/{len(tasks)} cache_hits={hit_count} api_calls={miss_count}"
                )
    finally:
        progress_bar.close()

    return results, CacheStats(hit_count=hit_count, miss_count=miss_count)


def cluster_centroid_indices(
    embedding_matrix: np.ndarray, indices: np.ndarray, sample_limit: int, seed: int
) -> list[int]:
    """Pick deterministic centroid-near and random examples from a cluster.

    Args:
        embedding_matrix: Full embedding matrix.
        indices: Section indices for one cluster.
        sample_limit: Maximum samples.
        seed: Random seed.

    Returns:
        Selected section indices.
    """

    if indices.size <= sample_limit:
        return indices.tolist()
    rng = random.Random(seed + int(indices[0]))
    cluster_vectors = embedding_matrix[indices]
    centroid = cluster_vectors.mean(axis=0)
    distances = np.linalg.norm(cluster_vectors - centroid, axis=1)
    ordered = np.argsort(distances)

    nearest_count = min(10, sample_limit, indices.size)
    selected = [int(indices[i]) for i in ordered[:nearest_count]]
    remaining_capacity = sample_limit - len(selected)
    if remaining_capacity <= 0:
        return selected

    remaining_pool = [int(indices[i]) for i in ordered[nearest_count:]]
    if remaining_pool:
        sampled = rng.sample(
            remaining_pool, k=min(remaining_capacity, len(remaining_pool))
        )
        selected.extend(sampled)
    return selected


def build_cluster_naming_prompt(cluster_id: int, samples: list[str]) -> str:
    """Build cluster naming prompt using representative examples.

    Args:
        cluster_id: Cluster id.
        samples: Steering text samples.

    Returns:
        Prompt text requesting strict JSON output.
    """

    numbered = "\n".join(f"{index + 1}. {text}" for index, text in enumerate(samples))
    return (
        "You are naming a cluster of steering strings.\\n"
        "Focus on the high-level cluster purpose, not low-level wording.\\n"
        "Return strict JSON only with keys: title, purpose, keywords.\\n"
        "keywords must be an array of 3-7 short phrases.\\n\\n"
        f"Cluster ID: {cluster_id}\\n"
        "Representative steering strings:\\n"
        f"{numbered}"
    )


def normalize_cluster_naming_payload(payload: Any, cluster_id: int) -> dict[str, Any]:
    """Normalize naming payload into title/purpose/keywords shape.

    Args:
        payload: Parsed payload from cache or model output.
        cluster_id: Cluster id used for fallback title.

    Returns:
        Normalized naming payload.
    """

    payload_dict = payload if isinstance(payload, dict) else {}
    title = str(payload_dict.get("title", "")).strip() or f"Cluster {cluster_id}"
    purpose = str(payload_dict.get("purpose", "")).strip()
    keywords = payload_dict.get("keywords", [])
    keyword_list = (
        [str(item).strip() for item in keywords] if isinstance(keywords, list) else []
    )
    return {"title": title, "purpose": purpose, "keywords": keyword_list}


def ensure_output_dirs(base_dir: Path) -> None:
    """Create required output directories.

    Args:
        base_dir: Cluster analysis output directory.
    """

    (base_dir / "cache").mkdir(parents=True, exist_ok=True)
    (base_dir / "plots").mkdir(parents=True, exist_ok=True)


def sample_tsne_indices(labels: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Stratified sample indices for t-SNE plotting.

    Args:
        labels: Cluster labels.
        sample_size: Target sample size.
        seed: Random seed.

    Returns:
        Selected indices.
    """

    total = labels.shape[0]
    if sample_size >= total:
        return np.arange(total)

    rng = np.random.default_rng(seed)
    unique_labels, counts = np.unique(labels, return_counts=True)
    allocations: dict[int, int] = {}
    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        raw = int(round(sample_size * count / total))
        allocations[int(label)] = max(1, min(count, raw))

    allocated = sum(allocations.values())
    label_cycle = [int(label) for label in unique_labels.tolist()]
    cycle_index = 0
    while allocated > sample_size:
        label = label_cycle[cycle_index % len(label_cycle)]
        if allocations[label] > 1:
            allocations[label] -= 1
            allocated -= 1
        cycle_index += 1
    cycle_index = 0
    while allocated < sample_size:
        label = label_cycle[cycle_index % len(label_cycle)]
        label_count = int((labels == label).sum())
        if allocations[label] < label_count:
            allocations[label] += 1
            allocated += 1
        cycle_index += 1

    selected: list[int] = []
    for label in unique_labels.tolist():
        label_int = int(label)
        label_indices = np.where(labels == label_int)[0]
        take_count = allocations[label_int]
        sampled = rng.choice(label_indices, size=take_count, replace=False)
        selected.extend(sampled.tolist())
    return np.asarray(sorted(selected), dtype=np.int64)


def render_tsne_plot(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    title: str,
    sample_size: int,
    seed: int,
) -> None:
    """Render and save a t-SNE scatter plot.

    Args:
        embeddings: Full embedding matrix.
        labels: Cluster labels.
        output_path: PNG destination.
        title: Plot title.
        sample_size: Number of points to draw.
        seed: Random seed.
    """

    total_start_time = time.perf_counter()
    sampled_indices = sample_tsne_indices(
        labels=labels, sample_size=sample_size, seed=seed
    )
    sampled_embeddings = embeddings[sampled_indices]
    sampled_labels = labels[sampled_indices]
    perplexity = min(30, max(5, sampled_embeddings.shape[0] // 50))
    debug_print(
        f"t-SNE start title={title} total_points={embeddings.shape[0]} "
        f"sampled_points={sampled_embeddings.shape[0]} perplexity={perplexity}"
    )
    tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca")
    fit_start_time = time.perf_counter()
    coords = tsne.fit_transform(sampled_embeddings)
    debug_print(
        f"t-SNE fit done title={title} seconds={time.perf_counter() - fit_start_time:.3f}"
    )

    plt.figure(figsize=(9, 6))
    unique_labels = np.unique(sampled_labels)
    for label in unique_labels:
        mask = sampled_labels == label
        legend_label = "noise" if int(label) == -1 else f"cluster {int(label)}"
        plt.scatter(
            coords[mask, 0], coords[mask, 1], s=8, alpha=0.7, label=legend_label
        )
    plt.title(title)
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    if unique_labels.shape[0] <= 25:
        plt.legend(loc="best", fontsize=7)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close()
    debug_print(
        f"t-SNE plot saved path={output_path} seconds={time.perf_counter() - total_start_time:.3f}"
    )


def render_histogram(
    series: Any,
    output_path: Path,
    title: str,
    xlabel: str,
    bins: int = 60,
    integer_bins: bool = False,
    annotate_percentiles: list[float] | None = None,
    annotation_colormap: str = "viridis",
    log_y: bool = False,
) -> None:
    """Render a single-series histogram plot.

    Args:
        series: Numeric series-like values.
        output_path: PNG path.
        title: Plot title.
        xlabel: X-axis label.
        bins: Histogram bins.
        integer_bins: Use integer-centered bin edges when True.
        annotate_percentiles: Optional percentiles in (0, 1) to annotate with vertical lines.
        annotation_colormap: Matplotlib colormap name used for percentile contour lines.
        log_y: Use log scale on the y-axis when True.
    """

    values = (
        to_numeric_series(values=series, label=f"hist_{title}").dropna().astype(float)
    )
    plt.figure(figsize=(8, 5))
    histogram_bins: Any = bins
    if integer_bins and not values.empty:
        minimum_value = int(np.floor(values.min()))
        maximum_value = int(np.ceil(values.max()))
        histogram_bins = np.arange(minimum_value - 0.5, maximum_value + 1.5, 1.0)
    plt.hist(
        values,
        bins=histogram_bins,
        alpha=0.85,
        color=LIGHT_RED_BAR_COLOR,
        edgecolor="#b86a6a",
    )
    if annotate_percentiles and not values.empty:
        axis = plt.gca()
        y_min, y_max = axis.get_ylim()
        y_base = y_max * 0.93 if y_max > 0 else 1.0
        y_step = y_max * 0.08 if y_max > 0 else 0.1
        x_min, x_max = axis.get_xlim()
        x_grid = np.linspace(x_min, x_max, 200)
        if log_y:
            safe_y_min = max(float(y_min), 1e-6)
            y_grid = np.geomspace(safe_y_min, max(float(y_max), safe_y_min * 10.0), 200)
        else:
            y_grid = np.linspace(float(y_min), float(y_max), 200)
        mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
        contour_values = mesh_x
        sorted_percentiles = sorted(annotate_percentiles)
        cmap = plt.get_cmap(annotation_colormap)
        for index, percentile in enumerate(sorted_percentiles):
            assert (
                0.0 < percentile < 1.0
            ), "annotate_percentiles values must be in (0, 1)"
            percentile_value = float(values.quantile(percentile))
            percentile_percent = percentile * 100.0
            if np.isclose(percentile_percent, round(percentile_percent)):
                percentile_label = str(int(round(percentile_percent)))
            else:
                percentile_label = f"{percentile_percent:.1f}".rstrip("0").rstrip(".")
            color_position = (
                0.0
                if len(sorted_percentiles) == 1
                else index / (len(sorted_percentiles) - 1)
            )
            contour_color = cmap(color_position)
            y_annotate = max(
                y_base - (index * y_step),
                y_max * 0.35 if y_max > 0 else 0.2,
            )
            contour_set = axis.contour(
                mesh_x,
                mesh_y,
                contour_values,
                levels=[percentile_value],
                colors=[contour_color],
                linestyles="--",
                linewidths=1.5,
            )
            axis.clabel(
                contour_set,
                levels=[percentile_value],
                fmt={percentile_value: f"p{percentile_label}"},
                inline=True,
                inline_spacing=8,
                fontsize=8,
                colors=[contour_color],
                manual=[(percentile_value, y_annotate)],
            )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    if log_y:
        plt.yscale("log")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close()


def render_overlay_histogram(
    left: Any,
    right: Any,
    output_path: Path,
    title: str,
    left_label: str,
    right_label: str,
    log_y: bool = False,
) -> None:
    """Render two-series overlay histogram.

    Args:
        left: First numeric series-like values.
        right: Second numeric series-like values.
        output_path: PNG path.
        title: Plot title.
        left_label: Legend label for first series.
        right_label: Legend label for second series.
        log_y: Use log scale on the y-axis when True.
    """

    left_values = (
        to_numeric_series(values=left, label=f"overlay_left_{title}")
        .dropna()
        .astype(float)
    )
    right_values = (
        to_numeric_series(values=right, label=f"overlay_right_{title}")
        .dropna()
        .astype(float)
    )
    plt.figure(figsize=(8, 5))
    plt.hist(
        left_values,
        bins=70,
        alpha=0.6,
        label=left_label,
        color=LIGHT_RED_BAR_COLOR_ALT,
    )
    plt.hist(
        right_values,
        bins=70,
        alpha=0.6,
        label=right_label,
        color=LIGHT_RED_BAR_COLOR,
    )
    plt.title(title)
    plt.xlabel("tokens")
    plt.ylabel("count")
    if log_y:
        plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close()


def format_dataset_source_label(source_name: str) -> str:
    """Convert raw dataset source keys into human-readable labels.

    Args:
        source_name: Raw source key from ``dataset_source``.

    Returns:
        Readable dataset source display name.

    Example:
        >>> format_dataset_source_label(
        ...     "saumyamalik/OpenThoughts3-full-filtered-science-decontam-v2"
        ... )
        'OpenThoughts 3 Science (Saumya Malik)'
    """

    trimmed_source = source_name.strip()
    if not trimmed_source:
        return "Unknown"
    alias_label = DATASET_SOURCE_LABEL_ALIASES.get(trimmed_source.lower())
    if alias_label is not None:
        return alias_label
    owner_part, separator, repo_part = trimmed_source.partition("/")
    owner_label = DATASET_OWNER_LABEL_ALIASES.get(
        owner_part.strip().lower(), owner_part.strip()
    )
    source_key_for_tokens = repo_part if separator else trimmed_source
    source_tokens = re.split(r"[\s_\-/]+", trimmed_source)
    if separator:
        source_tokens = re.split(r"[\s_\-/]+", source_key_for_tokens)
    uppercase_tokens = {
        "gsm8k",
        "mmlu",
        "gpqa",
        "aime",
        "arc",
        "bbh",
        "agi",
        "cot",
    }
    pretty_tokens: list[str] = []
    for source_token in source_tokens:
        if not source_token:
            continue
        normalized_token = source_token.strip()
        lower_token = normalized_token.lower()
        if lower_token in uppercase_tokens:
            pretty_tokens.append(lower_token.upper())
            continue
        if normalized_token.isupper() or normalized_token.isdigit():
            pretty_tokens.append(normalized_token)
            continue
        pretty_tokens.append(lower_token.capitalize())
    if not pretty_tokens:
        return "Unknown"
    repo_label = " ".join(pretty_tokens)
    if separator and owner_label:
        return f"{repo_label} ({owner_label})"
    return repo_label


def render_scatter_plot(
    x: Any,
    y: Any,
    output_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    point_size: float = 10.0,
    alpha: float = 0.35,
    group_labels: Any | None = None,
    group_label_name: str = "group",
    show_group_fit_lines: bool = False,
    fit_line_width: float = 1.8,
    group_display_formatter: Callable[[str], str] | None = None,
) -> None:
    """Render a scatter plot from two numeric series.

    Args:
        x: X-axis values.
        y: Y-axis values.
        output_path: PNG output path.
        title: Plot title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        point_size: Marker size.
        alpha: Marker alpha value.
        group_labels: Optional categorical labels used to color points.
        group_label_name: Legend title when ``group_labels`` is provided.
        show_group_fit_lines: Draw per-group linear best-fit lines when True.
        fit_line_width: Line width for per-group best-fit lines.
        group_display_formatter: Optional function to format group legend labels.

    Returns:
        None: Saves the plot to ``output_path``.

    Example:
        >>> render_scatter_plot(
        ...     x=[1, 2, 3],
        ...     y=[10, 20, 15],
        ...     output_path=Path("scatter.png"),
        ...     title="Example",
        ...     xlabel="x",
        ...     ylabel="y",
        ...     group_labels=["source_a", "source_b", "source_a"],
        ...     group_label_name="dataset source",
        ...     show_group_fit_lines=True,
        ...     group_display_formatter=format_dataset_source_label,
        ... )
    """

    x_values = to_numeric_series(values=x, label=f"scatter_x_{title}").astype(float)
    y_values = to_numeric_series(values=y, label=f"scatter_y_{title}").astype(float)
    scatter_df = pd.DataFrame({"x": x_values, "y": y_values})
    if group_labels is not None:
        group_series = pd.Series(group_labels)
        assert len(group_series) == len(scatter_df), (
            "group_labels length must match x/y length for scatter plot."
        )
        normalized_groups = (
            group_series.astype("string").fillna("unknown").str.strip()
        )
        normalized_groups = normalized_groups.mask(normalized_groups == "", "unknown")
        scatter_df["group"] = normalized_groups.astype(str)
    scatter_df = scatter_df.dropna(subset=["x", "y"])

    plt.figure(figsize=(8, 5))
    if group_labels is None:
        plt.scatter(
            scatter_df["x"],
            scatter_df["y"],
            s=point_size,
            alpha=alpha,
            color="#b94747",
            edgecolors="none",
        )
    else:
        unique_groups = sorted(scatter_df["group"].astype(str).unique().tolist())
        colormap_name = "tab20" if len(unique_groups) <= 20 else "gist_ncar"
        colormap = plt.get_cmap(colormap_name, max(len(unique_groups), 1))
        display_label_by_group: dict[str, str] = {}
        for group_name in unique_groups:
            formatted_name = (
                group_display_formatter(group_name)
                if group_display_formatter is not None
                else group_name
            )
            display_label_by_group[group_name] = formatted_name.strip() or group_name
        display_label_counts: dict[str, int] = {}
        for display_label in display_label_by_group.values():
            display_label_counts[display_label] = (
                display_label_counts.get(display_label, 0) + 1
            )
        for group_index, group_name in enumerate(unique_groups):
            group_mask = scatter_df["group"] == group_name
            group_df = scatter_df.loc[group_mask]
            group_color = colormap(group_index)
            display_group_name = display_label_by_group[group_name]
            if display_label_counts.get(display_group_name, 0) > 1:
                display_group_name = f"{display_group_name} ({group_name})"
            plt.scatter(
                group_df["x"],
                group_df["y"],
                s=point_size,
                alpha=alpha,
                color=group_color,
                edgecolors="none",
                label=display_group_name,
            )
            if show_group_fit_lines:
                unique_x_values = group_df["x"].nunique()
                if len(group_df) >= 2 and unique_x_values >= 2:
                    fit_slope, fit_intercept = np.polyfit(
                        group_df["x"].to_numpy(),
                        group_df["y"].to_numpy(),
                        deg=1,
                    )
                    fit_x_values = np.linspace(
                        group_df["x"].min(),
                        group_df["x"].max(),
                        num=100,
                    )
                    fit_y_values = fit_slope * fit_x_values + fit_intercept
                    plt.plot(
                        fit_x_values,
                        fit_y_values,
                        color=group_color,
                        linewidth=fit_line_width,
                        alpha=min(alpha + 0.35, 1.0),
                    )
        if unique_groups:
            plt.legend(title=group_label_name, loc="best", fontsize=7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close()


def render_cluster_size_plot(
    cluster_counts: Any,
    cluster_names: dict[int, str],
    output_path: Path,
    top_n: int = 5,
    middle_n: int = 5,
    bottom_n: int = 5,
    log_y: bool = False,
) -> None:
    """Render top/middle/bottom cluster-size bar chart using cluster names.

    Args:
        cluster_counts: Cluster-size series indexed by label.
        cluster_names: Mapping from cluster id to cluster display name.
        output_path: PNG path.
        top_n: Number of largest clusters to include.
        middle_n: Number of middle-ranked clusters to include.
        bottom_n: Number of smallest clusters to include.
        log_y: Use log scale on the y-axis when True.
    """

    numeric_counts = to_numeric_series(values=cluster_counts, label="cluster_counts")
    sorted_counts = numeric_counts.sort_values(ascending=False)
    cluster_ids = sorted_counts.index.tolist()
    total_clusters = len(cluster_ids)
    top_only_mode = middle_n <= 0 and bottom_n <= 0

    top_ids = cluster_ids[: min(top_n, total_clusters)]
    middle_start = max(0, (total_clusters // 2) - (middle_n // 2))
    middle_end = min(total_clusters, middle_start + middle_n)
    middle_ids = cluster_ids[middle_start:middle_end]
    bottom_ids = cluster_ids[max(0, total_clusters - bottom_n) :]

    selected_ids: list[Any] = []
    for cluster_id in [*top_ids, *middle_ids, *bottom_ids]:
        if cluster_id not in selected_ids:
            selected_ids.append(cluster_id)

    selected_counts = sorted_counts.loc[selected_ids]
    group_by_id: dict[Any, str] = {}
    for cluster_id in top_ids:
        group_by_id[cluster_id] = "Top"
    for cluster_id in middle_ids:
        group_by_id[cluster_id] = "Middle"
    for cluster_id in bottom_ids:
        group_by_id[cluster_id] = "Bottom"

    bar_labels: list[str] = []
    label_counts: dict[str, int] = {}
    for cluster_id in selected_counts.index.tolist():
        cluster_id_int = coerce_cluster_label(cluster_id)
        cluster_name = cluster_names.get(cluster_id_int, f"Cluster {cluster_id_int}")
        group_name = group_by_id.get(cluster_id, "Selected")
        base_label = (
            cluster_name.strip()
            if top_only_mode
            else f"{group_name}: {cluster_name.strip()}"
        )
        count = label_counts.get(base_label, 0) + 1
        label_counts[base_label] = count
        bar_labels.append(base_label if count == 1 else f"{base_label} ({count})")

    bar_values = [float(value) for value in selected_counts.tolist()]
    plt.figure(figsize=(10, 6))
    plt.bar(bar_labels, bar_values, color=LIGHT_RED_BAR_COLOR, edgecolor="#b86a6a")
    plot_title = (
        f"Final cluster sizes (Top {len(top_ids)})"
        if top_only_mode
        else "Final cluster sizes (Top/Middle/Bottom)"
    )
    plt.title(plot_title)
    plt.xlabel("cluster name")
    plt.ylabel("sections")
    if log_y:
        plt.yscale("log")
    plt.xticks(rotation=70, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=DEFAULT_PLOT_DPI)
    plt.close()


def stage_extract(config: RunConfig) -> dict[str, Any]:
    """Run extract stage and persist sections and think rows.

    Args:
        config: Runtime config.

    Returns:
        Stage metadata dictionary.
    """

    transformed_rows = load_jsonl(path=config.transformed_path)
    og_rows = load_jsonl(path=config.og_path)
    og_by_id = {row["id"]: row for row in og_rows if isinstance(row.get("id"), str)}

    section_records: list[SteeringRecord] = []
    think_rows: list[dict[str, Any]] = []
    anomalies: list[dict[str, Any]] = []

    for row_index, transformed_row in enumerate(transformed_rows):
        row_id = transformed_row.get("id")
        if not isinstance(row_id, str):
            continue
        og_row = og_by_id.get(row_id)
        if og_row is None:
            anomalies.append(
                {
                    "row_id": row_id,
                    "dataset_source": transformed_row.get("dataset_source", ""),
                    "anomaly": "missing_og_row",
                }
            )
            continue

        source_name = str(transformed_row.get("dataset_source", ""))
        transformed_messages = transformed_row.get("messages", [])
        og_messages = og_row.get("messages", [])
        if not isinstance(transformed_messages, list) or not isinstance(
            og_messages, list
        ):
            anomalies.append(
                {
                    "row_id": row_id,
                    "dataset_source": source_name,
                    "anomaly": "invalid_messages",
                }
            )
            continue

        transformed_content = extract_first_assistant_content(
            messages=transformed_messages
        )
        og_content = extract_first_assistant_content(messages=og_messages)
        if transformed_content is None or og_content is None:
            anomalies.append(
                {
                    "row_id": row_id,
                    "dataset_source": source_name,
                    "anomaly": "missing_assistant_content",
                }
            )
            continue

        new_think, new_anomalies = extract_single_think_block(
            content=transformed_content, row_id=row_id, source_name=source_name
        )
        og_think, og_anomalies = extract_single_think_block(
            content=og_content, row_id=row_id, source_name=source_name
        )
        anomalies.extend(new_anomalies)
        anomalies.extend(og_anomalies)
        if new_think is None or og_think is None:
            continue

        pairs, parse_meta = parse_steering_execution_pairs(think_text=new_think)
        if (
            parse_meta["unmatched_steering"]
            or parse_meta["unmatched_execution"]
            or parse_meta["paired_count"] == 0
        ):
            anomaly_payload = {
                "row_id": row_id,
                "dataset_source": source_name,
                "anomaly": "unmatched_sections",
                **parse_meta,
            }
            anomalies.append(anomaly_payload)

        think_rows.append(
            {
                "row_id": row_id,
                "dataset_source": source_name,
                "original_think_text": og_think,
                "new_think_text": new_think,
                "source_batch": (
                    str(
                        (transformed_row.get("transform_meta") or {}).get(
                            "source_batch", "unknown"
                        )
                    )
                    if isinstance(transformed_row.get("transform_meta"), dict)
                    else "unknown"
                ),
            }
        )

        for pair_index, (steering_text, execution_text) in enumerate(pairs):
            section_id = f"{row_id}::sec{pair_index}"
            section_records.append(
                SteeringRecord(
                    section_id=section_id,
                    row_id=row_id,
                    dataset_source=source_name,
                    steering_text=steering_text,
                    execution_text=execution_text,
                    original_think_text=og_think,
                    new_think_text=new_think,
                )
            )

    sections_df = pd.DataFrame([asdict(record) for record in section_records])
    think_df = pd.DataFrame(think_rows).drop_duplicates(subset=["row_id"])

    sections_parquet_path = config.output_dir / "sections.parquet"
    sections_csv_path = config.output_dir / "sections.csv"
    think_rows_path = config.output_dir / "think_rows.parquet"
    anomalies_path = config.output_dir / "parse_anomalies.jsonl"

    sections_df.to_parquet(sections_parquet_path, index=False)
    sections_df.to_csv(sections_csv_path, index=False)
    think_df.to_parquet(think_rows_path, index=False)
    write_jsonl(path=anomalies_path, rows=anomalies)

    return {
        "rows_in": len(transformed_rows),
        "rows_with_think": int(think_df.shape[0]),
        "sections_emitted": int(sections_df.shape[0]),
        "anomalies": len(anomalies),
        "sections_parquet": str(sections_parquet_path),
        "think_rows_parquet": str(think_rows_path),
        "anomalies_path": str(anomalies_path),
    }


async def stage_tokens_async(config: RunConfig, client: Any) -> dict[str, Any]:
    """Run token counting stage with async Gemini count_tokens calls.

    Args:
        config: Runtime config.
        client: Gemini-compatible client.

    Returns:
        Stage metadata.
    """

    debug_print("tokens stage: loading sections.parquet and think_rows.parquet")
    sections_df = pd.read_parquet(config.output_dir / "sections.parquet")
    think_df = pd.read_parquet(config.output_dir / "think_rows.parquet")
    debug_print(
        f"tokens stage: sections={sections_df.shape[0]} think_rows={think_df.shape[0]}"
    )

    retry_config = RetryConfig(
        timeout_seconds=config.api_timeout_seconds, max_retries=config.max_retries
    )
    token_cache = JsonlCache(
        cache_path=config.output_dir / "cache" / "token_counts.jsonl"
    )

    sections_df["steering_key"] = sections_df["steering_text"].apply(
        lambda text: hash_text("steering", config.naming_model, text)
    )
    sections_df["execution_key"] = sections_df["execution_text"].apply(
        lambda text: hash_text("execution", config.naming_model, text)
    )
    think_df["og_key"] = think_df["original_think_text"].apply(
        lambda text: hash_text("og_think", config.naming_model, text)
    )
    think_df["new_key"] = think_df["new_think_text"].apply(
        lambda text: hash_text("new_think", config.naming_model, text)
    )

    payloads: dict[str, str] = {}
    for key, value in zip(sections_df["steering_key"], sections_df["steering_text"]):
        payloads.setdefault(str(key), str(value))
    for key, value in zip(sections_df["execution_key"], sections_df["execution_text"]):
        payloads.setdefault(str(key), str(value))
    for key, value in zip(think_df["og_key"], think_df["original_think_text"]):
        payloads.setdefault(str(key), str(value))
    for key, value in zip(think_df["new_key"], think_df["new_think_text"]):
        payloads.setdefault(str(key), str(value))
    debug_print(
        f"tokens stage: unique_token_requests={len(payloads)} token_concurrency={config.token_concurrency} "
        f"token_requests_per_minute={config.token_requests_per_minute}"
    )

    async def _count_tokens(text: str) -> int:
        response = await client.aio.models.count_tokens(
            model=config.naming_model, contents=text
        )
        return int(response.total_tokens or 0)

    token_values, cache_stats = await run_async_text_jobs(
        job_payloads=payloads,
        cache=token_cache,
        concurrency=config.token_concurrency,
        async_fn=_count_tokens,
        retry_config=retry_config,
        requests_per_minute=config.token_requests_per_minute,
        progress_desc="tokens",
    )

    sections_df["steering_tokens"] = map_cached_numeric_values(
        keys=cast(pd.Series, sections_df["steering_key"]),
        cache_values=token_values,
        label="steering_tokens",
    )
    sections_df["execution_tokens"] = map_cached_numeric_values(
        keys=cast(pd.Series, sections_df["execution_key"]),
        cache_values=token_values,
        label="execution_tokens",
    )
    think_df["original_think_tokens"] = map_cached_numeric_values(
        keys=cast(pd.Series, think_df["og_key"]),
        cache_values=token_values,
        label="original_think_tokens",
    )
    think_df["new_think_tokens"] = map_cached_numeric_values(
        keys=cast(pd.Series, think_df["new_key"]),
        cache_values=token_values,
        label="new_think_tokens",
    )
    think_df["think_token_delta"] = (
        think_df["new_think_tokens"] - think_df["original_think_tokens"]
    )

    token_stats_df = sections_df[
        ["section_id", "row_id", "steering_tokens", "execution_tokens"]
    ].merge(
        think_df[
            ["row_id", "original_think_tokens", "new_think_tokens", "think_token_delta"]
        ],
        on="row_id",
        how="left",
    )

    token_stats_path = config.output_dir / "token_stats.parquet"
    think_token_stats_path = config.output_dir / "think_token_stats.parquet"
    summary_path = config.output_dir / "token_summary.json"

    token_stats_df.to_parquet(token_stats_path, index=False)
    think_df.to_parquet(think_token_stats_path, index=False)

    summary = {
        "rows": int(think_df.shape[0]),
        "sections": int(token_stats_df.shape[0]),
        "steering_tokens": summarize_series(token_stats_df["steering_tokens"]),
        "execution_tokens": summarize_series(token_stats_df["execution_tokens"]),
        "original_think_tokens": summarize_series(think_df["original_think_tokens"]),
        "new_think_tokens": summarize_series(think_df["new_think_tokens"]),
        "think_token_delta": summarize_series(think_df["think_token_delta"]),
        "cache": asdict(cache_stats),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "token_stats_path": str(token_stats_path),
        "think_token_stats_path": str(think_token_stats_path),
        "summary_path": str(summary_path),
        "cache": asdict(cache_stats),
        "token_requests": len(payloads),
    }


def split_embedding_batches(
    items: list[tuple[str, str]], batch_size: int
) -> list[list[tuple[str, str]]]:
    """Split embedding payloads into API-sized chunks.

    Args:
        items: Ordered `(cache_key, text)` payload list.
        batch_size: Desired batch size.

    Returns:
        Chunked payload batches.
    """

    effective_batch_size = max(1, min(batch_size, 100))
    return [
        items[index : index + effective_batch_size]
        for index in range(0, len(items), effective_batch_size)
    ]


def parse_embedding_batch_response(
    response: Any, batch_items: list[tuple[str, str]]
) -> dict[str, list[float]]:
    """Parse one embed response while preserving input order.

    Args:
        response: Embed API response object.
        batch_items: Ordered `(cache_key, text)` request payloads.

    Returns:
        Embedding vectors keyed by cache key.
    """

    embeddings = response.embeddings or []
    assert len(embeddings) == len(
        batch_items
    ), "Embedding response length does not match input batch length"
    vectors_by_key: dict[str, list[float]] = {}
    for item_index, (cache_key, _) in enumerate(batch_items):
        values = embeddings[item_index].values or []
        assert values, f"Embedding vector is empty for key {cache_key}"
        vectors_by_key[cache_key] = [float(value) for value in values]
    return vectors_by_key


def build_unique_text_index(texts: list[str]) -> tuple[list[str], np.ndarray]:
    """Build ordered unique text list and row-to-unique index map.

    Args:
        texts: Row-aligned text inputs.

    Returns:
        Tuple of unique text list and `int64` array mapping each row to a unique-text index.

    Example:
        >>> unique_texts, row_index = build_unique_text_index(texts=["A", "B", "A"])
        >>> unique_texts
        ['A', 'B']
        >>> row_index.tolist()
        [0, 1, 0]
    """

    unique_texts: list[str] = []
    text_to_unique_index: dict[str, int] = {}
    row_to_unique_index: list[int] = []
    for text in texts:
        text_value = str(text)
        unique_index = text_to_unique_index.get(text_value)
        if unique_index is None:
            unique_index = len(unique_texts)
            text_to_unique_index[text_value] = unique_index
            unique_texts.append(text_value)
        row_to_unique_index.append(unique_index)
    return unique_texts, np.asarray(row_to_unique_index, dtype=np.int64)


async def embed_texts_async(
    texts: list[str],
    model_name: str,
    cache_path: Path,
    concurrency: int,
    batch_size: int,
    requests_per_minute: int,
    client: Any,
    retry_config: RetryConfig,
    progress_desc: str,
) -> tuple[np.ndarray, CacheStats]:
    """Embed text list with caching and async Gemini embed_content.

    Args:
        texts: Input text list.
        model_name: Embedding model.
        cache_path: Embedding cache path.
        concurrency: Async concurrency.
        batch_size: Number of texts sent per embedding call.
        requests_per_minute: Rolling one-minute request cap.
        client: Gemini-compatible client.
        retry_config: Retry settings.
        progress_desc: Progress bar description label.

    Returns:
        Tuple of embedding matrix and cache stats.
    """

    debug_print(
        f"{progress_desc}: preparing embeddings rows={len(texts)} cache={cache_path} "
        f"batch_size={batch_size} concurrency={concurrency} requests_per_minute={requests_per_minute}"
    )
    cache = JsonlCache(cache_path=cache_path)
    key_to_text: dict[str, str] = {}
    row_keys: list[str] = []
    for text in texts:
        cache_key = hash_text(kind="embedding", model=model_name, text=text)
        row_keys.append(cache_key)
        key_to_text.setdefault(cache_key, text)

    embedding_values: dict[str, list[float]] = {}
    missing_items: list[tuple[str, str]] = []
    for cache_key, text in key_to_text.items():
        cached_value = cache.get(key=cache_key)
        if isinstance(cached_value, list) and cached_value:
            embedding_values[cache_key] = [float(value) for value in cached_value]
        else:
            missing_items.append((cache_key, text))

    hit_count = len(embedding_values)
    miss_count = 0
    progress_bar = tqdm(
        total=len(key_to_text), desc=progress_desc, unit="text", dynamic_ncols=True
    )
    if hit_count > 0:
        progress_bar.update(hit_count)

    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = build_rate_limiter(requests_per_minute=requests_per_minute)
    batches = split_embedding_batches(items=missing_items, batch_size=batch_size)
    debug_print(
        f"{progress_desc}: unique_texts={len(key_to_text)} cache_hits={hit_count} "
        f"cache_misses={len(missing_items)} batches={len(batches)}"
    )

    async def _embed_batch(
        batch_index: int, batch_items: list[tuple[str, str]]
    ) -> dict[str, list[float]]:
        batch_start_time = time.perf_counter()
        debug_print(
            f"{progress_desc}: batch_start index={batch_index + 1}/{len(batches)} size={len(batch_items)}"
        )
        async with semaphore:
            if rate_limiter is not None:
                await rate_limiter.acquire()

            batch_texts = [text for _, text in batch_items]

            async def _call() -> Any:
                return await client.aio.models.embed_content(
                    model=model_name, contents=batch_texts
                )

            response = await call_with_retry(
                call_factory=_call,
                retry_config=retry_config,
                request_label=f"embed_batch:{batch_items[0][0]}:{len(batch_items)}",
            )

        vectors_by_key = parse_embedding_batch_response(
            response=response, batch_items=batch_items
        )
        for cache_key, vector in vectors_by_key.items():
            await cache.set(key=cache_key, value=vector)
        debug_print(
            f"{progress_desc}: batch_done index={batch_index + 1}/{len(batches)} size={len(batch_items)} "
            f"seconds={time.perf_counter() - batch_start_time:.3f}"
        )
        return vectors_by_key

    tasks = [
        asyncio.create_task(_embed_batch(batch_index=batch_index, batch_items=batch))
        for batch_index, batch in enumerate(batches)
    ]
    try:
        for task in asyncio.as_completed(tasks):
            vectors_by_key = await task
            embedding_values.update(vectors_by_key)
            miss_count += len(vectors_by_key)
            progress_bar.update(len(vectors_by_key))
    finally:
        progress_bar.close()

    assert len(embedding_values) == len(key_to_text), "Not all embeddings were resolved"
    vectors = [embedding_values[key] for key in row_keys]
    matrix = np.asarray(vectors, dtype=np.float32)
    assert matrix.ndim == 2, "Embedding matrix must be 2D"
    debug_print(
        f"{progress_desc}: embedding matrix ready shape={matrix.shape} cache_hits={hit_count} api_embeddings={miss_count}"
    )
    return matrix, CacheStats(hit_count=hit_count, miss_count=miss_count)


async def cluster_texts_using_unique_strings(
    texts: list[str],
    model_name: str,
    cache_path: Path,
    concurrency: int,
    batch_size: int,
    requests_per_minute: int,
    min_cluster_size: int,
    max_cluster_size: int,
    client: Any,
    retry_config: RetryConfig,
    progress_desc: str,
) -> UniqueTextClusteringResult:
    """Cluster only unique strings and map results back to every original row.

    Args:
        texts: Row-aligned text inputs to cluster.
        model_name: Embedding model.
        cache_path: Embedding cache path.
        concurrency: Async concurrency for embedding requests.
        batch_size: Number of texts per embedding request.
        requests_per_minute: Rolling one-minute request cap.
        min_cluster_size: HDBSCAN minimum cluster size.
        max_cluster_size: HDBSCAN maximum cluster size.
        client: Gemini-compatible client.
        retry_config: Retry/timeout settings.
        progress_desc: Progress label for embedding requests.

    Returns:
        Row-aligned labels/embeddings plus unique-string clustering stats.
    """

    assert texts, "Cannot cluster empty text list"
    unique_texts, row_to_unique_index = build_unique_text_index(texts=texts)
    duplicate_row_count = len(texts) - len(unique_texts)
    debug_print(
        f"{progress_desc}: unique clustering rows={len(texts)} unique_texts={len(unique_texts)} "
        f"duplicate_rows={duplicate_row_count}"
    )

    unique_embeddings, cache_stats = await embed_texts_async(
        texts=unique_texts,
        model_name=model_name,
        cache_path=cache_path,
        concurrency=concurrency,
        batch_size=batch_size,
        requests_per_minute=requests_per_minute,
        client=client,
        retry_config=retry_config,
        progress_desc=progress_desc,
    )
    unique_labels = run_hdbscan(
        embedding_matrix=unique_embeddings,
        min_cluster_size=min_cluster_size,
        max_cluster_size=max_cluster_size,
    )

    row_labels = unique_labels[row_to_unique_index].astype(np.int64)
    row_embeddings = unique_embeddings[row_to_unique_index].astype(
        np.float32, copy=False
    )
    debug_print(
        f"{progress_desc}: mapped unique clusters back to rows total_rows={len(texts)} "
        f"unique_texts={len(unique_texts)}"
    )
    return UniqueTextClusteringResult(
        row_labels=row_labels,
        row_embeddings=row_embeddings,
        unique_text_count=len(unique_texts),
        duplicate_row_count=duplicate_row_count,
        cache_stats=cache_stats,
    )


def run_hdbscan(
    embedding_matrix: np.ndarray,
    min_cluster_size: int,
    max_cluster_size: int,
) -> np.ndarray:
    """Run HDBSCAN clustering.

    Args:
        embedding_matrix: Input embedding matrix.
        min_cluster_size: HDBSCAN min cluster size.
        max_cluster_size: HDBSCAN max cluster size.

    Returns:
        Cluster labels array with noise as -1.
    """

    debug_print(
        f"hdbscan start rows={embedding_matrix.shape[0]} dims={embedding_matrix.shape[1]} "
        f"min_cluster_size={min_cluster_size} max_cluster_size={max_cluster_size}"
    )
    start_time = time.perf_counter()
    clusterer: HDBSCAN = cast(
        HDBSCAN,
        cast(Any, HDBSCAN)(
            min_cluster_size=min_cluster_size,
            max_cluster_size=max_cluster_size,
            copy=False,
            n_jobs=-1,
        ),
    )
    labels = clusterer.fit_predict(embedding_matrix)
    unique_clusters = int(np.unique(labels).shape[0])
    noise_count = int((labels == -1).sum())
    debug_print(
        f"hdbscan done seconds={time.perf_counter() - start_time:.3f} "
        f"clusters_including_noise={unique_clusters} noise_count={noise_count}"
    )
    return labels.astype(int)


async def stage_cluster1_async(config: RunConfig, client: Any) -> dict[str, Any]:
    """Run first clustering stage using unique steering strings.

    Args:
        config: Runtime config.
        client: Gemini-compatible client.

    Returns:
        Stage metadata.
    """

    debug_print("cluster1 stage: loading sections.parquet")
    sections_df = pd.read_parquet(config.output_dir / "sections.parquet")
    steering_texts = sections_df["steering_text"].astype(str).tolist()
    debug_print(f"cluster1 stage: sections={len(steering_texts)}")
    retry_config = RetryConfig(
        timeout_seconds=config.api_timeout_seconds, max_retries=config.max_retries
    )

    clustering_result = await cluster_texts_using_unique_strings(
        texts=steering_texts,
        model_name=config.embedding_model,
        cache_path=config.output_dir / "cache" / "embeddings_pass1.jsonl",
        concurrency=config.embed_concurrency,
        batch_size=config.embed_batch_size,
        requests_per_minute=config.embed_requests_per_minute,
        min_cluster_size=config.min_cluster_size,
        max_cluster_size=config.max_cluster_size,
        client=client,
        retry_config=retry_config,
        progress_desc="embed pass1",
    )
    labels = clustering_result.row_labels
    embedding_matrix = clustering_result.row_embeddings
    sections_df["cluster_pass1"] = labels
    sections_df.to_parquet(config.output_dir / "clusters_pass1.parquet", index=False)
    np.save(config.output_dir / "embeddings_pass1.npy", embedding_matrix)

    render_tsne_plot(
        embeddings=embedding_matrix,
        labels=labels,
        output_path=config.output_dir / "plots" / "tsne_pass1.png",
        title="Pass1 t-SNE (steering embeddings)",
        sample_size=config.tsne_sample_size,
        seed=config.seed,
    )

    cluster_counts = (
        sections_df["cluster_pass1"].value_counts().sort_values(ascending=False)
    )
    noise_count = int((labels == -1).sum())
    return {
        "sections": int(sections_df.shape[0]),
        "unique_texts_clustered": clustering_result.unique_text_count,
        "duplicate_rows_mapped": clustering_result.duplicate_row_count,
        "clusters_including_noise": int(cluster_counts.shape[0]),
        "noise_count": noise_count,
        "cache": asdict(clustering_result.cache_stats),
        "clusters_path": str(config.output_dir / "clusters_pass1.parquet"),
    }


async def name_clusters_async(
    sections_df: pd.DataFrame,
    embedding_matrix: np.ndarray,
    cluster_column: str,
    model_name: str,
    concurrency: int,
    output_prompts_path: Path,
    output_responses_path: Path,
    requests_per_minute: int,
    progress_desc: str,
    client: Any,
    retry_config: RetryConfig,
    seed: int,
) -> tuple[dict[int, dict[str, Any]], CacheStats, int]:
    """Name clusters asynchronously with Gemini generate_content.

    Args:
        sections_df: Section dataframe with cluster labels.
        embedding_matrix: Embedding matrix aligned to sections rows.
        cluster_column: Cluster label column.
        model_name: Naming model.
        concurrency: Async concurrency.
        output_prompts_path: Prompt JSONL path.
        output_responses_path: Response JSONL path.
        requests_per_minute: Rolling one-minute request cap.
        progress_desc: Progress bar description label.
        client: Gemini-compatible client.
        retry_config: Retry settings.
        seed: Random seed.

    Returns:
        Tuple of cluster-name payload by cluster id, cache stats, and naming error count.
    """

    cluster_to_prompt: dict[int, str] = {}
    prompt_rows: list[dict[str, Any]] = []
    non_noise_df = sections_df[sections_df[cluster_column] != -1]
    debug_print(
        f"{progress_desc}: building prompts from non_noise_rows={non_noise_df.shape[0]} "
        f"total_rows={sections_df.shape[0]}"
    )
    for cluster_id, cluster_df in non_noise_df.groupby(cluster_column):
        cluster_id_int = coerce_cluster_label(cluster_id)
        cluster_indices = cluster_df.index.to_numpy(dtype=np.int64)
        selected_indices = cluster_centroid_indices(
            embedding_matrix=embedding_matrix,
            indices=cluster_indices,
            sample_limit=min(20, int(cluster_indices.shape[0])),
            seed=seed,
        )
        sample_texts = (
            sections_df.loc[selected_indices, "steering_text"].astype(str).tolist()
        )
        prompt_text = build_cluster_naming_prompt(
            cluster_id=cluster_id_int, samples=sample_texts
        )
        cluster_to_prompt[cluster_id_int] = prompt_text
        prompt_rows.append(
            {
                "cluster_id": cluster_id_int,
                "sample_count": len(sample_texts),
                "prompt": prompt_text,
            }
        )

    write_jsonl(path=output_prompts_path, rows=prompt_rows)
    naming_cache = JsonlCache(
        cache_path=output_responses_path.parent
        / f"cache_{output_responses_path.stem}.jsonl"
    )

    semaphore = asyncio.Semaphore(concurrency)
    rate_limiter = build_rate_limiter(requests_per_minute=requests_per_minute)
    hit_count = 0
    miss_count = 0
    error_count = 0

    async def _name_one(prompt_text: str) -> dict[str, Any]:
        config = types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "purpose": {"type": "string"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "purpose", "keywords"],
            },
        )
        response = await client.aio.models.generate_content(
            model=model_name, contents=prompt_text, config=config
        )
        parsed = extract_json_object(text=response.text or "")
        if parsed is None:
            parsed = {"title": "", "purpose": "", "keywords": []}
        return parsed

    async def _name_cluster(
        cluster_id: int, prompt_text: str
    ) -> tuple[int, dict[str, Any], bool, str | None]:
        cache_key = hash_text(kind="naming", model=model_name, text=prompt_text)
        cached_value = naming_cache.get(key=cache_key)
        if cached_value is not None:
            payload = normalize_cluster_naming_payload(
                payload=cached_value, cluster_id=cluster_id
            )
            return cluster_id, payload, True, None

        async with semaphore:
            if rate_limiter is not None:
                await rate_limiter.acquire()

            async def _call() -> Any:
                return await _name_one(prompt_text=prompt_text)

            try:
                raw_payload = await call_with_retry(
                    call_factory=_call,
                    retry_config=retry_config,
                    request_label=cache_key,
                )
            except Exception as exc:  # noqa: BLE001
                fallback_payload = {
                    "title": f"Cluster {cluster_id}",
                    "purpose": "",
                    "keywords": [],
                }
                return cluster_id, fallback_payload, False, str(exc)

            payload = normalize_cluster_naming_payload(
                payload=raw_payload, cluster_id=cluster_id
            )
            await naming_cache.set(key=cache_key, value=payload)
            return cluster_id, payload, False, None

    response_rows: list[dict[str, Any]] = []
    cluster_names: dict[int, dict[str, Any]] = {}
    tasks = [
        asyncio.create_task(
            _name_cluster(cluster_id=cluster_id, prompt_text=prompt_text)
        )
        for cluster_id, prompt_text in cluster_to_prompt.items()
    ]
    debug_print(
        f"{progress_desc}: naming_tasks={len(tasks)} concurrency={concurrency} "
        f"requests_per_minute={requests_per_minute}"
    )
    progress_bar = tqdm(
        total=len(tasks), desc=progress_desc, unit="cluster", dynamic_ncols=True
    )
    progress_interval = max(1, min(200, max(10, len(tasks) // 10)))
    try:
        for task in asyncio.as_completed(tasks):
            cluster_id, resolved_payload, cached, error_message = await task
            if cached:
                hit_count += 1
            else:
                miss_count += 1
            if error_message is not None:
                error_count += 1
            cluster_names[cluster_id] = resolved_payload
            response_rows.append(
                {
                    "cluster_id": cluster_id,
                    **resolved_payload,
                    "error": error_message,
                }
            )
            progress_bar.update(1)
            completed = hit_count + miss_count
            if completed % progress_interval == 0 or completed == len(tasks):
                debug_print(
                    f"{progress_desc}: completed={completed}/{len(tasks)} cache_hits={hit_count} "
                    f"api_calls={miss_count} errors={error_count}"
                )
    finally:
        progress_bar.close()

    write_jsonl(path=output_responses_path, rows=response_rows)
    return (
        cluster_names,
        CacheStats(hit_count=hit_count, miss_count=miss_count),
        error_count,
    )


async def stage_name1_async(config: RunConfig, client: Any) -> dict[str, Any]:
    """Run first naming stage for pass1 clusters.

    Args:
        config: Runtime config.
        client: Gemini-compatible client.

    Returns:
        Stage metadata.
    """

    clusters_df = pd.read_parquet(config.output_dir / "clusters_pass1.parquet")
    embeddings = np.load(config.output_dir / "embeddings_pass1.npy")
    retry_config = RetryConfig(
        timeout_seconds=config.api_timeout_seconds, max_retries=config.max_retries
    )

    cluster_names, cache_stats, naming_errors = await name_clusters_async(
        sections_df=clusters_df,
        embedding_matrix=embeddings,
        cluster_column="cluster_pass1",
        model_name=config.naming_model,
        concurrency=config.naming_concurrency,
        output_prompts_path=config.output_dir / "naming_prompts_pass1.jsonl",
        output_responses_path=config.output_dir / "naming_responses_pass1.jsonl",
        requests_per_minute=config.naming_requests_per_minute,
        progress_desc="name pass1",
        client=client,
        retry_config=retry_config,
        seed=config.seed,
    )

    clusters_df["cluster_name_pass1"] = clusters_df["cluster_pass1"].apply(
        lambda cluster_id: cluster_names.get(int(cluster_id), {}).get(
            "title",
            f"Noise {-1}" if int(cluster_id) == -1 else f"Cluster {int(cluster_id)}",
        )
    )
    clusters_df.to_parquet(config.output_dir / "clusters_pass1.parquet", index=False)

    return {
        "clusters_named": len(cluster_names),
        "naming_errors": naming_errors,
        "cache": asdict(cache_stats),
        "responses_path": str(config.output_dir / "naming_responses_pass1.jsonl"),
    }


async def stage_cluster2_async(config: RunConfig, client: Any) -> dict[str, Any]:
    """Run second clustering stage using unique title+steering strings.

    Args:
        config: Runtime config.
        client: Gemini-compatible client.

    Returns:
        Stage metadata.
    """

    debug_print("cluster2 stage: loading clusters_pass1.parquet")
    pass1_df = pd.read_parquet(config.output_dir / "clusters_pass1.parquet")
    if "cluster_name_pass1" not in pass1_df.columns:
        pass1_df["cluster_name_pass1"] = pass1_df["cluster_pass1"].apply(
            lambda cluster_id: f"Cluster {int(cluster_id)}"
        )

    recluster_texts = (
        pass1_df["cluster_name_pass1"].astype(str).fillna("Cluster")
        + " :: "
        + pass1_df["steering_text"].astype(str)
    ).tolist()

    debug_print(f"cluster2 stage: recluster rows={len(recluster_texts)}")
    retry_config = RetryConfig(
        timeout_seconds=config.api_timeout_seconds, max_retries=config.max_retries
    )
    clustering_result = await cluster_texts_using_unique_strings(
        texts=recluster_texts,
        model_name=config.embedding_model,
        cache_path=config.output_dir / "cache" / "embeddings_pass2.jsonl",
        concurrency=config.embed_concurrency,
        batch_size=config.embed_batch_size,
        requests_per_minute=config.embed_requests_per_minute,
        min_cluster_size=config.min_cluster_size,
        max_cluster_size=config.max_cluster_size,
        client=client,
        retry_config=retry_config,
        progress_desc="embed pass2",
    )
    labels = clustering_result.row_labels
    embeddings = clustering_result.row_embeddings
    pass2_df = pass1_df[
        [
            "section_id",
            "row_id",
            "dataset_source",
            "steering_text",
            "cluster_pass1",
            "cluster_name_pass1",
        ]
    ].copy()
    pass2_df["cluster_pass2_raw"] = labels
    pass2_df.to_parquet(config.output_dir / "clusters_pass2.parquet", index=False)
    np.save(config.output_dir / "embeddings_pass2.npy", embeddings)

    render_tsne_plot(
        embeddings=embeddings,
        labels=labels,
        output_path=config.output_dir / "plots" / "tsne_pass2.png",
        title="Pass2 t-SNE (title + steering embeddings)",
        sample_size=config.tsne_sample_size,
        seed=config.seed,
    )

    return {
        "sections": int(pass2_df.shape[0]),
        "unique_texts_clustered": clustering_result.unique_text_count,
        "duplicate_rows_mapped": clustering_result.duplicate_row_count,
        "noise_count": int((labels == -1).sum()),
        "cache": asdict(clustering_result.cache_stats),
        "clusters_path": str(config.output_dir / "clusters_pass2.parquet"),
    }


def stage_assign_noise(config: RunConfig) -> dict[str, Any]:
    """Assign pass2 noise points to nearest fixed-centroid k-means cluster.

    Args:
        config: Runtime config.

    Returns:
        Stage metadata.
    """

    debug_print(
        "assign_noise stage: loading clusters_pass2.parquet and embeddings_pass2.npy"
    )
    pass2_df = pd.read_parquet(config.output_dir / "clusters_pass2.parquet")
    embeddings = np.load(config.output_dir / "embeddings_pass2.npy")
    labels_raw = pass2_df["cluster_pass2_raw"].to_numpy(dtype=np.int64)
    debug_print(
        f"assign_noise stage: rows={pass2_df.shape[0]} embedding_shape={embeddings.shape}"
    )

    final_labels = labels_raw.copy()
    noise_mask = labels_raw == -1
    noise_assigned = np.zeros_like(labels_raw, dtype=bool)
    non_noise_labels = sorted(
        int(label) for label in np.unique(labels_raw) if int(label) != -1
    )
    assert non_noise_labels, "No non-noise clusters available for assignment"

    centroid_matrix = np.vstack(
        [embeddings[labels_raw == label].mean(axis=0) for label in non_noise_labels]
    ).astype(np.float32)

    kmeans: KMeans = cast(
        KMeans,
        cast(Any, KMeans)(
            n_clusters=len(non_noise_labels),
            init=centroid_matrix,
            n_init=1,
            random_state=config.seed,
        ),
    )
    kmeans_start_time = time.perf_counter()
    kmeans.fit(centroid_matrix)
    debug_print(
        f"assign_noise stage: centroid_kmeans_fit seconds={time.perf_counter() - kmeans_start_time:.3f} "
        f"centroids={len(non_noise_labels)}"
    )

    center_mapping = {}
    for km_index, center in enumerate(kmeans.cluster_centers_):
        distances = np.linalg.norm(centroid_matrix - center, axis=1)
        nearest_index = int(np.argmin(distances))
        center_mapping[km_index] = non_noise_labels[nearest_index]

    if noise_mask.any():
        debug_print(
            f"assign_noise stage: predicting noise rows={int(noise_mask.sum())}"
        )
        predicted = kmeans.predict(embeddings[noise_mask])
        mapped_labels = np.asarray(
            [center_mapping[int(label)] for label in predicted], dtype=np.int64
        )
        final_labels[noise_mask] = mapped_labels
        noise_assigned[noise_mask] = True

    clusters_final_df = pass2_df.copy()
    clusters_final_df["cluster_pass2"] = final_labels
    clusters_final_df["noise_assigned"] = noise_assigned
    clusters_final_df.to_parquet(
        config.output_dir / "clusters_final.parquet", index=False
    )
    render_tsne_plot(
        embeddings=embeddings,
        labels=final_labels,
        output_path=config.output_dir / "plots" / "tsne_pass2_noise_assigned.png",
        title="Pass2 t-SNE (after KMeans noise reassignment)",
        sample_size=config.tsne_sample_size,
        seed=config.seed,
    )
    debug_print(
        f"assign_noise stage: write complete noise_before={int((labels_raw == -1).sum())} "
        f"noise_assigned={int(noise_assigned.sum())}"
    )

    return {
        "rows": int(clusters_final_df.shape[0]),
        "noise_before": int((labels_raw == -1).sum()),
        "noise_after": int((final_labels == -1).sum()),
        "noise_assigned": int(noise_assigned.sum()),
        "clusters_final_path": str(config.output_dir / "clusters_final.parquet"),
        "tsne_plot_path": str(
            config.output_dir / "plots" / "tsne_pass2_noise_assigned.png"
        ),
    }


async def stage_name2_async(config: RunConfig, client: Any) -> dict[str, Any]:
    """Run final naming stage using pass2 final clusters.

    Args:
        config: Runtime config.
        client: Gemini-compatible client.

    Returns:
        Stage metadata.
    """

    final_df = pd.read_parquet(config.output_dir / "clusters_final.parquet")
    embeddings = np.load(config.output_dir / "embeddings_pass2.npy")
    retry_config = RetryConfig(
        timeout_seconds=config.api_timeout_seconds, max_retries=config.max_retries
    )

    cluster_names, cache_stats, naming_errors = await name_clusters_async(
        sections_df=final_df,
        embedding_matrix=embeddings,
        cluster_column="cluster_pass2",
        model_name=config.naming_model,
        concurrency=config.naming_concurrency,
        output_prompts_path=config.output_dir / "naming_prompts_pass2.jsonl",
        output_responses_path=config.output_dir / "naming_responses_pass2.jsonl",
        requests_per_minute=config.naming_requests_per_minute,
        progress_desc="name pass2",
        client=client,
        retry_config=retry_config,
        seed=config.seed,
    )

    def _resolve_final_name(cluster_id: Any) -> str:
        cluster_id_int = coerce_cluster_label(cluster_id)
        payload = cluster_names.get(cluster_id_int)
        if payload is None:
            return f"Cluster {cluster_id_int}"
        return str(payload.get("title", f"Cluster {cluster_id_int}"))

    final_df["cluster_name_pass2"] = final_df["cluster_pass2"].apply(
        _resolve_final_name
    )
    # Backward compatibility for older downstream consumers.
    final_df["cluster_name_final"] = final_df["cluster_name_pass2"]
    final_df.to_parquet(config.output_dir / "clusters_final.parquet", index=False)

    return {
        "clusters_named": len(cluster_names),
        "naming_errors": naming_errors,
        "cache": asdict(cache_stats),
        "responses_path": str(config.output_dir / "naming_responses_pass2.jsonl"),
    }


def stage_report(config: RunConfig) -> dict[str, Any]:
    """Generate plots and final markdown/json report.

    Args:
        config: Runtime config.

    Returns:
        Stage metadata.
    """

    token_stats_df = pd.read_parquet(config.output_dir / "token_stats.parquet")
    think_stats_df = pd.read_parquet(config.output_dir / "think_token_stats.parquet")
    final_df = pd.read_parquet(config.output_dir / "clusters_final.parquet")
    percentile_markers = [0.25, 0.50, 0.75, 0.95, 0.99, 0.995, 0.999]

    render_histogram(
        series=token_stats_df["steering_tokens"],
        output_path=config.output_dir / "plots" / "steering_token_hist.png",
        title="Steering token length distribution",
        xlabel="steering tokens",
        integer_bins=True,
        annotate_percentiles=percentile_markers,
        annotation_colormap="viridis",
    )
    render_histogram(
        series=token_stats_df["execution_tokens"],
        output_path=config.output_dir / "plots" / "execution_token_hist.png",
        title="Execution token length distribution",
        xlabel="execution tokens",
        bins=35,
        annotate_percentiles=percentile_markers,
        annotation_colormap="viridis",
    )
    execution_tokens = to_numeric_series(
        values=token_stats_df["execution_tokens"], label="execution_tokens_ratio"
    ).astype(float)
    steering_tokens = to_numeric_series(
        values=token_stats_df["steering_tokens"], label="steering_tokens_ratio"
    ).astype(float)
    valid_ratio_mask = steering_tokens > 0.0
    dropped_ratio_rows = int((~valid_ratio_mask).sum())
    if dropped_ratio_rows > 0:
        debug_print(
            f"ratio_hist dropped_rows_zero_steering_tokens={dropped_ratio_rows}"
        )
    execution_to_steering_ratio = (
        execution_tokens[valid_ratio_mask] / steering_tokens[valid_ratio_mask]
    )
    render_histogram(
        series=execution_to_steering_ratio,
        output_path=config.output_dir
        / "plots"
        / "execution_to_steering_ratio_hist.png",
        title="Execution / steering token ratio distribution",
        xlabel="execution_tokens / steering_tokens",
        bins=50,
        annotate_percentiles=percentile_markers,
        annotation_colormap="viridis",
    )
    render_overlay_histogram(
        left=think_stats_df["original_think_tokens"],
        right=think_stats_df["new_think_tokens"],
        output_path=config.output_dir / "plots" / "think_tokens_og_vs_new.png",
        title="OG vs transformed think token lengths",
        left_label="original",
        right_label="transformed",
        log_y=True,
    )
    render_histogram(
        series=think_stats_df["think_token_delta"],
        output_path=config.output_dir / "plots" / "think_token_diff_hist.png",
        title="Think token length difference (new - original)",
        xlabel="token delta",
        log_y=True,
    )
    required_think_columns = {"row_id", "dataset_source", "new_think_text"}
    missing_think_columns = required_think_columns.difference(think_stats_df.columns)
    assert (
        not missing_think_columns
    ), "think_token_stats.parquet is missing columns: " + ", ".join(
        sorted(missing_think_columns)
    )
    think_steer_segment_counts = (
        think_stats_df["new_think_text"].astype(str).apply(count_steering_segments)
    )
    render_scatter_plot(
        x=think_steer_segment_counts,
        y=think_stats_df["new_think_tokens"],
        output_path=config.output_dir
        / "plots"
        / "think_tokens_vs_steer_segments_scatter.png",
        title="Transformed think tokens vs steer segment count",
        xlabel="steer segments per think block",
        ylabel="transformed think tokens",
        point_size=10.0,
        alpha=0.35,
        group_labels=think_stats_df["dataset_source"],
        group_label_name="dataset source",
        show_group_fit_lines=True,
        group_display_formatter=format_dataset_source_label,
    )
    think_residual_records = collect_think_residual_records(
        think_df=think_stats_df, think_column="new_think_text"
    )
    think_residual_rows = [asdict(record) for record in think_residual_records]
    think_residuals_path = config.output_dir / "think_text_outside_segments.jsonl"
    write_jsonl(path=think_residuals_path, rows=think_residual_rows)
    debug_print(
        "think residual check "
        f"rows_with_non_whitespace_outside_segments={len(think_residual_records)}"
    )

    cluster_sizes = (
        final_df["cluster_pass2"].value_counts().sort_values(ascending=False)
    )
    assert "cluster_name_pass2" in final_df.columns, (
        "clusters_final.parquet is missing cluster_name_pass2. "
        "Run --stage name2 --no-resume before --stage report."
    )
    cluster_name_by_id: dict[int, str] = {}
    for cluster_id, cluster_df in final_df.groupby("cluster_pass2"):
        cluster_id_int = coerce_cluster_label(cluster_id)
        pass2_name_counts = (
            cluster_df["cluster_name_pass2"]
            .astype(str)
            .map(lambda name: name.strip())
            .loc[lambda series: (series != "") & (series.str.lower() != "nan")]
            .value_counts()
        )
        cluster_name_by_id[cluster_id_int] = (
            str(pass2_name_counts.index[0])
            if not pass2_name_counts.empty
            else f"Cluster {cluster_id_int}"
        )
    render_cluster_size_plot(
        cluster_counts=cluster_sizes,
        cluster_names=cluster_name_by_id,
        output_path=config.output_dir / "plots" / "final_cluster_sizes.png",
        top_n=20,
        middle_n=0,
        bottom_n=0,
        log_y=True,
    )

    top_big = cluster_sizes.head(5)
    top_small = cluster_sizes.tail(5)
    cluster_examples: dict[int, list[str]] = {}
    for cluster_id in cluster_sizes.index.tolist():
        subset = final_df[final_df["cluster_pass2"] == cluster_id].head(3)
        examples = subset["steering_text"].astype(str).tolist()
        cluster_examples[coerce_cluster_label(cluster_id)] = examples

    source_mix: dict[int, dict[str, int]] = {}
    for cluster_id, cluster_df in final_df.groupby("cluster_pass2"):
        counts = cluster_df["dataset_source"].value_counts().to_dict()
        source_mix[coerce_cluster_label(cluster_id)] = {
            str(key): int(value) for key, value in counts.items()
        }

    state = load_state(path=config.output_dir / "run_state.json")
    runtime_summary = {
        stage_name: stage_payload.get("metadata", {})
        for stage_name, stage_payload in (
            state.get("stages", {}) if isinstance(state.get("stages", {}), dict) else {}
        ).items()
        if isinstance(stage_payload, dict)
    }
    name2_runtime = runtime_summary.get("name2", {})
    naming_errors_final = (
        int(name2_runtime.get("naming_errors", 0))
        if isinstance(name2_runtime, dict)
        else 0
    )

    report_json = {
        "rows": int(final_df.shape[0]),
        "clusters": int(cluster_sizes.shape[0]),
        "naming_errors_final": naming_errors_final,
        "top_big": [
            {"cluster": coerce_cluster_label(index), "size": int(value)}
            for index, value in top_big.items()
        ],
        "top_small": [
            {"cluster": coerce_cluster_label(index), "size": int(value)}
            for index, value in top_small.items()
        ],
        "examples": {str(key): value for key, value in cluster_examples.items()},
        "source_mix": {str(key): value for key, value in source_mix.items()},
        "token_summaries": {
            "steering": summarize_series(token_stats_df["steering_tokens"]),
            "execution": summarize_series(token_stats_df["execution_tokens"]),
            "execution_to_steering_ratio": summarize_series(
                execution_to_steering_ratio
            ),
            "steer_segment_count_per_think": summarize_series(
                think_steer_segment_counts
            ),
            "og_think": summarize_series(think_stats_df["original_think_tokens"]),
            "new_think": summarize_series(think_stats_df["new_think_tokens"]),
            "think_delta": summarize_series(think_stats_df["think_token_delta"]),
        },
        "think_text_outside_segments_check": {
            "rows_with_non_whitespace_outside_segments": len(think_residual_records),
            "residuals_path": str(think_residuals_path),
            "examples": think_residual_rows[:10],
        },
        "runtime": runtime_summary,
    }
    report_json_path = config.output_dir / "cluster_report.json"
    report_json_path.write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append("# Steering Cluster Report")
    lines.append("")
    lines.append(f"- Rows analyzed: {final_df.shape[0]}")
    lines.append(f"- Final clusters: {cluster_sizes.shape[0]}")
    lines.append(
        "- Think rows with non-whitespace text outside "
        f"<steer>/<execute>: {len(think_residual_records)}"
    )
    lines.append(f"- Residual detail path: {think_residuals_path}")
    lines.append("")
    lines.append("## Top 5 Largest Clusters")
    lines.append("")
    for cluster_id, size in top_big.items():
        cluster_id_int = coerce_cluster_label(cluster_id)
        cluster_name = cluster_name_by_id.get(
            cluster_id_int, f"Cluster {cluster_id_int}"
        )
        lines.append(f"- Cluster {cluster_id_int} ({cluster_name}) size={int(size)}")
        for example in cluster_examples[cluster_id_int][:3]:
            lines.append(f"  - {example[:220].replace(chr(10), ' ')}")
    lines.append("")
    lines.append("## Top 5 Smallest Clusters")
    lines.append("")
    for cluster_id, size in top_small.items():
        cluster_id_int = coerce_cluster_label(cluster_id)
        cluster_name = cluster_name_by_id.get(
            cluster_id_int, f"Cluster {cluster_id_int}"
        )
        lines.append(f"- Cluster {cluster_id_int} ({cluster_name}) size={int(size)}")
        for example in cluster_examples[cluster_id_int][:3]:
            lines.append(f"  - {example[:220].replace(chr(10), ' ')}")
    lines.append("")
    lines.append("## Noise Reassignment")
    lines.append("")
    noise_assigned = (
        int(final_df["noise_assigned"].sum())
        if "noise_assigned" in final_df.columns
        else 0
    )
    lines.append(
        f"- Reassigned rows originally labeled as pass2 noise: {noise_assigned}"
    )
    lines.append(f"- Final naming errors (fallback title used): {naming_errors_final}")
    lines.append("")
    lines.append("## Source Distribution by Cluster")
    lines.append("")
    for cluster_id in cluster_sizes.index.tolist()[:15]:
        cluster_id_int = coerce_cluster_label(cluster_id)
        lines.append(f"- Cluster {cluster_id_int}")
        source_counts = source_mix[cluster_id_int]
        for source_name, count in list(source_counts.items())[:5]:
            lines.append(f"  - {source_name}: {count}")
    lines.append("")
    lines.append("## Runtime and Cache Summary")
    lines.append("")
    for stage_name, payload in runtime_summary.items():
        lines.append(f"- {stage_name}: {json.dumps(payload, ensure_ascii=False)}")
    lines.append("")
    lines.append("## Think Residual Text Check")
    lines.append("")
    if think_residual_records:
        for record in think_residual_records[:10]:
            lines.append(
                "- row_id="
                f"{record.row_id} source={record.dataset_source} "
                f"steer_segments={record.steer_segment_count} "
                f"residual_chars={record.residual_char_count} "
                f"residual_preview={record.residual_preview}"
            )
    else:
        lines.append(
            "- No rows contain non-whitespace text outside <steer>/<execute> blocks."
        )
    lines.append("")
    lines.append("## Additional Suggestions")
    lines.append("")
    lines.append(
        "1. Add stability analysis across random seeds using adjusted Rand index."
    )
    lines.append("2. Add per-source cluster purity metrics and drift checks.")
    lines.append(
        "3. Add representative central vs diverse steering exemplars per cluster."
    )
    lines.append(
        "4. Add optional UMAP projection for faster large-scale visual checks."
    )

    report_md_path = config.output_dir / "cluster_report.md"
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "report_md": str(report_md_path),
        "report_json": str(report_json_path),
        "clusters": int(cluster_sizes.shape[0]),
        "rows": int(final_df.shape[0]),
        "think_rows_with_non_whitespace_outside_segments": len(think_residual_records),
        "think_residuals_path": str(think_residuals_path),
    }


def parse_args() -> RunConfig:
    """Parse command-line arguments into RunConfig.

    Returns:
        Parsed runtime configuration.
    """

    parser = argparse.ArgumentParser(
        description="Analyze steering strings with two-pass clustering and Gemini naming"
    )
    parser.add_argument(
        "--transformed-path", type=Path, default=DEFAULT_TRANSFORMED_PATH
    )
    parser.add_argument("--og-path", type=Path, default=DEFAULT_OG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--naming-model", default=DEFAULT_NAMING_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE
    )
    parser.add_argument(
        "--max-cluster-size", type=int, default=DEFAULT_MAX_CLUSTER_SIZE
    )
    parser.add_argument(
        "--naming-concurrency", type=int, default=DEFAULT_NAMING_CONCURRENCY
    )
    parser.add_argument(
        "--embed-concurrency", type=int, default=DEFAULT_EMBED_CONCURRENCY
    )
    parser.add_argument(
        "--token-concurrency", type=int, default=DEFAULT_TOKEN_CONCURRENCY
    )
    parser.add_argument(
        "--embed-batch-size", type=int, default=DEFAULT_EMBED_BATCH_SIZE
    )
    parser.add_argument(
        "--naming-requests-per-minute",
        type=int,
        default=DEFAULT_NAMING_REQUESTS_PER_MINUTE,
    )
    parser.add_argument(
        "--embed-requests-per-minute",
        type=int,
        default=DEFAULT_EMBED_REQUESTS_PER_MINUTE,
    )
    parser.add_argument(
        "--token-requests-per-minute",
        type=int,
        default=DEFAULT_TOKEN_REQUESTS_PER_MINUTE,
    )
    parser.add_argument(
        "--tsne-sample-size", type=int, default=DEFAULT_TSNE_SAMPLE_SIZE
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--stage", choices=[*STAGE_ORDER, "all"], default="all")
    parser.add_argument(
        "--api-timeout-seconds", type=int, default=DEFAULT_API_TIMEOUT_SECONDS
    )
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)

    args = parser.parse_args()
    return RunConfig(
        transformed_path=args.transformed_path,
        og_path=args.og_path,
        output_dir=args.output_dir,
        env_file=args.env_file,
        naming_model=args.naming_model,
        embedding_model=args.embedding_model,
        min_cluster_size=args.min_cluster_size,
        max_cluster_size=args.max_cluster_size,
        naming_concurrency=args.naming_concurrency,
        embed_concurrency=args.embed_concurrency,
        token_concurrency=args.token_concurrency,
        embed_batch_size=args.embed_batch_size,
        naming_requests_per_minute=args.naming_requests_per_minute,
        embed_requests_per_minute=args.embed_requests_per_minute,
        token_requests_per_minute=args.token_requests_per_minute,
        tsne_sample_size=args.tsne_sample_size,
        seed=args.seed,
        resume=args.resume,
        stage=args.stage,
        api_timeout_seconds=args.api_timeout_seconds,
        max_retries=args.max_retries,
    )


def run_stage(config: RunConfig, stage: str, client: Any | None) -> dict[str, Any]:
    """Run one stage and return metadata.

    Args:
        config: Runtime config.
        stage: Stage name.
        client: Gemini-compatible client when required.

    Returns:
        Stage metadata.
    """

    start_time = time.time()
    if stage == "extract":
        metadata = stage_extract(config=config)
    elif stage == "tokens":
        assert client is not None, "Gemini client required for tokens stage"
        metadata = asyncio.run(stage_tokens_async(config=config, client=client))
    elif stage == "cluster1":
        assert client is not None, "Gemini client required for cluster1 stage"
        metadata = asyncio.run(stage_cluster1_async(config=config, client=client))
    elif stage == "name1":
        assert client is not None, "Gemini client required for name1 stage"
        metadata = asyncio.run(stage_name1_async(config=config, client=client))
    elif stage == "cluster2":
        assert client is not None, "Gemini client required for cluster2 stage"
        metadata = asyncio.run(stage_cluster2_async(config=config, client=client))
    elif stage == "assign_noise":
        metadata = stage_assign_noise(config=config)
    elif stage == "name2":
        assert client is not None, "Gemini client required for name2 stage"
        metadata = asyncio.run(stage_name2_async(config=config, client=client))
    elif stage == "report":
        metadata = stage_report(config=config)
    else:
        raise SystemExit(f"Unsupported stage: {stage}")

    metadata["runtime_seconds"] = round(time.time() - start_time, 3)
    return metadata


def should_use_api(stage: str) -> bool:
    """Return True if stage requires Gemini API access.

    Args:
        stage: Stage name.

    Returns:
        API requirement flag.
    """

    return stage in {"tokens", "cluster1", "name1", "cluster2", "name2"}


def main() -> None:
    """Entrypoint for steering clustering and report pipeline."""

    config = parse_args()
    ensure_output_dirs(base_dir=config.output_dir)

    if not config.transformed_path.exists():
        raise SystemExit(f"Missing transformed dataset: {config.transformed_path}")
    if not config.og_path.exists():
        raise SystemExit(f"Missing OG dataset: {config.og_path}")

    state_path = config.output_dir / "run_state.json"
    state = load_state(path=state_path)

    stages = list(STAGE_ORDER) if config.stage == "all" else [config.stage]

    api_key: str | None = None
    client: Any | None = None

    for stage in stages:
        if stage_should_skip(config=config, state=state, stage=stage):
            print(f"Skipping stage (resume): {stage}")
            continue

        if should_use_api(stage=stage):
            if api_key is None:
                api_key = resolve_api_key(env_file=config.env_file)
                client = build_client(api_key=api_key)
            assert client is not None

        print(f"Running stage: {stage}")
        stage_metadata = run_stage(config=config, stage=stage, client=client)
        mark_stage_complete(state=state, stage=stage, metadata=stage_metadata)
        save_state(path=state_path, state=state)
        print(json.dumps({"stage": stage, "metadata": stage_metadata}, indent=2))


if __name__ == "__main__":
    main()
