from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

THINK_PATTERN = r"<think>(.*?)</think>"
STAGE_ORDER: tuple[str, ...] = ("sample", "filter", "stratify", "transform")

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
        rows_per_shard: Rows sampled per shard.
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
        max_rows: Target transformed row count.
        dry_run: Skip API calls when true.
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
        system_prompt_path: System prompt path.
        user_prompt_path: User prompt path.
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
        dry_run: Skip writes/API requests.
        stage: Requested stage.
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
        stratified_path: Stratified sample JSONL.
        transformed_path: Transformed JSONL.
        state_path: Pipeline state path.
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
        updated_at: UTC timestamp.
        metadata: Arbitrary stage metadata.
    """

    completed: bool
    config_hash: str
    updated_at: str
    metadata: dict[str, object]


@dataclass(frozen=True)
class ThinkTask:
    """Mapping of one transformed think block.

    Args:
        row_index: Index in row buffer.
        message_index: Message index in row.
        block_index: Think block index in message.
    """

    row_index: int
    message_index: int
    block_index: int
