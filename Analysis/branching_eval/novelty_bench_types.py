"""Typed NoveltyBench config, prompt loading, and generation cleaning helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from branching_eval.config_types import (
    BranchingEvalConfig,
    load_branching_eval_config,
)
from io_utils import read_jsonl

DatasetSplit = Literal["curated", "wildchat"]
OutputTextMode = Literal["raw", "after_think", "strip_internal_tags"]

THINK_CLOSE_PATTERN = re.compile(r"</think>", flags=re.IGNORECASE)
THINK_OPEN_PATTERN = re.compile(r"<think\b[^>]*>", flags=re.IGNORECASE)
INTERNAL_TAG_PATTERN = re.compile(
    r"</?(?:think|steer|exec)\b[^>]*>",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class NoveltyBenchConfig:
    """NoveltyBench-specific generation settings.

    Args:
        dataset_name: Hugging Face dataset id.
        dataset_split: Dataset split, either `curated` or `wildchat`.
        num_generations: Required generations per prompt for `distinct_k`.
        prompt_column: Dataset column containing user prompts.
        id_column: Dataset column containing stable prompt ids.
        shuffle: Whether to shuffle prompts before applying limits.
        shuffle_seed: Seed used when `shuffle` is enabled.
        max_concurrent_prompts: Maximum prompts in flight for one run.
        output_text_mode: Cleaning mode applied before writing generations.
        prompt_max_attempts: Maximum whole-prompt attempts after vLLM transport errors.
        prompt_retry_base_delay_seconds: Initial whole-prompt retry backoff.

    Returns:
        Dataclass containing NoveltyBench generation behavior.

    Example:
        >>> cfg = NoveltyBenchConfig(dataset_split="curated", num_generations=10)
        >>> cfg.dataset_split
        'curated'
    """

    dataset_name: str = "yimingzhang/novelty-bench"
    dataset_split: DatasetSplit = "curated"
    num_generations: int = 10
    prompt_column: str = "prompt"
    id_column: str = "id"
    shuffle: bool = False
    shuffle_seed: int = 1234
    max_concurrent_prompts: int = 1
    output_text_mode: OutputTextMode = "after_think"
    prompt_max_attempts: int = 3
    prompt_retry_base_delay_seconds: float = 2.0

    def validate(self) -> None:
        """Validate NoveltyBench settings.

        Args:
            None.

        Returns:
            None.
        """

        assert self.dataset_name.strip(), "dataset_name must be non-empty"
        assert self.dataset_split in {"curated", "wildchat"}, "unsupported split"
        assert self.num_generations >= 1, "num_generations must be >= 1"
        assert self.prompt_column.strip(), "prompt_column must be non-empty"
        assert self.id_column.strip(), "id_column must be non-empty"
        assert self.max_concurrent_prompts >= 1, "max_concurrent_prompts must be >= 1"
        assert self.prompt_max_attempts >= 1, "prompt_max_attempts must be >= 1"
        assert (
            self.prompt_retry_base_delay_seconds >= 0.0
        ), "prompt_retry_base_delay_seconds must be non-negative"
        assert self.output_text_mode in {
            "raw",
            "after_think",
            "strip_internal_tags",
        }, "unsupported output_text_mode"


@dataclass(frozen=True)
class NoveltyBenchPrompt:
    """One prompt loaded from NoveltyBench.

    Args:
        doc_id: Sequential id used for run events and CLI filtering.
        benchmark_id: Stable NoveltyBench prompt id.
        prompt_text: User prompt text sent to the model.

    Returns:
        Dataclass used by the generation runner.
    """

    doc_id: int
    benchmark_id: str
    prompt_text: str


@dataclass(frozen=True)
class NoveltyPromptResult:
    """Completed generations for one NoveltyBench prompt.

    Args:
        prompt: Source prompt metadata.
        generations: Cleaned generations written to official JSONL.
        raw_generations: Raw model outputs before cleaning.

    Returns:
        Result row plus diagnostics for sidecar metadata.
    """

    prompt: NoveltyBenchPrompt
    generations: tuple[str, ...]
    raw_generations: tuple[str, ...]

    def official_row(self, *, model_name: str) -> dict[str, Any]:
        """Return official-compatible NoveltyBench `generations.jsonl` row.

        Args:
            model_name: Model label to write in the row.

        Returns:
            JSON-ready row consumed by NoveltyBench partition/score scripts.
        """

        return {
            "id": self.prompt.benchmark_id,
            "prompt": self.prompt.prompt_text,
            "model": model_name,
            "generations": list(self.generations),
        }

    def metadata_row(self) -> dict[str, Any]:
        """Return sidecar metadata for generation auditing.

        Args:
            None.

        Returns:
            JSON-ready diagnostic row.
        """

        return {
            "doc_id": self.prompt.doc_id,
            "id": self.prompt.benchmark_id,
            "raw_generation_char_counts": [
                len(generation) for generation in self.raw_generations
            ],
            "generation_char_counts": [
                len(generation) for generation in self.generations
            ],
            "generation_complete": [
                is_clean_generation_complete(generation=generation)
                for generation in self.generations
            ],
        }


def load_config_bundle(
    *,
    config_path: Path,
    dataset_split_override: str | None,
    num_generations_override: int | None,
    output_text_mode_override: str | None,
) -> tuple[BranchingEvalConfig, NoveltyBenchConfig]:
    """Load core branching config plus NoveltyBench-specific settings.

    Args:
        config_path: YAML config path.
        dataset_split_override: Optional CLI split override.
        num_generations_override: Optional CLI generation-count override.
        output_text_mode_override: Optional CLI output-cleaning override.

    Returns:
        Tuple of `(branching_config, novelty_config)`.
    """

    branching_config = load_branching_eval_config(config_path=config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "config payload must be a mapping"
    novelty_config = parse_novelty_config(payload=payload)
    if dataset_split_override is not None:
        novelty_config = replace(
            novelty_config,
            dataset_split=_as_dataset_split(value=dataset_split_override),
        )
    if num_generations_override is not None:
        novelty_config = replace(
            novelty_config,
            num_generations=int(num_generations_override),
        )
    if output_text_mode_override is not None:
        novelty_config = replace(
            novelty_config,
            output_text_mode=_as_output_text_mode(value=output_text_mode_override),
        )
    novelty_config.validate()
    return branching_config, novelty_config


def parse_novelty_config(*, payload: dict[str, Any]) -> NoveltyBenchConfig:
    """Parse `novelty_bench` section from a YAML payload.

    Args:
        payload: Raw YAML mapping.

    Returns:
        Validated NoveltyBench config.
    """

    novelty_payload = payload.get("novelty_bench", {})
    if not isinstance(novelty_payload, dict):
        config = NoveltyBenchConfig()
        config.validate()
        return config
    config = NoveltyBenchConfig(
        dataset_name=str(
            novelty_payload.get("dataset_name", "yimingzhang/novelty-bench")
        ),
        dataset_split=_as_dataset_split(
            value=novelty_payload.get("dataset_split", "curated")
        ),
        num_generations=int(novelty_payload.get("num_generations", 10)),
        prompt_column=str(novelty_payload.get("prompt_column", "prompt")),
        id_column=str(novelty_payload.get("id_column", "id")),
        shuffle=bool(novelty_payload.get("shuffle", False)),
        shuffle_seed=int(novelty_payload.get("shuffle_seed", 1234)),
        max_concurrent_prompts=int(novelty_payload.get("max_concurrent_prompts", 1)),
        output_text_mode=_as_output_text_mode(
            value=novelty_payload.get("output_text_mode", "after_think")
        ),
        prompt_max_attempts=int(novelty_payload.get("prompt_max_attempts", 3)),
        prompt_retry_base_delay_seconds=float(
            novelty_payload.get("prompt_retry_base_delay_seconds", 2.0)
        ),
    )
    config.validate()
    return config


def load_novelty_bench_prompts(
    *,
    novelty_config: NoveltyBenchConfig,
    limit: int | None,
    doc_ids: tuple[int, ...] | None,
) -> list[NoveltyBenchPrompt]:
    """Load NoveltyBench prompts from Hugging Face.

    Args:
        novelty_config: Dataset and field settings.
        limit: Optional prompt cap applied before `doc_ids`.
        doc_ids: Optional sequential doc ids to keep.

    Returns:
        Prompt records in execution order.
    """

    dataset = load_hf_dataset(
        dataset_name=novelty_config.dataset_name,
        dataset_split=novelty_config.dataset_split,
    )
    if novelty_config.shuffle:
        dataset = dataset.shuffle(seed=novelty_config.shuffle_seed)
    rows = list(dataset)
    if limit is not None:
        rows = rows[: max(0, limit)]
    prompts = [
        prompt_from_dataset_row(
            row=dict(row),
            doc_id=doc_id,
            novelty_config=novelty_config,
        )
        for doc_id, row in enumerate(rows)
    ]
    return filter_prompts_by_doc_ids(prompts=prompts, doc_ids=doc_ids)


def load_hf_dataset(*, dataset_name: str, dataset_split: str) -> Any:
    """Load a Hugging Face dataset split.

    Args:
        dataset_name: Dataset id.
        dataset_split: Split name.

    Returns:
        Dataset object from `datasets.load_dataset`.
    """

    from datasets import load_dataset

    return load_dataset(dataset_name, split=dataset_split)


def prompt_from_dataset_row(
    *,
    row: dict[str, Any],
    doc_id: int,
    novelty_config: NoveltyBenchConfig,
) -> NoveltyBenchPrompt:
    """Build a prompt record from one dataset row.

    Args:
        row: Raw dataset row mapping.
        doc_id: Sequential doc index.
        novelty_config: Column configuration.

    Returns:
        Parsed prompt record.
    """

    assert (
        novelty_config.id_column in row
    ), f"Missing id column: {novelty_config.id_column}"
    assert (
        novelty_config.prompt_column in row
    ), f"Missing prompt column: {novelty_config.prompt_column}"
    return NoveltyBenchPrompt(
        doc_id=doc_id,
        benchmark_id=str(row[novelty_config.id_column]),
        prompt_text=str(row[novelty_config.prompt_column]),
    )


def filter_prompts_by_doc_ids(
    *, prompts: list[NoveltyBenchPrompt], doc_ids: tuple[int, ...] | None
) -> list[NoveltyBenchPrompt]:
    """Restrict prompt records to explicit doc ids.

    Args:
        prompts: Ordered prompt records.
        doc_ids: Optional doc ids to keep.

    Returns:
        Filtered prompt records.
    """

    if doc_ids is None:
        return prompts
    requested_doc_ids = set(doc_ids)
    filtered = [prompt for prompt in prompts if prompt.doc_id in requested_doc_ids]
    missing = sorted(requested_doc_ids - {prompt.doc_id for prompt in filtered})
    assert not missing, f"requested doc ids are unavailable: {missing}"
    return filtered


def clean_generation_text(*, text: str, mode: OutputTextMode) -> str:
    """Clean one generation before NoveltyBench scoring.

    Args:
        text: Raw model completion.
        mode: Cleaning strategy.

    Returns:
        User-visible generation text.

    Example:
        >>> clean_generation_text(text="<think>x</think> answer", mode="after_think")
        'answer'
        >>> clean_generation_text(text="<think>x</think>", mode="after_think")
        ''
    """

    if mode == "raw":
        return text.strip()
    if mode == "after_think":
        match = THINK_CLOSE_PATTERN.search(text)
        if match is not None:
            return strip_internal_tags(text=text[match.end() :])
        if THINK_OPEN_PATTERN.search(text):
            return ""
        return strip_internal_tags(text=text)
    if mode == "strip_internal_tags":
        return strip_internal_tags(text=text)
    raise AssertionError(f"Unsupported output text mode: {mode}")


def strip_internal_tags(*, text: str) -> str:
    """Remove internal reasoning-control tags from text.

    Args:
        text: Candidate generation text.

    Returns:
        Text with project-internal tags removed and whitespace trimmed.
    """

    return INTERNAL_TAG_PATTERN.sub("", text).strip()


def is_clean_generation_complete(*, generation: str) -> bool:
    """Return whether cleaned text is safe to write as a final answer.

    Args:
        generation: Cleaned generation intended for NoveltyBench scoring.

    Returns:
        True when nonempty and free of internal reasoning tags.
    """

    return bool(generation.strip()) and INTERNAL_TAG_PATTERN.search(generation) is None


def read_completed_prompt_ids(
    *, generations_path: Path, num_generations: int
) -> set[str]:
    """Read prompt ids already completed in an output JSONL file.

    Args:
        generations_path: Existing `generations.jsonl`.
        num_generations: Required generations per row.

    Returns:
        Set of benchmark ids with complete rows.
    """

    if not generations_path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(path=generations_path):
        generations = row.get("generations")
        if (
            isinstance(generations, list)
            and len(generations) == num_generations
            and all(
                is_clean_generation_complete(generation=str(generation))
                for generation in generations
            )
        ):
            completed.add(str(row.get("id")))
    return completed


def _as_dataset_split(*, value: object) -> DatasetSplit:
    text = str(value)
    assert text in {"curated", "wildchat"}, f"unsupported dataset_split: {text}"
    return text  # type: ignore[return-value]


def _as_output_text_mode(*, value: object) -> OutputTextMode:
    text = str(value)
    assert text in {
        "raw",
        "after_think",
        "strip_internal_tags",
    }, f"unsupported output_text_mode: {text}"
    return text  # type: ignore[return-value]
