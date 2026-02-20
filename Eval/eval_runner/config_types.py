"""Standalone typed configuration objects for benchmark evaluation."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class LmEvalConfig:
    """Configuration for benchmark evaluation with `lm_eval`.

    Args:
        tasks: Benchmark task names passed to `lm_eval.simple_evaluate`.
        batch_size: Batch size passed to `lm_eval.simple_evaluate`.
        num_fewshot: Optional few-shot count override.
        aime_avg_k: Number of sampled responses per AIME question for avg@k.
        apply_chat_template: Enables `--apply_chat_template`.
        think_end_token: Token/string used to strip reasoning trace in eval.
        trust_remote_code: Enables model loading for repos with custom code.
        log_samples: Enables `--log_samples` to save per-example generations.
        temperature: Default generation temperature passed via `--gen_kwargs`.
        top_p: Default nucleus sampling value passed via `--gen_kwargs`.
        max_gen_toks: Default max generated tokens passed via `--gen_kwargs`.
        vllm_disable_log_stats: Forwarded to vLLM engine to control throughput logs.
        vllm_log_stats_interval: vLLM stats log heartbeat interval in seconds.
        vllm_logging_level: vLLM logger level (`INFO`, `DEBUG`, ...).
    """

    tasks: tuple[str, ...] = ("minerva_math500", "aime24", "aime25")
    batch_size: int | str = 4
    num_fewshot: int | None = None
    aime_avg_k: int = 32
    apply_chat_template: bool = True
    think_end_token: str = "</think>"
    trust_remote_code: bool = True
    model_type: str = "hf"
    tensor_parallel_size: int = -1
    gpu_memory_utilization: float = 0.9
    data_parallel_size: int = 1
    log_samples: bool = False
    temperature: float = 0.6
    top_p: float = 0.95
    max_gen_toks: int = 32768
    vllm_disable_log_stats: bool = False
    vllm_log_stats_interval: float = 10.0
    vllm_logging_level: str = "INFO"


@dataclass(frozen=True)
class EvalRunMetadata:
    """Optional run metadata used when standalone eval logs to W&B.

    Args:
        run_name: Base run name from a train run config.
        wandb_project: W&B project for init.
        wandb_entity: Optional W&B entity for init.
    """

    run_name: str
    wandb_project: str
    wandb_entity: str | None = None


def _resolve_path(base_dir: Path, value: str) -> Path:
    """Resolve a relative or absolute path.

    Args:
        base_dir: Parent directory used for relative values.
        value: Path string to resolve.

    Returns:
        Resolved absolute path.
    """
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


def _load_yaml_mapping(yaml_path: Path) -> dict[str, Any]:
    """Load one YAML file into a mapping.

    Args:
        yaml_path: YAML file path.

    Returns:
        Parsed mapping payload.
    """
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), f"YAML payload must be a mapping: {yaml_path}"
    return payload


def _normalize_lm_eval_path(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    """Normalize `lm_eval` string pointers to absolute file paths.

    Args:
        payload: Raw run config mapping.
        source_path: Source YAML path used for relative resolution.

    Returns:
        Mapping where `lm_eval` paths are absolute when present.
    """
    lm_eval_value = payload.get("lm_eval")
    if not isinstance(lm_eval_value, str):
        return payload
    normalized = dict(payload)
    normalized["lm_eval"] = str(
        _resolve_path(base_dir=source_path.parent, value=lm_eval_value)
    )
    return normalized


def _load_run_payload_with_base(
    config_path: Path,
    visited_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Load a run config with recursive `base_config` merging.

    Args:
        config_path: Run config path.
        visited_paths: Optional recursion guard set.

    Returns:
        Fully merged run config mapping.
    """
    active_visited_paths = set() if visited_paths is None else set(visited_paths)
    resolved_config_path = config_path.resolve()
    assert resolved_config_path not in active_visited_paths, "Circular base_config detected."
    active_visited_paths.add(resolved_config_path)

    payload = _normalize_lm_eval_path(
        payload=_load_yaml_mapping(yaml_path=resolved_config_path),
        source_path=resolved_config_path,
    )
    base_value = payload.get("base_config")
    if base_value is None:
        return payload
    base_path = _resolve_path(base_dir=resolved_config_path.parent, value=str(base_value))
    base_payload = _load_run_payload_with_base(
        config_path=base_path,
        visited_paths=active_visited_paths,
    )
    merged_payload = dict(base_payload)
    merged_payload.update(payload)
    return merged_payload


def _looks_like_run_config(payload: dict[str, Any]) -> bool:
    """Check whether a payload is a full train-run config.

    Args:
        payload: Parsed YAML mapping.

    Returns:
        `True` when payload likely contains run metadata and lm_eval settings.
    """
    if "run_name" not in payload:
        return False
    if "wandb_project" in payload or "base_config" in payload:
        return True
    return "lm_eval" in payload and "model_name_or_path" in payload


def _extract_lm_eval_payload(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Extract and resolve `lm_eval` mapping from run or eval-only payload.

    Args:
        payload: Source payload (run config or lm_eval-only mapping).
        base_dir: Parent directory used to resolve relative `lm_eval` paths.

    Returns:
        Parsed `lm_eval` configuration mapping.
    """
    lm_eval_value: Any = payload.get("lm_eval", payload)
    if isinstance(lm_eval_value, str):
        lm_eval_path = _resolve_path(base_dir=base_dir, value=lm_eval_value)
        return _load_yaml_mapping(yaml_path=lm_eval_path)
    assert isinstance(lm_eval_value, dict), "`lm_eval` must be a mapping or YAML path."
    return lm_eval_value


def _build_lm_eval_config(lm_eval_payload: dict[str, Any]) -> LmEvalConfig:
    """Build `LmEvalConfig` from one parsed mapping.

    Args:
        lm_eval_payload: Parsed `lm_eval` mapping.

    Returns:
        Typed eval configuration object.
    """
    aime_avg_k = int(lm_eval_payload.get("aime_avg_k", 32))
    assert aime_avg_k >= 1, "`lm_eval.aime_avg_k` must be >= 1."
    vllm_log_stats_interval = float(lm_eval_payload.get("vllm_log_stats_interval", 10.0))
    assert vllm_log_stats_interval > 0.0, "`lm_eval.vllm_log_stats_interval` must be > 0."
    return LmEvalConfig(
        tasks=tuple(lm_eval_payload.get("tasks", LmEvalConfig.tasks)),
        batch_size=lm_eval_payload.get("batch_size", 4),
        num_fewshot=lm_eval_payload.get("num_fewshot"),
        aime_avg_k=aime_avg_k,
        apply_chat_template=bool(lm_eval_payload.get("apply_chat_template", True)),
        think_end_token=str(lm_eval_payload.get("think_end_token", "</think>")),
        trust_remote_code=bool(lm_eval_payload.get("trust_remote_code", True)),
        model_type=str(lm_eval_payload.get("model_type", "hf")),
        tensor_parallel_size=_resolve_tensor_parallel_size(lm_eval_payload=lm_eval_payload),
        gpu_memory_utilization=float(
            lm_eval_payload.get("gpu_memory_utilization", 0.9)
        ),
        data_parallel_size=int(lm_eval_payload.get("data_parallel_size", 1)),
        log_samples=bool(lm_eval_payload.get("log_samples", False)),
        temperature=float(lm_eval_payload.get("temperature", 0.6)),
        top_p=float(lm_eval_payload.get("top_p", 0.95)),
        max_gen_toks=int(lm_eval_payload.get("max_gen_toks", 32768)),
        vllm_disable_log_stats=bool(lm_eval_payload.get("vllm_disable_log_stats", False)),
        vllm_log_stats_interval=vllm_log_stats_interval,
        vllm_logging_level=str(lm_eval_payload.get("vllm_logging_level", "INFO")).upper(),
    )


def _resolve_tensor_parallel_size(lm_eval_payload: dict[str, Any]) -> int:
    """Resolve tensor-parallel size from config or local GPU availability.

    Args:
        lm_eval_payload: Parsed `lm_eval` mapping.

    Returns:
        Explicit configured size, or detected GPU count fallback.
    """
    configured_size = int(lm_eval_payload.get("tensor_parallel_size", -1))
    if configured_size > 0:
        return configured_size
    if importlib.util.find_spec(name="torch") is None:
        return 1
    import torch

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def _build_eval_run_metadata(run_payload: dict[str, Any]) -> EvalRunMetadata:
    """Build optional W&B metadata from a run payload.

    Args:
        run_payload: Merged run config payload.

    Returns:
        Metadata used by standalone W&B initialization.
    """
    return EvalRunMetadata(
        run_name=str(run_payload["run_name"]),
        wandb_project=str(run_payload["wandb_project"]),
        wandb_entity=(
            None
            if run_payload.get("wandb_entity") is None
            else str(run_payload["wandb_entity"])
        ),
    )


def parse_eval_config(config_path: Path) -> tuple[LmEvalConfig, EvalRunMetadata | None]:
    """Parse a full run config or an lm_eval-only config.

    Args:
        config_path: YAML path containing either run config or lm_eval payload.

    Returns:
        Tuple of `(LmEvalConfig, optional EvalRunMetadata)`.

    Example:
        >>> cfg, meta = parse_eval_config(config_path=Path("configs/lm_eval_vllm.yaml"))
        >>> cfg.aime_avg_k
        32
    """
    payload = _load_yaml_mapping(yaml_path=config_path)
    if not _looks_like_run_config(payload=payload):
        lm_eval_payload = _extract_lm_eval_payload(
            payload=payload,
            base_dir=config_path.parent,
        )
        return _build_lm_eval_config(lm_eval_payload=lm_eval_payload), None

    merged_run_payload = _load_run_payload_with_base(config_path=config_path)
    lm_eval_payload = _extract_lm_eval_payload(
        payload=merged_run_payload,
        base_dir=config_path.parent,
    )
    return (
        _build_lm_eval_config(lm_eval_payload=lm_eval_payload),
        _build_eval_run_metadata(run_payload=merged_run_payload),
    )
