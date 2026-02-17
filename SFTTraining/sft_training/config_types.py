"""Typed configuration objects for SFT training and benchmark evaluation."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml


def _resolve_path(base_dir: Path, value: str) -> Path:
    """Resolve `value` against `base_dir` when `value` is not absolute.

    Args:
        base_dir: Directory containing the config file.
        value: Relative or absolute filesystem path.

    Returns:
        Absolute resolved path.
    """
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (base_dir / candidate).resolve()


@dataclass(frozen=True)
class LmEvalConfig:
    """Configuration for benchmark evaluation with `lm-eval`.

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


@dataclass(frozen=True)
class LoraConfig:
    """Configuration for optional LoRA adapter tuning.

    Args:
        r: LoRA rank.
        lora_alpha: LoRA scaling factor.
        lora_dropout: Dropout applied on LoRA updates.
        target_modules: Module names where LoRA layers are injected.
        bias: PEFT bias mode (`none`, `all`, or `lora_only`).
        task_type: PEFT task type; defaults to `CAUSAL_LM`.

    Example:
        >>> cfg = LoraConfig(
        ...     r=16,
        ...     lora_alpha=32,
        ...     target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ... )
        >>> cfg.r
        16
    """

    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: Literal["none", "all", "lora_only"] = "none"
    task_type: str = "CAUSAL_LM"


@dataclass(frozen=True)
class RunConfig:
    """Top-level run configuration used by `train.py`.

    Args:
        run_name: W&B and checkpoint run name.
        model_name_or_path: Base model identifier.
        dataset_path: Path to transformed JSONL dataset.
        output_dir: Output directory for checkpoints/logs.
        deepspeed_config_path: Optional path to deepspeed config JSON.
        wandb_project: W&B project name.
        wandb_entity: Optional W&B entity name.
        seed: Random seed for split/training.
        num_train_epochs: Number of training epochs.
        learning_rate: Optimizer learning rate.
        max_seq_length: Max sequence length for SFT.
        per_device_train_batch_size: Train micro-batch size per GPU.
        per_device_eval_batch_size: Eval micro-batch size per GPU.
        gradient_accumulation_steps: Gradient accumulation count.
        warmup_ratio: Warmup ratio for scheduler.
        optim: Optimizer name passed to TRL/Transformers trainer.
        adam_beta1: Adam beta1 coefficient.
        adam_beta2: Adam beta2 coefficient.
        adam_epsilon: Adam epsilon numerical stabilizer.
        weight_decay: AdamW weight decay.
        eval_split_ratio: Fraction of rows reserved for eval loss.
        save_total_limit: Maximum number of stored checkpoints.
        save_only_model: Save model weights only in checkpoints.
        lora: Optional LoRA adapter configuration.
        lm_eval: Nested benchmark evaluation config.

    Example:
        >>> config = RunConfig.from_yaml(
        ...     yaml_path=Path("configs/runs/olmo3_7b_instruct_to_think.yaml")
        ... )
        >>> config.num_train_epochs
        8
    """

    run_name: str
    model_name_or_path: str
    dataset_path: Path
    output_dir: Path
    deepspeed_config_path: Path | None
    wandb_project: str
    wandb_entity: str | None = None
    seed: int = 42
    num_train_epochs: int = 8
    learning_rate: float = 1e-5
    max_seq_length: int = 8192
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.03
    optim: str = "adamw_torch_fused"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    weight_decay: float = 0.0
    eval_split_ratio: float = 0.05
    save_total_limit: int = 8
    save_only_model: bool = False
    lora: LoraConfig | None = None
    lm_eval: LmEvalConfig = field(default_factory=LmEvalConfig)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> RunConfig:
        """Load a typed run config from YAML.

        Args:
            yaml_path: Filesystem path to run YAML.

        Returns:
            Parsed `RunConfig`.
        """
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict), "Run config must be a mapping."

        base_config_path = payload.get("base_config")
        if base_config_path:
            base_path = _resolve_path(
                base_dir=yaml_path.parent, value=str(base_config_path)
            )
            base_payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
            if not isinstance(base_payload, dict):
                raise ValueError(f"Base config must be a mapping: {base_path}")

            # Pre-resolve critical paths relative to the base config file location
            base_dir = base_path.parent
            if "dataset_path" in base_payload:
                base_payload["dataset_path"] = str(
                    _resolve_path(base_dir, base_payload["dataset_path"])
                )
            if "lm_eval" in base_payload and isinstance(base_payload["lm_eval"], str):
                base_payload["lm_eval"] = str(
                    _resolve_path(base_dir, base_payload["lm_eval"])
                )

            # Merge: payload overrides base_payload
            base_payload.update(payload)
            payload = base_payload

        return cls._from_payload(payload=payload, base_dir=yaml_path.parent)

    @classmethod
    def _from_payload(cls, payload: dict[str, Any], base_dir: Path) -> RunConfig:
        """Build `RunConfig` from a validated payload mapping.

        Args:
            payload: Raw mapping loaded from YAML.
            base_dir: Parent directory of the YAML file.

        Returns:
            Parsed `RunConfig`.
        """
        return RunConfig(
            **_parse_common_kwargs(payload=payload, base_dir=base_dir),
            **_parse_numeric_kwargs(payload=payload),
            lora=_parse_lora_config(payload=payload),
            lm_eval=_parse_lm_eval_config(payload=payload, base_dir=base_dir),
        )


def _parse_lm_eval_config(payload: dict[str, Any], base_dir: Path) -> LmEvalConfig:
    """Parse nested `lm_eval` config payload.

    Args:
        payload: Raw top-level run config payload.
        base_dir: Parent directory for path resolution.

    Returns:
        Parsed `LmEvalConfig`.
    """
    lm_payload = payload.get("lm_eval", {})
    if isinstance(lm_payload, str):
        path = _resolve_path(base_dir=base_dir, value=str(lm_payload))
        lm_payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert isinstance(lm_payload, dict), "`lm_eval` must be a mapping or path to YAML."
    aime_avg_k = int(lm_payload.get("aime_avg_k", 32))
    assert aime_avg_k >= 1, "`lm_eval.aime_avg_k` must be >= 1."
    return LmEvalConfig(
        tasks=tuple(lm_payload.get("tasks", LmEvalConfig.tasks)),
        batch_size=lm_payload.get("batch_size", 4),
        num_fewshot=lm_payload.get("num_fewshot"),
        aime_avg_k=aime_avg_k,
        apply_chat_template=bool(lm_payload.get("apply_chat_template", True)),
        think_end_token=str(lm_payload.get("think_end_token", "</think>")),
        trust_remote_code=bool(lm_payload.get("trust_remote_code", True)),
        model_type=str(lm_payload.get("model_type", "hf")),
        tensor_parallel_size=_resolve_tensor_parallel_size(lm_payload),
        gpu_memory_utilization=float(lm_payload.get("gpu_memory_utilization", 0.9)),
        data_parallel_size=int(lm_payload.get("data_parallel_size", 1)),
        log_samples=bool(lm_payload.get("log_samples", False)),
        temperature=float(lm_payload.get("temperature", 0.6)),
        top_p=float(lm_payload.get("top_p", 0.95)),
        max_gen_toks=int(lm_payload.get("max_gen_toks", 32768)),
    )


def _parse_lora_config(payload: dict[str, Any]) -> LoraConfig | None:
    """Parse optional LoRA payload from run config.

    Args:
        payload: Raw top-level run config payload.

    Returns:
        Parsed `LoraConfig` when present, otherwise `None`.
    """
    lora_payload = payload.get("lora")
    if lora_payload is None:
        return None
    assert isinstance(lora_payload, dict), "`lora` must be a mapping."
    target_modules_raw = lora_payload.get("target_modules", LoraConfig.target_modules)
    if isinstance(target_modules_raw, str):
        target_modules = (target_modules_raw,)
    else:
        target_modules = tuple(str(name) for name in target_modules_raw)
    assert target_modules, "`lora.target_modules` must not be empty."
    rank_value = int(lora_payload.get("r", 16))
    alpha_value = int(lora_payload.get("lora_alpha", 32))
    dropout_value = float(lora_payload.get("lora_dropout", 0.05))
    bias_value_raw = str(lora_payload.get("bias", "none"))
    assert rank_value >= 1, "`lora.r` must be >= 1."
    assert alpha_value >= 1, "`lora.lora_alpha` must be >= 1."
    assert 0.0 <= dropout_value < 1.0, "`lora.lora_dropout` must be in [0, 1)."
    valid_bias_values = ("none", "all", "lora_only")
    assert bias_value_raw in valid_bias_values, "Invalid `lora.bias` value."
    bias_value = cast(Literal["none", "all", "lora_only"], bias_value_raw)
    return LoraConfig(
        r=rank_value,
        lora_alpha=alpha_value,
        lora_dropout=dropout_value,
        target_modules=target_modules,
        bias=bias_value,
        task_type=str(lora_payload.get("task_type", "CAUSAL_LM")),
    )


def _resolve_tensor_parallel_size(payload: dict[str, Any]) -> int:
    """Determine effective tensor parallel size from config or environment.

    Args:
        payload: Checkpoint of raw config payload.

    Returns:
        Explicit count or autodetected device count.
    """
    configured = int(payload.get("tensor_parallel_size", -1))
    if configured > 0:
        return configured
    if importlib.util.find_spec(name="torch") is None:
        return 1
    import torch

    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 1


def _parse_common_kwargs(payload: dict[str, Any], base_dir: Path) -> dict[str, Any]:
    """Parse non-numeric run fields and filesystem paths.

    Args:
        payload: Raw top-level run config payload.
        base_dir: Config parent directory for path resolution.

    Returns:
        Mapping of `RunConfig` keyword arguments.
    """
    return {
        "run_name": str(payload["run_name"]),
        "model_name_or_path": str(payload["model_name_or_path"]),
        "dataset_path": _resolve_path(
            base_dir=base_dir, value=str(payload["dataset_path"])
        ),
        "output_dir": _resolve_path(
            base_dir=base_dir, value=str(payload["output_dir"])
        ),
        "deepspeed_config_path": _parse_deepspeed_path(
            payload=payload, base_dir=base_dir
        ),
        "wandb_project": str(payload["wandb_project"]),
        "wandb_entity": payload.get("wandb_entity"),
    }


def _parse_deepspeed_path(payload: dict[str, Any], base_dir: Path) -> Path | None:
    """Parse optional deepspeed config path from run config.

    Args:
        payload: Raw top-level run config payload.
        base_dir: Config parent directory for path resolution.

    Returns:
        Resolved path when provided, otherwise `None`.
    """
    deepspeed_value = payload.get("deepspeed_config_path")
    if deepspeed_value is None:
        return None
    return _resolve_path(base_dir=base_dir, value=str(deepspeed_value))


def _parse_numeric_kwargs(payload: dict[str, Any]) -> dict[str, Any]:
    """Parse numeric run hyperparameters with defaults.

    Args:
        payload: Raw top-level run config payload.

    Returns:
        Mapping of `RunConfig` numeric keyword arguments.
    """
    return {
        "seed": int(payload.get("seed", 42)),
        "num_train_epochs": int(payload.get("num_train_epochs", 8)),
        "learning_rate": float(payload.get("learning_rate", 1e-5)),
        "max_seq_length": int(payload.get("max_seq_length", 8192)),
        "per_device_train_batch_size": int(
            payload.get("per_device_train_batch_size", 1)
        ),
        "per_device_eval_batch_size": int(payload.get("per_device_eval_batch_size", 1)),
        "gradient_accumulation_steps": int(
            payload.get("gradient_accumulation_steps", 2)
        ),
        "warmup_ratio": float(payload.get("warmup_ratio", 0.03)),
        "optim": str(payload.get("optim", "adamw_torch_fused")),
        "adam_beta1": float(payload.get("adam_beta1", 0.9)),
        "adam_beta2": float(payload.get("adam_beta2", 0.999)),
        "adam_epsilon": float(payload.get("adam_epsilon", 1e-8)),
        "weight_decay": float(payload.get("weight_decay", 0.0)),
        "eval_split_ratio": float(payload.get("eval_split_ratio", 0.05)),
        "save_total_limit": int(payload.get("save_total_limit", 8)),
        "save_only_model": bool(payload.get("save_only_model", False)),
    }
