"""Typed config objects for branching lm_eval experiments.

This module defines the canonical YAML-backed configuration schema used by the
`branching_eval` framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from branching_eval.selector_types import SelectorMode


@dataclass(frozen=True)
class TaskConfig:
    """Task-level configuration.

    Args:
        task_names: Ordered lm_eval task names.

    Returns:
        Dataclass containing task selection.
    """

    task_names: tuple[str, ...] = ("aime24",)


@dataclass(frozen=True)
class ModelSpec:
    """One model serving specification.

    Args:
        model_id: Stable experiment label (`non_sft`, `sft`, etc.).
        checkpoint_or_repo: Optional checkpoint path or HF repo id.
        base_model: Optional base model id for LoRA serving.
        lora_adapter: Optional LoRA adapter directory.
        lora_name: Optional LoRA module alias.
        trigger_steer_default: Enables steer trigger for branching by default.
        trigger_entropy_default: Enables entropy trigger for branching by default.

    Returns:
        Dataclass describing one model input.

    Example:
        >>> spec = ModelSpec(model_id="non_sft", checkpoint_or_repo="Qwen/Qwen3-8B")
        >>> spec.has_lora
        False
    """

    model_id: str
    checkpoint_or_repo: str | None = None
    base_model: str | None = None
    lora_adapter: str | None = None
    lora_name: str | None = None
    trigger_steer_default: bool = False
    trigger_entropy_default: bool = True

    @property
    def has_lora(self) -> bool:
        """Return whether this spec represents a base+LoRA serving mode.

        Args:
            None.

        Returns:
            True when LoRA fields are present.
        """

        return self.base_model is not None and self.lora_adapter is not None

    def served_model_arg(self) -> str:
        """Resolve primary model argument passed to `vllm serve`.

        Args:
            None.

        Returns:
            Model string used in serve command.
        """

        if self.checkpoint_or_repo is not None:
            return self.checkpoint_or_repo
        assert self.base_model is not None, "base_model required for LoRA serving"
        return self.base_model

    def validate(self) -> None:
        """Validate model specification.

        Args:
            None.

        Returns:
            None.
        """

        assert self.model_id.strip(), "model_id must be non-empty"
        has_checkpoint = self.checkpoint_or_repo is not None
        has_lora_fields = self.base_model is not None or self.lora_adapter is not None
        assert (
            has_checkpoint or has_lora_fields
        ), "provide checkpoint_or_repo or (base_model + lora_adapter)"
        if has_checkpoint:
            assert self.base_model is None, "base_model must be empty with checkpoint"
            assert (
                self.lora_adapter is None
            ), "lora_adapter must be empty with checkpoint"
            assert self.lora_name is None, "lora_name must be empty with checkpoint"
            return
        assert self.base_model is not None, "base_model required"
        assert self.lora_adapter is not None, "lora_adapter required"
        assert (
            self.lora_name is not None and self.lora_name.strip()
        ), "lora_name required"


@dataclass(frozen=True)
class ServeConfig:
    """vLLM serving configuration.

    Args:
        host: Host used for OpenAI-compatible endpoint.
        base_port: Initial port used by auto-serving.
        tensor_parallel_size: vLLM tensor parallel size.
        gpu_memory_utilization: vLLM GPU memory utilization.
        dtype: vLLM dtype value.
        scheduling_policy: vLLM scheduler policy (`fcfs` or `priority`).
        trust_remote_code: Forwarded model loading flag.
        max_logprobs: vLLM max logprobs engine cap.
        startup_timeout_seconds: Max wait for server health.
        poll_interval_seconds: Health-check poll interval.

    Returns:
        Dataclass containing serving defaults.
    """

    host: str = "127.0.0.1"
    base_port: int = 8020
    tensor_parallel_size: int = 2
    gpu_memory_utilization: float = 0.9
    dtype: str = "auto"
    scheduling_policy: str = "priority"
    trust_remote_code: bool = True
    max_logprobs: int = 20
    startup_timeout_seconds: float = 180.0
    poll_interval_seconds: float = 1.0


@dataclass(frozen=True)
class DecodingConfig:
    """Decoding settings shared across baseline and branching modes.

    Args:
        temperature: Sampling temperature.
        top_p: Nucleus sampling cutoff.
        max_gen_toks: Max completion token budget per rollout.
        top_logprobs: Requested top-logprob alternatives.
        decode_chunk_tokens: Tokens per decode chunk before event re-check.

    Returns:
        Dataclass containing generation settings.
    """

    temperature: float = 0.6
    top_p: float = 0.95
    max_gen_toks: int = 16384
    top_logprobs: int = 20
    decode_chunk_tokens: int = 512


@dataclass(frozen=True)
class BranchingConfig:
    """Branching policy and limits.

    Args:
        branch_prob: Probability of branching at eligible trigger points.
        max_branch_points_per_rollout: Maximum chosen branch points on one path.
        num_candidates: Candidate count generated per branch point.
        branch_fanout: Number of selected candidates kept per branch point.
        max_clusters: Max clusters for `cluster_across` selection.
        candidate_span_tokens: Span used for entropy-trigger candidate generation.
        max_steer_tokens: Max generated tokens for steer-trigger candidates.
        entropy_threshold: Optional explicit entropy threshold override.
        entropy_profile_name: Calibration profile key.

    Returns:
        Dataclass with branching behavior controls.
    """

    branch_prob: float = 0.05
    max_branch_points_per_rollout: int = 2
    num_candidates: int = 100
    branch_fanout: int = 4
    max_clusters: int = 4
    candidate_span_tokens: int = 15
    max_steer_tokens: int = 15
    entropy_threshold: float | None = None
    entropy_profile_name: str = "aime24_default"


@dataclass(frozen=True)
class CalibrationConfig:
    """Entropy calibration lookup configuration.

    Args:
        entropy_threshold_path: JSON file with calibrated thresholds.

    Returns:
        Dataclass describing threshold source.
    """

    entropy_threshold_path: Path = Path(
        "branching_eval/calibration/entropy_thresholds.json"
    )


@dataclass(frozen=True)
class ArtifactConfig:
    """Artifact output configuration.

    Args:
        output_root: Root directory for run outputs.
        reuse_candidate_pools: Reuse persisted candidate pools across reruns.

    Returns:
        Dataclass containing output behavior.
    """

    output_root: Path = Path("output/branching_eval")
    reuse_candidate_pools: bool = True


@dataclass(frozen=True)
class RunMatrixConfig:
    """Experiment matrix switches and defaults.

    Args:
        include_baselines: Enables baseline N-rollout runs.
        baseline_rollouts: Baseline rollout count (`N`).
        include_branching: Enables branching runs.
        selectors: Selector modes executed for branching.
        seed_values: Seeds used for matrix expansion.
        default_limit: Optional doc limit for quick runs.

    Returns:
        Dataclass controlling matrix expansion.
    """

    include_baselines: bool = True
    baseline_rollouts: int = 16
    include_branching: bool = True
    selectors: tuple[SelectorMode, ...] = (
        "cluster_across",
        "embed_diverse",
        "within_cluster",
        "random",
    )
    seed_values: tuple[int, ...] = (1234,)
    default_limit: int | None = None


@dataclass(frozen=True)
class BranchingEvalConfig:
    """Root configuration for branching lm_eval experiments.

    Args:
        tasks: Task configuration.
        models: Ordered model specifications.
        serve: vLLM serving configuration.
        decoding: Shared decoding settings.
        branching: Branching policy settings.
        calibration: Entropy threshold source config.
        artifacts: Artifact output settings.
        run_matrix: Matrix orchestration settings.

    Returns:
        Fully validated branching evaluation configuration.

    Example:
        >>> cfg = BranchingEvalConfig(models=(ModelSpec(model_id="non_sft", checkpoint_or_repo="Qwen/Qwen3-8B"),))
        >>> cfg.run_matrix.baseline_rollouts
        16
    """

    tasks: TaskConfig = field(default_factory=TaskConfig)
    models: tuple[ModelSpec, ...] = field(default_factory=tuple)
    serve: ServeConfig = field(default_factory=ServeConfig)
    decoding: DecodingConfig = field(default_factory=DecodingConfig)
    branching: BranchingConfig = field(default_factory=BranchingConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    artifacts: ArtifactConfig = field(default_factory=ArtifactConfig)
    run_matrix: RunMatrixConfig = field(default_factory=RunMatrixConfig)

    def validate(self) -> None:
        """Validate root configuration.

        Args:
            None.

        Returns:
            None.
        """

        assert self.models, "at least one model spec is required"
        for model_spec in self.models:
            model_spec.validate()
        assert self.serve.base_port > 0, "base_port must be positive"
        assert self.serve.tensor_parallel_size >= 1, "tensor_parallel_size must be >= 1"
        assert self.serve.max_logprobs >= 0, "max_logprobs must be >= 0"
        assert self.serve.scheduling_policy in {
            "fcfs",
            "priority",
        }, "serve.scheduling_policy must be one of {fcfs, priority}"
        assert 0.0 <= self.decoding.temperature, "temperature must be >= 0"
        assert 0.0 < self.decoding.top_p <= 1.0, "top_p must be in (0, 1]"
        assert self.decoding.max_gen_toks >= 1, "max_gen_toks must be >= 1"
        assert (
            self.decoding.decode_chunk_tokens >= 1
        ), "decode_chunk_tokens must be >= 1"
        assert self.branching.branch_fanout >= 1, "branch_fanout must be >= 1"
        assert (
            self.branching.max_branch_points_per_rollout >= 1
        ), "max_branch_points_per_rollout must be >= 1"
        assert (
            self.branching.num_candidates >= self.branching.branch_fanout
        ), "num_candidates must be >= branch_fanout"
        assert 0.0 <= self.branching.branch_prob <= 1.0, "branch_prob must be in [0, 1]"
        assert self.run_matrix.baseline_rollouts >= 1, "baseline_rollouts must be >= 1"
        assert self.run_matrix.seed_values, "seed_values must be non-empty"


@dataclass(frozen=True)
class ExperimentSpec:
    """One concrete experiment from matrix expansion.

    Args:
        task_name: lm_eval task name.
        model_id: Model label from `ModelSpec`.
        mode: `baseline` or `branching`.
        selector: Selector mode for branching runs.
        seed: RNG seed value.
        baseline_rollouts: Baseline rollout count (`N`) for baseline mode.
        trigger_steer: Enables steer trigger.
        trigger_entropy: Enables entropy trigger.

    Returns:
        Concrete experiment run specification.
    """

    task_name: str
    model_id: str
    mode: str
    selector: SelectorMode | None
    seed: int
    baseline_rollouts: int
    trigger_steer: bool
    trigger_entropy: bool


@dataclass(frozen=True)
class RuntimeModelConfig:
    """Resolved runtime model configuration.

    Args:
        model_spec: Source model specification.
        model_name_for_generation: Model name used in generation requests.
        base_url: OpenAI-compatible base URL for this model.
        server_port: Bound server port.

    Returns:
        Runtime model connection metadata.
    """

    model_spec: ModelSpec
    model_name_for_generation: str
    base_url: str
    server_port: int


def load_branching_eval_config(*, config_path: Path) -> BranchingEvalConfig:
    """Load one YAML config file into a typed branching-eval config.

    Args:
        config_path: YAML configuration path.

    Returns:
        Parsed and validated `BranchingEvalConfig`.

    Example:
        >>> _ = load_branching_eval_config(config_path=Path('missing.yaml'))  # doctest: +SKIP
    """

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "config payload must be a mapping"
    base_dir = config_path.resolve().parent
    config = BranchingEvalConfig(
        tasks=_parse_tasks(payload=payload),
        models=_parse_models(payload=payload),
        serve=_parse_serve(payload=payload),
        decoding=_parse_decoding(payload=payload),
        branching=_parse_branching(payload=payload),
        calibration=_parse_calibration(payload=payload, base_dir=base_dir),
        artifacts=_parse_artifacts(payload=payload, base_dir=base_dir),
        run_matrix=_parse_run_matrix(payload=payload),
    )
    config.validate()
    return config


def _parse_tasks(*, payload: dict[str, Any]) -> TaskConfig:
    task_payload = payload.get("tasks", {})
    if isinstance(task_payload, list):
        return TaskConfig(task_names=tuple(str(item) for item in task_payload))
    if not isinstance(task_payload, dict):
        return TaskConfig()
    task_names = task_payload.get("task_names")
    if task_names is None:
        return TaskConfig()
    assert isinstance(task_names, list), "tasks.task_names must be a list"
    return TaskConfig(task_names=tuple(str(item) for item in task_names))


def _parse_models(*, payload: dict[str, Any]) -> tuple[ModelSpec, ...]:
    model_rows = payload.get("models", [])
    assert isinstance(model_rows, list), "models must be a list"
    parsed: list[ModelSpec] = []
    for row in model_rows:
        assert isinstance(row, dict), "each model spec must be a mapping"
        parsed.append(
            ModelSpec(
                model_id=str(row.get("model_id", "")),
                checkpoint_or_repo=_optional_str(value=row.get("checkpoint_or_repo")),
                base_model=_optional_str(value=row.get("base_model")),
                lora_adapter=_optional_str(value=row.get("lora_adapter")),
                lora_name=_optional_str(value=row.get("lora_name")),
                trigger_steer_default=bool(row.get("trigger_steer_default", False)),
                trigger_entropy_default=bool(row.get("trigger_entropy_default", True)),
            )
        )
    return tuple(parsed)


def _parse_serve(*, payload: dict[str, Any]) -> ServeConfig:
    serve_payload = payload.get("serve", {})
    if not isinstance(serve_payload, dict):
        return ServeConfig()
    return ServeConfig(
        host=str(serve_payload.get("host", "127.0.0.1")),
        base_port=int(serve_payload.get("base_port", 8020)),
        tensor_parallel_size=int(serve_payload.get("tensor_parallel_size", 2)),
        gpu_memory_utilization=float(serve_payload.get("gpu_memory_utilization", 0.9)),
        dtype=str(serve_payload.get("dtype", "auto")),
        scheduling_policy=str(serve_payload.get("scheduling_policy", "priority")),
        trust_remote_code=bool(serve_payload.get("trust_remote_code", True)),
        max_logprobs=int(serve_payload.get("max_logprobs", 20)),
        startup_timeout_seconds=float(
            serve_payload.get("startup_timeout_seconds", 180.0)
        ),
        poll_interval_seconds=float(serve_payload.get("poll_interval_seconds", 1.0)),
    )


def _parse_decoding(*, payload: dict[str, Any]) -> DecodingConfig:
    decoding_payload = payload.get("decoding", {})
    if not isinstance(decoding_payload, dict):
        return DecodingConfig()
    return DecodingConfig(
        temperature=float(decoding_payload.get("temperature", 0.6)),
        top_p=float(decoding_payload.get("top_p", 0.95)),
        max_gen_toks=int(decoding_payload.get("max_gen_toks", 16384)),
        top_logprobs=int(decoding_payload.get("top_logprobs", 20)),
        decode_chunk_tokens=int(decoding_payload.get("decode_chunk_tokens", 512)),
    )


def _parse_branching(*, payload: dict[str, Any]) -> BranchingConfig:
    branch_payload = payload.get("branching", {})
    if not isinstance(branch_payload, dict):
        return BranchingConfig()
    return BranchingConfig(
        branch_prob=float(branch_payload.get("branch_prob", 0.05)),
        max_branch_points_per_rollout=int(
            branch_payload.get("max_branch_points_per_rollout", 2)
        ),
        num_candidates=int(branch_payload.get("num_candidates", 100)),
        branch_fanout=int(branch_payload.get("branch_fanout", 4)),
        max_clusters=int(branch_payload.get("max_clusters", 4)),
        candidate_span_tokens=int(branch_payload.get("candidate_span_tokens", 15)),
        max_steer_tokens=int(branch_payload.get("max_steer_tokens", 15)),
        entropy_threshold=_optional_float(
            value=branch_payload.get("entropy_threshold")
        ),
        entropy_profile_name=str(
            branch_payload.get("entropy_profile_name", "aime24_default")
        ),
    )


def _parse_calibration(*, payload: dict[str, Any], base_dir: Path) -> CalibrationConfig:
    calibration_payload = payload.get("calibration", {})
    if not isinstance(calibration_payload, dict):
        return CalibrationConfig()
    raw_path = Path(
        str(
            calibration_payload.get(
                "entropy_threshold_path",
                "branching_eval/calibration/entropy_thresholds.json",
            )
        )
    )
    resolved_path = (
        raw_path if raw_path.is_absolute() else (base_dir / raw_path).resolve()
    )
    return CalibrationConfig(entropy_threshold_path=resolved_path)


def _parse_artifacts(*, payload: dict[str, Any], base_dir: Path) -> ArtifactConfig:
    artifacts_payload = payload.get("artifacts", {})
    if not isinstance(artifacts_payload, dict):
        return ArtifactConfig()
    raw_root = Path(str(artifacts_payload.get("output_root", "output/branching_eval")))
    output_root = (
        raw_root if raw_root.is_absolute() else (base_dir / raw_root).resolve()
    )
    return ArtifactConfig(
        output_root=output_root,
        reuse_candidate_pools=bool(
            artifacts_payload.get("reuse_candidate_pools", True)
        ),
    )


def _parse_run_matrix(*, payload: dict[str, Any]) -> RunMatrixConfig:
    matrix_payload = payload.get("run_matrix", {})
    if not isinstance(matrix_payload, dict):
        return RunMatrixConfig()
    selectors = _parse_selectors(value=matrix_payload.get("selectors"))
    seed_values_raw = matrix_payload.get("seed_values", [1234])
    assert isinstance(seed_values_raw, list), "run_matrix.seed_values must be a list"
    return RunMatrixConfig(
        include_baselines=bool(matrix_payload.get("include_baselines", True)),
        baseline_rollouts=int(matrix_payload.get("baseline_rollouts", 16)),
        include_branching=bool(matrix_payload.get("include_branching", True)),
        selectors=selectors,
        seed_values=tuple(int(seed_value) for seed_value in seed_values_raw),
        default_limit=_optional_int(value=matrix_payload.get("default_limit")),
    )


def _parse_selectors(*, value: object) -> tuple[SelectorMode, ...]:
    if value is None:
        return RunMatrixConfig.selectors
    assert isinstance(value, list), "run_matrix.selectors must be a list"
    selectors = tuple(str(item) for item in value)
    for selector in selectors:
        assert selector in {
            "cluster_across",
            "embed_diverse",
            "within_cluster",
            "random",
        }
    return selectors  # type: ignore[return-value]


def _optional_str(*, value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_int(*, value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def _optional_float(*, value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return float(stripped)
    return None
