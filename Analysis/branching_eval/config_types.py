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
        base_url: Optional external OpenAI-compatible vLLM base URL (`.../v1`).
        served_model_name: Optional request-time model name for external serving.
        clustering_base_url: Optional external base URL for cluster selectors.
        clustering_served_model_name: Optional request-time model for clustering.
        trigger_steer_default: Enables steer trigger for branching by default.

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
    base_url: str | None = None
    served_model_name: str | None = None
    clustering_base_url: str | None = None
    clustering_served_model_name: str | None = None
    trigger_steer_default: bool = False

    @property
    def uses_external_server(self) -> bool:
        """Return whether this spec points at a pre-existing vLLM server.

        Args:
            None.

        Returns:
            True when `base_url` is configured.
        """

        return self.base_url is not None

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
        if self.clustering_base_url is not None:
            assert (
                self.clustering_base_url.strip()
            ), "clustering_base_url must be non-empty when provided"
            assert self.clustering_base_url.rstrip("/").endswith(
                "/v1"
            ), "clustering_base_url must end with /v1"
            assert (
                self.clustering_served_model_name is not None
                and self.clustering_served_model_name.strip()
            ), "clustering_served_model_name required with clustering_base_url"
        if self.uses_external_server:
            assert (
                self.base_url is not None and self.base_url.strip()
            ), "base_url must be non-empty when provided"
            assert self.base_url.rstrip("/").endswith(
                "/v1"
            ), "base_url must end with /v1"
            assert (
                self.served_model_name is not None and self.served_model_name.strip()
            ), "served_model_name required with external base_url"
            assert (
                self.checkpoint_or_repo is None
            ), "checkpoint_or_repo must be empty with external base_url"
            assert (
                self.base_model is None
            ), "base_model must be empty with external base_url"
            assert (
                self.lora_adapter is None
            ), "lora_adapter must be empty with external base_url"
            assert (
                self.lora_name is None
            ), "lora_name must be empty with external base_url"
            return
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
        kv_offloading_size_gb: KV cache CPU offload buffer size in GiB.
        kv_offloading_backend: KV offload backend (`native` or `lmcache`).
        trust_remote_code: Forwarded model loading flag.
        max_logprobs: vLLM max logprobs engine cap.
        max_model_len: Optional vLLM context window for server startup.
        startup_timeout_seconds: Max wait for server health.
        request_timeout_seconds: Max wait for one vLLM HTTP request.
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
    kv_offloading_size_gb: float = 64.0
    kv_offloading_backend: str = "native"
    trust_remote_code: bool = True
    max_logprobs: int = 20
    max_model_len: int | None = None
    startup_timeout_seconds: float = 180.0
    request_timeout_seconds: float = 600.0
    poll_interval_seconds: float = 1.0


@dataclass(frozen=True)
class DecodingConfig:
    """Decoding settings shared across baseline and branching modes.

    Args:
        temperature: Sampling temperature.
        steer_temperature: Optional sampling temperature for generated steer
            continuations. When absent, steer requests use `temperature`.
        initial_assistant_prefix: Optional assistant text prefilled before
            generation starts, for models that expect an opening control tag.
        top_p: Nucleus sampling cutoff.
        steer_top_p: Optional nucleus sampling cutoff for generated steer
            continuations. When absent, steer requests use `top_p`.
        top_k: Optional top-k sampling cutoff.
        min_p: Optional min-p sampling cutoff.
        presence_penalty: Optional OpenAI/vLLM presence penalty.
        repetition_penalty: Optional repetition penalty for decode requests.
        max_gen_toks: Max completion token budget per rollout.
        max_model_len: Optional vLLM context window for token-id requests.
        top_logprobs: Requested top-logprob alternatives.
        decode_chunk_tokens: Tokens per decode chunk before event re-check.
        debug_assert_text_token_alignment: Enables expensive tokenizer/text
            alignment assertions for debug runs.

    Returns:
        Dataclass containing generation settings.
    """

    temperature: float = 0.6
    steer_temperature: float | None = None
    initial_assistant_prefix: str = ""
    top_p: float = 0.95
    steer_top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    max_gen_toks: int = 16384
    max_model_len: int | None = None
    top_logprobs: int = 20
    decode_chunk_tokens: int = 512
    debug_assert_text_token_alignment: bool = False

    def request_temperature(self, *, request_kind: str) -> float:
        """Resolve the sampling temperature for one vLLM request kind.

        Args:
            request_kind: Runtime request label such as `decode_chunk` or
                `steer_single_candidate`.

        Returns:
            Temperature to pass to vLLM for this request.

        Example:
            >>> cfg = DecodingConfig(temperature=0.3, steer_temperature=0.8)
            >>> cfg.request_temperature(request_kind="steer_single_candidate")
            0.8
        """

        if request_kind == "candidate_pool_steer_boundary":
            if self.steer_temperature is not None:
                return self.steer_temperature
            return 1.0
        if request_kind == "steer_single_candidate":
            if self.steer_temperature is not None:
                return self.steer_temperature
        return self.temperature

    def request_top_p(self, *, request_kind: str) -> float:
        """Resolve nucleus sampling cutoff for one vLLM request kind.

        Args:
            request_kind: Runtime request label such as `decode_chunk` or
                `steer_single_candidate`.

        Returns:
            Top-p value to pass to vLLM for this request.
        """

        if request_kind in {"candidate_pool_steer_boundary", "steer_single_candidate"}:
            if self.steer_top_p is not None:
                return self.steer_top_p
        return self.top_p


@dataclass(frozen=True)
class BranchingConfig:
    """Branching policy and limits.

    Args:
        branch_prob: Probability of branching at eligible trigger points.
        max_branch_points_per_rollout: Maximum chosen branch points on one path.
        max_concurrent_branches: Maximum in-flight decode/expansion tasks shared
            across active docs in one run.
        num_candidates: Candidate count generated per branch point.
        branch_fanout: Number of selected candidates kept per branch point.
        max_clusters: Max clusters for `cluster_across` selection.
        max_steer_tokens: Max generated tokens for steer-trigger candidates.
        steer_repetition_penalty: Repetition penalty applied to steer-token requests.
        repetition_checking_enabled: Whether repeated exec/steer block detection can
            force-close a rollout.
        epsilon_greedy_prob: Probability of one-path exploration at eligible
            triggers for `epsilon_greedy` runs.
        off_policy_min_candidates: Inclusive lower bound for verbalized options
            requested by `epsilon_greedy_off_policy`.
        off_policy_max_candidates: Inclusive upper bound for verbalized options
            requested by `epsilon_greedy_off_policy`.
        use_full_stop_strings: Whether generation requests should stop on full
            close tags such as `</steer>` and `</exec>`. Defaults to legacy
            partial-prefix stops for tokenizer-agnostic runs.

    Returns:
        Dataclass with branching behavior controls.
    """

    branch_prob: float = 0.05
    max_branch_points_per_rollout: int = 2
    max_concurrent_branches: int = 40
    num_candidates: int = 100
    branch_fanout: int = 4
    max_clusters: int = 4
    max_steer_tokens: int = 15
    steer_repetition_penalty: float = 1.01
    repetition_checking_enabled: bool = True
    epsilon_greedy_prob: float = 0.05
    off_policy_min_candidates: int = 3
    off_policy_max_candidates: int = 10
    verbalized_off_policy_enabled: bool = False
    use_full_stop_strings: bool = False


@dataclass(frozen=True)
class ArtifactConfig:
    """Artifact output configuration.

    Args:
        output_root: Root directory for run outputs.

    Returns:
        Dataclass containing output behavior.
    """

    output_root: Path = Path("output/branching_eval")


@dataclass(frozen=True)
class RunMatrixConfig:
    """Experiment matrix switches and defaults.

    Args:
        include_baselines: Enables baseline N-rollout runs.
        include_structured_baselines: Enables steer/exec-structured N-rollout runs
            without tree branching.
        baseline_rollouts: Baseline rollout count (`N`).
        include_branching: Enables branching runs.
        include_epsilon_greedy: Enables single-path epsilon-greedy runs.
        selectors: Selector modes executed for branching.
        seed_values: Seeds used for matrix expansion.
        default_limit: Optional doc limit for quick runs.
        max_concurrent_docs: Maximum in-flight docs per run.

    Returns:
        Dataclass controlling matrix expansion.
    """

    include_baselines: bool = True
    include_structured_baselines: bool = False
    baseline_rollouts: int = 16
    include_branching: bool = True
    include_epsilon_greedy: bool = False
    selectors: tuple[SelectorMode, ...] = (
        "cluster_across",
        "embed_diverse_topk_random",
        "within_cluster",
        "random",
    )
    seed_values: tuple[int, ...] = (1234,)
    default_limit: int | None = None
    max_concurrent_docs: int = 2


@dataclass(frozen=True)
class BranchingEvalConfig:
    """Root configuration for branching lm_eval experiments.

    Args:
        tasks: Task configuration.
        models: Ordered model specifications.
        serve: vLLM serving configuration.
        decoding: Shared decoding settings.
        branching: Branching policy settings.
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
        assert (
            self.serve.kv_offloading_size_gb >= 0.0
        ), "serve.kv_offloading_size_gb must be >= 0"
        assert self.serve.kv_offloading_backend in {
            "native",
            "lmcache",
        }, "serve.kv_offloading_backend must be one of {native, lmcache}"
        assert 0.0 <= self.decoding.temperature, "temperature must be >= 0"
        assert (
            self.decoding.steer_temperature is None
            or self.decoding.steer_temperature >= 0.0
        ), "steer_temperature must be >= 0 when provided"
        assert 0.0 < self.decoding.top_p <= 1.0, "top_p must be in (0, 1]"
        assert (
            self.decoding.steer_top_p is None or 0.0 < self.decoding.steer_top_p <= 1.0
        ), "steer_top_p must be in (0, 1] when provided"
        assert (
            self.decoding.top_k is None or self.decoding.top_k >= 0
        ), "top_k must be >= 0 when provided"
        assert (
            self.decoding.min_p is None or 0.0 <= self.decoding.min_p <= 1.0
        ), "min_p must be in [0, 1] when provided"
        assert (
            self.decoding.repetition_penalty is None
            or self.decoding.repetition_penalty > 0.0
        ), "repetition_penalty must be > 0 when provided"
        assert self.decoding.max_gen_toks >= 1, "max_gen_toks must be >= 1"
        assert (
            self.decoding.max_model_len is None or self.decoding.max_model_len >= 1
        ), "max_model_len must be >= 1 when provided"
        assert (
            self.decoding.decode_chunk_tokens >= 1
        ), "decode_chunk_tokens must be >= 1"
        assert self.branching.branch_fanout >= 1, "branch_fanout must be >= 1"
        assert (
            self.branching.max_branch_points_per_rollout >= 1
        ), "max_branch_points_per_rollout must be >= 1"
        assert (
            self.branching.max_concurrent_branches >= 1
        ), "max_concurrent_branches must be >= 1"
        assert (
            self.branching.num_candidates >= self.branching.branch_fanout
        ), "num_candidates must be >= branch_fanout"
        assert (
            self.branching.off_policy_min_candidates >= 1
        ), "off_policy_min_candidates must be >= 1"
        assert (
            self.branching.off_policy_max_candidates
            >= self.branching.off_policy_min_candidates
        ), "off_policy_max_candidates must be >= off_policy_min_candidates"
        if self.branching.verbalized_off_policy_enabled:
            assert (
                self.branching.branch_fanout <= self.branching.off_policy_min_candidates
            ), (
                "verbalized off-policy requires branch_fanout <= "
                "off_policy_min_candidates"
            )
        assert 0.0 <= self.branching.branch_prob <= 1.0, "branch_prob must be in [0, 1]"
        assert (
            0.0 <= self.branching.epsilon_greedy_prob <= 1.0
        ), "epsilon_greedy_prob must be in [0, 1]"
        assert (
            self.branching.steer_repetition_penalty >= 1.0
        ), "steer_repetition_penalty must be >= 1.0"
        assert self.run_matrix.baseline_rollouts >= 1, "baseline_rollouts must be >= 1"
        assert self.run_matrix.seed_values, "seed_values must be non-empty"


@dataclass(frozen=True)
class ExperimentSpec:
    """One concrete experiment from matrix expansion.

    Args:
        task_name: lm_eval task name.
        model_id: Model label from `ModelSpec`.
        mode: `baseline`, `structured_baseline`, `branching`, or `epsilon_greedy`.
        selector: Selector mode for branching runs.
        seed: RNG seed value.
        baseline_rollouts: Baseline rollout count (`N`) for baseline mode.
        trigger_steer: Enables steer trigger.

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


@dataclass(frozen=True)
class RuntimeModelConfig:
    """Resolved runtime model configuration.

    Args:
        model_spec: Source model specification.
        model_name_for_generation: Model name used in generation requests.
        base_url: OpenAI-compatible base URL for this model.
        server_port: Bound server port when locally managed, else `None`.

    Returns:
        Runtime model connection metadata.
    """

    model_spec: ModelSpec
    model_name_for_generation: str
    base_url: str
    server_port: int | None


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
                base_url=_optional_str(value=row.get("base_url")),
                served_model_name=_optional_str(value=row.get("served_model_name")),
                clustering_base_url=_optional_str(value=row.get("clustering_base_url")),
                clustering_served_model_name=_optional_str(
                    value=row.get("clustering_served_model_name")
                ),
                trigger_steer_default=bool(row.get("trigger_steer_default", False)),
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
        kv_offloading_size_gb=float(serve_payload.get("kv_offloading_size_gb", 64.0)),
        kv_offloading_backend=str(serve_payload.get("kv_offloading_backend", "native")),
        trust_remote_code=bool(serve_payload.get("trust_remote_code", True)),
        max_logprobs=int(serve_payload.get("max_logprobs", 20)),
        max_model_len=_optional_int(value=serve_payload.get("max_model_len")),
        startup_timeout_seconds=float(
            serve_payload.get("startup_timeout_seconds", 180.0)
        ),
        request_timeout_seconds=float(
            serve_payload.get("request_timeout_seconds", 600.0)
        ),
        poll_interval_seconds=float(serve_payload.get("poll_interval_seconds", 1.0)),
    )


def _parse_decoding(*, payload: dict[str, Any]) -> DecodingConfig:
    decoding_payload = payload.get("decoding", {})
    if not isinstance(decoding_payload, dict):
        return DecodingConfig()
    return DecodingConfig(
        temperature=float(decoding_payload.get("temperature", 0.6)),
        steer_temperature=_optional_float(
            value=decoding_payload.get("steer_temperature")
        ),
        initial_assistant_prefix=str(
            decoding_payload.get("initial_assistant_prefix", "")
        ),
        top_p=float(decoding_payload.get("top_p", 0.95)),
        steer_top_p=_optional_float(value=decoding_payload.get("steer_top_p")),
        top_k=_optional_int(value=decoding_payload.get("top_k")),
        min_p=_optional_float(value=decoding_payload.get("min_p")),
        presence_penalty=_optional_float(
            value=decoding_payload.get("presence_penalty")
        ),
        repetition_penalty=_optional_float(
            value=decoding_payload.get("repetition_penalty")
        ),
        max_gen_toks=int(decoding_payload.get("max_gen_toks", 16384)),
        max_model_len=_optional_int(value=decoding_payload.get("max_model_len")),
        top_logprobs=int(decoding_payload.get("top_logprobs", 20)),
        decode_chunk_tokens=int(decoding_payload.get("decode_chunk_tokens", 512)),
        debug_assert_text_token_alignment=bool(
            decoding_payload.get("debug_assert_text_token_alignment", False)
        ),
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
        max_concurrent_branches=int(branch_payload.get("max_concurrent_branches", 20)),
        num_candidates=int(branch_payload.get("num_candidates", 100)),
        branch_fanout=int(branch_payload.get("branch_fanout", 4)),
        max_clusters=int(branch_payload.get("max_clusters", 4)),
        max_steer_tokens=int(branch_payload.get("max_steer_tokens", 15)),
        steer_repetition_penalty=float(
            branch_payload.get("steer_repetition_penalty", 1.01)
        ),
        repetition_checking_enabled=_parse_bool(
            value=branch_payload.get("repetition_checking_enabled", True)
        ),
        epsilon_greedy_prob=float(branch_payload.get("epsilon_greedy_prob", 0.05)),
        off_policy_min_candidates=int(
            branch_payload.get("off_policy_min_candidates", 3)
        ),
        off_policy_max_candidates=int(
            branch_payload.get("off_policy_max_candidates", 10)
        ),
        verbalized_off_policy_enabled=_parse_bool(
            value=branch_payload.get("verbalized_off_policy_enabled", False)
        ),
        use_full_stop_strings=_parse_bool(
            value=branch_payload.get("use_full_stop_strings", False)
        ),
    )


def _parse_bool(*, value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        assert normalized in {
            "true",
            "false",
            "1",
            "0",
        }, f"Expected boolean string, got {value!r}"
        return normalized in {"true", "1"}
    return bool(value)


def _parse_artifacts(*, payload: dict[str, Any], base_dir: Path) -> ArtifactConfig:
    artifacts_payload = payload.get("artifacts", {})
    if not isinstance(artifacts_payload, dict):
        return ArtifactConfig()
    raw_root = Path(str(artifacts_payload.get("output_root", "output/branching_eval")))
    output_root = (
        raw_root if raw_root.is_absolute() else (base_dir / raw_root).resolve()
    )
    return ArtifactConfig(output_root=output_root)


def _parse_run_matrix(*, payload: dict[str, Any]) -> RunMatrixConfig:
    matrix_payload = payload.get("run_matrix", {})
    if not isinstance(matrix_payload, dict):
        return RunMatrixConfig()
    selectors = _parse_selectors(value=matrix_payload.get("selectors"))
    seed_values_raw = matrix_payload.get("seed_values", [1234])
    assert isinstance(seed_values_raw, list), "run_matrix.seed_values must be a list"
    return RunMatrixConfig(
        include_baselines=bool(matrix_payload.get("include_baselines", True)),
        include_structured_baselines=bool(
            matrix_payload.get("include_structured_baselines", False)
        ),
        baseline_rollouts=int(matrix_payload.get("baseline_rollouts", 16)),
        include_branching=bool(matrix_payload.get("include_branching", True)),
        include_epsilon_greedy=bool(
            matrix_payload.get("include_epsilon_greedy", False)
        ),
        selectors=selectors,
        seed_values=tuple(int(seed_value) for seed_value in seed_values_raw),
        default_limit=_optional_int(value=matrix_payload.get("default_limit")),
        max_concurrent_docs=int(matrix_payload.get("max_concurrent_docs", 2)),
    )


def _parse_selectors(*, value: object) -> tuple[SelectorMode, ...]:
    if value is None:
        return RunMatrixConfig.selectors
    assert isinstance(value, list), "run_matrix.selectors must be a list"
    selectors = tuple(str(item) for item in value)
    for selector in selectors:
        assert selector in {
            "cluster_across",
            "embed_diverse_topk_random",
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
