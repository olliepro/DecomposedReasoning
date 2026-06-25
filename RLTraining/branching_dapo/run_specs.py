"""Typed launch specifications for branching-DAPO RL runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

DEFAULT_SCRATCH_ROOT = Path("/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning")
DEFAULT_DATASET_CACHE_ROOT = (
    DEFAULT_SCRATCH_ROOT / "Analysis/cache/datasets/submissions"
)
DEFAULT_QWEN35_MODEL_PATH = (
    DEFAULT_SCRATCH_ROOT
    / "SFTTraining/vllm_exports/qwen35_2b_lr7e6_checkpoint-480_official_layout"
)


def env_value(env: Mapping[str, str], name: str, default: str) -> str:
    """Return a non-empty environment value or its default."""

    value = env.get(name, "")
    return value if value else default


def optional_env_value(env: Mapping[str, str], name: str) -> str:
    """Return an environment value while preserving empty-as-empty semantics."""

    return env.get(name, "")


@dataclass(frozen=True)
class SlurmResources:
    """Slurm resource request for one submitted RL job."""

    partition: str
    gpu_count: str
    gpu_type: str
    gpu_gres_override: str
    cpu_count: str
    memory: str
    time_limit: str
    exclusive: bool
    exclude_nodes: str
    sbatch_script: Path

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, rltraining_dir: Path
    ) -> "SlurmResources":
        """Build a Slurm resource spec from operator environment variables."""

        return cls(
            partition=env_value(env, "PARTITION", "preemptible-quad"),
            gpu_count=env_value(env, "GPU_COUNT", "4"),
            gpu_type=env_value(env, "GPU_TYPE", "a100"),
            gpu_gres_override=optional_env_value(env, "GPU_GRES"),
            cpu_count=env_value(env, "CPU_COUNT", "48"),
            memory=env_value(env, "MEMORY", "256G"),
            time_limit=env_value(env, "TIME_LIMIT", "02:00:00"),
            exclusive=env_value(env, "EXCLUSIVE", "false") == "true",
            exclude_nodes=optional_env_value(env, "EXCLUDE_NODES"),
            sbatch_script=Path(
                env_value(
                    env,
                    "SBATCH_SCRIPT",
                    str(rltraining_dir / "slurm/branching_dapo_qwen35_smoke.sbatch"),
                )
            ),
        )

    def gpu_gres(self) -> str:
        """Return the `--gres` value passed to sbatch."""

        if self.gpu_gres_override:
            return self.gpu_gres_override
        if self.gpu_type:
            return f"gpu:{self.gpu_type}:{self.gpu_count}"
        return f"gpu:{self.gpu_count}"

    def sbatch_args(self, *, job_name: str, test_only: bool) -> list[str]:
        """Return sbatch CLI arguments for this resource request."""

        args = [
            "sbatch",
            "--parsable",
            f"--partition={self.partition}",
            f"--gres={self.gpu_gres()}",
            f"--cpus-per-task={self.cpu_count}",
            f"--mem={self.memory}",
            f"--time={self.time_limit}",
            f"--job-name={job_name}",
            "--export=ALL",
        ]
        if self.exclusive:
            args.append("--exclusive")
        if self.exclude_nodes:
            args.append(f"--exclude={self.exclude_nodes}")
        if test_only:
            args.append("--test-only")
        args.append(str(self.sbatch_script))
        return args


@dataclass(frozen=True)
class TrainingShape:
    """Batch, sequence, and optimization shape shared by matrix modes."""

    train_prompt_bsz: str
    train_prompt_mini_bsz: str
    train_prompt_micro_bsz_per_gpu: str
    max_prompt_length: str
    max_response_length: str
    max_model_len: str
    max_steer_tokens: str
    actor_lr: str
    total_training_steps: str
    save_freq: str
    clip_ratio_low: str
    clip_ratio_high: str
    branching_alpha: str

    @classmethod
    def from_env(cls, env: Mapping[str, str], *, gpu_count: str) -> "TrainingShape":
        """Build the common training shape from environment variables."""

        max_prompt_length = env_value(env, "MAX_PROMPT_LENGTH", "1024")
        max_response_length = env_value(env, "MAX_RESPONSE_LENGTH", "32768")
        max_model_len = env_value(
            env,
            "MAX_MODEL_LEN",
            str(int(max_prompt_length) + int(max_response_length)),
        )
        return cls(
            train_prompt_bsz=env_value(env, "TRAIN_PROMPT_BSZ", gpu_count),
            train_prompt_mini_bsz=env_value(env, "TRAIN_PROMPT_MINI_BSZ", gpu_count),
            train_prompt_micro_bsz_per_gpu=env_value(
                env, "TRAIN_PROMPT_MICRO_BSZ_PER_GPU", "1"
            ),
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            max_model_len=max_model_len,
            max_steer_tokens=env_value(env, "MAX_STEER_TOKENS", "15"),
            actor_lr=env_value(env, "ACTOR_LR", "1e-6"),
            total_training_steps=env_value(env, "TOTAL_TRAINING_STEPS", "2"),
            save_freq=env_value(env, "SAVE_FREQ", "2"),
            clip_ratio_low=env_value(env, "CLIP_RATIO_LOW", "0.2"),
            clip_ratio_high=env_value(env, "CLIP_RATIO_HIGH", "0.28"),
            branching_alpha=env_value(env, "BRANCHING_ALPHA", "0.5"),
        )


@dataclass(frozen=True)
class DecodeSettings:
    """Sampling and vLLM settings shared by matrix modes."""

    exec_temperature: str
    steer_temperature: str
    exec_top_p: str
    steer_top_p: str
    presence_penalty: str
    repetition_penalty: str
    steer_repetition_penalty: str
    repetition_checking_enabled: str
    rollout_top_logprobs: str
    rollout_gpu_memory_utilization: str
    rollout_max_num_batched_tokens: str
    rollout_max_num_seqs: str
    rollout_gdn_prefill_backend: str
    rollout_disable_custom_all_reduce: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "DecodeSettings":
        """Build common decode settings from the operator environment."""

        repetition_penalty = env_value(env, "REPETITION_PENALTY", "1.0")
        return cls(
            exec_temperature=env_value(env, "EXEC_TEMPERATURE", "0.7"),
            steer_temperature=env_value(env, "STEER_TEMPERATURE", "1.0"),
            exec_top_p=env_value(env, "EXEC_TOP_P", "0.95"),
            steer_top_p=env_value(env, "STEER_TOP_P", "0.95"),
            presence_penalty=env_value(env, "PRESENCE_PENALTY", "1.5"),
            repetition_penalty=repetition_penalty,
            steer_repetition_penalty=env_value(
                env, "STEER_REPETITION_PENALTY", repetition_penalty
            ),
            repetition_checking_enabled=env_value(
                env, "REPETITION_CHECKING_ENABLED", "True"
            ),
            rollout_top_logprobs=env_value(env, "ROLLOUT_TOP_LOGPROBS", "1"),
            rollout_gpu_memory_utilization=env_value(
                env, "ROLLOUT_GPU_MEMORY_UTILIZATION", "0.80"
            ),
            rollout_max_num_batched_tokens=env_value(
                env, "ROLLOUT_MAX_NUM_BATCHED_TOKENS", "420000"
            ),
            rollout_max_num_seqs=env_value(env, "ROLLOUT_MAX_NUM_SEQS", "4096"),
            rollout_gdn_prefill_backend=optional_env_value(
                env, "ROLLOUT_GDN_PREFILL_BACKEND"
            ),
            rollout_disable_custom_all_reduce=env_value(
                env, "ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE", "false"
            ),
        )


@dataclass(frozen=True)
class ActorRuntimeSettings:
    """Actor/model runtime switches passed through to the Qwen launcher."""

    use_remove_padding: str
    use_fused_kernels: str
    fused_kernel_backend: str
    attn_implementation: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "ActorRuntimeSettings":
        """Build actor runtime switches from environment variables."""

        return cls(
            use_remove_padding=env_value(env, "ACTOR_USE_REMOVE_PADDING", "true"),
            use_fused_kernels=env_value(env, "ACTOR_USE_FUSED_KERNELS", "true"),
            fused_kernel_backend=env_value(env, "ACTOR_FUSED_KERNEL_BACKEND", "triton"),
            attn_implementation=env_value(env, "ACTOR_ATTN_IMPLEMENTATION", "sdpa"),
        )


@dataclass(frozen=True)
class ArtifactRetention:
    """Branching artifact retention policy for completed RL steps."""

    persistent_log_interval_steps: str

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "ArtifactRetention":
        """Build artifact retention settings from environment variables."""

        return cls(
            persistent_log_interval_steps=env_value(
                env, "PERSISTENT_LOG_INTERVAL_STEPS", "10"
            )
        )


@dataclass(frozen=True)
class RunIdentity:
    """Naming and data-source values shared by the matrix."""

    run_label: str
    trainer_loggers: str
    project_name: str
    experiment_name: str
    train_file: str
    val_file: str
    reward_require_steer_exec: str
    topology_policy: str
    submission_id: str
    hf_datasets_cache_root: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "RunIdentity":
        """Build identity and dataset settings from environment variables."""

        return cls(
            run_label=env_value(
                env, "RUN_LABEL", env_value(env, "SMOKE_RUN_LABEL", "smoke")
            ),
            trainer_loggers=env_value(env, "TRAINER_LOGGERS", '["console","wandb"]'),
            project_name=env_value(env, "PROJECT_NAME", "branching_dapo_qwen35_smoke"),
            experiment_name=optional_env_value(env, "EXPERIMENT_NAME"),
            train_file=optional_env_value(env, "TRAIN_FILE"),
            val_file=optional_env_value(env, "VAL_FILE"),
            reward_require_steer_exec=env_value(
                env, "REWARD_REQUIRE_STEER_EXEC", "True"
            ),
            topology_policy=env_value(env, "TOPOLOGY_POLICY", "auto"),
            submission_id=env_value(env, "SUBMISSION_ID", default_submission_id()),
            hf_datasets_cache_root=Path(
                env_value(
                    env, "HF_DATASETS_CACHE_ROOT", str(DEFAULT_DATASET_CACHE_ROOT)
                )
            ),
        )


@dataclass(frozen=True)
class ModeSpec:
    """Resolved rollout-mode settings for one matrix job."""

    mode: str
    selector_mode: str
    branch_prob: str
    branch_fanout: str
    max_branch_points: str
    num_candidates: str
    epsilon_greedy_prob: str
    n_resp_per_prompt: str
    algorithm_adv_estimator: str
    rollout_gpu_memory_utilization: str
    use_full_stop_strings: str
    initial_assistant_prefix: str
    off_policy_min_candidates: str = "3"
    off_policy_max_candidates: str = "10"

    @property
    def mode_slug(self) -> str:
        """Return a shell-safe mode slug used in job names."""

        return self.mode.replace("_", "-")


@dataclass(frozen=True)
class Qwen35MatrixSpec:
    """Resolved Qwen3.5 branching-DAPO matrix submission spec."""

    modes: tuple[str, ...]
    resources: SlurmResources
    shape: TrainingShape
    decode: DecodeSettings
    actor_runtime: ActorRuntimeSettings
    artifact_retention: ArtifactRetention
    identity: RunIdentity
    rltraining_dir: Path
    base_env: Mapping[str, str]

    @classmethod
    def from_env(
        cls, env: Mapping[str, str], *, rltraining_dir: Path
    ) -> "Qwen35MatrixSpec":
        """Build a complete Qwen3.5 matrix spec from environment variables."""

        resources = SlurmResources.from_env(env=env, rltraining_dir=rltraining_dir)
        return cls(
            modes=tuple(env_value(env, "SMOKE_ROLLOUT_MODES", default_modes()).split()),
            resources=resources,
            shape=TrainingShape.from_env(env=env, gpu_count=resources.gpu_count),
            decode=DecodeSettings.from_env(env=env),
            actor_runtime=ActorRuntimeSettings.from_env(env=env),
            artifact_retention=ArtifactRetention.from_env(env=env),
            identity=RunIdentity.from_env(env),
            rltraining_dir=rltraining_dir,
            base_env=env,
        )

    def mode_spec(self, mode: str) -> ModeSpec:
        """Return resolved rollout settings for one requested mode."""

        if mode == "branching":
            return self._branching_mode()
        if mode in {"baseline", "no_branching", "structured_baseline"}:
            return self._baseline_mode(mode=mode)
        if mode == "epsilon_greedy":
            return self._epsilon_mode()
        if mode == "epsilon_greedy_off_policy":
            return self._epsilon_off_policy_mode()
        raise ValueError(f"Unsupported mode: {mode}")

    def sbatch_env_for_mode(
        self, mode_spec: ModeSpec, *, run_spec_json: Path
    ) -> dict[str, str]:
        """Return exact environment variables added for one sbatch call."""

        env = {
            "SMOKE_ROLLOUT_MODE": mode_spec.mode,
            "SMOKE_RUN_LABEL": self.identity.run_label,
            "MODEL_NAME_SLUG": env_value(self.base_env, "MODEL_NAME_SLUG", "qwen35_2b"),
            "MODEL_PATH": env_value(
                self.base_env, "MODEL_PATH", str(DEFAULT_QWEN35_MODEL_PATH)
            ),
            "CACHE_ROOT": env_value(
                self.base_env,
                "CACHE_ROOT",
                str(DEFAULT_SCRATCH_ROOT / "RLTraining/qwen35_branching_dapo"),
            ),
            "NGPUS_PER_NODE": self.resources.gpu_count,
            "GEN_TP": self.resources.gpu_count,
            "FSDP_SIZE": self.resources.gpu_count,
            "RAY_NUM_CPUS": self.resources.cpu_count,
            "TRAIN_PROMPT_BSZ": self.shape.train_prompt_bsz,
            "TRAIN_PROMPT_MINI_BSZ": self.shape.train_prompt_mini_bsz,
            "TRAIN_PROMPT_MICRO_BSZ_PER_GPU": self.shape.train_prompt_micro_bsz_per_gpu,
            "N_RESP_PER_PROMPT": mode_spec.n_resp_per_prompt,
            "MAX_PROMPT_LENGTH": self.shape.max_prompt_length,
            "MAX_RESPONSE_LENGTH": self.shape.max_response_length,
            "MAX_MODEL_LEN": self.shape.max_model_len,
            "MAX_STEER_TOKENS": self.shape.max_steer_tokens,
            "ACTOR_LR": self.shape.actor_lr,
            "TOTAL_TRAINING_STEPS": self.shape.total_training_steps,
            "SAVE_FREQ": self.shape.save_freq,
            "CLIP_RATIO_LOW": self.shape.clip_ratio_low,
            "CLIP_RATIO_HIGH": self.shape.clip_ratio_high,
            "BRANCHING_ALPHA": self.shape.branching_alpha,
            "HF_DATASETS_CACHE": self._dataset_cache(mode_spec=mode_spec),
            "ALGORITHM_ADV_ESTIMATOR": mode_spec.algorithm_adv_estimator,
            "PROJECT_NAME": self.identity.project_name,
            "EXPERIMENT_NAME": self.identity.experiment_name,
            "TRAIN_FILE": self.identity.train_file,
            "VAL_FILE": self.identity.val_file,
            "REWARD_REQUIRE_STEER_EXEC": self.identity.reward_require_steer_exec,
            "TOPOLOGY_POLICY": self.identity.topology_policy,
            "SELECTOR_MODE": mode_spec.selector_mode,
            "BRANCH_PROB": mode_spec.branch_prob,
            "BRANCH_FANOUT": mode_spec.branch_fanout,
            "MAX_BRANCH_POINTS_PER_ROLLOUT": mode_spec.max_branch_points,
            "NUM_CANDIDATES": mode_spec.num_candidates,
            "OFF_POLICY_MIN_CANDIDATES": mode_spec.off_policy_min_candidates,
            "OFF_POLICY_MAX_CANDIDATES": mode_spec.off_policy_max_candidates,
            "EPSILON_GREEDY_PROB": mode_spec.epsilon_greedy_prob,
            "INITIAL_ASSISTANT_PREFIX": mode_spec.initial_assistant_prefix,
            "PERSISTENT_LOG_INTERVAL_STEPS": (
                self.artifact_retention.persistent_log_interval_steps
            ),
            "RUN_SPEC_JSON": str(run_spec_json),
        }
        env.update(self._decode_env(mode_spec=mode_spec))
        env.update(self._actor_runtime_env())
        return env

    def _branching_mode(self) -> ModeSpec:
        env = self.base_env
        return ModeSpec(
            mode="branching",
            selector_mode=env_value(
                env,
                "BRANCHING_SELECTOR_MODE",
                env_value(env, "SELECTOR_MODE", "cluster_across"),
            ),
            branch_prob=env_value(
                env, "BRANCHING_BRANCH_PROB", env_value(env, "BRANCH_PROB", "0.10")
            ),
            branch_fanout=env_value(
                env, "BRANCHING_BRANCH_FANOUT", env_value(env, "BRANCH_FANOUT", "2")
            ),
            max_branch_points=env_value(
                env,
                "BRANCHING_MAX_BRANCH_POINTS_PER_ROLLOUT",
                env_value(env, "MAX_BRANCH_POINTS_PER_ROLLOUT", "4"),
            ),
            num_candidates=env_value(
                env, "BRANCHING_NUM_CANDIDATES", env_value(env, "NUM_CANDIDATES", "50")
            ),
            epsilon_greedy_prob=env_value(
                env,
                "BRANCHING_EPSILON_GREEDY_PROB",
                env_value(env, "EPSILON_GREEDY_PROB", "0.1"),
            ),
            n_resp_per_prompt=env_value(env, "N_RESP_PER_PROMPT", "1"),
            algorithm_adv_estimator=env_value(
                env, "ALGORITHM_ADV_ESTIMATOR", "branch_interpolated_grpo"
            ),
            rollout_gpu_memory_utilization=env_value(
                env,
                "BRANCHING_ROLLOUT_GPU_MEMORY_UTILIZATION",
                self.decode.rollout_gpu_memory_utilization,
            ),
            use_full_stop_strings=env_value(env, "USE_FULL_STOP_STRINGS", "False"),
            initial_assistant_prefix=env_value(env, "INITIAL_ASSISTANT_PREFIX", ""),
        )

    def _baseline_mode(self, *, mode: str) -> ModeSpec:
        env = self.base_env
        return ModeSpec(
            mode=mode,
            selector_mode=env_value(env, "BASELINE_SELECTOR_MODE", "random"),
            branch_prob="0.0",
            branch_fanout=env_value(env, "BRANCH_FANOUT", "2"),
            max_branch_points=env_value(env, "MAX_BRANCH_POINTS_PER_ROLLOUT", "1"),
            num_candidates=env_value(env, "NUM_CANDIDATES", "4"),
            epsilon_greedy_prob="0.0",
            n_resp_per_prompt=env_value(env, "N_RESP_PER_PROMPT", "16"),
            algorithm_adv_estimator=env_value(env, "ALGORITHM_ADV_ESTIMATOR", "grpo"),
            rollout_gpu_memory_utilization=env_value(
                env,
                "BASELINE_ROLLOUT_GPU_MEMORY_UTILIZATION",
                self.decode.rollout_gpu_memory_utilization,
            ),
            use_full_stop_strings=env_value(env, "USE_FULL_STOP_STRINGS", "False"),
            initial_assistant_prefix=env_value(env, "INITIAL_ASSISTANT_PREFIX", ""),
        )

    def _epsilon_mode(self) -> ModeSpec:
        env = self.base_env
        return ModeSpec(
            mode="epsilon_greedy",
            selector_mode=env_value(
                env, "EPSILON_SELECTOR_MODE", "embed_diverse_topk_random"
            ),
            branch_prob="0.0",
            branch_fanout=env_value(env, "EPSILON_BRANCH_FANOUT", "1"),
            max_branch_points=env_value(
                env,
                "EPSILON_MAX_BRANCH_POINTS_PER_ROLLOUT",
                env_value(env, "MAX_BRANCH_POINTS_PER_ROLLOUT", "1"),
            ),
            num_candidates=env_value(
                env, "EPSILON_NUM_CANDIDATES", env_value(env, "NUM_CANDIDATES", "50")
            ),
            epsilon_greedy_prob=env_value(env, "EPSILON_GREEDY_PROB", "0.1"),
            n_resp_per_prompt=env_value(env, "N_RESP_PER_PROMPT", "16"),
            algorithm_adv_estimator=env_value(env, "ALGORITHM_ADV_ESTIMATOR", "grpo"),
            rollout_gpu_memory_utilization=env_value(
                env,
                "EPSILON_ROLLOUT_GPU_MEMORY_UTILIZATION",
                self.decode.rollout_gpu_memory_utilization,
            ),
            use_full_stop_strings=env_value(env, "USE_FULL_STOP_STRINGS", "False"),
            initial_assistant_prefix=env_value(env, "INITIAL_ASSISTANT_PREFIX", ""),
        )

    def _epsilon_off_policy_mode(self) -> ModeSpec:
        env = self.base_env
        min_candidates = env_value(env, "OFF_POLICY_MIN_CANDIDATES", "3")
        max_candidates = env_value(env, "OFF_POLICY_MAX_CANDIDATES", "10")
        fanout = env_value(
            env,
            "OFF_POLICY_BRANCH_FANOUT",
            env_value(env, "BRANCH_FANOUT", "2"),
        )
        assert int(min_candidates) <= int(max_candidates), (
            "epsilon_greedy_off_policy requires OFF_POLICY_MIN_CANDIDATES <= "
            "OFF_POLICY_MAX_CANDIDATES"
        )
        assert int(fanout) <= int(min_candidates), (
            "epsilon_greedy_off_policy requires BRANCH_FANOUT <= "
            "OFF_POLICY_MIN_CANDIDATES"
        )
        return ModeSpec(
            mode="epsilon_greedy_off_policy",
            selector_mode=env_value(
                env, "OFF_POLICY_SELECTOR_MODE", "embed_diverse_topk_random"
            ),
            branch_prob="0.0",
            branch_fanout=fanout,
            max_branch_points=env_value(
                env,
                "OFF_POLICY_MAX_BRANCH_POINTS_PER_ROLLOUT",
                env_value(env, "MAX_BRANCH_POINTS_PER_ROLLOUT", "1"),
            ),
            num_candidates=max_candidates,
            epsilon_greedy_prob=env_value(
                env,
                "OFF_POLICY_EPSILON_GREEDY_PROB",
                env_value(env, "EPSILON_GREEDY_PROB", "0.1"),
            ),
            n_resp_per_prompt=env_value(env, "OFF_POLICY_N_RESP_PER_PROMPT", "1"),
            algorithm_adv_estimator=env_value(
                env,
                "OFF_POLICY_ALGORITHM_ADV_ESTIMATOR",
                env_value(env, "ALGORITHM_ADV_ESTIMATOR", "branch_interpolated_grpo"),
            ),
            rollout_gpu_memory_utilization=env_value(
                env,
                "OFF_POLICY_ROLLOUT_GPU_MEMORY_UTILIZATION",
                self.decode.rollout_gpu_memory_utilization,
            ),
            use_full_stop_strings=env_value(env, "USE_FULL_STOP_STRINGS", "False"),
            initial_assistant_prefix=env_value(env, "INITIAL_ASSISTANT_PREFIX", ""),
            off_policy_min_candidates=min_candidates,
            off_policy_max_candidates=max_candidates,
        )

    def _dataset_cache(self, *, mode_spec: ModeSpec) -> str:
        env_cache = optional_env_value(self.base_env, "HF_DATASETS_CACHE")
        if env_cache:
            return env_cache
        return str(
            self.identity.hf_datasets_cache_root
            / self.identity.submission_id
            / mode_spec.mode
        )

    def _decode_env(self, *, mode_spec: ModeSpec) -> dict[str, str]:
        return {
            "EXEC_TEMPERATURE": self.decode.exec_temperature,
            "STEER_TEMPERATURE": self.decode.steer_temperature,
            "EXEC_TOP_P": self.decode.exec_top_p,
            "STEER_TOP_P": self.decode.steer_top_p,
            "PRESENCE_PENALTY": self.decode.presence_penalty,
            "REPETITION_PENALTY": self.decode.repetition_penalty,
            "STEER_REPETITION_PENALTY": self.decode.steer_repetition_penalty,
            "REPETITION_CHECKING_ENABLED": self.decode.repetition_checking_enabled,
            "USE_FULL_STOP_STRINGS": mode_spec.use_full_stop_strings,
            "ROLLOUT_TOP_LOGPROBS": self.decode.rollout_top_logprobs,
            "ROLLOUT_GPU_MEMORY_UTILIZATION": mode_spec.rollout_gpu_memory_utilization,
            "ROLLOUT_MAX_NUM_BATCHED_TOKENS": self.decode.rollout_max_num_batched_tokens,
            "ROLLOUT_MAX_NUM_SEQS": self.decode.rollout_max_num_seqs,
            "ROLLOUT_GDN_PREFILL_BACKEND": self.decode.rollout_gdn_prefill_backend,
            "ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE": (
                self.decode.rollout_disable_custom_all_reduce
            ),
        }

    def _actor_runtime_env(self) -> dict[str, str]:
        return {
            "ACTOR_USE_REMOVE_PADDING": self.actor_runtime.use_remove_padding,
            "ACTOR_USE_FUSED_KERNELS": self.actor_runtime.use_fused_kernels,
            "ACTOR_FUSED_KERNEL_BACKEND": self.actor_runtime.fused_kernel_backend,
            "ACTOR_ATTN_IMPLEMENTATION": self.actor_runtime.attn_implementation,
        }


def default_modes() -> str:
    """Return default Qwen35 matrix rollout modes."""

    return "branching no_branching structured_baseline epsilon_greedy"


def default_submission_id() -> str:
    """Return a collision-resistant submission id for cache scoping."""

    from datetime import datetime, timezone

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{os.getpid()}"


def build_rltraining_dir() -> Path:
    """Resolve RLTraining directory from this module location."""

    return Path(__file__).resolve().parents[1]


def manifest_root(spec: Qwen35MatrixSpec) -> Path:
    """Return the submission manifest root for this matrix."""

    cache_root = Path(
        env_value(
            spec.base_env,
            "CACHE_ROOT",
            str(DEFAULT_SCRATCH_ROOT / "RLTraining/qwen35_branching_dapo"),
        )
    )
    return cache_root / "submissions" / spec.identity.submission_id


def write_manifest(
    *,
    spec: Qwen35MatrixSpec,
    mode_spec: ModeSpec,
    sbatch_env: Mapping[str, str],
    sbatch_args: Sequence[str],
    path: Path,
) -> None:
    """Write the resolved launch specification as JSON."""

    payload = {
        "mode": json_safe(asdict(mode_spec)),
        "resources": json_safe(asdict(spec.resources)),
        "shape": json_safe(asdict(spec.shape)),
        "decode": json_safe(asdict(spec.decode)),
        "actor_runtime": json_safe(asdict(spec.actor_runtime)),
        "artifact_retention": json_safe(asdict(spec.artifact_retention)),
        "identity": json_safe(asdict(spec.identity)),
        "sbatch_env": dict(sorted(sbatch_env.items())),
        "sbatch_args": list(sbatch_args),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def json_safe(value: object) -> object:
    """Return a JSON-serializable equivalent for dataclass payload values."""

    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [json_safe(child) for child in value]
    return value


def submit_matrix(*, spec: Qwen35MatrixSpec, dry_run: bool, test_only: bool) -> int:
    """Submit or print all jobs in a resolved Qwen35 matrix."""

    exit_code = 0
    for mode in spec.modes:
        mode_spec = spec.mode_spec(mode)
        mode_root = manifest_root(spec) / mode_spec.mode
        run_spec_json = mode_root / "run_spec.json"
        sbatch_env = spec.sbatch_env_for_mode(
            mode_spec=mode_spec, run_spec_json=run_spec_json
        )
        sbatch_args = spec.resources.sbatch_args(
            job_name=f"q35rl-{mode_spec.mode_slug}", test_only=test_only
        )
        write_manifest(
            spec=spec,
            mode_spec=mode_spec,
            sbatch_env=sbatch_env,
            sbatch_args=sbatch_args,
            path=run_spec_json,
        )
        if dry_run:
            print(json.dumps({"mode": mode, "run_spec_json": str(run_spec_json)}))
            continue
        result = run_sbatch(sbatch_args=sbatch_args, sbatch_env=sbatch_env)
        output = (result.stdout.strip() or result.stderr.strip()) or "ok"
        print(f"{mode}: {output}")
        if result.returncode:
            exit_code = result.returncode
        elif result.stderr.strip() and result.stdout.strip():
            sys.stderr.write(result.stderr)
        if result.returncode == 0 and not test_only:
            copy_checkpoint_manifest(
                spec=spec,
                mode_spec=mode_spec,
                source_path=run_spec_json,
                sbatch_output=result.stdout,
            )
    return exit_code


def copy_checkpoint_manifest(
    *,
    spec: Qwen35MatrixSpec,
    mode_spec: ModeSpec,
    source_path: Path,
    sbatch_output: str,
) -> None:
    """Copy the launch manifest to the predicted checkpoint directory."""

    job_id = parse_sbatch_job_id(sbatch_output)
    if not job_id:
        return
    experiment_name = checkpoint_experiment_name(
        spec=spec, mode_spec=mode_spec, job_id=job_id
    )
    target_path = checkpoint_manifest_path(spec=spec, experiment_name=experiment_name)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(source_path.read_text())


def parse_sbatch_job_id(output: str) -> str | None:
    """Parse a Slurm job id from `sbatch --parsable` output."""

    token = output.strip().split(maxsplit=1)[0] if output.strip() else ""
    job_id = token.split(";", maxsplit=1)[0]
    return job_id or None


def checkpoint_experiment_name(
    *, spec: Qwen35MatrixSpec, mode_spec: ModeSpec, job_id: str
) -> str:
    """Return the experiment name used by the Qwen35 Slurm wrapper."""

    if spec.identity.experiment_name:
        return spec.identity.experiment_name
    model_slug = env_value(spec.base_env, "MODEL_NAME_SLUG", "qwen35_2b")
    return f"{model_slug}_{mode_spec.mode}_{spec.identity.run_label}_{job_id}"


def checkpoint_manifest_path(*, spec: Qwen35MatrixSpec, experiment_name: str) -> Path:
    """Return the tracked checkpoint-manifest target path."""

    cache_root = Path(
        env_value(
            spec.base_env,
            "CACHE_ROOT",
            str(DEFAULT_SCRATCH_ROOT / "RLTraining/qwen35_branching_dapo"),
        )
    )
    return cache_root / "checkpoints" / experiment_name / "run_spec.json"


def run_sbatch(
    *, sbatch_args: Sequence[str], sbatch_env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    """Run sbatch with the resolved per-job environment."""

    child_env = os.environ.copy()
    child_env.update(sbatch_env)
    return subprocess.run(
        list(sbatch_args),
        check=False,
        env=child_env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    """Parse CLI arguments for the run-spec launcher."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    matrix = subparsers.add_parser(
        "submit-qwen35-matrix",
        help="Submit the Qwen3.5 rollout matrix from typed run specs.",
    )
    matrix.add_argument("--dry-run", action="store_true")
    matrix.add_argument("--test-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for typed run-spec launchers."""

    args = parse_args(sys.argv[1:] if argv is None else argv)
    rltraining_dir = build_rltraining_dir()
    spec = Qwen35MatrixSpec.from_env(env=os.environ, rltraining_dir=rltraining_dir)
    if args.command == "submit-qwen35-matrix":
        return submit_matrix(spec=spec, dry_run=args.dry_run, test_only=args.test_only)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
