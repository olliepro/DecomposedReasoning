"""Typed data objects for vLLM native tree-search experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

BenchmarkMode = Literal[
    "grammar_temp",
    "eps_on_policy_diverse",
    "eps_off_policy_verbalized",
]
DiversityVectorSource = Literal["lexical", "model_hidden_state"]

DEFAULT_MODEL_PATH = Path(
    "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/"
    "SFTTraining/outputs/"
    "qwen35_4b_base_warmup_offpolicy_added_tokens_system_prompt_lr7e6_"
    "zero3_12ep_decay_20260625/final_model"
)
DEFAULT_SCRATCH_ROOT = Path(
    "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/vLLM_Experimental"
)
DEFAULT_DOC_IDS = (0, 1, 2, 3, 4, 5, 6, 7)


@dataclass(frozen=True)
class TreeSearchParams:
    """Runtime behavior carried in `SamplingParams.extra_args`."""

    mode: BenchmarkMode = "grammar_temp"
    fire_rate: float = 0.10
    candidate_count: int = 50
    branch_fanout: int = 2
    branch_depth: int = 4
    off_policy_min_candidates: int = 3
    off_policy_max_candidates: int = 10
    branch_max_tokens: int = 700
    max_model_len: int = 17_408
    max_num_batched_tokens: int = 65_536
    max_steer_tokens: int = 30
    max_exec_tokens: int = 512
    steer_temperature: float = 1.0
    exec_temperature: float = 0.7
    post_think_temperature: float = 0.7
    diversity_vector_source: DiversityVectorSource = "lexical"
    seed: int = 1234
    native_scheduler_kv_fork: bool = False
    native_branch_wave_size: int = 50
    native_branch_dynamic_admission: bool = True
    native_branch_min_free_blocks: int = 256
    native_branch_free_block_fraction: float = 0.05
    native_branch_seq_reserve: int = 8
    native_branch_priority_boost: int = 1000
    native_branch_block_safety_multiplier: float = 1.25
    native_branch_blocked_log_interval_s: float = 5.0
    native_branch_max_live_pools: int = 2
    native_branch_max_queued_pools: int = 8

    def validate(self) -> None:
        """Assert the config is internally consistent."""

        assert self.mode in {
            "grammar_temp",
            "eps_on_policy_diverse",
            "eps_off_policy_verbalized",
        }, f"unsupported benchmark mode: {self.mode}"
        assert 0.0 <= self.fire_rate <= 1.0, "fire_rate must be in [0, 1]."
        assert self.candidate_count >= 1, "candidate_count must be positive."
        assert self.branch_fanout >= 1, "branch_fanout must be positive."
        assert self.branch_depth >= 1, "branch_depth must be positive."
        assert (
            self.off_policy_min_candidates >= 1
        ), "off_policy_min_candidates must be positive."
        assert (
            self.off_policy_max_candidates >= self.off_policy_min_candidates
        ), "off_policy_max_candidates must be >= off_policy_min_candidates."
        assert self.branch_max_tokens >= 1, "branch_max_tokens must be positive."
        assert self.max_steer_tokens >= 1, "max_steer_tokens must be positive."
        assert self.max_exec_tokens >= 1, "max_exec_tokens must be positive."
        assert self.diversity_vector_source in {
            "lexical",
            "model_hidden_state",
        }, "unsupported diversity vector source."
        assert (
            self.native_branch_wave_size >= 1
        ), "native_branch_wave_size must be positive."
        assert (
            self.native_branch_min_free_blocks >= 0
        ), "native_branch_min_free_blocks must be non-negative."
        assert (
            0.0 <= self.native_branch_free_block_fraction < 1.0
        ), "native_branch_free_block_fraction must be in [0, 1)."
        assert (
            self.native_branch_seq_reserve >= 0
        ), "native_branch_seq_reserve must be non-negative."
        assert (
            self.native_branch_priority_boost >= 0
        ), "native_branch_priority_boost must be non-negative."
        assert (
            self.native_branch_block_safety_multiplier >= 1.0
        ), "native_branch_block_safety_multiplier must be >= 1."
        assert (
            self.native_branch_blocked_log_interval_s >= 0.0
        ), "native_branch_blocked_log_interval_s must be non-negative."
        assert (
            self.native_branch_max_live_pools >= 0
        ), "native_branch_max_live_pools must be non-negative."
        assert (
            self.native_branch_max_queued_pools >= 0
        ), "native_branch_max_queued_pools must be non-negative."
        if self.mode != "grammar_temp":
            assert (
                self.native_scheduler_kv_fork
            ), "branch modes require native scheduler KV fork."
        if self.mode == "eps_off_policy_verbalized":
            assert (
                self.branch_fanout <= self.off_policy_min_candidates
            ), "off-policy fanout must fit the minimum verbalized option count."

    def extra_args_payload(self) -> dict[str, object]:
        """Return the payload stored under `vllm_experimental`."""

        self.validate()
        return asdict(self)

    def to_json(self) -> str:
        """Serialize to stable JSON for manifests."""

        return json.dumps(self.extra_args_payload(), sort_keys=True)


@dataclass(frozen=True)
class SlurmResources:
    """One-GPU Ascend benchmark Slurm request."""

    account: str = "PAA0201"
    partition: str = "debug"
    gpu_type: str = "a100"
    gpu_count: int = 1
    cpu_count: int = 16
    memory: str = "128G"
    time_limit: str = "00:30:00"

    def sbatch_args(
        self, *, job_name: str, script_path: Path, test_only: bool
    ) -> list[str]:
        """Return sbatch arguments for this resource request."""

        args = [
            "sbatch",
            "--parsable",
            f"--account={self.account}",
            f"--partition={self.partition}",
            f"--gres=gpu:{self.gpu_type}:{self.gpu_count}",
            f"--cpus-per-task={self.cpu_count}",
            f"--mem={self.memory}",
            f"--time={self.time_limit}",
            f"--job-name={job_name}",
            "--export=ALL",
        ]
        if test_only:
            args.append("--test-only")
        args.append(str(script_path))
        return args


@dataclass(frozen=True)
class HardwareFingerprint:
    """Hardware facts used to reject mixed-GPU comparisons."""

    node: str
    gpu_name: str
    gpu_memory_total_mib: int
    cuda_visible_devices: str

    def comparable_to(self, other: "HardwareFingerprint") -> bool:
        """Return whether two benchmark rows may be compared."""

        return (
            self.gpu_name == other.gpu_name
            and self.gpu_memory_total_mib == other.gpu_memory_total_mib
        )


@dataclass(frozen=True)
class VllmRuntimeSpec:
    """Materialized vLLM runtime metadata."""

    source_package: Path
    runtime_root: Path
    vllm_version: str
    source_hash: str
    patch_hash: str
    patch_files: tuple[str, ...] = field(default_factory=tuple)

    def manifest_payload(self) -> dict[str, object]:
        """Return JSON-serializable runtime metadata."""

        return {
            "source_package": str(self.source_package),
            "runtime_root": str(self.runtime_root),
            "vllm_version": self.vllm_version,
            "source_hash": self.source_hash,
            "patch_hash": self.patch_hash,
            "patch_files": list(self.patch_files),
        }


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark sweep configuration for one mode."""

    run_name: str
    mode: BenchmarkMode
    model_path: Path = DEFAULT_MODEL_PATH
    task_name: str = "aime25"
    doc_ids: tuple[int, ...] = DEFAULT_DOC_IDS
    prompt_concurrency: tuple[int, ...] = (1, 2, 4, 8, 16)
    request_prompt_batch_size: tuple[int, ...] = (1, 4, 8)
    max_model_len: int = 17_408
    max_num_batched_tokens: int = 65_536
    max_num_seqs: int = 384
    params: TreeSearchParams = field(default_factory=TreeSearchParams)

    def validate(self) -> None:
        """Assert the benchmark config is ready to run."""

        assert self.run_name.strip(), "run_name must be non-empty."
        assert self.doc_ids, "doc_ids must be non-empty."
        assert self.model_path.is_absolute(), "model_path must be absolute."
        assert self.prompt_concurrency, "prompt_concurrency must be non-empty."
        assert self.request_prompt_batch_size, "request_prompt_batch_size is required."
        assert self.max_num_seqs >= 1, "max_num_seqs must be positive."
        self.params.validate()

    def sweep_rows(self) -> list[dict[str, object]]:
        """Return one row per load point."""

        self.validate()
        rows: list[dict[str, object]] = []
        for concurrency in self.prompt_concurrency:
            for batch_size in self.request_prompt_batch_size:
                rows.append(
                    {
                        "run_name": self.run_name,
                        "mode": self.mode,
                        "task_name": self.task_name,
                        "doc_ids": list(self.doc_ids),
                        "prompt_concurrency": concurrency,
                        "request_prompt_batch_size": batch_size,
                        "max_model_len": self.max_model_len,
                        "max_num_batched_tokens": self.max_num_batched_tokens,
                        "max_num_seqs": self.max_num_seqs,
                        "model_path": str(self.model_path),
                        "tree_search": self.params.extra_args_payload(),
                    }
                )
        return rows
