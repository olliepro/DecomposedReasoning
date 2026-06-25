"""Typed launch specifications for branching-eval runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

import yaml

from branching_eval.selector_types import SelectorMode

DEFAULT_SCRATCH_ROOT = Path("/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning")
DEFAULT_ANALYSIS_DIR = Path(
    "/users/PAA0201/ollieproudman/work/DecomposedReasoning/Analysis"
)
DEFAULT_SPEC_ROOT = DEFAULT_ANALYSIS_DIR / "branching_eval/generated_run_specs"
DEFAULT_OUTPUT_PARENT = DEFAULT_SCRATCH_ROOT / "Analysis/branching_eval"

EvalMode = Literal["baseline", "structured", "branching", "epsilon", "all"]


def env_value(env: Mapping[str, str], name: str, default: str) -> str:
    """Return a non-empty environment value or its default."""

    value = env.get(name, "")
    return value if value else default


def optional_env_value(env: Mapping[str, str], name: str) -> str:
    """Return an environment value while preserving empty-as-empty semantics."""

    return env.get(name, "")


def split_doc_ids(raw_doc_ids: str) -> tuple[int, ...]:
    """Parse comma/space separated doc ids from an environment value."""

    if not raw_doc_ids:
        return ()
    normalized = raw_doc_ids.replace(",", " ")
    return tuple(int(value) for value in normalized.split())


@dataclass(frozen=True)
class SlurmResources:
    """Slurm resource request for one branching-eval job."""

    account: str
    partition: str
    gpu_count: str
    gpu_type: str
    gpu_gres_override: str
    cpu_count: str
    memory: str
    time_limit: str
    exclude_nodes: str
    sbatch_script: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "SlurmResources":
        """Build Slurm resources from operator environment variables."""

        return cls(
            account=env_value(env, "ACCOUNT", "PAA0201"),
            partition=env_value(env, "PARTITION", "quad"),
            gpu_count=env_value(env, "GPU_COUNT", "1"),
            gpu_type=env_value(env, "GPU_TYPE", "a100"),
            gpu_gres_override=optional_env_value(env, "GPU_GRES"),
            cpu_count=env_value(env, "CPU_COUNT", "32"),
            memory=env_value(env, "MEMORY", "250G"),
            time_limit=env_value(env, "TIME_LIMIT", "12:00:00"),
            exclude_nodes=optional_env_value(env, "EXCLUDE_NODES"),
            sbatch_script=Path(
                env_value(
                    env,
                    "SBATCH_SCRIPT",
                    str(DEFAULT_ANALYSIS_DIR / "slurm/branching_eval.sbatch"),
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
            f"--account={self.account}",
            f"--partition={self.partition}",
            f"--gres={self.gpu_gres()}",
            f"--cpus-per-task={self.cpu_count}",
            f"--mem={self.memory}",
            f"--time={self.time_limit}",
            f"--job-name={job_name}",
            "--export=ALL",
        ]
        if self.exclude_nodes:
            args.append(f"--exclude={self.exclude_nodes}")
        if test_only:
            args.append("--test-only")
        args.append(str(self.sbatch_script))
        return args


@dataclass(frozen=True)
class EvalShape:
    """Shared model, decode, and matrix shape for one eval run."""

    task_name: str
    model_id: str
    model_path: str
    mode: EvalMode
    run_name: str
    selector: SelectorMode
    seed: int
    doc_ids: tuple[int, ...]
    limit: int | None
    output_root: Path
    tensor_parallel_size: int
    gpu_memory_utilization: float
    max_logprobs: int
    max_gen_toks: int
    max_model_len: int | None
    temperature: float
    steer_temperature: float
    top_p: float
    decode_chunk_tokens: int
    baseline_rollouts: int
    branch_prob: float
    max_branch_points_per_rollout: int
    max_concurrent_branches: int
    num_candidates: int
    branch_fanout: int
    max_steer_tokens: int
    epsilon_greedy_prob: float
    max_concurrent_docs: int

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "EvalShape":
        """Build an eval shape from environment variables."""

        run_name = env_value(env, "RUN_NAME", "branching-eval-smoke")
        return cls(
            task_name=env_value(env, "TASK_NAME", "aime25"),
            model_id=env_value(env, "MODEL_ID", "sft"),
            model_path=env_value(env, "MODEL_PATH", "Qwen/Qwen3-8B"),
            mode=_parse_mode(raw_mode=env_value(env, "EVAL_MODE", "structured")),
            run_name=run_name,
            selector=_parse_selector(
                raw_selector=env_value(env, "SELECTOR", "embed_diverse_topk_random")
            ),
            seed=int(env_value(env, "SEED", "1234")),
            doc_ids=split_doc_ids(raw_doc_ids=optional_env_value(env, "DOC_IDS")),
            limit=_optional_int(value=optional_env_value(env, "LIMIT")),
            output_root=Path(
                env_value(env, "OUTPUT_ROOT", str(DEFAULT_OUTPUT_PARENT / run_name))
            ),
            tensor_parallel_size=int(env_value(env, "TENSOR_PARALLEL_SIZE", "1")),
            gpu_memory_utilization=float(
                env_value(env, "GPU_MEMORY_UTILIZATION", "0.9")
            ),
            max_logprobs=int(env_value(env, "MAX_LOGPROBS", "20")),
            max_gen_toks=int(env_value(env, "MAX_GEN_TOKS", "32768")),
            max_model_len=_optional_int(value=optional_env_value(env, "MAX_MODEL_LEN")),
            temperature=float(env_value(env, "TEMPERATURE", "0.7")),
            steer_temperature=float(env_value(env, "STEER_TEMPERATURE", "1.0")),
            top_p=float(env_value(env, "TOP_P", "0.95")),
            decode_chunk_tokens=int(env_value(env, "DECODE_CHUNK_TOKENS", "512")),
            baseline_rollouts=int(env_value(env, "BASELINE_ROLLOUTS", "48")),
            branch_prob=float(env_value(env, "BRANCH_PROB", "0.0")),
            max_branch_points_per_rollout=int(
                env_value(env, "MAX_BRANCH_POINTS_PER_ROLLOUT", "4")
            ),
            max_concurrent_branches=int(
                env_value(env, "MAX_CONCURRENT_BRANCHES", "20")
            ),
            num_candidates=int(env_value(env, "NUM_CANDIDATES", "100")),
            branch_fanout=int(env_value(env, "BRANCH_FANOUT", "2")),
            max_steer_tokens=int(env_value(env, "MAX_STEER_TOKENS", "20")),
            epsilon_greedy_prob=float(env_value(env, "EPSILON_GREEDY_PROB", "0.33")),
            max_concurrent_docs=int(env_value(env, "MAX_CONCURRENT_DOCS", "2")),
        )

    def cli_args(self, *, config_path: Path) -> list[str]:
        """Return `run_branching_lm_eval.py` arguments for this run."""

        args = ["--config", str(config_path)]
        if self.limit is not None:
            args += ["--limit", str(self.limit)]
        for doc_id in self.doc_ids:
            args += ["--doc-id", str(doc_id)]
        return args

    def config_payload(self) -> dict[str, object]:
        """Return the generated YAML payload consumed by branching eval."""

        return {
            "tasks": {"task_names": [self.task_name]},
            "models": [self.model_payload()],
            "serve": self.serve_payload(),
            "decoding": self.decoding_payload(),
            "branching": self.branching_payload(),
            "artifacts": {"output_root": str(self.output_root)},
            "run_matrix": self.run_matrix_payload(),
        }

    def model_payload(self) -> dict[str, object]:
        """Return the generated model spec block."""

        return {
            "model_id": self.model_id,
            "checkpoint_or_repo": self.model_path,
            "trigger_steer_default": True,
        }

    def serve_payload(self) -> dict[str, object]:
        """Return the generated serving block."""

        return {
            "host": "127.0.0.1",
            "base_port": 8020,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "kv_offloading_size_gb": 0.0,
            "dtype": "auto",
            "scheduling_policy": "priority",
            "trust_remote_code": True,
            "max_logprobs": self.max_logprobs,
            "startup_timeout_seconds": 900.0,
            "request_timeout_seconds": 900.0,
            "poll_interval_seconds": 1.0,
        }

    def decoding_payload(self) -> dict[str, object]:
        """Return the generated decoding block."""

        payload: dict[str, object] = {
            "temperature": self.temperature,
            "steer_temperature": self.steer_temperature,
            "top_p": self.top_p,
            "max_gen_toks": self.max_gen_toks,
            "top_logprobs": self.max_logprobs,
            "decode_chunk_tokens": self.decode_chunk_tokens,
        }
        if self.max_model_len is not None:
            payload["max_model_len"] = self.max_model_len
        return payload

    def branching_payload(self) -> dict[str, object]:
        """Return the generated branching block."""

        return {
            "branch_prob": self.branch_prob,
            "max_branch_points_per_rollout": self.max_branch_points_per_rollout,
            "max_concurrent_branches": self.max_concurrent_branches,
            "num_candidates": self.num_candidates,
            "branch_fanout": self.branch_fanout,
            "max_clusters": 4,
            "max_steer_tokens": self.max_steer_tokens,
            "steer_repetition_penalty": 1.01,
            "epsilon_greedy_prob": self.epsilon_greedy_prob,
        }

    def run_matrix_payload(self) -> dict[str, object]:
        """Return the generated run-matrix block."""

        return {
            "include_baselines": self.mode in {"baseline", "all"},
            "include_structured_baselines": self.mode in {"structured", "all"},
            "baseline_rollouts": self.baseline_rollouts,
            "include_branching": self.mode in {"branching", "all"},
            "include_epsilon_greedy": self.mode in {"epsilon", "all"},
            "selectors": [self.selector],
            "seed_values": [self.seed],
            "default_limit": self.limit,
            "max_concurrent_docs": self.max_concurrent_docs,
        }


@dataclass(frozen=True)
class EvalRunSpec:
    """Complete launch spec for one generated branching-eval run."""

    shape: EvalShape
    slurm: SlurmResources
    spec_dir: Path

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> "EvalRunSpec":
        """Build a full run spec from environment variables."""

        shape = EvalShape.from_env(env=env)
        spec_root = Path(env_value(env, "SPEC_ROOT", str(DEFAULT_SPEC_ROOT)))
        return cls(
            shape=shape,
            slurm=SlurmResources.from_env(env=env),
            spec_dir=spec_root / shape.run_name,
        )

    @property
    def config_path(self) -> Path:
        """Return the generated YAML path."""

        return self.spec_dir / "config.yaml"

    @property
    def manifest_path(self) -> Path:
        """Return the generated run-spec manifest path."""

        return self.spec_dir / "run_spec.json"

    def write(self) -> None:
        """Write generated YAML and manifest files."""

        self.spec_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            yaml.safe_dump(self.shape.config_payload(), sort_keys=False),
            encoding="utf-8",
        )
        self.manifest_path.write_text(
            json.dumps(asdict(self), indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )

    def local_command(self) -> list[str]:
        """Return the local eval command using the generated config."""

        return [
            "uv",
            "run",
            "python",
            "run_branching_lm_eval.py",
        ] + self.shape.cli_args(config_path=self.config_path)

    def submit(self, *, test_only: bool) -> subprocess.CompletedProcess[str]:
        """Submit or forecast this run with sbatch."""

        self.write()
        env = os.environ.copy()
        env.update(self.sbatch_env())
        args = self.slurm.sbatch_args(job_name=self.shape.run_name, test_only=test_only)
        return subprocess.run(
            args=args,
            cwd=DEFAULT_ANALYSIS_DIR,
            env=env,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def sbatch_env(self) -> dict[str, str]:
        """Return environment variables consumed by the Slurm wrapper."""

        return {
            "BRANCH_CONFIG": str(self.config_path),
            "BRANCH_DOC_IDS": ",".join(str(doc_id) for doc_id in self.shape.doc_ids),
            "BRANCH_LIMIT": "" if self.shape.limit is None else str(self.shape.limit),
            "RUN_SPEC_JSON": str(self.manifest_path),
        }


def _parse_mode(*, raw_mode: str) -> EvalMode:
    modes: set[str] = {"baseline", "structured", "branching", "epsilon", "all"}
    assert raw_mode in modes, f"EVAL_MODE must be one of {sorted(modes)}"
    return raw_mode  # type: ignore[return-value]


def _parse_selector(*, raw_selector: str) -> SelectorMode:
    selectors: set[str] = {
        "cluster_across",
        "embed_diverse_topk_random",
        "within_cluster",
        "random",
    }
    assert raw_selector in selectors, f"SELECTOR must be one of {sorted(selectors)}"
    return raw_selector  # type: ignore[return-value]


def _optional_int(*, value: str) -> int | None:
    if not value:
        return None
    return int(value)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse run-spec CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("write", "dry-run", "submit", "test-only"):
        subparsers.add_parser(command)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the run-spec CLI."""

    args = parse_args(argv=argv)
    spec = EvalRunSpec.from_env(env=os.environ)
    if args.command in {"write", "dry-run"}:
        spec.write()
        print(f"config={spec.config_path}")
        print(f"manifest={spec.manifest_path}")
        print("local_command=" + " ".join(spec.local_command()))
        return 0
    result = spec.submit(test_only=args.command == "test-only")
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
