#!/usr/bin/env python3
"""Submit a chain of dependent branching-DAPO Slurm jobs.

Inputs:
    Command-line options describing the experiment name, chunk sizing,
    checkpoint root, and Slurm resource shape.
Outputs:
    Prints the shared W&B run ID, checkpoint root, and submitted Slurm job IDs.

Example:
    python RLTraining/slurm/submit_chunked_branching_dapo.py \
        --experiment-name my_200_step_run \
        --project-name branching_dapo \
        --total-steps 200 \
        --steps-per-job 20
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class ChunkedLaunchConfig:
    """Configuration for a chained Slurm submission.

    Inputs:
        experiment_name: Shared experiment name across all chunks.
        project_name: W&B project name and trainer project name.
        total_steps: Absolute final training step target.
        steps_per_job: Number of new steps each chunk is responsible for.
        save_frequency: Checkpoint cadence in absolute steps.
        cache_root: Scratch root for checkpoints and local W&B files.
        account: Slurm account.
        partition: Slurm partition.
        time_limit: Slurm walltime per chunk.
        gpus: GPUs per chunk.
        cpus_per_task: CPUs per chunk.
        memory: Memory per chunk.
        python_bin: Python executable used by the sbatch wrapper.
        repo_root: Repository root containing RLTraining.
    Outputs:
        An immutable data object used to build sbatch commands.
    """

    experiment_name: str
    project_name: str
    total_steps: int
    steps_per_job: int
    save_frequency: int
    cache_root: Path
    account: str
    partition: str
    time_limit: str
    gpus: int
    cpus_per_task: int
    memory: str
    python_bin: Path
    repo_root: Path

    def checkpoint_root(self) -> Path:
        """Return the scratch checkpoint directory for the shared experiment."""

        return self.cache_root / "checkpoints" / self.experiment_name

    def log_root(self) -> Path:
        """Return the repo-local Slurm log directory for this chain."""

        return self.repo_root / "RLTraining" / "logs" / self.experiment_name

    def manifest_path(self) -> Path:
        """Return the repo-local manifest path for this chain submission."""

        return self.log_root() / "chain_manifest.json"


@dataclass(frozen=True)
class SubmittedChunk:
    """Metadata for one submitted Slurm chunk.

    Inputs:
        chunk_index: One-based chunk number.
        total_training_steps: Absolute trainer step target for this chunk.
        job_id: Slurm job ID returned by sbatch.
    Outputs:
        Serializable submission metadata for the launch manifest.
    """

    chunk_index: int
    total_training_steps: int
    job_id: str


def parse_args(repo_root: Path) -> ChunkedLaunchConfig:
    """Parse CLI arguments into a validated launch config.

    Inputs:
        repo_root: Repository root used to resolve default paths.
    Outputs:
        A validated ChunkedLaunchConfig instance.
    """

    default_cache_root = Path(
        "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/branching_dapo"
    )
    default_python = repo_root / "RLTraining" / ".venv" / "bin" / "python"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-name",
        default=f"full_200step_fsdp4_bsz8_16k_branch_chain_{timestamp}",
    )
    parser.add_argument("--project-name", default="branching_dapo")
    parser.add_argument("--total-steps", type=int, default=200)
    parser.add_argument("--steps-per-job", type=int, default=20)
    parser.add_argument("--save-frequency", type=int, default=10)
    parser.add_argument("--cache-root", type=Path, default=default_cache_root)
    parser.add_argument("--account", default="PAS3268")
    parser.add_argument("--partition", default="gpu")
    parser.add_argument("--time-limit", default="05:00:00")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--cpus-per-task", type=int, default=32)
    parser.add_argument("--memory", default="256G")
    parser.add_argument("--python-bin", type=Path, default=default_python)
    args = parser.parse_args()
    assert args.total_steps > 0, "total_steps must be positive"
    assert args.steps_per_job > 0, "steps_per_job must be positive"
    assert args.total_steps % args.steps_per_job == 0, "total_steps must be divisible by steps_per_job"
    assert args.save_frequency > 0, "save_frequency must be positive"
    return ChunkedLaunchConfig(
        experiment_name=args.experiment_name,
        project_name=args.project_name,
        total_steps=args.total_steps,
        steps_per_job=args.steps_per_job,
        save_frequency=args.save_frequency,
        cache_root=args.cache_root,
        account=args.account,
        partition=args.partition,
        time_limit=args.time_limit,
        gpus=args.gpus,
        cpus_per_task=args.cpus_per_task,
        memory=args.memory,
        python_bin=args.python_bin,
        repo_root=repo_root,
    )


def resolve_wandb_run_id() -> str:
    """Generate a stable W&B run ID for a multi-job chain.

    Inputs:
        None.
    Outputs:
        A short hexadecimal run ID suitable for WANDB_RUN_ID.
    """

    return secrets.token_hex(8)


def build_sbatch_command(
    *,
    config: ChunkedLaunchConfig,
    chunk_index: int,
    total_training_steps: int,
    dependency_job_id: str | None,
) -> list[str]:
    """Build the sbatch command for one chunk.

    Inputs:
        config: Shared chain configuration.
        chunk_index: One-based chunk number.
        total_training_steps: Absolute trainer step target for this chunk.
        dependency_job_id: Upstream job ID or None for the first chunk.
    Outputs:
        A subprocess-ready sbatch command list.
    """

    job_name = f"{config.experiment_name}_chunk{chunk_index:02d}"
    output_path = config.log_root() / f"{job_name}-%j.out"
    error_path = config.log_root() / f"{job_name}-%j.err"
    sbatch_script = config.repo_root / "RLTraining" / "slurm" / "branching_dapo_chunk.sbatch"
    command = [
        "sbatch",
        "--parsable",
        f"--account={config.account}",
        f"--partition={config.partition}",
        "--nodes=1",
        "--ntasks=1",
        f"--cpus-per-task={config.cpus_per_task}",
        f"--mem={config.memory}",
        f"--time={config.time_limit}",
        f"--gres=gpu:{config.gpus}",
        f"--job-name={job_name}",
        f"--output={output_path}",
        f"--error={error_path}",
    ]
    if dependency_job_id is not None:
        command.append(f"--dependency=afterok:{dependency_job_id}")
    command.append(str(sbatch_script))
    return command


def build_submission_environment(
    *,
    base_env: Mapping[str, str],
    config: ChunkedLaunchConfig,
    total_training_steps: int,
    wandb_run_id: str,
) -> dict[str, str]:
    """Create the environment propagated to sbatch.

    Inputs:
        base_env: Current process environment.
        config: Shared chain configuration.
        total_training_steps: Absolute trainer step target for this chunk.
        wandb_run_id: Shared W&B run ID across all chunks.
    Outputs:
        An environment dictionary suitable for subprocess.run.
    """

    env = dict(base_env)
    env["RL_PROJECT_NAME"] = config.project_name
    env["RL_EXPERIMENT_NAME"] = config.experiment_name
    env["RL_TOTAL_TRAINING_STEPS"] = str(total_training_steps)
    env["RL_SAVE_FREQ"] = str(config.save_frequency)
    env["RL_CACHE_ROOT"] = str(config.cache_root)
    env["RL_WANDB_RUN_ID"] = wandb_run_id
    env["WANDB_RESUME"] = "allow"
    env["RL_PYTHON_BIN"] = str(config.python_bin)
    env["RLTRAINING_DIR_OVERRIDE"] = str(config.repo_root / "RLTraining")
    env["REPO_ROOT_OVERRIDE"] = str(config.repo_root)
    return env


def submit_chunk(
    *,
    command: Sequence[str],
    env: dict[str, str],
) -> str:
    """Submit one Slurm chunk and return the parsed job ID.

    Inputs:
        command: sbatch command list produced by build_sbatch_command().
        env: Submission environment containing shared resume metadata.
    Outputs:
        The Slurm job ID string returned by `sbatch --parsable`.
    """

    completed = subprocess.run(
        args=list(command),
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, (
        f"sbatch failed with code {completed.returncode}: "
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    job_id = completed.stdout.strip().split(";")[0]
    assert job_id, f"Unexpected sbatch output: {completed.stdout!r}"
    return job_id


def write_manifest(
    *,
    config: ChunkedLaunchConfig,
    wandb_run_id: str,
    submitted_chunks: Sequence[SubmittedChunk],
) -> None:
    """Persist the chain submission manifest on scratch.

    Inputs:
        config: Shared launch configuration.
        wandb_run_id: Shared W&B run ID for all chunks.
        submitted_chunks: Ordered list of submitted chunk metadata.
    Outputs:
        Writes a JSON manifest to scratch for later inspection.
    """

    manifest = {
        "config": {
            **asdict(config),
            "cache_root": str(config.cache_root),
            "python_bin": str(config.python_bin),
            "repo_root": str(config.repo_root),
        },
        "wandb_run_id": wandb_run_id,
        "checkpoint_root": str(config.checkpoint_root()),
        "submitted_chunks": [asdict(chunk) for chunk in submitted_chunks],
    }
    manifest_path = config.manifest_path()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def ensure_writable_dir(path: Path) -> None:
    """Assert that a directory can be created and written.

    Inputs:
        path: Directory path that must exist and be writable.
    Outputs:
        Creates the directory if needed and raises on write failure.
    """

    path.mkdir(parents=True, exist_ok=True)
    probe_path = path / ".write_probe"
    probe_path.write_text("ok\n", encoding="utf-8")
    probe_path.unlink()


def main() -> None:
    """Submit all chunked jobs and print the chain summary.

    Inputs:
        Command-line options parsed by parse_args().
    Outputs:
        Prints the shared W&B run ID, checkpoint root, manifest path,
        and the ordered list of submitted Slurm job IDs.
    """

    repo_root = Path(__file__).resolve().parents[2]
    config = parse_args(repo_root=repo_root)
    ensure_writable_dir(config.log_root())
    ensure_writable_dir(config.manifest_path().parent)
    wandb_run_id = resolve_wandb_run_id()
    dependency_job_id: str | None = None
    submitted_chunks: list[SubmittedChunk] = []
    total_chunks = config.total_steps // config.steps_per_job
    for chunk_index in range(1, total_chunks + 1):
        total_training_steps = chunk_index * config.steps_per_job
        command = build_sbatch_command(
            config=config,
            chunk_index=chunk_index,
            total_training_steps=total_training_steps,
            dependency_job_id=dependency_job_id,
        )
        env = build_submission_environment(
            base_env=os.environ,
            config=config,
            total_training_steps=total_training_steps,
            wandb_run_id=wandb_run_id,
        )
        job_id = submit_chunk(command=command, env=env)
        submitted_chunks.append(
            SubmittedChunk(
                chunk_index=chunk_index,
                total_training_steps=total_training_steps,
                job_id=job_id,
            )
        )
        dependency_job_id = job_id
    write_manifest(
        config=config,
        wandb_run_id=wandb_run_id,
        submitted_chunks=submitted_chunks,
    )
    print(f"experiment_name={config.experiment_name}")
    print(f"wandb_run_id={wandb_run_id}")
    print(f"checkpoint_root={config.checkpoint_root()}")
    print(f"manifest_path={config.manifest_path()}")
    for chunk in submitted_chunks:
        print(
            f"chunk={chunk.chunk_index:02d} total_training_steps={chunk.total_training_steps} job_id={chunk.job_id}"
        )


if __name__ == "__main__":
    main()
