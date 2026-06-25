"""Tests for typed RL launch specifications."""

from __future__ import annotations

import json
from pathlib import Path

from branching_dapo.run_specs import (
    Qwen35MatrixSpec,
    checkpoint_experiment_name,
    checkpoint_manifest_path,
    parse_sbatch_job_id,
    write_manifest,
)


def make_spec(env: dict[str, str]) -> Qwen35MatrixSpec:
    """Build a Qwen35 matrix spec with a stable fake repo path."""

    return Qwen35MatrixSpec.from_env(env=env, rltraining_dir=Path("/repo/RLTraining"))


def test_branchp10_spec_keeps_intended_shape(tmp_path: Path) -> None:
    """The branchp10 run shape should not inherit smoke-sized defaults."""

    spec = make_spec(
        {
            "SMOKE_ROLLOUT_MODES": "branching",
            "RUN_LABEL": "gs50_branch_all_lr2e6_branchp10_steer30",
            "TRAIN_PROMPT_BSZ": "8",
            "MAX_PROMPT_LENGTH": "1024",
            "MAX_RESPONSE_LENGTH": "16384",
            "MAX_STEER_TOKENS": "30",
            "N_RESP_PER_PROMPT": "16",
            "ACTOR_LR": "2e-6",
            "PROJECT_NAME": "branching_dapo_qwen35_gs50",
            "MODEL_NAME_SLUG": "qwen35_4b_branch_gs50",
            "MODEL_PATH": "/scratch/model",
            "CACHE_ROOT": "/scratch/cache",
            "BRANCHING_SELECTOR_MODE": "embed_diverse_topk_random",
            "BRANCHING_BRANCH_PROB": "0.10",
            "BRANCHING_BRANCH_FANOUT": "2",
            "BRANCHING_MAX_BRANCH_POINTS_PER_ROLLOUT": "4",
            "BRANCHING_NUM_CANDIDATES": "50",
            "BRANCHING_EPSILON_GREEDY_PROB": "0.1",
            "PERSISTENT_LOG_INTERVAL_STEPS": "25",
            "REPETITION_CHECKING_ENABLED": "False",
        }
    )
    mode = spec.mode_spec("branching")
    env = spec.sbatch_env_for_mode(
        mode_spec=mode, run_spec_json=tmp_path / "run_spec.json"
    )

    assert env["TRAIN_PROMPT_BSZ"] == "8"
    assert env["MAX_PROMPT_LENGTH"] == "1024"
    assert env["MAX_RESPONSE_LENGTH"] == "16384"
    assert env["N_RESP_PER_PROMPT"] == "16"
    assert env["ACTOR_LR"] == "2e-6"
    assert env["MODEL_NAME_SLUG"] == "qwen35_4b_branch_gs50"
    assert env["MODEL_PATH"] == "/scratch/model"
    assert env["CACHE_ROOT"] == "/scratch/cache"
    assert env["ROLLOUT_MAX_NUM_BATCHED_TOKENS"] == "420000"
    assert env["SELECTOR_MODE"] == "embed_diverse_topk_random"
    assert env["BRANCH_PROB"] == "0.10"
    assert env["BRANCH_FANOUT"] == "2"
    assert env["MAX_BRANCH_POINTS_PER_ROLLOUT"] == "4"
    assert env["NUM_CANDIDATES"] == "50"
    assert env["EPSILON_GREEDY_PROB"] == "0.1"
    assert env["PERSISTENT_LOG_INTERVAL_STEPS"] == "25"
    assert env["REPETITION_CHECKING_ENABLED"] == "False"

    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        spec=spec,
        mode_spec=mode,
        sbatch_env=env,
        sbatch_args=["sbatch", "fake.sbatch"],
        path=manifest_path,
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["artifact_retention"]["persistent_log_interval_steps"] == "25"
    assert manifest["decode"]["repetition_checking_enabled"] == "False"
    assert manifest["sbatch_env"]["REPETITION_CHECKING_ENABLED"] == "False"


def test_matrix_modes_have_distinct_defaults(tmp_path: Path) -> None:
    """Per-mode defaults live in one typed resolver."""

    spec = make_spec({"SUBMISSION_ID": "sub-1"})
    branching = spec.sbatch_env_for_mode(
        mode_spec=spec.mode_spec("branching"),
        run_spec_json=tmp_path / "branching.json",
    )
    structured = spec.sbatch_env_for_mode(
        mode_spec=spec.mode_spec("structured_baseline"),
        run_spec_json=tmp_path / "structured.json",
    )
    epsilon = spec.sbatch_env_for_mode(
        mode_spec=spec.mode_spec("epsilon_greedy"),
        run_spec_json=tmp_path / "epsilon.json",
    )

    assert branching["ALGORITHM_ADV_ESTIMATOR"] == "branch_interpolated_grpo"
    assert branching["SELECTOR_MODE"] == "cluster_across"
    assert branching["N_RESP_PER_PROMPT"] == "1"
    assert branching["PERSISTENT_LOG_INTERVAL_STEPS"] == "10"
    assert branching["REPETITION_CHECKING_ENABLED"] == "True"
    assert structured["ALGORITHM_ADV_ESTIMATOR"] == "grpo"
    assert structured["BRANCH_PROB"] == "0.0"
    assert structured["N_RESP_PER_PROMPT"] == "16"
    assert epsilon["SELECTOR_MODE"] == "embed_diverse_topk_random"
    assert epsilon["BRANCH_FANOUT"] == "1"
    assert epsilon["NUM_CANDIDATES"] == "50"


def test_checkpoint_manifest_path_matches_slurm_default_name() -> None:
    """The Python launcher should predict the Slurm wrapper experiment name."""

    spec = make_spec(
        {
            "SUBMISSION_ID": "sub-1",
            "RUN_LABEL": "label-a",
            "MODEL_NAME_SLUG": "q35_4b",
            "CACHE_ROOT": "/scratch/cache-root",
        }
    )
    mode = spec.mode_spec("branching")

    assert parse_sbatch_job_id("12345;cluster\n") == "12345"
    experiment_name = checkpoint_experiment_name(
        spec=spec, mode_spec=mode, job_id="12345"
    )
    path = checkpoint_manifest_path(spec=spec, experiment_name=experiment_name)

    assert experiment_name == "q35_4b_branching_label-a_12345"
    assert path == Path(
        "/scratch/cache-root/checkpoints/q35_4b_branching_label-a_12345/run_spec.json"
    )
