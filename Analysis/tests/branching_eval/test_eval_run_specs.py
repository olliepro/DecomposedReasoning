"""Tests for typed branching-eval run specs."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from branching_eval.run_specs import EvalRunSpec, split_doc_ids


def test_split_doc_ids_accepts_commas_and_spaces() -> None:
    """Doc-id parsing should support shell-friendly values."""

    assert split_doc_ids(raw_doc_ids="6,12 18") == (6, 12, 18)


def test_generated_structured_config_writes_manifest(tmp_path: Path) -> None:
    """Generated run specs should write config YAML and JSON sidecar."""

    spec = EvalRunSpec.from_env(
        env={
            "RUN_NAME": "unit-structured",
            "MODEL_PATH": "/tmp/checkpoint",
            "EVAL_MODE": "structured",
            "DOC_IDS": "6,12",
            "SPEC_ROOT": str(tmp_path / "specs"),
            "OUTPUT_ROOT": str(tmp_path / "outputs/unit-structured"),
            "GPU_COUNT": "2",
            "TENSOR_PARALLEL_SIZE": "2",
        }
    )

    spec.write()

    payload = yaml.safe_load(spec.config_path.read_text(encoding="utf-8"))
    assert payload["models"][0]["checkpoint_or_repo"] == "/tmp/checkpoint"
    assert payload["serve"]["tensor_parallel_size"] == 2
    assert payload["artifacts"]["output_root"] == str(
        tmp_path / "outputs/unit-structured"
    )
    assert payload["run_matrix"]["include_structured_baselines"] is True
    assert payload["run_matrix"]["include_epsilon_greedy"] is False
    assert spec.local_command()[-4:] == ["--doc-id", "6", "--doc-id", "12"]
    manifest = json.loads(spec.manifest_path.read_text(encoding="utf-8"))
    assert manifest["shape"]["run_name"] == "unit-structured"


def test_sbatch_args_include_resources_and_test_only(tmp_path: Path) -> None:
    """Slurm args should be explicit and forecastable."""

    spec = EvalRunSpec.from_env(
        env={
            "RUN_NAME": "unit-epsilon",
            "MODEL_PATH": "Qwen/Qwen3-8B",
            "EVAL_MODE": "epsilon",
            "SPEC_ROOT": str(tmp_path / "specs"),
            "ACCOUNT": "PAA0201",
            "PARTITION": "preemptible-quad",
            "GPU_COUNT": "4",
            "GPU_TYPE": "a100",
        }
    )

    args = spec.slurm.sbatch_args(job_name=spec.shape.run_name, test_only=True)

    assert "--test-only" in args
    assert "--account=PAA0201" in args
    assert "--partition=preemptible-quad" in args
    assert "--gres=gpu:a100:4" in args
