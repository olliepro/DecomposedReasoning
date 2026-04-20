"""Unit tests for checkpoint sweep discovery, ranking, and dry-run submission."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from eval_runner.checkpoint_sweep import (
    collect_checkpoint_rankings,
    discover_checkpoint_targets,
    find_aime_metric,
    write_ranking_artifacts,
)


def write_eval_result(
    *,
    output_path: Path,
    task_name: str,
    metric_name: str,
    metric_value: float,
) -> None:
    """Write one synthetic `lm-eval` result file for tests.

    Args:
        output_path: Result JSON path to write.
        task_name: Benchmark task name.
        metric_name: Raw metric key inside `results[task_name]`.
        metric_value: Raw metric value.

    Returns:
        None.
    """

    payload = {"results": {task_name: {metric_name: metric_value}}}
    output_path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_checkpoint_targets_orders_raw_dirs(tmp_path: Path) -> None:
    """Discovery should return raw checkpoints in numeric order and final model last."""
    for name in ("checkpoint-112", "checkpoint-56", "notes", "final_model"):
        (tmp_path / name).mkdir()

    targets = discover_checkpoint_targets(run_dir=tmp_path)

    assert [target.label for target in targets] == [
        "checkpoint-56",
        "checkpoint-112",
        "final_model",
    ]


def test_find_aime_metric_supports_dynamic_avg_k() -> None:
    """AIME metric lookup should accept any flattened `avg_at_k` key."""
    metrics = {"bench/aime24/avg_at_7": 0.25}
    assert find_aime_metric(metrics=metrics, task_name="aime24") == 0.25


def test_write_ranking_artifacts_uses_aime_first_policy(tmp_path: Path) -> None:
    """Ranking should prioritize mean AIME score, then Math500, then later step."""
    benchmark_dir = tmp_path / "benchmark_evals"
    benchmark_dir.mkdir()
    for label in ("checkpoint-56", "checkpoint-112", "final_model"):
        (tmp_path / label).mkdir()

    write_eval_result(
        output_path=benchmark_dir / "checkpoint-56_aime24.json",
        task_name="aime24",
        metric_name="avg_at_32,avg@32",
        metric_value=0.40,
    )
    write_eval_result(
        output_path=benchmark_dir / "checkpoint-56_aime25.json",
        task_name="aime25",
        metric_name="avg_at_32,avg@32",
        metric_value=0.50,
    )
    write_eval_result(
        output_path=benchmark_dir / "checkpoint-56_minerva_math500.json",
        task_name="minerva_math500",
        metric_name="math_verify,none",
        metric_value=0.20,
    )

    write_eval_result(
        output_path=benchmark_dir / "checkpoint-112_aime24.json",
        task_name="aime24",
        metric_name="avg_at_32,avg@32",
        metric_value=0.40,
    )
    write_eval_result(
        output_path=benchmark_dir / "checkpoint-112_aime25.json",
        task_name="aime25",
        metric_name="avg_at_32,avg@32",
        metric_value=0.50,
    )
    write_eval_result(
        output_path=benchmark_dir / "checkpoint-112_minerva_math500.json",
        task_name="minerva_math500",
        metric_name="math_verify,none",
        metric_value=0.30,
    )

    write_eval_result(
        output_path=benchmark_dir / "final_model_aime24.json",
        task_name="aime24",
        metric_name="avg_at_32,avg@32",
        metric_value=0.20,
    )
    write_eval_result(
        output_path=benchmark_dir / "final_model_aime25.json",
        task_name="aime25",
        metric_name="avg_at_32,avg@32",
        metric_value=0.20,
    )
    write_eval_result(
        output_path=benchmark_dir / "final_model_minerva_math500.json",
        task_name="minerva_math500",
        metric_name="math_verify,none",
        metric_value=0.95,
    )

    ranking_entries, incomplete_entries = write_ranking_artifacts(run_dir=tmp_path)

    assert incomplete_entries == []
    assert [entry.checkpoint_label for entry in ranking_entries] == [
        "checkpoint-112",
        "checkpoint-56",
        "final_model",
    ]
    assert (benchmark_dir / "best_checkpoint.txt").read_text(encoding="utf-8") == str(
        tmp_path / "checkpoint-112"
    )


def test_collect_checkpoint_rankings_marks_incomplete_outputs(tmp_path: Path) -> None:
    """Checkpoints with missing task outputs should be tracked as incomplete."""
    benchmark_dir = tmp_path / "benchmark_evals"
    benchmark_dir.mkdir()
    (tmp_path / "checkpoint-56").mkdir()
    write_eval_result(
        output_path=benchmark_dir / "checkpoint-56_aime24.json",
        task_name="aime24",
        metric_name="avg_at_32,avg@32",
        metric_value=0.1,
    )

    ranking_entries, incomplete_entries = collect_checkpoint_rankings(run_dir=tmp_path)

    assert ranking_entries == []
    assert len(incomplete_entries) == 1
    assert incomplete_entries[0].missing_tasks == ("aime25", "minerva_math500")


def test_matrix_dry_run_resolves_olmo_raw_checkpoints_and_group_env(tmp_path: Path) -> None:
    """Matrix dry run should enumerate raw checkpoints and export grouped W&B env."""
    run_dir = tmp_path / "olmo_run"
    for label in (
        "checkpoint-56",
        "checkpoint-112",
        "checkpoint-168",
        "checkpoint-224",
        "checkpoint-280",
        "checkpoint-336",
        "checkpoint-392",
        "checkpoint-448",
        "final_model",
    ):
        (run_dir / label).mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "config.yaml"
    sbatch_path = tmp_path / "standalone_eval.sbatch"
    config_path.write_text("{}", encoding="utf-8")
    sbatch_path.write_text("#!/bin/bash\n", encoding="utf-8")

    script_path = (
        Path(__file__).resolve().parents[1] / "slurm" / "matrix.sh"
    )
    env = dict(os.environ)
    env.update(
        {
            "RUN_DIR": str(run_dir),
            "CONFIG": str(config_path),
            "SBATCH_SCRIPT": str(sbatch_path),
            "TASKS": "minerva_math500 aime24 aime25",
            "INCLUDE_BASELINE_MODEL": "0",
            "MATRIX_DRY_RUN": "1",
            "PYTHON_BIN": sys.executable,
            "SFT_JOB_TIMESTAMP": "2026-03-17T12-00-00Z",
        }
    )

    result = subprocess.run(
        ["bash", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
        cwd=str(Path(__file__).resolve().parents[1]),
    )

    assert "Submitted 27 eval jobs." in result.stdout
    assert "checkpoint-56" in result.stdout
    assert "final_model" in result.stdout
    assert "SFT_WANDB_GROUP=olmo_run_checkpoint_eval_2026-03-17T12-00-00Z" in result.stdout
    assert "SFT_WANDB_RUN_ID=" in result.stdout
