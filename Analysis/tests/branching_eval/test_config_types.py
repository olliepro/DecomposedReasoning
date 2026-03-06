"""Tests for branching-eval configuration parsing."""

from __future__ import annotations

from pathlib import Path

import yaml

from branching_eval.config_types import load_branching_eval_config


def test_config_defaults_parse_from_minimal_payload(tmp_path: Path) -> None:
    """Minimal config should parse with defaults and validate."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"models": [{"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}]}
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    assert config.tasks.task_names == ("aime24",)
    assert config.decoding.temperature == 0.6
    assert config.decoding.decode_chunk_tokens == 512
    assert config.branching.num_candidates == 100
    assert config.branching.steer_repetition_penalty == 1.01
    assert config.serve.scheduling_policy == "priority"
    assert config.serve.kv_offloading_size_gb == 64.0
    assert config.serve.kv_offloading_backend == "native"


def test_calibration_and_output_paths_resolve_relative_to_config(
    tmp_path: Path,
) -> None:
    """Relative file paths should resolve from config parent directory."""

    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "branching.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}
                ],
                "calibration": {"entropy_threshold_path": "calibration.json"},
                "artifacts": {"output_root": "runs"},
            }
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    assert (
        config.calibration.entropy_threshold_path
        == (config_dir / "calibration.json").resolve()
    )
    assert config.artifacts.output_root == (config_dir / "runs").resolve()


def test_scheduling_policy_parses_from_serve_block(tmp_path: Path) -> None:
    """Serve scheduling and KV offload settings should parse from YAML."""

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "models": [
                    {"model_id": "non_sft", "checkpoint_or_repo": "Qwen/Qwen3-8B"}
                ],
                "serve": {
                    "scheduling_policy": "priority",
                    "kv_offloading_size_gb": 12.0,
                    "kv_offloading_backend": "lmcache",
                },
                "branching": {"steer_repetition_penalty": 1.2},
            }
        ),
        encoding="utf-8",
    )
    config = load_branching_eval_config(config_path=config_path)
    assert config.serve.scheduling_policy == "priority"
    assert config.serve.kv_offloading_size_gb == 12.0
    assert config.serve.kv_offloading_backend == "lmcache"
    assert config.branching.steer_repetition_penalty == 1.2
