"""Unit tests for typed run configuration parsing."""

from __future__ import annotations

import yaml
from pathlib import Path

import pytest

from sft_training.config_types import RunConfig


def test_run_config_from_yaml_resolves_paths() -> None:
    """Run config loader should resolve relative dataset/output paths."""
    yaml_path = Path("configs/runs/olmo3_7b_instruct_to_think.yaml")
    config = RunConfig.from_yaml(yaml_path=yaml_path)
    assert config.dataset_path.is_absolute()
    assert config.output_dir.is_absolute()
    assert config.deepspeed_config_path is None
    assert config.num_train_epochs == 8
    assert config.lm_eval.tasks == ("minerva_math500", "aime24", "aime25")
    assert config.lm_eval.aime_avg_k == 32


def test_run_config_allows_null_deepspeed_path(tmp_path: Path) -> None:
    """Run config loader should accept a null deepspeed config value."""
    payload = {
        "run_name": "smoke",
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_path": "../BuildSFTDataset/output/transformed_output.jsonl",
        "output_dir": "/tmp/sft",
        "deepspeed_config_path": None,
        "wandb_project": "proj",
    }
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = RunConfig.from_yaml(yaml_path=yaml_path)
    assert config.deepspeed_config_path is None


def test_run_config_resolves_deepspeed_path_when_provided(tmp_path: Path) -> None:
    """Run config loader should resolve provided DeepSpeed paths to absolute paths."""
    deepspeed_path = tmp_path / "deepspeed_config.json"
    deepspeed_path.write_text("{}", encoding="utf-8")
    payload = {
        "run_name": "smoke",
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_path": "../BuildSFTDataset/output/transformed_output.jsonl",
        "output_dir": "/tmp/sft",
        "deepspeed_config_path": "./deepspeed_config.json",
        "wandb_project": "proj",
    }
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = RunConfig.from_yaml(yaml_path=yaml_path)
    assert config.deepspeed_config_path is not None
    assert config.deepspeed_config_path.is_absolute()
    assert config.deepspeed_config_path == deepspeed_path.resolve()


def test_run_config_parses_lm_eval_log_samples(tmp_path: Path) -> None:
    """Run config loader should parse lm_eval.log_samples boolean overrides."""
    payload = {
        "run_name": "smoke",
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_path": "../BuildSFTDataset/output/transformed_output.jsonl",
        "output_dir": "/tmp/sft",
        "deepspeed_config_path": None,
        "wandb_project": "proj",
        "lm_eval": {"tasks": ["minerva_math500"], "log_samples": True},
    }
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = RunConfig.from_yaml(yaml_path=yaml_path)
    assert config.lm_eval.log_samples is True
    assert config.lm_eval.temperature == 0.6
    assert config.lm_eval.top_p == 0.95
    assert config.lm_eval.max_gen_toks == 32768


def test_run_config_parses_lm_eval_sampling_overrides(tmp_path: Path) -> None:
    """Run config loader should parse lm_eval temperature and top_p overrides."""
    payload = {
        "run_name": "smoke",
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_path": "../BuildSFTDataset/output/transformed_output.jsonl",
        "output_dir": "/tmp/sft",
        "deepspeed_config_path": None,
        "wandb_project": "proj",
        "lm_eval": {
            "tasks": ["minerva_math500"],
            "aime_avg_k": 7,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_gen_toks": 4096,
        },
    }
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    config = RunConfig.from_yaml(yaml_path=yaml_path)
    assert config.lm_eval.aime_avg_k == 7
    assert config.lm_eval.temperature == 0.2
    assert config.lm_eval.top_p == 0.7
    assert config.lm_eval.max_gen_toks == 4096


def test_run_config_rejects_invalid_aime_avg_k(tmp_path: Path) -> None:
    """Run config loader should reject non-positive `lm_eval.aime_avg_k` values."""
    payload = {
        "run_name": "smoke",
        "model_name_or_path": "Qwen/Qwen2.5-0.5B-Instruct",
        "dataset_path": "../BuildSFTDataset/output/transformed_output.jsonl",
        "output_dir": "/tmp/sft",
        "deepspeed_config_path": None,
        "wandb_project": "proj",
        "lm_eval": {"tasks": ["aime24"], "aime_avg_k": 0},
    }
    yaml_path = tmp_path / "run.yaml"
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(AssertionError):
        RunConfig.from_yaml(yaml_path=yaml_path)
