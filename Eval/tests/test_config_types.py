"""Unit tests for standalone eval configuration parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eval_runner.config_types import parse_eval_config


def test_parse_eval_only_config_uses_defaults(tmp_path: Path) -> None:
    """Parser should apply default lm_eval values when keys are omitted.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "lm_eval.yaml"
    config_path.write_text(yaml.safe_dump({}), encoding="utf-8")

    config, metadata = parse_eval_config(config_path=config_path)

    assert config.tasks == ("minerva_math500", "aime24", "aime25")
    assert config.aime_avg_k == 32
    assert metadata is None


def test_parse_eval_only_config_supports_overrides(tmp_path: Path) -> None:
    """Parser should apply explicit lm_eval overrides from yaml.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "lm_eval.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tasks": ["aime24"],
                "aime_avg_k": 7,
                "temperature": 0.2,
                "top_p": 0.7,
            }
        ),
        encoding="utf-8",
    )

    config, metadata = parse_eval_config(config_path=config_path)

    assert config.tasks == ("aime24",)
    assert config.aime_avg_k == 7
    assert config.temperature == 0.2
    assert config.top_p == 0.7
    assert metadata is None


def test_parse_eval_config_rejects_invalid_aime_avg_k(tmp_path: Path) -> None:
    """Parser should reject non-positive `aime_avg_k` values.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "lm_eval.yaml"
    config_path.write_text(yaml.safe_dump({"aime_avg_k": 0}), encoding="utf-8")

    with pytest.raises(AssertionError):
        parse_eval_config(config_path=config_path)


def test_parse_run_config_supports_base_merge_and_lm_eval_path(tmp_path: Path) -> None:
    """Parser should merge base config and resolve lm_eval path pointers.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    base_lm_eval_path = tmp_path / "lm_eval_base.yaml"
    base_lm_eval_path.write_text(
        yaml.safe_dump({"tasks": ["aime25"], "aime_avg_k": 64}),
        encoding="utf-8",
    )
    base_run_path = tmp_path / "base_run.yaml"
    base_run_path.write_text(
        yaml.safe_dump(
            {
                "wandb_project": "proj",
                "wandb_entity": "entity",
                "lm_eval": "lm_eval_base.yaml",
            }
        ),
        encoding="utf-8",
    )
    run_config_path = tmp_path / "run.yaml"
    run_config_path.write_text(
        yaml.safe_dump(
            {
                "base_config": "base_run.yaml",
                "run_name": "my_run",
                "model_name_or_path": "Qwen/Qwen3-8B",
                "output_dir": "/tmp/out",
            }
        ),
        encoding="utf-8",
    )

    config, metadata = parse_eval_config(config_path=run_config_path)

    assert config.tasks == ("aime25",)
    assert config.aime_avg_k == 64
    assert metadata is not None
    assert metadata.run_name == "my_run"
    assert metadata.wandb_project == "proj"
    assert metadata.wandb_entity == "entity"
