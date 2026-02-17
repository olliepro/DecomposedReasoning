"""Unit tests for standalone eval logging helpers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest
import yaml

import eval_runner.standalone as standalone_module
from eval_runner.config_types import EvalRunMetadata, LmEvalConfig
from eval_runner.standalone import (
    apply_eval_overrides,
    build_eval_log_payload,
    checkpoint_step_from_name,
    infer_eval_step,
    load_eval_config,
    maybe_init_wandb_eval_run,
    maybe_log_eval_metrics_to_wandb,
    parse_args,
)


def test_checkpoint_step_from_name_parses_checkpoint_steps() -> None:
    """Step parser should extract numeric suffix from checkpoint names."""
    assert checkpoint_step_from_name(checkpoint_name="checkpoint-200") == 200


def test_checkpoint_step_from_name_ignores_invalid_names() -> None:
    """Step parser should return None when name format is unsupported."""
    assert checkpoint_step_from_name(checkpoint_name="final_model") is None
    assert checkpoint_step_from_name(checkpoint_name="checkpoint-last") is None


def test_infer_eval_step_for_final_model_uses_latest_checkpoint(tmp_path: Path) -> None:
    """Final-model eval should infer train step from the latest checkpoint folder."""
    (tmp_path / "checkpoint-2").mkdir()
    (tmp_path / "checkpoint-10").mkdir()
    final_model_path = tmp_path / "final_model"
    final_model_path.mkdir()
    assert infer_eval_step(checkpoint=final_model_path) == 10


def test_build_eval_log_payload_adds_step_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    """Payload builder should inject explicit step metrics for checkpoint evals."""
    result_path = tmp_path / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        standalone_module,
        "load_and_flatten_metrics",
        lambda result_json_path: {"bench/math500/math_verify": 0.5},
    )
    payload, step = build_eval_log_payload(
        result_path=result_path,
        checkpoint=Path("checkpoint-12"),
    )
    assert step == 12
    assert payload["bench/math500/math_verify"] == 0.5
    assert payload["train/global_step"] == 12.0
    assert payload["eval/checkpoint_step"] == 12.0


def test_build_eval_log_payload_without_step_keeps_metrics(
    tmp_path: Path, monkeypatch
) -> None:
    """Payload builder should leave metrics unchanged when step is unavailable."""
    result_path = tmp_path / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        standalone_module,
        "load_and_flatten_metrics",
        lambda result_json_path: {"bench/math500/math_verify": 0.2},
    )
    payload, step = build_eval_log_payload(
        result_path=result_path,
        checkpoint=Path("model_without_step"),
    )
    assert step is None
    assert payload == {"bench/math500/math_verify": 0.2}


def test_apply_eval_overrides_keeps_config_without_cli_value() -> None:
    """CLI override helper should preserve config when no override is provided."""
    config = LmEvalConfig(log_samples=False, temperature=0.6, top_p=0.95)
    args = argparse.Namespace(log_samples=None, temperature=None, top_p=None)
    updated = apply_eval_overrides(config=config, args=args)
    assert updated == config


def test_apply_eval_overrides_updates_log_samples() -> None:
    """CLI override helper should update log_samples when requested."""
    config = LmEvalConfig(log_samples=False, temperature=0.6, top_p=0.95)
    args = argparse.Namespace(log_samples=True, temperature=None, top_p=None)
    updated = apply_eval_overrides(config=config, args=args)
    assert updated.log_samples is True


def test_apply_eval_overrides_updates_sampling_values() -> None:
    """CLI override helper should update temperature and top_p when requested."""
    config = LmEvalConfig(temperature=0.6, top_p=0.95)
    args = argparse.Namespace(log_samples=None, temperature=0.2, top_p=0.7)
    updated = apply_eval_overrides(config=config, args=args)
    assert updated.temperature == 0.2
    assert updated.top_p == 0.7


def test_apply_eval_overrides_rejects_invalid_top_p() -> None:
    """CLI override helper should reject invalid top_p values."""
    config = LmEvalConfig(temperature=0.6, top_p=0.95)
    args = argparse.Namespace(log_samples=None, temperature=None, top_p=1.5)
    with pytest.raises(AssertionError):
        apply_eval_overrides(config=config, args=args)


def test_parse_args_supports_limit(monkeypatch) -> None:
    """Parser should accept the standalone `--limit` argument.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "standalone.py",
            "--checkpoint",
            "/tmp/checkpoint",
            "--config",
            "/tmp/config.yaml",
            "--limit",
            "5",
        ],
    )
    args = parse_args()
    assert args.limit == 5


def test_load_eval_config_supports_lm_eval_only_yaml(tmp_path: Path) -> None:
    """Loader should parse standalone lm_eval-only yaml files.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    config_path = tmp_path / "lm_eval.yaml"
    config_path.write_text(yaml.safe_dump({"tasks": ["aime24"]}), encoding="utf-8")

    config, metadata = load_eval_config(config_path=config_path)

    assert config.tasks == ("aime24",)
    assert metadata is None


def test_load_eval_config_supports_run_yaml_with_base(tmp_path: Path) -> None:
    """Loader should parse full run yaml files with `base_config` inheritance.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    base_eval_path = tmp_path / "base_eval.yaml"
    base_eval_path.write_text(yaml.safe_dump({"tasks": ["aime25"]}), encoding="utf-8")
    base_run_path = tmp_path / "base_run.yaml"
    base_run_path.write_text(
        yaml.safe_dump(
            {
                "wandb_project": "proj",
                "wandb_entity": "entity",
                "lm_eval": "base_eval.yaml",
            }
        ),
        encoding="utf-8",
    )
    run_config_path = tmp_path / "run.yaml"
    run_config_path.write_text(
        yaml.safe_dump(
            {
                "base_config": "base_run.yaml",
                "run_name": "example",
                "model_name_or_path": "Qwen/Qwen3-8B",
                "output_dir": "/tmp/out",
            }
        ),
        encoding="utf-8",
    )

    config, metadata = load_eval_config(config_path=run_config_path)

    assert config.tasks == ("aime25",)
    assert isinstance(metadata, EvalRunMetadata)
    assert metadata.run_name == "example"
    assert metadata.wandb_project == "proj"


def test_maybe_init_wandb_eval_run_reuses_shared_train_run(monkeypatch) -> None:
    """Eval init should resume the shared train run when run id is provided."""
    captured_init_kwargs: dict[str, object] = {}
    captured_metric_names: list[str] = []

    def fake_init(**kwargs: object) -> None:
        captured_init_kwargs.update(kwargs)

    def fake_define_metric(*, name: str, step_metric: str | None = None) -> None:
        metric_repr = name if step_metric is None else f"{name}:{step_metric}"
        captured_metric_names.append(metric_repr)

    monkeypatch.setenv(name="SFT_WANDB_RUN_ID", value="shared123")
    monkeypatch.setenv(
        name="SFT_WANDB_RUN_NAME",
        value="my_train_run",
    )
    monkeypatch.setenv(
        name="SFT_WANDB_GROUP",
        value="my_group",
    )
    monkeypatch.setenv(
        name="SFT_JOB_TIMESTAMP",
        value="2026-02-12_21-51-51",
    )
    monkeypatch.setattr(standalone_module.wandb, "init", fake_init, raising=False)
    monkeypatch.setattr(
        standalone_module.wandb,
        "define_metric",
        fake_define_metric,
        raising=False,
    )

    run_config = EvalRunMetadata(
        run_name="base",
        wandb_project="project",
    )
    maybe_init_wandb_eval_run(
        run_config=run_config,
        checkpoint=Path("checkpoint-8"),
        config_path=Path("/tmp/run.yaml"),
    )

    assert captured_init_kwargs["id"] == "shared123"
    assert captured_init_kwargs["resume"] == "allow"
    assert captured_init_kwargs["name"] == "my_train_run"
    assert captured_init_kwargs["group"] == "my_group"
    assert captured_init_kwargs["job_type"] == "train"
    config_payload = captured_init_kwargs["config"]
    assert isinstance(config_payload, dict)
    assert config_payload["eval_checkpoint_step"] == 8
    assert "train/global_step" in captured_metric_names
    assert "bench/*:train/global_step" in captured_metric_names


def test_maybe_log_eval_metrics_without_explicit_step_for_shared_run(
    monkeypatch,
) -> None:
    """Shared run mode should not pass `step` to `wandb.log`."""
    captured_log_kwargs: dict[str, object] = {}

    monkeypatch.setenv(name="SFT_WANDB_RUN_ID", value="shared123")
    monkeypatch.setattr(
        standalone_module,
        "build_eval_log_payload",
        lambda result_path, checkpoint: ({"bench/math500/math_verify": 0.3}, 2),
    )
    monkeypatch.setattr(standalone_module.wandb, "run", object(), raising=False)
    monkeypatch.setattr(
        standalone_module.wandb,
        "log",
        lambda **kwargs: captured_log_kwargs.update(kwargs),
        raising=False,
    )
    monkeypatch.setattr(standalone_module.wandb, "finish", lambda: None, raising=False)

    maybe_log_eval_metrics_to_wandb(result_path=Path("/tmp/out.json"), checkpoint=Path("checkpoint-2"))

    assert "step" not in captured_log_kwargs
    assert captured_log_kwargs["data"] == {"bench/math500/math_verify": 0.3}


def test_maybe_log_eval_metrics_with_explicit_step_without_shared_run(
    monkeypatch,
) -> None:
    """Standalone eval run should use explicit W&B step when available."""
    captured_log_kwargs: dict[str, object] = {}

    monkeypatch.delenv(name="SFT_WANDB_RUN_ID", raising=False)
    monkeypatch.setattr(
        standalone_module,
        "build_eval_log_payload",
        lambda result_path, checkpoint: ({"bench/math500/math_verify": 0.3}, 2),
    )
    monkeypatch.setattr(standalone_module.wandb, "run", object(), raising=False)
    monkeypatch.setattr(
        standalone_module.wandb,
        "log",
        lambda **kwargs: captured_log_kwargs.update(kwargs),
        raising=False,
    )
    monkeypatch.setattr(standalone_module.wandb, "finish", lambda: None, raising=False)

    maybe_log_eval_metrics_to_wandb(result_path=Path("/tmp/out.json"), checkpoint=Path("checkpoint-2"))

    assert captured_log_kwargs["step"] == 2
    assert captured_log_kwargs["data"] == {"bench/math500/math_verify": 0.3}
