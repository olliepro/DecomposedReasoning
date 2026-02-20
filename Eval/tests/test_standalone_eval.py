"""Unit tests for standalone eval logging helpers."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pytest
import yaml

import eval_runner.standalone as standalone_module
from eval_runner.config_types import EvalRunMetadata, LmEvalConfig
from eval_runner.standalone import (
    ApproxCompletedRequestsTracker,
    apply_eval_overrides,
    build_eval_log_payload,
    checkpoint_step_from_name,
    configure_vllm_runtime_logging,
    parse_vllm_stats_log_line,
    VllmStatsWandbHandler,
    infer_eval_step,
    load_eval_config,
    maybe_attach_vllm_stats_wandb_handler,
    detach_vllm_stats_wandb_handler,
    maybe_init_wandb_eval_run,
    maybe_log_eval_metrics_to_wandb,
    parse_args,
)


def test_checkpoint_step_from_name_parses_checkpoint_steps() -> None:
    """Step parser should extract numeric suffix from checkpoint names."""
    assert checkpoint_step_from_name(checkpoint_name="checkpoint-200") == 200
    assert checkpoint_step_from_name(checkpoint_name="checkpoint-200-hf") == 200
    assert checkpoint_step_from_name(checkpoint_name="checkpoint-200-vllm") == 200


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
    args = argparse.Namespace(log_samples=None, temperature=None, top_p=None, tasks=None)
    updated = apply_eval_overrides(config=config, args=args)
    assert updated == config


def test_apply_eval_overrides_updates_log_samples() -> None:
    """CLI override helper should update log_samples when requested."""
    config = LmEvalConfig(log_samples=False, temperature=0.6, top_p=0.95)
    args = argparse.Namespace(log_samples=True, temperature=None, top_p=None, tasks=None)
    updated = apply_eval_overrides(config=config, args=args)
    assert updated.log_samples is True


def test_apply_eval_overrides_updates_sampling_values() -> None:
    """CLI override helper should update temperature and top_p when requested."""
    config = LmEvalConfig(temperature=0.6, top_p=0.95)
    args = argparse.Namespace(
        log_samples=None,
        temperature=0.2,
        top_p=0.7,
        tasks=None,
    )
    updated = apply_eval_overrides(config=config, args=args)
    assert updated.temperature == 0.2
    assert updated.top_p == 0.7


def test_apply_eval_overrides_rejects_invalid_top_p() -> None:
    """CLI override helper should reject invalid top_p values."""
    config = LmEvalConfig(temperature=0.6, top_p=0.95)
    args = argparse.Namespace(
        log_samples=None,
        temperature=None,
        top_p=1.5,
        tasks=None,
    )
    with pytest.raises(AssertionError):
        apply_eval_overrides(config=config, args=args)


def test_apply_eval_overrides_updates_tasks() -> None:
    """CLI override helper should update tasks when requested."""
    config = LmEvalConfig(tasks=("aime24", "aime25"))
    args = argparse.Namespace(
        log_samples=None,
        temperature=None,
        top_p=None,
        tasks=["aime24"],
    )
    updated = apply_eval_overrides(config=config, args=args)
    assert updated.tasks == ("aime24",)


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


def test_parse_args_supports_tasks_override(monkeypatch) -> None:
    """Parser should accept the standalone `--tasks` argument."""
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "standalone.py",
            "--checkpoint",
            "/tmp/checkpoint",
            "--config",
            "/tmp/config.yaml",
            "--tasks",
            "aime24",
            "aime25",
        ],
    )
    args = parse_args()
    assert args.tasks == ["aime24", "aime25"]


def test_configure_vllm_runtime_logging_sets_expected_env(monkeypatch) -> None:
    """vLLM runtime logging helper should set env vars for stats heartbeat."""
    monkeypatch.delenv(name="VLLM_LOGGING_LEVEL", raising=False)
    monkeypatch.delenv(name="VLLM_LOG_STATS_INTERVAL", raising=False)

    applied, message = configure_vllm_runtime_logging(
        config=LmEvalConfig(
            model_type="vllm",
            vllm_log_stats_interval=5.0,
            vllm_logging_level="INFO",
            vllm_disable_log_stats=False,
        )
    )

    assert applied is True
    assert "VLLM_LOGGING_LEVEL=INFO" in message
    assert "VLLM_LOG_STATS_INTERVAL=5.0" in message
    assert "disable_log_stats=False" in message


def test_parse_vllm_stats_log_line_extracts_runtime_metrics() -> None:
    """Parser should extract throughput and queue stats from vLLM log lines."""
    sample_line = (
        "INFO 02-20 13:21:15 [loggers.py:257] Engine 000: "
        "Avg prompt throughput: 0.0 tokens/s, "
        "Avg generation throughput: 1504.7 tokens/s, "
        "Running: 68 reqs, Waiting: 394 reqs, "
        "GPU KV cache usage: 99.6%, Prefix cache hit rate: 86.6%"
    )
    payload = parse_vllm_stats_log_line(log_line=sample_line)
    assert payload is not None
    assert payload["runtime/vllm/prompt_tokens_per_s"] == 0.0
    assert payload["runtime/vllm/generation_tokens_per_s"] == 1504.7
    assert payload["runtime/vllm/running_reqs"] == 68.0
    assert payload["runtime/vllm/waiting_reqs"] == 394.0
    assert payload["runtime/vllm/gpu_kv_cache_usage_pct"] == 99.6
    assert payload["runtime/vllm/prefix_cache_hit_rate_pct"] == 86.6


def test_approx_completed_requests_tracker_uses_peak_outstanding() -> None:
    """Completed requests should be peak outstanding minus current outstanding."""
    tracker = ApproxCompletedRequestsTracker()

    first_payload = tracker.with_completed_requests(
        payload={
            "runtime/vllm/running_reqs": 6.0,
            "runtime/vllm/waiting_reqs": 4.0,
        }
    )
    second_payload = tracker.with_completed_requests(
        payload={
            "runtime/vllm/running_reqs": 3.0,
            "runtime/vllm/waiting_reqs": 2.0,
        }
    )

    assert first_payload["runtime/vllm/completed_requests"] == 0.0
    assert second_payload["runtime/vllm/completed_requests"] == 5.0


def test_vllm_stats_wandb_handler_logs_parsed_payload(monkeypatch) -> None:
    """vLLM log handler should forward parsed stats payloads to W&B."""
    captured_log_payloads: list[dict[str, float]] = []
    monkeypatch.setattr(standalone_module.wandb, "run", object(), raising=False)
    monkeypatch.setattr(
        standalone_module.wandb,
        "log",
        lambda *, data: captured_log_payloads.append(data),
        raising=False,
    )
    handler = VllmStatsWandbHandler()
    record = logging.LogRecord(
        name="vllm.v1.metrics.loggers",
        level=logging.INFO,
        pathname="",
        lineno=0,
        msg=(
            "Engine 000: Avg prompt throughput: 1.2 tokens/s, "
            "Avg generation throughput: 3.4 tokens/s, Running: 2 reqs, "
            "Waiting: 5 reqs, GPU KV cache usage: 90.0%, Prefix cache hit rate: 12.5%"
        ),
        args=(),
        exc_info=None,
    )
    handler.emit(record=record)
    assert captured_log_payloads
    assert captured_log_payloads[0]["runtime/vllm/generation_tokens_per_s"] == 3.4


def test_attach_and_detach_vllm_stats_handler(monkeypatch) -> None:
    """Attach helper should install and remove handler cleanly."""
    monkeypatch.setattr(standalone_module.wandb, "run", object(), raising=False)
    vllm_logger = logging.getLogger("vllm.v1.metrics.loggers")
    existing_handlers = list(vllm_logger.handlers)
    handler, message = maybe_attach_vllm_stats_wandb_handler(
        config=LmEvalConfig(model_type="vllm", vllm_disable_log_stats=False)
    )
    assert handler is not None
    assert "attached" in message
    detach_vllm_stats_wandb_handler(handler=handler)
    assert all(installed_handler in vllm_logger.handlers for installed_handler in existing_handlers)


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


def test_maybe_init_wandb_eval_run_appends_checkpoint_to_name_without_shared_run(
    monkeypatch,
) -> None:
    """Eval run names should include checkpoint token when not sharing run id."""
    captured_init_kwargs: dict[str, object] = {}

    def fake_init(**kwargs: object) -> None:
        captured_init_kwargs.update(kwargs)

    monkeypatch.delenv(name="SFT_WANDB_RUN_ID", raising=False)
    monkeypatch.setenv(name="SFT_WANDB_RUN_NAME", value="eval_base")
    monkeypatch.setenv(name="SFT_WANDB_GROUP", value="group_base")
    monkeypatch.setattr(standalone_module.wandb, "init", fake_init, raising=False)
    monkeypatch.setattr(
        standalone_module.wandb,
        "define_metric",
        lambda **_: None,
        raising=False,
    )

    run_config = EvalRunMetadata(
        run_name="base",
        wandb_project="project",
    )
    maybe_init_wandb_eval_run(
        run_config=run_config,
        checkpoint=Path("checkpoint-8-hf"),
        config_path=Path("/tmp/run.yaml"),
        task_names=("aime24",),
    )

    assert captured_init_kwargs["name"] == "eval_base_checkpoint-8_task-aime24"
    assert captured_init_kwargs["group"] == "group_base"
    assert captured_init_kwargs["job_type"] == "eval"


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
