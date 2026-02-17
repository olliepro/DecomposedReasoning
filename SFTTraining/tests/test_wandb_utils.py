"""Unit tests for W&B run naming utilities."""

from __future__ import annotations

from sft_training.wandb_utils import build_wandb_run_context


def test_build_wandb_run_context_uses_env_overrides() -> None:
    """Explicit env values should control run/group naming."""
    context = build_wandb_run_context(
        base_run_name="smoke",
        job_type="eval",
        env={
            "SFT_JOB_TIMESTAMP": "2026-02-12_15-00-00",
            "SFT_WANDB_GROUP": "custom_group",
            "SFT_WANDB_RUN_NAME": "custom_run",
        },
    )
    assert context.timestamp == "2026-02-12_15-00-00"
    assert context.group_name == "custom_group"
    assert context.run_name == "custom_run"
    assert context.job_type == "eval"


def test_build_wandb_run_context_formats_defaults_from_timestamp() -> None:
    """Default names should include base run and job type."""
    context = build_wandb_run_context(
        base_run_name="smoke",
        job_type="train",
        env={"SFT_JOB_TIMESTAMP": "2026-02-12_15-00-00"},
    )
    assert context.group_name == "smoke_2026-02-12_15-00-00"
    assert context.run_name == "smoke_train_2026-02-12_15-00-00"
