"""Utilities for consistent W&B run naming across train and eval jobs."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping


def utc_job_timestamp() -> str:
    """Return a filesystem-safe UTC timestamp string.

    Returns:
        UTC timestamp formatted as `YYYY-MM-DD_HH-MM-SS`.
    """
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")


@dataclass(frozen=True)
class WandbRunContext:
    """W&B naming context shared by one logical train/eval pair.

    Args:
        base_run_name: Base name from run configuration.
        timestamp: Shared timestamp token for this launch.
        group_name: W&B group name linking related runs.
        run_name: Concrete run name for one job.
        job_type: W&B job type (for example `train` or `eval`).
    """

    base_run_name: str
    timestamp: str
    group_name: str
    run_name: str
    job_type: str

    def as_init_kwargs(self) -> dict[str, str]:
        """Convert context to keyword args for `wandb.init`.

        Returns:
            Mapping containing `name`, `group`, and `job_type`.
        """
        return {
            "name": self.run_name,
            "group": self.group_name,
            "job_type": self.job_type,
        }


def build_wandb_run_context(
    base_run_name: str,
    job_type: str,
    env: Mapping[str, str] | None = None,
) -> WandbRunContext:
    """Build timestamped W&B run/group names for one job.

    Args:
        base_run_name: Base run name from configuration.
        job_type: Job kind label (for example `train` or `eval`).
        env: Optional environment mapping used for overrides.

    Returns:
        Structured W&B run context.

    Example:
        >>> context = build_wandb_run_context(
        ...     base_run_name="smoke_test_run",
        ...     job_type="eval",
        ...     env={"SFT_JOB_TIMESTAMP": "2026-02-12_15-00-00"},
        ... )
        >>> context.run_name
        'smoke_test_run_eval_2026-02-12_15-00-00'
    """
    active_env = os.environ if env is None else env
    timestamp = active_env.get("SFT_JOB_TIMESTAMP", utc_job_timestamp())
    group_name = active_env.get("SFT_WANDB_GROUP", f"{base_run_name}_{timestamp}")
    run_name = active_env.get(
        "SFT_WANDB_RUN_NAME",
        f"{base_run_name}_{job_type}_{timestamp}",
    )
    return WandbRunContext(
        base_run_name=base_run_name,
        timestamp=timestamp,
        group_name=group_name,
        run_name=run_name,
        job_type=job_type,
    )
