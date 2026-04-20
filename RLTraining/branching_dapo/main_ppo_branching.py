"""Launcher that registers repo-local branching DAPO extensions before PPO startup."""

# pyright: reportMissingImports=false

from __future__ import annotations

import hydra
import ray

from branching_dapo.bootstrap import ensure_repo_paths
from branching_dapo.task_runner import BranchingTaskRunner

ensure_repo_paths()

import branching_dapo.advantage  # noqa: F401,E402
from verl.experimental.reward_loop import migrate_legacy_reward_impl  # noqa: E402
from verl.trainer.main_ppo import run_ppo  # noqa: E402
from verl.utils.device import auto_set_device  # noqa: E402


@hydra.main(config_path="../verl/verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config) -> None:
    """Hydra entrypoint for repo-local branching PPO training.

    Args:
        config: PPO trainer Hydra configuration.

    Returns:
        None.
    """

    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(
        config,
        task_runner_class=ray.remote(num_cpus=1)(BranchingTaskRunner),
    )


if __name__ == "__main__":
    main()  # pyright: ignore[reportCallIssue]
