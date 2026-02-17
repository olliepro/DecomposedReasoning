"""Standalone benchmark evaluation entrypoint."""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import wandb

from eval_runner.config_types import EvalRunMetadata, LmEvalConfig, parse_eval_config
from eval_runner.run_lm_eval import (
    find_sample_log_files,
    load_and_flatten_metrics,
    run_lm_eval_for_checkpoint,
)
from eval_runner.topology_env import maybe_set_cross_numa_vllm_env
from eval_runner.wandb_utils import build_wandb_run_context


def checkpoint_step_from_name(checkpoint_name: str) -> int | None:
    """Parse global-step value from `checkpoint-<step>` directory names.

    Args:
        checkpoint_name: Directory basename.

    Returns:
        Parsed integer step when the name follows `checkpoint-<step>`.
    """
    prefix = "checkpoint-"
    if not checkpoint_name.startswith(prefix):
        return None
    step_suffix = checkpoint_name[len(prefix) :]
    if not step_suffix.isdigit():
        return None
    return int(step_suffix)


def infer_eval_step(checkpoint: Path) -> int | None:
    """Infer a train global step for eval logging from checkpoint path.

    Args:
        checkpoint: Evaluated checkpoint directory path.

    Returns:
        Train global step when inferable, otherwise `None`.

    Example:
        >>> infer_eval_step(checkpoint=Path("checkpoint-200"))
        200
    """
    direct_step = checkpoint_step_from_name(checkpoint_name=checkpoint.name)
    if direct_step is not None:
        return direct_step
    if checkpoint.name != "final_model" or not checkpoint.parent.exists():
        return None
    discovered_steps = [
        parsed_step
        for checkpoint_dir in checkpoint.parent.glob("checkpoint-*")
        for parsed_step in [
            checkpoint_step_from_name(checkpoint_name=checkpoint_dir.name)
        ]
        if parsed_step is not None
    ]
    if not discovered_steps:
        return None
    return max(discovered_steps)


def build_eval_log_payload(
    result_path: Path,
    checkpoint: Path,
) -> tuple[dict[str, float], int | None]:
    """Build eval metrics payload and optional W&B step value.

    Args:
        result_path: Aggregated `lm-eval` output JSON path.
        checkpoint: Evaluated checkpoint directory path.

    Returns:
        Tuple `(payload, eval_step)` for `wandb.log`.
    """
    payload = load_and_flatten_metrics(result_json_path=result_path)
    eval_step = infer_eval_step(checkpoint=checkpoint)
    if eval_step is None:
        return payload, None
    payload_with_step = dict(payload)
    payload_with_step["train/global_step"] = float(eval_step)
    payload_with_step["eval/checkpoint_step"] = float(eval_step)
    return payload_with_step, eval_step


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for standalone evaluation."""
    parser = argparse.ArgumentParser(description="Run standalone benchmark evaluation.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the model checkpoint directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the run/lm_eval YAML config.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output JSON path. Defaults to benchmark_evals/standalone.json in checkpoint dir.",
    )
    parser.add_argument(
        "--log-samples",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override lm_eval log_samples to persist per-example outputs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override lm_eval generation temperature (defaults to config value).",
    )
    parser.add_argument(
        "--top-p",
        dest="top_p",
        type=float,
        default=None,
        help="Override lm_eval nucleus sampling `top_p` (defaults to config value).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional sample cap forwarded to lm_eval `limit`.",
    )
    return parser.parse_args()


def apply_eval_overrides(config: LmEvalConfig, args: argparse.Namespace) -> LmEvalConfig:
    """Apply optional CLI overrides to eval config.

    Args:
        config: Parsed benchmark evaluation configuration.
        args: Parsed standalone CLI arguments.

    Returns:
        Effective evaluation configuration.
    """
    override_kwargs: dict[str, Any] = {}
    if args.log_samples is not None:
        override_kwargs["log_samples"] = bool(args.log_samples)
    if args.temperature is not None:
        assert args.temperature >= 0.0, "`--temperature` must be >= 0."
        override_kwargs["temperature"] = float(args.temperature)
    if args.top_p is not None:
        assert 0.0 < args.top_p <= 1.0, "`--top-p` must be in (0, 1]."
        override_kwargs["top_p"] = float(args.top_p)
    if not override_kwargs:
        return config
    return replace(config, **override_kwargs)


def load_eval_config(config_path: Path) -> tuple[LmEvalConfig, EvalRunMetadata | None]:
    """Load eval config and optional full run config.

    Args:
        config_path: Path to run config or lm_eval-only config.

    Returns:
        Tuple `(lm_eval_config, eval_run_metadata_or_none)`.
    """
    return parse_eval_config(config_path=config_path)


def resolve_output_json_path(checkpoint: Path, output: Path | None) -> Path:
    """Resolve default output JSON path for standalone eval.

    Args:
        checkpoint: Model checkpoint path or repo id.
        output: Optional explicit output path.

    Returns:
        Concrete output JSON path for `lm-eval`.
    """
    if output is not None:
        return output
    if not checkpoint.exists():
        model_slug = str(checkpoint).replace("/", "--")
        return Path("benchmark_evals_base") / f"{model_slug}.json"
    return checkpoint / "benchmark_evals" / "standalone.json"


def maybe_init_wandb_eval_run(
    run_config: EvalRunMetadata | None,
    checkpoint: Path,
    config_path: Path,
) -> None:
    """Initialize W&B run for eval-only process when run config is available.

    Args:
        run_config: Optional full run configuration.
        checkpoint: Evaluated checkpoint path.
        config_path: Source config path used for eval.

    Returns:
        None.
    """
    if run_config is None:
        return
    shared_run_id = os.environ.get("SFT_WANDB_RUN_ID")
    eval_step = infer_eval_step(checkpoint=checkpoint)
    run_context = build_wandb_run_context(
        base_run_name=run_config.run_name,
        job_type="eval",
    )
    init_config: dict[str, Any] = {
        "eval_config_path": str(config_path),
        "eval_checkpoint_path": str(checkpoint),
    }
    if eval_step is not None:
        init_config["eval_checkpoint_step"] = int(eval_step)
    init_kwargs: dict[str, Any] = {}
    if shared_run_id:
        init_kwargs["id"] = shared_run_id
        init_kwargs["resume"] = "allow"
    wandb.init(
        project=run_config.wandb_project,
        entity=run_config.wandb_entity,
        name=run_context.run_name,
        group=run_context.group_name,
        job_type="train" if shared_run_id else run_context.job_type,
        config=init_config,
        **init_kwargs,
    )
    wandb.define_metric(name="train/global_step")
    wandb.define_metric(name="eval/checkpoint_step")
    wandb.define_metric(name="bench/*", step_metric="train/global_step")


def maybe_log_eval_metrics_to_wandb(result_path: Path, checkpoint: Path) -> None:
    """Log flattened benchmark metrics to active W&B eval run.

    Args:
        result_path: Aggregated eval result JSON path.
        checkpoint: Evaluated checkpoint path.

    Returns:
        None.
    """
    if wandb.run is None:
        return
    shared_run_id = os.environ.get("SFT_WANDB_RUN_ID")
    payload, eval_step = build_eval_log_payload(
        result_path=result_path,
        checkpoint=checkpoint,
    )
    if shared_run_id:
        # Shared train/eval runs must rely on metric-defined step axes.
        wandb.log(data=payload)
    elif eval_step is None:
        wandb.log(data=payload)
    else:
        wandb.log(data=payload, step=eval_step)
    wandb.finish()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.limit is not None:
        assert args.limit >= 1, "`--limit` must be >= 1."
    lm_eval_config, run_config = load_eval_config(config_path=args.config)
    lm_eval_config = apply_eval_overrides(config=lm_eval_config, args=args)
    env_applied, env_reason = maybe_set_cross_numa_vllm_env(
        model_type=lm_eval_config.model_type
    )
    output_json_path = resolve_output_json_path(
        checkpoint=args.checkpoint,
        output=args.output,
    )
    maybe_init_wandb_eval_run(
        run_config=run_config,
        checkpoint=args.checkpoint,
        config_path=args.config,
    )

    print(f"Running evaluation for model/checkpoint: {args.checkpoint}")
    print(f"Using tasks: {lm_eval_config.tasks}")
    policy_status = "applied" if env_applied else "skipped"
    print(f"Cross-NUMA env policy {policy_status}: {env_reason}")

    result_path = run_lm_eval_for_checkpoint(
        checkpoint_path=args.checkpoint,
        output_json_path=output_json_path,
        config=lm_eval_config,
        limit=args.limit,
    )
    if lm_eval_config.log_samples:
        sample_log_paths = find_sample_log_files(result_json_path=result_path)
        if sample_log_paths:
            print("Sample logs saved to:")
            for sample_log_path in sample_log_paths:
                print(f"  - {sample_log_path}")
        else:
            print("Sample logging enabled, but no sample JSONL files were found.")

    maybe_log_eval_metrics_to_wandb(
        result_path=result_path,
        checkpoint=args.checkpoint,
    )
    print(f"Evaluation complete. Results saved to: {result_path}")


if __name__ == "__main__":
    main()
