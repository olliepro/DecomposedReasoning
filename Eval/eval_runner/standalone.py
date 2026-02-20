"""Standalone benchmark evaluation entrypoint."""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import threading
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

VLLM_STATS_PATTERN = re.compile(
    pattern=(
        r"Avg prompt throughput: (?P<prompt_toks_per_s>\d+(?:\.\d+)?) tokens/s, "
        r"Avg generation throughput: (?P<generation_toks_per_s>\d+(?:\.\d+)?) tokens/s, "
        r"Running: (?P<running_reqs>\d+) reqs, "
        r"Waiting: (?P<waiting_reqs>\d+) reqs"
        r"(?:, Preemptions: (?P<preemptions>\d+))?, "
        r"GPU KV cache usage: (?P<gpu_kv_cache_usage_pct>\d+(?:\.\d+)?)%, "
        r"Prefix cache hit rate: (?P<prefix_cache_hit_rate_pct>\d+(?:\.\d+)?)%"
    )
)
VLLM_STATS_LOGGER_NAMES: tuple[str, ...] = (
    "vllm.v1.metrics.loggers",
    "vllm.engine.metrics",
    "vllm",
)


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
    numeric_prefix = ""
    for character in step_suffix:
        if not character.isdigit():
            break
        numeric_prefix += character
    if not numeric_prefix:
        return None
    return int(numeric_prefix)


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
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Optional task override list (e.g. --tasks aime24 aime25).",
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
    if args.tasks is not None:
        normalized_tasks = tuple(task_name.strip() for task_name in args.tasks if task_name.strip())
        assert normalized_tasks, "`--tasks` must include at least one non-empty task name."
        override_kwargs["tasks"] = normalized_tasks
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
    task_names: tuple[str, ...] | None = None,
) -> None:
    """Initialize W&B run for eval-only process when run config is available.

    Args:
        run_config: Optional full run configuration.
        checkpoint: Evaluated checkpoint path.
        config_path: Source config path used for eval.
        task_names: Optional explicit task list used for this eval invocation.

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
    run_name = run_context.run_name
    if not shared_run_id:
        checkpoint_token = (
            f"checkpoint-{eval_step}" if eval_step is not None else checkpoint.name
        ).replace("/", "--")
        run_name = f"{run_context.run_name}_{checkpoint_token}"
        if task_names:
            if len(task_names) == 1:
                task_token = f"task-{task_names[0]}"
            else:
                task_token = f"tasks-{'-'.join(task_names)}"
            run_name = f"{run_name}_{task_token.replace('/', '--')}"
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
        name=run_name,
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


def configure_vllm_runtime_logging(config: LmEvalConfig) -> tuple[bool, str]:
    """Configure vLLM runtime logging env for periodic progress visibility.

    Args:
        config: Effective standalone eval configuration.

    Returns:
        Tuple `(applied, message)` describing runtime logging policy.
    """
    if config.model_type != "vllm":
        return False, "model_type is not vllm"
    assert config.vllm_log_stats_interval > 0.0, "`vllm_log_stats_interval` must be > 0."
    os.environ["VLLM_LOGGING_LEVEL"] = config.vllm_logging_level.upper()
    os.environ["VLLM_LOG_STATS_INTERVAL"] = str(config.vllm_log_stats_interval)
    status_message = (
        "enabled with "
        f"VLLM_LOGGING_LEVEL={os.environ['VLLM_LOGGING_LEVEL']}, "
        f"VLLM_LOG_STATS_INTERVAL={os.environ['VLLM_LOG_STATS_INTERVAL']}, "
        f"disable_log_stats={config.vllm_disable_log_stats}"
    )
    return True, status_message


def parse_vllm_stats_log_line(log_line: str) -> dict[str, float] | None:
    """Parse one vLLM periodic stats log line into W&B metric payload.

    Args:
        log_line: One vLLM logger output line.

    Returns:
        Flattened metric payload, or `None` when the line is unrelated.
    """
    stats_match = VLLM_STATS_PATTERN.search(log_line)
    if stats_match is None:
        return None
    payload: dict[str, float] = {
        "runtime/vllm/prompt_tokens_per_s": float(stats_match.group("prompt_toks_per_s")),
        "runtime/vllm/generation_tokens_per_s": float(stats_match.group("generation_toks_per_s")),
        "runtime/vllm/running_reqs": float(stats_match.group("running_reqs")),
        "runtime/vllm/waiting_reqs": float(stats_match.group("waiting_reqs")),
        "runtime/vllm/gpu_kv_cache_usage_pct": float(stats_match.group("gpu_kv_cache_usage_pct")),
        "runtime/vllm/prefix_cache_hit_rate_pct": float(stats_match.group("prefix_cache_hit_rate_pct")),
    }
    preemptions_text = stats_match.group("preemptions")
    if preemptions_text is not None:
        payload["runtime/vllm/preemptions"] = float(preemptions_text)
    return payload


class VllmStatsWandbHandler(logging.Handler):
    """Forward vLLM periodic stats logs to W&B as runtime metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.captured_stats_line_count = 0
        self._completed_requests_tracker = ApproxCompletedRequestsTracker()

    def emit(self, record: logging.LogRecord) -> None:
        if wandb.run is None:
            return
        parsed_payload = parse_vllm_stats_log_line(log_line=record.getMessage())
        if parsed_payload is None:
            return
        parsed_payload = self._completed_requests_tracker.with_completed_requests(
            payload=parsed_payload
        )
        self.captured_stats_line_count += 1
        wandb.log(data=parsed_payload)


class ApproxCompletedRequestsTracker:
    """Estimate completed requests from running/waiting queue snapshots."""

    def __init__(self) -> None:
        self._max_outstanding_requests = 0.0

    def with_completed_requests(self, *, payload: dict[str, float]) -> dict[str, float]:
        """Add rough completed request estimate to one parsed vLLM payload.

        Args:
            payload: Parsed vLLM stats payload containing running/waiting counts.

        Returns:
            Payload extended with `runtime/vllm/completed_requests`.
        """
        running_requests = payload.get("runtime/vllm/running_reqs", 0.0)
        waiting_requests = payload.get("runtime/vllm/waiting_reqs", 0.0)
        outstanding_requests = running_requests + waiting_requests
        self._max_outstanding_requests = max(
            self._max_outstanding_requests, outstanding_requests
        )
        completed_requests = max(
            self._max_outstanding_requests - outstanding_requests, 0.0
        )
        payload_with_completed = dict(payload)
        payload_with_completed["runtime/vllm/completed_requests"] = completed_requests
        return payload_with_completed


class VllmStatsStreamCapture:
    """Parse native vLLM stats lines from stdout/stderr at fd level."""

    def __init__(self) -> None:
        self.captured_stats_line_count = 0
        self._saved_fds: dict[int, int] = {}
        self._read_fds: dict[int, int] = {}
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self._completed_requests_tracker = ApproxCompletedRequestsTracker()

    def _handle_line(self, *, line: str) -> None:
        if wandb.run is None:
            return
        parsed_payload = parse_vllm_stats_log_line(log_line=line)
        if parsed_payload is None:
            return
        parsed_payload = self._completed_requests_tracker.with_completed_requests(
            payload=parsed_payload
        )
        with self._lock:
            self.captured_stats_line_count += 1
        wandb.log(data=parsed_payload)

    def _reader_loop(self, *, read_fd: int, mirror_fd: int) -> None:
        line_buffer = ""
        while True:
            try:
                chunk = os.read(read_fd, 4096)
            except OSError:
                break
            if not chunk:
                break
            try:
                os.write(mirror_fd, chunk)
            except OSError:
                pass
            line_buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in line_buffer:
                line_text, line_buffer = line_buffer.split("\n", 1)
                self._handle_line(line=line_text)
        if line_buffer:
            self._handle_line(line=line_buffer)

    def start(self) -> None:
        """Start stdout/stderr fd capture."""
        sys.stdout.flush()
        sys.stderr.flush()
        for stream_fd in (1, 2):
            saved_fd = os.dup(stream_fd)
            read_fd, write_fd = os.pipe()
            os.dup2(write_fd, stream_fd)
            os.close(write_fd)
            self._saved_fds[stream_fd] = saved_fd
            self._read_fds[stream_fd] = read_fd
            reader_thread = threading.Thread(
                target=self._reader_loop,
                kwargs={"read_fd": read_fd, "mirror_fd": saved_fd},
                daemon=True,
            )
            reader_thread.start()
            self._threads.append(reader_thread)

    def stop(self) -> None:
        """Restore stdout/stderr and stop capture threads."""
        sys.stdout.flush()
        sys.stderr.flush()
        for stream_fd, saved_fd in self._saved_fds.items():
            os.dup2(saved_fd, stream_fd)
        for reader_thread in self._threads:
            reader_thread.join(timeout=2.0)
        for read_fd in self._read_fds.values():
            try:
                os.close(read_fd)
            except OSError:
                pass
        for saved_fd in self._saved_fds.values():
            try:
                os.close(saved_fd)
            except OSError:
                pass


def maybe_attach_vllm_stats_wandb_handler(
    config: LmEvalConfig,
) -> tuple[logging.Handler | None, str]:
    """Attach a logging handler that forwards vLLM stats lines to W&B.

    Args:
        config: Effective standalone eval configuration.

    Returns:
        Tuple `(handler_or_none, status_message)`.
    """
    if config.model_type != "vllm":
        return None, "model_type is not vllm"
    if wandb.run is None:
        return None, "wandb run is not active"
    if config.vllm_disable_log_stats:
        return None, "vllm_disable_log_stats=True"
    stats_handler = VllmStatsWandbHandler()
    stats_handler.setLevel(level=logging.INFO)
    for logger_name in VLLM_STATS_LOGGER_NAMES:
        logging.getLogger(logger_name).addHandler(hdlr=stats_handler)
    return stats_handler, f"attached to {', '.join(VLLM_STATS_LOGGER_NAMES)}"


def detach_vllm_stats_wandb_handler(handler: logging.Handler | None) -> None:
    """Detach previously attached vLLM stats forwarding handler.

    Args:
        handler: Previously attached handler instance.

    Returns:
        None.
    """
    if handler is None:
        return
    for logger_name in VLLM_STATS_LOGGER_NAMES:
        logging.getLogger(logger_name).removeHandler(hdlr=handler)


def maybe_start_vllm_stats_stream_capture(
    config: LmEvalConfig,
) -> tuple[VllmStatsStreamCapture | None, str]:
    """Start fd-level vLLM stats capture for native log-line parsing.

    Args:
        config: Effective standalone eval configuration.

    Returns:
        Tuple `(capture_or_none, status_message)`.
    """
    if config.model_type != "vllm":
        return None, "model_type is not vllm"
    if wandb.run is None:
        return None, "wandb run is not active"
    if config.vllm_disable_log_stats:
        return None, "vllm_disable_log_stats=True"
    stream_capture = VllmStatsStreamCapture()
    stream_capture.start()
    return stream_capture, "started fd-level stdout/stderr capture"


def stop_vllm_stats_stream_capture(capture: VllmStatsStreamCapture | None) -> None:
    """Stop fd-level capture when it was started.

    Args:
        capture: Previously started stream capture object.

    Returns:
        None.
    """
    if capture is None:
        return
    capture.stop()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    if args.limit is not None:
        assert args.limit >= 1, "`--limit` must be >= 1."
    lm_eval_config, run_config = load_eval_config(config_path=args.config)
    lm_eval_config = apply_eval_overrides(config=lm_eval_config, args=args)
    logging_applied, logging_message = configure_vllm_runtime_logging(config=lm_eval_config)
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
        task_names=lm_eval_config.tasks,
    )
    stats_capture, stats_capture_reason = maybe_start_vllm_stats_stream_capture(
        config=lm_eval_config
    )

    print(f"Running evaluation for model/checkpoint: {args.checkpoint}")
    print(f"Using tasks: {lm_eval_config.tasks}")
    logging_status = "applied" if logging_applied else "skipped"
    print(f"vLLM runtime logging {logging_status}: {logging_message}")
    capture_status = "applied" if stats_capture is not None else "skipped"
    print(f"vLLM stats to W&B {capture_status}: {stats_capture_reason}")
    policy_status = "applied" if env_applied else "skipped"
    print(f"Cross-NUMA env policy {policy_status}: {env_reason}")

    try:
        result_path = run_lm_eval_for_checkpoint(
            checkpoint_path=args.checkpoint,
            output_json_path=output_json_path,
            config=lm_eval_config,
            limit=args.limit,
        )
    finally:
        stop_vllm_stats_stream_capture(capture=stats_capture)
        if stats_capture is not None:
            print(
                "vLLM stats lines captured in this process: "
                f"{stats_capture.captured_stats_line_count}"
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
