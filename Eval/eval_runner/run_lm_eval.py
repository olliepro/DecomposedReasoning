"""Utilities for checkpoint-based `lm-eval` benchmark execution."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from eval_runner.aime_avgk import build_aime_avgk_process_results
from eval_runner.config_types import LmEvalConfig

ModelArgValue = str | int | float | bool
ModelArgs = dict[str, ModelArgValue]
GenKwargs = dict[str, ModelArgValue]
AIME_TASK_NAMES: tuple[str, ...] = ("aime24", "aime25")
AVG_AT_K_PATTERN = re.compile(pattern=r"^avg_at_(\d+)$")


@dataclass(frozen=True)
class BenchmarkMetricSpec:
    """One canonical benchmark metric extraction rule.

    Args:
        task_name: Task key under `payload["results"]`.
        metric_key: Canonical flattened key for logging.
        metric_names: Accepted metric names from lm-eval task results.

    Returns:
        Dataclass container describing one metric extraction rule.
    """

    task_name: str
    metric_key: str
    metric_names: tuple[str, ...]


BENCHMARK_METRIC_SPECS: tuple[BenchmarkMetricSpec, ...] = (
    BenchmarkMetricSpec(
        task_name="minerva_math500",
        metric_key="bench/math500/math_verify",
        metric_names=("math_verify",),
    ),
    BenchmarkMetricSpec(
        task_name="gpqa_main_n_shot",
        metric_key="bench/gpqa_main_n_shot/acc",
        metric_names=("acc", "accuracy"),
    ),
)


def _build_aime_metric_name(aime_avg_k: int) -> str:
    """Build canonical AIME avg@k metric name.

    Args:
        aime_avg_k: Number of sampled responses per question.

    Returns:
        Metric name in `avg_at_<k>` format.
    """
    return f"avg_at_{aime_avg_k}"


def _build_aime_filter_name(aime_avg_k: int) -> str:
    """Build filter label for AIME avg@k sampling.

    Args:
        aime_avg_k: Number of sampled responses per question.

    Returns:
        Filter label in `avg@<k>` format.
    """
    return f"avg@{aime_avg_k}"


def _build_aime_task_override(task_name: str, aime_avg_k: int) -> dict[str, Any]:
    """Build one `lm-eval` task override dict for custom AIME avg@k scoring.

    Args:
        task_name: Base task name (`aime24` or `aime25`).
        aime_avg_k: Number of sampled responses per question.

    Returns:
        Task override payload consumed by `lm_eval.get_task_dict`.
    """
    metric_name = _build_aime_metric_name(aime_avg_k=aime_avg_k)
    filter_name = _build_aime_filter_name(aime_avg_k=aime_avg_k)
    return {
        "task": task_name,
        "repeats": aime_avg_k,
        "process_results": build_aime_avgk_process_results(metric_name=metric_name),
        "metric_list": [
            {"metric": metric_name, "aggregation": "mean", "higher_is_better": True}
        ],
        "filter_list": [
            {"name": filter_name, "filter": [{"function": "take_first_k", "k": aime_avg_k}]}
        ],
    }


def _uses_aime_avgk(tasks: tuple[str, ...]) -> bool:
    """Check whether configured tasks include AIME avg@k targets.

    Args:
        tasks: Configured task name tuple.

    Returns:
        `True` when at least one AIME task is present.
    """
    return any(task_name in AIME_TASK_NAMES for task_name in tasks)


def _validate_aime_sampling(config: LmEvalConfig) -> None:
    """Validate sampling settings for AIME avg@k.

    Args:
        config: Benchmark evaluation configuration.

    Returns:
        None.
    """
    if not _uses_aime_avgk(tasks=config.tasks):
        return
    assert config.aime_avg_k >= 1, "`aime_avg_k` must be >= 1."
    assert config.temperature > 0.0, (
        "AIME avg@k requires stochastic generation; set temperature > 0.0."
    )


def _build_task_entries(config: LmEvalConfig) -> list[str | dict[str, Any]]:
    """Build task entries for `simple_evaluate`, including AIME overrides.

    Args:
        config: Benchmark evaluation configuration.

    Returns:
        Task entries compatible with `simple_evaluate(tasks=...)`.
    """
    task_entries: list[str | dict[str, Any]] = []
    for task_name in config.tasks:
        if task_name in AIME_TASK_NAMES:
            task_entries.append(
                _build_aime_task_override(task_name=task_name, aime_avg_k=config.aime_avg_k)
            )
            continue
        task_entries.append(task_name)
    return task_entries


def build_model_args(checkpoint_path: Path, config: LmEvalConfig) -> ModelArgs:
    """Build the `model_args` value for `lm_eval.simple_evaluate`.

    Args:
        checkpoint_path: Saved model checkpoint directory.
        config: Benchmark evaluation configuration.

    Returns:
        Model argument mapping accepted by `simple_evaluate`.

    Example:
        >>> args = build_model_args(
        ...     checkpoint_path=Path("checkpoint-100"),
        ...     config=LmEvalConfig(model_type="vllm"),
        ... )
        >>> args["pretrained"]
        'checkpoint-100'
    """
    model_args: ModelArgs = {"pretrained": str(checkpoint_path)}
    if config.model_type == "vllm":
        model_args.update(
            {
                "dtype": "auto",
                "tensor_parallel_size": config.tensor_parallel_size,
                "gpu_memory_utilization": config.gpu_memory_utilization,
                "data_parallel_size": config.data_parallel_size,
                "disable_log_stats": config.vllm_disable_log_stats,
            }
        )
    else:
        model_args["dtype"] = "bfloat16"
    model_args.update(
        {
            "trust_remote_code": config.trust_remote_code,
            "think_end_token": config.think_end_token,
        }
    )
    return model_args


def build_gen_kwargs(config: LmEvalConfig) -> GenKwargs:
    """Build generation kwargs for sampling-based generation tasks.

    Args:
        config: Benchmark evaluation configuration.

    Returns:
        Generation kwargs accepted by `simple_evaluate`.
    """
    return {
        "do_sample": config.temperature > 0.0,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "max_gen_toks": config.max_gen_toks,
    }


def build_simple_evaluate_kwargs(
    checkpoint_path: Path,
    config: LmEvalConfig,
    limit: int | None = None,
) -> dict[str, Any]:
    """Build keyword args for `lm_eval.simple_evaluate`.

    Args:
        checkpoint_path: Saved model checkpoint directory.
        config: Benchmark evaluation configuration.
        limit: Optional sample cap passed to `simple_evaluate(limit=...)`.

    Returns:
        Keyword args dictionary for `lm_eval.simple_evaluate`.

    Example:
        >>> kwargs = build_simple_evaluate_kwargs(
        ...     checkpoint_path=Path("checkpoint-100"),
        ...     config=LmEvalConfig(),
        ... )
        >>> len(kwargs["tasks"]) >= 1
        True
    """
    _validate_aime_sampling(config=config)
    eval_kwargs: dict[str, Any] = {
        "model": config.model_type,
        "model_args": build_model_args(checkpoint_path=checkpoint_path, config=config),
        "tasks": _build_task_entries(config=config),
        "batch_size": config.batch_size,
        "num_fewshot": config.num_fewshot,
        "apply_chat_template": config.apply_chat_template,
        "log_samples": config.log_samples,
    }
    if config.model_type == "vllm":
        eval_kwargs["gen_kwargs"] = build_gen_kwargs(config=config)
    if limit is not None:
        assert limit >= 1, "`limit` must be >= 1."
        eval_kwargs["limit"] = limit
    return eval_kwargs


def _create_evaluation_tracker(output_json_path: Path) -> Any:
    """Create the lm-eval tracker used to persist aggregated and sample outputs.

    Args:
        output_json_path: Requested output path from standalone eval.

    Returns:
        Tracker object implementing save methods for results and samples.
    """
    from lm_eval.loggers import EvaluationTracker

    return EvaluationTracker(output_path=str(output_json_path))


def _run_simple_evaluate(eval_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Execute one lm-eval run through the Python API.

    Args:
        eval_kwargs: Keyword args for `lm_eval.simple_evaluate`.

    Returns:
        Raw lm-eval result payload.
    """
    import lm_eval

    results = lm_eval.simple_evaluate(**eval_kwargs)
    assert results is not None, "Expected non-empty lm-eval results payload."
    assert isinstance(results, dict), "Expected dict payload from simple_evaluate."
    return cast(dict[str, Any], results)


def _extract_samples(
    results_payload: dict[str, Any],
    log_samples: bool,
) -> dict[str, list[dict[str, Any]]] | None:
    """Pop optional samples payload from lm-eval output.

    Args:
        results_payload: Raw `simple_evaluate` return payload.
        log_samples: Whether sample logging is enabled for this run.

    Returns:
        Per-task samples mapping when enabled, otherwise `None`.
    """
    if not log_samples:
        return None
    sample_payload = results_payload.pop("samples")
    assert isinstance(sample_payload, dict), "Expected samples payload mapping."
    return cast(dict[str, list[dict[str, Any]]], sample_payload)


def _persist_lm_eval_outputs(
    results_payload: dict[str, Any],
    evaluation_tracker: Any,
    log_samples: bool,
) -> None:
    """Persist lm-eval outputs to timestamped JSON/JSONL artifacts.

    Args:
        results_payload: Raw `simple_evaluate` return payload.
        evaluation_tracker: Tracker object for result persistence.
        log_samples: Whether sample logs should be saved.

    Returns:
        None.
    """
    samples_by_task = _extract_samples(
        results_payload=results_payload,
        log_samples=log_samples,
    )
    evaluation_tracker.save_results_aggregated(
        results=results_payload,
        samples=samples_by_task,
    )
    if samples_by_task is None:
        return
    task_configs = results_payload.get("configs", {})
    assert isinstance(task_configs, dict), "Expected task configs in results payload."
    for task_name in task_configs:
        task_samples = samples_by_task.get(task_name, [])
        assert isinstance(task_samples, list), f"Expected sample list for: {task_name}"
        evaluation_tracker.save_results_samples(
            task_name=task_name,
            samples=task_samples,
        )


def find_result_file(output_json_path: Path) -> Path:
    """Locate the timestamped aggregated result JSON written by `lm-eval`.

    Args:
        output_json_path: The same path passed to `--output_path`.

    Returns:
        Concrete result JSON file path.
    """
    assert output_json_path.suffix == ".json", "Expected JSON output path suffix."
    pattern = f"{output_json_path.stem}_*.json"
    result_files = sorted(output_json_path.parent.glob(pattern))
    assert result_files, f"No lm-eval result JSON found for pattern: {pattern}"
    return result_files[-1]


def find_sample_log_files(result_json_path: Path) -> list[Path]:
    """Locate per-sample JSONL files emitted for a specific eval run.

    Args:
        result_json_path: Timestamped aggregated result JSON path.

    Returns:
        Sorted JSONL paths from `--log_samples` for the same run timestamp.
    """
    timestamp = result_json_path.stem.rsplit("_", maxsplit=1)[-1]
    pattern = f"samples_*_{timestamp}.jsonl"
    return sorted(result_json_path.parent.glob(pattern))


def run_lm_eval_for_checkpoint(
    checkpoint_path: Path,
    output_json_path: Path,
    config: LmEvalConfig,
    limit: int | None = None,
) -> Path:
    """Run `lm_eval.simple_evaluate` for one checkpoint.

    Args:
        checkpoint_path: Saved model checkpoint directory.
        output_json_path: Requested result output JSON path.
        config: Benchmark evaluation configuration.
        limit: Optional sample cap passed to `simple_evaluate(limit=...)`.

    Returns:
        Timestamped path of the produced `lm-eval` aggregated result JSON.
    """
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    eval_kwargs = build_simple_evaluate_kwargs(
        checkpoint_path=checkpoint_path,
        config=config,
        limit=limit,
    )
    evaluation_tracker = _create_evaluation_tracker(output_json_path=output_json_path)
    results_payload = _run_simple_evaluate(eval_kwargs=eval_kwargs)
    _persist_lm_eval_outputs(
        results_payload=results_payload,
        evaluation_tracker=evaluation_tracker,
        log_samples=config.log_samples,
    )
    return find_result_file(output_json_path=output_json_path)


def _extract_metric(
    task_payload: dict[str, Any], metric_names: tuple[str, ...]
) -> float:
    """Extract a metric from one task payload by metric-name prefix.

    Args:
        task_payload: Task metrics mapping from `results[task_name]`.
        metric_names: Candidate metric keys (prefix matches allowed).

    Returns:
        The extracted metric value.
    """
    for metric_name in metric_names:
        for key, value in task_payload.items():
            if key == metric_name or key.startswith(f"{metric_name},"):
                return float(value)
    raise KeyError(f"None of metric candidates found: {metric_names}")


def _extract_avg_at_k_metric_name(task_payload: dict[str, Any]) -> str:
    """Extract `avg_at_<k>` metric name from one AIME task payload.

    Args:
        task_payload: Metrics mapping from `results[aime_task]`.

    Returns:
        Unique `avg_at_<k>` metric name.
    """
    metric_names = {
        key.split(",")[0]
        for key in task_payload
        if AVG_AT_K_PATTERN.fullmatch(key.split(",")[0])
    }
    assert metric_names, "AIME payload is missing an `avg_at_<k>` metric."
    assert len(metric_names) == 1, "Expected exactly one `avg_at_<k>` metric in AIME payload."
    return next(iter(metric_names))


def flatten_benchmark_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Map raw `lm-eval` payload to canonical benchmark metric keys.

    Args:
        payload: Parsed `lm-eval` aggregated result JSON payload.

    Returns:
        Flat benchmark metric mapping with stable key names.
    """
    results = payload.get("results", {})
    assert isinstance(results, dict), "lm-eval payload missing `results` mapping."
    flattened_metrics: dict[str, float] = {}
    for metric_spec in BENCHMARK_METRIC_SPECS:
        task_payload = results.get(metric_spec.task_name)
        if task_payload is None:
            continue
        assert isinstance(
            task_payload, dict
        ), f"Task payload must be a dict: {metric_spec.task_name}"
        flattened_metrics[metric_spec.metric_key] = _extract_metric(
            task_payload=task_payload,
            metric_names=metric_spec.metric_names,
        )
    for task_name in AIME_TASK_NAMES:
        task_payload = results.get(task_name)
        if task_payload is None:
            continue
        assert isinstance(task_payload, dict), f"Task payload must be a dict: {task_name}"
        metric_name = _extract_avg_at_k_metric_name(task_payload=task_payload)
        flattened_metrics[f"bench/{task_name}/{metric_name}"] = _extract_metric(
            task_payload=task_payload,
            metric_names=(metric_name,),
        )
    assert flattened_metrics, "No recognized benchmark task metrics were present."
    return flattened_metrics


def load_and_flatten_metrics(result_json_path: Path) -> dict[str, float]:
    """Load and flatten benchmark metrics from one `lm-eval` result file.

    Args:
        result_json_path: Aggregated result JSON path emitted by `lm-eval`.

    Returns:
        Flat benchmark metric mapping.
    """
    payload = json.loads(result_json_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict), "Expected dict payload from lm-eval JSON."
    return flatten_benchmark_metrics(payload=payload)
