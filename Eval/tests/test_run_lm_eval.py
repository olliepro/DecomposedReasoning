"""Unit tests for `eval_runner.run_lm_eval`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import eval_runner.run_lm_eval as run_lm_eval_module
from eval_runner.config_types import LmEvalConfig
from eval_runner.run_lm_eval import (
    AIME_TASK_NAMES,
    build_gen_kwargs,
    build_model_args,
    build_simple_evaluate_kwargs,
    flatten_benchmark_metrics,
    find_sample_log_files,
    find_result_file,
    run_lm_eval_for_checkpoint,
)


def test_build_simple_evaluate_kwargs_includes_required_args(tmp_path: Path) -> None:
    """Kwarg builder should emit canonical fields for Python API execution."""
    checkpoint_path = tmp_path / "checkpoint-100"
    eval_kwargs = build_simple_evaluate_kwargs(
        checkpoint_path=checkpoint_path,
        config=LmEvalConfig(),
    )
    task_entries = eval_kwargs["tasks"]
    assert isinstance(task_entries, list)
    aime_entries = [entry for entry in task_entries if isinstance(entry, dict)]
    minerva_entries = [entry for entry in task_entries if entry == "minerva_math500"]
    assert eval_kwargs["model"] == "hf"
    assert minerva_entries == ["minerva_math500"]
    assert len(aime_entries) == 2
    assert sorted([entry["task"] for entry in aime_entries]) == sorted(AIME_TASK_NAMES)
    assert eval_kwargs["apply_chat_template"] is True
    assert eval_kwargs["log_samples"] is False
    assert "gen_kwargs" not in eval_kwargs


@pytest.mark.parametrize(argnames="aime_avg_k", argvalues=[1, 7, 32, 64])
def test_build_simple_evaluate_kwargs_supports_any_aime_avg_k(
    tmp_path: Path, aime_avg_k: int
) -> None:
    """Kwarg builder should parameterize AIME overrides for any configured k."""
    eval_kwargs = build_simple_evaluate_kwargs(
        checkpoint_path=tmp_path / "checkpoint-99",
        config=LmEvalConfig(tasks=("aime24",), aime_avg_k=aime_avg_k),
    )
    task_entries = eval_kwargs["tasks"]
    assert isinstance(task_entries, list)
    assert len(task_entries) == 1
    aime_task = task_entries[0]
    assert isinstance(aime_task, dict)
    assert aime_task["task"] == "aime24"
    assert aime_task["repeats"] == aime_avg_k
    assert aime_task["metric_list"][0]["metric"] == f"avg_at_{aime_avg_k}"
    assert aime_task["filter_list"][0]["name"] == f"avg@{aime_avg_k}"
    assert aime_task["filter_list"][0]["filter"][0] == {
        "function": "take_first_k",
        "k": aime_avg_k,
    }


def test_build_simple_evaluate_kwargs_rejects_greedy_aime_avg_k(tmp_path: Path) -> None:
    """AIME avg@k mode should reject greedy decoding (temperature=0)."""
    with pytest.raises(AssertionError):
        build_simple_evaluate_kwargs(
            checkpoint_path=tmp_path / "checkpoint-99",
            config=LmEvalConfig(tasks=("aime24",), temperature=0.0),
        )


def test_build_model_args_for_vllm_includes_parallel_and_sampling_fields(
    tmp_path: Path,
) -> None:
    """Model arg builder should include expected vLLM loading options."""
    model_args = build_model_args(
        checkpoint_path=tmp_path / "checkpoint-11",
        config=LmEvalConfig(
            model_type="vllm",
            tensor_parallel_size=2,
            gpu_memory_utilization=0.8,
            data_parallel_size=1,
        ),
    )
    assert model_args["dtype"] == "auto"
    assert model_args["tensor_parallel_size"] == 2
    assert model_args["gpu_memory_utilization"] == 0.8
    assert model_args["data_parallel_size"] == 1
    assert model_args["disable_log_stats"] is False
    assert model_args["trust_remote_code"] is True
    assert model_args["think_end_token"] == "</think>"


def test_build_gen_kwargs_uses_temperature_and_top_p() -> None:
    """Generation kwargs helper should emit sampling and max token values."""
    config = LmEvalConfig(temperature=0.6, top_p=0.95, max_gen_toks=32768)
    assert build_gen_kwargs(config=config) == {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_gen_toks": 32768,
    }


def test_build_gen_kwargs_sets_do_sample_false_for_greedy_temperature() -> None:
    """Generation kwargs helper should emit do_sample=false for temperature=0."""
    config = LmEvalConfig(temperature=0.0, top_p=0.95, max_gen_toks=32768)
    assert build_gen_kwargs(config=config) == {
        "do_sample": False,
        "temperature": 0.0,
        "top_p": 0.95,
        "max_gen_toks": 32768,
    }


def test_build_simple_evaluate_kwargs_adds_gen_kwargs_for_vllm(tmp_path: Path) -> None:
    """vLLM kwarg builder should include generation kwargs for task generation."""
    eval_kwargs = build_simple_evaluate_kwargs(
        checkpoint_path=tmp_path / "checkpoint-11",
        config=LmEvalConfig(
            model_type="vllm",
            temperature=0.6,
            top_p=0.95,
            max_gen_toks=32768,
        ),
    )
    assert eval_kwargs["gen_kwargs"] == {
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_gen_toks": 32768,
    }


def test_build_simple_evaluate_kwargs_includes_limit_when_set(tmp_path: Path) -> None:
    """Kwarg builder should forward optional eval sample limits.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    eval_kwargs = build_simple_evaluate_kwargs(
        checkpoint_path=tmp_path / "checkpoint-11",
        config=LmEvalConfig(tasks=("minerva_math500",)),
        limit=5,
    )
    assert eval_kwargs["limit"] == 5


def test_build_simple_evaluate_kwargs_rejects_invalid_limit(tmp_path: Path) -> None:
    """Kwarg builder should reject non-positive sample limits.

    Args:
        tmp_path: Temporary directory fixture.

    Returns:
        None.
    """
    with pytest.raises(AssertionError):
        build_simple_evaluate_kwargs(
            checkpoint_path=tmp_path / "checkpoint-11",
            config=LmEvalConfig(tasks=("minerva_math500",)),
            limit=0,
        )


def test_flatten_benchmark_metrics_extracts_expected_keys() -> None:
    """Metric flattening should normalize Minerva + secondary task metrics."""
    payload = {
        "results": {
            "minerva_math500": {"exact_match,none": 0.01, "math_verify,none": 0.31},
            "aime24": {"avg_at_32,avg@32": 0.12},
            "aime25": {"avg_at_32,avg@32": 0.09},
            "gpqa_main_n_shot": {"acc,none": 0.42},
        }
    }
    metrics = flatten_benchmark_metrics(payload=payload)
    assert metrics["bench/math500/math_verify"] == 0.31
    assert metrics["bench/aime24/avg_at_32"] == 0.12
    assert metrics["bench/aime25/avg_at_32"] == 0.09
    assert metrics["bench/gpqa_main_n_shot/acc"] == 0.42


def test_flatten_benchmark_metrics_handles_subset_tasks() -> None:
    """Metric flattening should support configs that evaluate a task subset."""
    payload = {
        "results": {
            "minerva_math500": {"exact_match,none": 0.01, "math_verify,none": 0.07}
        }
    }
    metrics = flatten_benchmark_metrics(payload=payload)
    assert metrics == {"bench/math500/math_verify": 0.07}


def test_flatten_benchmark_metrics_extracts_dynamic_aime_avg_k_keys() -> None:
    """Metric flattening should preserve configured AIME k in key names."""
    payload = {
        "results": {
            "aime24": {"avg_at_7,avg@7": 0.25, "avg_at_7_stderr,avg@7": 0.03},
            "aime25": {"avg_at_64,avg@64": 0.5},
        }
    }
    metrics = flatten_benchmark_metrics(payload=payload)
    assert metrics["bench/aime24/avg_at_7"] == 0.25
    assert metrics["bench/aime25/avg_at_64"] == 0.5


def test_flatten_benchmark_metrics_rejects_aime_without_avg_at_k() -> None:
    """Flattening should fail when AIME payload lacks custom avg@k metric."""
    payload = {"results": {"aime24": {"exact_match,none": 0.1}}}
    with pytest.raises(AssertionError):
        flatten_benchmark_metrics(payload=payload)


def test_find_result_file_discovers_timestamped_output(tmp_path: Path) -> None:
    """Result discovery should return the newest timestamped JSON file."""
    output_json_path = tmp_path / "epoch_2_step_30.json"
    old_file = tmp_path / "epoch_2_step_30_2026-01-01T00-00-00.json"
    new_file = tmp_path / "epoch_2_step_30_2026-01-01T00-00-01.json"
    old_file.write_text(json.dumps({"ok": 1}), encoding="utf-8")
    new_file.write_text(json.dumps({"ok": 2}), encoding="utf-8")
    result_path = find_result_file(output_json_path=output_json_path)
    assert result_path == new_file


def test_find_sample_log_files_uses_result_timestamp(tmp_path: Path) -> None:
    """Sample discovery should match only jsonl files from one eval timestamp."""
    result_path = tmp_path / "standalone_2026-02-12T18-04-24.799797.json"
    keep_file = tmp_path / "samples_minerva_math500_2026-02-12T18-04-24.799797.jsonl"
    skip_file = tmp_path / "samples_minerva_math500_2026-02-12T18-04-25.799797.jsonl"
    keep_file.write_text("{}", encoding="utf-8")
    skip_file.write_text("{}", encoding="utf-8")
    sample_paths = find_sample_log_files(result_json_path=result_path)
    assert sample_paths == [keep_file]


def test_run_lm_eval_for_checkpoint_uses_python_api_and_saves_outputs(
    tmp_path: Path, monkeypatch
) -> None:
    """Checkpoint runner should call Python API and persist result artifacts."""
    captured_eval_kwargs: dict[str, Any] = {}
    captured_aggregated: dict[str, Any] = {}
    captured_sample_logs: list[tuple[str, list[dict[str, Any]]]] = []

    class FakeTracker:
        """Simple evaluation tracker double for save call assertions."""

        def save_results_aggregated(
            self,
            *,
            results: dict[str, Any],
            samples: dict[str, list[dict[str, Any]]] | None = None,
        ) -> None:
            captured_aggregated["results"] = results
            captured_aggregated["samples"] = samples

        def save_results_samples(
            self, *, task_name: str, samples: list[dict[str, Any]]
        ) -> None:
            captured_sample_logs.append((task_name, samples))

    fake_result_payload = {
        "results": {"minerva_math500": {"math_verify,none": 0.25}},
        "configs": {"minerva_math500": {"task": "minerva_math500"}},
        "samples": {"minerva_math500": [{"doc_id": 0}]},
    }
    result_path = tmp_path / "epoch_1_fake.json"
    result_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        run_lm_eval_module,
        "_create_evaluation_tracker",
        lambda output_json_path: FakeTracker(),
    )
    monkeypatch.setattr(
        run_lm_eval_module,
        "_run_simple_evaluate",
        lambda eval_kwargs: captured_eval_kwargs.update(eval_kwargs)
        or dict(fake_result_payload),
    )
    monkeypatch.setattr(
        run_lm_eval_module,
        "find_result_file",
        lambda output_json_path: result_path,
    )

    returned_path = run_lm_eval_for_checkpoint(
        checkpoint_path=tmp_path / "checkpoint-7",
        output_json_path=tmp_path / "epoch_1.json",
        config=LmEvalConfig(tasks=("minerva_math500",), log_samples=True),
        limit=5,
    )

    assert captured_eval_kwargs["tasks"] == ["minerva_math500"]
    assert captured_eval_kwargs["log_samples"] is True
    assert captured_eval_kwargs["limit"] == 5
    saved_results = captured_aggregated["results"]
    assert isinstance(saved_results, dict)
    assert "samples" not in saved_results
    assert captured_sample_logs == [("minerva_math500", [{"doc_id": 0}])]
    assert returned_path == result_path
