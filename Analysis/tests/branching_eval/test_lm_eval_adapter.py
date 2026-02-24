"""Tests for lm_eval aggregation compatibility paths."""

from __future__ import annotations

from branching_eval.lm_eval_adapter import LmEvalAdapter


class FakeTaskAggregationSpec:
    """Task mock with zero-arg aggregation spec API."""

    def aggregation(self) -> dict[str, object]:
        return {
            "acc": lambda values: sum(float(value) for value in values) / len(values),
            "len": max,
        }


class FakeTaskAggregationDirect:
    """Task mock with direct aggregation API."""

    def aggregation(self, per_doc_metrics: list[dict[str, float]]) -> dict[str, float]:
        acc_values = [float(row["acc"]) for row in per_doc_metrics]
        return {"acc": sum(acc_values) / len(acc_values)}


class FakeTaskNoAggregation:
    """Task mock without callable aggregation method."""

    aggregation = None


def _adapter_with_task(task: object) -> LmEvalAdapter:
    adapter = LmEvalAdapter.__new__(LmEvalAdapter)
    adapter.task_name = "aime24"
    adapter._task = task
    return adapter


def test_aggregate_task_metrics_supports_spec_api() -> None:
    """Adapter should apply reducer functions from aggregation spec mappings."""

    adapter = _adapter_with_task(FakeTaskAggregationSpec())
    aggregated = adapter.aggregate_task_metrics(
        per_doc_metrics=[
            {"acc": 1.0, "len": 3.0},
            {"acc": 0.0, "len": 5.0},
        ]
    )
    assert aggregated["acc"] == 0.5
    assert aggregated["len"] == 5.0


def test_aggregate_task_metrics_supports_direct_api() -> None:
    """Adapter should support direct aggregation(per_doc_metrics) methods."""

    adapter = _adapter_with_task(FakeTaskAggregationDirect())
    aggregated = adapter.aggregate_task_metrics(
        per_doc_metrics=[
            {"acc": 1.0},
            {"acc": 0.0},
        ]
    )
    assert aggregated["acc"] == 0.5


def test_aggregate_task_metrics_falls_back_to_numeric_mean() -> None:
    """Adapter should fallback to numeric mean when aggregation is unavailable."""

    adapter = _adapter_with_task(FakeTaskNoAggregation())
    aggregated = adapter.aggregate_task_metrics(
        per_doc_metrics=[
            {"acc": 1.0, "tag": "x"},
            {"acc": 0.0, "tag": "y"},
        ]
    )
    assert aggregated["acc"] == 0.5
    assert aggregated["tag"] == "x"
