"""lm_eval task adapter for doc iteration, scoring, and aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from branching_eval.aime_bridge import verify_aime_response


@dataclass(frozen=True)
class DocRecord:
    """One task document with resolved prompt text.

    Args:
        doc_id: Sequential document id.
        doc_payload: Raw task document payload.
        prompt_text: Prompt string from `task.doc_to_text`.

    Returns:
        Dataclass containing one evaluation document.
    """

    doc_id: int
    doc_payload: dict[str, Any]
    prompt_text: str


class LmEvalAdapter:
    """Wrapper around one lm_eval task object.

    Args:
        task_name: lm_eval task name.

    Returns:
        Task adapter exposing docs, scoring, and aggregation methods.

    Example:
        >>> adapter = LmEvalAdapter(task_name="aime24")  # doctest: +SKIP
    """

    def __init__(self, *, task_name: str) -> None:
        self.task_name = task_name
        self._task = self._load_task(task_name=task_name)

    def docs(self, *, limit: int | None) -> list[DocRecord]:
        """Return task docs with resolved prompts.

        Args:
            limit: Optional doc cap.

        Returns:
            Ordered `DocRecord` list.
        """

        docs = self._read_docs()
        if limit is not None:
            docs = docs[: max(0, limit)]
        return [
            DocRecord(
                doc_id=doc_id,
                doc_payload=doc_payload,
                prompt_text=str(self._task.doc_to_text(doc_payload)),
            )
            for doc_id, doc_payload in enumerate(docs)
        ]

    def score_response(
        self, *, doc: dict[str, Any], response_text: str
    ) -> dict[str, Any]:
        """Score one response with task `process_results`.

        Args:
            doc: Task document payload.
            response_text: Model output text.

        Returns:
            Task metric mapping.
        """

        metrics = self._task.process_results(doc, [response_text])
        assert isinstance(metrics, dict), "process_results must return a dict"
        return dict(metrics)

    def verification(self, *, doc: dict[str, Any], response_text: str) -> int:
        """Return binary verification score for one response.

        Args:
            doc: Task document payload.
            response_text: Model output text.

        Returns:
            Verification score in `{0, 1}`.
        """

        if self.task_name in {"aime24", "aime25"}:
            return verify_aime_response(doc=doc, response_text=response_text)
        scored = self.score_response(doc=doc, response_text=response_text)
        return _fallback_binary_score(metrics=scored)

    def aggregate_doc_metrics(
        self, *, rollout_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reduce one document's rollout metric rows to a single mapping.

        Args:
            rollout_metrics: Per-rollout metric dict rows.

        Returns:
            Mean-reduced per-doc metric mapping.
        """

        if not rollout_metrics:
            return {}
        keys = sorted(rollout_metrics[0])
        reduced: dict[str, Any] = {}
        for key in keys:
            numeric_values = [
                float(metric[key])
                for metric in rollout_metrics
                if isinstance(metric.get(key), (int, float))
            ]
            if numeric_values:
                reduced[key] = sum(numeric_values) / len(numeric_values)
                continue
            reduced[key] = rollout_metrics[0].get(key)
        return reduced

    def aggregate_task_metrics(
        self, *, per_doc_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Aggregate reduced per-doc metrics with task aggregation.

        Args:
            per_doc_metrics: One reduced metric mapping per document.

        Returns:
            Aggregated task metrics mapping.
        """

        if not per_doc_metrics:
            return {}
        aggregation_method = getattr(self._task, "aggregation", None)
        if not callable(aggregation_method):
            return _mean_numeric_metrics(per_doc_metrics=per_doc_metrics)
        spec = _aggregation_spec(task_aggregation=aggregation_method)
        if spec is not None:
            return _aggregate_with_spec(
                per_doc_metrics=per_doc_metrics,
                aggregation_spec=spec,
            )
        direct = _direct_aggregation_call(
            task_aggregation=aggregation_method,
            per_doc_metrics=per_doc_metrics,
        )
        if direct is not None:
            return direct
        return _mean_numeric_metrics(per_doc_metrics=per_doc_metrics)

    def _load_task(self, *, task_name: str) -> Any:
        from lm_eval import tasks

        task_dict = tasks.get_task_dict([task_name])
        assert task_dict, f"No task loaded: {task_name}"
        task = next(iter(task_dict.values()))
        if hasattr(task, "build_all_requests"):
            task.build_all_requests()
        elif hasattr(task, "build_requests"):
            task.build_requests()
        return task

    def _read_docs(self) -> list[dict[str, Any]]:
        docs = self._task.validation_docs() or self._task.test_docs()
        assert docs is not None, f"No docs available for task: {self.task_name}"
        return [dict(doc) for doc in docs]


def _fallback_binary_score(*, metrics: dict[str, Any]) -> int:
    for value in metrics.values():
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, (int, float)):
            return int(round(float(value)))
    return 0


def _aggregation_spec(*, task_aggregation: Any) -> dict[str, Any] | None:
    try:
        resolved = task_aggregation()
    except TypeError:
        return None
    if not isinstance(resolved, dict):
        return None
    return dict(resolved)


def _direct_aggregation_call(
    *, task_aggregation: Any, per_doc_metrics: list[dict[str, Any]]
) -> dict[str, Any] | None:
    try:
        aggregated = task_aggregation(per_doc_metrics)
    except TypeError:
        return None
    if not isinstance(aggregated, dict):
        return None
    return dict(aggregated)


def _aggregate_with_spec(
    *, per_doc_metrics: list[dict[str, Any]], aggregation_spec: dict[str, Any]
) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    for metric_name, reducer in aggregation_spec.items():
        values = [row[metric_name] for row in per_doc_metrics if metric_name in row]
        if not values:
            continue
        if callable(reducer):
            try:
                aggregated[metric_name] = reducer(values)
                continue
            except Exception:
                pass
        numeric_values = [
            float(value) for value in values if isinstance(value, (int, float))
        ]
        if numeric_values:
            aggregated[metric_name] = sum(numeric_values) / len(numeric_values)
            continue
        aggregated[metric_name] = values[0]
    return aggregated


def _mean_numeric_metrics(*, per_doc_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    keys = sorted({key for row in per_doc_metrics for key in row})
    reduced: dict[str, Any] = {}
    for key in keys:
        values = [row.get(key) for row in per_doc_metrics]
        numeric_values = [
            float(value) for value in values if isinstance(value, (int, float))
        ]
        if numeric_values:
            reduced[key] = sum(numeric_values) / len(numeric_values)
            continue
        present_values = [value for value in values if value is not None]
        if present_values:
            reduced[key] = present_values[0]
    return reduced
