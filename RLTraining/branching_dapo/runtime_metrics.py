"""Driver-local metric stash for branching DAPO rollout and advantage summaries."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Mapping


@dataclass
class _MetricStore:
    """Mutable metric store used between branching helpers and trainer logging.

    Args:
        generation_metrics: Latest rollout-side branching metrics.
        advantage_metrics: Latest advantage-side branching metrics.
        lock: Thread lock guarding concurrent access.

    Returns:
        Internal metric store for one driver process.
    """

    generation_metrics: dict[str, float] = field(default_factory=dict)
    advantage_metrics: dict[str, float] = field(default_factory=dict)
    lock: Lock = field(default_factory=Lock)

    def update_generation_metrics(self, *, metrics: Mapping[str, float]) -> None:
        """Overwrite stored rollout-side metrics.

        Args:
            metrics: Numeric rollout-side metrics for the current step.

        Returns:
            None.
        """

        with self.lock:
            self.generation_metrics = {str(key): float(value) for key, value in metrics.items()}

    def update_advantage_metrics(self, *, metrics: Mapping[str, float]) -> None:
        """Overwrite stored advantage-side metrics.

        Args:
            metrics: Numeric advantage-side metrics for the current step.

        Returns:
            None.
        """

        with self.lock:
            self.advantage_metrics = {str(key): float(value) for key, value in metrics.items()}

    def pop_metrics(self) -> dict[str, float]:
        """Return and clear the stored branching metrics.

        Args:
            None.

        Returns:
            Combined metric mapping for the latest completed step.
        """

        with self.lock:
            metrics = {**self.generation_metrics, **self.advantage_metrics}
            self.generation_metrics = {}
            self.advantage_metrics = {}
        return metrics


_STORE = _MetricStore()


def record_generation_metrics(*, metrics: Mapping[str, float]) -> None:
    """Record rollout-side branching metrics for later logging.

    Args:
        metrics: Numeric rollout-side metrics.

    Returns:
        None.
    """

    _STORE.update_generation_metrics(metrics=metrics)


def record_advantage_metrics(*, metrics: Mapping[str, float]) -> None:
    """Record advantage-side branching metrics for later logging.

    Args:
        metrics: Numeric advantage-side metrics.

    Returns:
        None.
    """

    _STORE.update_advantage_metrics(metrics=metrics)


def consume_runtime_metrics() -> dict[str, float]:
    """Consume the latest rollout and advantage metrics.

    Args:
        None.

    Returns:
        Combined metric mapping.
    """

    return _STORE.pop_metrics()
