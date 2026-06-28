"""Native scheduler event metric aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def event_int(*, event: dict[str, object], key: str, default: int = 0) -> int:
    """Read an integer field from a native scheduler event."""

    raw = event.get(key, default)
    assert isinstance(raw, (int, float, str)), f"{key} must be scalar"
    return int(raw)


def event_float(*, event: dict[str, object], key: str, default: float = 0.0) -> float:
    """Read a float field from a native scheduler event."""

    raw = event.get(key, default)
    assert isinstance(raw, (int, float, str)), f"{key} must be scalar"
    return float(raw)


def event_int_list(*, event: dict[str, object], key: str) -> list[int]:
    """Read an integer list field from a native scheduler event."""

    raw = event.get(key, [])
    assert isinstance(raw, list), f"{key} must be a list"
    return [int(value) for value in raw]


@dataclass
class NativeMetricAccumulator:
    """Incremental native scheduler event metrics."""

    boundary_fire_count: int = 0
    branch_count: int = 0
    kv_blocks_allocated: int = 0
    kv_blocks_freed: int = 0
    async_tokens_discarded: int = 0
    prefill_tokens_avoided: int = 0
    branch_pool_queued_count: int = 0
    branch_pool_admitted_count: int = 0
    branch_pool_blocked_count: int = 0
    max_live_branch_pools: int = 0
    max_queued_branch_pools: int = 0
    min_branch_free_blocks: int | None = None
    min_branch_seq_slots: int | None = None
    hidden_vector_child_count: int = 0
    selected_ids: list[int] = field(default_factory=list)
    returned_ids: list[list[int]] = field(default_factory=list)
    returned_branch_counts: list[int] = field(default_factory=list)
    branch_depth_used: list[int] = field(default_factory=list)
    branch_depth_limits: list[int] = field(default_factory=list)
    top_k_candidate_ids: list[list[int]] = field(default_factory=list)
    unique_candidate_counts: list[int] = field(default_factory=list)
    candidate_bounds: list[int] = field(default_factory=list)
    diversity_sources: list[str] = field(default_factory=list)
    pool_hidden_pairwise_diversities: list[float] = field(default_factory=list)
    selected_hidden_diversities: list[float] = field(default_factory=list)

    def observe(self, *, event: dict[str, object]) -> None:
        """Update metrics from one native scheduler event."""

        event_name = event.get("event")
        if event_name in {"branch_start", "branch_wave_start"}:
            self.observe_start(event=event, is_boundary=event_name == "branch_start")
        if event_name == "branch_pool_queued":
            self.observe_pool_queue(event=event)
        if event_name == "branch_pool_blocked":
            self.observe_pool_blocked(event=event)
        if event_name == "branch_pool_admitted":
            self.observe_pool_admitted(event=event)
        if event_name in {"branch_promote", "branch_return"}:
            self.observe_completion(
                event=event, is_return=event_name == "branch_return"
            )

    def observe_start(self, *, event: dict[str, object], is_boundary: bool) -> None:
        """Update metrics from a branch-start or branch-wave event."""

        candidate_count = event_int(event=event, key="candidate_count")
        if is_boundary:
            self.boundary_fire_count += 1
            self.async_tokens_discarded += event_int(
                event=event,
                key="async_tokens_to_discard",
            )
        self.branch_count += candidate_count
        self.prefill_tokens_avoided += (
            event_int(event=event, key="fork_tokens") * candidate_count
        )
        self.kv_blocks_allocated += (
            event_int(event=event, key="shared_blocks") * candidate_count
        )

    def observe_pool_queue(self, *, event: dict[str, object]) -> None:
        """Update metrics when a branch pool is queued."""

        self.branch_pool_queued_count += 1
        self.observe_pool_pressure(event=event)

    def observe_pool_blocked(self, *, event: dict[str, object]) -> None:
        """Update metrics when admission is blocked by resource pressure."""

        self.branch_pool_blocked_count += 1
        self.observe_pool_pressure(event=event)

    def observe_pool_admitted(self, *, event: dict[str, object]) -> None:
        """Update metrics when a queued pool is admitted to vLLM."""

        self.branch_pool_admitted_count += 1
        self.branch_count += event_int(event=event, key="candidate_count")
        self.prefill_tokens_avoided += event_int(
            event=event, key="fork_tokens"
        ) * event_int(event=event, key="candidate_count")
        self.kv_blocks_allocated += event_int(
            event=event, key="shared_blocks"
        ) * event_int(event=event, key="candidate_count")
        self.observe_pool_pressure(event=event)

    def observe_pool_pressure(self, *, event: dict[str, object]) -> None:
        """Track high/low-water marks for dynamic branch admission."""

        self.max_live_branch_pools = max(
            self.max_live_branch_pools,
            event_int(event=event, key="live_branch_pool_count"),
        )
        self.max_queued_branch_pools = max(
            self.max_queued_branch_pools,
            event_int(event=event, key="queued_branch_pool_count"),
        )
        free_blocks = event_int(event=event, key="free_blocks", default=-1)
        if free_blocks >= 0:
            self.min_branch_free_blocks = (
                free_blocks
                if self.min_branch_free_blocks is None
                else min(self.min_branch_free_blocks, free_blocks)
            )
        seq_slots = event_int(event=event, key="seq_slots", default=-1)
        if seq_slots >= 0:
            self.min_branch_seq_slots = (
                seq_slots
                if self.min_branch_seq_slots is None
                else min(self.min_branch_seq_slots, seq_slots)
            )

    def observe_completion(self, *, event: dict[str, object], is_return: bool) -> None:
        """Update metrics from branch promotion or branch return."""

        if is_return:
            self.returned_ids.append(
                event_int_list(event=event, key="returned_candidate_ids")
            )
            self.returned_branch_counts.append(
                event_int(event=event, key="returned_branch_count")
            )
        else:
            self.selected_ids.append(
                event_int(event=event, key="selected_candidate_id")
            )
        self.top_k_candidate_ids.append(
            event_int_list(event=event, key="top_k_candidate_ids")
        )
        self.branch_depth_used.append(event_int(event=event, key="branch_depth_used"))
        self.branch_depth_limits.append(
            event_int(event=event, key="branch_depth_limit")
        )
        self.unique_candidate_counts.append(
            event_int(event=event, key="unique_candidate_count")
        )
        self.candidate_bounds.append(event_int(event=event, key="candidate_bound"))
        self.kv_blocks_freed += event_int(event=event, key="children_freed")
        if "diversity_vector_source" in event:
            self.diversity_sources.append(str(event["diversity_vector_source"]))
        self.hidden_vector_child_count += event_int(
            event=event,
            key="hidden_vector_child_count",
        )
        self.pool_hidden_pairwise_diversities.append(
            event_float(event=event, key="pool_hidden_pairwise_diversity")
        )
        self.selected_hidden_diversities.append(
            event_float(event=event, key="selected_hidden_diversity")
        )

    def diversity_source(self) -> str:
        """Return the strongest observed diversity-source label."""

        if any(source == "model_hidden_state" for source in self.diversity_sources):
            return "model_hidden_state"
        return self.diversity_sources[-1] if self.diversity_sources else "none"

    def payload(self) -> dict[str, object]:
        """Return JSON-serializable aggregate metrics."""

        return {
            "boundary_fire_count": self.boundary_fire_count,
            "branch_count": self.branch_count,
            "selected_candidate_ids": self.selected_ids,
            "returned_candidate_ids": self.returned_ids,
            "returned_branch_counts": self.returned_branch_counts,
            "branch_depth_used": self.branch_depth_used,
            "branch_depth_limits": self.branch_depth_limits,
            "top_k_candidate_ids": self.top_k_candidate_ids,
            "unique_candidate_counts": self.unique_candidate_counts,
            "candidate_bounds": self.candidate_bounds,
            "prefill_tokens_avoided": self.prefill_tokens_avoided,
            "kv_blocks_allocated": self.kv_blocks_allocated,
            "kv_blocks_copied": 0,
            "kv_blocks_freed": self.kv_blocks_freed,
            "async_tokens_discarded": self.async_tokens_discarded,
            "branch_pool_queued_count": self.branch_pool_queued_count,
            "branch_pool_admitted_count": self.branch_pool_admitted_count,
            "branch_pool_blocked_count": self.branch_pool_blocked_count,
            "max_live_branch_pools": self.max_live_branch_pools,
            "max_queued_branch_pools": self.max_queued_branch_pools,
            "min_branch_free_blocks": self.min_branch_free_blocks or 0,
            "min_branch_seq_slots": self.min_branch_seq_slots or 0,
            "diversity_vector_source": self.diversity_source(),
            "hidden_vector_child_count": self.hidden_vector_child_count,
            "pool_hidden_pairwise_diversity": max(
                self.pool_hidden_pairwise_diversities or [0.0]
            ),
            "selected_hidden_diversity": max(self.selected_hidden_diversities or [0.0]),
        }


def empty_native_event_metrics() -> dict[str, object]:
    """Return zero-valued native branch metrics."""

    return NativeMetricAccumulator().payload()


def read_native_event_metrics(*, path: Path) -> dict[str, object]:
    """Summarize native scheduler event JSONL for benchmark metrics."""

    if not path.exists():
        return empty_native_event_metrics()
    metrics = NativeMetricAccumulator()
    for line in path.read_text(encoding="utf-8").splitlines():
        event = dict(json.loads(line))
        metrics.observe(event=event)
    return metrics.payload()
