"""Typed selector-mode definitions for branch candidate selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SelectorMode = Literal["cluster_across", "embed_diverse", "within_cluster", "random"]
WithinClusterHeuristic = Literal["random_non_other_min_k"]


@dataclass(frozen=True)
class SelectorParams:
    """Selector parameter bundle used by branch-point selection.

    Args:
        branch_fanout: Number of candidates to select (`K`).
        max_clusters: Max clusters considered by `cluster_across`.
        within_cluster_heuristic: Heuristic name for within-cluster selection.

    Returns:
        Dataclass containing selector-specific parameters.

    Example:
        >>> params = SelectorParams(branch_fanout=4, max_clusters=4)
        >>> params.branch_fanout
        4
    """

    branch_fanout: int
    max_clusters: int = 4
    within_cluster_heuristic: WithinClusterHeuristic = "random_non_other_min_k"


@dataclass(frozen=True)
class SelectionOutcome:
    """Selection result for one branch point and one selector mode.

    Args:
        selector_mode: Selector mode used for this outcome.
        selected_candidate_ids: Candidate ids chosen by this selector.
        cluster_by_candidate_id: Optional cluster label by candidate id.
        embedding_by_candidate_id: Optional embedding vectors by candidate id.

    Returns:
        Dataclass containing selection outputs and diagnostics.
    """

    selector_mode: SelectorMode
    selected_candidate_ids: tuple[int, ...]
    cluster_by_candidate_id: dict[int, str] | None = None
    embedding_by_candidate_id: dict[int, tuple[float, ...]] | None = None
