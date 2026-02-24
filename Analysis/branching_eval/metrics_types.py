"""Variance and summary metric dataclasses for branching analysis."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median

from branching_eval.tree_types import BranchTree, LeafRollout


@dataclass(frozen=True)
class LengthSummary:
    """Summary stats for rollout lengths.

    Args:
        count: Number of length observations.
        mean_value: Arithmetic mean.
        median_value: Median value.
        std_value: Population standard deviation.

    Returns:
        Dataclass containing length summary statistics.
    """

    count: int
    mean_value: float
    median_value: float
    std_value: float


@dataclass(frozen=True)
class BreakpointVariance:
    """Breakpoint-attributed variance values for one document.

    Args:
        bp1_verification_unweighted: Unweighted variance across first-level child means.
        bp1_verification_weighted: Weighted variance across first-level child means.
        bp1_length_unweighted: Unweighted variance across first-level child length means.
        bp1_length_weighted: Weighted variance across first-level child length means.
        bp2_verification_unweighted: Mean of child-level bp2 verification variances.
        bp2_verification_weighted: Leaf-weighted bp2 verification variance.
        bp2_length_unweighted: Mean of child-level bp2 length variances.
        bp2_length_weighted: Leaf-weighted bp2 length variance.

    Returns:
        Dataclass holding per-doc breakpoint variance outputs.
    """

    bp1_verification_unweighted: float
    bp1_verification_weighted: float
    bp1_length_unweighted: float
    bp1_length_weighted: float
    bp2_verification_unweighted: float
    bp2_verification_weighted: float
    bp2_length_unweighted: float
    bp2_length_weighted: float


@dataclass(frozen=True)
class DocDiagnostics:
    """Per-document branching diagnostics.

    Args:
        doc_id: Document id.
        selector_mode: Selector mode for this tree.
        verification_variance_leaf: Leaf-level verification variance.
        length_variance_leaf: Leaf-level length variance.
        breakpoint_variance: Breakpoint-attributed variance metrics.
        length_summary: Length summary stats.

    Returns:
        Dataclass containing per-doc diagnostics.
    """

    doc_id: int
    selector_mode: str
    verification_variance_leaf: float
    length_variance_leaf: float
    breakpoint_variance: BreakpointVariance
    length_summary: LengthSummary


@dataclass(frozen=True)
class AggregateDiagnostics:
    """Cross-document aggregate diagnostics.

    Args:
        selector_mode: Selector mode.
        doc_count: Number of documents aggregated.
        mean_bp1_verification_unweighted: Mean of per-doc bp1 unweighted verification variance.
        mean_bp1_verification_weighted: Mean of per-doc bp1 weighted verification variance.
        mean_bp2_verification_unweighted: Mean of per-doc bp2 unweighted verification variance.
        mean_bp2_verification_weighted: Mean of per-doc bp2 weighted verification variance.
        mean_bp1_length_unweighted: Mean of per-doc bp1 unweighted length variance.
        mean_bp1_length_weighted: Mean of per-doc bp1 weighted length variance.
        mean_bp2_length_unweighted: Mean of per-doc bp2 unweighted length variance.
        mean_bp2_length_weighted: Mean of per-doc bp2 weighted length variance.

    Returns:
        Dataclass with aggregate breakpoint diagnostics.
    """

    selector_mode: str
    doc_count: int
    mean_bp1_verification_unweighted: float
    mean_bp1_verification_weighted: float
    mean_bp2_verification_unweighted: float
    mean_bp2_verification_weighted: float
    mean_bp1_length_unweighted: float
    mean_bp1_length_weighted: float
    mean_bp2_length_unweighted: float
    mean_bp2_length_weighted: float


def summarize_lengths(*, lengths: list[int]) -> LengthSummary:
    """Compute summary statistics for rollout lengths.

    Args:
        lengths: Length observations.

    Returns:
        `LengthSummary` for the input lengths.
    """

    if not lengths:
        return LengthSummary(count=0, mean_value=0.0, median_value=0.0, std_value=0.0)
    mean_value = float(mean(lengths))
    variance_value = _variance(values=[float(value) for value in lengths])
    return LengthSummary(
        count=len(lengths),
        mean_value=mean_value,
        median_value=float(median(lengths)),
        std_value=float(variance_value**0.5),
    )


def compute_leaf_variance(*, leaves: list[LeafRollout]) -> tuple[float, float]:
    """Compute leaf-level verification and length variances.

    Args:
        leaves: Completed leaf rollout rows.

    Returns:
        Tuple `(verification_variance, length_variance)`.
    """

    if not leaves:
        return 0.0, 0.0
    verifications = [float(leaf.verification) for leaf in leaves]
    lengths = [float(leaf.length_tokens_total) for leaf in leaves]
    return _variance(values=verifications), _variance(values=lengths)


def compute_breakpoint_variance(*, tree: BranchTree) -> BreakpointVariance:
    """Compute bp1/bp2 variance metrics from one branch tree.

    Args:
        tree: Branch tree with leaves and edge lineage.

    Returns:
        Breakpoint variance metrics for this tree.
    """

    child_to_leaves = _group_first_branch_child_leaves(tree=tree)
    if not child_to_leaves:
        return BreakpointVariance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    child_verification_means = []
    child_length_means = []
    child_weights = []
    bp2_verification_values = []
    bp2_length_values = []
    for leaves in child_to_leaves.values():
        child_weights.append(float(len(leaves)))
        child_verification_means.append(
            mean(float(leaf.verification) for leaf in leaves)
        )
        child_length_means.append(
            mean(float(leaf.length_tokens_total) for leaf in leaves)
        )
        bp2_verification_values.append(
            _variance(values=[float(leaf.verification) for leaf in leaves])
        )
        bp2_length_values.append(
            _variance(values=[float(leaf.length_tokens_total) for leaf in leaves])
        )
    return BreakpointVariance(
        bp1_verification_unweighted=_variance(values=child_verification_means),
        bp1_verification_weighted=_weighted_variance(
            values=child_verification_means,
            weights=child_weights,
        ),
        bp1_length_unweighted=_variance(values=child_length_means),
        bp1_length_weighted=_weighted_variance(
            values=child_length_means, weights=child_weights
        ),
        bp2_verification_unweighted=float(mean(bp2_verification_values)),
        bp2_verification_weighted=_weighted_mean(
            values=bp2_verification_values, weights=child_weights
        ),
        bp2_length_unweighted=float(mean(bp2_length_values)),
        bp2_length_weighted=_weighted_mean(
            values=bp2_length_values, weights=child_weights
        ),
    )


def aggregate_diagnostics(*, diagnostics: list[DocDiagnostics]) -> AggregateDiagnostics:
    """Aggregate per-document diagnostics across documents.

    Args:
        diagnostics: Per-doc diagnostics for one selector mode.

    Returns:
        Aggregate diagnostics dataclass.
    """

    if not diagnostics:
        return AggregateDiagnostics(
            selector_mode="unknown",
            doc_count=0,
            mean_bp1_verification_unweighted=0.0,
            mean_bp1_verification_weighted=0.0,
            mean_bp2_verification_unweighted=0.0,
            mean_bp2_verification_weighted=0.0,
            mean_bp1_length_unweighted=0.0,
            mean_bp1_length_weighted=0.0,
            mean_bp2_length_unweighted=0.0,
            mean_bp2_length_weighted=0.0,
        )
    selector_mode = diagnostics[0].selector_mode
    return AggregateDiagnostics(
        selector_mode=selector_mode,
        doc_count=len(diagnostics),
        mean_bp1_verification_unweighted=_mean_by(
            diagnostics, "bp1_verification_unweighted"
        ),
        mean_bp1_verification_weighted=_mean_by(
            diagnostics, "bp1_verification_weighted"
        ),
        mean_bp2_verification_unweighted=_mean_by(
            diagnostics, "bp2_verification_unweighted"
        ),
        mean_bp2_verification_weighted=_mean_by(
            diagnostics, "bp2_verification_weighted"
        ),
        mean_bp1_length_unweighted=_mean_by(diagnostics, "bp1_length_unweighted"),
        mean_bp1_length_weighted=_mean_by(diagnostics, "bp1_length_weighted"),
        mean_bp2_length_unweighted=_mean_by(diagnostics, "bp2_length_unweighted"),
        mean_bp2_length_weighted=_mean_by(diagnostics, "bp2_length_weighted"),
    )


def _mean_by(diagnostics: list[DocDiagnostics], field_name: str) -> float:
    values = [
        float(getattr(doc.breakpoint_variance, field_name)) for doc in diagnostics
    ]
    return float(mean(values))


def _variance(*, values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = float(mean(values))
    return float(sum((value - mean_value) ** 2 for value in values) / len(values))


def _weighted_mean(*, values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 0.0
    weighted_sum = sum(value * weight for value, weight in zip(values, weights))
    return float(weighted_sum / total_weight)


def _weighted_variance(*, values: list[float], weights: list[float]) -> float:
    total_weight = sum(weights)
    if total_weight <= 0.0:
        return 0.0
    weighted_mean = _weighted_mean(values=values, weights=weights)
    numerator = sum(
        weight * ((value - weighted_mean) ** 2)
        for value, weight in zip(values, weights)
    )
    return float(numerator / total_weight)


def _group_first_branch_child_leaves(
    *, tree: BranchTree
) -> dict[str, list[LeafRollout]]:
    edge_by_child: dict[str, str] = {}
    for edge in tree.edges:
        edge_by_child[edge.child_node_id] = edge.parent_node_id
    grouped: dict[str, list[LeafRollout]] = {}
    for leaf in tree.leaves:
        first_child = _first_branch_child(
            node_id=leaf.node_id, edge_by_child=edge_by_child
        )
        key = first_child or "no_branch"
        grouped.setdefault(key, []).append(leaf)
    return grouped


def _first_branch_child(*, node_id: str, edge_by_child: dict[str, str]) -> str | None:
    chain = []
    current = node_id
    while current in edge_by_child:
        chain.append(current)
        current = edge_by_child[current]
    if not chain:
        return None
    return chain[-1]
