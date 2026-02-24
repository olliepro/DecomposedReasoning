"""Tests for breakpoint and leaf variance diagnostics."""

from __future__ import annotations

from branching_eval.metrics_types import (
    compute_breakpoint_variance,
    compute_leaf_variance,
)
from branching_eval.tree_types import BranchTree, LeafRollout, TreeEdge, TreeNode


def test_breakpoint_variance_matches_hand_computed_fixture() -> None:
    """Breakpoint variance should match expected hand-computed values."""

    tree = BranchTree(
        doc_id=0,
        task_name="aime24",
        model_id="m",
        selector_mode="random",
        root_prompt="p",
    )
    tree.add_node(
        node=TreeNode(
            node_id="node_root",
            parent_node_id=None,
            prompt_text="p",
            assistant_prefix="",
            prompt_token_ids=None,
            branch_points_used=0,
        )
    )
    tree.add_node(
        node=TreeNode(
            node_id="node_a",
            parent_node_id="node_root",
            prompt_text="p",
            assistant_prefix="a",
            prompt_token_ids=None,
            branch_points_used=1,
        )
    )
    tree.add_node(
        node=TreeNode(
            node_id="node_b",
            parent_node_id="node_root",
            prompt_text="p",
            assistant_prefix="b",
            prompt_token_ids=None,
            branch_points_used=1,
        )
    )
    tree.edges = [
        TreeEdge(
            edge_id="e1",
            parent_node_id="node_root",
            child_node_id="node_a",
            candidate_pool_id="pool",
            candidate_id=0,
            selector_mode="random",
        ),
        TreeEdge(
            edge_id="e2",
            parent_node_id="node_root",
            child_node_id="node_b",
            candidate_pool_id="pool",
            candidate_id=1,
            selector_mode="random",
        ),
    ]
    tree.leaves = [
        make_leaf(node_id="node_a", verification=1, length=10),
        make_leaf(node_id="node_a", verification=0, length=20),
        make_leaf(node_id="node_b", verification=0, length=30),
        make_leaf(node_id="node_b", verification=0, length=40),
    ]

    variance = compute_breakpoint_variance(tree=tree)
    assert abs(variance.bp1_verification_unweighted - 0.0625) < 1e-9
    assert abs(variance.bp1_verification_weighted - 0.0625) < 1e-9
    assert abs(variance.bp2_verification_unweighted - 0.125) < 1e-9
    assert abs(variance.bp2_verification_weighted - 0.125) < 1e-9
    assert abs(variance.bp1_length_unweighted - 100.0) < 1e-9
    assert abs(variance.bp1_length_weighted - 100.0) < 1e-9
    assert abs(variance.bp2_length_unweighted - 25.0) < 1e-9
    assert abs(variance.bp2_length_weighted - 25.0) < 1e-9


def test_leaf_variance_for_binary_and_length_values() -> None:
    """Leaf variance should return expected binary and length values."""

    leaves = [
        make_leaf(node_id="n", verification=1, length=5),
        make_leaf(node_id="n", verification=0, length=7),
        make_leaf(node_id="n", verification=1, length=9),
    ]
    verification_variance, length_variance = compute_leaf_variance(leaves=leaves)
    assert abs(verification_variance - (2.0 / 9.0)) < 1e-9
    assert abs(length_variance - (8.0 / 3.0)) < 1e-9


def make_leaf(*, node_id: str, verification: int, length: int) -> LeafRollout:
    """Build a leaf fixture row."""

    return LeafRollout(
        leaf_id=f"leaf_{node_id}_{verification}_{length}",
        node_id=node_id,
        text="x",
        token_ids=tuple(range(length)),
        tokens=(),
        verification=verification,
        length_tokens_total=length,
        length_tokens_exec=None,
        stop_reason="stop",
        task_metrics={},
    )
