"""Tests for replay-based partial-doc resume state reconstruction."""

from __future__ import annotations

from pathlib import Path

from branching_eval.artifact_store import ArtifactStore
from branching_eval.event_types import EventContext
from branching_eval.resume_state import replay_branching_resume_state


def _context(*, store: ArtifactStore, doc_id: int, doc_attempt: int) -> EventContext:
    """Build canonical context for one doc attempt in test run logs."""

    return EventContext(
        run_id=store.run_id,
        doc_id=doc_id,
        doc_attempt=doc_attempt,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )


def test_resume_replay_reconstructs_partial_frontier(tmp_path: Path) -> None:
    """Replay should reconstruct child frontier state for incomplete attempt."""

    store = ArtifactStore(run_dir=tmp_path / "run", reuse_candidate_pools=False)
    context = _context(store=store, doc_id=0, doc_attempt=0)
    store.append_event(context=context, event_type="doc_started", payload={})
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": "node_root",
            "parent_node_id": None,
            "branch_points_used": 0,
        },
    )
    store.append_event(
        context=context,
        event_type="node_created",
        payload={
            "node_id": "node_child",
            "parent_node_id": "node_root",
            "branch_points_used": 1,
        },
    )
    store.append_event(
        context=context,
        event_type="edge_selected",
        payload={
            "edge_id": "edge_root_child",
            "parent_node_id": "node_root",
            "child_node_id": "node_child",
            "candidate_pool_id": "pool_0",
            "candidate_id": 0,
            "selector_mode": "random",
            "candidate_text_normalized": "prefix ",
            "candidate_token_ids_normalized": [10],
        },
    )
    store.append_event(
        context=context,
        event_type="decode_chunk",
        payload={
            "node_id": "node_child",
            "chunk_text": "tail",
            "chunk_was_normalized": False,
            "chunk_token_ids": [11, 12],
            "chunk_token_ids_source": "generated_token_delta",
            "finish_reason": "length",
            "generated_tokens_before_chunk": 0,
            "generated_tokens_after_chunk": 2,
            "branching_enabled": True,
        },
    )

    state = replay_branching_resume_state(
        events=store.read_events(),
        prompt_text="Solve this.",
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    assert state.frontier
    assert len(state.frontier) == 1
    frontier = state.frontier[0]
    assert frontier.node_id == "node_child"
    assert frontier.assistant_prefix == "prefix tail"
    assert frontier.token_ids == (10, 11, 12)


def test_resume_replay_tracks_pre_scored_leaves(tmp_path: Path) -> None:
    """Replay should expose pre-scored leaves for duplicate-score suppression."""

    store = ArtifactStore(run_dir=tmp_path / "run", reuse_candidate_pools=False)
    context = _context(store=store, doc_id=1, doc_attempt=0)
    store.append_event(context=context, event_type="doc_started", payload={})
    store.append_event(
        context=context,
        event_type="leaf_scored",
        payload={
            "leaf_id": "leaf_1",
            "node_id": "node_root",
            "text": "answer",
            "token_ids": [1, 2, 3],
            "tokens": [],
            "verification": 1,
            "length_tokens_total": 3,
            "length_tokens_exec": 3,
            "stop_reason": "length",
            "task_metrics": {"acc": 1.0},
        },
    )

    state = replay_branching_resume_state(
        events=store.read_events(),
        prompt_text="Solve this.",
        run_id=store.run_id,
        doc_id=1,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    assert "leaf_1" in state.pre_scored_leaves
    assert any(leaf.leaf_id == "leaf_1" for leaf in state.tree.leaves)
