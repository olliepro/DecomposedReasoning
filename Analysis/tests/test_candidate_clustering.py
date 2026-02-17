"""Tests for prompt-based steer-candidate clustering helpers."""

from __future__ import annotations

import candidate_clustering
from pathlib import Path

from candidate_clustering import (
    ClusteringConfig,
    DedupItem,
    build_cluster_prompt,
    cluster_candidates_by_step,
    coerce_assignments,
    strip_steer_suffix,
)


def test_strip_steer_suffix_removes_trailing_close_tag() -> None:
    """Strip helper should remove trailing close-tag content."""

    assert strip_steer_suffix(text="Try factoring</steer>") == "Try factoring"
    assert strip_steer_suffix(text="Try factoring</steer>   ") == "Try factoring"
    assert (
        strip_steer_suffix(text='Try "A.L.G.O.R.I.T.H.M" pun</ste ')
        == 'Try "A.L.G.O.R.I.T.H.M" pun'
    )
    assert strip_steer_suffix(text="Try factoring</st") == "Try factoring"
    assert strip_steer_suffix(text="Try factoring") == "Try factoring"


def test_build_cluster_prompt_previous_context_behavior() -> None:
    """Prompt should include previous context only when count is positive."""

    items = (DedupItem(item_id=1, text="Try substitution", count=3),)
    with_context = build_cluster_prompt(
        previous_count=2,
        previous_chain="Factor >> Substitute",
        items=items,
    )
    without_context = build_cluster_prompt(
        previous_count=0,
        previous_chain="",
        items=items,
    )
    assert "Current step index" not in with_context
    assert "Previous 2 selected steps: Factor >> Substitute" in with_context
    assert "Previous 0 selected steps" not in without_context


def test_coerce_assignments_fills_missing_with_other() -> None:
    """Cluster assignment coercion should cover every dedup item id."""

    items = (
        DedupItem(item_id=1, text="A", count=5),
        DedupItem(item_id=2, text="B", count=3),
        DedupItem(item_id=3, text="C", count=2),
    )
    assignments = coerce_assignments(
        items=items,
        clusters=[{"name": "cluster_a", "member_ids": [1, 2, 2, 99]}],
    )
    assert assignments[1] == "cluster_a"
    assert assignments[2] == "cluster_a"
    assert assignments[3] == "other"
    assert sorted(assignments) == [1, 2, 3]


def test_cluster_candidates_by_step_builds_assignments_and_summaries(
    tmp_path: Path,
) -> None:
    """Disabled mode should still build complete fallback cluster artifacts."""

    candidates = [
        {"step_index": 0, "candidate_index": 0, "text": "Try substitution</steer>"},
        {"step_index": 0, "candidate_index": 1, "text": "Try substitution</steer>"},
        {"step_index": 0, "candidate_index": 2, "text": "Try substitution</steer>"},
        {"step_index": 0, "candidate_index": 3, "text": "Factor first</steer>"},
        {"step_index": 0, "candidate_index": 4, "text": "Graph it</steer>"},
    ]
    artifacts = cluster_candidates_by_step(
        candidates=candidates,
        steps=[],
        config=ClusteringConfig(
            enabled=False,
            gemini_model="gemini-3-flash",
            temperature=0.2,
            seed=7,
            previous_steps_window=5,
            env_paths=(tmp_path / ".env",),
        ),
    )
    assert artifacts.mode == "disabled"
    assert 0 in artifacts.summaries_by_step
    assert len(artifacts.assignments_by_candidate) == 5
    assert artifacts.assignments_by_candidate[(0, 0)].clean_text == "Try substitution"
    total = sum(summary.count for summary in artifacts.summaries_by_step[0])
    assert total == 5
    assert artifacts.summaries_by_step[0][0].name == "try_substitution"


def test_cluster_candidates_by_step_reuses_prompt_cache(
    tmp_path: Path, monkeypatch
) -> None:
    """Clustering should reuse cached Gemini responses across reruns."""

    call_count = 0

    def fake_call(
        *, api_key: str, prompt: str, model_id: str, temperature: float
    ) -> list[dict[str, object]]:
        nonlocal call_count
        call_count += 1
        return [{"name": "solver_plan", "member_ids": [1, 2]}]

    monkeypatch.setenv("VERTEX_KEY", "test-key")
    monkeypatch.setattr(candidate_clustering, "call_gemini_clusters", fake_call)
    candidates = [
        {"step_index": 0, "candidate_index": 0, "text": "Try substitution</steer>"},
        {"step_index": 0, "candidate_index": 1, "text": "Factor first</steer>"},
    ]
    config = ClusteringConfig(
        enabled=True,
        gemini_model="gemini-3-flash-preview",
        temperature=0.2,
        previous_steps_window=5,
        max_concurrent_requests=4,
        cache_path=tmp_path / "cluster_cache.json",
        env_paths=(),
    )
    first = cluster_candidates_by_step(candidates=candidates, steps=[], config=config)
    second = cluster_candidates_by_step(candidates=candidates, steps=[], config=config)
    assert call_count == 1
    assert config.cache_path is not None
    assert config.cache_path.exists()
    assert first.summaries_by_step[0][0].name == "solver_plan"
    assert second.summaries_by_step[0][0].name == "solver_plan"
