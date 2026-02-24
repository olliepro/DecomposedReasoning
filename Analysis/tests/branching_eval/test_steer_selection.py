"""Tests for steer-branch candidate-id sanitization in branch executor."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from branching_eval.artifact_store import ArtifactStore
from branching_eval.branch_executor import BranchExecutor
from branching_eval.config_types import BranchingConfig, DecodingConfig
from branching_eval.tree_types import CandidatePoolRecord, CandidateRecord
from vllm_client import VllmClient


class MinimalClient:
    """Minimal client stub for executor construction in selection tests."""

    def tokenize(
        self,
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        _ = model, text, add_special_tokens
        return ()


def build_executor(tmp_path: Path, *, branch_fanout: int = 2) -> BranchExecutor:
    """Build an executor instance for steer-selection unit tests."""

    store = ArtifactStore(run_dir=tmp_path / "run", reuse_candidate_pools=True)
    return BranchExecutor(
        client=cast(VllmClient, MinimalClient()),
        prompt_text="prompt",
        model_name="fake",
        decoding=DecodingConfig(
            temperature=0.6,
            top_p=0.95,
            max_gen_toks=8,
            top_logprobs=5,
        ),
        branching=BranchingConfig(
            branch_prob=0.05,
            max_branch_points_per_rollout=2,
            num_candidates=100,
            branch_fanout=branch_fanout,
            max_clusters=4,
            candidate_span_tokens=3,
            max_steer_tokens=3,
            entropy_threshold=0.2,
        ),
        artifact_store=store,
        requested_selectors=("random",),
        active_selector="random",
        seed=7,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
        env_paths=(),
        cluster_cache_path=tmp_path / "cluster_cache.json",
        embedding_cache_path=tmp_path / "embedding_cache.json",
    )


def steer_pool(*, trigger_type: str) -> CandidatePoolRecord:
    """Build candidate-pool fixture with duplicate steer texts."""

    return CandidatePoolRecord(
        candidate_pool_id="pool",
        cache_key="key",
        branch_point_id="bp",
        node_id="node",
        trigger_type=trigger_type,
        entropy_value=None,
        candidates=(
            CandidateRecord(
                candidate_id=0,
                text="same",
                token_ids=(10,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer>",
            ),
            CandidateRecord(
                candidate_id=1,
                text="same",
                token_ids=(11,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer>",
            ),
            CandidateRecord(
                candidate_id=2,
                text="other",
                token_ids=(12,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer>",
            ),
        ),
    )


def test_steer_selection_deduplicates_and_backfills(tmp_path: Path) -> None:
    """Steer selection should dedupe by text and backfill without replacement."""

    executor = build_executor(tmp_path=tmp_path, branch_fanout=2)
    selected = executor._selected_ids_for_branch(
        pool=steer_pool(trigger_type="steer_boundary"),
        selected_ids=(0, 1),
    )
    assert selected == (0, 2)


def test_steer_selection_ignores_repeated_ids_without_replacement(tmp_path: Path) -> None:
    """Steer selection should never include repeated candidate IDs."""

    executor = build_executor(tmp_path=tmp_path, branch_fanout=2)
    selected = executor._selected_ids_for_branch(
        pool=steer_pool(trigger_type="steer_boundary"),
        selected_ids=(1, 1, 1),
    )
    assert selected == (1, 2)


def test_non_steer_selection_preserves_selector_output(tmp_path: Path) -> None:
    """Non-steer trigger should preserve selector ids unchanged."""

    executor = build_executor(tmp_path=tmp_path, branch_fanout=2)
    selected = executor._selected_ids_for_branch(
        pool=steer_pool(trigger_type="high_entropy"),
        selected_ids=(0, 0, 1),
    )
    assert selected == (0, 0, 1)
