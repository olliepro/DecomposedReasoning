"""Tests for branching execution, selectors, and candidate-pool reuse."""

from __future__ import annotations

import asyncio
from pathlib import Path
import time
from typing import cast

import pytest

from branching_eval.artifact_store import ArtifactStore
from branching_eval.branch_executor import BranchExecutor, DecodeOutcome, PathState
from branching_eval.config_types import BranchingConfig, DecodingConfig
from branching_eval.selector_runtime import (
    _cluster_across_ids,
    _diverse_embedding_ids,
    _random_ids,
    _within_cluster_ids,
)
from branching_eval.steer_decode_flow import continue_with_single_steer_candidate
from branching_eval.tree_types import (
    BranchTree,
    CandidatePoolRecord,
    CandidateRecord,
    TokenTrace,
)
from io_utils import read_jsonl
from vllm_client import GenerationChoice, ParsedToken, VllmClient


class FakeClient:
    """Deterministic fake completions client for branching tests."""

    def __init__(self) -> None:
        self.candidate_calls = 0
        self.decode_calls = 0

    def completions(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            prompt,
            prompt_token_ids,
            temperature,
            top_p,
            max_tokens,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
        )
        if n == 100:
            self.candidate_calls += 1
            return tuple(
                make_choice(index=index, text=f"cand_{index}", entropy_logprob=-0.4)
                for index in range(100)
            )
        if n > 1:
            return tuple(
                make_choice(index=index, text=f"base_{index}", entropy_logprob=-0.2)
                for index in range(n)
            )
        self.decode_calls += 1
        return (make_choice(index=0, text=" x", entropy_logprob=-0.05),)

    async def completions_async(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[GenerationChoice, ...]:
        """Async wrapper matching executor async completion interface."""

        return self.completions(
            model=model,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
            priority=priority,
            repetition_penalty=repetition_penalty,
        )

    def tokenize(
        self,
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        _ = model, add_special_tokens
        return tuple(range(1000, 1000 + len(text)))

    async def tokenize_async(
        self,
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        """Async wrapper matching executor tokenizer interface."""

        return self.tokenize(
            model=model,
            text=text,
            add_special_tokens=add_special_tokens,
        )


class AsyncDecodeClient(FakeClient):
    """Fake client exposing async completions with deterministic delay."""

    def __init__(self) -> None:
        super().__init__()
        self.request_start_times: list[float] = []

    async def completions_async(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            prompt,
            prompt_token_ids,
            temperature,
            top_p,
            max_tokens,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
        )
        self.request_start_times.append(time.perf_counter())
        await asyncio.sleep(0.05)
        if n != 1:
            return super().completions(
                model=model,
                prompt=prompt,
                prompt_token_ids=prompt_token_ids,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                n=n,
                seed=seed,
                stop=stop,
                top_logprobs=top_logprobs,
                priority=priority,
                repetition_penalty=repetition_penalty,
            )
        return (
            GenerationChoice(
                index=0,
                text="done",
                finish_reason="stop",
                stop_reason=None,
                tokens=(
                    ParsedToken(
                        token="done",
                        logprob=-0.1,
                        top_entries=(("x", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 3),
                token_ids=(42,),
            ),
        )


def make_choice(*, index: int, text: str, entropy_logprob: float) -> GenerationChoice:
    """Build one deterministic generation choice."""

    parsed_token = ParsedToken(
        token=text,
        logprob=entropy_logprob,
        top_entries=(("a", -0.2), ("b", -1.0)),
    )
    return GenerationChoice(
        index=index,
        text=text,
        finish_reason="length",
        stop_reason=None,
        tokens=(parsed_token,),
        prompt_token_ids=(1, 2, 3),
        token_ids=(10 + index,),
    )


def _path_state(*, node_id: str, branch_points_used: int = 0) -> PathState:
    """Build one minimal path state row for scheduler tests."""

    return PathState(
        node_id=node_id,
        assistant_prefix="",
        prompt_token_ids=None,
        token_ids=(),
        token_traces=(),
        branch_points_used=branch_points_used,
    )


def _decode_outcome(
    *,
    event_type: str,
    assistant_prefix: str,
    token_ids: tuple[int, ...],
    generated_tokens: int,
    stop_reason: str,
    trigger_type: str | None = None,
) -> DecodeOutcome:
    """Build one decode outcome row with default empty metrics fields."""

    return DecodeOutcome(
        event_type=event_type,
        trigger_type=trigger_type,
        entropy_value=None,
        assistant_prefix=assistant_prefix,
        prompt_token_ids=None,
        token_ids=token_ids,
        token_traces=(),
        generated_tokens=generated_tokens,
        stop_reason=stop_reason,
    )


def build_executor(
    tmp_path: Path,
    *,
    branch_prob: float = 1.0,
    trigger_steer_enabled: bool = False,
    trigger_entropy_enabled: bool = True,
) -> BranchExecutor:
    """Build branch executor with fake client and random-only selector."""

    store = ArtifactStore(run_dir=tmp_path / "run", reuse_candidate_pools=True)
    return BranchExecutor(
        client=cast(VllmClient, FakeClient()),
        prompt_text="Solve this.",
        model_name="fake",
        decoding=DecodingConfig(
            temperature=0.6, top_p=0.95, max_gen_toks=8, top_logprobs=5
        ),
        branching=BranchingConfig(
            branch_prob=branch_prob,
            max_branch_points_per_rollout=2,
            num_candidates=100,
            branch_fanout=2,
            max_clusters=4,
            candidate_span_tokens=3,
            max_steer_tokens=3,
            entropy_threshold=0.2,
        ),
        artifact_store=store,
        requested_selectors=("random",),
        active_selector="random",
        seed=7,
        trigger_steer_enabled=trigger_steer_enabled,
        trigger_entropy_enabled=trigger_entropy_enabled,
        env_paths=(),
        cluster_cache_path=tmp_path / "cluster_cache.json",
        embedding_cache_path=tmp_path / "embedding_cache.json",
    )


def test_shared_pool_records_all_requested_selectors(
    tmp_path: Path, monkeypatch
) -> None:
    """Each branch point should persist selection outcomes for all requested modes."""

    executor = build_executor(tmp_path=tmp_path)
    executor.requested_selectors = (
        "cluster_across",
        "embed_diverse",
        "within_cluster",
        "random",
    )

    def fake_select_all_modes(**_: object):
        from branching_eval.selector_types import SelectionOutcome

        return (
            SelectionOutcome(
                selector_mode="cluster_across", selected_candidate_ids=(0, 1)
            ),
            SelectionOutcome(
                selector_mode="embed_diverse", selected_candidate_ids=(2, 3)
            ),
            SelectionOutcome(
                selector_mode="within_cluster", selected_candidate_ids=(4, 5)
            ),
            SelectionOutcome(selector_mode="random", selected_candidate_ids=(6, 7)),
        )

    monkeypatch.setattr(
        "branching_eval.branch_executor.select_candidates_all_modes",
        fake_select_all_modes,
    )
    tree = executor.run_branching_rollouts(doc_id=9, task_name="aime24", model_id="m")
    assert tree.branch_points
    assert all(len(branch_point.selections) == 4 for branch_point in tree.branch_points)


def test_selector_event_logs_cluster_details(tmp_path: Path, monkeypatch) -> None:
    """Selector event should include cluster assignment and grouped summaries."""

    executor = build_executor(tmp_path=tmp_path)
    executor.requested_selectors = ("cluster_across", "random")
    cluster_assignment = {
        index: ("alpha" if index % 2 == 0 else "beta") for index in range(8)
    }

    def fake_select_all_modes(**_: object):
        from branching_eval.selector_types import SelectionOutcome

        return (
            SelectionOutcome(
                selector_mode="cluster_across",
                selected_candidate_ids=(0, 3, 4),
                cluster_by_candidate_id=cluster_assignment,
            ),
            SelectionOutcome(selector_mode="random", selected_candidate_ids=(6, 7)),
        )

    monkeypatch.setattr(
        "branching_eval.branch_executor.select_candidates_all_modes",
        fake_select_all_modes,
    )
    _ = executor.run_branching_rollouts(doc_id=10, task_name="aime24", model_id="m")
    rows = read_jsonl(path=tmp_path / "run" / "tree_events.jsonl")
    selector_rows = [row for row in rows if row.get("event_type") == "selector_applied"]
    assert selector_rows
    payload = dict(selector_rows[0].get("payload", {}))
    assignment_by_mode = dict(payload.get("cluster_assignments_by_mode", {}))
    grouped_by_mode = dict(payload.get("cluster_groups_by_mode", {}))
    assert "cluster_across" in assignment_by_mode
    assert "cluster_across" in grouped_by_mode
    assert assignment_by_mode["cluster_across"][0] == {
        "candidate_id": 0,
        "cluster_name": "alpha",
    }
    first_cluster = grouped_by_mode["cluster_across"][0]
    assert isinstance(first_cluster.get("candidate_ids"), list)
    assert isinstance(first_cluster.get("selected_candidate_ids"), list)
    assert "candidate_count" in first_cluster
    assert "selected_candidate_count" in first_cluster


def test_branching_leaf_bound_is_k_squared(tmp_path: Path) -> None:
    """Branching should produce at most `K^2` leaves for two branch points."""

    executor = build_executor(tmp_path=tmp_path)
    tree = executor.run_branching_rollouts(doc_id=0, task_name="aime24", model_id="m")
    assert len(tree.leaves) <= 4
    assert all(node.branch_points_used <= 2 for node in tree.nodes.values())


def test_async_frontier_decode_runs_states_concurrently(tmp_path: Path) -> None:
    """Async decode should process sibling frontier states in parallel."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=False,
        trigger_entropy_enabled=False,
    )
    async_client = AsyncDecodeClient()
    executor.client = cast(VllmClient, async_client)
    tree = _minimal_tree()
    frontier = [
        PathState(
            node_id="node_a",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
        PathState(
            node_id="node_b",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    ]
    next_frontier = asyncio.run(
        executor._decode_frontier_batch_async(
            tree=tree,
            frontier=frontier,
            doc_id=0,
            leaf_limit=16,
        )
    )
    assert not next_frontier
    assert len(tree.leaves) == 2
    assert len(async_client.request_start_times) == 2
    start_gap = max(async_client.request_start_times) - min(
        async_client.request_start_times
    )
    assert start_gap < 0.03


def test_streaming_scheduler_starts_child_before_slow_sibling_finishes(
    tmp_path: Path, monkeypatch
) -> None:
    """Streaming scheduler should avoid waiting for whole-depth completion."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=False,
        trigger_entropy_enabled=False,
    )
    tree = _minimal_tree()
    frontier = [
        PathState(
            node_id="node_a",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
        PathState(
            node_id="node_b",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    ]
    start_times: dict[str, float] = {}
    finish_times: dict[str, float] = {}

    async def fake_decode_until_event(
        *, tree: BranchTree, state: PathState
    ) -> DecodeOutcome:
        _ = tree
        start_times[state.node_id] = time.monotonic()
        if state.node_id == "node_a":
            await asyncio.sleep(0.01)
            finish_times[state.node_id] = time.monotonic()
            return DecodeOutcome(
                event_type="trigger",
                trigger_type="steer_boundary",
                entropy_value=None,
                assistant_prefix="a-prefix",
                prompt_token_ids=None,
                token_ids=(),
                token_traces=(),
                generated_tokens=1,
                stop_reason="",
            )
        if state.node_id == "node_b":
            await asyncio.sleep(0.20)
            finish_times[state.node_id] = time.monotonic()
            return DecodeOutcome(
                event_type="terminated",
                trigger_type=None,
                entropy_value=None,
                assistant_prefix="b-prefix",
                prompt_token_ids=None,
                token_ids=(),
                token_traces=(),
                generated_tokens=1,
                stop_reason="model_finished",
            )
        assert state.node_id == "node_a_child"
        await asyncio.sleep(0.01)
        finish_times[state.node_id] = time.monotonic()
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix="child-prefix",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            generated_tokens=1,
            stop_reason="model_finished",
        )

    async def fake_expand_branchable(
        *,
        tree: BranchTree,
        branchable: list[tuple[PathState, DecodeOutcome]],
        doc_id: int,
    ) -> list[PathState]:
        _ = tree, doc_id
        assert len(branchable) == 1
        state, _outcome = branchable[0]
        if state.node_id != "node_a":
            return []
        return [
            PathState(
                node_id="node_a_child",
                assistant_prefix="",
                prompt_token_ids=None,
                token_ids=(),
                token_traces=(),
                branch_points_used=1,
            )
        ]

    monkeypatch.setattr(executor, "_decode_until_event_async", fake_decode_until_event)
    monkeypatch.setattr(
        executor, "_expand_branchable_states_async", fake_expand_branchable
    )
    asyncio.run(
        executor._decode_frontier_streaming_async(
            tree=tree,
            frontier=frontier,
            doc_id=0,
            leaf_limit=16,
        )
    )
    assert "node_b" in finish_times
    assert "node_a_child" in start_times
    assert start_times["node_a_child"] < finish_times["node_b"]
    assert len(tree.leaves) == 2


def test_streaming_scheduler_resumes_maxed_state_without_blocking_siblings(
    tmp_path: Path, monkeypatch
) -> None:
    """Maxed-path continuation should start before a slow sibling terminates."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=False,
        trigger_entropy_enabled=False,
    )
    tree = _minimal_tree()
    frontier = [
        _path_state(node_id="node_maxed", branch_points_used=2),
        _path_state(node_id="node_slow", branch_points_used=0),
    ]
    start_times: dict[str, float] = {}
    finish_times: dict[str, float] = {}

    async def fake_decode_until_event(
        *,
        tree: BranchTree,
        state: PathState,
        branching_enabled: bool = True,
        steer_normalization_enabled: bool | None = None,
    ) -> DecodeOutcome:
        _ = tree
        if state.node_id == "node_maxed" and branching_enabled:
            start_times["maxed_trigger"] = time.monotonic()
            await asyncio.sleep(0.01)
            finish_times["maxed_trigger"] = time.monotonic()
            return _decode_outcome(
                event_type="trigger",
                trigger_type="steer_boundary",
                assistant_prefix="maxed-prefix",
                token_ids=(11,),
                generated_tokens=1,
                stop_reason="",
            )
        if state.node_id == "node_maxed" and not branching_enabled:
            assert steer_normalization_enabled is True
            start_times["maxed_resume"] = time.monotonic()
            await asyncio.sleep(0.01)
            finish_times["maxed_resume"] = time.monotonic()
            return _decode_outcome(
                event_type="terminated",
                assistant_prefix="maxed-done",
                token_ids=(11, 12),
                generated_tokens=2,
                stop_reason="model_finished",
            )
        assert state.node_id == "node_slow"
        start_times["slow"] = time.monotonic()
        await asyncio.sleep(0.20)
        finish_times["slow"] = time.monotonic()
        return _decode_outcome(
            event_type="terminated",
            assistant_prefix="slow-done",
            token_ids=(21,),
            generated_tokens=1,
            stop_reason="model_finished",
        )

    monkeypatch.setattr(executor, "_decode_until_event_async", fake_decode_until_event)
    asyncio.run(
        executor._decode_frontier_streaming_async(
            tree=tree,
            frontier=frontier,
            doc_id=0,
            leaf_limit=16,
        )
    )
    assert "maxed_resume" in start_times
    assert "slow" in finish_times
    assert start_times["maxed_resume"] < finish_times["slow"]
    assert len(tree.leaves) == 2
    rows = read_jsonl(path=tmp_path / "run" / "tree_events.jsonl")
    assert any(
        row.get("event_type") == "trigger_skipped_max_branch_points" for row in rows
    )


def test_streaming_scheduler_does_not_block_maxed_resume_on_slow_expansion(
    tmp_path: Path, monkeypatch
) -> None:
    """Slow trigger expansion should not delay maxed-state continuation decode."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=False,
        trigger_entropy_enabled=False,
    )
    tree = _minimal_tree()
    frontier = [
        _path_state(node_id="node_trigger", branch_points_used=0),
        _path_state(node_id="node_maxed", branch_points_used=2),
    ]
    timings: dict[str, float] = {}

    async def fake_decode_until_event(
        *,
        tree: BranchTree,
        state: PathState,
        branching_enabled: bool = True,
        steer_normalization_enabled: bool | None = None,
    ) -> DecodeOutcome:
        _ = tree
        if state.node_id == "node_trigger":
            await asyncio.sleep(0.01)
            return _decode_outcome(
                event_type="trigger",
                trigger_type="steer_boundary",
                assistant_prefix="trigger-prefix",
                token_ids=(31,),
                generated_tokens=1,
                stop_reason="",
            )
        if state.node_id == "node_maxed" and branching_enabled:
            await asyncio.sleep(0.02)
            return _decode_outcome(
                event_type="trigger",
                trigger_type="steer_boundary",
                assistant_prefix="maxed-prefix",
                token_ids=(11,),
                generated_tokens=1,
                stop_reason="",
            )
        assert state.node_id == "node_maxed"
        assert branching_enabled is False
        assert steer_normalization_enabled is True
        timings["maxed_resume_start"] = time.monotonic()
        await asyncio.sleep(0.01)
        return _decode_outcome(
            event_type="terminated",
            assistant_prefix="maxed-done",
            token_ids=(11, 12),
            generated_tokens=2,
            stop_reason="model_finished",
        )

    async def fake_expand_branchable(
        *,
        tree: BranchTree,
        branchable: list[tuple[PathState, DecodeOutcome]],
        doc_id: int,
    ) -> list[PathState]:
        _ = tree, doc_id, branchable
        await asyncio.sleep(0.20)
        timings["expansion_finish"] = time.monotonic()
        return []

    monkeypatch.setattr(executor, "_decode_until_event_async", fake_decode_until_event)
    monkeypatch.setattr(
        executor, "_expand_branchable_states_async", fake_expand_branchable
    )
    asyncio.run(
        executor._decode_frontier_streaming_async(
            tree=tree,
            frontier=frontier,
            doc_id=0,
            leaf_limit=16,
        )
    )
    assert "maxed_resume_start" in timings
    assert "expansion_finish" in timings
    assert timings["maxed_resume_start"] < timings["expansion_finish"]
    assert len(tree.leaves) == 1


def test_baseline_rollout_count_is_n(tmp_path: Path) -> None:
    """Baseline mode should return exactly N rollouts."""

    executor = build_executor(tmp_path=tmp_path)
    leaves = executor.run_standard_rollouts(rollout_count=16)
    assert len(leaves) == 16


def test_branching_pool_reuse_avoids_regeneration(tmp_path: Path) -> None:
    """Candidate pool should be reused from cache on repeated resolution."""

    executor = build_executor(tmp_path=tmp_path)
    assert isinstance(executor.client, FakeClient)
    _ = executor.run_branching_rollouts(doc_id=1, task_name="aime24", model_id="m")
    first_calls = executor.client.candidate_calls
    _ = executor.run_branching_rollouts(doc_id=1, task_name="aime24", model_id="m")
    assert executor.client.candidate_calls == first_calls


def test_branching_writes_incremental_tree_events_jsonl(tmp_path: Path) -> None:
    """Branching run should append live tree events to JSONL."""

    executor = build_executor(tmp_path=tmp_path)
    tree = executor.run_branching_rollouts(doc_id=3, task_name="aime24", model_id="m")
    events_path = tmp_path / "run" / "tree_events.jsonl"
    assert events_path.exists()
    rows = read_jsonl(path=events_path)
    assert rows
    event_types = {str(row.get("event_type", "")) for row in rows}
    assert "rollout_started" in event_types
    assert "node_created" in event_types
    assert "trigger_fired" in event_types
    assert "decode_chunk" in event_types
    assert "trigger_skipped_max_branch_points" in event_types
    assert "candidate_pool_resolved" in event_types
    assert "selector_applied" in event_types
    assert "edge_selected" in event_types
    assert "leaf_completed" in event_types
    assert "rollout_finished" in event_types
    assert all(int(row["doc_id"]) == tree.doc_id for row in rows)
    decode_rows = [row for row in rows if row.get("event_type") == "decode_chunk"]
    assert decode_rows
    assert isinstance(decode_rows[0]["payload"].get("chunk_text"), str)
    assert isinstance(decode_rows[0]["payload"].get("chunk_was_normalized"), bool)
    assert isinstance(decode_rows[0]["payload"].get("chunk_token_ids_source"), str)
    assert "chunk_text_raw" not in decode_rows[0]["payload"]
    assert "chunk_token_ids_raw" not in decode_rows[0]["payload"]
    pool_rows = [
        row for row in rows if row.get("event_type") == "candidate_pool_resolved"
    ]
    assert pool_rows
    candidates_payload = pool_rows[0]["payload"].get("candidates")
    assert isinstance(candidates_payload, list)
    assert candidates_payload
    first_candidate = candidates_payload[0]
    assert isinstance(first_candidate, dict)
    assert "text" in first_candidate


def test_max_branch_points_does_not_force_leaf_termination(tmp_path: Path) -> None:
    """Leaves should continue to normal termination after max branch points."""

    executor = build_executor(tmp_path=tmp_path)
    tree = executor.run_branching_rollouts(doc_id=4, task_name="aime24", model_id="m")
    assert tree.leaves
    assert all(leaf.stop_reason != "max_branch_points_reached" for leaf in tree.leaves)


def test_decode_chunk_event_uses_state_delta(tmp_path: Path) -> None:
    """Decode chunk event should log from before/after executor state."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    tree = BranchTree(
        doc_id=0,
        task_name="aime24",
        model_id="m",
        selector_mode="random",
        root_prompt="Solve this.",
    )
    executor._append_decode_chunk_event(
        tree=tree,
        node_id="node_0",
        raw_chunk_text="<steer",
        raw_chunk_token_ids=(42,),
        finish_reason="length",
        generated_tokens_before_chunk=10,
        generated_tokens_after_chunk=11,
        branching_enabled=True,
        prefix_before="<exec>plan\n",
        prefix_after="<exec>plan\n<steer></steer></exec>\n\n<steer>",
        prompt_token_ids_before=(9, 8),
        prompt_token_ids_after=(9, 8, 50, 51, 52, 53),
        token_ids_before=(1, 2, 3),
        token_ids_after=(1, 2, 3, 42, 43, 44),
    )
    rows = read_jsonl(path=tmp_path / "run" / "tree_events.jsonl")
    assert rows
    payload = rows[-1]["payload"]
    assert payload["chunk_text"] == "<steer></steer></exec>\n\n<steer>"
    assert payload["chunk_was_normalized"] is True
    assert payload["chunk_token_ids"] == [50, 51, 52, 53]
    assert payload["chunk_token_ids_source"] == "prompt_token_delta"
    assert "chunk_text_raw" not in payload
    assert "chunk_token_ids_raw" not in payload


def test_selector_cluster_across_respects_cluster_cap() -> None:
    """Cluster-across mode should sample one id across capped clusters."""

    pool = sample_pool(candidate_count=8)
    cluster_assignment = {
        0: "a",
        1: "a",
        2: "b",
        3: "b",
        4: "c",
        5: "c",
        6: "d",
        7: "d",
    }
    selected = _cluster_across_ids(
        pool=pool,
        cluster_assignment=cluster_assignment,
        max_count=4,
        max_clusters=3,
        rng=__import__("random").Random(3),
    )
    assert len(selected) == 3


def test_selector_diverse_embedding_returns_k_ids() -> None:
    """Embedding-diverse mode should return `K` ids when enough points exist."""

    embeddings = {
        0: (1.0, 0.0),
        1: (0.0, 1.0),
        2: (-1.0, 0.0),
        3: (0.0, -1.0),
    }
    selected = _diverse_embedding_ids(
        embedding_by_candidate=embeddings,
        max_count=3,
        rng=__import__("random").Random(2),
    )
    assert len(selected) == 3
    assert all(candidate_id in embeddings for candidate_id in selected)


def test_selector_within_cluster_prefers_non_other() -> None:
    """Within-cluster mode should choose from non-other cluster when feasible."""

    pool = sample_pool(candidate_count=6)
    cluster_assignment = {
        0: "other",
        1: "other",
        2: "main",
        3: "main",
        4: "main",
        5: "main",
    }
    selected = _within_cluster_ids(
        pool=pool,
        cluster_assignment=cluster_assignment,
        max_count=3,
        rng=__import__("random").Random(4),
    )
    assert len(selected) == 3
    assert all(cluster_assignment[candidate_id] == "main" for candidate_id in selected)


def test_selector_random_returns_unique_k_ids() -> None:
    """Random mode should return `K` unique ids sampled from the pool."""

    pool = sample_pool(candidate_count=10)
    selected = _random_ids(pool=pool, max_count=4, rng=__import__("random").Random(5))
    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert all(0 <= candidate_id < 10 for candidate_id in selected)


def test_entropy_span_pool_generation_uses_no_stop(tmp_path: Path, monkeypatch) -> None:
    """Entropy-trigger pool generation should use `stop=None` without steer normalization."""

    executor = build_executor(tmp_path=tmp_path)
    called = {}

    async def fake_generate_many(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        n: int,
        stop: tuple[str, ...] | None,
        temperature: float | None = None,
        **kwargs: object,
    ):
        called["assistant_prefix"] = assistant_prefix
        called["prompt_token_ids"] = prompt_token_ids
        called["max_tokens"] = max_tokens
        called["n"] = n
        called["stop"] = stop
        called["temperature"] = temperature
        return tuple(
            make_choice(index=index, text=f"c{index}", entropy_logprob=-0.4)
            for index in range(n)
        )

    monkeypatch.setattr(executor, "_generate_many_async", fake_generate_many)
    state = executor.run_branching_rollouts(doc_id=2, task_name="aime24", model_id="m")
    assert state.candidate_pools
    assert called["stop"] is None


def test_entropy_span_generation_preserves_partial_steer_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    """Entropy-trigger candidates should not normalize partial `<steer` prefixes."""

    executor = build_executor(tmp_path=tmp_path)
    captured_prefix: dict[str, str | float] = {}

    async def fake_generate_many(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        n: int,
        stop: tuple[str, ...] | None,
        temperature: float | None = None,
        **kwargs: object,
    ):
        captured_prefix["assistant_prefix"] = assistant_prefix
        captured_prefix["temperature"] = -1.0 if temperature is None else temperature
        _ = prompt_token_ids, max_tokens, stop
        return tuple(
            make_choice(index=index, text=f"c{index}", entropy_logprob=-0.4)
            for index in range(n)
        )

    monkeypatch.setattr(executor, "_generate_many_async", fake_generate_many)
    _ = executor._generate_candidate_pool(
        cache_key="cache_key",
        state=PathState(
            node_id="node_x",
            assistant_prefix="abc<steer",
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
        trigger_type="high_entropy",
        entropy_value=1.2,
        assistant_prefix="abc<steer",
        prompt_token_ids=(1, 2, 3),
    )
    assert captured_prefix["assistant_prefix"] == "abc<steer"
    assert captured_prefix["temperature"] == 1.0


def test_steer_pool_generation_applies_full_boundary_normalization(
    tmp_path: Path, monkeypatch
) -> None:
    """Steer-trigger pool generation should normalize to valid steer boundary."""

    executor = build_executor(tmp_path=tmp_path)
    captured_prefix: dict[str, str | float] = {}
    observed_prompt_token_ids: tuple[int, ...] | None = None

    async def fake_generate_many(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        n: int,
        stop: tuple[str, ...] | None,
        temperature: float | None = None,
        **kwargs: object,
    ):
        nonlocal observed_prompt_token_ids
        captured_prefix["assistant_prefix"] = assistant_prefix
        captured_prefix["stop"] = "" if stop is None else ",".join(stop)
        captured_prefix["temperature"] = -1.0 if temperature is None else temperature
        observed_prompt_token_ids = prompt_token_ids
        _ = max_tokens
        return tuple(
            make_choice(index=index, text=f"c{index}", entropy_logprob=-0.4)
            for index in range(n)
        )

    monkeypatch.setattr(executor, "_generate_many_async", fake_generate_many)
    _ = executor._generate_candidate_pool(
        cache_key="cache_key",
        state=PathState(
            node_id="node_s",
            assistant_prefix="<exec>abc<steer",
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
        trigger_type="steer_boundary",
        entropy_value=None,
        assistant_prefix="<exec>abc<steer",
        prompt_token_ids=(1, 2, 3),
    )
    assert captured_prefix["stop"] == "</steer"
    assert isinstance(captured_prefix["assistant_prefix"], str)
    assert captured_prefix["assistant_prefix"].endswith("</steer></exec>\n\n<steer>")
    assert captured_prefix["temperature"] == 1.0
    normalized_prefix = str(captured_prefix["assistant_prefix"])
    injected_suffix = normalized_prefix[len("<exec>abc<steer") :]
    expected_suffix_token_ids = tuple(range(1000, 1000 + len(injected_suffix)))
    assert observed_prompt_token_ids == (1, 2, 3) + expected_suffix_token_ids


def test_steer_prefix_normalization_asserts_detokenized_alignment(
    tmp_path: Path, monkeypatch
) -> None:
    """Prefix normalization should assert when suffix detokenization mismatches."""

    executor = build_executor(tmp_path=tmp_path)

    def fake_detokenize(*, model: str, token_ids: tuple[int, ...]) -> str:
        _ = model, token_ids
        return "mismatch"

    monkeypatch.setattr(executor.client, "detokenize", fake_detokenize, raising=False)
    with pytest.raises(
        AssertionError, match="normalize_steer_prefix_suffix: detokenized text mismatch"
    ):
        _ = executor._normalize_steer_prefix_prompt_ids(
            assistant_prefix="<exec>abc<steer",
            prompt_token_ids=(1, 2, 3),
        )


def test_steer_pool_candidates_preserve_vllm_output(
    tmp_path: Path, monkeypatch
) -> None:
    """Steer-trigger candidate artifacts should preserve raw vLLM text/tokens."""

    executor = build_executor(tmp_path=tmp_path)

    async def fake_generate_many(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        n: int,
        stop: tuple[str, ...] | None,
        temperature: float | None = None,
        **kwargs: object,
    ):
        _ = assistant_prefix, prompt_token_ids, max_tokens, n, stop, temperature
        return (
            GenerationChoice(
                index=0,
                text="abc</steer>",
                finish_reason="stop",
                stop_reason="</steer>",
                tokens=(
                    ParsedToken(
                        token="abc</steer>",
                        logprob=-0.1,
                        top_entries=(("x", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2),
                token_ids=(10, 11, 12),
            ),
        ) * 100

    monkeypatch.setattr(executor, "_generate_many_async", fake_generate_many)
    pool = executor._generate_candidate_pool(
        cache_key="cache_key",
        state=PathState(
            node_id="node_s",
            assistant_prefix="<think><steer>",
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
        trigger_type="steer_boundary",
        entropy_value=None,
        assistant_prefix="<think><steer>",
        prompt_token_ids=(1, 2, 3),
    )
    assert pool.candidates[0].text == "abc</steer>"
    assert pool.candidates[0].token_ids == (10, 11, 12)
    assert len(pool.candidates[0].tokens) == 1


def test_steer_pool_candidates_fail_when_text_after_close(
    tmp_path: Path, monkeypatch
) -> None:
    """Steer candidate generation should fail on text after first `</steer>`."""

    executor = build_executor(tmp_path=tmp_path)

    async def fake_generate_many(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        n: int,
        stop: tuple[str, ...] | None,
        temperature: float | None = None,
        **kwargs: object,
    ):
        _ = assistant_prefix, prompt_token_ids, max_tokens, n, stop, temperature
        return (
            GenerationChoice(
                index=0,
                text="abc</steer>tail",
                finish_reason="stop",
                stop_reason="</steer>",
                tokens=(
                    ParsedToken(
                        token="abc</steer>tail",
                        logprob=-0.1,
                        top_entries=(("x", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2),
                token_ids=(10, 11, 12),
            ),
        ) * 100

    monkeypatch.setattr(executor, "_generate_many_async", fake_generate_many)
    with pytest.raises(AssertionError, match="unexpected text after first </steer>"):
        _ = executor._generate_candidate_pool(
            cache_key="cache_key",
            state=PathState(
                node_id="node_s",
                assistant_prefix="<think><steer>",
                prompt_token_ids=(1, 2, 3),
                token_ids=(),
                token_traces=(),
                branch_points_used=0,
            ),
            trigger_type="steer_boundary",
            entropy_value=None,
            assistant_prefix="<think><steer>",
            prompt_token_ids=(1, 2, 3),
        )


def test_steer_decode_uses_rollout_stop_sequence(tmp_path: Path, monkeypatch) -> None:
    """Steer-enabled rollout decoding should request legacy stop marker."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    observed_stop: dict[str, tuple[str, ...] | None] = {}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, n
        observed_stop["stop"] = stop
        return make_choice(index=0, text="base", entropy_logprob=-0.2)

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    tree = _minimal_tree()
    _ = executor._decode_until_event(
        tree=tree,
        state=PathState(
            node_id="node_root",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert observed_stop["stop"] == ("<steer",)


def test_steer_stop_reason_triggers_branch_event(tmp_path: Path, monkeypatch) -> None:
    """Explicit steer stop reason should be interpreted as steer trigger."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n
        return GenerationChoice(
            index=0,
            text="plain text",
            finish_reason="stop",
            stop_reason="<steer",
            tokens=(
                ParsedToken(
                    token="plain text",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(42,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert outcome.event_type == "trigger"
    assert outcome.trigger_type == "steer_boundary"


def test_steer_think_close_natural_finish_terminates(
    tmp_path: Path, monkeypatch
) -> None:
    """Steer decode should terminate with `think_end` on natural `</think>` close."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n
        return GenerationChoice(
            index=0,
            text="</think>",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="</think>",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2),
            token_ids=(42,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "think_end"


def test_steer_think_partial_continues_without_stop(
    tmp_path: Path, monkeypatch
) -> None:
    """After partial `</think`, steer path should continue with `stop=None`."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    observed_stops: list[tuple[str, ...] | None] = []
    call_counter = {"count": 0}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, n
        observed_stops.append(stop)
        call_counter["count"] += 1
        if call_counter["count"] == 1:
            return GenerationChoice(
                index=0,
                text="</th",
                finish_reason="length",
                stop_reason=None,
                tokens=(
                    ParsedToken(
                        token="</th",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2),
                token_ids=(42,),
            )
        return GenerationChoice(
            index=0,
            text="ink done",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="ink done",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 42),
            token_ids=(43,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "think_end"
    assert observed_stops[0] == ("<steer",)
    assert observed_stops[1] is None


def test_steer_length_boundary_forces_branch_trigger(
    tmp_path: Path, monkeypatch
) -> None:
    """Steer length finish should force a steer-boundary trigger."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n
        return GenerationChoice(
            index=0,
            text="plain",
            finish_reason="length",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="plain",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2),
            token_ids=(50,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert outcome.event_type == "trigger"
    assert outcome.trigger_type == "steer_boundary"
    assert outcome.assistant_prefix.endswith("<steer>")


def test_steer_stop_respects_branch_prob_and_continues(
    tmp_path: Path, monkeypatch
) -> None:
    """With `branch_prob=0`, explicit steer stop should not branch and should continue."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    calls = {"generate": 0, "continued": 0}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n
        calls["generate"] += 1
        if calls["generate"] == 1:
            return GenerationChoice(
                index=0,
                text="...<steer",
                finish_reason="stop",
                stop_reason="<steer",
                tokens=(
                    ParsedToken(
                        token="...<steer",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2),
                token_ids=(42,),
            )
        return GenerationChoice(
            index=0,
            text="done",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="done",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 42),
            token_ids=(43,),
        )

    async def fake_continue_with_single(
        *,
        executor: BranchExecutor,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        token_ids: tuple[int, ...],
        token_traces: tuple[TokenTrace, ...],
        generated_tokens: int,
        request_stream_id: str,
    ) -> DecodeOutcome:
        _ = (
            executor,
            assistant_prefix,
            prompt_token_ids,
            token_ids,
            token_traces,
            generated_tokens,
            request_stream_id,
        )
        calls["continued"] += 1
        return DecodeOutcome(
            event_type="continued",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix="seed</steer>\n",
            prompt_token_ids=(1, 2, 42),
            token_ids=(42,),
            token_traces=(),
            generated_tokens=1,
            stop_reason="",
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    monkeypatch.setattr(
        "branching_eval.branch_executor.continue_with_single_steer_candidate_async",
        fake_continue_with_single,
    )
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert calls["continued"] == 1
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "model_finished"
    events = read_jsonl(path=tmp_path / "run" / "tree_events.jsonl")
    steer_events = [
        row for row in events if row.get("event_type") == "steer_block_generated"
    ]
    assert steer_events
    payload = dict(steer_events[-1].get("payload", {}))
    assert payload.get("source") == "explicit_stop_nonbranch"


def test_maxed_path_keeps_steer_normalization_without_branching(
    tmp_path: Path, monkeypatch
) -> None:
    """Branch-disabled decode should preserve steer normalization and continue."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=1.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    observed_stops: list[tuple[str, ...] | None] = []
    calls = {"continued": 0}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, n
        observed_stops.append(stop)
        if len(observed_stops) == 1:
            return GenerationChoice(
                index=0,
                text="...<steer",
                finish_reason="stop",
                stop_reason="<steer",
                tokens=(
                    ParsedToken(
                        token="...<steer",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2),
                token_ids=(42,),
            )
        return GenerationChoice(
            index=0,
            text="done",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="done",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 42),
            token_ids=(43,),
        )

    async def fake_continue_with_single(
        *,
        executor: BranchExecutor,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        token_ids: tuple[int, ...],
        token_traces: tuple[TokenTrace, ...],
        generated_tokens: int,
        request_stream_id: str,
    ) -> DecodeOutcome:
        _ = (
            executor,
            assistant_prefix,
            prompt_token_ids,
            token_ids,
            token_traces,
            generated_tokens,
            request_stream_id,
        )
        calls["continued"] += 1
        return DecodeOutcome(
            event_type="continued",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix="seed</steer>\n<exec>\n",
            prompt_token_ids=(1, 2, 42),
            token_ids=(42,),
            token_traces=(),
            generated_tokens=1,
            stop_reason="",
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    monkeypatch.setattr(
        "branching_eval.branch_executor.continue_with_single_steer_candidate_async",
        fake_continue_with_single,
    )
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=2,
        ),
        branching_enabled=False,
        steer_normalization_enabled=True,
    )
    assert observed_stops[0] == ("<steer",)
    assert calls["continued"] == 1
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "model_finished"
    events = read_jsonl(path=tmp_path / "run" / "tree_events.jsonl")
    steer_events = [
        row for row in events if row.get("event_type") == "steer_block_generated"
    ]
    assert steer_events
    assert all(
        bool(row.get("payload", {}).get("branching_enabled")) is False
        for row in events
        if row.get("event_type") == "decode_chunk"
    )


def test_steer_exec_repeat_loop_terminates_branch(tmp_path: Path, monkeypatch) -> None:
    """Steer decode should terminate when exec blocks repeat 3 times at >=85% match."""

    monkeypatch.setattr(
        "branching_eval.branch_executor.REPEAT_TERMINATION_MIN_GENERATED_TOKENS",
        0,
    )
    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    repeated_exec_chunk = "Repeat this execution block exactly.</exec>\n\n"
    call_counts = {"generate": 0, "continued": 0}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
        call_counts["generate"] += 1
        assert (
            call_counts["generate"] <= 3
        ), "repeat-loop termination should stop decode"
        return GenerationChoice(
            index=0,
            text=repeated_exec_chunk,
            finish_reason="stop",
            stop_reason="<steer",
            tokens=(
                ParsedToken(
                    token=repeated_exec_chunk,
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(40 + call_counts["generate"],),
        )

    async def fake_continue_with_single(
        *,
        executor: BranchExecutor,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        token_ids: tuple[int, ...],
        token_traces: tuple[TokenTrace, ...],
        generated_tokens: int,
        request_stream_id: str,
    ) -> DecodeOutcome:
        _ = (
            executor,
            prompt_token_ids,
            token_ids,
            token_traces,
            generated_tokens,
            request_stream_id,
        )
        call_counts["continued"] += 1
        return DecodeOutcome(
            event_type="continued",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=f"{assistant_prefix}Repeat steering plan</steer>\n<exec>\n",
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="",
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    monkeypatch.setattr(
        "branching_eval.branch_executor.continue_with_single_steer_candidate_async",
        fake_continue_with_single,
    )
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="<think><steer>Repeat</steer>\n<exec>",
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "repeated_exec_block_loop"
    assert call_counts["generate"] == 3
    assert call_counts["continued"] == 2
    events = read_jsonl(path=tmp_path / "run" / "tree_events.jsonl")
    repeat_events = [
        row for row in events if row.get("event_type") == "exec_repeat_terminated"
    ]
    assert repeat_events
    payload = dict(repeat_events[-1].get("payload", {}))
    assert int(payload["repeated_exec_blocks"]) >= 3
    assert float(payload["similarity_threshold"]) == 0.85
    assert int(payload["similarity_lookback_window"]) == 3
    assert int(payload["termination_block_count"]) == 3


def test_steer_exec_repeat_loop_terminates_on_alternating_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    """Alternating repeat blocks should terminate with lookback-window matching."""

    monkeypatch.setattr(
        "branching_eval.branch_executor.REPEAT_TERMINATION_MIN_GENERATED_TOKENS",
        0,
    )
    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    alternating_exec_chunks = (
        "Alpha execution loop chunk.</exec>\n\n",
        "Beta execution loop chunk.</exec>\n\n",
    )
    call_counts = {"generate": 0, "continued": 0}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
        call_counts["generate"] += 1
        assert (
            call_counts["generate"] <= 4
        ), "alternating repeat-loop termination should stop decode"
        chunk_index = (call_counts["generate"] - 1) % len(alternating_exec_chunks)
        exec_chunk = alternating_exec_chunks[chunk_index]
        return GenerationChoice(
            index=0,
            text=exec_chunk,
            finish_reason="stop",
            stop_reason="<steer",
            tokens=(
                ParsedToken(
                    token=exec_chunk,
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(90 + call_counts["generate"],),
        )

    async def fake_continue_with_single(
        *,
        executor: BranchExecutor,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        token_ids: tuple[int, ...],
        token_traces: tuple[TokenTrace, ...],
        generated_tokens: int,
        request_stream_id: str,
    ) -> DecodeOutcome:
        _ = (
            executor,
            prompt_token_ids,
            token_ids,
            token_traces,
            generated_tokens,
            request_stream_id,
        )
        call_counts["continued"] += 1
        return DecodeOutcome(
            event_type="continued",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=f"{assistant_prefix}Alternate steering</steer>\n<exec>\n",
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="",
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    monkeypatch.setattr(
        "branching_eval.branch_executor.continue_with_single_steer_candidate_async",
        fake_continue_with_single,
    )
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="<think><steer>Alternating</steer>\n<exec>",
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "repeated_exec_block_loop"
    assert call_counts["generate"] == 3
    assert call_counts["continued"] == 2
    events = read_jsonl(path=tmp_path / "run" / "tree_events.jsonl")
    repeat_events = [
        row for row in events if row.get("event_type") == "exec_repeat_terminated"
    ]
    assert repeat_events
    payload = dict(repeat_events[-1].get("payload", {}))
    assert int(payload["repeated_exec_blocks"]) >= 3
    assert float(payload["last_similarity_ratio"]) >= 0.85
    assert int(payload["similarity_lookback_window"]) == 3
    assert int(payload["termination_block_count"]) == 3


def test_steer_block_repeat_loop_terminates_on_alternating_blocks(
    tmp_path: Path, monkeypatch
) -> None:
    """Alternating steer blocks should terminate at 4 matches with lookback 3."""

    monkeypatch.setattr(
        "branching_eval.branch_executor.REPEAT_TERMINATION_MIN_GENERATED_TOKENS",
        0,
    )
    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    exec_chunks = (
        "Red square orbit.</exec>\n\n",
        "Blue triangle pulse.</exec>\n\n",
        "Gold spiral vector.</exec>\n\n",
        "Green prism lattice.</exec>\n\n",
        "Silver wave tangent.</exec>\n\n",
    )
    steer_chunks = ("Alpha steering loop.", "Beta steering loop.")
    call_counts = {"generate": 0, "continued": 0}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
        call_counts["generate"] += 1
        assert call_counts["generate"] <= len(
            exec_chunks
        ), "steer repeat-loop termination should stop decode"
        exec_chunk = exec_chunks[call_counts["generate"] - 1]
        return GenerationChoice(
            index=0,
            text=exec_chunk,
            finish_reason="stop",
            stop_reason="<steer",
            tokens=(
                ParsedToken(
                    token=exec_chunk,
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(130 + call_counts["generate"],),
        )

    async def fake_continue_with_single(
        *,
        executor: BranchExecutor,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        token_ids: tuple[int, ...],
        token_traces: tuple[TokenTrace, ...],
        generated_tokens: int,
        request_stream_id: str,
    ) -> DecodeOutcome:
        _ = (
            executor,
            prompt_token_ids,
            token_ids,
            token_traces,
            generated_tokens,
            request_stream_id,
        )
        call_counts["continued"] += 1
        steer_chunk_index = (call_counts["continued"] - 1) % len(steer_chunks)
        steer_chunk = steer_chunks[steer_chunk_index]
        return DecodeOutcome(
            event_type="continued",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=f"{assistant_prefix}{steer_chunk}</steer>\n<exec>\n",
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="",
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    monkeypatch.setattr(
        "branching_eval.branch_executor.continue_with_single_steer_candidate_async",
        fake_continue_with_single,
    )
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="<think><steer>Seed steering</steer>\n<exec>",
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "repeated_steer_block_loop"
    assert call_counts["generate"] == len(exec_chunks)
    assert call_counts["continued"] == len(exec_chunks)
    events = read_jsonl(path=tmp_path / "run" / "tree_events.jsonl")
    repeat_events = [
        row for row in events if row.get("event_type") == "steer_repeat_terminated"
    ]
    assert repeat_events
    payload = dict(repeat_events[-1].get("payload", {}))
    assert int(payload["repeated_steer_blocks"]) >= 4
    assert float(payload["similarity_threshold"]) == 0.85
    assert int(payload["similarity_lookback_window"]) == 3
    assert int(payload["termination_block_count"]) == 4
    assert float(payload["last_similarity_ratio"]) >= 0.85


def test_single_steer_continuation_keeps_vllm_tokens_and_appends_suffix(
    tmp_path: Path, monkeypatch
) -> None:
    """Single steer continuation should append injected suffix tokens only."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )

    def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n
        return GenerationChoice(
            index=0,
            text="abc</steer>",
            finish_reason="stop",
            stop_reason="</steer>",
            tokens=(
                ParsedToken(
                    token="abc</steer>",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(10, 11, 12),
        )

    monkeypatch.setattr(executor, "_generate_choice", fake_generate_choice)
    outcome = continue_with_single_steer_candidate(
        executor=executor,
        assistant_prefix="seed</exec>\n\n<steer>",
        prompt_token_ids=(1, 2, 3),
        token_ids=(),
        token_traces=(),
        generated_tokens=0,
        request_stream_id="decode:node_root",
    )
    assert outcome.event_type == "continued"
    assert outcome.assistant_prefix.endswith("abc</steer>\n<exec>\n")
    assert outcome.generated_tokens == 3
    assert outcome.token_ids[:3] == (10, 11, 12)
    assert outcome.token_ids[3:] == tuple(range(1000, 1008))
    assert outcome.prompt_token_ids is not None
    assert outcome.prompt_token_ids[:6] == (1, 2, 3, 10, 11, 12)
    assert outcome.prompt_token_ids[6:] == tuple(range(1000, 1008))


def test_decode_respects_path_level_max_gen_toks(tmp_path: Path, monkeypatch) -> None:
    """Decode should terminate immediately when path token budget is already exhausted."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=False,
        trigger_entropy_enabled=False,
    )
    max_budget = int(executor.decoding.max_gen_toks)
    call_count = {"decode_calls": 0}

    async def fail_generate_choice_async(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
        call_count["decode_calls"] += 1
        raise AssertionError("decode call should not occur when budget is exhausted")

    monkeypatch.setattr(executor, "_generate_choice_async", fail_generate_choice_async)
    state = PathState(
        node_id="node_root",
        assistant_prefix="prefix",
        prompt_token_ids=tuple(range(max_budget)),
        token_ids=tuple(range(max_budget)),
        token_traces=(),
        branch_points_used=0,
    )
    outcome = executor._decode_until_event(tree=_minimal_tree(), state=state)
    assert call_count["decode_calls"] == 0
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "max_gen_toks_reached"
    assert outcome.generated_tokens == max_budget
    assert outcome.token_ids == tuple(range(max_budget))


def test_single_steer_continuation_fails_when_text_after_close(
    tmp_path: Path, monkeypatch
) -> None:
    """Single steer continuation should fail on text after first `</steer>`."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )

    def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n
        return GenerationChoice(
            index=0,
            text="abc</steer>tail",
            finish_reason="stop",
            stop_reason="</steer>",
            tokens=(
                ParsedToken(
                    token="abc</steer>tail",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(10, 11, 12),
        )

    monkeypatch.setattr(executor, "_generate_choice", fake_generate_choice)
    with pytest.raises(AssertionError, match="unexpected text after first </steer>"):
        _ = continue_with_single_steer_candidate(
            executor=executor,
            assistant_prefix="seed</exec>\n\n<steer>",
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
            token_traces=(),
            generated_tokens=0,
            request_stream_id="decode:node_root",
        )


def test_steer_selected_child_is_closed_before_continuation(tmp_path: Path) -> None:
    """Steer-selected children should append legacy close-tag suffix when needed."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    tree = _minimal_tree()
    pool = CandidatePoolRecord(
        candidate_pool_id="pool_s",
        cache_key="key_s",
        branch_point_id="bp_s",
        node_id="node_root",
        trigger_type="steer_boundary",
        entropy_value=None,
        candidates=(
            CandidateRecord(
                candidate_id=0,
                text="unfinished",
                token_ids=(11, 12),
                tokens=(),
                finish_reason="length",
                stop_reason=None,
            ),
        ),
    )
    outcome = DecodeOutcome(
        event_type="trigger",
        trigger_type="steer_boundary",
        entropy_value=None,
        assistant_prefix="<think><steer>",
        prompt_token_ids=(1, 2, 3),
        token_ids=(7, 8),
        token_traces=(),
        generated_tokens=2,
        stop_reason="",
    )
    children = asyncio.run(
        executor._expand_children_async(
            tree=tree,
            parent_state=PathState(
                node_id="node_root",
                assistant_prefix="<think><steer>",
                prompt_token_ids=(1, 2, 3),
                token_ids=(7, 8),
                token_traces=(),
                branch_points_used=0,
            ),
            outcome=outcome,
            pool=pool,
            selected_ids=(0,),
        )
    )
    assert len(children) == 1
    assert children[0].assistant_prefix.endswith("unfinished</steer>\n<exec>\n")
    assert children[0].prompt_token_ids is not None
    assert children[0].prompt_token_ids[:5] == (1, 2, 3, 11, 12)
    assert children[0].prompt_token_ids[5:] == tuple(range(1000, 1016))
    events = read_jsonl(path=tmp_path / "run" / "tree_events.jsonl")
    edge_events = [row for row in events if row.get("event_type") == "edge_selected"]
    assert edge_events
    edge_payload = dict(edge_events[-1].get("payload", {}))
    assert edge_payload.get("candidate_text_normalized", "").endswith(
        "unfinished</steer>\n<exec>\n"
    )


def test_selected_child_normalization_asserts_detokenized_alignment(
    tmp_path: Path, monkeypatch
) -> None:
    """Selected steer child normalization should fail on text/token mismatch."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )

    def fake_detokenize(*, model: str, token_ids: tuple[int, ...]) -> str:
        _ = model, token_ids
        return "bad"

    monkeypatch.setattr(executor.client, "detokenize", fake_detokenize, raising=False)
    with pytest.raises(
        AssertionError, match="normalized_child_candidate: detokenized text mismatch"
    ):
        _ = executor._normalized_child_candidate(
            trigger_type="steer_boundary",
            candidate=CandidateRecord(
                candidate_id=0,
                text="unfinished",
                token_ids=(11, 12),
                tokens=(),
                finish_reason="length",
                stop_reason=None,
            ),
        )


def sample_pool(*, candidate_count: int) -> CandidatePoolRecord:
    """Build a candidate pool fixture with sequential ids."""

    candidates = tuple(
        CandidateRecord(
            candidate_id=index,
            text=f"cand_{index}",
            token_ids=(index,),
            tokens=(
                TokenTrace(
                    token_index=0,
                    token_id=index,
                    token_text=f"cand_{index}",
                    logprob=-0.1,
                    probability=0.9,
                    entropy=0.2,
                ),
            ),
            finish_reason="stop",
            stop_reason=None,
        )
        for index in range(candidate_count)
    )
    return CandidatePoolRecord(
        candidate_pool_id="pool",
        cache_key="key",
        branch_point_id="bp",
        node_id="node",
        trigger_type="high_entropy",
        entropy_value=1.0,
        candidates=candidates,
    )


def _minimal_tree() -> BranchTree:
    tree = BranchTree(
        doc_id=0,
        task_name="aime24",
        model_id="m",
        selector_mode="random",
        root_prompt="p",
    )
    return tree
