"""Tests for canonical event-source logging, prefix invariants, and resume replay."""

from __future__ import annotations

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Any, cast

from branching_eval.artifact_store import ArtifactStore
from branching_eval.branch_executor import BranchExecutor
from branching_eval.config_types import BranchingConfig, DecodingConfig
from branching_eval.event_types import EventContext
from branching_eval.lm_eval_adapter import DocRecord
from branching_eval.metrics_types import (
    BreakpointVariance,
    DocDiagnostics,
    LengthSummary,
)
from branching_eval.run_matrix import (
    build_doc_execution_plans,
    recompute_outputs_from_events,
)
from vllm_client import GenerationChoice, ParsedToken, VllmClient
import pytest


class PrefixClient:
    """Minimal async client used to test request-stream prefix invariants."""

    def __init__(self) -> None:
        self.calls = 0
        self.priorities: list[int | None] = []

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
            n,
            seed,
            stop,
            top_logprobs,
            repetition_penalty,
        )
        self.priorities.append(priority)
        self.calls += 1
        output_token_id = 100 + self.calls
        return (
            GenerationChoice(
                index=0,
                text=f"tok_{self.calls}",
                finish_reason="length",
                stop_reason=None,
                tokens=(
                    ParsedToken(
                        token=f"tok_{self.calls}",
                        logprob=-0.2,
                        top_entries=(("alt", -0.3),),
                    ),
                ),
                prompt_token_ids=prompt_token_ids,
                token_ids=(output_token_id,),
            ),
        )

    async def tokenize_async(
        self,
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        _ = model, add_special_tokens
        return tuple(ord(char) for char in text)


def build_executor(
    *, tmp_path: Path, enable_request_priorities: bool = False
) -> BranchExecutor:
    """Build executor with lightweight async client for prefix tests."""

    store = ArtifactStore(run_dir=tmp_path / "run", reuse_candidate_pools=True)
    executor = BranchExecutor(
        client=cast(VllmClient, PrefixClient()),
        prompt_text="Solve this.",
        model_name="fake-model",
        decoding=DecodingConfig(
            temperature=0.6,
            top_p=0.95,
            max_gen_toks=16,
            top_logprobs=5,
        ),
        branching=BranchingConfig(
            branch_prob=1.0,
            max_branch_points_per_rollout=2,
            num_candidates=4,
            branch_fanout=2,
            max_clusters=2,
            candidate_span_tokens=3,
            max_steer_tokens=3,
            entropy_threshold=0.2,
        ),
        artifact_store=store,
        requested_selectors=("random",),
        active_selector="random",
        seed=7,
        trigger_steer_enabled=False,
        trigger_entropy_enabled=False,
        env_paths=(),
        cluster_cache_path=tmp_path / "cluster_cache.json",
        embedding_cache_path=tmp_path / "embedding_cache.json",
        enable_request_priorities=enable_request_priorities,
    )
    executor.set_event_context(
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    return executor


def test_vllm_request_prefix_invariant_and_delta_extraction(tmp_path: Path) -> None:
    """Sequential streamed requests enforce prev-input+output prefix and log exact delta."""

    executor = build_executor(tmp_path=tmp_path)
    client = cast(PrefixClient, executor.client)
    stream_id = "decode:node_root"
    kwargs = {
        "completions_async": client.completions_async,
        "model": "fake-model",
        "prompt": None,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 1,
        "n": 1,
        "seed": 7,
        "stop": None,
        "top_logprobs": 5,
        "assistant_prefix": "",
        "request_kind": "decode_chunk",
        "request_stream_id": stream_id,
        "prefix_chain_enabled": True,
    }

    async def run_sequence() -> None:
        _ = await executor._request_completions_with_limit(
            prompt_token_ids=(1, 2, 3),
            **kwargs,
        )
        _ = await executor._request_completions_with_limit(
            prompt_token_ids=(1, 2, 3, 101),
            **kwargs,
        )
        with pytest.raises(AssertionError, match="request stream prefix mismatch"):
            _ = await executor._request_completions_with_limit(
                prompt_token_ids=(1, 99, 3),
                **kwargs,
            )

    asyncio.run(run_sequence())
    rows = executor.artifact_store.read_event_rows()
    request_rows = [row for row in rows if row.get("event_type") == "vllm_request"]
    assert len(request_rows) >= 2
    assert request_rows[0]["payload"]["delta_input_token_ids"] == [1, 2, 3]
    assert request_rows[1]["payload"]["delta_input_token_ids"] == []
    assert request_rows[1]["payload"]["delta_token_count"] == 0


def test_vllm_response_alternatives_truncated_to_four(tmp_path: Path) -> None:
    """Serialized response token alternatives should keep top four alternates only."""

    executor = build_executor(tmp_path=tmp_path)
    choice = GenerationChoice(
        index=0,
        text="answer",
        finish_reason="length",
        stop_reason=None,
        tokens=(
            ParsedToken(
                token="selected",
                logprob=-0.1,
                top_entries=(
                    ("selected", -0.1),
                    ("alt_a", -0.2),
                    ("alt_b", -0.3),
                    ("alt_c", -0.4),
                    ("alt_d", -0.5),
                    ("alt_e", -0.6),
                ),
            ),
        ),
        prompt_token_ids=None,
        token_ids=(123,),
    )

    payload = executor._serialize_choice_for_vllm_event(choice=choice)
    token_rows = payload["tokens"]
    assert isinstance(token_rows, list)
    assert len(token_rows) == 1
    alternatives = token_rows[0]["top_logprob_alternatives"]
    assert [row["token_text"] for row in alternatives] == [
        "alt_a",
        "alt_b",
        "alt_c",
        "alt_d",
    ]


def test_vllm_request_includes_branch_priority_fields(tmp_path: Path) -> None:
    """Priority-enabled executor should emit branch-number-derived request priority."""

    executor = build_executor(tmp_path=tmp_path, enable_request_priorities=True)
    client = cast(PrefixClient, executor.client)

    async def run_requests() -> None:
        _ = await executor._generate_many_async(
            assistant_prefix="",
            prompt_token_ids=(1, 2, 3),
            max_tokens=1,
            n=1,
            stop=None,
            request_kind="decode_chunk",
            request_stream_id="decode:node_node_root_0_9",
            enforce_prefix_chain=False,
        )
        _ = await executor._generate_many_async(
            assistant_prefix="",
            prompt_token_ids=(1, 2, 3),
            max_tokens=1,
            n=1,
            stop=None,
            request_kind="decode_chunk",
            request_stream_id="decode:node_node_root_1_7",
            enforce_prefix_chain=False,
        )

    asyncio.run(run_requests())
    request_rows = [
        row
        for row in executor.artifact_store.read_event_rows()
        if row["event_type"] == "vllm_request"
    ]
    assert len(request_rows) >= 2
    payload_first = request_rows[0]["payload"]
    payload_second = request_rows[1]["payload"]
    assert payload_first["request_branch_number"] == "1.1"
    assert payload_second["request_branch_number"] == "1.2"
    assert int(payload_first["request_priority"]) < int(
        payload_second["request_priority"]
    )
    assert client.priorities[0] == payload_first["request_priority"]
    assert client.priorities[1] == payload_second["request_priority"]


def test_event_writer_monotonic_indices_and_float_quantization(tmp_path: Path) -> None:
    """Concurrent appends should remain monotonic and quantize floats to 4 decimals."""

    store = ArtifactStore(run_dir=tmp_path / "run", reuse_candidate_pools=False)
    context = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )

    def append_one(value: int) -> None:
        store.append_event(
            context=context,
            event_type="worker_event",
            payload={"score": value + 0.123456},
        )

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(append_one, range(40)))
    events = store.read_events()
    indices = sorted(event.event_index for event in events)
    assert indices == list(range(40))
    one_payload = events[0].payload
    assert isinstance(one_payload["score"], float)
    assert one_payload["score"] == round(float(one_payload["score"]), 4)


def test_resume_planning_skips_finished_and_restarts_incomplete(tmp_path: Path) -> None:
    """Resume plans should skip completed docs and continue incomplete attempts in place."""

    store = ArtifactStore(run_dir=tmp_path / "run", reuse_candidate_pools=False)
    context_doc0 = EventContext(
        run_id=store.run_id,
        doc_id=0,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    context_doc1 = EventContext(
        run_id=store.run_id,
        doc_id=1,
        doc_attempt=0,
        task_name="aime24",
        model_id="fake",
        selector_mode="random",
    )
    store.append_event(context=context_doc0, event_type="doc_started", payload={})
    store.append_event(
        context=context_doc0,
        event_type="doc_finished",
        payload={"status": "completed", "doc_metrics": {}, "diagnostics": {}},
    )
    store.append_event(context=context_doc1, event_type="doc_started", payload={})
    docs = [
        DocRecord(doc_id=0, doc_payload={}, prompt_text="p0"),
        DocRecord(doc_id=1, doc_payload={}, prompt_text="p1"),
        DocRecord(doc_id=2, doc_payload={}, prompt_text="p2"),
    ]
    plans, skipped = build_doc_execution_plans(
        docs=docs,
        existing_events=store.read_events(),
        resume_enabled=True,
    )
    assert skipped == 1
    assert [plan.doc_record.doc_id for plan in plans] == [1, 2]
    assert [plan.doc_attempt for plan in plans] == [0, 0]
    assert plans[0].resumed_reason == "resume_from_partial_logs"
    assert plans[1].resumed_reason is None
    assert plans[0].resume_in_place is True
    assert plans[1].resume_in_place is False


def test_recompute_outputs_uses_latest_completed_attempt_only(tmp_path: Path) -> None:
    """Aggregate recompute should ignore stale older attempts for same doc."""

    store = ArtifactStore(run_dir=tmp_path / "run", reuse_candidate_pools=False)
    contexts = {
        (0, 0): EventContext(
            run_id=store.run_id,
            doc_id=0,
            doc_attempt=0,
            task_name="aime24",
            model_id="fake",
            selector_mode="random",
        ),
        (0, 1): EventContext(
            run_id=store.run_id,
            doc_id=0,
            doc_attempt=1,
            task_name="aime24",
            model_id="fake",
            selector_mode="random",
        ),
        (1, 0): EventContext(
            run_id=store.run_id,
            doc_id=1,
            doc_attempt=0,
            task_name="aime24",
            model_id="fake",
            selector_mode="random",
        ),
    }

    def diag_payload(*, doc_id: int) -> dict[str, Any]:
        diag = DocDiagnostics(
            doc_id=doc_id,
            selector_mode="random",
            verification_variance_leaf=0.1,
            length_variance_leaf=0.2,
            breakpoint_variance=BreakpointVariance(
                bp1_verification_unweighted=0.1,
                bp1_verification_weighted=0.1,
                bp1_length_unweighted=0.1,
                bp1_length_weighted=0.1,
                bp2_verification_unweighted=0.1,
                bp2_verification_weighted=0.1,
                bp2_length_unweighted=0.1,
                bp2_length_weighted=0.1,
            ),
            length_summary=LengthSummary(
                count=1,
                mean_value=10.0,
                median_value=10.0,
                std_value=0.0,
            ),
        )
        return asdict(diag)

    store.append_event(
        context=contexts[(0, 0)],
        event_type="doc_finished",
        payload={
            "status": "completed",
            "doc_metrics": {"acc": 0.0},
            "diagnostics": diag_payload(doc_id=0),
            "leaf_lengths": [8],
        },
    )
    store.append_event(
        context=contexts[(0, 1)],
        event_type="doc_finished",
        payload={
            "status": "completed",
            "doc_metrics": {"acc": 1.0},
            "diagnostics": diag_payload(doc_id=0),
            "leaf_lengths": [10],
        },
    )
    store.append_event(
        context=contexts[(1, 0)],
        event_type="doc_finished",
        payload={
            "status": "completed",
            "doc_metrics": {"acc": 0.5},
            "diagnostics": diag_payload(doc_id=1),
            "leaf_lengths": [12],
        },
    )

    class FakeAdapter:
        def aggregate_task_metrics(
            self, *, per_doc_metrics: list[dict[str, Any]]
        ) -> dict[str, Any]:
            values = [float(item["acc"]) for item in per_doc_metrics]
            return {"acc": mean(values) if values else 0.0}

    summary = recompute_outputs_from_events(
        store=store,
        adapter=cast(Any, FakeAdapter()),
        selector_mode="random",
    )
    assert summary["doc_count_completed"] == 2
    payload = json.loads((store.run_dir / "lm_eval_aggregates.json").read_text())
    assert payload["acc"] == pytest.approx(0.75)
