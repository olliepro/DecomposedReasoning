"""Tests for selector-runtime clustering backends."""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import branching_eval.selector_runtime as selector_runtime_module
from candidate_clustering import DedupItem
from branching_eval.embedding_selection import (
    OPENAI_EMBEDDING_DIMENSIONS,
    _openai_embedding_batch_async,
    cleaned_candidate_groups,
    greedy_diverse_indices,
    normalized_embedding_matrix,
    openai_diverse_topk_random_ids_async,
)
from branching_eval.selector_runtime import (
    select_candidates_all_modes,
    select_candidates_all_modes_async,
)
from branching_eval.selector_types import SelectorParams
from branching_eval.tree_types import CandidatePoolRecord, CandidateRecord
from io_utils import read_jsonl


def candidate_pool_fixture() -> CandidatePoolRecord:
    """Build one candidate pool with unique texts for clustering tests."""

    return CandidatePoolRecord(
        candidate_pool_id="pool",
        branch_point_id="bp",
        node_id="node",
        trigger_type="steer_boundary",
        entropy_value=None,
        candidates=(
            CandidateRecord(
                candidate_id=0,
                text="factor the expression",
                token_ids=(10,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer>",
            ),
            CandidateRecord(
                candidate_id=1,
                text="rewrite the expression",
                token_ids=(11,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer>",
            ),
            CandidateRecord(
                candidate_id=2,
                text="solve by substitution",
                token_ids=(12,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer>",
            ),
        ),
    )


def test_cluster_selectors_retry_low_assignment_rate_live_requests(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Cluster selectors should retry weak coverage responses on live requests."""

    responses = [
        '{"groups":[{"name":"alpha","key":"alpha"}],"assignments":{"1":"alpha"}}',
        (
            '{"groups":[{"name":"alpha","key":"alpha"},'
            '{"name":"beta","key":"beta"}],'
            '"assignments":{"1":"alpha","2":"alpha","3":"beta"}}'
        ),
    ]
    call_count = {"value": 0}

    async def fake_call_once(
        *,
        http_session,
        api_key: str,
        items,
        prompt: str,
        attempt_number: int,
        cluster_log_path: Path | None,
    ) -> tuple[list[dict[str, object]], str]:
        _ = api_key, http_session
        response_text = responses[min(call_count["value"], len(responses) - 1)]
        call_count["value"] += 1
        selector_runtime_module._append_cluster_log(
            cluster_log_path=cluster_log_path,
            payload={
                "event": "attempt_raw_response",
                "attempt_number": attempt_number,
                "model_id": selector_runtime_module.OPENROUTER_CLUSTER_MODEL,
                "item_count": len(items),
                "temperature": selector_runtime_module._cluster_attempt_temperature(
                    attempt_number=attempt_number
                ),
                "prompt": prompt,
                "raw_text": response_text,
            },
        )
        clusters = selector_runtime_module.parse_clusters_payload(
            raw_text=response_text
        )
        selector_runtime_module.validate_clusters_response(
            items=items,
            clusters=clusters,
        )
        selector_runtime_module._append_cluster_log(
            cluster_log_path=cluster_log_path,
            payload={
                "event": "attempt_success",
                "attempt_number": attempt_number,
                "model_id": selector_runtime_module.OPENROUTER_CLUSTER_MODEL,
                "item_count": len(items),
                "assignment_rate": selector_runtime_module._cluster_assignment_rate(
                    items=items,
                    clusters=clusters,
                ),
            },
        )
        return clusters, response_text

    monkeypatch.setattr(
        "branching_eval.selector_runtime._call_openrouter_clusters_once_async",
        fake_call_once,
    )
    monkeypatch.setattr(
        "branching_eval.selector_runtime._dedup_items",
        lambda **_: (
            DedupItem(item_id=1, text="factor the expression", count=1),
            DedupItem(item_id=2, text="rewrite the expression", count=1),
            DedupItem(item_id=3, text="solve by substitution", count=1),
        ),
    )

    selector_params = SelectorParams(branch_fanout=2, max_clusters=4)
    pool = candidate_pool_fixture()

    outcomes = select_candidates_all_modes(
        pool=pool,
        selector_params=selector_params,
        selector_modes=("cluster_across",),
        rng=random.Random(0),
        openrouter_api_key="test-openrouter-key",
        cluster_log_path=tmp_path / "clustering_debug.jsonl",
    )
    cached_outcomes = select_candidates_all_modes(
        pool=pool,
        selector_params=selector_params,
        selector_modes=("cluster_across",),
        rng=random.Random(1),
        openrouter_api_key="test-openrouter-key",
        cluster_log_path=tmp_path / "clustering_debug.jsonl",
    )

    expected = {0: "alpha", 1: "alpha", 2: "beta"}
    log_rows = read_jsonl(path=tmp_path / "clustering_debug.jsonl")
    assert outcomes[0].cluster_by_candidate_id == expected
    assert cached_outcomes[0].cluster_by_candidate_id == expected
    assert call_count["value"] == 3
    assert [row["event"] for row in log_rows] == [
        "attempt_raw_response",
        "attempt_failure",
        "attempt_raw_response",
        "attempt_success",
        "attempt_raw_response",
        "attempt_success",
    ]
    assert log_rows[0]["raw_text"] == responses[0]
    assert log_rows[2]["raw_text"] == responses[1]
    assert log_rows[4]["raw_text"] == responses[1]


def test_cluster_selectors_async_retry_path(tmp_path: Path, monkeypatch) -> None:
    """Async selector resolution should retry through the async clustering path."""

    async def fake_call_once(
        *,
        http_session,
        api_key: str,
        items,
        prompt: str,
        attempt_number: int,
        cluster_log_path: Path | None,
    ) -> tuple[list[dict[str, object]], str]:
        _ = api_key, prompt, cluster_log_path, attempt_number, http_session
        response_text = (
            '{"groups":[{"name":"alpha","key":"alpha"},'
            '{"name":"beta","key":"beta"}],'
            '"assignments":{"1":"alpha","2":"alpha","3":"beta"}}'
        )
        clusters = selector_runtime_module.parse_clusters_payload(
            raw_text=response_text
        )
        selector_runtime_module.validate_clusters_response(
            items=items,
            clusters=clusters,
        )
        return clusters, response_text

    monkeypatch.setattr(
        "branching_eval.selector_runtime._call_openrouter_clusters_once_async",
        fake_call_once,
    )
    monkeypatch.setattr(
        "branching_eval.selector_runtime._dedup_items",
        lambda **_: (
            DedupItem(item_id=1, text="factor the expression", count=1),
            DedupItem(item_id=2, text="rewrite the expression", count=1),
            DedupItem(item_id=3, text="solve by substitution", count=1),
        ),
    )

    selector_params = SelectorParams(branch_fanout=2, max_clusters=4)
    pool = candidate_pool_fixture()

    outcomes = asyncio.run(
        select_candidates_all_modes_async(
            pool=pool,
            selector_params=selector_params,
            selector_modes=("cluster_across",),
            rng=random.Random(0),
            openrouter_api_key="test-openrouter-key",
            cluster_log_path=tmp_path / "clustering_debug_async.jsonl",
        )
    )

    assert outcomes[0].cluster_by_candidate_id == {0: "alpha", 1: "alpha", 2: "beta"}


def duplicate_candidate_pool_fixture() -> CandidatePoolRecord:
    """Build one pool with duplicate cleaned texts for OpenAI selector tests."""

    return CandidatePoolRecord(
        candidate_pool_id="dup-pool",
        branch_point_id="bp",
        node_id="node",
        trigger_type="steer_boundary",
        entropy_value=None,
        candidates=(
            CandidateRecord(
                candidate_id=0,
                text="alpha</steer>",
                token_ids=(10,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer>",
            ),
            CandidateRecord(
                candidate_id=1,
                text="alpha",
                token_ids=(11,),
                tokens=(),
                finish_reason="stop",
                stop_reason=None,
            ),
            CandidateRecord(
                candidate_id=2,
                text="beta",
                token_ids=(12,),
                tokens=(),
                finish_reason="stop",
                stop_reason=None,
            ),
            CandidateRecord(
                candidate_id=3,
                text="gamma",
                token_ids=(13,),
                tokens=(),
                finish_reason="stop",
                stop_reason=None,
            ),
            CandidateRecord(
                candidate_id=4,
                text="delta",
                token_ids=(14,),
                tokens=(),
                finish_reason="stop",
                stop_reason=None,
            ),
        ),
    )


def _padded_vector(*, x: float, y: float) -> tuple[float, ...]:
    """Build one 130-d vector whose first two coordinates are controllable."""

    return (x, y) + (0.0,) * 126 + (999.0, -999.0)


def test_cleaned_candidate_groups_skips_empty_texts() -> None:
    """OpenAI selector grouping should not emit blank embedding inputs."""

    pool = CandidatePoolRecord(
        candidate_pool_id="empty-pool",
        branch_point_id="bp",
        node_id="node",
        trigger_type="steer_boundary",
        entropy_value=None,
        candidates=(
            CandidateRecord(
                candidate_id=0,
                text="</steer>",
                token_ids=(10,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer>",
            ),
            CandidateRecord(
                candidate_id=1,
                text=" useful candidate </steer>",
                token_ids=(11,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer>",
            ),
        ),
    )

    groups = cleaned_candidate_groups(candidates=pool.candidates)

    assert [group.text for group in groups] == ["useful candidate"]
    assert [group.candidate_ids for group in groups] == [(1,)]


def test_normalized_embedding_matrix_normalizes_rows() -> None:
    """Embedding matrix helper should normalize rows without resizing them."""

    texts = ("left", "right")
    embeddings_by_text = {
        "left": (3.0, 4.0),
        "right": (6.0, 8.0),
    }

    matrix = normalized_embedding_matrix(
        texts=texts,
        embeddings_by_text=embeddings_by_text,
    )

    assert matrix.shape == (2, 2)
    assert abs(float(matrix[0, 0]) - 0.6) < 1e-6
    assert abs(float(matrix[0, 1]) - 0.8) < 1e-6
    assert abs(float(matrix[1, 0]) - 0.6) < 1e-6
    assert abs(float(matrix[1, 1]) - 0.8) < 1e-6


def test_openai_diverse_topk_random_dedupes_and_samples_from_shortlist(
    monkeypatch,
) -> None:
    """OpenAI selector should batch unique cleaned texts and sample from shortlist."""

    pool = duplicate_candidate_pool_fixture()
    seen_batches: list[tuple[str, ...]] = []

    async def fake_batch(  # type: ignore[no-untyped-def]
        *,
        session,
        texts,
        model,
        openai_api_key,
        output_dimensions,
    ):
        _ = session, model, openai_api_key
        assert output_dimensions == OPENAI_EMBEDDING_DIMENSIONS
        seen_batches.append(tuple(texts))
        vectors = {
            "alpha": _padded_vector(x=1.0, y=0.0),
            "beta": _padded_vector(x=0.0, y=1.0),
            "gamma": _padded_vector(x=-1.0, y=0.0),
            "delta": _padded_vector(x=0.0, y=-1.0),
        }
        return [vectors[text] for text in texts]

    monkeypatch.setattr(
        "branching_eval.embedding_selection._openai_embedding_batch_async",
        fake_batch,
    )

    shortlist_ids, selected_ids = asyncio.run(
        openai_diverse_topk_random_ids_async(
            pool=pool,
            branch_count=2,
            rng=random.Random(0),
            openai_api_key="test-openai-key",
        )
    )

    matrix = normalized_embedding_matrix(
        texts=("alpha", "beta", "gamma", "delta"),
        embeddings_by_text={
            "alpha": _padded_vector(x=1.0, y=0.0),
            "beta": _padded_vector(x=0.0, y=1.0),
            "gamma": _padded_vector(x=-1.0, y=0.0),
            "delta": _padded_vector(x=0.0, y=-1.0),
        },
    )
    shortlist = greedy_diverse_indices(matrix=matrix, max_count=3)
    shortlist_candidate_ids = {0, 2, 3}

    assert seen_batches == [("alpha", "beta", "gamma", "delta")]
    assert shortlist == (0, 2, 1)
    assert shortlist_ids == (0, 3, 2)
    assert len(selected_ids) == 2
    assert set(selected_ids).issubset(shortlist_candidate_ids)
    assert 1 not in selected_ids


def test_selector_runtime_openai_mode_omits_raw_embeddings(monkeypatch) -> None:
    """Selector runtime should not persist raw vectors for OpenAI selector mode."""

    pool = duplicate_candidate_pool_fixture()

    monkeypatch.setattr(
        "branching_eval.selector_runtime.openai_diverse_topk_random_ids",
        lambda **_: ((0, 2), (0, 2)),
    )

    outcomes = select_candidates_all_modes(
        pool=pool,
        selector_params=SelectorParams(branch_fanout=2, max_clusters=4),
        selector_modes=("embed_diverse_topk_random",),
        rng=random.Random(0),
        openrouter_api_key=None,
        openai_api_key="test-openai-key",
    )

    assert outcomes[0].selected_candidate_ids == (0, 2)
    assert outcomes[0].embedding_by_candidate_id is None
    assert outcomes[0].shortlist_candidate_ids == (0, 2)


def test_openai_diverse_topk_random_uses_provided_http_session(monkeypatch) -> None:
    """OpenAI selector should reuse the provided HTTP session for batch fetches."""

    pool = duplicate_candidate_pool_fixture()
    seen_sessions: list[object] = []
    sentinel_session = object()

    async def fake_batch(  # type: ignore[no-untyped-def]
        *,
        session,
        texts,
        model,
        openai_api_key,
        output_dimensions,
    ):
        _ = texts, model, openai_api_key, output_dimensions
        seen_sessions.append(session)
        return [_padded_vector(x=1.0, y=0.0) for _ in texts]

    monkeypatch.setattr(
        "branching_eval.embedding_selection._openai_embedding_batch_async",
        fake_batch,
    )

    asyncio.run(
        openai_diverse_topk_random_ids_async(
            pool=pool,
            branch_count=2,
            rng=random.Random(0),
            openai_api_key="test-openai-key",
            session=sentinel_session,  # type: ignore[arg-type]
        )
    )

    assert seen_sessions == [sentinel_session]


class FakeEmbeddingResponse:
    """Async context response fixture for OpenAI embedding HTTP tests."""

    def __init__(self, *, status: int, body: str) -> None:
        """Store fake response fields.

        Args:
            status: HTTP status exposed to the caller.
            body: Response text returned by `text`.

        Returns:
            None.
        """

        self.status = status
        self.body = body

    async def __aenter__(self) -> "FakeEmbeddingResponse":
        """Return the fake response for async context manager use."""

        return self

    async def __aexit__(self, *args: object) -> None:
        """Exit the fake async context manager."""

        _ = args

    async def text(self) -> str:
        """Return the configured response body."""

        return self.body


class FakeEmbeddingSession:
    """Minimal session fixture returning queued embedding responses."""

    def __init__(self, *, responses: list[FakeEmbeddingResponse]) -> None:
        """Initialize a fake session.

        Args:
            responses: Ordered responses to return from `post`.

        Returns:
            None.
        """

        self.responses = responses
        self.calls = 0

    def post(self, *args: Any, **kwargs: Any) -> FakeEmbeddingResponse:
        """Return the next queued fake response.

        Args:
            args: Positional request arguments, including the URL.
            kwargs: Request keyword arguments, captured only for signature parity.

        Returns:
            Next fake response.
        """

        _ = args, kwargs
        response = self.responses[self.calls]
        self.calls += 1
        return response


def _embedding_response_body(*, values: tuple[float, ...]) -> str:
    """Return a valid OpenAI embedding response JSON string."""

    return json.dumps({"data": [{"index": 0, "embedding": list(values)}]})


def test_openai_embedding_batch_retries_transient_http_error(monkeypatch) -> None:
    """OpenAI embedding batches should retry transient 5xx failures."""

    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr("branching_eval.embedding_selection.asyncio.sleep", fake_sleep)
    session = FakeEmbeddingSession(
        responses=[
            FakeEmbeddingResponse(status=502, body="bad gateway"),
            FakeEmbeddingResponse(
                status=200,
                body=_embedding_response_body(values=(1.0, 0.0)),
            ),
        ]
    )

    vectors = asyncio.run(
        _openai_embedding_batch_async(
            session=session,  # type: ignore[arg-type]
            texts=("alpha",),
            model="text-embedding-3-small",
            openai_api_key="test-openai-key",
            output_dimensions=2,
        )
    )

    assert session.calls == 2
    assert sleep_delays == [1.0]
    assert vectors == [(1.0, 0.0)]


def test_openai_embedding_batch_does_not_retry_auth_error(monkeypatch) -> None:
    """OpenAI embedding batches should fail immediately on non-retryable 4xx."""

    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    monkeypatch.setattr("branching_eval.embedding_selection.asyncio.sleep", fake_sleep)
    session = FakeEmbeddingSession(
        responses=[FakeEmbeddingResponse(status=401, body="bad key")]
    )

    try:
        asyncio.run(
            _openai_embedding_batch_async(
                session=session,  # type: ignore[arg-type]
                texts=("alpha",),
                model="text-embedding-3-small",
                openai_api_key="test-openai-key",
                output_dimensions=2,
            )
        )
    except RuntimeError as exc:
        assert "OpenAI embedding error 401 after 1 attempts" in str(exc)
    else:
        raise AssertionError("expected non-retryable 401 to fail")

    assert session.calls == 1
    assert sleep_delays == []
