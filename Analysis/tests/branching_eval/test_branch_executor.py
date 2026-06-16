"""Tests for branching execution, selectors, and candidate-pool reuse."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
import time
from typing import Any, cast

import pytest

import branching_eval.branch_executor as branch_executor_module
from branching_eval.artifact_store import ArtifactStore
from branching_eval.branch_executor import BranchExecutor, DecodeOutcome, PathState
from chat_templating import build_raw_im_prompt
from branching_eval.config_types import BranchingConfig, DecodingConfig
from branching_eval.event_types import EventContext
from branching_eval.selector_types import SelectionOutcome
from branching_eval.selector_runtime import (
    _cluster_across_ids,
    _random_ids,
    _within_cluster_ids,
)
from branching_eval.steer_decode_flow import (
    continue_with_single_steer_candidate,
    continue_with_single_steer_candidate_async,
    continue_after_think_close_async,
)
from branching_eval.tree_types import (
    BranchTree,
    CandidatePoolRecord,
    CandidateRecord,
    TokenTrace,
)
from vllm_client import GenerationChoice, ParsedToken, VllmClient, VllmRequestError


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
        resolved_prompt_token_ids: tuple[int, ...] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        parse_response_prompt_token_ids: bool = True,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            prompt,
            prompt_token_ids,
            resolved_prompt_token_ids,
            temperature,
            top_p,
            max_tokens,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
            parse_response_prompt_token_ids,
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
        return (
            make_choice(
                index=0,
                text=" x<steer>",
                entropy_logprob=-0.05,
                finish_reason="stop",
            ),
        )

    async def completions_async(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        resolved_prompt_token_ids: tuple[int, ...] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        parse_response_prompt_token_ids: bool = True,
    ) -> tuple[GenerationChoice, ...]:
        """Async wrapper matching executor async completion interface."""

        return self.completions(
            model=model,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            resolved_prompt_token_ids=resolved_prompt_token_ids,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
            priority=priority,
            repetition_penalty=repetition_penalty,
            parse_response_prompt_token_ids=parse_response_prompt_token_ids,
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


class SessionLifecycleFakeClient(FakeClient):
    """Fake client that records runtime session lifecycle calls."""

    def __init__(self) -> None:
        super().__init__()
        self.close_count = 0
        self.dropped_session_keys: list[str] = []

    async def close_async(self) -> None:
        """Record a whole-client close call."""

        self.close_count += 1

    async def recreate_async_session(self, *, session_key: str | None = None) -> None:
        """Record a keyed session drop call."""

        assert session_key is not None, "session_key must be provided"
        self.dropped_session_keys.append(session_key)


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
        resolved_prompt_token_ids: tuple[int, ...] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        parse_response_prompt_token_ids: bool = True,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            prompt,
            prompt_token_ids,
            resolved_prompt_token_ids,
            temperature,
            top_p,
            max_tokens,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
            parse_response_prompt_token_ids,
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


class PromptReuseClient(FakeClient):
    """Fake client that records parser-fallback prompt ids per request."""

    def __init__(self) -> None:
        super().__init__()
        self.prompt_text_seen: list[str | None] = []
        self.prompt_token_ids_seen: list[tuple[int, ...] | None] = []
        self.resolved_prompt_token_ids_seen: list[tuple[int, ...] | None] = []
        self.parse_response_prompt_token_ids_seen: list[bool] = []

    async def completions_async(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        resolved_prompt_token_ids: tuple[int, ...] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        parse_response_prompt_token_ids: bool = True,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            temperature,
            top_p,
            max_tokens,
            n,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
        )
        self.prompt_text_seen.append(prompt)
        self.prompt_token_ids_seen.append(prompt_token_ids)
        self.resolved_prompt_token_ids_seen.append(resolved_prompt_token_ids)
        self.parse_response_prompt_token_ids_seen.append(
            parse_response_prompt_token_ids
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
                prompt_token_ids=resolved_prompt_token_ids,
                token_ids=(42,),
            ),
        )


class ContextBudgetClient(FakeClient):
    """Fake client that records max_tokens after request-level context clamping."""

    def __init__(self) -> None:
        super().__init__()
        self.max_tokens_seen: list[int] = []
        self.prompt_text_seen: list[str | None] = []
        self.prompt_token_ids_seen: list[tuple[int, ...] | None] = []

    async def completions_async(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        resolved_prompt_token_ids: tuple[int, ...] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        parse_response_prompt_token_ids: bool = True,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            resolved_prompt_token_ids,
            temperature,
            top_p,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
            parse_response_prompt_token_ids,
        )
        self.max_tokens_seen.append(max_tokens)
        self.prompt_text_seen.append(prompt)
        self.prompt_token_ids_seen.append(prompt_token_ids)
        return tuple(
            make_choice(index=index, text="x", entropy_logprob=-0.1)
            for index in range(n)
        )


class TokenPromptTransientErrorClient(FakeClient):
    """Fake client that raises a transient request error for token prompts."""

    def __init__(self) -> None:
        super().__init__()
        self.supports_prompt_token_ids: bool | None = None
        self.prompt_text_seen: list[str | None] = []
        self.prompt_token_ids_seen: list[tuple[int, ...] | None] = []

    async def completions_async(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        resolved_prompt_token_ids: tuple[int, ...] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        parse_response_prompt_token_ids: bool = True,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            resolved_prompt_token_ids,
            temperature,
            top_p,
            max_tokens,
            n,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
            parse_response_prompt_token_ids,
        )
        self.prompt_text_seen.append(prompt)
        self.prompt_token_ids_seen.append(prompt_token_ids)
        if prompt_token_ids is not None:
            raise VllmRequestError("Server disconnected")
        return super().completions(
            model=model,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            resolved_prompt_token_ids=resolved_prompt_token_ids,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
            priority=priority,
            repetition_penalty=repetition_penalty,
            parse_response_prompt_token_ids=parse_response_prompt_token_ids,
        )


class BatchedDisconnectClient(FakeClient):
    """Fake client that disconnects once for a large batched request."""

    def __init__(self) -> None:
        super().__init__()
        self.requested_batch_sizes: list[int] = []

    async def completions_async(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        resolved_prompt_token_ids: tuple[int, ...] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        parse_response_prompt_token_ids: bool = True,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            prompt,
            prompt_token_ids,
            resolved_prompt_token_ids,
            temperature,
            top_p,
            max_tokens,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
            parse_response_prompt_token_ids,
        )
        self.requested_batch_sizes.append(n)
        if self.requested_batch_sizes == [10]:
            raise VllmRequestError("Server disconnected")
        return tuple(
            make_choice(index=index, text=f"split_{n}_{index}", entropy_logprob=-0.2)
            for index in range(n)
        )


class IncompleteBatchedClient(FakeClient):
    """Fake client that returns too few choices for the initial batch."""

    def __init__(self) -> None:
        super().__init__()
        self.requested_batch_sizes: list[int] = []

    async def completions_async(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
        resolved_prompt_token_ids: tuple[int, ...] | None = None,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        parse_response_prompt_token_ids: bool = True,
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            prompt,
            prompt_token_ids,
            resolved_prompt_token_ids,
            temperature,
            top_p,
            max_tokens,
            seed,
            stop,
            top_logprobs,
            priority,
            repetition_penalty,
            parse_response_prompt_token_ids,
        )
        self.requested_batch_sizes.append(n)
        if self.requested_batch_sizes == [10]:
            return tuple(
                make_choice(index=index, text=f"partial_{index}", entropy_logprob=-0.2)
                for index in range(9)
            )
        return tuple(
            make_choice(index=index, text=f"split_{n}_{index}", entropy_logprob=-0.2)
            for index in range(n)
        )


def make_choice(
    *,
    index: int,
    text: str,
    entropy_logprob: float,
    finish_reason: str = "length",
) -> GenerationChoice:
    """Build one deterministic generation choice."""

    parsed_token = ParsedToken(
        token=text,
        logprob=entropy_logprob,
        top_entries=(("a", -0.2), ("b", -1.0)),
    )
    return GenerationChoice(
        index=index,
        text=text,
        finish_reason=finish_reason,
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


def test_generate_many_async_reuses_locally_resolved_prompt_ids(
    tmp_path: Path,
) -> None:
    """Executor should pass known prompt ids and skip echoed response prompt ids."""

    client = PromptReuseClient()
    executor = build_executor(tmp_path=tmp_path)
    executor.client = cast(VllmClient, client)
    choice = asyncio.run(
        executor._generate_choice_async(
            assistant_prefix="prefix",
            prompt_token_ids=None,
            max_tokens=1,
            stop=None,
            n=1,
            request_stream_id="decode:node_root",
            enforce_prefix_chain=True,
        )
    )
    expected_prompt_ids = asyncio.run(
        executor.client.tokenize_async(
            model=executor.model_name,
            text=build_raw_im_prompt(
                prompt="Solve this.",
                assistant_prefix="prefix",
            ),
            add_special_tokens=False,
        )
    )
    assert client.prompt_text_seen == [None]
    assert client.prompt_token_ids_seen == [expected_prompt_ids]
    assert client.resolved_prompt_token_ids_seen == [expected_prompt_ids]
    assert client.parse_response_prompt_token_ids_seen == [False]
    assert choice.prompt_token_ids == expected_prompt_ids
    executor.artifact_store.close()


def test_generate_many_async_clamps_text_prompt_request_to_context(
    tmp_path: Path,
) -> None:
    """Text-prompt requests should be capped using locally resolved input ids."""

    client = ContextBudgetClient()
    executor = build_executor(tmp_path=tmp_path)
    executor.client = cast(VllmClient, client)
    prompt_text = build_raw_im_prompt(
        prompt="Solve this.",
        assistant_prefix="prefix",
    )
    input_token_ids = asyncio.run(
        client.tokenize_async(
            model=executor.model_name,
            text=prompt_text,
            add_special_tokens=False,
        )
    )
    executor.decoding = replace(
        executor.decoding,
        max_model_len=len(input_token_ids) + 2,
    )

    _ = asyncio.run(
        executor._generate_many_async(
            assistant_prefix="prefix",
            prompt_token_ids=None,
            max_tokens=8,
            stop=None,
            n=3,
            request_stream_id="baseline:0:0:pool",
            enforce_prefix_chain=False,
        )
    )

    assert client.prompt_text_seen == [prompt_text]
    assert client.prompt_token_ids_seen == [None]
    assert client.max_tokens_seen == [2]
    executor.artifact_store.close()


def test_active_prefix_chain_requires_prompt_token_ids(tmp_path: Path) -> None:
    """Active prefix-chain streams must not retokenize full prompt text."""

    client = PromptReuseClient()
    executor = build_executor(tmp_path=tmp_path)
    executor.client = cast(VllmClient, client)
    stream_id = "decode:node_root"
    executor._request_stream_state[stream_id] = (
        branch_executor_module._RequestStreamState(
            request_id="req_old",
            input_token_ids=(99, 100, 101),
            output_token_ids=(102,),
        )
    )

    with pytest.raises(AssertionError, match="lost prompt_token_ids"):
        asyncio.run(
            executor._generate_choice_async(
                assistant_prefix="fresh retokenized text",
                prompt_token_ids=None,
                max_tokens=1,
                stop=None,
                n=1,
                request_stream_id=stream_id,
                enforce_prefix_chain=True,
            )
        )

    assert client.prompt_text_seen == []
    assert client.prompt_token_ids_seen == []
    executor.artifact_store.close()


def test_transient_token_prompt_error_does_not_disable_token_prompt_support(
    tmp_path: Path,
) -> None:
    """Transient request errors must not poison token-prompt support cache."""

    client = TokenPromptTransientErrorClient()
    executor = build_executor(tmp_path=tmp_path)
    executor.client = cast(VllmClient, client)

    with pytest.raises(VllmRequestError, match="Server disconnected"):
        asyncio.run(
            executor._generate_many_async(
                assistant_prefix="prefix",
                prompt_token_ids=(1, 2, 3),
                max_tokens=1,
                stop=None,
                n=1,
                request_stream_id=None,
                enforce_prefix_chain=False,
            )
        )

    assert client.supports_prompt_token_ids is None
    assert client.prompt_text_seen == [None]
    assert client.prompt_token_ids_seen == [(1, 2, 3)]
    executor.artifact_store.close()


def test_retryable_batched_completion_failure_splits_request(
    tmp_path: Path,
) -> None:
    """Retryable disconnects should split large completion batches."""

    client = BatchedDisconnectClient()
    executor = build_executor(tmp_path=tmp_path)
    executor.client = cast(VllmClient, client)

    choices = asyncio.run(
        executor._generate_many_async(
            assistant_prefix="",
            prompt_token_ids=None,
            max_tokens=1,
            stop=None,
            n=10,
            request_stream_id="baseline:0:0:10",
            enforce_prefix_chain=False,
        )
    )

    assert client.requested_batch_sizes == [10, 5, 5]
    assert len(choices) == 10
    assert [choice.index for choice in choices] == list(range(10))
    executor.artifact_store.close()


def test_incomplete_batched_completion_response_splits_request(
    tmp_path: Path,
) -> None:
    """Incomplete multi-choice responses should split instead of leaking rows."""

    client = IncompleteBatchedClient()
    executor = build_executor(tmp_path=tmp_path)
    executor.client = cast(VllmClient, client)

    choices = asyncio.run(
        executor._generate_many_async(
            assistant_prefix="",
            prompt_token_ids=None,
            max_tokens=1,
            stop=None,
            n=10,
            request_stream_id="baseline:0:0:10",
            enforce_prefix_chain=False,
        )
    )

    assert client.requested_batch_sizes == [10, 5, 5]
    assert len(choices) == 10
    assert [choice.index for choice in choices] == list(range(10))
    executor.artifact_store.close()


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
    max_concurrent_branches: int = 20,
    branch_task_semaphore: asyncio.Semaphore | None = None,
    trigger_steer_enabled: bool = True,
    trigger_entropy_enabled: bool = True,
    debug_assert_text_token_alignment: bool = False,
) -> BranchExecutor:
    """Build branch executor with fake client and random-only selector."""

    store = ArtifactStore(run_dir=tmp_path / "run")
    return BranchExecutor(
        client=cast(VllmClient, FakeClient()),
        cluster_client=None,
        prompt_text="Solve this.",
        model_name="fake",
        cluster_model_name=None,
        decoding=DecodingConfig(
            temperature=0.6,
            top_p=0.95,
            max_gen_toks=8,
            top_logprobs=5,
            debug_assert_text_token_alignment=debug_assert_text_token_alignment,
        ),
        branching=BranchingConfig(
            branch_prob=branch_prob,
            max_branch_points_per_rollout=2,
            max_concurrent_branches=max_concurrent_branches,
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
        branch_task_semaphore=branch_task_semaphore,
    )


def read_tree_event_rows(*, store: ArtifactStore) -> list[dict[str, Any]]:
    """Read flushed tree event rows from one artifact store."""

    return store.read_event_rows()


def test_executor_drops_only_doc_scoped_sessions_when_client_is_shared(
    tmp_path: Path,
) -> None:
    """Concurrent-doc executors must not close the shared vLLM client."""

    store = ArtifactStore(run_dir=tmp_path / "run")
    client = SessionLifecycleFakeClient()
    session_key = ""
    try:
        executor = BranchExecutor(
            client=cast(VllmClient, client),
            cluster_client=None,
            prompt_text="Solve this.",
            model_name="fake",
            cluster_model_name=None,
            decoding=DecodingConfig(max_gen_toks=8, top_logprobs=0),
            branching=BranchingConfig(max_branch_points_per_rollout=1),
            artifact_store=store,
            requested_selectors=("random",),
            active_selector="random",
            seed=7,
            trigger_steer_enabled=True,
            env_paths=(),
            close_runtime_clients_on_finish=False,
        )
        executor.set_event_context(
            doc_id=7,
            doc_attempt=2,
            task_name="task",
            model_id="model",
            selector_mode="random",
        )
        session_key = executor._session_key_for_request_stream(
            request_stream_id="decode:node_root_rollout_0"
        )
        executor._runtime_session_keys.add(session_key)

        asyncio.run(executor._close_runtime_http_sessions_async())
    finally:
        store.close()

    assert session_key == "doc:7:attempt:2:branch:node_root_rollout_0"
    assert client.close_count == 0
    assert client.dropped_session_keys == [session_key]


def install_fake_repeat_close_continuation(
    *,
    monkeypatch: pytest.MonkeyPatch,
    call_counts: dict[str, int],
    final_text: str = "\nFinal answer.",
) -> None:
    """Install a fake repeat-loop post-think continuation.

    Args:
        monkeypatch: Pytest monkeypatch fixture.
        call_counts: Mutable call counters updated by the fake continuation.
        final_text: Text appended by the fake post-think generation.

    Returns:
        None.
    """

    async def fake_continue_after_think_close(
        *,
        executor: BranchExecutor,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        token_ids: tuple[int, ...],
        token_traces: tuple[TokenTrace, ...],
        generated_tokens: int,
        request_stream_id: str,
    ) -> DecodeOutcome:
        _ = executor, request_stream_id
        call_counts["post_think"] = call_counts.get("post_think", 0) + 1
        assert assistant_prefix.endswith(branch_executor_module.THINK_CLOSE_TAG)
        continuation_token_id = 9_000 + call_counts["post_think"]
        continuation_trace = TokenTrace(
            token_index=len(token_traces),
            token_id=continuation_token_id,
            token_text=final_text,
            logprob=-0.1,
            probability=0.9,
        )
        updated_prompt_ids = (
            None
            if prompt_token_ids is None
            else tuple(prompt_token_ids) + (continuation_token_id,)
        )
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=f"{assistant_prefix}{final_text}",
            prompt_token_ids=updated_prompt_ids,
            token_ids=tuple(token_ids) + (continuation_token_id,),
            token_traces=tuple(token_traces) + (continuation_trace,),
            generated_tokens=generated_tokens + 1,
            stop_reason="think_end",
        )

    monkeypatch.setattr(
        "branching_eval.branch_executor.continue_after_think_close_async",
        fake_continue_after_think_close,
    )


def test_generate_many_uses_steer_sampling_for_steer_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Steer request kinds should use optional steer sampling overrides."""

    executor = build_executor(tmp_path=tmp_path)
    executor.branching = replace(executor.branching, steer_repetition_penalty=1.0)
    executor.decoding = DecodingConfig(
        temperature=0.3,
        steer_temperature=0.8,
        top_p=0.95,
        steer_top_p=0.77,
        presence_penalty=1.5,
        repetition_penalty=1.0,
        max_gen_toks=8,
        top_logprobs=5,
    )
    observed_sampling: list[tuple[float, float, float, float]] = []

    async def fake_request_completions(
        **kwargs: object,
    ) -> tuple[GenerationChoice, ...]:
        temperature = kwargs["temperature"]
        top_p = kwargs["top_p"]
        presence_penalty = kwargs["presence_penalty"]
        repetition_penalty = kwargs["repetition_penalty"]
        assert isinstance(temperature, (float, int))
        assert isinstance(top_p, (float, int))
        assert isinstance(presence_penalty, (float, int))
        assert isinstance(repetition_penalty, (float, int))
        observed_sampling.append(
            (
                float(temperature),
                float(top_p),
                float(presence_penalty),
                float(repetition_penalty),
            )
        )
        return (make_choice(index=0, text="done", entropy_logprob=-0.1),)

    monkeypatch.setattr(
        executor,
        "_request_completions_with_limit",
        fake_request_completions,
    )

    asyncio.run(
        executor._generate_many_async(
            assistant_prefix="",
            prompt_token_ids=None,
            max_tokens=1,
            n=1,
            stop=None,
            request_kind="decode_chunk",
        )
    )
    asyncio.run(
        executor._generate_many_async(
            assistant_prefix="",
            prompt_token_ids=None,
            max_tokens=1,
            n=1,
            stop=None,
            request_kind="steer_single_candidate",
        )
    )

    assert observed_sampling == [(0.3, 0.95, 1.5, 1.0), (0.8, 0.77, 1.5, 1.0)]


def test_parallel_rollouts_seed_initial_assistant_prefix(tmp_path: Path) -> None:
    """Structured roots should start from configured assistant prefix."""

    executor = build_executor(tmp_path=tmp_path)
    executor.decoding = DecodingConfig(
        initial_assistant_prefix="<think>\n<steer>",
        max_gen_toks=8,
        top_logprobs=0,
    )
    executor.set_event_context(
        doc_id=0,
        doc_attempt=0,
        task_name="novelty_bench",
        model_id="fake",
        selector_mode="structured_baseline",
    )

    tree, scheduled = executor._initialize_parallel_rollout_tree(
        rollout_count=2,
        branching_enabled=False,
    )

    assert tree.nodes["node_root"].assistant_prefix == "<think>\n<steer>"
    assert all(item.state.assistant_prefix == "<think>\n<steer>" for item in scheduled)
    executor.artifact_store.close()


def test_open_initial_steer_prefix_uses_steer_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Open steer prefixes should sample steer content before exec content."""

    executor = build_executor(tmp_path=tmp_path, trigger_entropy_enabled=False)
    executor.decoding = DecodingConfig(
        temperature=0.3,
        steer_temperature=0.8,
        top_p=0.95,
        steer_top_p=0.77,
        initial_assistant_prefix="<think>\n<steer>",
        max_gen_toks=32,
        top_logprobs=0,
    )
    observed_requests: list[tuple[str, float, float, str, tuple[str, ...] | None]] = []

    async def fake_request_completions(
        **kwargs: object,
    ) -> tuple[GenerationChoice, ...]:
        request_kind = kwargs["request_kind"]
        temperature = kwargs["temperature"]
        top_p = kwargs["top_p"]
        assistant_prefix = kwargs["assistant_prefix"]
        prompt_token_ids = kwargs["prompt_token_ids"]
        stop = kwargs["stop"]
        assert isinstance(request_kind, str)
        assert isinstance(temperature, (float, int))
        assert isinstance(top_p, (float, int))
        assert isinstance(assistant_prefix, str)
        assert isinstance(prompt_token_ids, tuple)
        assert stop is None or isinstance(stop, tuple)
        observed_requests.append(
            (request_kind, float(temperature), float(top_p), assistant_prefix, stop)
        )
        if request_kind == "steer_single_candidate":
            return (
                GenerationChoice(
                    index=0,
                    text="Plan first</steer>",
                    finish_reason="stop",
                    stop_reason="</steer>",
                    tokens=(
                        ParsedToken(
                            token="Plan first</steer>",
                            logprob=-0.1,
                            top_entries=(("x", -0.2),),
                        ),
                    ),
                    prompt_token_ids=(1, 2, 3),
                    token_ids=(42,),
                ),
            )
        if request_kind == "think_close_continuation":
            return (
                GenerationChoice(
                    index=0,
                    text=" so \\boxed{7}",
                    finish_reason="stop",
                    stop_reason=None,
                    tokens=(
                        ParsedToken(
                            token=" so \\boxed{7}",
                            logprob=-0.1,
                            top_entries=(("x", -0.2),),
                        ),
                    ),
                    prompt_token_ids=(1, 2, 3, 4),
                    token_ids=(44,),
                ),
            )
        assert request_kind == "decode_chunk"
        return (
            GenerationChoice(
                index=0,
                text="</think>\nFinal answer",
                finish_reason="stop",
                stop_reason=None,
                tokens=(
                    ParsedToken(
                        token="</think>\nFinal answer",
                        logprob=-0.1,
                        top_entries=(("x", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 3),
                token_ids=(43,),
            ),
        )

    monkeypatch.setattr(
        executor,
        "_request_completions_with_limit",
        fake_request_completions,
    )
    tree = _minimal_tree()

    outcome = executor._decode_until_event(
        tree=tree,
        state=PathState(
            node_id="node_root",
            assistant_prefix=executor.decoding.initial_assistant_prefix,
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
        branching_enabled=False,
        steer_normalization_enabled=True,
    )

    assert "</think>" in outcome.assistant_prefix
    assert observed_requests[0][0] == "steer_single_candidate"
    assert observed_requests[0][1] == 0.8
    assert observed_requests[0][2] == 0.77
    assert observed_requests[0][3].endswith("<steer>")
    assert observed_requests[0][4] == ("</steer",)
    assert observed_requests[1][0] == "decode_chunk"
    assert observed_requests[1][1] == 0.3
    assert observed_requests[1][2] == 0.95
    assert observed_requests[1][3].endswith("</steer>\n<exec>")
    executor.artifact_store.close()


def test_exec_decode_stops_at_exec_close_before_steer_decision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exec chunks should stop at `</exec` and let steer decode choose next tag."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=32)
    observed_requests: list[tuple[str, str, tuple[str, ...] | None]] = []

    async def fake_request_completions(
        **kwargs: object,
    ) -> tuple[GenerationChoice, ...]:
        request_kind = kwargs["request_kind"]
        assistant_prefix = kwargs["assistant_prefix"]
        stop = kwargs["stop"]
        assert isinstance(request_kind, str)
        assert isinstance(assistant_prefix, str)
        assert stop is None or isinstance(stop, tuple)
        observed_requests.append((request_kind, assistant_prefix, stop))
        if request_kind == "decode_chunk":
            return (
                GenerationChoice(
                    index=0,
                    text="Compute the value.</exec",
                    finish_reason="stop",
                    stop_reason="</exec",
                    tokens=(
                        ParsedToken(
                            token="Compute the value.</exec",
                            logprob=-0.1,
                            top_entries=(("x", -0.2),),
                        ),
                    ),
                    prompt_token_ids=(1, 2, 3),
                    token_ids=(41,),
                ),
            )
        if request_kind == "steer_single_candidate":
            return (
                GenerationChoice(
                    index=0,
                    text="</think>",
                    finish_reason="stop",
                    stop_reason="</think>",
                    tokens=(
                        ParsedToken(
                            token="</think>",
                            logprob=-0.1,
                            top_entries=(("x", -0.2),),
                        ),
                    ),
                    prompt_token_ids=(1, 2, 3, 4),
                    token_ids=(42,),
                ),
            )
        assert request_kind == "think_close_continuation"
        return (
            GenerationChoice(
                index=0,
                text="\n\\boxed{7}",
                finish_reason="stop",
                stop_reason=None,
                tokens=(
                    ParsedToken(
                        token="\n\\boxed{7}",
                        logprob=-0.1,
                        top_entries=(("x", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 3, 4, 5),
                token_ids=(43,),
            ),
        )

    monkeypatch.setattr(
        executor,
        "_request_completions_with_limit",
        fake_request_completions,
    )

    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="<think><steer>Plan</steer>\n<exec>",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
        branching_enabled=False,
        steer_normalization_enabled=True,
    )

    assert observed_requests[0][0] == "decode_chunk"
    assert observed_requests[0][2] == ("</exec",)
    assert observed_requests[1][0] == "steer_single_candidate"
    assert observed_requests[1][1].endswith("</exec>\n")
    assert "</exec</exec>" not in observed_requests[1][1]
    assert not observed_requests[1][1].endswith("<steer>")
    assert observed_requests[1][2] == ("</steer", "</think>")
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "think_end"
    assert "</exec>\n</think>" in outcome.assistant_prefix
    executor.artifact_store.close()


def test_post_exec_malformed_steer_decision_logs_tree_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed post-exec steer decisions should be visible in tree events."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=32)

    async def fake_request_completions(
        **kwargs: object,
    ) -> tuple[GenerationChoice, ...]:
        request_kind = kwargs["request_kind"]
        assert isinstance(request_kind, str)
        if request_kind == "decode_chunk":
            return (
                GenerationChoice(
                    index=0,
                    text="Compute the value.",
                    finish_reason="stop",
                    stop_reason="</exec",
                    tokens=(
                        ParsedToken(
                            token="Compute the value.",
                            logprob=-0.1,
                            top_entries=(("x", -0.2),),
                        ),
                    ),
                    prompt_token_ids=(1, 2, 3),
                    token_ids=(41,),
                ),
            )
        assert request_kind == "steer_single_candidate"
        return (
            GenerationChoice(
                index=0,
                text="Continue without choosing a tag.",
                finish_reason="length",
                stop_reason=None,
                tokens=(
                    ParsedToken(
                        token="Continue without choosing a tag.",
                        logprob=-0.1,
                        top_entries=(("x", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 3, 4),
                token_ids=(42,),
            ),
        )

    monkeypatch.setattr(
        executor,
        "_request_completions_with_limit",
        fake_request_completions,
    )

    tree = _minimal_tree()
    outcome = executor._decode_until_event(
        tree=tree,
        state=PathState(
            node_id="node_root",
            assistant_prefix="<think><steer>Plan</steer>\n<exec>",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
        branching_enabled=False,
        steer_normalization_enabled=True,
    )

    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "missing_steer_or_think_close"
    assert outcome.assistant_prefix.endswith(
        "</exec>\nContinue without choosing a tag."
    )
    assert 42 in outcome.token_ids
    malformed_events = [
        row
        for row in read_tree_event_rows(store=executor.artifact_store)
        if row["event_type"] == "malformed_steer_decision"
    ]
    assert len(malformed_events) == 1
    payload = malformed_events[0]["payload"]
    assert payload["node_id"] == "node_root"
    assert payload["source"] == "explicit_stop_nonbranch"
    assert payload["stop_reason"] == "missing_steer_or_think_close"
    assert payload["assistant_prefix_tail"].endswith("</exec>\n")
    assert payload["candidate_text"] == "Continue without choosing a tag."
    executor.artifact_store.close()


def test_post_exec_steer_decision_without_tag_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-exec steer generation must produce `<steer>` or `</think>`."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=32)

    async def fake_generate_choice_async(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = prompt_token_ids, max_tokens, n, kwargs
        assert assistant_prefix.endswith("</exec>\n")
        assert stop == ("</steer", "</think>")
        return GenerationChoice(
            index=0,
            text="Continue without a tag.",
            finish_reason="length",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="Continue without a tag.",
                    logprob=-0.1,
                    top_entries=(("x", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(99,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice_async)

    outcome = asyncio.run(
        continue_with_single_steer_candidate_async(
            executor=executor,
            assistant_prefix="<think><steer>Plan</steer>\n<exec>Work</exec>\n",
            prompt_token_ids=(1, 2),
            token_ids=(10,),
            token_traces=(),
            generated_tokens=1,
            request_stream_id="decode:node_root",
        )
    )

    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "missing_steer_or_think_close"
    assert outcome.assistant_prefix.endswith("</exec>\nContinue without a tag.")
    assert outcome.token_ids == (10, 99)
    assert outcome.steer_phase_token_spans == ((1, 2),)
    executor.artifact_store.close()


def test_post_exec_steer_decision_with_text_before_tag_terminates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-exec steer decisions must start with `<steer>` or `</think>`."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=32)

    async def fake_generate_choice_async(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = prompt_token_ids, max_tokens, n, kwargs
        assert assistant_prefix.endswith("</exec>\n")
        assert stop == ("</steer", "</think>")
        return GenerationChoice(
            index=0,
            text="I should keep going. <steer>Plan</steer>",
            finish_reason="stop",
            stop_reason="</steer",
            tokens=(
                ParsedToken(
                    token="I should keep going. <steer>Plan</steer>",
                    logprob=-0.1,
                    top_entries=(("x", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(99,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice_async)

    outcome = asyncio.run(
        continue_with_single_steer_candidate_async(
            executor=executor,
            assistant_prefix="<think><steer>Plan</steer>\n<exec>Work</exec>\n",
            prompt_token_ids=(1, 2),
            token_ids=(10,),
            token_traces=(),
            generated_tokens=1,
            request_stream_id="decode:node_root",
        )
    )

    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "missing_steer_or_think_close"
    assert outcome.assistant_prefix.endswith(
        "</exec>\nI should keep going. <steer>Plan</steer>"
    )
    assert outcome.token_ids == (10, 99)
    assert outcome.steer_phase_token_spans == ((1, 2),)
    executor.artifact_store.close()


def test_branch_selected_malformed_steer_text_becomes_terminal_leaf(
    tmp_path: Path,
) -> None:
    """Malformed selected branch text should be kept on the terminal leaf."""

    executor = build_executor(tmp_path=tmp_path)
    tree = _minimal_tree()
    outcome = DecodeOutcome(
        event_type="trigger",
        trigger_type="steer_boundary",
        assistant_prefix="<think><steer>Plan</steer>\n<exec>Work</exec>\n",
        prompt_token_ids=(1, 2),
        token_ids=(10,),
        token_traces=(),
        generated_tokens=1,
        stop_reason="",
        steer_phase_token_spans=((0, 1),),
    )
    pool = CandidatePoolRecord(
        candidate_pool_id="pool_bad",
        branch_point_id="bp_bad",
        node_id="node_root",
        trigger_type="steer_boundary",
        candidates=(
            CandidateRecord(
                candidate_id=7,
                text="Continue without a steer tag.",
                token_ids=(99,),
                tokens=(
                    TokenTrace(
                        token_index=0,
                        token_id=99,
                        token_text="Continue without a steer tag.",
                        logprob=-0.1,
                        probability=0.9,
                    ),
                ),
                finish_reason="length",
                stop_reason=None,
            ),
        ),
    )

    children = asyncio.run(
        executor._expand_children_async(
            tree=tree,
            parent_state=PathState(
                node_id="node_root",
                assistant_prefix=outcome.assistant_prefix,
                prompt_token_ids=outcome.prompt_token_ids,
                token_ids=outcome.token_ids,
                token_traces=outcome.token_traces,
                branch_points_used=0,
                steer_phase_token_spans=outcome.steer_phase_token_spans,
            ),
            outcome=outcome,
            pool=pool,
            selected_ids=(7,),
        )
    )

    assert children == []
    assert len(tree.leaves) == 1
    leaf = tree.leaves[0]
    assert leaf.text.endswith("</exec>\nContinue without a steer tag.")
    assert leaf.token_ids == (10, 99)
    assert leaf.steer_phase_token_spans == ((0, 1), (1, 2))
    malformed_events = [
        row
        for row in read_tree_event_rows(store=executor.artifact_store)
        if row["event_type"] == "malformed_steer_decision"
    ]
    assert malformed_events[-1]["payload"]["candidate_text"] == (
        "Continue without a steer tag."
    )
    executor.artifact_store.close()


def test_think_close_continuation_resets_stale_prefix_chain(tmp_path: Path) -> None:
    """Post-think continuation should not reuse stale decode stream bases."""

    executor = build_executor(tmp_path=tmp_path)
    executor.decoding = replace(executor.decoding, max_gen_toks=64)
    stream_id = "decode:node_test"
    executor._request_stream_state[stream_id] = (
        branch_executor_module._RequestStreamState(
            request_id="req_old",
            input_token_ids=(1, 2, 3),
            output_token_ids=(4, 5, 6),
        )
    )

    outcome = asyncio.run(
        continue_after_think_close_async(
            executor=executor,
            assistant_prefix="reasoning</think>",
            prompt_token_ids=(9,),
            token_ids=(),
            token_traces=(),
            generated_tokens=0,
            request_stream_id=stream_id,
        )
    )

    assert outcome.event_type == "terminated"
    assert executor._request_stream_state[stream_id].request_id != "req_old"
    executor.artifact_store.close()


def test_think_close_continuation_caps_remaining_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-think continuation should not request beyond vLLM context length."""

    executor = build_executor(tmp_path=tmp_path)
    executor.decoding = replace(
        executor.decoding,
        max_gen_toks=64,
        max_model_len=10,
    )
    observed_max_tokens: list[int] = []

    async def fake_generate_choice_async(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, stop, n, kwargs
        assert prompt_token_ids == tuple(range(8))
        observed_max_tokens.append(max_tokens)
        return GenerationChoice(
            index=0,
            text="ok",
            finish_reason="length",
            stop_reason=None,
            tokens=(
                ParsedToken(token="o", logprob=-0.1, top_entries=(("o", -0.1),)),
                ParsedToken(token="k", logprob=-0.2, top_entries=(("k", -0.2),)),
            ),
            prompt_token_ids=prompt_token_ids,
            token_ids=(101, 102),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice_async)

    outcome = asyncio.run(
        continue_after_think_close_async(
            executor=executor,
            assistant_prefix="reasoning</think>",
            prompt_token_ids=tuple(range(8)),
            token_ids=(),
            token_traces=(),
            generated_tokens=20,
            request_stream_id="decode:node_test",
        )
    )

    assert observed_max_tokens == [2]
    assert outcome.stop_reason == "max_context_length_reached"
    executor.artifact_store.close()


def test_think_close_continuation_caps_post_think_tokens(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-think answer continuation should have a local 2K token cap."""

    executor = build_executor(tmp_path=tmp_path)
    executor.decoding = replace(
        executor.decoding,
        max_gen_toks=4096,
        max_model_len=None,
    )
    observed_max_tokens: list[int] = []

    async def fake_generate_choice_async(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, stop, n, kwargs
        observed_max_tokens.append(max_tokens)
        return GenerationChoice(
            index=0,
            text=" answer",
            finish_reason="length",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token=" answer", logprob=-0.1, top_entries=((" answer", -0.1),)
                ),
            ),
            prompt_token_ids=prompt_token_ids,
            token_ids=(101,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice_async)

    outcome = asyncio.run(
        continue_after_think_close_async(
            executor=executor,
            assistant_prefix="reasoning</think>",
            prompt_token_ids=(9,),
            token_ids=(),
            token_traces=(),
            generated_tokens=0,
            request_stream_id="decode:node_test",
        )
    )

    assert observed_max_tokens == [2048]
    assert outcome.stop_reason == "post_think_toks_reached"
    executor.artifact_store.close()


def test_think_close_continuation_skips_full_context_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Post-think continuation should terminate when no context remains."""

    executor = build_executor(tmp_path=tmp_path)
    executor.decoding = replace(
        executor.decoding,
        max_gen_toks=64,
        max_model_len=8,
    )
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
        raise AssertionError("continuation request should not be sent")

    monkeypatch.setattr(executor, "_generate_choice_async", fail_generate_choice_async)

    outcome = asyncio.run(
        continue_after_think_close_async(
            executor=executor,
            assistant_prefix="reasoning</think>",
            prompt_token_ids=tuple(range(8)),
            token_ids=(),
            token_traces=(),
            generated_tokens=20,
            request_stream_id="decode:node_test",
        )
    )

    assert call_count["decode_calls"] == 0
    assert outcome.stop_reason == "max_context_length_reached"
    executor.artifact_store.close()


def test_greedy_standard_rollouts_split_into_single_choice_requests(
    tmp_path: Path,
) -> None:
    """Greedy baseline exports should avoid vLLM's `n > 1` restriction."""

    executor = build_executor(tmp_path=tmp_path)
    executor.decoding = DecodingConfig(
        temperature=0.0,
        top_p=0.95,
        max_gen_toks=8,
        top_logprobs=5,
    )
    executor.set_event_context(
        doc_id=0,
        doc_attempt=0,
        task_name="novelty_bench",
        model_id="fake",
        selector_mode="baseline",
    )

    leaves = asyncio.run(executor.run_standard_rollouts_async(rollout_count=3))
    fake_client = cast(FakeClient, executor.client)

    assert len(leaves) == 3
    assert fake_client.decode_calls == 3


def test_shared_pool_records_all_requested_selectors(
    tmp_path: Path, monkeypatch
) -> None:
    """Each branch point should persist selection outcomes for all requested modes."""

    executor = build_executor(tmp_path=tmp_path)
    executor.requested_selectors = (
        "cluster_across",
        "embed_diverse_topk_random",
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
                selector_mode="embed_diverse_topk_random",
                selected_candidate_ids=(2, 3),
            ),
            SelectionOutcome(
                selector_mode="within_cluster", selected_candidate_ids=(4, 5)
            ),
            SelectionOutcome(selector_mode="random", selected_candidate_ids=(6, 7)),
        )

    async def fake_select_all_modes_async(**_: object):
        return fake_select_all_modes()

    monkeypatch.setattr(
        "branching_eval.branch_executor.select_candidates_all_modes",
        fake_select_all_modes,
    )
    monkeypatch.setattr(
        "branching_eval.branch_executor.select_candidates_all_modes_async",
        fake_select_all_modes_async,
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

    async def fake_select_all_modes_async(**_: object):
        return fake_select_all_modes()

    monkeypatch.setattr(
        "branching_eval.branch_executor.select_candidates_all_modes",
        fake_select_all_modes,
    )
    monkeypatch.setattr(
        "branching_eval.branch_executor.select_candidates_all_modes_async",
        fake_select_all_modes_async,
    )
    _ = executor.run_branching_rollouts(doc_id=10, task_name="aime24", model_id="m")
    rows = read_tree_event_rows(store=executor.artifact_store)
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


def test_leaf_budget_caps_realized_leaf_count(tmp_path: Path) -> None:
    """Runtime leaf budgets should cap realized leaves below the full tree capacity."""

    executor = build_executor(tmp_path=tmp_path)
    tree = executor.run_branching_rollouts(
        doc_id=1,
        task_name="aime24",
        model_id="m",
        leaf_budget=2,
    )

    rows = read_tree_event_rows(store=executor.artifact_store)
    rollout_started = next(
        row for row in rows if row.get("event_type") == "rollout_started"
    )
    assert rollout_started["payload"]["leaf_limit"] == 2
    assert len(tree.leaves) == 2


def test_leaf_budget_keeps_underproduced_no_trigger_rollouts_ragged(
    tmp_path: Path, monkeypatch
) -> None:
    """No-trigger runs should keep their single realized leaf instead of padding to budget."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=False,
        trigger_entropy_enabled=False,
    )

    async def fake_decode_until_event_async(
        *,
        tree: BranchTree,
        state: PathState,
        branching_enabled: bool = True,
        steer_normalization_enabled: bool | None = None,
    ) -> DecodeOutcome:
        _ = tree, state, branching_enabled, steer_normalization_enabled
        return _decode_outcome(
            event_type="terminated",
            assistant_prefix="done",
            token_ids=(11, 12),
            generated_tokens=2,
            stop_reason="model_finished",
        )

    monkeypatch.setattr(
        executor, "_decode_until_event_async", fake_decode_until_event_async
    )
    tree = executor.run_branching_rollouts(
        doc_id=2,
        task_name="aime24",
        model_id="m",
        leaf_budget=2,
    )

    rows = read_tree_event_rows(store=executor.artifact_store)
    rollout_started = next(
        row for row in rows if row.get("event_type") == "rollout_started"
    )
    rollout_finished = next(
        row for row in rows if row.get("event_type") == "rollout_finished"
    )
    assert rollout_started["payload"]["leaf_limit"] == 2
    assert len(tree.branch_points) == 0
    assert len(tree.leaves) == 1
    assert rollout_finished["payload"]["leaf_count"] == 1


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
        inline_epsilon_enabled: bool | None = None,
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
            assert inline_epsilon_enabled is True
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
    rows = read_tree_event_rows(store=executor.artifact_store)
    assert any(
        row.get("event_type") == "trigger_skipped_max_branch_points" for row in rows
    )


def test_epsilon_greedy_roots_skip_true_branch_scheduler(
    tmp_path: Path, monkeypatch
) -> None:
    """Epsilon-greedy roots should schedule inline exploration only."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=False,
        trigger_entropy_enabled=False,
    )
    executor.allow_true_branching = False
    executor.branching = replace(executor.branching, epsilon_greedy_prob=0.1)
    executor.set_event_context(
        doc_id=7,
        doc_attempt=0,
        task_name="task",
        model_id="model",
        selector_mode="random",
    )
    captured_scheduled: list[Any] = []

    async def fake_decode_frontier_streaming_async(
        *,
        tree: BranchTree,
        frontier: list[PathState],
        doc_id: int,
        leaf_limit: int,
        initial_scheduled: list[Any] | None = None,
    ) -> None:
        _ = tree, leaf_limit
        assert doc_id == 7
        assert frontier == []
        assert initial_scheduled is not None
        captured_scheduled.extend(initial_scheduled)

    monkeypatch.setattr(
        executor,
        "_decode_frontier_streaming_async",
        fake_decode_frontier_streaming_async,
    )

    tree = asyncio.run(executor.run_epsilon_greedy_rollouts_async(rollout_count=3))

    assert len(captured_scheduled) == 3
    assert len({scheduled.state.node_id for scheduled in captured_scheduled}) == 3
    assert all(not scheduled.branching_enabled for scheduled in captured_scheduled)
    assert all(
        scheduled.steer_normalization_enabled is True
        for scheduled in captured_scheduled
    )
    assert all(
        scheduled.inline_epsilon_enabled is True for scheduled in captured_scheduled
    )
    _ = tree
    rows = read_tree_event_rows(store=executor.artifact_store)
    assert rows[-1]["event_type"] == "rollout_finished"
    executor.artifact_store.close()


def test_streaming_scheduler_cancels_pending_tasks_after_decode_error(
    tmp_path: Path, monkeypatch
) -> None:
    """Decode scheduler should drain siblings when one task raises."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=False,
        trigger_entropy_enabled=False,
    )
    tree = _minimal_tree()
    frontier = [
        _path_state(node_id="node_fails", branch_points_used=0),
        _path_state(node_id="node_waits", branch_points_used=0),
    ]
    cancelled_states: list[str] = []

    async def fake_decode_until_event(
        *,
        tree: BranchTree,
        state: PathState,
        branching_enabled: bool = True,
        steer_normalization_enabled: bool | None = None,
        inline_epsilon_enabled: bool | None = None,
    ) -> DecodeOutcome:
        _ = tree, branching_enabled, steer_normalization_enabled, inline_epsilon_enabled
        if state.node_id == "node_fails":
            await asyncio.sleep(0.01)
            raise VllmRequestError("server disconnected")
        assert state.node_id == "node_waits"
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            cancelled_states.append(state.node_id)
            raise
        raise AssertionError("waiting decode should have been cancelled")

    monkeypatch.setattr(executor, "_decode_until_event_async", fake_decode_until_event)

    with pytest.raises(VllmRequestError, match="server disconnected"):
        asyncio.run(
            executor._decode_frontier_streaming_async(
                tree=tree,
                frontier=frontier,
                doc_id=0,
                leaf_limit=16,
            )
        )

    assert cancelled_states == ["node_waits"]


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
        inline_epsilon_enabled: bool | None = None,
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
        assert inline_epsilon_enabled is True
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


def test_streaming_scheduler_caps_shared_inflight_branch_tasks(
    tmp_path: Path, monkeypatch
) -> None:
    """Shared branch semaphore should cap decode work across executors."""

    active_task_count = 0
    max_active_task_count = 0

    async def fake_decode_until_event(
        *,
        tree: BranchTree,
        state: PathState,
        branching_enabled: bool = True,
        steer_normalization_enabled: bool | None = None,
    ) -> DecodeOutcome:
        nonlocal active_task_count, max_active_task_count
        _ = tree, state, branching_enabled, steer_normalization_enabled
        active_task_count += 1
        max_active_task_count = max(max_active_task_count, active_task_count)
        await asyncio.sleep(0.02)
        active_task_count -= 1
        return _decode_outcome(
            event_type="terminated",
            assistant_prefix="done",
            token_ids=(1,),
            generated_tokens=1,
            stop_reason="model_finished",
        )

    async def run_both_executors() -> None:
        branch_task_semaphore = asyncio.Semaphore(2)
        executor_a = build_executor(
            tmp_path=tmp_path / "run_a",
            branch_task_semaphore=branch_task_semaphore,
            trigger_steer_enabled=False,
            trigger_entropy_enabled=False,
        )
        executor_b = build_executor(
            tmp_path=tmp_path / "run_b",
            branch_task_semaphore=branch_task_semaphore,
            trigger_steer_enabled=False,
            trigger_entropy_enabled=False,
        )
        tree_a = _minimal_tree()
        tree_b = _minimal_tree()
        frontier_a = [
            _path_state(node_id=f"node_a_{index}", branch_points_used=0)
            for index in range(3)
        ]
        frontier_b = [
            _path_state(node_id=f"node_b_{index}", branch_points_used=0)
            for index in range(3)
        ]
        monkeypatch.setattr(
            executor_a, "_decode_until_event_async", fake_decode_until_event
        )
        monkeypatch.setattr(
            executor_b, "_decode_until_event_async", fake_decode_until_event
        )
        await asyncio.gather(
            executor_a._decode_frontier_streaming_async(
                tree=tree_a,
                frontier=frontier_a,
                doc_id=0,
                leaf_limit=16,
            ),
            executor_b._decode_frontier_streaming_async(
                tree=tree_b,
                frontier=frontier_b,
                doc_id=1,
                leaf_limit=16,
            ),
        )
        assert len(tree_a.leaves) == 3
        assert len(tree_b.leaves) == 3

    asyncio.run(run_both_executors())
    assert max_active_task_count == 2


def test_baseline_rollout_count_is_n(tmp_path: Path) -> None:
    """Baseline mode should return exactly N rollouts."""

    executor = build_executor(tmp_path=tmp_path)
    leaves = executor.run_standard_rollouts(rollout_count=16)
    assert len(leaves) == 16


def test_branching_pool_regenerates_without_cache(tmp_path: Path) -> None:
    """Candidate pool should be regenerated on each rollout."""

    executor = build_executor(tmp_path=tmp_path)
    assert isinstance(executor.client, FakeClient)
    _ = executor.run_branching_rollouts(doc_id=1, task_name="aime24", model_id="m")
    first_calls = executor.client.candidate_calls
    _ = executor.run_branching_rollouts(doc_id=1, task_name="aime24", model_id="m")
    assert executor.client.candidate_calls > first_calls


def test_branching_writes_incremental_tree_events_sqlite(tmp_path: Path) -> None:
    """Branching run should append live tree events to SQLite."""

    executor = build_executor(tmp_path=tmp_path)
    tree = executor.run_branching_rollouts(doc_id=3, task_name="aime24", model_id="m")
    executor.artifact_store.flush_events()
    assert (tmp_path / "run" / "tree_events.sqlite").exists()
    assert not (tmp_path / "run" / "tree_events.jsonl").exists()
    assert (tmp_path / "run" / "viz" / "start_viz.sh").exists()
    rows = read_tree_event_rows(store=executor.artifact_store)
    assert rows
    event_types = {str(row.get("event_type", "")) for row in rows}
    assert "rollout_started" in event_types
    assert "node_created" in event_types
    assert "trigger_fired" in event_types
    assert "decode_chunk" in event_types
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
    assert "text" not in first_candidate
    assert "text_preview" in first_candidate


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
        prefix_after="<exec>plan\n<steer></steer></exec>\n<steer>",
        prompt_token_ids_before=(9, 8),
        prompt_token_ids_after=(9, 8, 50, 51, 52, 53),
        token_ids_before=(1, 2, 3),
        token_ids_after=(1, 2, 3, 42, 43, 44),
    )
    rows = read_tree_event_rows(store=executor.artifact_store)
    assert rows
    payload = rows[-1]["payload"]
    assert payload["chunk_text"] == "<steer></steer></exec>\n<steer>"
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


def test_selected_ids_for_diverse_topk_random_does_not_backfill(tmp_path: Path) -> None:
    """OpenAI diverse-top-k selector should not backfill beyond chosen ids."""

    executor = build_executor(tmp_path=tmp_path)
    executor.active_selector = "embed_diverse_topk_random"
    pool = sample_pool(candidate_count=4)

    selected = executor._selected_ids_for_branch(pool=pool, selected_ids=(1,))

    assert selected == (1,)


def test_candidate_pool_generation_defers_alignment_until_selection(
    tmp_path: Path, monkeypatch
) -> None:
    """Candidate pools should skip detokenization until a selected child is normalized."""

    executor = build_executor(tmp_path=tmp_path)
    detokenize_calls: list[tuple[int, ...]] = []

    async def fake_generate_many_async(**_: object) -> tuple[GenerationChoice, ...]:
        return (
            make_choice(index=0, text="raw_0", entropy_logprob=-0.4),
            make_choice(index=1, text="raw_1", entropy_logprob=-0.4),
        )

    async def fake_detokenize_async(*, model: str, token_ids: tuple[int, ...]) -> str:
        _ = model
        detokenize_calls.append(token_ids)
        return f"decoded_{token_ids[0]}"

    monkeypatch.setattr(executor, "_generate_many_async", fake_generate_many_async)
    monkeypatch.setattr(
        executor.client,
        "detokenize_async",
        fake_detokenize_async,
        raising=False,
    )

    pool = asyncio.run(
        executor._generate_candidate_pool_async(
            candidate_pool_id="pool",
            state=_path_state(node_id="root"),
            trigger_type="steer_boundary",
            entropy_value=None,
            assistant_prefix="seed",
            prompt_token_ids=(1, 2, 3),
            candidate_token_budget=4,
        )
    )

    assert detokenize_calls == []
    aligned_candidate = asyncio.run(
        executor._candidate_with_aligned_text_async(candidate=pool.candidates[1])
    )

    assert detokenize_calls == [(11,)]
    assert aligned_candidate.text == "decoded_11"
    assert aligned_candidate.token_ids == (11,)


def test_candidate_pool_after_exec_allows_terminal_think_choice(
    tmp_path: Path, monkeypatch
) -> None:
    """Candidate generation after exec should let the model choose `</think>`."""

    executor = build_executor(tmp_path=tmp_path)
    calls: list[dict[str, object]] = []

    async def fake_generate_many_async(
        **kwargs: object,
    ) -> tuple[GenerationChoice, ...]:
        calls.append(dict(kwargs))
        return (
            make_choice(
                index=0, text="<steer>try factoring</steer>", entropy_logprob=-0.4
            ),
            GenerationChoice(
                index=1,
                text="",
                finish_reason="stop",
                stop_reason="</think>",
                tokens=(),
                prompt_token_ids=(1, 2, 3),
                token_ids=(),
            ),
        )

    monkeypatch.setattr(executor, "_generate_many_async", fake_generate_many_async)

    pool = asyncio.run(
        executor._generate_candidate_pool_async(
            candidate_pool_id="pool",
            state=_path_state(node_id="root"),
            trigger_type="steer_boundary",
            entropy_value=None,
            assistant_prefix="<exec>work",
            prompt_token_ids=(1, 2, 3),
            candidate_token_budget=4,
        )
    )

    assert calls[0]["assistant_prefix"] == "<exec>work\n</exec>\n"
    assert calls[0]["stop"] == ("</steer", "</think>")
    assert pool.candidates[0].text.startswith("<steer>")
    assert pool.candidates[1].text == "</think>"


def test_tokenize_helpers_share_executor_cache(tmp_path: Path, monkeypatch) -> None:
    """Sync and async tokenizer helpers should reuse one executor-local cache."""

    executor = build_executor(tmp_path=tmp_path)
    tokenize_calls: list[str] = []
    tokenize_async_calls: list[str] = []
    expected_text = "</steer>\n<exec>"

    def fake_tokenize(
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        _ = model, add_special_tokens
        tokenize_calls.append(text)
        return (7, 8, 9)

    async def fake_tokenize_async(
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        _ = model, add_special_tokens
        tokenize_async_calls.append(text)
        return (70, 80, 90)

    monkeypatch.setattr(executor.client, "tokenize", fake_tokenize, raising=False)
    monkeypatch.setattr(
        executor.client,
        "tokenize_async",
        fake_tokenize_async,
        raising=False,
    )

    assert executor._tokenize_text(text=expected_text) == (7, 8, 9)
    assert asyncio.run(executor._tokenize_text_async(text=expected_text)) == (
        7,
        8,
        9,
    )
    assert tokenize_calls == [expected_text]
    assert tokenize_async_calls == []


def test_detokenize_helpers_share_executor_cache(tmp_path: Path, monkeypatch) -> None:
    """Sync and async detokenizer helpers should reuse one executor-local cache."""

    executor = build_executor(tmp_path=tmp_path)
    detokenize_calls: list[tuple[int, ...]] = []
    detokenize_async_calls: list[tuple[int, ...]] = []
    expected_token_ids = (11, 12, 13)

    def fake_detokenize(*, model: str, token_ids: tuple[int, ...]) -> str:
        _ = model
        detokenize_calls.append(token_ids)
        return "decoded_sync"

    async def fake_detokenize_async(
        *,
        model: str,
        token_ids: tuple[int, ...],
    ) -> str:
        _ = model
        detokenize_async_calls.append(token_ids)
        return "decoded_async"

    monkeypatch.setattr(executor.client, "detokenize", fake_detokenize, raising=False)
    monkeypatch.setattr(
        executor.client,
        "detokenize_async",
        fake_detokenize_async,
        raising=False,
    )

    assert executor._detokenize_ids(token_ids=expected_token_ids) == "decoded_sync"
    assert (
        asyncio.run(executor._detokenize_ids_async(token_ids=expected_token_ids))
        == "decoded_sync"
    )
    assert detokenize_calls == [expected_token_ids]
    assert detokenize_async_calls == []


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
        candidate_pool_id="pool_00000001",
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
        candidate_token_budget=3,
    )
    assert captured_prefix["stop"] == "</steer"
    assert isinstance(captured_prefix["assistant_prefix"], str)
    assert captured_prefix["assistant_prefix"].endswith("</steer></exec>\n<steer>")
    assert captured_prefix["temperature"] == -1.0
    normalized_prefix = str(captured_prefix["assistant_prefix"])
    injected_suffix = normalized_prefix[len("<exec>abc<steer") :]
    expected_suffix_token_ids = tuple(range(1000, 1000 + len(injected_suffix)))
    assert observed_prompt_token_ids == (1, 2, 3) + expected_suffix_token_ids


def test_steer_pool_generation_defers_normalization_for_open_steer(
    tmp_path: Path, monkeypatch
) -> None:
    """Candidate pools should sample steer text before repairing open steer tags."""

    executor = build_executor(tmp_path=tmp_path)
    captured_prefix: dict[str, object] = {}

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
        _ = max_tokens, temperature, kwargs
        captured_prefix["assistant_prefix"] = assistant_prefix
        captured_prefix["prompt_token_ids"] = prompt_token_ids
        captured_prefix["stop"] = stop
        return tuple(
            GenerationChoice(
                index=index,
                text=f"candidate {index}</steer>",
                finish_reason="stop",
                stop_reason="</steer>",
                tokens=(),
                prompt_token_ids=(1, 2, 1000),
                token_ids=(10 + index,),
            )
            for index in range(n)
        )

    monkeypatch.setattr(executor, "_generate_many_async", fake_generate_many)

    pool = executor._generate_candidate_pool(
        candidate_pool_id="pool_000000_open",
        state=PathState(
            node_id="node_s",
            assistant_prefix="<steer",
            prompt_token_ids=(1, 2),
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
        trigger_type="steer_boundary",
        entropy_value=None,
        assistant_prefix="<steer",
        prompt_token_ids=(1, 2),
        candidate_token_budget=3,
    )

    assert captured_prefix == {
        "assistant_prefix": "<steer>",
        "prompt_token_ids": (1, 2, 1000),
        "stop": ("</steer",),
    }
    assert pool.candidates[0].text == "candidate 0</steer>"
    assert "<steer></steer><exec></exec>" not in str(
        captured_prefix["assistant_prefix"]
    )


def test_steer_prefix_normalization_asserts_detokenized_alignment(
    tmp_path: Path, monkeypatch
) -> None:
    """Prefix normalization should assert when suffix detokenization mismatches."""

    executor = build_executor(
        tmp_path=tmp_path,
        debug_assert_text_token_alignment=True,
    )

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
        candidate_pool_id="pool_00000002",
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
        candidate_token_budget=3,
    )
    assert pool.candidates[0].text == "abc</steer>"
    assert pool.candidates[0].token_ids == (10, 11, 12)
    assert len(pool.candidates[0].tokens) == 1


def test_steer_pool_candidates_trim_text_after_close(
    tmp_path: Path, monkeypatch
) -> None:
    """Steer candidate generation should trim text after first `</steer>`."""

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
    pool = executor._generate_candidate_pool(
        candidate_pool_id="pool_00000003",
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
        candidate_token_budget=3,
    )

    assert pool.candidates[0].text == "abc</steer>"
    assert pool.candidates[0].token_ids == ()
    assert pool.candidates[0].tokens == ()


def test_steer_pool_candidates_preserve_token_payload_on_boundary_trim(
    tmp_path: Path, monkeypatch
) -> None:
    """Steer candidate trimming should keep tokens when trim hits a boundary."""

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
                        token="abc</steer>",
                        logprob=-0.1,
                        top_entries=(("x", -0.2),),
                    ),
                    ParsedToken(
                        token="tail",
                        logprob=-0.2,
                        top_entries=(("y", -0.3),),
                    ),
                ),
                prompt_token_ids=(1, 2),
                token_ids=(10, 11),
            ),
        ) * 100

    monkeypatch.setattr(executor, "_generate_many_async", fake_generate_many)
    pool = executor._generate_candidate_pool(
        candidate_pool_id="pool_00000004",
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
        candidate_token_budget=3,
    )

    assert pool.candidates[0].text == "abc</steer>"
    assert pool.candidates[0].token_ids == (10,)
    assert tuple(token.token_text for token in pool.candidates[0].tokens) == (
        "abc</steer>",
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
    assert observed_stop["stop"] == ("</exec",)


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
            stop_reason="</exec",
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
    tree = _minimal_tree()
    outcome = executor._decode_until_event(
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
    assert outcome.event_type == "trigger"
    assert outcome.trigger_type == "steer_boundary"


def test_steer_stop_reason_without_logprob_trace_continues(
    tmp_path: Path, monkeypatch
) -> None:
    """Explicit steer stops should not become empty generations at top_logprobs=0."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = DecodingConfig(
        temperature=executor.decoding.temperature,
        top_p=executor.decoding.top_p,
        max_gen_toks=branch_executor_module.REPEAT_TERMINATION_MIN_GENERATED_TOKENS
        + 100,
        top_logprobs=executor.decoding.top_logprobs,
    )
    observed_continue_args: dict[str, object] = {}

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
        return GenerationChoice(
            index=0,
            text="plain<steer",
            finish_reason="stop",
            stop_reason="</exec",
            tokens=(),
            prompt_token_ids=(1, 2, 3),
            token_ids=(42, 43),
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
        _ = executor, assistant_prefix, prompt_token_ids, token_traces
        observed_continue_args["token_ids"] = token_ids
        observed_continue_args["generated_tokens"] = generated_tokens
        observed_continue_args["request_stream_id"] = request_stream_id
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix="plain<steer></steer>\n<exec>done",
            prompt_token_ids=(1, 2, 3, 42, 43, 99),
            token_ids=token_ids + (99,),
            token_traces=token_traces,
            generated_tokens=generated_tokens + 1,
            stop_reason="model_finished",
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

    assert observed_continue_args["token_ids"] == (42, 43)
    assert observed_continue_args["generated_tokens"] == 2
    assert outcome.stop_reason == "model_finished"
    assert outcome.generated_tokens == 3


def test_steer_think_close_natural_finish_continues_to_answer(
    tmp_path: Path, monkeypatch
) -> None:
    """A bare natural `</think>` close should continue into answer generation."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=64)
    observed_stops: list[tuple[str, ...] | None] = []
    call_count = {"count": 0}
    observed_prefixes: list[str] = []

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = prompt_token_ids, max_tokens, n, kwargs
        observed_prefixes.append(assistant_prefix)
        observed_stops.append(stop)
        call_count["count"] += 1
        if call_count["count"] == 2:
            return GenerationChoice(
                index=0,
                text="\n\n\\boxed{7}",
                finish_reason="stop",
                stop_reason=None,
                tokens=(
                    ParsedToken(
                        token="\n\n\\boxed{7}",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 42),
                token_ids=(43,),
            )
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
    tree = _minimal_tree()
    outcome = executor._decode_until_event(
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
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "think_end"
    assert observed_stops[-1] is None
    assert observed_prefixes[-1].endswith("</think>")
    assert outcome.assistant_prefix.endswith("</think>\n\n\\boxed{7}")
    assert outcome.generated_tokens == 2


def test_steer_think_close_im_end_stop_does_not_continue(
    tmp_path: Path, monkeypatch
) -> None:
    """A model EOS after `</think>` should not force answer continuation."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=64)
    observed_stops: list[tuple[str, ...] | None] = []
    call_count = {"count": 0}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, n, kwargs
        observed_stops.append(stop)
        call_count["count"] += 1
        return GenerationChoice(
            index=0,
            text="</think>",
            finish_reason="stop",
            stop_reason="<|im_end|>",
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
    assert outcome.stop_reason == "model_finished"
    assert outcome.assistant_prefix == "</think>"
    assert outcome.generated_tokens == 1
    assert call_count["count"] == 1
    assert observed_stops == [("</exec",)]


def test_decode_chunk_im_end_after_steer_does_not_normalize(
    tmp_path: Path, monkeypatch
) -> None:
    """EOS-stopped chunks should not branch or normalize after a steer marker."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=1.0,
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
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
        return GenerationChoice(
            index=0,
            text="abc<steer>",
            finish_reason="stop",
            stop_reason="<|im_end|>",
            tokens=(
                ParsedToken(
                    token="abc<steer>",
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
    assert outcome.stop_reason == "model_finished"
    assert outcome.assistant_prefix == "abc<steer>"
    assert "</steer>" not in outcome.assistant_prefix
    assert "<exec>" not in outcome.assistant_prefix


def test_decode_chunk_generated_im_end_token_stops_before_suffix(
    tmp_path: Path, monkeypatch
) -> None:
    """Generated EOS token should be stripped before any continuation."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=1.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    call_count = {"count": 0}

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
        call_count["count"] += 1
        return GenerationChoice(
            index=0,
            text="answer<|im_end|>tail",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="answer",
                    logprob=-0.1,
                    top_entries=(("answer", -0.1),),
                ),
                ParsedToken(
                    token="<|im_end|>",
                    logprob=-0.2,
                    top_entries=(("<|im_end|>", -0.2),),
                ),
                ParsedToken(
                    token="tail",
                    logprob=-0.3,
                    top_entries=(("tail", -0.3),),
                ),
            ),
            prompt_token_ids=(1, 2),
            token_ids=(10, 248046, 11),
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
    assert outcome.stop_reason == "model_finished"
    assert outcome.assistant_prefix == "answer"
    assert outcome.token_ids == (10,)
    assert outcome.generated_tokens == 1
    assert call_count["count"] == 1


def test_steer_think_close_with_unboxed_answer_continues(
    tmp_path: Path, monkeypatch
) -> None:
    """Unboxed post-think text should not terminate strict-answer rollouts."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=64)
    observed_stops: list[tuple[str, ...] | None] = []
    call_count = {"count": 0}
    observed_prefixes: list[str] = []

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = prompt_token_ids, max_tokens, n, kwargs
        observed_prefixes.append(assistant_prefix)
        observed_stops.append(stop)
        call_count["count"] += 1
        if call_count["count"] == 2:
            return GenerationChoice(
                index=0,
                text="\n\n\\boxed{7}",
                finish_reason="stop",
                stop_reason=None,
                tokens=(
                    ParsedToken(
                        token="\n\n\\boxed{7}",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 42, 43),
                token_ids=(44,),
            )
        return GenerationChoice(
            index=0,
            text="</think>\nAnswer: 7",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="</think>\nAnswer: 7",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2),
            token_ids=(42, 43),
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
    assert observed_stops[-1] is None
    assert observed_prefixes[-1].endswith("</think>\nAnswer: 7")
    assert outcome.assistant_prefix.endswith("</think>\nAnswer: 7\n\n\\boxed{7}")
    assert outcome.generated_tokens == 2


def test_steer_think_partial_continues_without_stop(
    tmp_path: Path, monkeypatch
) -> None:
    """After partial `</think`, steer path should continue with `stop=None`."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=64)
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
    tree = _minimal_tree()
    outcome = executor._decode_until_event(
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
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "think_end"
    assert observed_stops[0] == ("</exec",)
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
    executor.decoding = replace(
        executor.decoding,
        decode_chunk_tokens=1,
        max_gen_toks=8,
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
    tree = _minimal_tree()
    outcome = executor._decode_until_event(
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
    assert outcome.event_type == "trigger"
    assert outcome.trigger_type == "steer_boundary"
    assert outcome.assistant_prefix.endswith("<steer>")


def test_steer_chunk_length_normalizes_partial_exec_close_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    """Decode-chunk length boundaries should repair partial exec close prefixes."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(
        executor.decoding,
        decode_chunk_tokens=1,
        max_gen_toks=8,
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
        _ = assistant_prefix, prompt_token_ids, stop, n, kwargs
        assert max_tokens == 1
        return GenerationChoice(
            index=0,
            text="work</e",
            finish_reason="length",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="work</e",
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
            assistant_prefix="<think><exec>",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )
    assert outcome.event_type == "trigger"
    assert outcome.trigger_type == "steer_boundary"
    assert outcome.assistant_prefix.endswith("<exec>work</exec>\n")


def test_response_length_boundary_hard_stops_without_steer_normalization(
    tmp_path: Path, monkeypatch
) -> None:
    """Full response-token exhaustion should terminate instead of forcing steer."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(
        executor.decoding,
        decode_chunk_tokens=512,
        max_gen_toks=1,
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
        _ = assistant_prefix, prompt_token_ids, stop, n, kwargs
        assert max_tokens == 1
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
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "max_gen_toks_reached"
    assert outcome.assistant_prefix == "plain"


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
    continued_prefixes: list[str] = []

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
                text="...",
                finish_reason="stop",
                stop_reason="</exec",
                tokens=(
                    ParsedToken(
                        token="...",
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
        _ = executor, prompt_token_ids, token_ids, token_traces, generated_tokens
        _ = request_stream_id
        calls["continued"] += 1
        continued_prefixes.append(assistant_prefix)
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
            steer_phase_token_spans=((0, 1),),
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
    assert continued_prefixes == ["...</exec>\n"]
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "model_finished"
    assert outcome.steer_phase_token_spans == ((0, 1),)
    events = read_tree_event_rows(store=executor.artifact_store)
    decode_chunks = [
        str(row.get("payload", {}).get("chunk_text", ""))
        for row in events
        if row.get("event_type") == "decode_chunk"
    ]
    assert not any("<steer></steer><exec></exec>" in text for text in decode_chunks)
    steer_events = [
        row for row in events if row.get("event_type") == "steer_block_generated"
    ]
    assert steer_events
    payload = dict(steer_events[-1].get("payload", {}))
    assert payload.get("source") == "explicit_stop_nonbranch"


def test_bare_open_steer_continuation_samples_before_normalizing(
    tmp_path: Path, monkeypatch
) -> None:
    """Open steer tags should receive sampled text before boundary repair."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    observed_requests: list[
        tuple[str, tuple[int, ...] | None, tuple[str, ...] | None]
    ] = []

    def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = max_tokens, n, kwargs
        observed_requests.append((assistant_prefix, prompt_token_ids, stop))
        return GenerationChoice(
            index=0,
            text="Plan next</steer>",
            finish_reason="stop",
            stop_reason="</steer>",
            tokens=(
                ParsedToken(
                    token="Plan next</steer>",
                    logprob=-0.1,
                    top_entries=(("Plan", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 1000),
            token_ids=(10, 11, 12),
        )

    monkeypatch.setattr(executor, "_generate_choice", fake_generate_choice)
    monkeypatch.setattr(
        executor,
        "_update_request_stream_state_output_ids",
        lambda **_: None,
    )

    outcome = continue_with_single_steer_candidate(
        executor=executor,
        assistant_prefix="<steer",
        prompt_token_ids=(1, 2),
        token_ids=(),
        token_traces=(),
        generated_tokens=0,
        request_stream_id="decode:node_root",
    )

    assert observed_requests == [("<steer>", (1, 2, 1000), ("</steer",))]
    assert outcome.event_type == "continued"
    assert outcome.assistant_prefix.startswith("<steer>Plan next</steer>\n<exec>")
    assert "<steer></steer><exec></exec>" not in outcome.assistant_prefix


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
                stop_reason="</exec",
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
            assistant_prefix="seed</steer>\n<exec>",
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
    assert observed_stops[0] == ("</exec",)
    assert calls["continued"] == 1
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "model_finished"
    events = read_tree_event_rows(store=executor.artifact_store)
    steer_events = [
        row for row in events if row.get("event_type") == "steer_block_generated"
    ]
    assert steer_events
    assert all(
        bool(row.get("payload", {}).get("branching_enabled")) is False
        for row in events
        if row.get("event_type") == "decode_chunk"
    )


def test_initial_think_boundary_samples_steer_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The first Qwen think-open boundary should sample a steer choice first."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=80)
    requests: list[tuple[str, str, tuple[str, ...] | None]] = []
    steer_calls = {"count": 0}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = prompt_token_ids, max_tokens, n
        request_kind = str(kwargs["request_kind"])
        requests.append((request_kind, assistant_prefix, stop))
        if request_kind == "steer_single_candidate":
            steer_calls["count"] += 1
            if steer_calls["count"] == 1:
                assert assistant_prefix == "<think>\n"
                assert stop == ("</steer", "</think>")
                return GenerationChoice(
                    index=0,
                    text="<steer>Restate problem</steer>",
                    finish_reason="stop",
                    stop_reason="</steer",
                    tokens=(
                        ParsedToken(
                            token="<steer>Restate problem</steer>",
                            logprob=-0.1,
                            top_entries=(("a", -0.2),),
                        ),
                    ),
                    prompt_token_ids=(1, 2, 3),
                    token_ids=(101,),
                )
            return GenerationChoice(
                index=0,
                text="",
                finish_reason="stop",
                stop_reason="</think>",
                tokens=(),
                prompt_token_ids=(1, 2, 3, 101, 102),
                token_ids=(),
            )
        if request_kind == "decode_chunk":
            assert assistant_prefix.endswith("</steer>\n<exec>")
            return GenerationChoice(
                index=0,
                text="Use the constraints.",
                finish_reason="stop",
                stop_reason="</exec",
                tokens=(
                    ParsedToken(
                        token="Use the constraints.",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 3, 101),
                token_ids=(102,),
            )
        assert request_kind == "think_close_continuation"
        return GenerationChoice(
            index=0,
            text="\n\n\\boxed{7}",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="\n\n\\boxed{7}",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3, 101, 102, 103),
            token_ids=(104,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="<think>\n",
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
            token_traces=(),
            branch_points_used=0,
        ),
    )

    assert requests[0][0] == "steer_single_candidate"
    assert outcome.event_type == "terminated"
    events = read_tree_event_rows(store=executor.artifact_store)
    decode_chunks = [
        str(row.get("payload", {}).get("chunk_text", ""))
        for row in events
        if row.get("event_type") == "decode_chunk"
    ]
    assert not any("<steer>Restate problem" in text for text in decode_chunks)
    steer_chunks = [
        str(row.get("payload", {}).get("chunk_text", ""))
        for row in events
        if row.get("event_type") == "steer_block_generated"
    ]
    assert any("<steer>Restate problem" in text for text in steer_chunks)


def test_scheduled_root_decode_samples_initial_steer_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The scheduled node_root path should not treat first steer text as decode."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(
        executor.decoding,
        initial_assistant_prefix="<think>\n",
        max_gen_toks=80,
    )
    tree, frontier, _ = executor._initialize_tree(
        doc_id=0,
        doc_attempt=0,
        task_name="math",
        model_id="fake",
        leaf_budget=1,
    )
    requests: list[tuple[str, str]] = []
    steer_calls = {"count": 0}

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = prompt_token_ids, max_tokens, stop, n
        request_kind = str(kwargs["request_kind"])
        requests.append((request_kind, assistant_prefix))
        if request_kind == "steer_single_candidate":
            steer_calls["count"] += 1
            if steer_calls["count"] == 1:
                return GenerationChoice(
                    index=0,
                    text="<steer>Restate problem</steer>",
                    finish_reason="stop",
                    stop_reason="</steer",
                    tokens=(
                        ParsedToken(
                            token="<steer>Restate problem</steer>",
                            logprob=-0.1,
                            top_entries=(("a", -0.2),),
                        ),
                    ),
                    prompt_token_ids=(1, 2, 3),
                    token_ids=(101,),
                )
            return GenerationChoice(
                index=0,
                text="",
                finish_reason="stop",
                stop_reason="</think>",
                tokens=(),
                prompt_token_ids=(1, 2, 3, 101, 102),
                token_ids=(),
            )
        if request_kind == "decode_chunk":
            assert assistant_prefix.endswith("</steer>\n<exec>")
            return GenerationChoice(
                index=0,
                text="Use constraints.",
                finish_reason="stop",
                stop_reason="</exec",
                tokens=(
                    ParsedToken(
                        token="Use constraints.",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 3, 101),
                token_ids=(102,),
            )
        assert request_kind == "think_close_continuation"
        return GenerationChoice(
            index=0,
            text="\n\n\\boxed{7}",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="\n\n\\boxed{7}",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3, 101, 102),
            token_ids=(103,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    scheduled, outcome = asyncio.run(
        executor._decode_state_outcome_async(
            tree=tree,
            scheduled=branch_executor_module._ScheduledDecode(state=frontier[0]),
        )
    )

    assert scheduled.state.node_id == "node_root"
    assert requests[0] == ("steer_single_candidate", "<think>\n")
    assert outcome.event_type == "terminated"
    events = read_tree_event_rows(store=executor.artifact_store)
    decode_chunks = [
        str(row.get("payload", {}).get("chunk_text", ""))
        for row in events
        if row.get("event_type") == "decode_chunk"
    ]
    assert not any("<steer>Restate problem" in text for text in decode_chunks)


def test_post_exec_resumed_boundary_samples_steer_before_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A maxed/resumed post-exec path should not decode an exec chunk first."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=1.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=80)
    requests: list[tuple[str, str, tuple[str, ...] | None]] = []

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = prompt_token_ids, max_tokens, n
        request_kind = str(kwargs["request_kind"])
        requests.append((request_kind, assistant_prefix, stop))
        if request_kind == "steer_single_candidate":
            assert assistant_prefix.endswith("</exec>\n")
            assert stop == ("</steer", "</think>")
            return GenerationChoice(
                index=0,
                text="",
                finish_reason="stop",
                stop_reason="</think>",
                tokens=(),
                prompt_token_ids=(1, 2, 3),
                token_ids=(),
            )
        assert request_kind == "think_close_continuation"
        return GenerationChoice(
            index=0,
            text="\n\n\\boxed{7}",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="\n\n\\boxed{7}",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3, 4),
            token_ids=(104,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="<think>\n<steer>Plan</steer>\n<exec>Work</exec>\n",
            prompt_token_ids=(1, 2, 3),
            token_ids=(10, 11, 12),
            token_traces=(),
            branch_points_used=2,
        ),
        branching_enabled=False,
        steer_normalization_enabled=True,
    )

    assert requests[0][0] == "steer_single_candidate"
    assert all(request[0] != "decode_chunk" for request in requests)
    assert outcome.event_type == "terminated"
    steer_events = [
        row for row in read_tree_event_rows(store=executor.artifact_store)
        if row.get("event_type") == "steer_block_generated"
    ]
    assert steer_events
    assert steer_events[-1]["payload"]["source"] == "steer_decision_boundary_nonbranch"


def test_max_branch_skip_completes_partial_exec_stop_before_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Max-branch resume should not decode a steer decision as exec text."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=1.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.branching = replace(executor.branching, max_branch_points_per_rollout=4)
    executor.decoding = replace(executor.decoding, max_gen_toks=80)
    tree = _minimal_tree()
    requests: list[tuple[str, str, tuple[str, ...] | None]] = []

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = prompt_token_ids, max_tokens, n
        request_kind = str(kwargs["request_kind"])
        requests.append((request_kind, assistant_prefix, stop))
        if request_kind == "steer_single_candidate":
            assert assistant_prefix.endswith("</exec>\n")
            return GenerationChoice(
                index=0,
                text="<steer>Calculate via complement</steer>",
                finish_reason="stop",
                stop_reason="</steer",
                tokens=(
                    ParsedToken(
                        token="<steer>Calculate via complement</steer>",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=None,
                token_ids=(104,),
            )
        if request_kind == "decode_chunk":
            assert not assistant_prefix.endswith("</exec>\n")
            return GenerationChoice(
                index=0,
                text="Exec after steer.</exec",
                finish_reason="stop",
                stop_reason="</exec",
                tokens=(
                    ParsedToken(
                        token="Exec after steer.</exec",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=None,
                token_ids=(105,),
            )
        assert request_kind == "think_close_continuation"
        return GenerationChoice(
            index=0,
            text="\n\n\\boxed{7}",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="\n\n\\boxed{7}",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=None,
            token_ids=(106,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    state = PathState(
        node_id="node_root",
        assistant_prefix="<think>\n<steer>Plan</steer>\n<exec>Work",
        prompt_token_ids=None,
        token_ids=(10,),
        token_traces=(),
        branch_points_used=4,
    )
    outcome = DecodeOutcome(
        event_type="trigger",
        trigger_type="steer_boundary",
        assistant_prefix="<think>\n<steer>Plan</steer>\n<exec>Work</exec",
        prompt_token_ids=None,
        token_ids=(10, 11),
        token_traces=(),
        generated_tokens=2,
        stop_reason="",
        steer_phase_token_spans=((0, 1),),
    )

    scheduled, expansion = executor._handle_state_outcome(
        tree=tree,
        scheduled=branch_executor_module._ScheduledDecode(
            state=state,
            inline_epsilon_enabled=False,
        ),
        outcome=outcome,
        doc_id=0,
        leaf_limit=8,
    )
    assert expansion is None
    assert len(scheduled) == 1
    assert scheduled[0].state.assistant_prefix.endswith("</exec>")
    assert scheduled[0].state.steer_phase_token_spans == ((0, 1),)

    resumed = executor._decode_until_event(
        tree=tree,
        state=scheduled[0].state,
        branching_enabled=scheduled[0].branching_enabled,
        steer_normalization_enabled=scheduled[0].steer_normalization_enabled,
        inline_epsilon_enabled=scheduled[0].inline_epsilon_enabled,
    )

    assert requests[0][0] == "steer_single_candidate"
    assert "<steer>Calculate via complement</steer>" in resumed.assistant_prefix
    events = read_tree_event_rows(store=executor.artifact_store)
    decode_chunks = [
        str(row.get("payload", {}).get("chunk_text", ""))
        for row in events
        if row.get("event_type") == "decode_chunk"
    ]
    assert not any("<steer>Calculate via complement" in text for text in decode_chunks)


def test_inline_epsilon_selection_stays_on_same_node(
    tmp_path: Path, monkeypatch
) -> None:
    """Inline epsilon steer selection should emit diagnostics without new edges."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=1.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.allow_true_branching = False
    executor.decoding = replace(executor.decoding, max_gen_toks=64)
    executor.branching = replace(executor.branching, epsilon_greedy_prob=1.0)
    pool = CandidatePoolRecord(
        candidate_pool_id="pool_inline",
        branch_point_id="bp_inline",
        node_id="node_root",
        trigger_type="steer_boundary",
        entropy_value=None,
        candidates=(
            CandidateRecord(
                candidate_id=0,
                text="<steer>reject</steer>",
                token_ids=(10,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer",
            ),
            CandidateRecord(
                candidate_id=1,
                text="<steer>keep</steer>",
                token_ids=(11,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer",
            ),
        ),
    )
    call_count = {"decode": 0}

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
        call_count["decode"] += 1
        if call_count["decode"] == 1:
            return GenerationChoice(
                index=0,
                text="trigger",
                finish_reason="stop",
                stop_reason="</exec",
                tokens=(
                    ParsedToken(
                        token="trigger",
                        logprob=-0.1,
                        top_entries=(("a", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 3),
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
            prompt_token_ids=(1, 2, 3, 42, 11),
            token_ids=(43,),
        )

    async def fake_resolve_pool_async(**kwargs: object) -> CandidatePoolRecord:
        _ = kwargs
        return pool

    async def fake_resolve_selection_outcomes_async(
        *,
        pool: CandidatePoolRecord,
        selector_params: object = None,
        selector_modes: tuple[str, ...] | None = None,
    ) -> tuple[SelectionOutcome, ...]:
        _ = pool, selector_params
        assert selector_modes == ("embed_diverse_topk_random",)
        return (
            SelectionOutcome(
                selector_mode="embed_diverse_topk_random",
                selected_candidate_ids=(1,),
                shortlist_candidate_ids=(0, 1),
            ),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    monkeypatch.setattr(
        executor,
        "_resolve_candidate_pool_async",
        fake_resolve_pool_async,
    )
    monkeypatch.setattr(
        executor,
        "_resolve_selection_outcomes_async",
        fake_resolve_selection_outcomes_async,
    )
    tree = _minimal_tree()
    outcome = executor._decode_until_event(
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
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "model_finished"
    assert outcome.branch_points_used == 0
    events = read_tree_event_rows(store=executor.artifact_store)
    event_types = [str(row.get("event_type")) for row in events]
    assert "candidate_pool_resolved" in event_types
    assert "selector_applied" in event_types
    assert "selector_continued_inline" in event_types
    assert "edge_selected" not in event_types
    assert "node_created" not in event_types
    assert len(tree.branch_points) == 0


def test_inline_epsilon_continues_after_true_branch_cap(
    tmp_path: Path, monkeypatch
) -> None:
    """Maxed true branching should not disable later inline epsilon exploration."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=1.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=64)
    executor.branching = replace(
        executor.branching,
        epsilon_greedy_prob=1.0,
        max_branch_points_per_rollout=4,
    )
    pool = CandidatePoolRecord(
        candidate_pool_id="pool_late_inline",
        branch_point_id="bp_late_inline",
        node_id="node_maxed",
        trigger_type="steer_boundary",
        entropy_value=None,
        candidates=(
            CandidateRecord(
                candidate_id=0,
                text="<steer>late keep</steer>",
                token_ids=(11,),
                tokens=(),
                finish_reason="stop",
                stop_reason="</steer",
            ),
        ),
    )
    decode_count = 0

    async def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        nonlocal decode_count
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
        decode_count += 1
        if decode_count == 1:
            return GenerationChoice(
                index=0,
                text="trigger",
                finish_reason="stop",
                stop_reason="</exec",
                tokens=(
                    ParsedToken(
                        token="trigger",
                        logprob=-0.1,
                        top_entries=(("x", -0.2),),
                    ),
                ),
                prompt_token_ids=(1, 2, 3),
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
                    top_entries=(("x", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3, 42, 11),
            token_ids=(43,),
        )

    async def fake_resolve_pool_async(**kwargs: object) -> CandidatePoolRecord:
        _ = kwargs
        return pool

    async def fake_resolve_selection_outcomes_async(
        *,
        pool: CandidatePoolRecord,
        selector_params: object = None,
        selector_modes: tuple[str, ...] | None = None,
    ) -> tuple[SelectionOutcome, ...]:
        _ = pool, selector_params
        assert selector_modes == ("embed_diverse_topk_random",)
        return (
            SelectionOutcome(
                selector_mode="embed_diverse_topk_random",
                selected_candidate_ids=(0,),
                shortlist_candidate_ids=(0,),
            ),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    monkeypatch.setattr(
        executor, "_resolve_candidate_pool_async", fake_resolve_pool_async
    )
    monkeypatch.setattr(
        executor,
        "_resolve_selection_outcomes_async",
        fake_resolve_selection_outcomes_async,
    )

    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_maxed",
            assistant_prefix="prefix",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=4,
        ),
        branching_enabled=False,
        steer_normalization_enabled=True,
        inline_epsilon_enabled=True,
    )

    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "model_finished"
    assert outcome.branch_points_used == 4
    events = read_tree_event_rows(store=executor.artifact_store)
    event_types = [str(row.get("event_type")) for row in events]
    assert "selector_continued_inline" in event_types
    assert "edge_selected" not in event_types


def test_inline_epsilon_uses_deterministic_boundary_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inline epsilon should not square probability at generated boundaries."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.15,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.branching = replace(executor.branching, epsilon_greedy_prob=0.2)
    observed_detection_probs: list[float] = []

    async def fake_generate_choice(**kwargs: object) -> GenerationChoice:
        _ = kwargs
        return GenerationChoice(
            index=0,
            text="<steer>",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="<steer>",
                    logprob=-0.1,
                    top_entries=(("x", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(42,),
        )

    def fake_consume_choice_tokens(**kwargs: object) -> DecodeOutcome:
        observed_detection_probs.append(cast(float, kwargs["branch_prob"]))
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=str(kwargs["assistant_prefix"]) + "<steer>",
            prompt_token_ids=None,
            token_ids=(42,),
            token_traces=(),
            generated_tokens=1,
            stop_reason="model_finished",
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    monkeypatch.setattr(
        branch_executor_module,
        "consume_choice_tokens",
        fake_consume_choice_tokens,
    )

    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_maxed",
            assistant_prefix="prefix",
            prompt_token_ids=None,
            token_ids=(),
            token_traces=(),
            branch_points_used=4,
        ),
        branching_enabled=False,
        steer_normalization_enabled=True,
        inline_epsilon_enabled=True,
    )

    assert outcome.event_type == "terminated"
    assert observed_detection_probs == [1.0]


def test_steer_exec_repeat_loop_forces_think_close_continuation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated exec blocks should close thinking and continue to final answer."""

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
        ), "repeat-loop detection should stop exec-block decoding"
        return GenerationChoice(
            index=0,
            text=repeated_exec_chunk,
            finish_reason="stop",
            stop_reason="</exec",
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
            assistant_prefix=f"{assistant_prefix}Repeat steering plan</steer>\n<exec>",
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
    install_fake_repeat_close_continuation(
        monkeypatch=monkeypatch,
        call_counts=call_counts,
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
    assert outcome.stop_reason == "think_end"
    assert branch_executor_module.THINK_CLOSE_TAG in outcome.assistant_prefix
    assert "</exec>\n\n</think>\nFinal answer." in outcome.assistant_prefix
    assert outcome.assistant_prefix.endswith("\nFinal answer.")
    assert outcome.repeat_stop_reason == "repeated_exec_block_loop"
    assert outcome.repeat_block_kind == "exec"
    assert outcome.repeat_block_count is not None
    assert outcome.repeat_block_count >= 3
    assert call_counts["generate"] == 3
    assert call_counts["continued"] == 2
    assert call_counts["post_think"] == 1
    events = read_tree_event_rows(store=executor.artifact_store)
    repeat_events = [
        row for row in events if row.get("event_type") == "exec_repeat_terminated"
    ]
    assert repeat_events
    payload = dict(repeat_events[-1].get("payload", {}))
    assert int(payload["repeated_exec_blocks"]) >= 3
    assert float(payload["similarity_threshold"]) == 0.85
    assert int(payload["similarity_lookback_window"]) == 3
    assert int(payload["termination_block_count"]) == 3
    forced_close_events = [
        row for row in events if row.get("event_type") == "repeat_forced_think_close"
    ]
    assert forced_close_events
    close_payload = dict(forced_close_events[-1].get("payload", {}))
    assert close_payload["repeat_stop_reason"] == "repeated_exec_block_loop"
    assert close_payload["repeat_block_kind"] == "exec"
    assert int(close_payload["repeat_block_count"]) >= 3
    assert close_payload["normalization_suffix"] == ""


def test_steer_exec_repeat_loop_forces_close_on_alternating_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alternating repeat blocks should force-close with lookback-window matching."""

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
        ), "alternating repeat-loop detection should stop exec-block decoding"
        chunk_index = (call_counts["generate"] - 1) % len(alternating_exec_chunks)
        exec_chunk = alternating_exec_chunks[chunk_index]
        return GenerationChoice(
            index=0,
            text=exec_chunk,
            finish_reason="stop",
            stop_reason="</exec",
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
            assistant_prefix=f"{assistant_prefix}Alternate steering</steer>\n<exec>",
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
    install_fake_repeat_close_continuation(
        monkeypatch=monkeypatch,
        call_counts=call_counts,
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
    assert outcome.stop_reason == "think_end"
    assert branch_executor_module.THINK_CLOSE_TAG in outcome.assistant_prefix
    assert "</exec>\n\n</think>\nFinal answer." in outcome.assistant_prefix
    assert call_counts["generate"] == 3
    assert call_counts["continued"] == 2
    assert call_counts["post_think"] == 1
    events = read_tree_event_rows(store=executor.artifact_store)
    repeat_events = [
        row for row in events if row.get("event_type") == "exec_repeat_terminated"
    ]
    assert repeat_events
    payload = dict(repeat_events[-1].get("payload", {}))
    assert int(payload["repeated_exec_blocks"]) >= 3
    assert float(payload["last_similarity_ratio"]) >= 0.85
    assert int(payload["similarity_lookback_window"]) == 3
    assert int(payload["termination_block_count"]) == 3


def test_steer_block_repeat_loop_forces_close_on_alternating_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Alternating steer blocks should force-close at 4 matches with lookback 3."""

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
        "Red square orbit.",
        "Blue triangle pulse.",
        "Gold spiral vector.",
        "Green prism lattice.",
        "Silver wave tangent.",
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
        ), "steer repeat-loop detection should stop exec-block decoding"
        exec_chunk = exec_chunks[call_counts["generate"] - 1]
        return GenerationChoice(
            index=0,
            text=exec_chunk,
            finish_reason="stop",
            stop_reason="</exec",
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
            assistant_prefix=(
                f"{assistant_prefix}<steer>{steer_chunk}</steer>\n<exec>"
            ),
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
    install_fake_repeat_close_continuation(
        monkeypatch=monkeypatch,
        call_counts=call_counts,
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
    assert outcome.stop_reason == "think_end"
    assert branch_executor_module.THINK_CLOSE_TAG in outcome.assistant_prefix
    assert "I am repeating now. Time to stop." not in outcome.assistant_prefix
    assert "</exec>\n</think>\nFinal answer." in outcome.assistant_prefix
    assert outcome.repeat_stop_reason == "repeated_steer_block_loop"
    assert outcome.repeat_block_kind == "steer"
    assert outcome.repeat_block_count is not None
    assert outcome.repeat_block_count >= 4
    assert call_counts["generate"] == len(exec_chunks)
    assert call_counts["continued"] == len(exec_chunks)
    assert call_counts["post_think"] == 1
    events = read_tree_event_rows(store=executor.artifact_store)
    repeat_events = [
        row for row in events if row.get("event_type") == "steer_repeat_terminated"
    ]
    assert repeat_events
    payload = dict(repeat_events[-1].get("payload", {}))
    assert int(payload["repeated_steer_blocks"]) >= 4
    assert float(payload["similarity_threshold"]) == 0.92
    assert int(payload["similarity_lookback_window"]) == 3
    assert int(payload["termination_block_count"]) == 4
    assert float(payload["last_similarity_ratio"]) >= 0.92
    forced_close_events = [
        row for row in events if row.get("event_type") == "repeat_forced_think_close"
    ]
    assert forced_close_events
    close_payload = dict(forced_close_events[-1].get("payload", {}))
    assert close_payload["repeat_stop_reason"] == "repeated_steer_block_loop"
    assert close_payload["repeat_block_kind"] == "steer"
    assert int(close_payload["repeat_block_count"]) >= 4
    assert close_payload["normalization_suffix"] == ""
    assert close_payload["forced_close_text"] == "</think>"
    forced_span = outcome.steer_phase_token_spans[-1]
    assert forced_span[1] - forced_span[0] == len(
        close_payload["think_close_token_ids"]
    )


def test_repeat_forced_close_does_not_append_think_inside_steer() -> None:
    """Malformed repeat tails should not produce `<steer></think>`."""

    malformed_prefix = (
        "<think><steer>Conclude k=4 failure</steer>\n"
        "<exec><steer></steer><exec></exec>\n<steer>"
    )
    normalized_prefix = branch_executor_module._repeat_forced_think_close_prefix(
        text=malformed_prefix,
    )
    forced_text = normalized_prefix + branch_executor_module.THINK_CLOSE_TAG

    assert "<steer></think>" not in forced_text
    assert forced_text.endswith("<steer></steer>\n</exec>\n</think>")


def test_steer_repeat_loop_uses_production_token_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Repeated steer labels should terminate once production token floor is met."""

    executor = build_executor(
        tmp_path=tmp_path,
        branch_prob=0.0,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = DecodingConfig(
        temperature=executor.decoding.temperature,
        top_p=executor.decoding.top_p,
        max_gen_toks=branch_executor_module.REPEAT_TERMINATION_MIN_GENERATED_TOKENS
        + 100,
        top_logprobs=executor.decoding.top_logprobs,
        decode_chunk_tokens=executor.decoding.decode_chunk_tokens,
    )
    repeated_steer_text = "Final check with small n"
    call_counts = {"generate": 0, "continued": 0}
    exec_chunks = (
        "Check a two-row parity construction before counting cases.",
        "Use a geometric invariant to rule out the proposed packing.",
        "Translate the condition into a graph coloring constraint.",
        "Compare the boundary equations against the normalization target.",
        "Inspect the residual term after substituting the small example.",
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
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
        call_counts["generate"] += 1
        assert (
            call_counts["generate"] <= 5
        ), "production repeat-loop detection should stop steer decoding"
        chunk_text = exec_chunks[call_counts["generate"] - 1]
        return GenerationChoice(
            index=0,
            text=chunk_text,
            finish_reason="stop",
            stop_reason="</exec",
            tokens=(
                ParsedToken(
                    token=chunk_text,
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(7000 + call_counts["generate"],),
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
        _ = executor, request_stream_id
        call_counts["continued"] += 1
        continuation_token_id = 8000 + call_counts["continued"]
        continuation_text = f"<steer>{repeated_steer_text}</steer>\n<exec>"
        continuation_trace = TokenTrace(
            token_index=len(token_traces),
            token_id=continuation_token_id,
            token_text=continuation_text,
            logprob=-0.1,
            probability=0.9,
        )
        return DecodeOutcome(
            event_type="continued",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=f"{assistant_prefix}{continuation_text}",
            prompt_token_ids=(
                None
                if prompt_token_ids is None
                else tuple(prompt_token_ids) + (continuation_token_id,)
            ),
            token_ids=tuple(token_ids) + (continuation_token_id,),
            token_traces=tuple(token_traces) + (continuation_trace,),
            generated_tokens=generated_tokens + 1,
            stop_reason="",
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice)
    monkeypatch.setattr(
        "branching_eval.branch_executor.continue_with_single_steer_candidate_async",
        fake_continue_with_single,
    )
    install_fake_repeat_close_continuation(
        monkeypatch=monkeypatch,
        call_counts=call_counts,
    )
    initial_token_ids = tuple(
        range(branch_executor_module.REPEAT_TERMINATION_MIN_GENERATED_TOKENS)
    )

    outcome = executor._decode_until_event(
        tree=_minimal_tree(),
        state=PathState(
            node_id="node_root",
            assistant_prefix="<think><steer>Seed steering</steer>\n<exec>",
            prompt_token_ids=initial_token_ids,
            token_ids=initial_token_ids,
            token_traces=(),
            branch_points_used=0,
        ),
    )

    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "think_end"
    assert "I am repeating now. Time to stop." not in outcome.assistant_prefix
    assert outcome.assistant_prefix.count(repeated_steer_text) == 3
    assert outcome.repeat_stop_reason == "repeated_steer_block_loop"
    assert outcome.repeat_block_kind == "steer"
    assert call_counts["generate"] == 4
    assert call_counts["continued"] == 4
    assert call_counts["post_think"] == 1
    events = read_tree_event_rows(store=executor.artifact_store)
    repeat_events = [
        row for row in events if row.get("event_type") == "steer_repeat_terminated"
    ]
    assert repeat_events
    payload = dict(repeat_events[-1].get("payload", {}))
    assert int(payload["repeated_steer_blocks"]) >= 4
    assert payload["normalized_steer_block_preview"] == repeated_steer_text.lower()


def test_single_steer_continuation_keeps_vllm_tokens_and_appends_suffix(
    tmp_path: Path, monkeypatch
) -> None:
    """Single steer continuation should append injected suffix tokens only."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = DecodingConfig(
        temperature=executor.decoding.temperature,
        top_p=executor.decoding.top_p,
        max_gen_toks=32,
        top_logprobs=executor.decoding.top_logprobs,
        decode_chunk_tokens=executor.decoding.decode_chunk_tokens,
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

    stream_updates: list[tuple[int, ...]] = []

    def fake_update_stream_state(
        *, request_stream_id: str, consumed_output_token_ids: tuple[int, ...]
    ) -> None:
        _ = request_stream_id
        stream_updates.append(consumed_output_token_ids)

    monkeypatch.setattr(executor, "_generate_choice", fake_generate_choice)
    monkeypatch.setattr(
        executor,
        "_update_request_stream_state_output_ids",
        fake_update_stream_state,
    )
    outcome = continue_with_single_steer_candidate(
        executor=executor,
        assistant_prefix="seed</exec>\n<steer>",
        prompt_token_ids=(1, 2, 3),
        token_ids=(),
        token_traces=(),
        generated_tokens=0,
        request_stream_id="decode:node_root",
    )
    assert outcome.event_type == "continued"
    assert outcome.assistant_prefix.endswith("abc</steer>\n<exec>")
    assert outcome.generated_tokens == len(outcome.token_ids)
    assert outcome.token_ids[:3] == (10, 11, 12)
    assert outcome.token_ids[3:] == tuple(range(1000, 1007))
    assert outcome.prompt_token_ids is not None
    assert outcome.prompt_token_ids[:6] == (1, 2, 3, 10, 11, 12)
    assert outcome.prompt_token_ids[6:] == tuple(range(1000, 1007))
    assert stream_updates == [(10, 11, 12)]


def test_single_steer_continuation_im_end_does_not_normalize(
    tmp_path: Path, monkeypatch
) -> None:
    """EOS-stopped steer candidates should terminate without injected suffixes."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = DecodingConfig(
        temperature=executor.decoding.temperature,
        top_p=executor.decoding.top_p,
        max_gen_toks=32,
        top_logprobs=executor.decoding.top_logprobs,
        decode_chunk_tokens=executor.decoding.decode_chunk_tokens,
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
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
        return GenerationChoice(
            index=0,
            text="abc<steer>",
            finish_reason="stop",
            stop_reason="<|im_end|>",
            tokens=(
                ParsedToken(
                    token="abc<steer>",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3),
            token_ids=(10,),
        )

    stream_updates: list[tuple[int, ...]] = []

    def fake_update_stream_state(
        *, request_stream_id: str, consumed_output_token_ids: tuple[int, ...]
    ) -> None:
        _ = request_stream_id
        stream_updates.append(consumed_output_token_ids)

    monkeypatch.setattr(executor, "_generate_choice", fake_generate_choice)
    monkeypatch.setattr(
        executor,
        "_update_request_stream_state_output_ids",
        fake_update_stream_state,
    )
    outcome = continue_with_single_steer_candidate(
        executor=executor,
        assistant_prefix="seed</exec>\n<steer>",
        prompt_token_ids=(1, 2, 3),
        token_ids=(),
        token_traces=(),
        generated_tokens=0,
        request_stream_id="decode:node_root",
    )

    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "model_finished"
    assert outcome.assistant_prefix.endswith("<steer>abc<steer>")
    assert "</steer>" not in outcome.assistant_prefix
    assert "<exec>" not in outcome.assistant_prefix
    assert outcome.prompt_token_ids == (1, 2, 3, 10)
    assert stream_updates == [(10,)]


def test_single_steer_continuation_accepts_empty_logprob_trace(
    tmp_path: Path, monkeypatch
) -> None:
    """Single steer continuation should use text/token ids without logprobs."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = DecodingConfig(
        temperature=executor.decoding.temperature,
        top_p=executor.decoding.top_p,
        max_gen_toks=32,
        top_logprobs=0,
        decode_chunk_tokens=executor.decoding.decode_chunk_tokens,
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
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
        return GenerationChoice(
            index=0,
            text="abc</steer>",
            finish_reason="stop",
            stop_reason="</steer>",
            tokens=(),
            prompt_token_ids=(1, 2, 3),
            token_ids=(10, 11, 12),
        )

    monkeypatch.setattr(executor, "_generate_choice", fake_generate_choice)
    monkeypatch.setattr(
        executor,
        "_update_request_stream_state_output_ids",
        lambda **_: None,
    )

    outcome = continue_with_single_steer_candidate(
        executor=executor,
        assistant_prefix="seed</exec>\n<steer>",
        prompt_token_ids=(1, 2, 3),
        token_ids=(),
        token_traces=(),
        generated_tokens=0,
        request_stream_id="decode:node_root",
    )

    assert outcome.event_type == "continued"
    assert outcome.assistant_prefix.endswith("abc</steer>\n<exec>")
    assert outcome.generated_tokens == len(outcome.token_ids)
    assert outcome.token_ids[:3] == (10, 11, 12)


def test_single_steer_continuation_can_finish_thinking(
    tmp_path: Path, monkeypatch
) -> None:
    """Steer continuation should leave bare `</think>` for answer continuation."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    stop_values: list[tuple[str, ...] | None] = []

    def fake_generate_choice(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, n, kwargs
        stop_values.append(stop)
        return GenerationChoice(
            index=0,
            text="",
            finish_reason="stop",
            stop_reason="</think>",
            tokens=(),
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
        )

    monkeypatch.setattr(executor, "_generate_choice", fake_generate_choice)
    monkeypatch.setattr(
        executor,
        "_update_request_stream_state_output_ids",
        lambda **_: None,
    )
    outcome = continue_with_single_steer_candidate(
        executor=executor,
        assistant_prefix="seed</exec>\n",
        prompt_token_ids=(1, 2, 3),
        token_ids=(),
        token_traces=(),
        generated_tokens=0,
        request_stream_id="decode:node_root",
    )

    assert stop_values == [("</steer", "</think>")]
    assert outcome.event_type == "continued"
    assert outcome.stop_reason == ""
    assert outcome.assistant_prefix.endswith("</exec>\n</think>")
    assert outcome.token_ids == tuple(range(1000, 1008))
    assert outcome.prompt_token_ids == (1, 2, 3) + tuple(range(1000, 1008))


def test_single_steer_continuation_async_counts_normalized_suffix_tokens(
    tmp_path: Path, monkeypatch
) -> None:
    """Async steer continuation should count injected suffix tokens against the budget."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = DecodingConfig(
        temperature=executor.decoding.temperature,
        top_p=executor.decoding.top_p,
        max_gen_toks=32,
        top_logprobs=executor.decoding.top_logprobs,
        decode_chunk_tokens=executor.decoding.decode_chunk_tokens,
    )

    async def fake_generate_choice_async(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = assistant_prefix, prompt_token_ids, max_tokens, stop, n, kwargs
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

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice_async)
    monkeypatch.setattr(
        executor,
        "_update_request_stream_state_output_ids",
        lambda **_: None,
    )
    outcome = asyncio.run(
        continue_with_single_steer_candidate_async(
            executor=executor,
            assistant_prefix="seed</exec>\n<steer>",
            prompt_token_ids=(1, 2, 3),
            token_ids=(),
            token_traces=(),
            generated_tokens=0,
            request_stream_id="decode:node_root",
        )
    )
    assert outcome.event_type == "continued"
    assert outcome.generated_tokens == len(outcome.token_ids)
    assert outcome.assistant_prefix.endswith("abc</steer>\n<exec>")


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


def test_decode_continues_existing_bare_think_close_prefix(
    tmp_path: Path, monkeypatch
) -> None:
    """A child selected as bare `</think>` should continue into answer text."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = replace(executor.decoding, max_gen_toks=64)
    call_count = {"decode_calls": 0}
    observed_stops: list[tuple[str, ...] | None] = []
    observed_prefixes: list[str] = []

    async def fake_generate_choice_async(
        *,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        max_tokens: int,
        stop: tuple[str, ...] | None,
        n: int,
        **kwargs: object,
    ) -> GenerationChoice:
        _ = prompt_token_ids, max_tokens, n, kwargs
        observed_prefixes.append(assistant_prefix)
        observed_stops.append(stop)
        call_count["decode_calls"] += 1
        return GenerationChoice(
            index=0,
            text="\n\n\\boxed{42}",
            finish_reason="stop",
            stop_reason=None,
            tokens=(
                ParsedToken(
                    token="\n\n\\boxed{42}",
                    logprob=-0.1,
                    top_entries=(("a", -0.2),),
                ),
            ),
            prompt_token_ids=(1, 2, 3, 10),
            token_ids=(11,),
        )

    monkeypatch.setattr(executor, "_generate_choice_async", fake_generate_choice_async)
    state = PathState(
        node_id="node_root",
        assistant_prefix="seed</exec>\n</think>",
        prompt_token_ids=(1, 2, 3),
        token_ids=(10,),
        token_traces=(),
        branch_points_used=1,
    )
    outcome = executor._decode_until_event(tree=_minimal_tree(), state=state)

    assert call_count["decode_calls"] == 1
    assert observed_stops == [None]
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "think_end"
    assert observed_prefixes == ["seed</exec>\n</think>"]
    assert outcome.assistant_prefix.endswith("</think>\n\n\\boxed{42}")
    assert outcome.generated_tokens == 2
    assert outcome.branch_points_used == 1


def test_inline_epsilon_terminates_when_candidate_budget_is_exhausted(
    tmp_path: Path,
) -> None:
    """Inline epsilon should terminate cleanly when no candidate budget remains."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.allow_true_branching = False
    executor.branching = replace(executor.branching, epsilon_greedy_prob=1.0)
    executor.decoding = replace(executor.decoding, max_gen_toks=4)
    outcome = asyncio.run(
        executor._resolve_nonbranch_steer_trigger_async(
            tree=_minimal_tree(),
            state=PathState(
                node_id="node_root",
                assistant_prefix="prefix",
                prompt_token_ids=(1, 2, 3, 4),
                token_ids=(10, 11, 12, 13),
                token_traces=(),
                branch_points_used=2,
            ),
            trigger_outcome=DecodeOutcome(
                event_type="trigger",
                trigger_type="steer_boundary",
                entropy_value=None,
                assistant_prefix="prefix<steer>",
                prompt_token_ids=(1, 2, 3, 4),
                token_ids=(10, 11, 12, 13),
                token_traces=(),
                generated_tokens=4,
                stop_reason="",
            ),
            branching_enabled=True,
        )
    )
    assert outcome.event_type == "terminated"
    assert outcome.stop_reason == "max_gen_toks_reached"
    assert outcome.branch_points_used == 2


def test_single_steer_continuation_trims_text_after_close(
    tmp_path: Path, monkeypatch
) -> None:
    """Single steer continuation should trim text after first `</steer>`."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )
    executor.decoding = DecodingConfig(
        temperature=executor.decoding.temperature,
        top_p=executor.decoding.top_p,
        max_gen_toks=32,
        top_logprobs=executor.decoding.top_logprobs,
        decode_chunk_tokens=executor.decoding.decode_chunk_tokens,
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

    trim_stream_updates: list[tuple[int, ...]] = []

    def fake_trim_update_stream_state(
        *, request_stream_id: str, consumed_output_token_ids: tuple[int, ...]
    ) -> None:
        _ = request_stream_id
        trim_stream_updates.append(consumed_output_token_ids)

    monkeypatch.setattr(executor, "_generate_choice", fake_generate_choice)
    monkeypatch.setattr(
        executor,
        "_update_request_stream_state_output_ids",
        fake_trim_update_stream_state,
    )
    outcome = continue_with_single_steer_candidate(
        executor=executor,
        assistant_prefix="seed</exec>\n<steer>",
        prompt_token_ids=(1, 2, 3),
        token_ids=(),
        token_traces=(),
        generated_tokens=0,
        request_stream_id="decode:node_root",
    )

    assert outcome.event_type == "continued"
    assert "tail" not in outcome.assistant_prefix
    assert outcome.assistant_prefix.endswith("<steer>abc</steer>\n<exec>")
    assert trim_stream_updates == [()]


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
    assert children[0].assistant_prefix.endswith("unfinished</steer>\n<exec>")
    assert children[0].prompt_token_ids is not None
    assert children[0].prompt_token_ids[:5] == (1, 2, 3, 11, 12)
    assert children[0].prompt_token_ids[5:] == tuple(range(1000, 1015))
    events = read_tree_event_rows(store=executor.artifact_store)
    edge_events = [row for row in events if row.get("event_type") == "edge_selected"]
    assert edge_events
    edge_payload = dict(edge_events[-1].get("payload", {}))
    assert edge_payload.get("candidate_text_normalized", "").endswith(
        "unfinished</steer>\n<exec>"
    )


def test_steer_candidate_pool_refreshes_when_cache_disabled(
    tmp_path: Path, monkeypatch
) -> None:
    """Disabling pool cache should regenerate candidates for the same trigger."""

    executor = build_executor(tmp_path=tmp_path)
    calls: dict[str, int] = {"generate": 0}

    async def fake_candidate_budget_async(
        *,
        trigger_type: str,
        generated_tokens: int,
    ) -> int:
        _ = trigger_type, generated_tokens
        return 3

    async def fake_generate_candidate_pool_async(
        *,
        candidate_pool_id: str,
        state: PathState,
        trigger_type: str,
        entropy_value: float | None,
        assistant_prefix: str,
        prompt_token_ids: tuple[int, ...] | None,
        candidate_token_budget: int,
    ) -> CandidatePoolRecord:
        _ = (
            candidate_pool_id,
            state,
            trigger_type,
            entropy_value,
            assistant_prefix,
            prompt_token_ids,
            candidate_token_budget,
        )
        calls["generate"] += 1
        return CandidatePoolRecord(
            candidate_pool_id=f"pool_{calls['generate']}",
            branch_point_id="bp",
            node_id="node_root",
            trigger_type="steer_boundary",
            entropy_value=None,
            candidates=(
                CandidateRecord(
                    candidate_id=0,
                    text=f"c{calls['generate']}",
                    token_ids=(11,),
                    tokens=(),
                    finish_reason="stop",
                    stop_reason="</steer",
                ),
            ),
        )

    monkeypatch.setattr(
        executor,
        "_candidate_token_budget_async",
        fake_candidate_budget_async,
    )
    monkeypatch.setattr(
        executor,
        "_generate_candidate_pool_async",
        fake_generate_candidate_pool_async,
    )

    async def run_twice() -> tuple[
        CandidatePoolRecord,
        CandidatePoolRecord,
    ]:
        first = await executor._resolve_candidate_pool_async(
            doc_id=1,
            state=PathState(
                node_id="node_root",
                assistant_prefix="seed",
                prompt_token_ids=(1, 2),
                token_ids=(),
                token_traces=(),
                branch_points_used=0,
            ),
            trigger_type="steer_boundary",
            entropy_value=None,
            assistant_prefix="seed",
            prompt_token_ids=(1, 2),
            generated_tokens=2,
        )
        second = await executor._resolve_candidate_pool_async(
            doc_id=1,
            state=PathState(
                node_id="node_root",
                assistant_prefix="seed",
                prompt_token_ids=(1, 2),
                token_ids=(),
                token_traces=(),
                branch_points_used=0,
            ),
            trigger_type="steer_boundary",
            entropy_value=None,
            assistant_prefix="seed",
            prompt_token_ids=(1, 2),
            generated_tokens=2,
        )
        return (first, second)

    first_pool, second_pool = asyncio.run(run_twice())
    assert calls["generate"] == 2
    assert first_pool.candidate_pool_id != second_pool.candidate_pool_id


def test_selected_child_normalization_asserts_detokenized_alignment(
    tmp_path: Path, monkeypatch
) -> None:
    """Selected steer child normalization should fail on text/token mismatch."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
        debug_assert_text_token_alignment=True,
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


def test_selected_child_normalization_skips_alignment_asserts_by_default(
    tmp_path: Path, monkeypatch
) -> None:
    """Alignment asserts should be disabled unless debug mode enables them."""

    executor = build_executor(
        tmp_path=tmp_path,
        trigger_steer_enabled=True,
        trigger_entropy_enabled=False,
    )

    def fake_detokenize(*, model: str, token_ids: tuple[int, ...]) -> str:
        _ = model, token_ids
        return "bad"

    monkeypatch.setattr(executor.client, "detokenize", fake_detokenize, raising=False)
    normalized_text, normalized_token_ids = executor._normalized_child_candidate(
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
    assert normalized_text.startswith("bad")
    assert normalized_token_ids[:2] == (11, 12)


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
        branch_point_id="bp",
        node_id="node",
        trigger_type="steer_boundary",
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
