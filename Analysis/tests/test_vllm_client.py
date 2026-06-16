"""Tests for vLLM completions payload and response parsing helpers."""

from __future__ import annotations

import asyncio
from typing import Any

import aiohttp
import pytest

from analysis_types import RunConfig
from vllm_client import (
    ChatMessage,
    VllmClient,
    async_post_max_attempts_for_payload,
    async_post_retry_delay_seconds,
    build_chat_completions_payload,
    build_completions_payload,
    build_detokenize_payload,
    is_prompt_token_ids_unsupported_error,
    is_retryable_async_transport_error,
    is_retryable_vllm_request_error,
    normalize_vllm_base_url,
    parse_chat_completions_choices,
    parse_completions_choices,
    parse_detokenize_text,
    parse_tokenize_ids,
    VllmRequestError,
)


def test_parse_completions_choice_with_stop_reason_and_token_ids() -> None:
    """Completions parser should preserve stop reason and token-ID fields."""
    payload = {
        "prompt_token_ids": [1, 2, 3],
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": "<steer",
                "text": "abc",
                "token_ids": [10, 11],
                "logprobs": {
                    "tokens": ["a", "b"],
                    "token_logprobs": [-0.1, -0.3],
                    "top_logprobs": [{"a": -0.1}, {"b": -0.3}],
                },
            }
        ],
    }
    choices = parse_completions_choices(response_payload=payload)
    assert len(choices) == 1
    assert choices[0].stop_reason == "<steer"
    assert choices[0].prompt_token_ids == (1, 2, 3)
    assert choices[0].token_ids == (10, 11)
    assert len(choices[0].tokens) == 2


def test_parse_completions_choice_can_reuse_known_prompt_token_ids() -> None:
    """Completions parser should skip echoed prompt ids when fallback is known."""

    payload = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": None,
                "text": "abc",
                "prompt_token_ids": [99, 98, 97],
                "token_ids": [10, 11],
                "logprobs": {
                    "tokens": ["a", "b"],
                    "token_logprobs": [-0.1, -0.3],
                    "top_logprobs": [{"a": -0.1}, {"b": -0.3}],
                },
            }
        ],
    }
    choices = parse_completions_choices(
        response_payload=payload,
        fallback_prompt_token_ids=(1, 2, 3),
        parse_choice_prompt_token_ids=False,
    )
    assert len(choices) == 1
    assert choices[0].prompt_token_ids == (1, 2, 3)
    assert choices[0].token_ids == (10, 11)


def test_parse_completions_choice_truncates_generated_chat_eos() -> None:
    """Completions parser should treat generated chat EOS as terminal."""

    payload = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": None,
                "text": "answer<|im_end|>ignored",
                "token_ids": [10, 248046, 11],
                "logprobs": {
                    "tokens": ["answer", "<|im_end|>", "ignored"],
                    "token_logprobs": [-0.1, -0.2, -0.3],
                    "top_logprobs": [
                        {"answer": -0.1},
                        {"<|im_end|>": -0.2},
                        {"ignored": -0.3},
                    ],
                },
            }
        ],
    }

    (choice,) = parse_completions_choices(response_payload=payload)

    assert choice.text == "answer"
    assert choice.stop_reason == "<|im_end|>"
    assert choice.token_ids == (10,)
    assert tuple(token.token for token in choice.tokens) == ("answer",)


def test_parse_completions_choice_keeps_plain_slash_s_text() -> None:
    """Plain textual slash-s should not be treated as chat EOS."""

    payload = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": None,
                "text": "answer.</s>ignored",
                "token_ids": [10, 11, 12, 13, 14],
                "logprobs": {
                    "tokens": ["answer", ".</", "s", ">", "ignored"],
                    "token_logprobs": [-0.1, -0.2, -0.3, -0.4, -0.5],
                    "top_logprobs": [
                        {"answer": -0.1},
                        {".</": -0.2},
                        {"s": -0.3},
                        {">": -0.4},
                        {"ignored": -0.5},
                    ],
                },
            }
        ],
    }

    (choice,) = parse_completions_choices(response_payload=payload)

    assert choice.text == "answer.</s>ignored"
    assert choice.stop_reason is None
    assert choice.token_ids == (10, 11, 12, 13, 14)
    assert tuple(token.token for token in choice.tokens) == (
        "answer",
        ".</",
        "s",
        ">",
        "ignored",
    )


def test_build_completions_payload_for_text_prompt() -> None:
    """Completions payload should include text prompt settings and token-ID return flag."""
    payload = build_completions_payload(
        model="m",
        prompt="p",
        prompt_token_ids=None,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.0,
        presence_penalty=1.5,
        max_tokens=17,
        n=3,
        seed=11,
        stop=("<steer",),
        top_logprobs=12,
        priority=17,
        repetition_penalty=1.1,
    )
    assert payload["prompt"] == "p"
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["top_k"] == 20
    assert payload["min_p"] == 0.0
    assert payload["presence_penalty"] == 1.5
    assert payload["max_tokens"] == 17
    assert payload["n"] == 3
    assert payload["seed"] == 11
    assert payload["logprobs"] == 12
    assert payload["stop"] == ["<steer"]
    assert payload["include_stop_str_in_output"] is True
    assert payload["return_token_ids"] is True
    assert payload["return_prompt_token_ids"] is False
    assert payload["priority"] == 17
    assert payload["repetition_penalty"] == 1.1


def test_build_completions_payload_for_token_prompt() -> None:
    """Completions payload should support prompt-token-id continuation mode."""
    payload = build_completions_payload(
        model="m",
        prompt=None,
        prompt_token_ids=(7, 8, 9),
        temperature=0.6,
        top_p=0.8,
        max_tokens=19,
        n=2,
        seed=7,
        stop=None,
        top_logprobs=9,
    )
    assert payload["prompt"] == [7, 8, 9]
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.8
    assert payload["max_tokens"] == 19
    assert payload["n"] == 2
    assert payload["seed"] == 7
    assert payload["logprobs"] == 9
    assert payload["return_prompt_token_ids"] is False
    assert "prompt_token_ids" not in payload
    assert "include_stop_str_in_output" not in payload
    assert "priority" not in payload
    assert "repetition_penalty" not in payload


def test_build_chat_completions_payload_shape() -> None:
    """Chat payload should include messages and chat logprob settings."""

    payload = build_chat_completions_payload(
        model="m",
        messages=(ChatMessage(role="user", content="cluster these"),),
        temperature=0.2,
        top_p=0.95,
        max_tokens=64,
        n=1,
        seed=3,
        stop=None,
        top_logprobs=5,
        priority=11,
        repetition_penalty=1.03,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert payload["messages"] == [{"role": "user", "content": "cluster these"}]
    assert payload["logprobs"] is True
    assert payload["top_logprobs"] == 5
    assert payload["priority"] == 11
    assert payload["repetition_penalty"] == 1.03
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}


def test_parse_chat_completions_choice_with_message_content() -> None:
    """Chat parser should normalize assistant message content into text."""

    payload = {
        "prompt_token_ids": [1, 2, 3],
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": None,
                "message": {"role": "assistant", "content": '{"clusters":[]}'},
                "token_ids": [10, 11],
                "logprobs": {
                    "tokens": ["{", "}"],
                    "token_logprobs": [-0.1, -0.2],
                    "top_logprobs": [{"{": -0.1}, {"}": -0.2}],
                },
            }
        ],
    }
    choices = parse_chat_completions_choices(response_payload=payload)
    assert len(choices) == 1
    assert choices[0].text == '{"clusters":[]}'
    assert choices[0].prompt_token_ids == (1, 2, 3)
    assert choices[0].token_ids == (10, 11)


def test_parse_chat_completions_choice_prefers_reasoning_fields() -> None:
    """Chat parser should read Qwen reasoning fields before message content."""

    reasoning_payload = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": None,
                "message": {
                    "role": "assistant",
                    "reasoning": "reasoning text",
                    "content": "",
                },
            }
        ],
    }
    choices = parse_chat_completions_choices(response_payload=reasoning_payload)
    assert choices[0].text == "reasoning text"

    reasoning_content_payload = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": None,
                "message": {
                    "role": "assistant",
                    "reasoning_content": "reasoning content text",
                    "content": "fallback content",
                },
            }
        ],
    }
    choices = parse_chat_completions_choices(response_payload=reasoning_content_payload)
    assert choices[0].text == "reasoning content text"


def test_parse_chat_completions_choice_can_reuse_known_prompt_token_ids() -> None:
    """Chat parser should skip echoed prompt ids when fallback is known."""

    payload = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "stop_reason": None,
                "prompt_token_ids": [99, 98, 97],
                "message": {"role": "assistant", "content": '{"clusters":[]}'},
                "token_ids": [10, 11],
                "logprobs": {
                    "tokens": ["{", "}"],
                    "token_logprobs": [-0.1, -0.2],
                    "top_logprobs": [{"{": -0.1}, {"}": -0.2}],
                },
            }
        ],
    }
    choices = parse_chat_completions_choices(
        response_payload=payload,
        fallback_prompt_token_ids=(1, 2, 3),
        parse_choice_prompt_token_ids=False,
    )
    assert len(choices) == 1
    assert choices[0].prompt_token_ids == (1, 2, 3)
    assert choices[0].token_ids == (10, 11)


def test_parse_tokenize_ids_accepts_standard_key() -> None:
    """Tokenize parser should read token IDs from standard payload key."""
    token_ids = parse_tokenize_ids(response_payload={"token_ids": [1, 3, 5]})
    assert token_ids == (1, 3, 5)


def test_build_detokenize_payload_shape() -> None:
    """Detokenize payload should map token IDs to `tokens` request key."""

    payload = build_detokenize_payload(model="m", token_ids=(7, 8, 9))
    assert payload == {"model": "m", "tokens": [7, 8, 9]}


def test_parse_detokenize_text_accepts_prompt_key() -> None:
    """Detokenize parser should read decoded text from `prompt` field."""

    assert parse_detokenize_text(response_payload={"prompt": "abc"}) == "abc"


def test_normalize_vllm_base_url_accepts_bare_host_port() -> None:
    """Bare Ray server addresses should become absolute `/v1` URLs."""

    normalized = normalize_vllm_base_url(base_url="10.8.2.8:42269")
    assert normalized == "http://10.8.2.8:42269/v1"


def test_normalize_vllm_base_url_preserves_existing_v1_url() -> None:
    """Already-normalized base URLs should pass through unchanged."""

    normalized = normalize_vllm_base_url(base_url="http://127.0.0.1:8000/v1")
    assert normalized == "http://127.0.0.1:8000/v1"


def test_tokenize_posts_to_root_endpoint_only() -> None:
    """Tokenize path should be sent to root `/tokenize`, not `/v1/tokenize`."""

    class RootTokenizeClient(VllmClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://127.0.0.1:8000/v1")
            self.root_paths: list[str] = []

        def _post(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("tokenize should not call v1-relative endpoint")

        def _post_root(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            self.root_paths.append(path)
            return {"token_ids": [42, 99]}

    client = RootTokenizeClient()
    token_ids = client.tokenize(model="m", text="<steer>", add_special_tokens=False)
    assert token_ids == (42, 99)
    assert client.root_paths == ["/tokenize"]


def test_detokenize_posts_to_root_endpoint_only() -> None:
    """Detokenize path should be sent to root `/detokenize` endpoint."""

    class RootDetokenizeClient(VllmClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://127.0.0.1:8000/v1")
            self.root_paths: list[str] = []
            self.last_payload: dict[str, Any] = {}

        def _post(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("detokenize should not call v1-relative endpoint")

        def _post_root(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            self.root_paths.append(path)
            self.last_payload = payload
            return {"prompt": "<exec>\n"}

    client = RootDetokenizeClient()
    decoded = client.detokenize(model="m", token_ids=(12, 34))
    assert decoded == "<exec>\n"
    assert client.root_paths == ["/detokenize"]
    assert client.last_payload == {"model": "m", "tokens": [12, 34]}


def test_top_logprobs_cap_enforced() -> None:
    """Run config should cap requested top-logprobs at server max value."""
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        top_logprobs=40,
        max_server_logprobs=7,
    )
    assert config.capped_top_logprobs() == 7


def test_async_completions_posts_to_v1_endpoint() -> None:
    """Async completions should call `/v1/completions` via `_post_async`."""

    class AsyncCompletionsClient(VllmClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://127.0.0.1:8000/v1")
            self.paths: list[str] = []

        def _post(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("completions_async should not use sync _post")

        async def _post_async(
            self,
            *,
            path: str,
            payload: dict[str, Any],
            session_key: str | None = None,
            disable_timeout: bool = False,
        ) -> dict[str, Any]:
            _ = payload, session_key, disable_timeout
            self.paths.append(path)
            return {
                "prompt_token_ids": [1, 2],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "stop_reason": None,
                        "text": "abc",
                        "token_ids": [10],
                        "logprobs": {
                            "tokens": ["a"],
                            "token_logprobs": [-0.1],
                            "top_logprobs": [{"a": -0.1}],
                        },
                    }
                ],
            }

    client = AsyncCompletionsClient()
    choices = asyncio.run(
        client.completions_async(
            model="m",
            prompt="p",
            prompt_token_ids=None,
            temperature=0.6,
            top_p=0.95,
            max_tokens=8,
            n=1,
            seed=9,
            stop=("<steer",),
            top_logprobs=3,
        )
    )
    assert client.paths == ["/completions"]
    assert len(choices) == 1
    assert choices[0].text == "abc"


def test_async_post_url_reuses_sessions_by_key_and_recreates_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async HTTP requests should reuse and recreate sessions by logical key."""

    created_sessions: list[FakeSession] = []
    fail_next_post = False

    class FakeResponse:
        status = 200

        def __init__(self, *, session_id: int) -> None:
            self.session_id = session_id

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        async def text(self) -> str:
            return f'{{"session_id": {self.session_id}}}'

    class FakeSession:
        def __init__(self, *, connector: Any, timeout: Any, session_id: int) -> None:
            _ = connector, timeout
            self.session_id = session_id
            self.closed = False

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            nonlocal fail_next_post
            _ = url, json, headers
            if fail_next_post:
                fail_next_post = False
                raise aiohttp.ServerDisconnectedError()
            return FakeResponse(session_id=self.session_id)

        async def close(self) -> None:
            self.closed = True

    def fake_client_session(*, connector: Any, timeout: Any) -> FakeSession:
        session = FakeSession(
            connector=connector,
            timeout=timeout,
            session_id=len(created_sessions),
        )
        created_sessions.append(session)
        return session

    monkeypatch.setattr("vllm_client.aiohttp.ClientSession", fake_client_session)
    client = VllmClient(base_url="http://127.0.0.1:8000/v1")

    async def run_requests() -> None:
        first = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 1},
            session_key="branch:a",
        )
        second = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 2},
            session_key="branch:a",
        )
        third = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 3},
            session_key="branch:b",
        )
        nonlocal fail_next_post
        fail_next_post = True
        fourth = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 4},
            session_key="branch:a",
        )
        assert first == {"session_id": 0}
        assert second == {"session_id": 0}
        assert third == {"session_id": 1}
        assert fourth == {"session_id": 2}
        assert created_sessions[0].closed
        await client.close_async()

    asyncio.run(run_requests())
    assert len(created_sessions) == 3
    assert all(session.closed for session in created_sessions)


def test_prompt_token_ids_unsupported_classifier_ignores_disconnect() -> None:
    """Token-prompt support classifier should not treat disconnects as capability."""

    assert is_prompt_token_ids_unsupported_error(
        error=VllmRequestError(
            "Either prompt or prompt_embeds must be provided and non-empty."
        )
    )
    assert not is_prompt_token_ids_unsupported_error(
        error=VllmRequestError("Server disconnected")
    )


def test_retryable_async_transport_error_classifier() -> None:
    """Async retry classifier should include aiohttp body-read disconnects."""

    assert is_retryable_async_transport_error(
        error=aiohttp.ServerDisconnectedError(message="disconnected")
    )
    assert is_retryable_async_transport_error(error=asyncio.TimeoutError())
    assert is_retryable_async_transport_error(error=RuntimeError("Connection closed."))
    assert is_retryable_async_transport_error(
        error=RuntimeError("Connector is closed.")
    )
    assert not is_retryable_async_transport_error(error=RuntimeError("other failure"))


def test_retryable_vllm_request_error_classifier() -> None:
    """Request retry classifier should identify post-HTTP transport failures."""

    assert is_retryable_vllm_request_error(
        error=VllmRequestError("Server disconnected")
    )
    assert is_retryable_vllm_request_error(error=VllmRequestError("504 Gateway"))
    assert is_retryable_vllm_request_error(
        error=VllmRequestError("incomplete choices: expected 10, got 9")
    )
    assert is_retryable_vllm_request_error(
        error=VllmRequestError("invalid JSON response from vLLM")
    )
    assert is_retryable_vllm_request_error(
        error=VllmRequestError("Connector is closed.")
    )
    assert not is_retryable_vllm_request_error(error=VllmRequestError("bad prompt"))
    assert not is_retryable_vllm_request_error(error=RuntimeError("timeout"))


def test_async_post_max_attempts_for_payload_lets_batches_split() -> None:
    """Multi-choice payloads should not retry same large body internally."""

    assert async_post_max_attempts_for_payload(payload={"n": 10}) == 1
    assert async_post_max_attempts_for_payload(payload={"n": 1}) == 11
    assert async_post_max_attempts_for_payload(payload={}) == 11


def test_async_post_retry_delay_seconds_caps_exponential_backoff() -> None:
    """Async POST retry backoff should grow exponentially and then cap."""

    delays = [
        async_post_retry_delay_seconds(retry_index=retry_index)
        for retry_index in range(10)
    ]

    assert delays == [0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 8.0, 8.0, 8.0, 8.0]


def test_async_post_url_retries_connection_drop_on_same_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async HTTP requests should recreate a cached session after one drop."""

    session_create_count = 0
    close_count = 0
    post_attempts = 0
    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    class FakeResponse:
        status = 200

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        async def text(self) -> str:
            return '{"ok": true}'

    class FakeSession:
        closed = False

        def __init__(self, *, connector: Any, timeout: Any) -> None:
            _ = connector, timeout

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            nonlocal post_attempts
            _ = url, json, headers
            post_attempts += 1
            if post_attempts == 1:
                raise aiohttp.ServerDisconnectedError(message="disconnected")
            return FakeResponse()

        async def close(self) -> None:
            nonlocal close_count
            self.closed = True
            close_count += 1

    def fake_client_session(*, connector: Any, timeout: Any) -> FakeSession:
        nonlocal session_create_count
        session_create_count += 1
        return FakeSession(connector=connector, timeout=timeout)

    monkeypatch.setattr("vllm_client.aiohttp.ClientSession", fake_client_session)
    monkeypatch.setattr("vllm_client.asyncio.sleep", fake_sleep)
    client = VllmClient(base_url="http://127.0.0.1:8000/v1")

    async def run_request() -> None:
        response = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 1},
        )
        assert response == {"ok": True}
        await client.close_async()

    asyncio.run(run_request())
    assert session_create_count == 2
    assert close_count == 2
    assert sleep_delays == [0.25]


def test_async_post_url_bubbles_multi_choice_disconnect_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Large `n` generation failures should be handled by the batch splitter."""

    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    class FakeSession:
        closed = False

        def __init__(self, *, connector: Any, timeout: Any) -> None:
            _ = connector, timeout
            self.post_calls = 0

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> Any:
            _ = url, json, headers
            self.post_calls += 1
            raise aiohttp.ServerDisconnectedError(message="disconnected")

        async def close(self) -> None:
            self.closed = True

    session: FakeSession | None = None

    def fake_client_session(*, connector: Any, timeout: Any) -> FakeSession:
        nonlocal session
        session = FakeSession(connector=connector, timeout=timeout)
        return session

    monkeypatch.setattr("vllm_client.aiohttp.ClientSession", fake_client_session)
    monkeypatch.setattr("vllm_client.asyncio.sleep", fake_sleep)
    client = VllmClient(base_url="http://127.0.0.1:8000/v1")

    async def run_request() -> None:
        with pytest.raises(VllmRequestError, match="disconnected"):
            await client._post_url_async(
                url="http://127.0.0.1:8000/v1/completions",
                payload={"n": 10},
            )
        await client.close_async()

    asyncio.run(run_request())
    assert session is not None
    assert session.post_calls == 1
    assert sleep_delays == []


def test_async_post_url_retries_body_read_connection_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async HTTP requests should retry when aiohttp closes while reading body."""

    session_create_count = 0
    close_count = 0
    post_attempts = 0
    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    class FakeResponse:
        status = 200

        def __init__(self, *, fail_read: bool) -> None:
            self.fail_read = fail_read

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        async def text(self) -> str:
            if self.fail_read:
                raise RuntimeError("Connection closed.")
            return '{"ok": true}'

    class FakeSession:
        closed = False

        def __init__(self, *, connector: Any, timeout: Any) -> None:
            _ = connector, timeout

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            nonlocal post_attempts
            _ = url, json, headers
            post_attempts += 1
            return FakeResponse(fail_read=post_attempts == 1)

        async def close(self) -> None:
            nonlocal close_count
            self.closed = True
            close_count += 1

    def fake_client_session(*, connector: Any, timeout: Any) -> FakeSession:
        nonlocal session_create_count
        session_create_count += 1
        return FakeSession(connector=connector, timeout=timeout)

    monkeypatch.setattr("vllm_client.aiohttp.ClientSession", fake_client_session)
    monkeypatch.setattr("vllm_client.asyncio.sleep", fake_sleep)
    client = VllmClient(base_url="http://127.0.0.1:8000/v1")

    async def run_request() -> None:
        response = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 1},
        )
        assert response == {"ok": True}
        await client.close_async()

    asyncio.run(run_request())
    assert session_create_count == 2
    assert close_count == 2
    assert sleep_delays == [0.25]


def test_async_post_url_invalid_json_is_request_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truncated 200 OK bodies should surface as retryable request errors."""

    class FakeResponse:
        status = 200

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        async def text(self) -> str:
            return '{"choices": ['

    class FakeSession:
        closed = False

        def __init__(self, *, connector: Any, timeout: Any) -> None:
            _ = connector, timeout

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            _ = url, json, headers
            return FakeResponse()

        async def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        "vllm_client.aiohttp.ClientSession",
        lambda *, connector, timeout: FakeSession(
            connector=connector,
            timeout=timeout,
        ),
    )
    client = VllmClient(base_url="http://127.0.0.1:8000/v1")

    async def run_request() -> None:
        with pytest.raises(VllmRequestError, match="invalid JSON response"):
            await client._post_url_async(
                url="http://127.0.0.1:8000/v1/completions",
                payload={"n": 1},
            )
        await client.close_async()

    asyncio.run(run_request())


def test_async_post_url_retries_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async HTTP requests should retry when an aiohttp request times out."""

    session_create_count = 0
    post_attempts = 0
    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    class FakeResponse:
        status = 200

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        async def text(self) -> str:
            return '{"ok": true}'

    class FakeSession:
        closed = False

        def __init__(self, *, connector: Any, timeout: Any) -> None:
            _ = connector, timeout

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            nonlocal post_attempts
            _ = url, json, headers
            post_attempts += 1
            if post_attempts == 1:
                raise asyncio.TimeoutError()
            return FakeResponse()

        async def close(self) -> None:
            self.closed = True

    session: FakeSession | None = None

    def fake_client_session(*, connector: Any, timeout: Any) -> FakeSession:
        nonlocal session_create_count
        nonlocal session
        session_create_count += 1
        session = FakeSession(connector=connector, timeout=timeout)
        return session

    monkeypatch.setattr("vllm_client.aiohttp.ClientSession", fake_client_session)
    monkeypatch.setattr("vllm_client.asyncio.sleep", fake_sleep)
    client = VllmClient(base_url="http://127.0.0.1:8000/v1", timeout_seconds=600.0)

    async def run_request() -> None:
        response = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 1},
        )
        assert response == {"ok": True}
        await client.close_async()

    asyncio.run(run_request())
    assert session is not None
    assert post_attempts == 2
    assert session_create_count == 2
    assert sleep_delays == [0.25]


def test_async_post_url_can_disable_session_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long baseline requests can opt out of the client's default timeout."""

    post_timeouts: list[Any] = []

    class FakeResponse:
        status = 200

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        async def text(self) -> str:
            return '{"ok": true}'

    class FakeSession:
        closed = False

        def __init__(self, *, connector: Any, timeout: Any) -> None:
            _ = connector, timeout

        def post(
            self,
            *,
            url: str,
            json: dict[str, Any],
            headers: dict[str, str],
            timeout: Any | None = None,
        ) -> FakeResponse:
            _ = url, json, headers
            post_timeouts.append(timeout)
            return FakeResponse()

        async def close(self) -> None:
            self.closed = True

    def fake_client_session(*, connector: Any, timeout: Any) -> FakeSession:
        return FakeSession(connector=connector, timeout=timeout)

    monkeypatch.setattr("vllm_client.aiohttp.ClientSession", fake_client_session)
    client = VllmClient(base_url="http://127.0.0.1:8000/v1", timeout_seconds=600.0)

    async def run_request() -> None:
        response = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 1},
            disable_timeout=True,
        )
        assert response == {"ok": True}
        await client.close_async()

    asyncio.run(run_request())
    assert len(post_timeouts) == 1
    assert isinstance(post_timeouts[0], aiohttp.ClientTimeout)
    assert post_timeouts[0].total is None


def test_async_post_url_retries_transient_disconnects_up_to_eleventh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async HTTP requests should tolerate ten transient disconnects."""

    session_create_count = 0
    post_attempts = 0
    sleep_delays: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_delays.append(delay)

    class FakeResponse:
        status = 200

        async def __aenter__(self) -> FakeResponse:
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            _ = exc_type, exc, tb

        async def text(self) -> str:
            return '{"ok": true}'

    class FakeSession:
        closed = False

        def __init__(self, *, connector: Any, timeout: Any) -> None:
            _ = connector, timeout

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            nonlocal post_attempts
            _ = url, json, headers
            post_attempts += 1
            if post_attempts < 11:
                raise aiohttp.ServerDisconnectedError(message="disconnected")
            return FakeResponse()

        async def close(self) -> None:
            self.closed = True

    session: FakeSession | None = None

    def fake_client_session(*, connector: Any, timeout: Any) -> FakeSession:
        nonlocal session_create_count
        nonlocal session
        session_create_count += 1
        session = FakeSession(connector=connector, timeout=timeout)
        return session

    monkeypatch.setattr("vllm_client.aiohttp.ClientSession", fake_client_session)
    monkeypatch.setattr("vllm_client.asyncio.sleep", fake_sleep)
    client = VllmClient(base_url="http://127.0.0.1:8000/v1")

    async def run_request() -> None:
        response = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 1},
        )
        assert response == {"ok": True}
        await client.close_async()

    asyncio.run(run_request())
    assert session is not None
    assert post_attempts == 11
    assert session_create_count == 11
    assert sleep_delays == [
        0.25,
        0.5,
        1.0,
        2.0,
        4.0,
        8.0,
        8.0,
        8.0,
        8.0,
        8.0,
    ]


def test_chat_completions_posts_to_chat_endpoint() -> None:
    """Chat completions should call `/v1/chat/completions`."""

    class ChatCompletionsClient(VllmClient):
        def __init__(self) -> None:
            super().__init__(base_url="http://127.0.0.1:8000/v1")
            self.paths: list[str] = []

        def _post(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
            self.paths.append(path)
            assert payload["messages"] == [{"role": "user", "content": "cluster"}]
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            assert payload["stop"] == ["]}]}"]
            assert payload["include_stop_str_in_output"] is True
            return {
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "stop_reason": None,
                        "message": {"role": "assistant", "content": '{"clusters":[]}'},
                        "token_ids": [10],
                        "logprobs": {
                            "tokens": ["{"],
                            "token_logprobs": [-0.1],
                            "top_logprobs": [{"{": -0.1}],
                        },
                    }
                ]
            }

    client = ChatCompletionsClient()
    choices = client.chat_completions(
        model="m",
        messages=(ChatMessage(role="user", content="cluster"),),
        temperature=0.2,
        top_p=0.95,
        max_tokens=32,
        n=1,
        seed=0,
        stop=("]}]}",),
        top_logprobs=2,
        chat_template_kwargs={"enable_thinking": False},
    )
    assert client.paths == ["/chat/completions"]
    assert len(choices) == 1
    assert choices[0].text == '{"clusters":[]}'
