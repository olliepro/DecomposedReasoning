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
    async_post_retry_delay_seconds,
    build_chat_completions_payload,
    build_completions_payload,
    build_detokenize_payload,
    is_prompt_token_ids_unsupported_error,
    is_retryable_async_transport_error,
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


def test_build_completions_payload_for_text_prompt() -> None:
    """Completions payload should include text prompt settings and token-ID return flag."""
    payload = build_completions_payload(
        model="m",
        prompt="p",
        prompt_token_ids=None,
        temperature=0.7,
        top_p=0.9,
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
            self, *, path: str, payload: dict[str, Any]
        ) -> dict[str, Any]:
            _ = payload
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


def test_async_post_url_reuses_single_client_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async HTTP requests should reuse one `aiohttp.ClientSession` per client."""

    session_create_count = 0

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

        def __init__(self, *, timeout: Any) -> None:
            _ = timeout

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            _ = url, json, headers
            return FakeResponse()

        async def close(self) -> None:
            self.closed = True

    def fake_client_session(*, timeout: Any) -> FakeSession:
        nonlocal session_create_count
        session_create_count += 1
        return FakeSession(timeout=timeout)

    monkeypatch.setattr("vllm_client.aiohttp.ClientSession", fake_client_session)
    client = VllmClient(base_url="http://127.0.0.1:8000/v1")

    async def run_requests() -> None:
        first = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 1},
        )
        second = await client._post_url_async(
            url="http://127.0.0.1:8000/v1/completions",
            payload={"x": 2},
        )
        assert first == {"ok": True}
        assert second == {"ok": True}
        await client.close_async()

    asyncio.run(run_requests())
    assert session_create_count == 1


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
    assert not is_retryable_async_transport_error(error=RuntimeError("other failure"))


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
    """Async HTTP requests should not close a shared session on one dropped request."""

    session_create_count = 0
    close_count = 0
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

        def __init__(self, *, timeout: Any) -> None:
            _ = timeout
            self.post_calls = 0

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            _ = url, json, headers
            self.post_calls += 1
            if self.post_calls == 1:
                raise aiohttp.ServerDisconnectedError(message="disconnected")
            return FakeResponse()

        async def close(self) -> None:
            nonlocal close_count
            self.closed = True
            close_count += 1

    def fake_client_session(*, timeout: Any) -> FakeSession:
        nonlocal session_create_count
        session_create_count += 1
        return FakeSession(timeout=timeout)

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
    assert session_create_count == 1
    assert close_count == 1
    assert sleep_delays == [0.25]


def test_async_post_url_retries_body_read_connection_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async HTTP requests should retry when aiohttp closes while reading body."""

    session_create_count = 0
    close_count = 0
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

        def __init__(self, *, timeout: Any) -> None:
            _ = timeout
            self.post_calls = 0

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            _ = url, json, headers
            self.post_calls += 1
            return FakeResponse(fail_read=self.post_calls == 1)

        async def close(self) -> None:
            nonlocal close_count
            self.closed = True
            close_count += 1

    def fake_client_session(*, timeout: Any) -> FakeSession:
        nonlocal session_create_count
        session_create_count += 1
        return FakeSession(timeout=timeout)

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
    assert session_create_count == 1
    assert close_count == 1
    assert sleep_delays == [0.25]


def test_async_post_url_retries_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async HTTP requests should retry when an aiohttp request times out."""

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

        def __init__(self, *, timeout: Any) -> None:
            _ = timeout
            self.post_calls = 0

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            _ = url, json, headers
            self.post_calls += 1
            if self.post_calls == 1:
                raise asyncio.TimeoutError()
            return FakeResponse()

        async def close(self) -> None:
            self.closed = True

    session: FakeSession | None = None

    def fake_client_session(*, timeout: Any) -> FakeSession:
        nonlocal session
        session = FakeSession(timeout=timeout)
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
    assert session.post_calls == 2
    assert sleep_delays == [0.25]


def test_async_post_url_retries_transient_disconnects_up_to_eleventh_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async HTTP requests should tolerate ten transient disconnects."""

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

        def __init__(self, *, timeout: Any) -> None:
            _ = timeout
            self.post_calls = 0

        def post(
            self, *, url: str, json: dict[str, Any], headers: dict[str, str]
        ) -> FakeResponse:
            _ = url, json, headers
            self.post_calls += 1
            if self.post_calls < 11:
                raise aiohttp.ServerDisconnectedError(message="disconnected")
            return FakeResponse()

        async def close(self) -> None:
            self.closed = True

    session: FakeSession | None = None

    def fake_client_session(*, timeout: Any) -> FakeSession:
        nonlocal session
        session = FakeSession(timeout=timeout)
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
    assert session.post_calls == 11
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
