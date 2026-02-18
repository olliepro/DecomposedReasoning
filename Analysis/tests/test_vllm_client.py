"""Tests for vLLM completions payload and response parsing helpers."""

from __future__ import annotations

from typing import Any

from analysis_types import RunConfig
from vllm_client import (
    VllmClient,
    build_completions_payload,
    parse_completions_choices,
    parse_tokenize_ids,
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
    assert "prompt_token_ids" not in payload
    assert "include_stop_str_in_output" not in payload


def test_parse_tokenize_ids_accepts_standard_key() -> None:
    """Tokenize parser should read token IDs from standard payload key."""
    token_ids = parse_tokenize_ids(response_payload={"token_ids": [1, 3, 5]})
    assert token_ids == (1, 3, 5)


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
