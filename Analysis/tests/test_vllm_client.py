"""Tests for vLLM request payload and response parsing helpers."""

from __future__ import annotations

from analysis_types import ApiModeConfig, RunConfig
from vllm_client import (
    build_chat_payload,
    build_completions_payload,
    parse_chat_choices,
)


def test_chat_logprobs_shape_parsing() -> None:
    """Chat parser should handle token alternatives when present or missing."""
    payload = {
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "message": {"content": "ab"},
                "logprobs": {
                    "content": [
                        {
                            "token": "a",
                            "logprob": -0.1,
                            "top_logprobs": [
                                {"token": "a", "logprob": -0.1},
                                {"token": "b", "logprob": -1.2},
                            ],
                        },
                        {"token": "b", "logprob": -0.5},
                    ]
                },
            }
        ]
    }
    choices = parse_chat_choices(response_payload=payload)
    assert len(choices) == 1
    assert len(choices[0].tokens) == 2
    assert len(choices[0].tokens[0].top_entries) == 2
    assert len(choices[0].tokens[1].top_entries) == 0


def test_generation_defaults_explicit_override_for_completions_payload() -> None:
    """Completions payload should preserve explicit generation settings."""
    payload = build_completions_payload(
        model="m",
        prompt="p",
        temperature=0.7,
        top_p=0.9,
        max_tokens=17,
        n=3,
        seed=11,
        stop=("</think>",),
        top_logprobs=12,
    )
    assert payload["temperature"] == 0.7
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 17
    assert payload["n"] == 3
    assert payload["seed"] == 11
    assert payload["logprobs"] == 12


def test_generation_defaults_explicit_override_for_chat_payload() -> None:
    """Chat payload should preserve explicit generation settings."""
    payload = build_chat_payload(
        model="m",
        messages=[{"role": "user", "content": "p"}],
        temperature=0.6,
        top_p=0.8,
        max_tokens=19,
        n=2,
        seed=7,
        stop=None,
        top_logprobs=9,
        template_fields={"add_generation_prompt": True},
    )
    assert payload["temperature"] == 0.6
    assert payload["top_p"] == 0.8
    assert payload["max_tokens"] == 19
    assert payload["n"] == 2
    assert payload["seed"] == 7
    assert payload["top_logprobs"] == 9
    assert payload["logprobs"] is True


def test_top_logprobs_cap_enforced() -> None:
    """Run config should cap requested top-logprobs at server max value."""
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        top_logprobs=40,
        api_mode_config=ApiModeConfig(max_server_logprobs=7),
    )
    assert config.capped_top_logprobs() == 7
