"""Minimal vLLM OpenAI-compatible client for chat and completions endpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast
from urllib import error as urllib_error
from urllib import request as urllib_request

from token_metrics import canonicalize_top_logprobs


@dataclass(frozen=True)
class ParsedToken:
    """One parsed generated token with selected and alternative logprobs.

    Args:
        token: Generated token text.
        logprob: Selected token logprob.
        top_entries: Top alternative entries as `(token, logprob)`.

    Returns:
        Dataclass containing token-level response data.
    """

    token: str
    logprob: float
    top_entries: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class GenerationChoice:
    """One parsed generation choice from chat/completions endpoints.

    Args:
        index: Choice index.
        text: Generated text.
        finish_reason: Choice finish reason.
        tokens: Parsed generated token details.

    Returns:
        Dataclass containing one choice output.
    """

    index: int
    text: str
    finish_reason: str
    tokens: tuple[ParsedToken, ...]


class VllmRequestError(RuntimeError):
    """Request error returned by vLLM OpenAI-compatible server."""


class VllmClient:
    """HTTP client wrapper for vLLM OpenAI-compatible APIs.

    Args:
        base_url: Base URL ending in `/v1`.
        timeout_seconds: HTTP request timeout seconds.

    Returns:
        Client used for chat/completions requests.

    Example:
        >>> client = VllmClient(base_url="http://127.0.0.1:8000/v1")
        >>> client.base_url.endswith('/v1')
        True
    """

    def __init__(self, *, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def completions(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
    ) -> tuple[GenerationChoice, ...]:
        """Call `/v1/completions` and parse choices.

        Args:
            model: Model name.
            prompt: Prompt text.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            max_tokens: Max generated tokens per choice.
            n: Number of choices.
            seed: Seed value.
            stop: Optional stop markers.
            top_logprobs: Top alternatives count.

        Returns:
            Parsed generation choices.
        """
        payload = build_completions_payload(
            model=model,
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
        )
        response_payload = self._post(path="/completions", payload=payload)
        return parse_completions_choices(response_payload=response_payload)

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        template_fields: dict[str, Any],
    ) -> tuple[GenerationChoice, ...]:
        """Call `/v1/chat/completions` and parse choices.

        Args:
            model: Model name.
            messages: OpenAI-format message list.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            max_tokens: Max generated tokens per choice.
            n: Number of choices.
            seed: Seed value.
            stop: Optional stop markers.
            top_logprobs: Top alternatives count.
            template_fields: Extra chat-template fields.

        Returns:
            Parsed generation choices.
        """
        payload = build_chat_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
            template_fields=template_fields,
        )
        response_payload = self._post(path="/chat/completions", payload=payload)
        return parse_chat_choices(response_payload=response_payload)

    def _post(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute JSON POST request and parse JSON response.

        Args:
            path: API path relative to base URL.
            payload: JSON payload.

        Returns:
            Parsed JSON payload.
        """
        url = self.base_url + path
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url=url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib_request.urlopen(
                request, timeout=self.timeout_seconds
            ) as response:
                response_text = response.read().decode("utf-8")
        except urllib_error.HTTPError as http_error:
            error_text = http_error.read().decode("utf-8", errors="replace")
            raise VllmRequestError(error_text) from http_error
        except urllib_error.URLError as url_error:
            raise VllmRequestError(str(url_error)) from url_error
        payload_obj = json.loads(response_text)
        assert isinstance(payload_obj, dict), "response must be a JSON object"
        return payload_obj


def build_completions_payload(
    *,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
    seed: int,
    stop: tuple[str, ...] | None,
    top_logprobs: int,
) -> dict[str, Any]:
    """Build payload for `/v1/completions`.

    Args:
        model: Model name.
        prompt: Prompt text.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        max_tokens: Max generated tokens.
        n: Number of choices.
        seed: Seed value.
        stop: Optional stop markers.
        top_logprobs: Top alternatives count.

    Returns:
        JSON-ready request payload.
    """
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,
    }
    if top_logprobs > 0:
        payload["logprobs"] = top_logprobs
    if stop is not None:
        payload["stop"] = list(stop)
    return payload


def build_chat_payload(
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
    seed: int,
    stop: tuple[str, ...] | None,
    top_logprobs: int,
    template_fields: dict[str, Any],
) -> dict[str, Any]:
    """Build payload for `/v1/chat/completions`.

    Args:
        model: Model name.
        messages: OpenAI-style message list.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        max_tokens: Max generated tokens.
        n: Number of choices.
        seed: Seed value.
        stop: Optional stop markers.
        top_logprobs: Top alternatives count.
        template_fields: Extra chat-template request fields.

    Returns:
        JSON-ready request payload.
    """
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,
    }
    payload.update(template_fields)
    if top_logprobs > 0:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    if stop is not None:
        payload["stop"] = list(stop)
    return payload


def parse_completions_choices(
    *, response_payload: dict[str, Any]
) -> tuple[GenerationChoice, ...]:
    """Parse completion choices from response payload.

    Args:
        response_payload: JSON payload from `/v1/completions`.

    Returns:
        Parsed generation choices.
    """
    raw_choices = _require_choices(response_payload=response_payload)
    parsed_choices = [
        _parse_one_completion_choice(choice=choice) for choice in raw_choices
    ]
    return tuple(parsed_choices)


def parse_chat_choices(
    *, response_payload: dict[str, Any]
) -> tuple[GenerationChoice, ...]:
    """Parse chat choices from response payload.

    Args:
        response_payload: JSON payload from `/v1/chat/completions`.

    Returns:
        Parsed generation choices.
    """
    raw_choices = _require_choices(response_payload=response_payload)
    parsed_choices = [_parse_one_chat_choice(choice=choice) for choice in raw_choices]
    return tuple(parsed_choices)


def _require_choices(*, response_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract and validate `choices` array from response payload.

    Args:
        response_payload: API response object.

    Returns:
        Validated list of choice payloads.
    """
    choices = response_payload.get("choices", [])
    assert isinstance(choices, list), "response choices must be a list"
    return [choice for choice in choices if isinstance(choice, dict)]


def _parse_one_completion_choice(*, choice: dict[str, Any]) -> GenerationChoice:
    """Parse one completion choice.

    Args:
        choice: One choice payload from `/v1/completions`.

    Returns:
        Parsed generation choice.
    """
    logprobs_candidate = choice.get("logprobs")
    raw_logprobs = (
        cast(dict[str, Any], logprobs_candidate)
        if isinstance(logprobs_candidate, dict)
        else {}
    )
    tokens = _parse_completion_tokens(raw_logprobs=raw_logprobs)
    text = str(choice.get("text", ""))
    finish_reason = str(choice.get("finish_reason", "unknown"))
    index = int(choice.get("index", 0))
    return GenerationChoice(
        index=index, text=text, finish_reason=finish_reason, tokens=tokens
    )


def _parse_one_chat_choice(*, choice: dict[str, Any]) -> GenerationChoice:
    """Parse one chat completion choice.

    Args:
        choice: One choice payload from `/v1/chat/completions`.

    Returns:
        Parsed generation choice.
    """
    message_candidate = choice.get("message")
    message = (
        cast(dict[str, Any], message_candidate)
        if isinstance(message_candidate, dict)
        else {}
    )
    text = str(message.get("content", ""))
    finish_reason = str(choice.get("finish_reason", "unknown"))
    index = int(choice.get("index", 0))
    tokens = _parse_chat_tokens(choice=choice)
    return GenerationChoice(
        index=index, text=text, finish_reason=finish_reason, tokens=tokens
    )


def _parse_completion_tokens(
    *, raw_logprobs: dict[str, Any]
) -> tuple[ParsedToken, ...]:
    """Parse token records from completion logprob payload shape.

    Args:
        raw_logprobs: `choice.logprobs` payload.

    Returns:
        Parsed token sequence.
    """
    raw_tokens = (
        raw_logprobs.get("tokens", [])
        if isinstance(raw_logprobs.get("tokens"), list)
        else []
    )
    raw_token_logprobs = raw_logprobs.get("token_logprobs", [])
    raw_top_logprobs = (
        raw_logprobs.get("top_logprobs", [])
        if isinstance(raw_logprobs.get("top_logprobs"), list)
        else []
    )
    parsed_tokens: list[ParsedToken] = []
    for token_index, token in enumerate(raw_tokens):
        logprob = _logprob_at(
            raw_token_logprobs=raw_token_logprobs, token_index=token_index
        )
        top_entries = canonicalize_top_logprobs(
            raw_top_logprobs=_at(raw_list=raw_top_logprobs, token_index=token_index)
        )
        parsed_tokens.append(
            ParsedToken(
                token=str(token), logprob=logprob, top_entries=tuple(top_entries)
            )
        )
    return tuple(parsed_tokens)


def _parse_chat_tokens(*, choice: dict[str, Any]) -> tuple[ParsedToken, ...]:
    """Parse token records from chat logprob payload shape.

    Args:
        choice: Choice payload from `/v1/chat/completions`.

    Returns:
        Parsed token sequence.
    """
    logprobs_candidate = choice.get("logprobs")
    logprobs_payload = (
        cast(dict[str, Any], logprobs_candidate)
        if isinstance(logprobs_candidate, dict)
        else {}
    )
    content_items = (
        logprobs_payload.get("content", [])
        if isinstance(logprobs_payload.get("content"), list)
        else []
    )
    parsed_tokens: list[ParsedToken] = []
    for item in content_items:
        if not isinstance(item, dict):
            continue
        token = str(item.get("token", ""))
        logprob = float(item.get("logprob", -1e9))
        top_entries = canonicalize_top_logprobs(
            raw_top_logprobs=item.get("top_logprobs")
        )
        parsed_tokens.append(
            ParsedToken(token=token, logprob=logprob, top_entries=tuple(top_entries))
        )
    return tuple(parsed_tokens)


def _logprob_at(*, raw_token_logprobs: object, token_index: int) -> float:
    """Read one token logprob from possibly malformed payload.

    Args:
        raw_token_logprobs: Token logprob list candidate.
        token_index: Token index.

    Returns:
        Parsed logprob value.
    """
    if not isinstance(raw_token_logprobs, list):
        return -1e9
    entry = _at(raw_list=raw_token_logprobs, token_index=token_index)
    if entry is None:
        return -1e9
    if not isinstance(entry, (float, int, str)):
        return -1e9
    return float(entry)


def _at(*, raw_list: list[object], token_index: int) -> object | None:
    """Safely index list and return item when present.

    Args:
        raw_list: Source list.
        token_index: Requested index.

    Returns:
        Item at index or `None`.
    """
    if token_index < 0 or token_index >= len(raw_list):
        return None
    return raw_list[token_index]


def is_chat_template_error(*, error_text: str) -> bool:
    """Detect template-related chat API errors for fallback decisions.

    Args:
        error_text: Error response body text.

    Returns:
        `True` when error appears template-related.
    """
    lowered = error_text.lower()
    return "template" in lowered and "chat" in lowered
