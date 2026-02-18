"""Minimal vLLM OpenAI-compatible client for completions and tokenization."""

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
    """One parsed generation choice from `/v1/completions`.

    Args:
        index: Choice index.
        text: Generated text.
        finish_reason: Choice finish reason.
        stop_reason: Stop reason returned by vLLM, when available.
        tokens: Parsed generated token details.
        prompt_token_ids: Prompt token IDs used for this request.
        token_ids: Output token IDs generated for this choice.

    Returns:
        Dataclass containing one choice output.
    """

    index: int
    text: str
    finish_reason: str
    stop_reason: int | str | None
    tokens: tuple[ParsedToken, ...]
    prompt_token_ids: tuple[int, ...] | None
    token_ids: tuple[int, ...] | None


class VllmRequestError(RuntimeError):
    """Request error returned by vLLM OpenAI-compatible server."""


class VllmClient:
    """HTTP client wrapper for vLLM OpenAI-compatible APIs.

    Args:
        base_url: Base URL ending in `/v1`.
        timeout_seconds: HTTP request timeout seconds.

    Returns:
        Client used for completions requests and tokenization.

    Example:
        >>> client = VllmClient(base_url="http://127.0.0.1:8000/v1")
        >>> client.base_url.endswith('/v1')
        True
    """

    def __init__(self, *, base_url: str, timeout_seconds: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.supports_prompt_token_ids: bool | None = None

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
    ) -> tuple[GenerationChoice, ...]:
        """Call `/v1/completions` and parse choices.

        Args:
            model: Model name.
            prompt: Prompt text when prompting via text.
            prompt_token_ids: Prompt token IDs when prompting via token space.
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
            prompt_token_ids=prompt_token_ids,
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

    def tokenize(
        self,
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        """Tokenize one text fragment with vLLM server tokenizer.

        Args:
            model: Model name.
            text: Text fragment to tokenize.
            add_special_tokens: Whether tokenizer should add special tokens.

        Returns:
            Token IDs for the fragment.
        """
        payload = build_tokenize_payload(
            model=model,
            text=text,
            add_special_tokens=add_special_tokens,
        )
        response_payload = self._post_root(path="/tokenize", payload=payload)
        token_ids = parse_tokenize_ids(response_payload=response_payload)
        return token_ids

    def _post(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute JSON POST request and parse JSON response.

        Args:
            path: API path relative to base URL.
            payload: JSON payload.

        Returns:
            Parsed JSON payload.
        """
        url = self.base_url + path
        return self._post_url(url=url, payload=payload)

    def _post_root(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute JSON POST request against server root instead of base-url endpoint.

        Args:
            path: API path relative to server root.
            payload: JSON payload.

        Returns:
            Parsed JSON payload.
        """
        root_url = self.base_url
        if root_url.endswith("/v1"):
            root_url = root_url[:-3]
        url = root_url + path
        return self._post_url(url=url, payload=payload)

    def _post_url(self, *, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Execute JSON POST request to an explicit URL and parse response.

        Args:
            url: Absolute request URL.
            payload: JSON payload.

        Returns:
            Parsed JSON payload.
        """
        body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
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
    prompt: str | None,
    prompt_token_ids: tuple[int, ...] | None,
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
        prompt: Prompt text when using text prompting.
        prompt_token_ids: Prompt IDs when using token-space prompting.
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
    has_prompt_text = prompt is not None
    has_prompt_ids = prompt_token_ids is not None
    assert (
        has_prompt_text != has_prompt_ids
    ), "provide exactly one prompt representation"
    payload: dict[str, Any] = {
        "model": model,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,
        "return_token_ids": True,
    }
    if prompt is not None:
        payload["prompt"] = prompt
    if prompt_token_ids is not None:
        # vLLM legacy `/v1/completions` accepts token prompts through `prompt`.
        payload["prompt"] = list(prompt_token_ids)
    if top_logprobs > 0:
        payload["logprobs"] = top_logprobs
    if stop is not None:
        payload["stop"] = list(stop)
        payload["include_stop_str_in_output"] = True
    return payload


def build_tokenize_payload(
    *,
    model: str,
    text: str,
    add_special_tokens: bool,
) -> dict[str, Any]:
    """Build payload for `/tokenize`.

    Args:
        model: Model name.
        text: Text to tokenize.
        add_special_tokens: Whether special tokens should be added.

    Returns:
        JSON-ready request payload.
    """
    return {
        "model": model,
        "prompt": text,
        "add_special_tokens": add_special_tokens,
    }


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
    prompt_ids = _parse_token_ids(raw_value=response_payload.get("prompt_token_ids"))
    parsed_choices = [
        _parse_one_completion_choice(
            choice=choice,
            fallback_prompt_token_ids=prompt_ids,
        )
        for choice in raw_choices
    ]
    return tuple(parsed_choices)


def parse_tokenize_ids(*, response_payload: dict[str, Any]) -> tuple[int, ...]:
    """Parse token IDs from `/tokenize` response payload.

    Args:
        response_payload: JSON payload from `/tokenize`.

    Returns:
        Parsed token ID tuple.
    """
    for key in ("token_ids", "tokens", "ids"):
        parsed = _parse_token_ids(raw_value=response_payload.get(key))
        if parsed is not None:
            return parsed
    raise AssertionError("tokenize response missing token IDs")


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


def _parse_one_completion_choice(
    *,
    choice: dict[str, Any],
    fallback_prompt_token_ids: tuple[int, ...] | None,
) -> GenerationChoice:
    """Parse one completion choice.

    Args:
        choice: One choice payload from `/v1/completions`.
        fallback_prompt_token_ids: Request-level prompt IDs when provided.

    Returns:
        Parsed generation choice.
    """
    logprobs_candidate = choice.get("logprobs")
    raw_logprobs = (
        cast(dict[str, Any], logprobs_candidate)
        if isinstance(logprobs_candidate, dict)
        else {}
    )
    stop_reason = _parse_stop_reason(raw_value=choice.get("stop_reason"))
    prompt_token_ids = _parse_token_ids(raw_value=choice.get("prompt_token_ids"))
    if prompt_token_ids is None:
        prompt_token_ids = fallback_prompt_token_ids
    token_ids = _parse_choice_token_ids(choice=choice)
    tokens = _parse_completion_tokens(raw_logprobs=raw_logprobs)
    text = str(choice.get("text", ""))
    finish_reason = str(choice.get("finish_reason", "unknown"))
    index = int(choice.get("index", 0))
    return GenerationChoice(
        index=index,
        text=text,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        tokens=tokens,
        prompt_token_ids=prompt_token_ids,
        token_ids=token_ids,
    )


def _parse_choice_token_ids(*, choice: dict[str, Any]) -> tuple[int, ...] | None:
    """Parse generated token IDs from one completion choice payload.

    Args:
        choice: Choice payload from `/v1/completions`.

    Returns:
        Output token IDs, when available.
    """
    for key in ("token_ids", "output_token_ids"):
        parsed = _parse_token_ids(raw_value=choice.get(key))
        if parsed is not None:
            return parsed
    return None


def _parse_stop_reason(*, raw_value: object) -> int | str | None:
    """Parse stop-reason field from completion choice payload.

    Args:
        raw_value: Raw stop-reason value.

    Returns:
        Parsed stop reason as `int`, `str`, or `None`.
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, (int, str)):
        return raw_value
    return str(raw_value)


def _parse_token_ids(*, raw_value: object) -> tuple[int, ...] | None:
    """Parse token ID arrays from possibly nested payload shapes.

    Args:
        raw_value: Raw JSON field candidate.

    Returns:
        Parsed token IDs, or `None` when unavailable.
    """
    if isinstance(raw_value, list) and raw_value and isinstance(raw_value[0], list):
        inner = raw_value[0]
        if not isinstance(inner, list):
            return None
        return _parse_token_ids(raw_value=inner)
    if not isinstance(raw_value, list):
        return None
    token_ids: list[int] = []
    for item in raw_value:
        if not isinstance(item, (int, float, str)):
            return None
        token_ids.append(int(item))
    return tuple(token_ids)


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
            raw_token_logprobs=raw_token_logprobs,
            token_index=token_index,
        )
        top_entries = canonicalize_top_logprobs(
            raw_top_logprobs=_at(raw_list=raw_top_logprobs, token_index=token_index)
        )
        parsed_tokens.append(
            ParsedToken(
                token=str(token),
                logprob=logprob,
                top_entries=tuple(top_entries),
            )
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
