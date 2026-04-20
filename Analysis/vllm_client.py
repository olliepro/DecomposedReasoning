"""Minimal vLLM OpenAI-compatible client for completions and tokenization."""

from __future__ import annotations

import asyncio
import aiohttp
import json
from dataclasses import dataclass
from typing import Any, Union, cast
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


@dataclass(frozen=True)
class ChatMessage:
    """One chat message payload for `/v1/chat/completions`.

    Args:
        role: OpenAI-style chat role.
        content: Message body text.

    Returns:
        Dataclass containing one chat message row.
    """

    role: str
    content: str


ChatTemplateKwargs = dict[str, Union[bool, int, float, str, None]]


class VllmRequestError(RuntimeError):
    """Request error returned by vLLM OpenAI-compatible server."""


def normalize_vllm_base_url(*, base_url: str) -> str:
    """Normalize a raw vLLM server address into an absolute `/v1` base URL.

    Args:
        base_url: Raw host:port string or HTTP base URL.

    Returns:
        Absolute base URL ending in `/v1`.

    Example:
        >>> normalize_vllm_base_url(base_url="127.0.0.1:8000")
        'http://127.0.0.1:8000/v1'
    """

    normalized_url = base_url.strip().rstrip("/")
    assert normalized_url, "base_url must not be empty"
    if "://" not in normalized_url:
        normalized_url = f"http://{normalized_url}"
    if normalized_url.endswith("/v1"):
        return normalized_url
    return f"{normalized_url}/v1"


class VllmClient:
    """HTTP client wrapper for vLLM OpenAI-compatible APIs.

    Args:
        base_url: Base URL ending in `/v1`.
        timeout_seconds: HTTP request timeout seconds. `None` disables timeout.

    Returns:
        Client used for completions requests and tokenization.

    Example:
        >>> client = VllmClient(base_url="http://127.0.0.1:8000/v1")
        >>> client.base_url.endswith('/v1')
        True
    """

    def __init__(self, *, base_url: str, timeout_seconds: float | None = None) -> None:
        self.base_url = normalize_vllm_base_url(base_url=base_url)
        self.timeout_seconds = timeout_seconds
        self.supports_prompt_token_ids: bool | None = None
        self._async_session: aiohttp.ClientSession | None = None
        self._async_session_loop_id: int | None = None
        self._async_session_lock: asyncio.Lock | None = None
        self._async_session_lock_loop_id: int | None = None

    def _async_timeout(self) -> aiohttp.ClientTimeout:
        """Return configured timeout object for async HTTP requests.

        Args:
            None.

        Returns:
            `aiohttp.ClientTimeout` derived from `timeout_seconds`.
        """

        client_timeout_ctor = cast(Any, aiohttp.ClientTimeout)
        if self.timeout_seconds is None:
            return client_timeout_ctor(total=None)
        return client_timeout_ctor(total=self.timeout_seconds)

    def _new_async_session(self) -> aiohttp.ClientSession:
        """Create one async HTTP session with the configured timeout.

        Args:
            None.

        Returns:
            Fresh `aiohttp.ClientSession` for async vLLM requests.
        """

        return aiohttp.ClientSession(timeout=self._async_timeout())

    def _get_async_session_lock(self) -> asyncio.Lock:
        """Return the lifecycle lock for the current event loop.

        Args:
            None.

        Returns:
            Async lock guarding session creation and replacement.
        """

        loop_id = id(asyncio.get_running_loop())
        lock = self._async_session_lock
        if lock is not None and self._async_session_lock_loop_id == loop_id:
            return lock
        lock = asyncio.Lock()
        self._async_session_lock = lock
        self._async_session_lock_loop_id = loop_id
        return lock

    def _take_async_session(
        self, *, expected_session: aiohttp.ClientSession | None = None
    ) -> aiohttp.ClientSession | None:
        """Detach the cached async session from this client.

        Args:
            expected_session: Optional session identity to require before detach.

        Returns:
            Previously cached session, when present.
        """

        session = self._async_session
        if expected_session is not None and session is not expected_session:
            return None
        self._async_session = None
        self._async_session_loop_id = None
        return session

    async def _close_session_async(
        self, *, session: aiohttp.ClientSession | None
    ) -> None:
        """Close one async session when it is still open.

        Args:
            session: Cached session to close.

        Returns:
            None.
        """

        if session is None or session.closed:
            return
        await session.close()

    async def _drop_async_session_unlocked(
        self, *, expected_session: aiohttp.ClientSession | None = None
    ) -> None:
        """Clear and close one cached async session without taking the lock.

        Args:
            expected_session: Optional session identity to require before drop.

        Returns:
            None.
        """

        session = self._take_async_session(expected_session=expected_session)
        await self._close_session_async(session=session)

    async def _drop_async_session(
        self, *, expected_session: aiohttp.ClientSession | None = None
    ) -> None:
        """Clear and close the cached async session, if any.

        Args:
            expected_session: Optional session identity to require before drop.

        Returns:
            None.
        """

        async with self._get_async_session_lock():
            await self._drop_async_session_unlocked(expected_session=expected_session)

    async def _get_async_session(self) -> aiohttp.ClientSession:
        """Return one reusable async session for the active event loop.

        Args:
            None.

        Returns:
            Open `aiohttp.ClientSession` for reuse by async requests.
        """

        loop_id = id(asyncio.get_running_loop())
        async with self._get_async_session_lock():
            session = self._async_session
            if session is not None and session.closed:
                self._take_async_session(expected_session=session)
                session = None
            if session is not None and self._async_session_loop_id == loop_id:
                return session
            if session is not None:
                await self._drop_async_session_unlocked(expected_session=session)
            session = self._new_async_session()
            self._async_session = session
            self._async_session_loop_id = loop_id
            return session

    async def close_async(self) -> None:
        """Close the reusable async session when present.

        Args:
            None.

        Returns:
            None.
        """

        async with self._get_async_session_lock():
            await self._drop_async_session_unlocked()

    async def _post_url_with_session_async(
        self,
        *,
        session: aiohttp.ClientSession,
        url: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one async JSON POST request with a specific session.

        Args:
            session: Session used for this request attempt.
            url: Absolute request URL.
            payload: JSON payload.

        Returns:
            Parsed JSON payload from the response body.
        """
        async with session.post(
            url=url,
            json=payload,
            headers={"Content-Type": "application/json"},
        ) as response:
            response_text = await response.text()
            if response.status >= 400:
                raise VllmRequestError(response_text)
        payload_obj = json.loads(response_text)
        assert isinstance(payload_obj, dict), "response must be a JSON object"
        return payload_obj

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
            priority: Optional request priority for scheduler-policy `priority`.
            repetition_penalty: Optional repetition penalty for generated tokens.

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
            priority=priority,
            repetition_penalty=repetition_penalty,
        )
        response_payload = self._post(path="/completions", payload=payload)
        return parse_completions_choices(response_payload=response_payload)

    def chat_completions(
        self,
        *,
        model: str,
        messages: tuple[ChatMessage, ...],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        chat_template_kwargs: ChatTemplateKwargs | None = None,
    ) -> tuple[GenerationChoice, ...]:
        """Call `/v1/chat/completions` and parse choices.

        Args:
            model: Model name.
            messages: Ordered chat messages.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            max_tokens: Max generated tokens per choice.
            n: Number of choices.
            seed: Seed value.
            stop: Optional stop markers.
            top_logprobs: Top alternatives count.
            priority: Optional request priority for scheduler-policy `priority`.
            repetition_penalty: Optional repetition penalty for generated tokens.
            chat_template_kwargs: Optional chat-template override mapping passed
                through to vLLM.

        Returns:
            Parsed generation choices.
        """

        payload = build_chat_completions_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
            priority=priority,
            repetition_penalty=repetition_penalty,
            chat_template_kwargs=chat_template_kwargs,
        )
        response_payload = self._post(path="/chat/completions", payload=payload)
        return parse_chat_completions_choices(response_payload=response_payload)

    async def chat_completions_async(
        self,
        *,
        model: str,
        messages: tuple[ChatMessage, ...],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        priority: int | None = None,
        repetition_penalty: float | None = None,
        chat_template_kwargs: ChatTemplateKwargs | None = None,
    ) -> tuple[GenerationChoice, ...]:
        """Call `/v1/chat/completions` asynchronously and parse choices.

        Args:
            model: Model name.
            messages: Ordered chat messages.
            temperature: Sampling temperature.
            top_p: Nucleus sampling value.
            max_tokens: Max generated tokens per choice.
            n: Number of choices.
            seed: Seed value.
            stop: Optional stop markers.
            top_logprobs: Top alternatives count.
            priority: Optional request priority for scheduler-policy `priority`.
            repetition_penalty: Optional repetition penalty for generated tokens.
            chat_template_kwargs: Optional chat-template override mapping passed
                through to vLLM.

        Returns:
            Parsed generation choices.
        """

        payload = build_chat_completions_payload(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
            priority=priority,
            repetition_penalty=repetition_penalty,
            chat_template_kwargs=chat_template_kwargs,
        )
        response_payload = await self._post_async(
            path="/chat/completions", payload=payload
        )
        return parse_chat_completions_choices(response_payload=response_payload)

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
        """Call `/v1/completions` asynchronously and parse choices.

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
            priority: Optional request priority for scheduler-policy `priority`.
            repetition_penalty: Optional repetition penalty for generated tokens.

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
            priority=priority,
            repetition_penalty=repetition_penalty,
        )
        response_payload = await self._post_async(path="/completions", payload=payload)
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

    async def tokenize_async(
        self,
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        """Tokenize one text fragment asynchronously with vLLM tokenizer.

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
        response_payload = await self._post_root_async(
            path="/tokenize", payload=payload
        )
        return parse_tokenize_ids(response_payload=response_payload)

    def detokenize(
        self,
        *,
        model: str,
        token_ids: tuple[int, ...],
    ) -> str:
        """Detokenize one token-id sequence with vLLM server tokenizer.

        Args:
            model: Model name.
            token_ids: Token IDs to detokenize.

        Returns:
            Decoded text for the token sequence.
        """
        payload = build_detokenize_payload(
            model=model,
            token_ids=token_ids,
        )
        response_payload = self._post_root(path="/detokenize", payload=payload)
        return parse_detokenize_text(response_payload=response_payload)

    async def detokenize_async(
        self,
        *,
        model: str,
        token_ids: tuple[int, ...],
    ) -> str:
        """Detokenize one token-id sequence asynchronously with vLLM tokenizer.

        Args:
            model: Model name.
            token_ids: Token IDs to detokenize.

        Returns:
            Decoded text for the token sequence.
        """

        payload = build_detokenize_payload(
            model=model,
            token_ids=token_ids,
        )
        response_payload = await self._post_root_async(
            path="/detokenize", payload=payload
        )
        return parse_detokenize_text(response_payload=response_payload)

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

    async def _post_async(
        self, *, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute async JSON POST request and parse JSON response.

        Args:
            path: API path relative to base URL.
            payload: JSON payload.

        Returns:
            Parsed JSON payload.
        """

        url = self.base_url + path
        return await self._post_url_async(url=url, payload=payload)

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

    async def _post_root_async(
        self, *, path: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute async JSON POST request against server root endpoint.

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
        return await self._post_url_async(url=url, payload=payload)

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
        request_kwargs: dict[str, Any] = {}
        if self.timeout_seconds is not None:
            request_kwargs["timeout"] = self.timeout_seconds
        try:
            with urllib_request.urlopen(request, **request_kwargs) as response:
                response_text = response.read().decode("utf-8")
        except urllib_error.HTTPError as http_error:
            error_text = http_error.read().decode("utf-8", errors="replace")
            raise VllmRequestError(error_text) from http_error
        except urllib_error.URLError as url_error:
            raise VllmRequestError(str(url_error)) from url_error
        payload_obj = json.loads(response_text)
        assert isinstance(payload_obj, dict), "response must be a JSON object"
        return payload_obj

    async def _post_url_async(
        self, *, url: str, payload: dict[str, Any]
    ) -> dict[str, Any]:
        """Execute async JSON POST request to an explicit URL.

        Args:
            url: Absolute request URL.
            payload: JSON payload.

        Returns:
            Parsed JSON payload.
        """

        for attempt_index in range(2):
            session = await self._get_async_session()
            try:
                return await self._post_url_with_session_async(
                    session=session,
                    url=url,
                    payload=payload,
                )
            except aiohttp.ClientConnectionError as client_error:
                await self._drop_async_session(expected_session=session)
                if attempt_index == 0:
                    continue
                raise VllmRequestError(str(client_error)) from client_error
            except aiohttp.ClientError as client_error:
                raise VllmRequestError(str(client_error)) from client_error
        raise AssertionError("async vLLM request retry loop exhausted unexpectedly")


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
    priority: int | None = None,
    repetition_penalty: float | None = None,
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
        priority: Optional request priority for scheduler-policy `priority`.
        repetition_penalty: Optional repetition penalty for generated tokens.

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
    if priority is not None:
        payload["priority"] = int(priority)
    if repetition_penalty is not None:
        payload["repetition_penalty"] = float(repetition_penalty)
    return payload


def build_chat_completions_payload(
    *,
    model: str,
    messages: tuple[ChatMessage, ...],
    temperature: float,
    top_p: float,
    max_tokens: int,
    n: int,
    seed: int,
    stop: tuple[str, ...] | None,
    top_logprobs: int,
    priority: int | None = None,
    repetition_penalty: float | None = None,
    chat_template_kwargs: ChatTemplateKwargs | None = None,
) -> dict[str, Any]:
    """Build payload for `/v1/chat/completions`.

    Args:
        model: Model name.
        messages: Ordered chat messages.
        temperature: Sampling temperature.
        top_p: Nucleus sampling value.
        max_tokens: Max generated tokens.
        n: Number of choices.
        seed: Seed value.
        stop: Optional stop markers.
        top_logprobs: Top alternatives count.
        priority: Optional request priority for scheduler-policy `priority`.
        repetition_penalty: Optional repetition penalty for generated tokens.
        chat_template_kwargs: Optional chat-template override mapping passed to
            the vLLM server.

    Returns:
        JSON-ready request payload.
    """

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": message.role, "content": message.content} for message in messages
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "n": n,
        "seed": seed,
    }
    if top_logprobs > 0:
        payload["logprobs"] = True
        payload["top_logprobs"] = top_logprobs
    if stop is not None:
        payload["stop"] = list(stop)
        payload["include_stop_str_in_output"] = True
    if priority is not None:
        payload["priority"] = int(priority)
    if repetition_penalty is not None:
        payload["repetition_penalty"] = float(repetition_penalty)
    if chat_template_kwargs is not None:
        payload["chat_template_kwargs"] = dict(chat_template_kwargs)
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


def build_detokenize_payload(
    *,
    model: str,
    token_ids: tuple[int, ...],
) -> dict[str, Any]:
    """Build payload for `/detokenize`.

    Args:
        model: Model name.
        token_ids: Token IDs to detokenize.

    Returns:
        JSON-ready request payload.
    """
    return {
        "model": model,
        "tokens": list(token_ids),
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


def parse_chat_completions_choices(
    *, response_payload: dict[str, Any]
) -> tuple[GenerationChoice, ...]:
    """Parse completion choices from `/v1/chat/completions` response payload.

    Args:
        response_payload: JSON payload from `/v1/chat/completions`.

    Returns:
        Parsed generation choices.
    """

    raw_choices = _require_choices(response_payload=response_payload)
    prompt_ids = _parse_token_ids(raw_value=response_payload.get("prompt_token_ids"))
    parsed_choices = [
        _parse_one_chat_completion_choice(
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


def parse_detokenize_text(*, response_payload: dict[str, Any]) -> str:
    """Parse decoded text from `/detokenize` response payload.

    Args:
        response_payload: JSON payload from `/detokenize`.

    Returns:
        Decoded text.
    """
    for key in ("prompt", "text"):
        value = response_payload.get(key)
        if isinstance(value, str):
            return value
    raise AssertionError("detokenize response missing decoded text")


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


def _parse_one_chat_completion_choice(
    *,
    choice: dict[str, Any],
    fallback_prompt_token_ids: tuple[int, ...] | None,
) -> GenerationChoice:
    """Parse one chat-completion choice into canonical generation shape.

    Args:
        choice: One choice payload from `/v1/chat/completions`.
        fallback_prompt_token_ids: Request-level prompt IDs when provided.

    Returns:
        Parsed generation choice.
    """

    message = choice.get("message", {})
    message_dict = cast(dict[str, Any], message) if isinstance(message, dict) else {}
    content = message_dict.get("content", "")
    text = content if isinstance(content, str) else str(content)
    normalized_choice = dict(choice)
    normalized_choice["text"] = text
    return _parse_one_completion_choice(
        choice=normalized_choice,
        fallback_prompt_token_ids=fallback_prompt_token_ids,
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
