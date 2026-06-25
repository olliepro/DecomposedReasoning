"""OpenAI-compatible embedding helpers for branching diversity sampling.

This module centralizes candidate-text cleaning, embedding lookup, and greedy
diversity sampling for the `embed_diverse_topk_random` selector.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import aiohttp
import numpy as np

from branching_eval.tree_types import CandidatePoolRecord, CandidateRecord
from candidate_clustering import parse_dotenv, strip_steer_suffix

OPENAI_EMBEDDING_API_URL = "https://api.openai.com/v1/embeddings"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
OPENROUTER_EMBEDDING_API_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-3-small"
OPENAI_EMBEDDING_BATCH_SIZE = 1024
# Request the reduced vector size directly from OpenAI.
OPENAI_EMBEDDING_DIMENSIONS = 128
OPENAI_EMBEDDING_MAX_ATTEMPTS = 5
OPENAI_EMBEDDING_RETRY_BASE_DELAY_SECONDS = 5.0
OPENAI_EMBEDDING_RETRY_MAX_DELAY_SECONDS = 40.0
OPENAI_EMBEDDING_RETRY_STATUS_CODES = frozenset({408, 409, 425, 429})
OPENAI_EMBEDDING_ERROR_BODY_MAX_CHARS = 2000
OPENAI_EMBEDDING_COALESCE_DELAY_SECONDS = 0.002


@dataclass(frozen=True)
class _EmbeddingProvider:
    """One embedding API endpoint shape."""

    label: str
    api_url: str
    api_key: str
    model: str


@dataclass(frozen=True)
class _EmbeddingBatchKey:
    """Lookup key for coalescing compatible embedding requests."""

    loop_id: int
    session_id: int
    primary_provider: _EmbeddingProvider
    fallback_provider: _EmbeddingProvider | None
    output_dimensions: int | None
    batch_size: int


@dataclass(frozen=True)
class _PendingEmbeddingRequest:
    """One waiting embedding request inside a coalesced batch."""

    texts: tuple[str, ...]
    future: asyncio.Future[dict[str, tuple[float, ...]]]


@dataclass
class _PendingEmbeddingBatch:
    """Accumulated embedding requests sharing one provider request shape."""

    session: aiohttp.ClientSession
    primary_provider: _EmbeddingProvider
    fallback_provider: _EmbeddingProvider | None
    output_dimensions: int | None
    batch_size: int
    requests: list[_PendingEmbeddingRequest]


_PENDING_EMBEDDING_BATCHES: dict[_EmbeddingBatchKey, _PendingEmbeddingBatch] = {}


@dataclass(frozen=True)
class CleanedCandidateGroup:
    """One cleaned candidate text and all candidate ids that map to it.

    Args:
        text: Cleaned candidate text used for embedding.
        candidate_ids: Candidate ids in original pool order.

    Returns:
        Grouped candidate metadata for text-level selection.
    """

    text: str
    candidate_ids: tuple[int, ...]

    def first_candidate_id(self) -> int:
        """Return the first candidate id in original pool order.

        Args:
            None.

        Returns:
            First candidate id.
        """

        assert self.candidate_ids, "candidate_ids must be non-empty"
        return self.candidate_ids[0]


def resolve_openai_api_key(*, env_paths: tuple[Path, ...]) -> str | None:
    """Resolve OpenAI API key from env vars, dotenv files, or ``~/.zshrc``.

    Args:
        env_paths: Dotenv lookup paths.

    Returns:
        API key string or `None`.
    """

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    for path in env_paths:
        values = parse_dotenv(path=path)
        value = values.get("OPENAI_API_KEY", "").strip()
        if value:
            return value
    return _zsh_openai_api_key()


def openai_diverse_topk_random_ids(
    *,
    pool: CandidatePoolRecord,
    branch_count: int,
    rng: Any,
    openai_api_key: str | None,
    openrouter_api_key: str | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return shortlist and selected ids for the OpenAI diverse-top-k selector.

    Args:
        pool: Candidate pool.
        branch_count: Number of candidate ids to select.
        rng: Random object used for shortlist sampling.
        openai_api_key: OpenAI API key.
        openrouter_api_key: Optional OpenRouter fallback embedding key.

    Returns:
        Tuple of `(shortlist_candidate_ids, selected_candidate_ids)`.

    Example:
        >>> openai_diverse_topk_random_ids(  # doctest: +SKIP
        ...     pool=pool,
        ...     branch_count=2,
        ...     rng=random.Random(0),
        ...     openai_api_key="sk-...",
        ... )
        ((0, 2, 4), (0, 4))
    """

    return asyncio.run(
        openai_diverse_topk_random_ids_async(
            pool=pool,
            branch_count=branch_count,
            rng=rng,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
        )
    )


async def openai_diverse_topk_random_ids_async(
    *,
    pool: CandidatePoolRecord,
    branch_count: int,
    rng: Any,
    openai_api_key: str | None,
    openrouter_api_key: str | None = None,
    session: aiohttp.ClientSession | None = None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return shortlist and selected ids from the async OpenAI selector.

    Args:
        pool: Candidate pool.
        branch_count: Number of candidate ids to select.
        rng: Random object used for shortlist sampling.
        openai_api_key: OpenAI API key.
        openrouter_api_key: Optional OpenRouter fallback embedding key.
        session: Optional shared HTTP session reused across selector calls.

    Returns:
        Tuple of `(shortlist_candidate_ids, selected_candidate_ids)`.
    """

    assert branch_count >= 1, "branch_count must be >= 1"
    groups = cleaned_candidate_groups(candidates=pool.candidates)
    if not groups:
        selected_ids = fallback_candidate_ids(
            candidates=pool.candidates,
            branch_count=branch_count,
            rng=rng,
        )
        return selected_ids, selected_ids
    if len(groups) <= branch_count:
        selected_ids = tuple(group.first_candidate_id() for group in groups)
        return selected_ids, selected_ids
    texts = [group.text for group in groups]
    embeddings_by_text = await openai_embeddings_by_text_async(
        texts=texts,
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        model=OPENAI_EMBEDDING_MODEL,
        batch_size=OPENAI_EMBEDDING_BATCH_SIZE,
        output_dimensions=OPENAI_EMBEDDING_DIMENSIONS,
        session=session,
    )
    matrix = normalized_embedding_matrix(
        texts=texts,
        embeddings_by_text=embeddings_by_text,
    )
    shortlist_size = diverse_shortlist_size(
        n_unique_candidates=len(groups),
        branch_count=branch_count,
    )
    shortlist_indices = greedy_diverse_indices(matrix=matrix, max_count=shortlist_size)
    shortlist_candidate_ids = tuple(
        groups[index].first_candidate_id() for index in shortlist_indices
    )
    chosen_indices = shortlist_indices
    if len(shortlist_indices) > branch_count:
        sampled = set(rng.sample(list(shortlist_indices), k=branch_count))
        chosen_indices = tuple(index for index in shortlist_indices if index in sampled)
    selected_candidate_ids = tuple(
        groups[index].first_candidate_id() for index in chosen_indices
    )
    return shortlist_candidate_ids, selected_candidate_ids


def cleaned_candidate_groups(
    *, candidates: tuple[CandidateRecord, ...]
) -> tuple[CleanedCandidateGroup, ...]:
    """Group candidate ids by cleaned text in original pool order.

    Args:
        candidates: Raw candidate rows from one branch point.

    Returns:
        Ordered cleaned-text groups.
    """

    grouped_ids: dict[str, list[int]] = {}
    ordered_texts: list[str] = []
    for candidate in candidates:
        text = cleaned_candidate_text(text=candidate.text)
        if not text:
            continue
        if text not in grouped_ids:
            grouped_ids[text] = []
            ordered_texts.append(text)
        grouped_ids[text].append(candidate.candidate_id)
    return tuple(
        CleanedCandidateGroup(text=text, candidate_ids=tuple(grouped_ids[text]))
        for text in ordered_texts
    )


def cleaned_candidate_text(*, text: str) -> str:
    """Return the cleaned text used by embedding-backed selectors.

    Args:
        text: Raw candidate text.

    Returns:
        Cleaned candidate string.
    """

    return strip_steer_suffix(text=text).strip()


def fallback_candidate_ids(
    *, candidates: tuple[CandidateRecord, ...], branch_count: int, rng: Any
) -> tuple[int, ...]:
    """Return candidate ids when no non-empty embedding text is available.

    Args:
        candidates: Raw candidate rows from one candidate pool.
        branch_count: Maximum number of candidate ids to select.
        rng: Random object used for deterministic sampling.

    Returns:
        Candidate ids in original pool order.
    """

    assert branch_count >= 1, "branch_count must be >= 1"
    candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
    if len(candidate_ids) <= branch_count:
        return candidate_ids
    sampled_ids = set(rng.sample(list(candidate_ids), k=branch_count))
    return tuple(
        candidate_id for candidate_id in candidate_ids if candidate_id in sampled_ids
    )


def diverse_shortlist_size(*, n_unique_candidates: int, branch_count: int) -> int:
    """Return shortlist size for the diverse-top-k selector.

    Args:
        n_unique_candidates: Count of unique cleaned candidates.
        branch_count: Number of final candidates requested.

    Returns:
        Diverse shortlist size bounded by uniqueness.
    """

    assert n_unique_candidates >= 1, "n_unique_candidates must be >= 1"
    assert branch_count >= 1, "branch_count must be >= 1"
    candidate_bound = math.ceil(math.log2(n_unique_candidates) + 1.0)
    return min(n_unique_candidates, max(branch_count, candidate_bound))


def normalized_embedding_matrix(
    *,
    texts: Sequence[str],
    embeddings_by_text: Mapping[str, Sequence[float]],
) -> np.ndarray:
    """Build a normalized embedding matrix from text-keyed vectors.

    Args:
        texts: Texts in row order.
        embeddings_by_text: Embeddings keyed by text.

    Returns:
        Normalized 2D NumPy array.
    """

    vectors = [embeddings_by_text[text] for text in texts]
    full_dimension = len(vectors[0])
    assert all(
        len(vector) == full_dimension for vector in vectors
    ), "all embedding vectors must have the same length"
    matrix = np.asarray(
        vectors,
        dtype=np.float32,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    assert np.all(norms > 0.0), "embedding vectors must have non-zero norm"
    return matrix / norms


def greedy_diverse_indices(*, matrix: np.ndarray, max_count: int) -> tuple[int, ...]:
    """Return greedy farthest-point row indices from a normalized matrix.

    Args:
        matrix: Normalized embedding matrix.
        max_count: Maximum number of indices to return.

    Returns:
        Tuple of selected row indices.
    """

    total = int(matrix.shape[0])
    if total == 0 or max_count <= 0:
        return ()
    selected = [0]
    closest_similarity = matrix @ matrix[0]
    target_count = min(max_count, total)
    while len(selected) < target_count:
        remaining = [index for index in range(total) if index not in selected]
        next_index = min(
            remaining,
            key=lambda index: (float(closest_similarity[index]), index),
        )
        selected.append(next_index)
        closest_similarity = np.maximum(closest_similarity, matrix @ matrix[next_index])
    return tuple(selected)


def openai_embeddings_by_text(
    *,
    texts: Sequence[str],
    openai_api_key: str | None,
    openrouter_api_key: str | None = None,
    model: str = OPENAI_EMBEDDING_MODEL,
    batch_size: int = OPENAI_EMBEDDING_BATCH_SIZE,
    output_dimensions: int | None = None,
) -> dict[str, tuple[float, ...]]:
    """Resolve embeddings keyed by text.

    Args:
        texts: Ordered input texts.
        openai_api_key: OpenAI API key.
        openrouter_api_key: Optional OpenRouter fallback embedding key.
        model: Embedding model id.
        batch_size: Maximum texts per API request.
        output_dimensions: Optional output dimension count requested from OpenAI.

    Returns:
        Text-keyed embeddings in input order.
    """

    return asyncio.run(
        openai_embeddings_by_text_async(
            texts=texts,
            openai_api_key=openai_api_key,
            openrouter_api_key=openrouter_api_key,
            model=model,
            batch_size=batch_size,
            output_dimensions=output_dimensions,
        )
    )


async def openai_embeddings_by_text_async(
    *,
    texts: Sequence[str],
    openai_api_key: str | None,
    openrouter_api_key: str | None = None,
    model: str = OPENAI_EMBEDDING_MODEL,
    batch_size: int = OPENAI_EMBEDDING_BATCH_SIZE,
    output_dimensions: int | None = None,
    session: aiohttp.ClientSession | None = None,
) -> dict[str, tuple[float, ...]]:
    """Resolve embeddings for text strings in API batch order.

    Args:
        texts: Ordered input texts.
        openai_api_key: OpenAI API key.
        openrouter_api_key: Optional OpenRouter fallback embedding key.
        model: Embedding model id.
        batch_size: Maximum texts per API request.
        output_dimensions: Optional output dimension count requested from OpenAI.
        session: Optional shared HTTP session reused across batches or pools.

    Returns:
        Text-keyed embeddings aligned to the input order.
    """

    if not texts:
        return {}
    if session is None:
        async with aiohttp.ClientSession() as owned_session:
            return await openai_embeddings_by_text_async(
                texts=texts,
                openai_api_key=openai_api_key,
                openrouter_api_key=openrouter_api_key,
                model=model,
                batch_size=batch_size,
                output_dimensions=output_dimensions,
                session=owned_session,
            )
    return await _coalesced_openai_embeddings_by_text_async(
        texts=tuple(texts),
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        model=model,
        batch_size=batch_size,
        output_dimensions=output_dimensions,
        session=session,
    )


async def _fetch_openai_embeddings_by_text_async(
    *,
    texts: Sequence[str],
    primary_provider: _EmbeddingProvider,
    fallback_provider: _EmbeddingProvider | None,
    batch_size: int,
    output_dimensions: int | None,
    session: aiohttp.ClientSession,
) -> dict[str, tuple[float, ...]]:
    """Fetch embeddings for one already-coalesced text list."""

    try:
        return await _fetch_embeddings_with_provider_async(
            texts=texts,
            provider=primary_provider,
            batch_size=batch_size,
            output_dimensions=output_dimensions,
            session=session,
        )
    except RuntimeError:
        if fallback_provider is None:
            raise
    return await _fetch_embeddings_with_provider_async(
        texts=texts,
        provider=fallback_provider,
        batch_size=batch_size,
        output_dimensions=output_dimensions,
        session=session,
    )


async def _fetch_embeddings_with_provider_async(
    *,
    texts: Sequence[str],
    provider: _EmbeddingProvider,
    batch_size: int,
    output_dimensions: int | None,
    session: aiohttp.ClientSession,
) -> dict[str, tuple[float, ...]]:
    """Fetch embeddings from one concrete provider."""

    batches = _batch_texts(
        texts=list(texts),
        batch_size=batch_size,
    )
    batch_vectors = await asyncio.gather(
        *[
            _openai_embedding_batch_async(
                session=session,
                texts=batch,
                model=provider.model,
                openai_api_key=provider.api_key,
                output_dimensions=output_dimensions,
                api_url=provider.api_url,
                provider_label=provider.label,
            )
            for batch in batches
        ]
    )
    flattened = [vector for batch in batch_vectors for vector in batch]
    assert len(flattened) == len(texts), "embedding response length mismatch"
    return {text: vector for text, vector in zip(texts, flattened)}


async def _coalesced_openai_embeddings_by_text_async(
    *,
    texts: tuple[str, ...],
    openai_api_key: str | None,
    openrouter_api_key: str | None,
    model: str,
    batch_size: int,
    output_dimensions: int | None,
    session: aiohttp.ClientSession,
) -> dict[str, tuple[float, ...]]:
    """Coalesce concurrent embedding calls sharing one HTTP session."""

    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, tuple[float, ...]]] = loop.create_future()
    primary_provider, fallback_provider = _embedding_providers(
        openai_api_key=openai_api_key,
        openrouter_api_key=openrouter_api_key,
        model=model,
    )
    key = _EmbeddingBatchKey(
        loop_id=id(loop),
        session_id=id(session),
        primary_provider=primary_provider,
        fallback_provider=fallback_provider,
        output_dimensions=output_dimensions,
        batch_size=batch_size,
    )
    batch = _PENDING_EMBEDDING_BATCHES.get(key)
    if batch is None:
        batch = _PendingEmbeddingBatch(
            session=session,
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            output_dimensions=output_dimensions,
            batch_size=batch_size,
            requests=[],
        )
        _PENDING_EMBEDDING_BATCHES[key] = batch
        loop.create_task(_flush_coalesced_openai_embeddings_async(key=key))
    batch.requests.append(_PendingEmbeddingRequest(texts=texts, future=future))
    return await future


async def _flush_coalesced_openai_embeddings_async(*, key: _EmbeddingBatchKey) -> None:
    """Resolve one pending coalesced embedding batch after a short debounce."""

    await asyncio.sleep(OPENAI_EMBEDDING_COALESCE_DELAY_SECONDS)
    batch = _PENDING_EMBEDDING_BATCHES.pop(key, None)
    if batch is None:
        return
    try:
        embeddings = await _resolve_coalesced_openai_embedding_batch_async(batch=batch)
    except Exception as exc:
        for request in batch.requests:
            if not request.future.done():
                request.future.set_exception(exc)
        return
    for request in batch.requests:
        if request.future.done():
            continue
        request.future.set_result({text: embeddings[text] for text in request.texts})


async def _resolve_coalesced_openai_embedding_batch_async(
    *, batch: _PendingEmbeddingBatch
) -> dict[str, tuple[float, ...]]:
    """Fetch unique embeddings needed by all requests in a coalesced batch."""

    unique_texts = tuple(
        dict.fromkeys(text for request in batch.requests for text in request.texts)
    )
    return await _fetch_openai_embeddings_by_text_async(
        texts=unique_texts,
        primary_provider=batch.primary_provider,
        fallback_provider=batch.fallback_provider,
        batch_size=batch.batch_size,
        output_dimensions=batch.output_dimensions,
        session=batch.session,
    )


def _embedding_providers(
    *, openai_api_key: str | None, openrouter_api_key: str | None, model: str
) -> tuple[_EmbeddingProvider, _EmbeddingProvider | None]:
    """Return the primary and optional fallback embedding providers."""

    if openai_api_key:
        primary = _EmbeddingProvider(
            label="OpenAI",
            api_url=OPENAI_EMBEDDING_API_URL,
            api_key=openai_api_key,
            model=model,
        )
        fallback = None
        if openrouter_api_key:
            fallback = _EmbeddingProvider(
                label="OpenRouter",
                api_url=OPENROUTER_EMBEDDING_API_URL,
                api_key=openrouter_api_key,
                model=OPENROUTER_EMBEDDING_MODEL,
            )
        return primary, fallback
    if openrouter_api_key:
        return (
            _EmbeddingProvider(
                label="OpenRouter",
                api_url=OPENROUTER_EMBEDDING_API_URL,
                api_key=openrouter_api_key,
                model=OPENROUTER_EMBEDDING_MODEL,
            ),
            None,
        )
    raise RuntimeError(
        "OPENAI_API_KEY or OPEN_ROUTER_KEY required for embedding selector"
    )


async def _openai_embedding_batch_async(
    *,
    session: aiohttp.ClientSession,
    texts: Sequence[str],
    model: str,
    openai_api_key: str,
    output_dimensions: int | None,
    api_url: str = OPENAI_EMBEDDING_API_URL,
    provider_label: str = "OpenAI",
) -> list[tuple[float, ...]]:
    """Fetch one embedding batch through an OpenAI-compatible API.

    Args:
        session: Shared HTTP session.
        texts: Batch texts in request order.
        model: Embedding model id.
        openai_api_key: OpenAI API key.
        output_dimensions: Optional reduced output dimension count.
        api_url: OpenAI-compatible embeddings endpoint.
        provider_label: Provider name used in error messages.

    Returns:
        Embedding vectors in request order.
    """

    assert OPENAI_EMBEDDING_MAX_ATTEMPTS >= 1, "max attempts must be positive"
    payload = {
        "input": list(texts),
        "model": model,
        "encoding_format": "float",
    }
    if output_dimensions is not None:
        payload["dimensions"] = output_dimensions
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    response_text = ""
    for attempt_number in range(1, OPENAI_EMBEDDING_MAX_ATTEMPTS + 1):
        try:
            async with session.post(
                api_url,
                headers=headers,
                json=payload,
            ) as response:
                response_text = await response.text()
                if response.status < 400:
                    break
                if not _should_retry_openai_embedding_status(
                    status_code=response.status,
                    attempt_number=attempt_number,
                ):
                    raise RuntimeError(
                        _openai_embedding_error_message(
                            status_code=response.status,
                            response_text=response_text,
                            attempt_number=attempt_number,
                            provider_label=provider_label,
                        )
                    )
                await asyncio.sleep(
                    _openai_embedding_retry_delay_seconds(attempt_number=attempt_number)
                )
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            if attempt_number >= OPENAI_EMBEDDING_MAX_ATTEMPTS:
                raise RuntimeError(
                    f"{provider_label} embedding transport error after "
                    f"{attempt_number} attempts: {exc}"
                ) from exc
            await asyncio.sleep(
                _openai_embedding_retry_delay_seconds(attempt_number=attempt_number)
            )
    else:
        raise AssertionError("unreachable embedding retry loop state")
    response_payload = json.loads(response_text)
    data = response_payload.get("data")
    assert isinstance(data, list), "embedding response missing data list"
    assert len(data) == len(texts), "embedding batch length mismatch"
    ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
    return [_parse_openai_embedding_row(row=row) for row in ordered]


def _should_retry_openai_embedding_status(
    *, status_code: int, attempt_number: int
) -> bool:
    """Return whether an embedding HTTP status should be retried.

    Args:
        status_code: HTTP response status from OpenAI.
        attempt_number: One-based request attempt number.

    Returns:
        `True` when another attempt should be made.
    """

    assert attempt_number >= 1, "attempt_number must be one-based"
    is_transient_server_error = 500 <= status_code <= 599
    return (
        status_code in OPENAI_EMBEDDING_RETRY_STATUS_CODES or is_transient_server_error
    ) and attempt_number < OPENAI_EMBEDDING_MAX_ATTEMPTS


def _openai_embedding_retry_delay_seconds(*, attempt_number: int) -> float:
    """Return exponential retry delay for an OpenAI embedding attempt.

    Args:
        attempt_number: One-based request attempt number that just failed.

    Returns:
        Delay in seconds before the next retry.

    Example:
        >>> _openai_embedding_retry_delay_seconds(attempt_number=1)
        5.0
    """

    assert attempt_number >= 1, "attempt_number must be one-based"
    delay = OPENAI_EMBEDDING_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt_number - 1))
    return min(delay, OPENAI_EMBEDDING_RETRY_MAX_DELAY_SECONDS)


def _openai_embedding_error_message(
    *,
    status_code: int,
    response_text: str,
    attempt_number: int,
    provider_label: str = "OpenAI",
) -> str:
    """Format a bounded OpenAI embedding HTTP error message.

    Args:
        status_code: HTTP response status.
        response_text: Raw response body.
        attempt_number: One-based attempt number.
        provider_label: Provider name used in the error.

    Returns:
        Runtime error message safe for logs.
    """

    body = response_text[:OPENAI_EMBEDDING_ERROR_BODY_MAX_CHARS]
    if len(response_text) > OPENAI_EMBEDDING_ERROR_BODY_MAX_CHARS:
        body = f"{body}...<truncated>"
    return (
        f"{provider_label} embedding error {status_code} after "
        f"{attempt_number} attempts: "
        f"{body}"
    )


def _parse_openai_embedding_row(*, row: object) -> tuple[float, ...]:
    assert isinstance(row, dict), "embedding row must be a mapping"
    embedding = row.get("embedding")
    assert isinstance(embedding, list), "embedding row missing embedding list"
    vector = tuple(float(value) for value in embedding)
    assert vector, "embedding vectors must be non-empty"
    return vector


def _zsh_openai_api_key() -> str | None:
    result = subprocess.run(
        [
            "zsh",
            "-c",
            'source ~/.zshrc >/dev/null 2>&1; printf %s "${OPENAI_API_KEY:-}"',
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value or None


def _batch_texts(*, texts: list[str], batch_size: int) -> tuple[tuple[str, ...], ...]:
    return tuple(
        tuple(texts[index : index + batch_size])
        for index in range(0, len(texts), batch_size)
    )


__all__ = [
    "CleanedCandidateGroup",
    "OPENAI_EMBEDDING_BATCH_SIZE",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_DIMENSIONS",
    "OPENROUTER_EMBEDDING_API_URL",
    "OPENROUTER_EMBEDDING_MODEL",
    "cleaned_candidate_groups",
    "cleaned_candidate_text",
    "diverse_shortlist_size",
    "greedy_diverse_indices",
    "normalized_embedding_matrix",
    "openai_embeddings_by_text",
    "openai_embeddings_by_text_async",
    "openai_diverse_topk_random_ids",
    "openai_diverse_topk_random_ids_async",
    "resolve_openai_api_key",
]
