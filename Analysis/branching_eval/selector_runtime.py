"""Selector execution helpers for branching candidate pools."""

from __future__ import annotations

import asyncio
import json
import aiohttp
import random
from collections import Counter
from pathlib import Path
from typing import Any

from candidate_clustering import (
    DedupItem,
    build_cluster_prompt,
    coerce_assignments,
    parse_clusters_payload,
    parse_dotenv,
    strip_steer_suffix,
    validate_clusters_response,
)
from branching_eval.embedding_selection import (
    openai_diverse_topk_random_ids,
    openai_diverse_topk_random_ids_async,
    resolve_openai_api_key,
)
from branching_eval.selector_types import SelectionOutcome, SelectorMode, SelectorParams
from branching_eval.tree_types import CandidatePoolRecord, CandidateRecord
from io_utils import append_jsonl
from vllm_client import VllmClient

CLUSTER_PROMPT_MAX_TOKENS = 8192
CLUSTER_PROMPT_TOP_P = 0.95
CLUSTER_PROMPT_TOP_LOGPROBS = 0
CLUSTER_PROMPT_SEED = 0
CLUSTER_PROMPT_BASE_TEMPERATURE = 0.6
CLUSTER_PROMPT_TEMPERATURE_STEP = 0.05
CLUSTER_PROMPT_STOP = ("]}]}",)
CLUSTER_PROMPT_MAX_ATTEMPTS = 10
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_CLUSTER_MODEL = "openai/gpt-oss-20b"


def select_candidates_all_modes(
    *,
    pool: CandidatePoolRecord,
    selector_params: SelectorParams,
    selector_modes: tuple[SelectorMode, ...],
    rng: random.Random,
    openrouter_api_key: str | None,
    openai_api_key: str | None = None,
    cluster_client: VllmClient | None = None,
    cluster_model_name: str | None = None,
    cluster_log_path: Path | None = None,
) -> tuple[SelectionOutcome, ...]:
    """Compute selection outcomes for requested selector modes.

    Args:
        pool: Candidate pool.
        selector_params: Selector parameter bundle.
        selector_modes: Selector modes to execute.
        rng: RNG instance.
        openrouter_api_key: OpenRouter API key for clustering selectors.
        openai_api_key: OpenAI API key for diverse-top-k selector mode.
        cluster_client: Optional served-model client for cluster prompts.
        cluster_model_name: Optional served-model name for cluster prompts.
        cluster_log_path: Optional JSONL debug log for clustering requests.

    Returns:
        Selection outcomes in requested-mode order.
    """

    require_clusters = any(
        mode in {"cluster_across", "within_cluster"} for mode in selector_modes
    )
    require_openai_embeddings = any(
        mode == "embed_diverse_topk_random" for mode in selector_modes
    )
    if require_clusters and not openrouter_api_key:
        raise RuntimeError("cluster selectors require OPEN_ROUTER_KEY")
    if require_openai_embeddings and not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY required for embed_diverse_topk_random selector mode"
        )
    cluster_assignment = None
    if require_clusters:
        assert openrouter_api_key is not None
        cluster_assignment = _cluster_assignments(
            pool=pool,
            rng=rng,
            openrouter_api_key=openrouter_api_key,
            cluster_log_path=cluster_log_path,
        )
    diverse_topk_random = None
    if require_openai_embeddings:
        assert openai_api_key is not None
        diverse_topk_random = openai_diverse_topk_random_ids(
            pool=pool,
            branch_count=selector_params.branch_fanout,
            rng=rng,
            openai_api_key=openai_api_key,
        )
    return tuple(
        SelectionOutcome(
            selector_mode=selector_mode,
            selected_candidate_ids=_select_ids(
                pool=pool,
                selector_mode=selector_mode,
                selector_params=selector_params,
                rng=rng,
                cluster_assignment=cluster_assignment,
                openai_selected_candidate_ids=(
                    None if diverse_topk_random is None else diverse_topk_random[1]
                ),
            ),
            cluster_by_candidate_id=cluster_assignment,
            shortlist_candidate_ids=_select_shortlist_ids(
                selector_mode=selector_mode,
                openai_shortlist_candidate_ids=(
                    None if diverse_topk_random is None else diverse_topk_random[0]
                ),
            ),
        )
        for selector_mode in selector_modes
    )


async def select_candidates_all_modes_async(
    *,
    pool: CandidatePoolRecord,
    selector_params: SelectorParams,
    selector_modes: tuple[SelectorMode, ...],
    rng: random.Random,
    openrouter_api_key: str | None,
    openai_api_key: str | None = None,
    cluster_client: VllmClient | None = None,
    cluster_model_name: str | None = None,
    cluster_log_path: Path | None = None,
    http_session: aiohttp.ClientSession | None = None,
) -> tuple[SelectionOutcome, ...]:
    """Compute selection outcomes asynchronously for requested selector modes.

    Args:
        pool: Candidate pool.
        selector_params: Selector parameter bundle.
        selector_modes: Selector modes to execute.
        rng: RNG instance.
        openrouter_api_key: OpenRouter API key for clustering selectors.
        openai_api_key: OpenAI API key for diverse-top-k selector mode.
        cluster_client: Optional served-model client for cluster prompts.
        cluster_model_name: Optional served-model name for cluster prompts.
        cluster_log_path: Optional JSONL debug log for clustering requests.
        http_session: Optional shared HTTP session reused across selector calls.

    Returns:
        Selection outcomes in requested-mode order.
    """

    require_clusters = any(
        mode in {"cluster_across", "within_cluster"} for mode in selector_modes
    )
    require_openai_embeddings = any(
        mode == "embed_diverse_topk_random" for mode in selector_modes
    )
    if require_clusters and not openrouter_api_key:
        raise RuntimeError("cluster selectors require OPEN_ROUTER_KEY")
    if require_openai_embeddings and not openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY required for embed_diverse_topk_random selector mode"
        )
    if http_session is None:
        async with aiohttp.ClientSession() as owned_session:
            return await select_candidates_all_modes_async(
                pool=pool,
                selector_params=selector_params,
                selector_modes=selector_modes,
                rng=rng,
                openrouter_api_key=openrouter_api_key,
                openai_api_key=openai_api_key,
                cluster_client=cluster_client,
                cluster_model_name=cluster_model_name,
                cluster_log_path=cluster_log_path,
                http_session=owned_session,
            )
    cluster_assignment = None
    if require_clusters:
        assert openrouter_api_key is not None
        cluster_assignment = await _cluster_assignments_async(
            pool=pool,
            rng=rng,
            openrouter_api_key=openrouter_api_key,
            cluster_log_path=cluster_log_path,
            http_session=http_session,
        )
    diverse_topk_random = None
    if require_openai_embeddings:
        assert openai_api_key is not None
        diverse_topk_random = await openai_diverse_topk_random_ids_async(
            pool=pool,
            branch_count=selector_params.branch_fanout,
            rng=rng,
            openai_api_key=openai_api_key,
            session=http_session,
        )
    return tuple(
        SelectionOutcome(
            selector_mode=selector_mode,
            selected_candidate_ids=_select_ids(
                pool=pool,
                selector_mode=selector_mode,
                selector_params=selector_params,
                rng=rng,
                cluster_assignment=cluster_assignment,
                openai_selected_candidate_ids=(
                    None if diverse_topk_random is None else diverse_topk_random[1]
                ),
            ),
            cluster_by_candidate_id=cluster_assignment,
            shortlist_candidate_ids=_select_shortlist_ids(
                selector_mode=selector_mode,
                openai_shortlist_candidate_ids=(
                    None if diverse_topk_random is None else diverse_topk_random[0]
                ),
            ),
        )
        for selector_mode in selector_modes
    )


def _cluster_attempt_temperature(*, attempt_number: int) -> float:
    """Return the clustering temperature for a retry attempt.

    Args:
        attempt_number: One-based retry attempt counter.

    Returns:
        Temperature for the current attempt.
    """

    assert attempt_number >= 1, "attempt_number must be one-based"
    return CLUSTER_PROMPT_BASE_TEMPERATURE + (
        CLUSTER_PROMPT_TEMPERATURE_STEP * (attempt_number - 1)
    )


def resolve_openrouter_api_key(*, env_paths: tuple[Path, ...]) -> str | None:
    """Resolve OpenRouter API key from env vars or dotenv files.

    Args:
        env_paths: Dotenv paths.

    Returns:
        API key or None.
    """

    import os

    env_key = os.getenv("OPEN_ROUTER_KEY")
    if env_key:
        return env_key
    for path in env_paths:
        values = parse_dotenv(path=path)
        value = values.get("OPEN_ROUTER_KEY")
        if value:
            return value
    return None


def _select_ids(
    *,
    pool: CandidatePoolRecord,
    selector_mode: SelectorMode,
    selector_params: SelectorParams,
    rng: random.Random,
    cluster_assignment: dict[int, str] | None,
    openai_selected_candidate_ids: tuple[int, ...] | None,
) -> tuple[int, ...]:
    if selector_mode == "random":
        return _random_ids(pool=pool, max_count=selector_params.branch_fanout, rng=rng)
    if selector_mode == "embed_diverse_topk_random":
        assert openai_selected_candidate_ids is not None, "selected ids required"
        return openai_selected_candidate_ids
    assert cluster_assignment is not None, "cluster assignments required"
    if selector_mode == "cluster_across":
        return _cluster_across_ids(
            pool=pool,
            cluster_assignment=cluster_assignment,
            max_count=selector_params.branch_fanout,
            max_clusters=selector_params.max_clusters,
            rng=rng,
        )
    return _within_cluster_ids(
        pool=pool,
        cluster_assignment=cluster_assignment,
        max_count=selector_params.branch_fanout,
        rng=rng,
    )


def _select_shortlist_ids(
    *,
    selector_mode: SelectorMode,
    openai_shortlist_candidate_ids: tuple[int, ...] | None,
) -> tuple[int, ...] | None:
    """Return shortlist ids for selector modes with pre-selection diagnostics."""

    if selector_mode != "embed_diverse_topk_random":
        return None
    assert openai_shortlist_candidate_ids is not None, "shortlist ids required"
    return openai_shortlist_candidate_ids


def _cluster_assignments(
    *,
    pool: CandidatePoolRecord,
    rng: random.Random,
    openrouter_api_key: str,
    cluster_log_path: Path | None,
) -> dict[int, str]:
    items = _dedup_items(candidates=pool.candidates, rng=rng)
    prompt = build_cluster_prompt(
        previous_selected_count=0,
        previous_selected_chain="",
        previous_execution_tail="",
        items=items,
    )
    clusters = _live_openrouter_clusters(
        items=items,
        prompt=prompt,
        api_key=openrouter_api_key,
        cluster_log_path=cluster_log_path,
    )
    assignments = coerce_assignments(items=items, clusters=clusters)
    cluster_by_text = {
        candidate.text: assignments[
            _item_id_by_clean_text(items=items)[strip_steer_suffix(text=candidate.text)]
        ]
        for candidate in pool.candidates
    }
    return {
        candidate.candidate_id: cluster_by_text.get(candidate.text, "other")
        for candidate in pool.candidates
    }


async def _cluster_assignments_async(
    *,
    pool: CandidatePoolRecord,
    rng: random.Random,
    openrouter_api_key: str,
    cluster_log_path: Path | None,
    http_session: aiohttp.ClientSession,
) -> dict[int, str]:
    """Resolve cluster assignments asynchronously for served-model clustering.

    Args:
        pool: Candidate pool to cluster.
        openrouter_api_key: OpenRouter API key for clustering.
        cluster_log_path: Optional JSONL debug log path.

    Returns:
        Candidate-id to cluster-name mapping.
    """

    items = _dedup_items(candidates=pool.candidates, rng=rng)
    prompt = build_cluster_prompt(
        previous_selected_count=0,
        previous_selected_chain="",
        previous_execution_tail="",
        items=items,
    )
    clusters = await _live_openrouter_clusters_async(
        items=items,
        prompt=prompt,
        api_key=openrouter_api_key,
        cluster_log_path=cluster_log_path,
        http_session=http_session,
    )
    assignments = coerce_assignments(items=items, clusters=clusters)
    cluster_by_text = {
        candidate.text: assignments[
            _item_id_by_clean_text(items=items)[strip_steer_suffix(text=candidate.text)]
        ]
        for candidate in pool.candidates
    }
    return {
        candidate.candidate_id: cluster_by_text.get(candidate.text, "other")
        for candidate in pool.candidates
    }


def _live_openrouter_clusters(
    *,
    items: tuple[DedupItem, ...],
    prompt: str,
    api_key: str,
    cluster_log_path: Path | None,
) -> list[dict[str, Any]]:
    """Resolve clusters from one live OpenRouter request sequence."""

    return asyncio.run(
        _live_openrouter_clusters_sync_async(
            items=items,
            prompt=prompt,
            api_key=api_key,
            cluster_log_path=cluster_log_path,
        )
    )


async def _live_openrouter_clusters_sync_async(
    *,
    items: tuple[DedupItem, ...],
    prompt: str,
    api_key: str,
    cluster_log_path: Path | None,
) -> list[dict[str, Any]]:
    """Resolve clusters with a one-off session for sync entrypoints."""

    async with aiohttp.ClientSession() as session:
        return await _call_openrouter_clusters_with_retries_async(
            http_session=session,
            api_key=api_key,
            items=items,
            prompt=prompt,
            cluster_log_path=cluster_log_path,
        )


async def _live_openrouter_clusters_async(
    *,
    items: tuple[DedupItem, ...],
    prompt: str,
    api_key: str,
    cluster_log_path: Path | None,
    http_session: aiohttp.ClientSession,
) -> list[dict[str, Any]]:
    """Resolve clusters from one live async OpenRouter request sequence."""

    return await _call_openrouter_clusters_with_retries_async(
        api_key=api_key,
        items=items,
        prompt=prompt,
        cluster_log_path=cluster_log_path,
        http_session=http_session,
    )


async def _call_openrouter_clusters_once_async(
    *,
    http_session: aiohttp.ClientSession,
    api_key: str,
    items: tuple[DedupItem, ...],
    prompt: str,
    attempt_number: int,
    cluster_log_path: Path | None,
) -> tuple[list[dict[str, Any]], str]:
    """Call OpenRouter once and parse cluster JSON assignments.

    Args:
        api_key: OpenRouter API key.
        items: Deduplicated prompt items expected in the response.
        prompt: Cluster prompt text.
        attempt_number: One-based retry attempt counter.
        cluster_log_path: Optional JSONL debug log path.

    Returns:
        Parsed cluster rows and raw response text.
    """

    attempt_temperature = _cluster_attempt_temperature(attempt_number=attempt_number)
    payload = {
        "model": OPENROUTER_CLUSTER_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": attempt_temperature,
        "top_p": CLUSTER_PROMPT_TOP_P,
        "max_tokens": CLUSTER_PROMPT_MAX_TOKENS,
        "reasoning": {"effort": "low"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    async with http_session.post(
        OPENROUTER_CHAT_COMPLETIONS_URL,
        headers=headers,
        json=payload,
    ) as response:
        response_text = await response.text()
        if response.status >= 400:
            raise RuntimeError(f"OpenRouter error {response.status}: {response_text}")
    response_payload = json.loads(response_text)
    choices = response_payload.get("choices")
    assert isinstance(choices, list) and choices, "OpenRouter response missing choices"
    message = choices[0].get("message", {})
    assert isinstance(message, dict), "OpenRouter response missing message object"
    raw_text = str(message.get("content", "") or "")
    _append_cluster_log(
        cluster_log_path=cluster_log_path,
        payload={
            "event": "attempt_raw_response",
            "attempt_number": attempt_number,
            "model_id": OPENROUTER_CLUSTER_MODEL,
            "item_count": len(items),
            "temperature": attempt_temperature,
            "prompt": prompt,
            "raw_text": raw_text,
        },
    )
    clusters = parse_clusters_payload(raw_text=raw_text)
    validate_clusters_response(items=items, clusters=clusters)
    _append_cluster_log(
        cluster_log_path=cluster_log_path,
        payload={
            "event": "attempt_success",
            "attempt_number": attempt_number,
            "model_id": OPENROUTER_CLUSTER_MODEL,
            "item_count": len(items),
            "assignment_rate": _cluster_assignment_rate(items=items, clusters=clusters),
        },
    )
    return clusters, raw_text


async def _call_openrouter_clusters_with_retries_async(
    *,
    http_session: aiohttp.ClientSession,
    api_key: str,
    items: tuple[DedupItem, ...],
    prompt: str,
    cluster_log_path: Path | None,
) -> list[dict[str, Any]]:
    """Call OpenRouter with retries for invalid or weak cluster output.

    Args:
        api_key: OpenRouter API key.
        items: Deduplicated prompt items expected in the response.
        prompt: Cluster prompt text.
        cluster_log_path: Optional JSONL debug log path.

    Returns:
        Parsed cluster rows from one successful response.
    """

    last_error: Exception | None = None
    for attempt_index in range(CLUSTER_PROMPT_MAX_ATTEMPTS):
        try:
            clusters, _ = await _call_openrouter_clusters_once_async(
                http_session=http_session,
                api_key=api_key,
                items=items,
                prompt=prompt,
                attempt_number=attempt_index + 1,
                cluster_log_path=cluster_log_path,
            )
            return clusters
        except Exception as error:
            last_error = error
            _append_cluster_log(
                cluster_log_path=cluster_log_path,
                payload={
                    "event": "attempt_failure",
                    "attempt_number": attempt_index + 1,
                    "model_id": OPENROUTER_CLUSTER_MODEL,
                    "item_count": len(items),
                    "error": str(error),
                },
            )
            if attempt_index < (CLUSTER_PROMPT_MAX_ATTEMPTS - 1):
                await asyncio.sleep(0.35 * (2**attempt_index))
    assert last_error is not None, "OpenRouter clustering failed without exception"
    raise last_error


def _cluster_assignment_rate(
    *, items: tuple[DedupItem, ...], clusters: list[dict[str, Any]]
) -> float:
    """Compute explicit cluster assignment coverage over deduplicated items.

    Args:
        items: Deduplicated candidate items in the prompt.
        clusters: Parsed cluster rows.

    Returns:
        Fraction of item ids assigned by the cluster response.
    """

    item_ids = {item.item_id for item in items}
    if not item_ids:
        return 0.0
    assigned_item_ids = {
        int(member_id)
        for cluster in clusters
        for member_id in cluster.get("member_ids", [])
        if int(member_id) in item_ids
    }
    return len(assigned_item_ids) / len(item_ids)


def _append_cluster_log(
    *, cluster_log_path: Path | None, payload: dict[str, Any]
) -> None:
    """Append one clustering debug row when logging is enabled.

    Args:
        cluster_log_path: Optional JSONL log path.
        payload: Row payload to append.

    Returns:
        None.
    """

    if cluster_log_path is None:
        return
    append_jsonl(path=cluster_log_path, payload=payload)


def _cluster_across_ids(
    *,
    pool: CandidatePoolRecord,
    cluster_assignment: dict[int, str],
    max_count: int,
    max_clusters: int,
    rng: random.Random,
) -> tuple[int, ...]:
    grouped = _group_candidates_by_cluster(
        pool=pool, cluster_assignment=cluster_assignment
    )
    cluster_names = sorted(grouped)
    cluster_limit = min(max_count, max_clusters, len(cluster_names))
    if cluster_limit <= 0:
        return ()
    chosen_clusters = rng.sample(population=cluster_names, k=cluster_limit)
    selected = [rng.choice(grouped[name]) for name in chosen_clusters if grouped[name]]
    return tuple(selected)


def _within_cluster_ids(
    *,
    pool: CandidatePoolRecord,
    cluster_assignment: dict[int, str],
    max_count: int,
    rng: random.Random,
) -> tuple[int, ...]:
    grouped = _group_candidates_by_cluster(
        pool=pool, cluster_assignment=cluster_assignment
    )
    eligible = [
        name
        for name, members in grouped.items()
        if name != "other" and len(members) >= max_count
    ]
    if eligible:
        chosen_cluster = rng.choice(sorted(eligible))
        return tuple(rng.sample(grouped[chosen_cluster], k=max_count))
    fallback_clusters = sorted(grouped, key=lambda name: (-len(grouped[name]), name))
    members = grouped[fallback_clusters[0]]
    if len(members) <= max_count:
        return tuple(members)
    return tuple(rng.sample(members, k=max_count))


def _group_candidates_by_cluster(
    *,
    pool: CandidatePoolRecord,
    cluster_assignment: dict[int, str],
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for candidate in pool.candidates:
        cluster_name = cluster_assignment.get(candidate.candidate_id, "other")
        grouped.setdefault(cluster_name, []).append(candidate.candidate_id)
    return grouped


def _random_ids(
    *, pool: CandidatePoolRecord, max_count: int, rng: random.Random
) -> tuple[int, ...]:
    candidate_ids = [candidate.candidate_id for candidate in pool.candidates]
    if len(candidate_ids) <= max_count:
        return tuple(candidate_ids)
    return tuple(rng.sample(candidate_ids, k=max_count))


def _dedup_items(
    *, candidates: tuple[CandidateRecord, ...], rng: random.Random
) -> tuple[DedupItem, ...]:
    counts = Counter(
        strip_steer_suffix(text=candidate.text) for candidate in candidates
    )
    ordered = list(sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    rng.shuffle(ordered)
    return tuple(
        DedupItem(item_id=index + 1, text=text, count=count)
        for index, (text, count) in enumerate(ordered)
    )


def _item_id_by_clean_text(*, items: tuple[DedupItem, ...]) -> dict[str, int]:
    """Return item-id lookup keyed by cleaned cluster-prompt text.

    Args:
        items: Deduplicated cleaned prompt items.

    Returns:
        Mapping from cleaned text to item id.
    """

    return {item.text: item.item_id for item in items}


__all__ = [
    "resolve_openai_api_key",
    "resolve_openrouter_api_key",
    "select_candidates_all_modes",
    "select_candidates_all_modes_async",
    "_cluster_across_ids",
    "_within_cluster_ids",
]
