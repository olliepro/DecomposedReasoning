"""Selector execution helpers, embedding cache, and candidate-pool keys."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, cast

from candidate_clustering import (
    ClusteringCache,
    DedupItem,
    build_cluster_prompt,
    cached_or_live_clusters,
    coerce_assignments,
    parse_dotenv,
)
from branching_eval.config_types import BranchingConfig, DecodingConfig
from branching_eval.selector_types import SelectionOutcome, SelectorMode, SelectorParams
from branching_eval.tree_types import CandidatePoolRecord, CandidateRecord


class EmbeddingCache:
    """Persistent Gemini embedding cache keyed by text hash.

    Args:
        cache_path: Cache file path.
        model_name: Gemini embedding model name.

    Returns:
        Cache helper for embedding lookups.
    """

    def __init__(self, *, cache_path: Path, model_name: str) -> None:
        self.cache_path = cache_path
        self.model_name = model_name
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries = self._load_entries()

    def embeddings_for_texts(
        self,
        *,
        texts: list[str],
        gemini_api_key: str,
    ) -> dict[str, tuple[float, ...]]:
        """Return embeddings for texts using cache and Gemini fallback.

        Args:
            texts: Input candidate texts.
            gemini_api_key: Gemini API key.

        Returns:
            Mapping from text to embedding vector.
        """

        missing = [
            text for text in texts if self._cache_key(text=text) not in self.entries
        ]
        if missing:
            self._resolve_missing(missing=missing, gemini_api_key=gemini_api_key)
        return {text: tuple(self.entries[self._cache_key(text=text)]) for text in texts}

    def _resolve_missing(self, *, missing: list[str], gemini_api_key: str) -> None:
        from google import genai

        client = genai.Client(api_key=gemini_api_key)
        for batch in _batch_items(items=missing, batch_size=32):
            response = client.models.embed_content(
                model=self.model_name,
                contents=cast(Any, batch),
            )
            vectors = _embedding_vectors_from_response(
                response=response, expected_size=len(batch)
            )
            for text, vector in zip(batch, vectors):
                self.entries[self._cache_key(text=text)] = list(vector)
        self._flush()

    def _flush(self) -> None:
        payload = {"model": self.model_name, "entries": self.entries}
        self.cache_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load_entries(self) -> dict[str, list[float]]:
        if not self.cache_path.exists():
            return {}
        payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            return {}
        entries = payload.get("entries", {})
        if not isinstance(entries, dict):
            return {}
        return {
            str(key): [float(value) for value in values]
            for key, values in entries.items()
            if isinstance(values, list)
        }

    def _cache_key(self, *, text: str) -> str:
        payload = f"embedding\n{self.model_name}\n{text}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def select_candidates_all_modes(
    *,
    pool: CandidatePoolRecord,
    selector_params: SelectorParams,
    selector_modes: tuple[SelectorMode, ...],
    rng: random.Random,
    cluster_cache: ClusteringCache,
    embedding_cache: EmbeddingCache,
    gemini_api_key: str | None,
) -> tuple[SelectionOutcome, ...]:
    """Compute selection outcomes for requested selector modes.

    Args:
        pool: Candidate pool.
        selector_params: Selector parameter bundle.
        selector_modes: Selector modes to execute.
        rng: RNG instance.
        cluster_cache: Gemini clustering cache.
        embedding_cache: Gemini embedding cache.
        gemini_api_key: Gemini API key.

    Returns:
        Selection outcomes in requested-mode order.
    """

    require_clusters = any(
        mode in {"cluster_across", "within_cluster"} for mode in selector_modes
    )
    require_embeddings = any(mode == "embed_diverse" for mode in selector_modes)
    if (require_clusters or require_embeddings) and not gemini_api_key:
        raise RuntimeError("Gemini API key required for requested selector modes")
    cluster_assignment = None
    if require_clusters:
        assert gemini_api_key is not None
        cluster_assignment = _cluster_assignments(
            pool=pool,
            gemini_api_key=gemini_api_key,
            cluster_cache=cluster_cache,
        )
    embedding_by_candidate = None
    if require_embeddings:
        assert gemini_api_key is not None
        embedding_by_candidate = _embedding_by_candidate(
            pool=pool,
            embedding_cache=embedding_cache,
            gemini_api_key=gemini_api_key,
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
                embedding_by_candidate=embedding_by_candidate,
            ),
            cluster_by_candidate_id=cluster_assignment,
            embedding_by_candidate_id=embedding_by_candidate,
        )
        for selector_mode in selector_modes
    )


def parse_selection_outcomes_from_cache(
    *, cached: dict[str, Any]
) -> tuple[SelectionOutcome, ...]:
    """Parse serialized selection cache into typed outcomes.

    Args:
        cached: Cached selection payload mapping.

    Returns:
        Parsed `SelectionOutcome` rows.
    """

    outcomes: list[SelectionOutcome] = []
    for selector_mode in sorted(cached):
        parsed_mode = _coerce_selector_mode(value=selector_mode)
        if parsed_mode is None:
            continue
        payload = cached[selector_mode]
        if not isinstance(payload, dict):
            continue
        selected_ids_raw = payload.get("selected_candidate_ids", [])
        selected_candidate_ids = tuple(int(item) for item in selected_ids_raw)
        outcomes.append(
            SelectionOutcome(
                selector_mode=parsed_mode,
                selected_candidate_ids=selected_candidate_ids,
                cluster_by_candidate_id=_parse_optional_cluster_map(
                    raw=payload.get("cluster_by_candidate_id")
                ),
                embedding_by_candidate_id=_parse_optional_embedding_map(
                    raw=payload.get("embedding_by_candidate_id")
                ),
            )
        )
    return tuple(outcomes)


def resolve_gemini_api_key(*, env_paths: tuple[Path, ...]) -> str | None:
    """Resolve Gemini API key from env vars or dotenv files.

    Args:
        env_paths: Dotenv paths.

    Returns:
        API key or None.
    """

    import os

    env_key = (
        os.getenv("VERTEX_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if env_key:
        return env_key
    for path in env_paths:
        values = parse_dotenv(path=path)
        for key in ("VERTEX_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            value = values.get(key)
            if value:
                return value
    return None


def build_candidate_pool_cache_key(
    *,
    doc_id: int,
    node_id: str,
    trigger_type: str,
    seed: int,
    model_name: str,
    decoding: DecodingConfig,
    branching: BranchingConfig,
) -> str:
    """Build stable candidate-pool cache key.

    Args:
        doc_id: Document id.
        node_id: Node id.
        trigger_type: Trigger type.
        seed: Seed value.
        model_name: Model name.
        decoding: Decoding settings.
        branching: Branching settings.

    Returns:
        SHA256 cache key.
    """

    payload = {
        "doc_id": doc_id,
        "node_id": node_id,
        "trigger_type": trigger_type,
        "seed": seed,
        "model_name": model_name,
        "temperature": decoding.temperature,
        "top_p": decoding.top_p,
        "max_gen_toks": decoding.max_gen_toks,
        "num_candidates": branching.num_candidates,
        "candidate_span_tokens": branching.candidate_span_tokens,
        "max_steer_tokens": branching.max_steer_tokens,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _select_ids(
    *,
    pool: CandidatePoolRecord,
    selector_mode: SelectorMode,
    selector_params: SelectorParams,
    rng: random.Random,
    cluster_assignment: dict[int, str] | None,
    embedding_by_candidate: dict[int, tuple[float, ...]] | None,
) -> tuple[int, ...]:
    if selector_mode == "random":
        return _random_ids(pool=pool, max_count=selector_params.branch_fanout, rng=rng)
    if selector_mode == "embed_diverse":
        assert embedding_by_candidate is not None, "embedding data required"
        return _diverse_embedding_ids(
            embedding_by_candidate=embedding_by_candidate,
            max_count=selector_params.branch_fanout,
            rng=rng,
        )
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


def _cluster_assignments(
    *,
    pool: CandidatePoolRecord,
    gemini_api_key: str,
    cluster_cache: ClusteringCache,
) -> dict[int, str]:
    items = _dedup_items(candidates=pool.candidates)
    prompt = build_cluster_prompt(
        previous_selected_count=0,
        previous_selected_chain="",
        previous_execution_tail="",
        items=items,
    )
    clusters = cached_or_live_clusters(
        cache=cluster_cache,
        prompt=prompt,
        model_id="gemini-3-flash-preview",
        temperature=0.2,
        api_key=gemini_api_key,
    )
    cluster_cache.flush()
    assignments = coerce_assignments(items=items, clusters=clusters)
    cluster_by_text = {item.text: assignments[item.item_id] for item in items}
    return {
        candidate.candidate_id: cluster_by_text.get(candidate.text, "other")
        for candidate in pool.candidates
    }


def _embedding_by_candidate(
    *,
    pool: CandidatePoolRecord,
    embedding_cache: EmbeddingCache,
    gemini_api_key: str,
) -> dict[int, tuple[float, ...]]:
    texts = [candidate.text for candidate in pool.candidates]
    embeddings_by_text = embedding_cache.embeddings_for_texts(
        texts=texts,
        gemini_api_key=gemini_api_key,
    )
    return {
        candidate.candidate_id: embeddings_by_text[candidate.text]
        for candidate in pool.candidates
    }


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


def _diverse_embedding_ids(
    *,
    embedding_by_candidate: dict[int, tuple[float, ...]],
    max_count: int,
    rng: random.Random,
) -> tuple[int, ...]:
    candidate_ids = sorted(embedding_by_candidate)
    if len(candidate_ids) <= max_count:
        return tuple(candidate_ids)
    selected: list[int] = [rng.choice(candidate_ids)]
    remaining = [
        candidate_id for candidate_id in candidate_ids if candidate_id not in selected
    ]
    while remaining and len(selected) < max_count:
        best_candidate = max(
            remaining,
            key=lambda candidate_id: _min_distance_to_selected(
                candidate_id=candidate_id,
                selected_ids=selected,
                embedding_by_candidate=embedding_by_candidate,
            ),
        )
        selected.append(best_candidate)
        remaining.remove(best_candidate)
    return tuple(selected)


def _min_distance_to_selected(
    *,
    candidate_id: int,
    selected_ids: list[int],
    embedding_by_candidate: dict[int, tuple[float, ...]],
) -> float:
    vector = embedding_by_candidate[candidate_id]
    distances = [
        _cosine_distance(vector_a=vector, vector_b=embedding_by_candidate[selected_id])
        for selected_id in selected_ids
    ]
    return min(distances) if distances else 0.0


def _cosine_distance(
    *, vector_a: tuple[float, ...], vector_b: tuple[float, ...]
) -> float:
    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    cosine = dot_product / (norm_a * norm_b)
    return 1.0 - max(-1.0, min(1.0, cosine))


def _random_ids(
    *, pool: CandidatePoolRecord, max_count: int, rng: random.Random
) -> tuple[int, ...]:
    candidate_ids = [candidate.candidate_id for candidate in pool.candidates]
    if len(candidate_ids) <= max_count:
        return tuple(candidate_ids)
    return tuple(rng.sample(candidate_ids, k=max_count))


def _batch_items(*, items: list[str], batch_size: int) -> list[list[str]]:
    return [
        items[index : index + batch_size] for index in range(0, len(items), batch_size)
    ]


def _coerce_selector_mode(*, value: object) -> SelectorMode | None:
    if value in {"cluster_across", "embed_diverse", "within_cluster", "random"}:
        return cast(SelectorMode, value)
    return None


def _embedding_vectors_from_response(
    *, response: Any, expected_size: int
) -> list[tuple[float, ...]]:
    raw_embeddings = getattr(response, "embeddings", None)
    assert isinstance(raw_embeddings, list), "embedding response missing list"
    assert len(raw_embeddings) == expected_size, "embedding response length mismatch"
    vectors = [
        tuple(float(value) for value in getattr(item, "values", []) or [])
        for item in raw_embeddings
    ]
    assert all(vector for vector in vectors), "embedding vectors must be non-empty"
    return vectors


def _dedup_items(*, candidates: tuple[CandidateRecord, ...]) -> tuple[DedupItem, ...]:
    counts = Counter(candidate.text for candidate in candidates)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        DedupItem(item_id=index + 1, text=text, count=count)
        for index, (text, count) in enumerate(ordered)
    )


def _parse_optional_cluster_map(*, raw: object) -> dict[int, str] | None:
    if not isinstance(raw, dict):
        return None
    return {int(key): str(value) for key, value in raw.items()}


def _parse_optional_embedding_map(
    *, raw: object
) -> dict[int, tuple[float, ...]] | None:
    if not isinstance(raw, dict):
        return None
    parsed: dict[int, tuple[float, ...]] = {}
    for key, value in raw.items():
        if not isinstance(value, (list, tuple)):
            continue
        parsed[int(key)] = tuple(float(item) for item in value)
    return parsed


__all__ = [
    "EmbeddingCache",
    "build_candidate_pool_cache_key",
    "parse_selection_outcomes_from_cache",
    "resolve_gemini_api_key",
    "select_candidates_all_modes",
    "_cluster_across_ids",
    "_diverse_embedding_ids",
    "_within_cluster_ids",
]
