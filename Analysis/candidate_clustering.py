"""Prompt-based steer candidate clustering for report generation.

This module intentionally keeps one clustering strategy: Gemini structured-output
prompting over deduplicated steer strings.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import threading
import time
from copy import deepcopy
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

STEER_TRAILING_SUFFIXES = (
    "</steer>",
    "</steer",
    "</stee",
    "</ste",
    "</st",
    "</s",
    "</",
)
NON_SNAKE_PATTERN = re.compile(r"[^a-z0-9]+")
CODE_BLOCK_JSON_PATTERN = re.compile(
    r"```(?:json)?\s*(\{.*\})\s*```", flags=re.IGNORECASE | re.DOTALL
)
OTHER_CLUSTER_NAME = "other"
MODEL_ALIAS_FALLBACKS = {
    "gemini-3-flash": ("gemini-3-flash-preview",),
}


@dataclass(frozen=True)
class ClusteringConfig:
    """Configuration for prompt-based steer-string clustering.

    Args:
        enabled: Enables Gemini prompt clustering when true.
        gemini_model: Gemini model id used for clustering.
        temperature: Sampling temperature for the clustering prompt.
        seed: Deterministic tie-break seed used in local fallback behavior.
        previous_steps_window: Number of prior selected steps to include.
        max_concurrent_requests: Maximum concurrent Gemini clustering requests.
        cache_path: Optional persistent cache path for prompt clusters.
        env_paths: Dotenv paths searched for API keys.

    Example:
        >>> ClusteringConfig(enabled=True, gemini_model="gemini-3-flash")
        ClusteringConfig(enabled=True, gemini_model='gemini-3-flash', temperature=0.2, seed=0, previous_steps_window=5, max_concurrent_requests=50, cache_path=None, env_paths=())
    """

    enabled: bool = True
    gemini_model: str = "gemini-3-flash-preview"
    temperature: float = 0.2
    seed: int = 0
    previous_steps_window: int = 5
    max_concurrent_requests: int = 50
    cache_path: Path | None = None
    env_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class ClusterSummary:
    """Cluster summary row for one step.

    Args:
        cluster_id: Stable integer id within a step.
        name: Human-readable cluster name.
        count: Total candidate count assigned to the cluster.
    """

    cluster_id: int
    name: str
    count: int


@dataclass(frozen=True)
class CandidateClusterAssignment:
    """Per-candidate cluster assignment used by the report payload.

    Args:
        step_index: Step index.
        candidate_index: Candidate index within step.
        cluster_id: Cluster id within step.
        cluster_name: Cluster display name.
        clean_text: Candidate steer text with close-tag removed.
    """

    step_index: int
    candidate_index: int
    cluster_id: int
    cluster_name: str
    clean_text: str


@dataclass(frozen=True)
class ClusteringArtifacts:
    """Cluster assignments and summaries consumed by the report renderer.

    Args:
        mode: Clustering mode label.
        warnings: Non-fatal clustering warnings.
        summaries_by_step: Step-indexed cluster summaries.
        assignments_by_candidate: Candidate assignment map.

    Example:
        >>> artifacts = ClusteringArtifacts(mode="none", warnings=(), summaries_by_step={}, assignments_by_candidate={})
        >>> artifacts.candidate_assignment(step_index=0, candidate_index=0) is None
        True
    """

    mode: str
    warnings: tuple[str, ...]
    summaries_by_step: dict[int, tuple[ClusterSummary, ...]]
    assignments_by_candidate: dict[tuple[int, int], CandidateClusterAssignment]

    def candidate_assignment(
        self, *, step_index: int, candidate_index: int
    ) -> CandidateClusterAssignment | None:
        """Look up one candidate assignment.

        Args:
            step_index: Step index.
            candidate_index: Candidate index.

        Returns:
            Matching assignment or `None`.
        """

        return self.assignments_by_candidate.get((step_index, candidate_index))

    def summary_dicts_for_step(self, *, step_index: int) -> list[dict[str, Any]]:
        """Serialize cluster summaries for one step.

        Args:
            step_index: Step index.

        Returns:
            List of summary dictionaries.
        """

        return [
            {
                "cluster_id": summary.cluster_id,
                "name": summary.name,
                "count": summary.count,
            }
            for summary in self.summaries_by_step.get(step_index, ())
        ]


@dataclass(frozen=True)
class CandidateRow:
    """Normalized candidate row.

    Args:
        step_index: Step index.
        candidate_index: Candidate index.
        clean_text: Close-tag-stripped steer text.
    """

    step_index: int
    candidate_index: int
    clean_text: str


@dataclass(frozen=True)
class DedupItem:
    """Deduplicated steer string item used in prompts.

    Args:
        item_id: Stable 1-based id.
        text: Unique steer string.
        count: Number of row-level occurrences.
    """

    item_id: int
    text: str
    count: int

    def prompt_line(self) -> str:
        """Render one item line for the clustering prompt.

        Returns:
            Prompt line containing id, count, and text.
        """

        return f"{self.item_id}: count={self.count} | text={self.text}"


@dataclass(frozen=True)
class StepClusterResult:
    """Step-level cluster materialization for merge into report artifacts.

    Args:
        step_index: Step index for these assignments.
        summaries: Cluster summaries for the step.
        assignments: Candidate cluster assignments for the step.
        warning: Optional warning message for the step.
    """

    step_index: int
    summaries: tuple[ClusterSummary, ...]
    assignments: dict[tuple[int, int], CandidateClusterAssignment]
    warning: str | None


@dataclass
class ClusteringCache:
    """Thread-safe on-disk cache for prompt-clustering responses.

    Args:
        path: Cache file path.
        entries: Cache entries keyed by prompt hash.
        lock: Internal thread lock.
    """

    path: Path
    entries: dict[str, list[dict[str, Any]]]
    lock: threading.Lock

    @classmethod
    def from_path(cls, *, path: Path) -> ClusteringCache:
        """Load cache file from disk or initialize empty cache.

        Args:
            path: Cache file path.

        Returns:
            Loaded cache instance.
        """

        if not path.exists():
            return cls(path=path, entries={}, lock=threading.Lock())
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(raw, dict), "invalid cluster cache file content"
        entries = raw.get("entries", {})
        assert isinstance(entries, dict), "invalid cluster cache entries"
        validated = {
            str(key): value
            for key, value in entries.items()
            if isinstance(key, str) and isinstance(value, list)
        }
        return cls(path=path, entries=validated, lock=threading.Lock())

    def get(self, *, key: str) -> list[dict[str, Any]] | None:
        """Get cached clusters by key.

        Args:
            key: Cache key.

        Returns:
            Cached clusters or `None`.
        """

        with self.lock:
            value = self.entries.get(key)
            if value is None:
                return None
            return deepcopy(value)

    def set(self, *, key: str, clusters: list[dict[str, Any]]) -> None:
        """Set cached clusters for a key.

        Args:
            key: Cache key.
            clusters: Parsed cluster rows.

        Returns:
            None.
        """

        with self.lock:
            self.entries[key] = deepcopy(clusters)

    def flush(self) -> None:
        """Write cache to disk with stable formatting.

        Args:
            None.

        Returns:
            None.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock:
            payload = {
                "version": 1,
                "entries": dict(sorted(self.entries.items())),
            }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )


def strip_steer_suffix(*, text: str) -> str:
    """Remove trailing steer close-tag from a candidate string.

    Args:
        text: Raw candidate text.

    Returns:
        Cleaned candidate text.

    Example:
        >>> strip_steer_suffix(text="Try substitution</steer>")
        'Try substitution'
        >>> strip_steer_suffix(text='Try "A.L.G.O.R.I.T.H.M" pun</ste ')
        'Try "A.L.G.O.R.I.T.H.M" pun'
    """
    suffix_span = trailing_steer_suffix_span(text=text)
    if suffix_span is None:
        return text.strip()
    return text[: suffix_span[0]].strip()


def trailing_steer_suffix_span(*, text: str) -> tuple[int, int] | None:
    """Find a trailing full or truncated steer close-tag suffix span.

    Args:
        text: Raw candidate or stitched token text.

    Returns:
        `(start, end)` char span for trailing close-tag-like suffix, else `None`.

    Example:
        >>> trailing_steer_suffix_span(text="x</steer>")
        (1, 9)
        >>> trailing_steer_suffix_span(text="x</ste ")
        (1, 6)
    """

    right = len(text.rstrip())
    if right <= 0:
        return None
    candidate = text[:right]
    lowered = candidate.lower()
    for suffix in STEER_TRAILING_SUFFIXES:
        if lowered.endswith(suffix):
            return (right - len(suffix), right)
    return None


def parse_dotenv(*, path: Path) -> dict[str, str]:
    """Parse dotenv entries from a file.

    Args:
        path: Dotenv file path.

    Returns:
        Parsed key/value mapping.
    """

    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_api_key(*, env_paths: tuple[Path, ...]) -> str | None:
    """Resolve Gemini API key from env vars or dotenv files.

    Args:
        env_paths: Dotenv path lookup order.

    Returns:
        Resolved API key or `None`.
    """

    env_key = (
        os.getenv("VERTEX_KEY")
        or os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )
    if env_key:
        return env_key
    for path in env_paths:
        values = parse_dotenv(path=path)
        for key_name in ("VERTEX_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            value = values.get(key_name)
            if value:
                return value
    return None


def parse_optional_int(*, value: object) -> int | None:
    """Parse optional integer values.

    Args:
        value: Input value.

    Returns:
        Parsed integer or `None`.
    """

    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


def candidate_rows(*, candidates: list[dict[str, Any]]) -> list[CandidateRow]:
    """Normalize candidate rows from artifact payloads.

    Args:
        candidates: Raw candidate rows.

    Returns:
        Parsed candidate rows.
    """

    rows: list[CandidateRow] = []
    for row in candidates:
        step_index = parse_optional_int(value=row.get("step_index"))
        candidate_index = parse_optional_int(value=row.get("candidate_index"))
        if step_index is None or candidate_index is None:
            continue
        rows.append(
            CandidateRow(
                step_index=step_index,
                candidate_index=candidate_index,
                clean_text=strip_steer_suffix(text=str(row.get("text", ""))),
            )
        )
    return rows


def selected_text_by_step(*, steps: list[dict[str, Any]]) -> dict[int, str]:
    """Build selected steer text mapping from step artifact rows.

    Args:
        steps: Step artifact rows.

    Returns:
        Step-indexed selected steer text.
    """

    selected: dict[int, str] = {}
    for row in steps:
        step_index = parse_optional_int(value=row.get("step_index"))
        if step_index is None:
            continue
        text = strip_steer_suffix(text=str(row.get("selected_text", "")))
        if text:
            selected[step_index] = text
    return selected


def previous_steps_chain(
    *, step_index: int, selected_texts: dict[int, str], window: int
) -> tuple[int, str]:
    """Build previous selected-step context for the Gemini prompt.

    Args:
        step_index: Current step index.
        selected_texts: Step-indexed selected steer text.
        window: Max number of previous steps.

    Returns:
        Tuple of context count and `>>`-joined context string.
    """

    start = max(0, step_index - max(1, window))
    history = [
        selected_texts[index]
        for index in range(start, step_index)
        if index in selected_texts
    ]
    return len(history), " >> ".join(history)


def dedup_items_for_step(*, rows: list[CandidateRow]) -> tuple[DedupItem, ...]:
    """Deduplicate step rows by steer text.

    Args:
        rows: Candidate rows within one step.

    Returns:
        Deduplicated items sorted by count descending then alphabetical text.
    """

    counts = Counter(row.clean_text for row in rows)
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(
        DedupItem(item_id=index + 1, text=text, count=count)
        for index, (text, count) in enumerate(ordered)
    )


def build_cluster_prompt(
    *, previous_count: int, previous_chain: str, items: tuple[DedupItem, ...]
) -> str:
    """Build strict-JSON clustering prompt for one step.

    Args:
        previous_count: Number of prior selected steps included.
        previous_chain: Previous selected-step chain.
        items: Deduplicated candidate strings for the step.

    Returns:
        Prompt text.

    Example:
        >>> prompt = build_cluster_prompt(previous_count=0, previous_chain="", items=(DedupItem(item_id=1, text="Try factoring", count=2),))
        >>> "Current step index" in prompt
        False
    """

    lines = [
        "When asked what to do next in a few words people said these things.",
        "Cluster them into high-level reasoning groups.",
        "Return strict JSON only with this exact shape:",
        '{"clusters":[{"name":"high_level_group_name","member_ids":[1,2]}]}',
        "Rules:",
        "- use lowercase snake_case names with max 3 words",
        "- include all ids exactly once across all member_ids",
        "- do not invent ids",
        "- if uncertain, place items into other",
    ]
    if previous_count > 0:
        lines.append(f"Previous {previous_count} selected steps: {previous_chain}")
    lines.append("Items:")
    lines.extend(item.prompt_line() for item in items)
    return "\n".join(lines)


def normalize_cluster_name(*, name: str) -> str:
    """Normalize cluster name into lowercase snake_case.

    Args:
        name: Raw cluster name.

    Returns:
        Normalized cluster name.
    """

    normalized = NON_SNAKE_PATTERN.sub("_", name.strip().lower()).strip("_")
    return normalized or OTHER_CLUSTER_NAME


def extract_json_text(*, raw_text: str) -> str:
    """Extract JSON object text from Gemini response.

    Args:
        raw_text: Raw model response text.

    Returns:
        JSON object string.
    """

    text = raw_text.strip()
    code_match = CODE_BLOCK_JSON_PATTERN.match(text)
    if code_match is not None:
        return code_match.group(1)
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    assert start >= 0 and end > start, "Gemini response missing JSON object"
    return text[start : end + 1]


def parse_clusters_payload(*, raw_text: str) -> list[dict[str, Any]]:
    """Parse Gemini JSON response to cluster rows.

    Args:
        raw_text: Raw model response text.

    Returns:
        Parsed cluster payload rows.
    """

    payload = json.loads(extract_json_text(raw_text=raw_text))
    clusters = payload.get("clusters")
    assert isinstance(clusters, list), "Gemini payload missing clusters list"
    return [cluster for cluster in clusters if isinstance(cluster, dict)]


def prompt_cache_key(*, model_id: str, temperature: float, prompt: str) -> str:
    """Compute cache key for one clustering prompt request.

    Args:
        model_id: Gemini model id.
        temperature: Sampling temperature.
        prompt: Prompt text.

    Returns:
        Stable SHA256 key.
    """

    payload = json.dumps(
        {
            "model_id": model_id,
            "temperature": float(temperature),
            "prompt": prompt,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_clustering_cache(*, path: Path | None) -> ClusteringCache | None:
    """Load optional clustering cache from disk.

    Args:
        path: Optional cache path.

    Returns:
        Cache instance or `None` when disabled.
    """

    if path is None:
        return None
    return ClusteringCache.from_path(path=path)


def model_attempt_order(*, model_id: str) -> tuple[str, ...]:
    """Build ordered model-id attempts including known aliases.

    Args:
        model_id: Requested model id.

    Returns:
        Ordered unique model-id attempts.
    """

    ordered = [model_id]
    ordered.extend(MODEL_ALIAS_FALLBACKS.get(model_id, ()))
    unique: list[str] = []
    for item in ordered:
        if item and item not in unique:
            unique.append(item)
    return tuple(unique)


def is_model_not_found_error(*, error: Exception) -> bool:
    """Check whether an exception indicates a model-not-found error.

    Args:
        error: Raised exception.

    Returns:
        True when the error message indicates a 404 model lookup failure.
    """

    message = str(error).lower()
    return "404" in message and "not_found" in message and "models/" in message


def coerce_assignments(
    *, items: tuple[DedupItem, ...], clusters: list[dict[str, Any]]
) -> dict[int, str]:
    """Coerce parsed clusters into complete id-to-name assignments.

    Args:
        items: Deduplicated items.
        clusters: Parsed cluster rows from Gemini.

    Returns:
        Complete item-id assignment mapping.
    """

    valid_ids = {item.item_id for item in items}
    assignment: dict[int, str] = {}
    for cluster in clusters:
        cluster_name = normalize_cluster_name(name=str(cluster.get("name", "")))
        members = cluster.get("member_ids", [])
        if not isinstance(members, list):
            continue
        for value in members:
            item_id = parse_optional_int(value=value)
            if item_id is None or item_id not in valid_ids:
                continue
            if item_id in assignment:
                continue
            assignment[item_id] = cluster_name
    for item in items:
        assignment.setdefault(item.item_id, OTHER_CLUSTER_NAME)
    assert len(assignment) == len(items), "incomplete cluster assignment"
    return assignment


def call_gemini_once(
    *, api_key: str, prompt: str, model_id: str, temperature: float
) -> list[dict[str, Any]]:
    """Call Gemini once and parse cluster JSON.

    Args:
        api_key: Gemini API key.
        prompt: Prompt text.
        model_id: Gemini model id.
        temperature: Sampling temperature.

    Returns:
        Parsed cluster rows from one response.
    """

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    thinking_config = (
        types.ThinkingConfig(thinking_level=cast(Any, "LOW"))
        if "gemini-3" in model_id
        else None
    )
    response = client.models.generate_content(
        model=model_id,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            response_mime_type="application/json",
            thinking_config=thinking_config,
        ),
    )
    return parse_clusters_payload(raw_text=str(getattr(response, "text", "") or ""))


def call_gemini_clusters(
    *, api_key: str, prompt: str, model_id: str, temperature: float
) -> list[dict[str, Any]]:
    """Call Gemini with retries and model-alias fallback.

    Args:
        api_key: Gemini API key.
        prompt: Prompt text.
        model_id: Requested Gemini model id.
        temperature: Sampling temperature.

    Returns:
        Parsed cluster rows.
    """

    last_error: Exception | None = None
    for candidate_model_id in model_attempt_order(model_id=model_id):
        for attempt_index in range(3):
            try:
                return call_gemini_once(
                    api_key=api_key,
                    prompt=prompt,
                    model_id=candidate_model_id,
                    temperature=temperature,
                )
            except Exception as error:
                last_error = error
                is_last_attempt = attempt_index >= 2
                if is_model_not_found_error(error=error):
                    break
                if is_last_attempt:
                    continue
                time.sleep(0.35 * (2**attempt_index))
    assert last_error is not None, "Gemini call failed without an exception"
    raise last_error


def cached_or_live_clusters(
    *,
    cache: ClusteringCache | None,
    prompt: str,
    model_id: str,
    temperature: float,
    api_key: str,
) -> list[dict[str, Any]]:
    """Resolve clusters from cache or live Gemini call.

    Args:
        cache: Optional clustering cache.
        prompt: Cluster prompt text.
        model_id: Gemini model id.
        temperature: Sampling temperature.
        api_key: Gemini API key.

    Returns:
        Parsed cluster rows.
    """

    key = prompt_cache_key(model_id=model_id, temperature=temperature, prompt=prompt)
    if cache is not None:
        cached = cache.get(key=key)
        if cached is not None:
            return cached
    clusters = call_gemini_clusters(
        api_key=api_key,
        prompt=prompt,
        model_id=model_id,
        temperature=temperature,
    )
    if cache is not None:
        cache.set(key=key, clusters=clusters)
    return clusters


def fallback_assignments(*, items: tuple[DedupItem, ...]) -> dict[int, str]:
    """Build deterministic local fallback assignments by exact text.

    Args:
        items: Deduplicated items.

    Returns:
        Item-id to cluster-name assignments.
    """

    assignments: dict[int, str] = {}
    for item in items:
        cluster_name = normalize_cluster_name(name=item.text)
        assignments[item.item_id] = cluster_name or OTHER_CLUSTER_NAME
    return assignments


def cluster_ids_by_name(*, counts: Counter[str]) -> dict[str, int]:
    """Build stable cluster ids sorted by count then name.

    Args:
        counts: Row-level counts by cluster name.

    Returns:
        Cluster name to id mapping.
    """

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {name: index for index, (name, _) in enumerate(ordered)}


def assignments_for_step(
    *,
    rows: list[CandidateRow],
    item_assignments: dict[int, str],
    items: tuple[DedupItem, ...],
) -> tuple[
    tuple[ClusterSummary, ...], dict[tuple[int, int], CandidateClusterAssignment]
]:
    """Expand dedup assignments back to per-candidate assignments.

    Args:
        rows: Candidate rows in one step.
        item_assignments: Cluster assignment by dedup item id.
        items: Deduplicated items.

    Returns:
        Tuple of step summaries and candidate assignment map.
    """

    item_id_by_text = {item.text: item.item_id for item in items}
    cluster_counts: Counter[str] = Counter()
    candidate_map: dict[tuple[int, int], CandidateClusterAssignment] = {}
    for row in rows:
        item_id = item_id_by_text[row.clean_text]
        cluster_name = item_assignments[item_id]
        cluster_counts[cluster_name] += 1
    name_to_id = cluster_ids_by_name(counts=cluster_counts)
    for row in rows:
        item_id = item_id_by_text[row.clean_text]
        cluster_name = item_assignments[item_id]
        candidate_map[(row.step_index, row.candidate_index)] = (
            CandidateClusterAssignment(
                step_index=row.step_index,
                candidate_index=row.candidate_index,
                cluster_id=name_to_id[cluster_name],
                cluster_name=cluster_name,
                clean_text=row.clean_text,
            )
        )
    summaries = tuple(
        ClusterSummary(cluster_id=name_to_id[name], name=name, count=count)
        for name, count in sorted(
            cluster_counts.items(), key=lambda item: (-item[1], item[0])
        )
    )
    return summaries, candidate_map


def cluster_step_result(
    *,
    step_index: int,
    step_rows: list[CandidateRow],
    selected_texts: dict[int, str],
    config: ClusteringConfig,
    api_key: str | None,
    mode: str,
    cache: ClusteringCache | None = None,
) -> StepClusterResult:
    """Cluster one step and return assignments/summaries.

    Args:
        step_index: Step index.
        step_rows: Candidate rows for the step.
        selected_texts: Selected steer text map for prompt context.
        config: Clustering configuration.
        api_key: Gemini API key when available.
        mode: Clustering mode label.
        cache: Optional cache used for prompt results.

    Returns:
        Step-level clustering result.
    """

    items = dedup_items_for_step(rows=step_rows)
    warning: str | None = None
    if mode == "prompt_gemini" and api_key is not None:
        previous_count, previous_chain = previous_steps_chain(
            step_index=step_index,
            selected_texts=selected_texts,
            window=config.previous_steps_window,
        )
        prompt = build_cluster_prompt(
            previous_count=previous_count,
            previous_chain=previous_chain,
            items=items,
        )
        try:
            clusters = cached_or_live_clusters(
                cache=cache,
                prompt=prompt,
                model_id=config.gemini_model,
                temperature=config.temperature,
                api_key=api_key,
            )
            item_assignments = coerce_assignments(items=items, clusters=clusters)
        except Exception as error:
            warning = (
                f"step {step_index}: Gemini clustering failed "
                f"({type(error).__name__}); using fallback"
            )
            item_assignments = fallback_assignments(items=items)
    else:
        item_assignments = fallback_assignments(items=items)
    summaries, assignments = assignments_for_step(
        rows=step_rows,
        item_assignments=item_assignments,
        items=items,
    )
    return StepClusterResult(
        step_index=step_index,
        summaries=summaries,
        assignments=assignments,
        warning=warning,
    )


async def cluster_steps_async(
    *,
    rows_by_step: dict[int, list[CandidateRow]],
    selected_texts: dict[int, str],
    config: ClusteringConfig,
    api_key: str,
    cache: ClusteringCache | None,
) -> list[StepClusterResult]:
    """Cluster all steps asynchronously using bounded concurrency.

    Args:
        rows_by_step: Candidate rows grouped by step.
        selected_texts: Selected steer text map.
        config: Clustering configuration.
        api_key: Gemini API key.
        cache: Optional cluster-response cache.

    Returns:
        Step results from asynchronous clustering tasks.
    """

    semaphore = asyncio.Semaphore(max(1, int(config.max_concurrent_requests)))

    async def run_step(step_index: int) -> StepClusterResult:
        async with semaphore:
            return await asyncio.to_thread(
                cluster_step_result,
                step_index=step_index,
                step_rows=rows_by_step[step_index],
                selected_texts=selected_texts,
                config=config,
                api_key=api_key,
                mode="prompt_gemini",
                cache=cache,
            )

    tasks = [run_step(step_index) for step_index in sorted(rows_by_step)]
    return list(await asyncio.gather(*tasks))


def cluster_candidates_by_step(
    *,
    candidates: list[dict[str, Any]],
    config: ClusteringConfig,
    steps: list[dict[str, Any]] | None = None,
) -> ClusteringArtifacts:
    """Cluster candidates step-by-step using Gemini prompting.

    Args:
        candidates: Candidate artifact rows.
        config: Clustering configuration.
        steps: Optional step rows to provide previous-step context.

    Returns:
        Clustering artifacts used by the report builder.

    Example:
        >>> artifacts = cluster_candidates_by_step(candidates=[], config=ClusteringConfig(enabled=False))
        >>> artifacts.mode
        'disabled'
    """

    rows = candidate_rows(candidates=candidates)
    rows_by_step: dict[int, list[CandidateRow]] = defaultdict(list)
    for row in rows:
        rows_by_step[row.step_index].append(row)
    selected_texts = selected_text_by_step(steps=steps or [])
    warnings: list[str] = []
    api_key = resolve_api_key(env_paths=config.env_paths)
    cache = load_clustering_cache(path=config.cache_path)
    mode = "prompt_gemini"
    if not config.enabled:
        mode = "disabled"
    elif not api_key:
        mode = "fallback_no_api_key"
        warnings.append(
            "Gemini API key not found; using exact-text fallback clustering"
        )

    if mode == "prompt_gemini" and api_key is not None:
        step_results = asyncio.run(
            cluster_steps_async(
                rows_by_step=rows_by_step,
                selected_texts=selected_texts,
                config=config,
                api_key=api_key,
                cache=cache,
            )
        )
    else:
        step_results = [
            cluster_step_result(
                step_index=step_index,
                step_rows=rows_by_step[step_index],
                selected_texts=selected_texts,
                config=config,
                api_key=api_key,
                mode=mode,
                cache=cache,
            )
            for step_index in sorted(rows_by_step)
        ]
    summaries_by_step: dict[int, tuple[ClusterSummary, ...]] = {}
    assignments_by_candidate: dict[tuple[int, int], CandidateClusterAssignment] = {}
    for result in sorted(step_results, key=lambda item: item.step_index):
        summaries_by_step[result.step_index] = result.summaries
        assignments_by_candidate.update(result.assignments)
        if result.warning:
            warnings.append(result.warning)

    assert len(assignments_by_candidate) == len(
        rows
    ), "not all candidates were assigned"
    if cache is not None:
        cache.flush()
    return ClusteringArtifacts(
        mode=mode,
        warnings=tuple(warnings),
        summaries_by_step=summaries_by_step,
        assignments_by_candidate=assignments_by_candidate,
    )
