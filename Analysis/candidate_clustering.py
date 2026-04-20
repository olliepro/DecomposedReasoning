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
STEER_EXEC_PAIR_PATTERN = re.compile(
    r"<steer\b[^>]*>(.*?)</steer>\s*<exec\b[^>]*>(.*?)</exec>",
    flags=re.IGNORECASE | re.DOTALL,
)
OTHER_CLUSTER_NAME = "other"
MIN_CLUSTER_ASSIGNMENT_RATE = 0.70
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


def normalize_steer_text(*, text: str) -> str:
    """Normalize steer text for alignment comparisons.

    Args:
        text: Raw steer text.

    Returns:
        Canonically normalized steer text.
    """

    collapsed = re.sub(r"\s+", " ", strip_steer_suffix(text=text))
    return collapsed.strip().lower()


def steer_exec_pairs_from_final_text(*, final_text: str) -> list[tuple[str, str]]:
    """Extract ordered `(steer_text, execution_text)` pairs from final text.

    Args:
        final_text: Final assistant text with steer/exec tags.

    Returns:
        Ordered steer/exec pairs.
    """

    return [
        (str(steer).strip(), str(execution).strip())
        for steer, execution in STEER_EXEC_PAIR_PATTERN.findall(final_text)
    ]


def execution_text_by_step_from_final_text(
    *, final_text: str, selected_text_by_step_index: dict[int, str]
) -> dict[int, str]:
    """Align execution blocks to steps by matching steer text in order.

    Args:
        final_text: Final assistant text containing `<steer>/<exec>` blocks.
        selected_text_by_step_index: Selected steer text keyed by step index.

    Returns:
        Mapping from zero-based step index to execution text.
    """

    pairs = steer_exec_pairs_from_final_text(final_text=final_text)
    if not pairs:
        return {}
    mapped: dict[int, str] = {}
    cursor = 0
    for step_index in sorted(selected_text_by_step_index):
        target = normalize_steer_text(
            text=selected_text_by_step_index.get(step_index, "")
        )
        if not target:
            continue
        for pair_index in range(cursor, len(pairs)):
            pair_steer, pair_execution = pairs[pair_index]
            if normalize_steer_text(text=pair_steer) != target:
                continue
            mapped[step_index] = str(pair_execution).strip()
            cursor = pair_index + 1
            break
    return mapped


def tail_words(*, text: str, word_count: int) -> str:
    """Return the last `word_count` whitespace-delimited words from text.

    Args:
        text: Source text.
        word_count: Number of trailing words to keep.

    Returns:
        Trailing words joined by single spaces.
    """

    words = [word for word in re.split(r"\s+", text.strip()) if word]
    if not words:
        return ""
    return " ".join(words[-max(1, int(word_count)) :])


def previous_execution_tail(
    *, step_index: int, execution_texts: dict[int, str], tail_word_count: int
) -> tuple[int, str]:
    """Build previous-step execution context using a trailing word window.

    Args:
        step_index: Current step index.
        execution_texts: Execution text keyed by step index.
        tail_word_count: Number of trailing words to include.

    Returns:
        Tuple of context count and context string.
    """

    previous_step_index = step_index - 1
    if previous_step_index < 0:
        return 0, ""
    previous_execution = str(execution_texts.get(previous_step_index, "")).strip()
    if not previous_execution:
        return 0, ""
    return 1, tail_words(text=previous_execution, word_count=tail_word_count)


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
    *,
    previous_selected_count: int,
    previous_selected_chain: str,
    previous_execution_tail: str,
    items: tuple[DedupItem, ...],
) -> str:
    """Build strict-JSON clustering prompt for one step.

    Args:
        previous_selected_count: Number of prior selected steps included.
        previous_selected_chain: Previous selected-step chain.
        previous_execution_tail: Last few words from previous step execution.
        items: Deduplicated candidate strings for the step.

    Returns:
        Prompt text.

    Example:
        >>> prompt = build_cluster_prompt(previous_selected_count=0, previous_selected_chain="", previous_execution_tail="", items=(DedupItem(item_id=1, text="Try factoring", count=2),))
        >>> "Current step index" in prompt
        False
    """

    lines = [
        "When asked what to do next in a few words, people gave the following responses. Group them into functionally distinct clusters, especially looking for rare or unique clusters, where within each group the target of the action is identical.",
        "",
        "Return strict JSON only with this exact top-level shape:",
        "",
        "{",
        '  "groups":[',
        '    {"name":"Group Name","key":"short_key"},',
        "    ...",
        '    {"name":"other","key":"other"}',
        "  ],",
        '  "assignments":{',
        '    "1":"short_key",',
        '    "2":"another_key",',
        "    ...",
        "  }",
        "}",
        "",
        "Return one complete JSON object only. Do not include any text before or after it.",
        "",
        "Rules for the JSON:",
        "",
        "* The top-level object must contain exactly two keys: groups and assignments",
        "* Groups must be a JSON array of group objects",
        "* Each group object must contain exactly two keys: name and key",
        "* name must be a clear human-readable label for the group",
        "* assignments must be a JSON object mapping every id to exactly one group key",
        "* Every assignment value must match a key present in the groups object",
        "* Include every id exactly once in assignments",
        '* Truly ungroupable items can be assigned to the group with key "other" and name "other"',
        "* Group keys should be short, preferably 1-2 small words, use snake_case",
        "* Groups should not contain duplicates by key or name",
        "",
        "Group guidance:",
        "",
        "* Group by functional intent, not surface wording",
        "* Treat near-synonyms as the same when they imply the same next step",
        "* Do not create groups just because of single, common verbs. Examine the complex meaning.",
        "* Keep groups highly distinct from one another (e.g. 2+2=4 shouldnt be grouped with 2+3=5)",
        "* Avoid overfitting to repeated words",
        "* If two items sound similar but imply different actions, keep them separate",
        "* You need enough groups for high coverage.",
        "* Do not assign to imaginary groups.",
        "",
        "Target number of groups:",
        "",
        "* Return 1 group if there is no variety and no minority groups",
        "* Return 2-4 in low variety settings to capture/separate out small interesting minorities",
        "* Return 5-9 groups for moderate variety with clear themes",
        "* Return 10-20 groups only if there is substantial functional variety",
        "",
        'On average, there are 6-14 groups. There should never be more than 5 items assigned to "other".',
        "",
        "Procedure:",
        "",
        "1. Think about & decide on the groups, end thinking.",
        "2. Output the groups array",
        "3. Then assign every id to exactly one group key",
        "",
        "Valid example:",
        "",
        "{",
        '  "groups":[',
        '    {"name":"Simplify via Factorization","key":"factor"},',
        '    {"name":"A=5 condition","key":"a5_cond"},',
        '    {"name":"other","key":"other"}',
        "  ],",
        '  "assignments":{',
        '    "1":"factor",',
        '    "2":"a5_cond",',
        '    "3":"other",',
        '    "4":"factor",',
        '    "5":"a5_cond"',
        "  }",
        "}",
        "",
        "Or:",
        "",
        "{",
        '  "groups":[',
        '    {"name":"Identify or Locate Vertex B","key":"vertex_b"},',
        '    {"name":"Focus on Inradius","key":"inradius"},',
        '    {"name":"Consider 1234 Case","key":"1234"},',
        '    {"name":"other","key":"other"}',
        "  ],",
        '  "assignments":{',
        '    "1":"1234",',
        '    "2":"vertex_b",',
        '    "3":"inradius",',
        '    "4":"1234",',
        '    "5":"other",',
        "    ...",
        '    "93":"1234",',
        '    "94":"inradius",',
        '    "95":"1234"',
        "  }",
        "}",
        "",
        "Another valid example:",
        "",
        "{",
        '  "groups":[',
        '    {"name":"Define Important Terms","key":"terms"},',
        '    {"name":"Restate Problem","key":"reflect"},',
        '    {"name":"Use O1","key":"O1"},',
        '    {"name":"Use O3","key":"O3"}',
        "  ],",
        '  "assignments":{',
        '    "1":"O1",',
        '    "2":"reflect",',
        '    "3":"terms",',
        '    "4":"reflect",',
        "    ...",
        '    "95":"terms"',
        "  }",
        "}",
        "",
        "Notice how in all of these examples, the most salient features of the groups were the targets of the actions, e.g. the groups were highly specific to the outcome of the action rather than the banal words which are vague. This is because the goal is to identify rare or unique elements in the blob of text.",
    ]
    context_lines: list[str] = []
    if previous_selected_count > 0 and previous_selected_chain:
        parts = [
            part.strip()
            for part in previous_selected_chain.split(" >> ")
            if part.strip()
        ]
        if parts:
            context_lines.append(f"Previous {previous_selected_count} selected steps:")
            context_lines.append(f"- {parts[0]}")
            context_lines.extend(f"  >> {part}" for part in parts[1:])
    if previous_execution_tail:
        if context_lines:
            context_lines.append("")
        context_lines.append("The last few words of the previous step's execution are:")
        context_lines.append(f'"{previous_execution_tail}"')
    if context_lines:
        lines.extend(
            [
                "",
                "## Context on the current process",
                "",
                *context_lines,
            ]
        )
    lines.extend(["", "## Options to group:"])
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
    """Parse cluster JSON response and validate its top-level structure.

    Args:
        raw_text: Raw model response text.

    Returns:
        Parsed cluster payload rows.
    """

    payload = json.loads(extract_json_text(raw_text=raw_text))
    assert isinstance(payload, dict), "cluster payload must be a JSON object"
    raw_groups = payload.get("groups")
    raw_assignments = payload.get("assignments")
    assert isinstance(raw_groups, list), "cluster payload missing groups list"
    assert isinstance(raw_assignments, dict), "cluster payload missing assignments map"
    group_name_by_key: dict[str, str] = {}
    for group in raw_groups:
        assert isinstance(group, dict), "group entries must be JSON objects"
        name = group.get("name")
        key = group.get("key")
        assert isinstance(name, str), "group name must be a string"
        assert isinstance(key, str), "group key must be a string"
        group_name_by_key[key] = name
    member_ids_by_key: dict[str, list[int]] = {key: [] for key in group_name_by_key}
    for raw_item_id, raw_group_key in raw_assignments.items():
        item_id = parse_optional_int(value=raw_item_id)
        assert item_id is not None, "assignment ids must be parseable integers"
        assert isinstance(raw_group_key, str), "assignment group key must be a string"
        member_ids_by_key.setdefault(raw_group_key, []).append(item_id)
    parsed_clusters: list[dict[str, Any]] = []
    for group_key, group_name in group_name_by_key.items():
        parsed_clusters.append(
            {
                "name": group_name,
                "key": group_key,
                "member_ids": member_ids_by_key.get(group_key, []),
            }
        )
    return parsed_clusters


def validate_clusters_response(
    *, items: tuple[DedupItem, ...], clusters: list[dict[str, Any]]
) -> None:
    """Validate parsed clusters and require strong explicit assignment coverage.

    Args:
        items: Deduplicated prompt items expected in the response.
        clusters: Parsed cluster rows.

    Returns:
        None.
    """

    valid_ids = {item.item_id for item in items}
    if not valid_ids:
        return
    assigned_ids: set[int] = set()
    for cluster in clusters:
        members = cluster.get("member_ids")
        assert isinstance(members, list), "cluster member_ids must be a list"
        for value in members:
            item_id = parse_optional_int(value=value)
            if item_id is None or item_id not in valid_ids:
                continue
            assigned_ids.add(item_id)
    assignment_rate = len(assigned_ids) / len(valid_ids)
    assert assignment_rate > MIN_CLUSTER_ASSIGNMENT_RATE, (
        "cluster assignment rate must be > "
        f"{MIN_CLUSTER_ASSIGNMENT_RATE:.0%}, got {assignment_rate:.0%}"
    )


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
        cluster_key = str(cluster.get("key", "")).strip()
        cluster_name = (
            normalize_cluster_name(name=cluster_key)
            if cluster_key
            else normalize_cluster_name(name=str(cluster.get("name", "")))
        )
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
    *,
    api_key: str,
    prompt: str,
    model_id: str,
    temperature: float,
    items: tuple[DedupItem, ...],
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
        types.ThinkingConfig(thinking_level=cast(Any, "MINIMAL"))
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
    clusters = parse_clusters_payload(raw_text=str(getattr(response, "text", "") or ""))
    validate_clusters_response(items=items, clusters=clusters)
    return clusters


def call_gemini_clusters(
    *,
    api_key: str,
    prompt: str,
    model_id: str,
    temperature: float,
    items: tuple[DedupItem, ...],
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
                    items=items,
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
    items: tuple[DedupItem, ...],
    prompt: str,
    model_id: str,
    temperature: float,
    api_key: str,
) -> list[dict[str, Any]]:
    """Resolve clusters from cache or live Gemini call.

    Args:
        cache: Optional clustering cache.
        items: Deduplicated prompt items expected in the response.
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
            try:
                validate_clusters_response(items=items, clusters=cached)
                return cached
            except Exception:
                pass
    clusters = call_gemini_clusters(
        api_key=api_key,
        prompt=prompt,
        model_id=model_id,
        temperature=temperature,
        items=items,
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
    execution_texts: dict[int, str],
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
        execution_texts: Step-indexed execution text for prompt context.
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
        previous_selected_count, previous_selected_chain = previous_steps_chain(
            step_index=step_index,
            selected_texts=selected_texts,
            window=config.previous_steps_window,
        )
        _, previous_execution_tail_text = previous_execution_tail(
            step_index=step_index,
            execution_texts=execution_texts,
            tail_word_count=20,
        )
        prompt = build_cluster_prompt(
            previous_selected_count=previous_selected_count,
            previous_selected_chain=previous_selected_chain,
            previous_execution_tail=previous_execution_tail_text,
            items=items,
        )
        try:
            clusters = cached_or_live_clusters(
                cache=cache,
                items=items,
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
    execution_texts: dict[int, str],
    config: ClusteringConfig,
    api_key: str,
    cache: ClusteringCache | None,
) -> list[StepClusterResult]:
    """Cluster all steps asynchronously using bounded concurrency.

    Args:
        rows_by_step: Candidate rows grouped by step.
        selected_texts: Selected steer text map.
        execution_texts: Step-indexed execution text.
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
                execution_texts=execution_texts,
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
    final_text: str = "",
) -> ClusteringArtifacts:
    """Cluster candidates step-by-step using Gemini prompting.

    Args:
        candidates: Candidate artifact rows.
        config: Clustering configuration.
        steps: Optional step rows to provide previous-step context.
        final_text: Final assistant text with `<steer>/<exec>` blocks.

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
    execution_texts = execution_text_by_step_from_final_text(
        final_text=final_text,
        selected_text_by_step_index=selected_texts,
    )
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
                execution_texts=execution_texts,
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
                execution_texts=execution_texts,
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
