"""Parse clustering debug logs, embed option strings, and diversity-sample them.

This module reconstructs the option strings and cluster assignments from
successful clustering attempts recorded in ``clustering_debug.jsonl`` files.
It then selects one successful prompt, embeds that prompt's cleaned option
strings with OpenAI's embeddings API, and picks a small diverse subset for
quick inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence
from branching_eval.embedding_selection import (
    greedy_diverse_indices,
    normalized_embedding_matrix,
    openai_embeddings_by_text,
    resolve_openai_api_key,
)

EMBEDDING_MODEL = "text-embedding-3-small"
OPENAI_EMBEDDING_FULL_DIMENSIONS_BY_MODEL = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}
OPTIONS_MARKER = "## Options to group:"
ITEM_LINE_PATTERN = re.compile(
    r"^(?P<item_id>\d+): count=(?P<count>\d+) \| text=(?P<text>.*)$"
)
STEER_TRAILING_SUFFIXES = (
    "</steer>",
    "</steer",
    "</stee",
    "</ste",
    "</st",
    "</s",
    "</",
)


@dataclass(frozen=True)
class PromptClusterItem:
    """One cleaned item listed in the prompt options.

    Args:
        item_id: One-based item id shown in the clustering prompt.
        source_count: Multiplicity attached to the item in the prompt.
        text: Cleaned option string.
    """

    item_id: int
    source_count: int
    text: str


@dataclass(frozen=True)
class ClusterGroup:
    """One named group returned by the clustering model.

    Args:
        key: Short assignment key.
        name: Human-readable cluster name.
    """

    key: str
    name: str

    def display_label(self) -> str:
        """Return a stable display label for terminal and notebook views."""

        return f"{self.name} [{self.key}]"


@dataclass(frozen=True)
class SuccessfulClusterAttempt:
    """One successful prompt-clustering attempt reconstructed from debug logs.

    Args:
        success_index: Stable one-based success index within the file.
        attempt_number: Retry number that succeeded.
        model_id: Model id recorded in the log row.
        items: Parsed prompt options.
        groups: Parsed output groups.
        assignments: Mapping from item id to cluster key.

    Example:
        >>> attempt = SuccessfulClusterAttempt(
        ...     success_index=1,
        ...     attempt_number=1,
        ...     model_id="demo",
        ...     items=(PromptClusterItem(item_id=1, source_count=2, text="Try factoring"),),
        ...     groups=(ClusterGroup(key="factor", name="Factor"),),
        ...     assignments={1: "factor"},
        ... )
        >>> attempt.cluster_name_by_key()["factor"]
        'Factor'
    """

    success_index: int
    attempt_number: int
    model_id: str
    items: tuple[PromptClusterItem, ...]
    groups: tuple[ClusterGroup, ...]
    assignments: dict[int, str]

    def cluster_name_by_key(self) -> dict[str, str]:
        """Return a key-to-name mapping for this attempt."""

        return {group.key: group.name for group in self.groups}


@dataclass(frozen=True)
class AssignedClusterRow:
    """Flattened row for one text item and its assigned cluster.

    Args:
        attempt_index: One-based successful attempt index.
        item_id: Item id inside the attempt.
        source_count: Multiplicity attached to the item in the source prompt.
        text: Cleaned option string.
        cluster_key: Cluster key assigned to this item.
        cluster_name: Human-readable cluster name.
    """

    attempt_index: int
    item_id: int
    source_count: int
    text: str
    cluster_key: str
    cluster_name: str


@dataclass(frozen=True)
class ClusterCount:
    """Observed cluster frequency for one text string.

    Args:
        cluster_key: Cluster key used in assignments.
        cluster_name: Cluster name used in assignments.
        count: Number of successful-attempt rows with this assignment.
    """

    cluster_key: str
    cluster_name: str
    count: int

    def display_label(self) -> str:
        """Return a stable terminal-friendly label."""

        return f"{self.cluster_name} [{self.cluster_key}] x{self.count}"


@dataclass(frozen=True)
class TextClusterSummary:
    """Aggregate view of one cleaned option string across successful attempts.

    Args:
        text: Cleaned option string.
        occurrences: Number of successful-attempt rows carrying the string.
        attempts: Number of successful attempts where the string appears.
        cluster_counts: Observed cluster labels and counts for this string.
    """

    text: str
    occurrences: int
    attempts: int
    cluster_counts: tuple[ClusterCount, ...]

    def cluster_summary(self) -> str:
        """Render cluster counts as a single compact string."""

        return "; ".join(count.display_label() for count in self.cluster_counts)

    def to_display_row(self) -> dict[str, object]:
        """Return a notebook-friendly dictionary row.

        Returns:
            Dictionary with text, counts, and cluster summary.
        """

        return {
            "text": self.text,
            "occurrences": self.occurrences,
            "attempts": self.attempts,
            "clusters": self.cluster_summary(),
        }


@dataclass(frozen=True)
class ClusterSamplingRun:
    """Full result of parsing, embedding, and diversity sampling.

    Args:
        input_path: Source ``clustering_debug.jsonl`` path.
        embedding_model: Embedding model used for vectors.
        embedding_dimensions: Dimensions returned by the embedding API.
        successful_attempt_count: Number of successful clustering attempts.
        selected_success_index: One-based successful attempt index used for sampling.
        selected_attempt_number: Retry attempt number of the selected success.
        selected_item_count: Number of prompt items in the selected success.
        sampled_summaries: Sampled text summaries selected by diversity sampling.
        text_summaries: All per-item summaries for the selected success.
    """

    input_path: Path
    embedding_model: str
    embedding_dimensions: int | None
    successful_attempt_count: int
    selected_success_index: int
    selected_attempt_number: int
    selected_item_count: int
    sampled_summaries: tuple[TextClusterSummary, ...]
    text_summaries: tuple[TextClusterSummary, ...]

    def display_rows(self) -> list[dict[str, object]]:
        """Return sampled summaries as display dictionaries."""

        return [summary.to_display_row() for summary in self.sampled_summaries]


@dataclass(frozen=True)
class _PendingRawResponse:
    """Raw response row waiting to be paired with success or failure."""

    attempt_number: int
    item_count: int
    model_id: str
    prompt: str
    raw_text: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for clustering-debug sampling."""

    parser = argparse.ArgumentParser(
        description=(
            "Parse successful clustering attempts, select one prompt, embed "
            "its cleaned option strings, and print a small diverse sample."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to clustering_debug.jsonl.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=5,
        help="How many sampled strings to print (clamped to 3-5 by default).",
    )
    parser.add_argument(
        "--success-index",
        type=int,
        default=1,
        help="One-based successful prompt index to sample within.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=EMBEDDING_MODEL,
        help="OpenAI embedding model id.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="Embedding batch size.",
    )
    return parser.parse_args()


def choose_sample_count(*, unique_text_count: int, requested_sample_count: int) -> int:
    """Clamp the requested sample count into a practical range.

    Args:
        unique_text_count: Number of candidate texts available.
        requested_sample_count: Requested sample size.

    Returns:
        Clamped sample size.
    """

    if unique_text_count <= 0:
        return 0
    lower_bound = 3 if unique_text_count >= 3 else unique_text_count
    upper_bound = 10 if unique_text_count >= 5 else unique_text_count
    return max(lower_bound, min(requested_sample_count, upper_bound))


def clean_cluster_text(*, text: str) -> str:
    """Normalize one prompt option string.

    Args:
        text: Raw option string.

    Returns:
        Whitespace-normalized text with any trailing steer suffix removed.

    Example:
        >>> clean_cluster_text(text="  Try substitution </steer>  ")
        'Try substitution'
    """

    stripped = " ".join(text.split())
    lowered = stripped.lower()
    for suffix in STEER_TRAILING_SUFFIXES:
        if lowered.endswith(suffix):
            stripped = stripped[: -len(suffix)].strip()
            break
    return stripped


def extract_json_object_text(*, raw_text: str) -> str:
    """Extract the outer JSON object from a raw model response.

    Args:
        raw_text: Raw response text from the clustering model.

    Returns:
        JSON object text.
    """

    text = raw_text.strip()
    if text.startswith("```"):
        first_brace = text.find("{")
        last_brace = text.rfind("}")
        assert first_brace >= 0 and last_brace >= 0, "response missing JSON object"
        return text[first_brace : last_brace + 1]
    if text.startswith("{") and text.endswith("}"):
        return text
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    assert first_brace >= 0 and last_brace >= 0, "response missing JSON object"
    return text[first_brace : last_brace + 1]


def parse_prompt_items(*, prompt: str) -> tuple[PromptClusterItem, ...]:
    """Parse the cleaned prompt options from one clustering prompt.

    Args:
        prompt: Logged prompt text.

    Returns:
        Parsed prompt items in prompt order.

    Example:
        >>> items = parse_prompt_items(prompt="x\\n## Options to group:\\n1: count=2 | text=Try factoring")
        >>> items[0].text
        'Try factoring'
    """

    marker_index = prompt.find(OPTIONS_MARKER)
    assert marker_index >= 0, "prompt missing options section"
    options_text = prompt[marker_index + len(OPTIONS_MARKER) :].strip()
    parsed_items: list[PromptClusterItem] = []
    for raw_line in options_text.splitlines():
        match = ITEM_LINE_PATTERN.match(raw_line.strip())
        if match is None:
            continue
        parsed_items.append(
            PromptClusterItem(
                item_id=int(match.group("item_id")),
                source_count=int(match.group("count")),
                text=clean_cluster_text(text=match.group("text")),
            )
        )
    assert parsed_items, "prompt options section did not yield any items"
    return tuple(parsed_items)


def parse_cluster_response(
    *, raw_text: str
) -> tuple[tuple[ClusterGroup, ...], dict[int, str]]:
    """Parse groups and assignments from one raw clustering response.

    Args:
        raw_text: Raw response text logged for the successful attempt.

    Returns:
        Tuple of parsed groups and item-id assignments.
    """

    payload = json.loads(extract_json_object_text(raw_text=raw_text))
    assert isinstance(payload, dict), "cluster response must decode to a mapping"
    groups_payload = payload.get("groups")
    assignments_payload = payload.get("assignments")
    assert isinstance(groups_payload, list), "cluster response missing groups"
    assert isinstance(assignments_payload, dict), "cluster response missing assignments"
    groups = tuple(parse_group(group=group) for group in groups_payload)
    assignments = parse_assignments(assignments_payload=assignments_payload)
    validate_group_keys(groups=groups)
    return groups, assignments


def parse_group(*, group: object) -> ClusterGroup:
    """Parse one group object from the response payload."""

    assert isinstance(group, dict), "group rows must be mappings"
    key = str(group.get("key", "")).strip()
    name = str(group.get("name", "")).strip()
    assert key, "group key must be non-empty"
    assert name, "group name must be non-empty"
    return ClusterGroup(key=key, name=name)


def parse_assignments(*, assignments_payload: dict[str, Any]) -> dict[int, str]:
    """Parse assignment ids into integer-keyed mapping."""

    assignments: dict[int, str] = {}
    for raw_key, raw_value in assignments_payload.items():
        assignment_key = str(raw_value).strip()
        assert str(raw_key).isdigit(), f"invalid assignment id: {raw_key}"
        assert assignment_key, "assignment key must be non-empty"
        assignments[int(raw_key)] = assignment_key
    return assignments


def validate_group_keys(*, groups: tuple[ClusterGroup, ...]) -> None:
    """Assert that returned groups have unique keys."""

    group_keys = {group.key for group in groups}
    assert len(group_keys) == len(groups), "duplicate group keys in response"


def iter_jsonl_rows(*, input_path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON mappings from a JSONL file."""

    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line_text = line.strip()
            if not line_text:
                continue
            payload = json.loads(line_text)
            assert isinstance(payload, dict), "JSONL rows must decode to mappings"
            yield payload


def extract_successful_cluster_attempts(
    *, input_path: Path
) -> tuple[SuccessfulClusterAttempt, ...]:
    """Reconstruct successful clustering attempts from a debug log file.

    Args:
        input_path: ``clustering_debug.jsonl`` path.

    Returns:
        Parsed successful attempts in file order.

    Example:
        >>> from pathlib import Path
        >>> temp = Path('/tmp/nonexistent.jsonl')
        >>> temp.exists()
        False
    """

    pending_raw_responses: list[_PendingRawResponse] = []
    successful_attempts: list[SuccessfulClusterAttempt] = []
    for row in iter_jsonl_rows(input_path=input_path):
        event_name = str(row.get("event", ""))
        if event_name == "attempt_raw_response":
            pending_raw_responses.append(parse_pending_raw_response(row=row))
            continue
        if event_name not in {"attempt_success", "attempt_failure"}:
            continue
        pending = pop_matching_pending_raw(
            pending_raw_responses=pending_raw_responses,
            attempt_number=parse_int_field(row=row, field_name="attempt_number"),
            item_count=parse_int_field(row=row, field_name="item_count"),
            model_id=str(row.get("model_id", "")),
        )
        if event_name != "attempt_success":
            continue
        successful_attempts.append(
            build_successful_attempt(
                success_index=len(successful_attempts) + 1,
                pending=pending,
            )
        )
    return tuple(successful_attempts)


def parse_pending_raw_response(*, row: dict[str, Any]) -> _PendingRawResponse:
    """Parse one raw response row into a pending item."""

    return _PendingRawResponse(
        attempt_number=parse_int_field(row=row, field_name="attempt_number"),
        item_count=parse_int_field(row=row, field_name="item_count"),
        model_id=str(row.get("model_id", "")),
        prompt=str(row.get("prompt", "")),
        raw_text=str(row.get("raw_text", "")),
    )


def parse_int_field(*, row: dict[str, Any], field_name: str) -> int:
    """Parse an integer field from one log row."""

    value = row.get(field_name)
    assert isinstance(value, int), f"{field_name} must be an int"
    return value


def pop_matching_pending_raw(
    *,
    pending_raw_responses: list[_PendingRawResponse],
    attempt_number: int,
    item_count: int,
    model_id: str,
) -> _PendingRawResponse:
    """Pop the most recent raw response matching a success or failure row."""

    for index in range(len(pending_raw_responses) - 1, -1, -1):
        pending = pending_raw_responses[index]
        if (
            pending.attempt_number == attempt_number
            and pending.item_count == item_count
            and pending.model_id == model_id
        ):
            return pending_raw_responses.pop(index)
    raise AssertionError("could not match success/failure row to a raw response")


def build_successful_attempt(
    *, success_index: int, pending: _PendingRawResponse
) -> SuccessfulClusterAttempt:
    """Build one parsed successful attempt from the paired raw response."""

    items = parse_prompt_items(prompt=pending.prompt)
    groups, assignments = parse_cluster_response(raw_text=pending.raw_text)
    assignments = filter_assignments_for_items(items=items, assignments=assignments)
    return SuccessfulClusterAttempt(
        success_index=success_index,
        attempt_number=pending.attempt_number,
        model_id=pending.model_id,
        items=items,
        groups=groups,
        assignments=assignments,
    )


def filter_assignments_for_items(
    *, items: tuple[PromptClusterItem, ...], assignments: dict[int, str]
) -> dict[int, str]:
    """Drop stray assignment ids and keep only ids present in the prompt."""

    expected_ids = {item.item_id for item in items}
    return {
        item_id: cluster_key
        for item_id, cluster_key in assignments.items()
        if item_id in expected_ids
    }


def flatten_assigned_rows(
    *, attempts: Sequence[SuccessfulClusterAttempt]
) -> tuple[AssignedClusterRow, ...]:
    """Flatten successful attempts into item-level assigned rows.

    Args:
        attempts: Successful attempts.

    Returns:
        Flattened assigned rows.
    """

    assigned_rows: list[AssignedClusterRow] = []
    for attempt in attempts:
        cluster_name_by_key = attempt.cluster_name_by_key()
        for item in attempt.items:
            cluster_key = attempt.assignments.get(item.item_id, "other")
            cluster_name = cluster_name_by_key.get(
                cluster_key,
                humanize_cluster_key(cluster_key=cluster_key),
            )
            assigned_rows.append(
                AssignedClusterRow(
                    attempt_index=attempt.success_index,
                    item_id=item.item_id,
                    source_count=item.source_count,
                    text=item.text,
                    cluster_key=cluster_key,
                    cluster_name=cluster_name,
                )
            )
    return tuple(assigned_rows)


def assigned_rows_for_attempt(
    *, attempt: SuccessfulClusterAttempt
) -> tuple[AssignedClusterRow, ...]:
    """Return item-level assigned rows for one successful attempt.

    Args:
        attempt: Successful prompt-clustering attempt.

    Returns:
        Assigned rows for that prompt only.
    """

    return flatten_assigned_rows(attempts=(attempt,))


def humanize_cluster_key(*, cluster_key: str) -> str:
    """Return a readable fallback label for an unknown cluster key."""

    cleaned = cluster_key.replace("_", " ").strip()
    return cleaned.title() if cleaned else "Other"


def select_successful_attempt(
    *, attempts: Sequence[SuccessfulClusterAttempt], success_index: int
) -> SuccessfulClusterAttempt:
    """Select one successful attempt by one-based index.

    Args:
        attempts: Parsed successful attempts.
        success_index: One-based success index to select.

    Returns:
        Selected successful attempt.
    """

    assert attempts, "no successful clustering attempts found"
    assert 1 <= success_index <= len(attempts), "success_index out of range"
    return attempts[success_index - 1]


def summaries_for_attempt(
    *, attempt: SuccessfulClusterAttempt
) -> tuple[TextClusterSummary, ...]:
    """Build per-item summaries for one successful attempt.

    Args:
        attempt: Successful prompt-clustering attempt.

    Returns:
        Per-item summaries sorted by prompt multiplicity then text.

    Example:
        >>> attempt = SuccessfulClusterAttempt(
        ...     success_index=1,
        ...     attempt_number=1,
        ...     model_id="demo",
        ...     items=(PromptClusterItem(item_id=1, source_count=2, text="Try factoring"),),
        ...     groups=(ClusterGroup(key="factor", name="Factor"),),
        ...     assignments={1: "factor"},
        ... )
        >>> summaries_for_attempt(attempt=attempt)[0].occurrences
        2
    """

    summaries: list[TextClusterSummary] = []
    for row in assigned_rows_for_attempt(attempt=attempt):
        summaries.append(
            TextClusterSummary(
                text=row.text,
                occurrences=row.source_count,
                attempts=1,
                cluster_counts=(
                    ClusterCount(
                        cluster_key=row.cluster_key,
                        cluster_name=row.cluster_name,
                        count=row.source_count,
                    ),
                ),
            )
        )
    return tuple(
        sorted(summaries, key=lambda summary: (-summary.occurrences, summary.text))
    )


def aggregate_text_summaries(
    *, assigned_rows: Sequence[AssignedClusterRow]
) -> tuple[TextClusterSummary, ...]:
    """Aggregate item-level rows into one summary per cleaned string.

    Args:
        assigned_rows: Flattened item-level assigned rows.

    Returns:
        Text summaries sorted by descending frequency.
    """

    counters_by_text: dict[str, Counter[tuple[str, str]]] = {}
    attempts_by_text: dict[str, set[int]] = {}
    counts_by_text: Counter[str] = Counter()
    for row in assigned_rows:
        counts_by_text[row.text] += 1
        counters_by_text.setdefault(row.text, Counter())[
            (row.cluster_key, row.cluster_name)
        ] += 1
        attempts_by_text.setdefault(row.text, set()).add(row.attempt_index)
    summaries = [
        TextClusterSummary(
            text=text,
            occurrences=counts_by_text[text],
            attempts=len(attempts_by_text[text]),
            cluster_counts=build_cluster_counts(cluster_counter=counters_by_text[text]),
        )
        for text in counts_by_text
    ]
    return tuple(
        sorted(
            summaries,
            key=lambda summary: (-summary.occurrences, -summary.attempts, summary.text),
        )
    )


def build_cluster_counts(
    *, cluster_counter: Counter[tuple[str, str]]
) -> tuple[ClusterCount, ...]:
    """Convert raw cluster counters into sorted dataclass rows."""

    counts = [
        ClusterCount(
            cluster_key=cluster_key,
            cluster_name=cluster_name,
            count=count,
        )
        for (cluster_key, cluster_name), count in cluster_counter.items()
    ]
    return tuple(
        sorted(
            counts,
            key=lambda count: (-count.count, count.cluster_name, count.cluster_key),
        )
    )


def resolve_required_openai_api_key() -> str:
    """Resolve the OpenAI API key required for sampling requests.

    Returns:
        Non-empty OpenAI API key string.
    """

    api_key = resolve_openai_api_key(env_paths=())
    assert api_key is not None, "OPENAI_API_KEY is required for embedding sampling"
    return api_key


def embedding_vector_dimensions(
    *, embeddings_by_text: dict[str, tuple[float, ...]]
) -> int | None:
    """Return the shared vector length for one text-to-embedding mapping.

    Args:
        embeddings_by_text: Text-to-embedding mapping.

    Returns:
        Shared vector length, or `None` when the mapping is empty.
    """

    if not embeddings_by_text:
        return None
    lengths = {len(vector) for vector in embeddings_by_text.values()}
    assert len(lengths) == 1, "embeddings must all have the same length"
    return next(iter(lengths))


def embed_text_summaries(
    *,
    summaries: Sequence["TextClusterSummary"],
    model: str,
    batch_size: int,
    embedding_dimensions: int | None,
) -> dict[str, tuple[float, ...]]:
    """Embed unique summary texts with the shared OpenAI helper.

    Args:
        summaries: Summary rows to embed.
        model: Embedding model id.
        batch_size: Maximum texts per embedding request.
        embedding_dimensions: Optional output dimension count requested from OpenAI.

    Returns:
        Text-to-embedding mapping for the summary texts.
    """

    if not summaries:
        return {}
    unique_texts = tuple(dict.fromkeys(summary.text for summary in summaries))
    return openai_embeddings_by_text(
        texts=unique_texts,
        openai_api_key=resolve_required_openai_api_key(),
        model=model,
        batch_size=batch_size,
        output_dimensions=embedding_dimensions,
    )


def sample_diverse_text_summaries(
    *,
    summaries: Sequence["TextClusterSummary"],
    embeddings_by_text: Mapping[str, Sequence[float]],
    sample_count: int,
) -> tuple["TextClusterSummary", ...]:
    """Pick a small diverse set of texts using the shared greedy selector.

    Args:
        summaries: All text summaries.
        embeddings_by_text: Text-to-embedding mapping.
        sample_count: Number of texts to return.

    Returns:
        Diversity-sampled summaries.
    """

    if sample_count <= 0:
        return ()
    ordered_summaries = tuple(
        sorted(
            summaries,
            key=lambda summary: (-summary.occurrences, -summary.attempts, summary.text),
        )
    )
    embeddings_for_sampling = {
        summary.text: tuple(float(value) for value in embeddings_by_text[summary.text])
        for summary in ordered_summaries
    }
    matrix = normalized_embedding_matrix(
        texts=[summary.text for summary in ordered_summaries],
        embeddings_by_text=embeddings_for_sampling,
    )
    selected_indices = greedy_diverse_indices(
        matrix=matrix,
        max_count=min(sample_count, len(ordered_summaries)),
    )
    return tuple(ordered_summaries[index] for index in selected_indices)


def run_sampling_pipeline(
    *,
    input_path: Path,
    success_index: int,
    requested_sample_count: int,
    embedding_model: str,
    batch_size: int,
    embedding_dimensions: int | None = None,
) -> ClusterSamplingRun:
    """Run the full parse/embed/sample pipeline for one debug log.

    Args:
        input_path: Source ``clustering_debug.jsonl`` path.
        success_index: One-based successful prompt index to sample within.
        requested_sample_count: Requested sample size before clamping.
        embedding_model: Embedding model id.
        batch_size: Embedding batch size.
        embedding_dimensions: Optional embedding dimension count requested from OpenAI.

    Returns:
        Sampling run bundle with parsed summaries and sampled texts.
    """

    attempts = extract_successful_cluster_attempts(input_path=input_path)
    selected_attempt = select_successful_attempt(
        attempts=attempts,
        success_index=success_index,
    )
    summaries = summaries_for_attempt(attempt=selected_attempt)
    embeddings_by_text = embed_text_summaries(
        summaries=summaries,
        model=embedding_model,
        batch_size=batch_size,
        embedding_dimensions=embedding_dimensions,
    )
    sample_count = choose_sample_count(
        unique_text_count=len(summaries),
        requested_sample_count=requested_sample_count,
    )
    sampled_summaries = sample_diverse_text_summaries(
        summaries=summaries,
        embeddings_by_text=embeddings_by_text,
        sample_count=sample_count,
    )
    return ClusterSamplingRun(
        input_path=input_path,
        embedding_model=embedding_model,
        embedding_dimensions=embedding_vector_dimensions(
            embeddings_by_text=embeddings_by_text,
        ),
        successful_attempt_count=len(attempts),
        selected_success_index=selected_attempt.success_index,
        selected_attempt_number=selected_attempt.attempt_number,
        selected_item_count=len(selected_attempt.items),
        sampled_summaries=sampled_summaries,
        text_summaries=summaries,
    )


def print_sampling_run(*, run: ClusterSamplingRun) -> None:
    """Print a concise terminal summary for one sampling run."""

    print(f"Input file: {run.input_path}")
    print(f"Embedding model: {run.embedding_model}")
    print(f"Embedding dimensions: {run.embedding_dimensions}")
    print(
        f"Parsed {run.successful_attempt_count} successful attempts and selected "
        f"success_index={run.selected_success_index} "
        f"(attempt_number={run.selected_attempt_number}) with "
        f"{run.selected_item_count} cleaned prompt items."
    )
    print("")
    print(f"Diversity sample ({len(run.sampled_summaries)} strings):")
    for index, summary in enumerate(run.sampled_summaries, start=1):
        print(f"{index}. {summary.text}")
        print(f"   occurrences={summary.occurrences} attempts={summary.attempts}")
        wrapped_clusters = textwrap.wrap(
            summary.cluster_summary(),
            width=100,
            subsequent_indent="   ",
        )
        print(f"   clusters={wrapped_clusters[0]}")
        for continuation in wrapped_clusters[1:]:
            print(f"   {continuation}")


def main() -> None:
    """Run the CLI."""

    args = parse_args()
    run = run_sampling_pipeline(
        input_path=args.input_path,
        success_index=args.success_index,
        requested_sample_count=args.sample_count,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
    )
    print_sampling_run(run=run)


if __name__ == "__main__":
    main()
