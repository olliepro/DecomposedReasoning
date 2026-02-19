"""Static HTML report builder with cluster-driven step exploration UI."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import array
import base64
from candidate_clustering import (
    ClusteringArtifacts,
    strip_steer_suffix,
    trailing_steer_suffix_span,
)


class TokenDictionary:
    """Accumulate unique tokens and encode sequences as indices."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {}
        self.tokens: list[str] = []

    def get_id(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.tokens)
            self.tokens.append(token)
        return self.token_to_id[token]

    def encode(self, tokens: list[str]) -> str:
        """Encode list of tokens to Base64 Uint32 array."""
        # 'I' is 4-byte unsigned int
        indices = [self.get_id(t) for t in tokens]
        arr = array.array("I", indices)
        return base64.b64encode(arr.tobytes()).decode("ascii")

    def encode_ids(self, ids: list[int]) -> str:
        """Encode list of token IDs to Base64 Uint32 array."""
        arr = array.array("I", ids)
        return base64.b64encode(arr.tobytes()).decode("ascii")


def encode_float_array(values: list[float]) -> str:
    """Encode float list to Base64 Float32 array."""
    # 'f' is 4-byte float
    arr = array.array("f", values)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def encode_uint8_array(values: list[int]) -> str:
    """Encode int list to Base64 Uint8 array."""
    # 'B' is 1-byte unsigned char
    arr = array.array("B", values)
    return base64.b64encode(arr.tobytes()).decode("ascii")


def compress_token_rows(
    *, rows: list[dict[str, Any]], token_dict: TokenDictionary
) -> dict[str, Any]:
    """Compress token rows into SoA format with Base64 arrays.

    Args:
        rows: Source token rows.
        token_dict: Shared token dictionary.

    Returns:
        Compressed token block.
    """
    if not rows:
        return {}
    tokens = [str(r.get("token", "")) for r in rows]
    probs = [float(r.get("probability", 0.0)) for r in rows]
    entropies = [float(r.get("entropy", 0.0)) for r in rows]

    # Process top_logprobs
    top_ids = []
    top_probs = []
    top_counts = []

    for row in rows:
        # Robust probability extraction with fallback
        prob = float(row.get("probability", 0.0))
        if prob == 0.0:
            # Fallback 1: Try logprob
            if "logprob" in row:
                try:
                    import math

                    prob = math.exp(float(row["logprob"]))
                except Exception:
                    pass

            # Fallback 2: Try to find selected token in alternatives
            if prob == 0.0:
                selected_token = str(row.get("token", "")).strip()
                alternatives = row.get("alternatives", [])
                for alt in alternatives:
                    if str(alt.get("token", "")).strip() == selected_token:
                        # Found it!
                        prob = float(alt.get("probability", 0.0))
                        if prob == 0.0 and "logprob" in alt:
                            try:
                                import math

                                prob = math.exp(float(alt["logprob"]))
                            except Exception:
                                pass
                        break

        probs.append(prob)

        tops = row.get("alternatives", []) or []
        top_counts.append(len(tops))
        for t in tops:
            top_ids.append(token_dict.get_id(str(t.get("token", ""))))
            # Alternatives use 'probability' not 'logprob', but fallback for robustness
            prob = float(t.get("probability", 0.0))
            if prob == 0.0 and "logprob" in t:
                # If probability is missing or zero, try to recover from logprob
                import math

                prob = math.exp(float(t["logprob"]))
            top_probs.append(prob)

    return {
        "ids": token_dict.encode(tokens),
        "probs": encode_float_array(probs),
        "entropies": encode_float_array(entropies),
        "top_ids": token_dict.encode_ids(top_ids),
        "top_probs": encode_float_array(top_probs),
        "top_counts": encode_uint8_array(top_counts),
    }


STEER_CLOSE_PATTERN = re.compile(r"</steer>", flags=re.IGNORECASE)
THINK_CLOSE_PATTERN = re.compile(r"</think\s*>", flags=re.IGNORECASE)
STEER_EXEC_PAIR_PATTERN = re.compile(
    r"<steer\b[^>]*>(.*?)</steer>\s*<exec(?:ute)?\b[^>]*>(.*?)</exec(?:ute)?>",
    flags=re.IGNORECASE | re.DOTALL,
)


def empty_clustering_artifacts() -> ClusteringArtifacts:
    """Build empty clustering artifacts used when clustering is absent.

    Args:
        None.

    Returns:
        Empty `ClusteringArtifacts` object.
    """

    return ClusteringArtifacts(
        mode="none",
        warnings=(),
        summaries_by_step={},
        assignments_by_candidate={},
    )


def parse_optional_int(*, value: object) -> int | None:
    """Parse integer-like values and return `None` for non-integer inputs.

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


def selected_candidates_by_step(*, steps: list[dict[str, Any]]) -> dict[int, int]:
    """Build mapping from step index to selected candidate index.

    Args:
        steps: Branch-step artifact rows.

    Returns:
        Step-to-candidate selection mapping.
    """

    selected: dict[int, int] = {}
    for row in steps:
        step = parse_optional_int(value=row.get("step_index"))
        candidate = parse_optional_int(value=row.get("selected_candidate_index"))
        if step is None or candidate is None or candidate < 0:
            continue
        selected[step] = candidate
    return selected


def selected_text_by_step(*, steps: list[dict[str, Any]]) -> dict[int, str]:
    """Build mapping from step index to cleaned selected steer text.

    Args:
        steps: Branch-step artifact rows.

    Returns:
        Step-to-cleaned selected steer text mapping.
    """

    selected_texts: dict[int, str] = {}
    for row in steps:
        step = parse_optional_int(value=row.get("step_index"))
        if step is None:
            continue
        text = strip_steer_suffix(text=str(row.get("selected_text", "")))
        selected_texts[step] = text
    return selected_texts


def step_status_by_index(
    *, steps: list[dict[str, Any]]
) -> dict[int, dict[str, str | bool]]:
    """Build map from step index to terminal status metadata.

    Args:
        steps: Branch-step artifact rows.

    Returns:
        Step-indexed mapping with `terminated` and `termination_reason`.
    """

    statuses: dict[int, dict[str, str | bool]] = {}
    for row in steps:
        step = parse_optional_int(value=row.get("step_index"))
        if step is None:
            continue
        statuses[step] = {
            "terminated": bool(row.get("terminated", False)),
            "termination_reason": str(row.get("termination_reason", "")),
        }
    return statuses


def build_alternative_rows(*, raw: object) -> list[dict[str, Any]]:
    """Normalize top-token alternative rows for tooltip rendering.

    Args:
        raw: Raw alternatives payload.

    Returns:
        List of `token` and `probability` mappings.
    """

    if not isinstance(raw, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in raw[:10]:
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "token": str(item.get("token", "")),
                "probability": float(item.get("probability", 0.0)),
            }
        )
    return rows


def candidate_token_rows(
    *, token_stats: list[dict[str, Any]]
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group candidate-token rows by `(step_index, candidate_index)`.

    Args:
        token_stats: Token-stat artifact rows.

    Returns:
        Grouped and token-index sorted candidate rows.
    """

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in token_stats:
        if str(row.get("source", "")) != "candidate":
            continue
        step = parse_optional_int(value=row.get("step_index"))
        candidate = parse_optional_int(value=row.get("candidate_index"))
        token_index = parse_optional_int(value=row.get("token_index"))
        if step is None or candidate is None or token_index is None:
            continue
        grouped[(step, candidate)].append(
            {
                "token_index": token_index,
                "token": str(row.get("token", "")),
                "probability": float(row.get("probability", 0.0)),
                "logprob": float(row.get("logprob", 0.0)),
                "entropy": float(row.get("entropy", 0.0)),
                "alternatives": build_alternative_rows(raw=row.get("alternatives")),
            }
        )
    for key in grouped:
        grouped[key].sort(key=lambda item: int(item["token_index"]))
    return grouped


def rollout_token_rows_by_step(
    *, token_stats: list[dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    """Group selected-rollout token rows by step index.

    Args:
        token_stats: Token-stat artifact rows.

    Returns:
        Step-index grouped rollout token rows in file order.
    """

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in token_stats:
        if str(row.get("source", "")) != "rollout":
            continue
        step = parse_optional_int(value=row.get("step_index"))
        token_index = parse_optional_int(value=row.get("token_index"))
        if step is None or token_index is None:
            continue
        grouped[step].append(
            {
                "token_index": token_index,
                "token": str(row.get("token", "")),
                "probability": float(row.get("probability", 0.0)),
                "logprob": float(row.get("logprob", 0.0)),
                "entropy": float(row.get("entropy", 0.0)),
                "alternatives": build_alternative_rows(raw=row.get("alternatives")),
            }
        )
    return grouped


def rollout_probability_distribution(
    *, token_stats: list[dict[str, Any]]
) -> list[float]:
    """Collect sorted rollout-token probabilities for global percentile coloring.

    Args:
        token_stats: Token-stat artifact rows.

    Returns:
        Sorted rollout probabilities clamped to `[0, 1]`.
    """

    probabilities: list[float] = []
    for row in token_stats:
        if str(row.get("source", "")) != "rollout":
            continue
        probability = float(row.get("probability", 0.0))
        probabilities.append(max(0.0, min(1.0, probability)))
    return sorted(probabilities)


def rollout_entropy_distribution(*, token_stats: list[dict[str, Any]]) -> list[float]:
    """Collect sorted rollout-token entropies for global percentile coloring.

    Args:
        token_stats: Token-stat artifact rows.

    Returns:
        Sorted non-negative rollout entropies.
    """

    entropies: list[float] = []
    for row in token_stats:
        if str(row.get("source", "")) != "rollout":
            continue
        entropy = float(row.get("entropy", 0.0))
        entropies.append(max(0.0, entropy))
    return sorted(entropies)


def candidate_text_by_key(
    *, candidates: list[dict[str, Any]]
) -> dict[tuple[int, int], str]:
    """Build map from `(step_index, candidate_index)` to candidate text.

    Args:
        candidates: Candidate artifact rows.

    Returns:
        Candidate text keyed by step/candidate index.
    """

    texts: dict[tuple[int, int], str] = {}
    for row in candidates:
        step = parse_optional_int(value=row.get("step_index"))
        candidate = parse_optional_int(value=row.get("candidate_index"))
        if step is None or candidate is None:
            continue
        texts[(step, candidate)] = str(row.get("text", ""))
    return texts


def token_rows_to_char_length(
    *, rows: list[dict[str, Any]], target_char_length: int
) -> list[dict[str, Any]]:
    """Trim token rows to a character-length boundary.

    Args:
        rows: Ordered token rows.
        target_char_length: Target character length.

    Returns:
        Token rows whose end offsets are within target char length.
    """

    if not rows:
        return []
    if int(target_char_length) <= 0:
        return []
    _, spans = stitch_token_rows(rows=rows)
    trimmed: list[dict[str, Any]] = []
    for row, span in zip(rows, spans):
        start = int(span[0])
        end = int(span[1])
        if end <= int(target_char_length):
            trimmed.append(row)
            continue
        if start < int(target_char_length):
            token_text = str(row.get("token", ""))
            keep_chars = max(0, int(target_char_length) - start)
            if keep_chars > 0:
                trimmed.append({**row, "token": token_text[:keep_chars]})
        break
    return trimmed


def synthetic_text_token(*, token: str) -> dict[str, Any]:
    """Build synthetic token row used to preserve exact trajectory text.

    Args:
        token: Text fragment.

    Returns:
        Synthetic row with neutral metadata and synthetic marker.
    """

    return {
        "token_index": -1,
        "token": token,
        "probability": 0.0,
        "entropy": 0.0,
        "alternatives": [],
        "synthetic": True,
    }


def row_fragments_in_span(
    *,
    rows: list[dict[str, Any]],
    spans: list[tuple[int, int]],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    """Extract row fragments that cover `[start, end)` source char span.

    Args:
        rows: Source rows.
        spans: Source row char spans.
        start: Inclusive source char start.
        end: Exclusive source char end.

    Returns:
        Source row fragments preserving per-token metadata.
    """

    if int(start) >= int(end):
        return []
    fragments: list[dict[str, Any]] = []
    for row, span in zip(rows, spans):
        span_start = int(span[0])
        span_end = int(span[1])
        if span_end <= int(start):
            continue
        if span_start >= int(end):
            break
        token = str(row.get("token", ""))
        slice_start = max(int(start), span_start) - span_start
        slice_end = min(int(end), span_end) - span_start
        if slice_start >= slice_end:
            continue
        fragments.append({**row, "token": token[slice_start:slice_end]})
    return fragments


def align_rows_to_target_text(
    *, rows: list[dict[str, Any]], target_text: str
) -> list[dict[str, Any]]:
    """Align token rows to exact target text and correct drift.

    Args:
        rows: Candidate trajectory token rows.
        target_text: Exact final trajectory text.

    Returns:
        Token rows aligned to target text.
    """

    if not target_text:
        return []
    source_rows = [row for row in rows if str(row.get("token", ""))]
    if not source_rows:
        return [synthetic_text_token(token=target_text)]
    source_text, source_spans = stitch_token_rows(rows=source_rows)
    if source_text == target_text:
        return source_rows
    aligned: list[dict[str, Any]] = []
    matcher = SequenceMatcher(a=source_text, b=target_text, autojunk=False)
    for (
        tag,
        source_start,
        source_end,
        target_start,
        target_end,
    ) in matcher.get_opcodes():
        if tag == "equal":
            aligned.extend(
                row_fragments_in_span(
                    rows=source_rows,
                    spans=source_spans,
                    start=int(source_start),
                    end=int(source_end),
                )
            )
            continue
        if tag in {"replace", "insert"} and int(target_start) < int(target_end):
            aligned.append(
                synthetic_text_token(token=target_text[target_start:target_end])
            )
    aligned_text = "".join(str(row.get("token", "")) for row in aligned)
    if aligned_text == target_text:
        return aligned
    return [synthetic_text_token(token=target_text)]


def trajectory_token_rows(
    *,
    steps: list[dict[str, Any]],
    token_stats: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    final_text: str,
) -> list[dict[str, Any]]:
    """Build ordered trajectory token rows across rollout and selected steers.

    Args:
        steps: Branch-step artifact rows.
        token_stats: Token-stat artifact rows.
        candidates: Candidate artifact rows.
        final_text: Final selected trajectory string.

    Returns:
        Ordered token rows for entire selected trajectory.
    """

    rollout_by_step = rollout_token_rows_by_step(token_stats=token_stats)
    candidate_by_key = candidate_token_rows(token_stats=token_stats)
    candidate_text_by_step_candidate = candidate_text_by_key(candidates=candidates)
    step_rows_by_index: dict[int, dict[str, Any]] = {}
    for step_row in steps:
        step_index = parse_optional_int(value=step_row.get("step_index"))
        if step_index is None:
            continue
        step_rows_by_index[int(step_index)] = step_row
    selected_map = selected_candidates_by_step(steps=steps)
    step_indices = sorted(
        set(rollout_by_step) | set(step_rows_by_index) | set(selected_map)
    )
    ordered_rows: list[dict[str, Any]] = []
    built_length = 0
    for step_index in step_indices:
        step_row = step_rows_by_index.get(step_index, {})
        prefix_char_end = parse_optional_int(value=step_row.get("prefix_char_end"))
        rollout_rows = rollout_by_step.get(step_index, [])
        if prefix_char_end is not None:
            remaining_rollout_chars = max(0, int(prefix_char_end) - int(built_length))
            rollout_rows = token_rows_to_char_length(
                rows=rollout_rows,
                target_char_length=remaining_rollout_chars,
            )
        ordered_rows.extend(rollout_rows)
        built_length += sum(len(str(row.get("token", ""))) for row in rollout_rows)
        selected_candidate = parse_optional_int(
            value=step_row.get("selected_candidate_index", selected_map.get(step_index))
        )
        if selected_candidate is None or int(selected_candidate) < 0:
            continue
        selected_rows = candidate_by_key.get((step_index, int(selected_candidate)), [])
        selected_text = candidate_text_by_step_candidate.get(
            (step_index, int(selected_candidate)),
            str(step_row.get("selected_text", "")),
        )
        selected_rows = token_rows_to_char_length(
            rows=selected_rows,
            target_char_length=len(selected_text),
        )
        ordered_rows.extend(selected_rows)
        built_length += sum(len(str(row.get("token", ""))) for row in selected_rows)
    capped_rows = token_rows_to_char_length(
        rows=ordered_rows, target_char_length=len(final_text)
    )
    return align_rows_to_target_text(rows=capped_rows, target_text=final_text)


def stitch_token_rows(
    *, rows: list[dict[str, Any]]
) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate token texts and track start/end char spans per token.

    Args:
        rows: Candidate token rows.

    Returns:
        Tuple of stitched text and token span list.
    """

    stitched = ""
    spans: list[tuple[int, int]] = []
    for row in rows:
        token_text = str(row.get("token", ""))
        start = len(stitched)
        stitched += token_text
        spans.append((start, len(stitched)))
    return stitched, spans


def steer_boundary(*, stitched: str) -> tuple[int, int] | None:
    """Find first `</steer>` boundary in stitched candidate text.

    Args:
        stitched: Concatenated token text.

    Returns:
        `(start, end)` char span for boundary, else `None`.
    """

    match = STEER_CLOSE_PATTERN.search(stitched)
    if match is not None:
        return (int(match.start()), int(match.end()))
    return trailing_steer_suffix_span(text=stitched)


def steer_token_rows_without_close_tag(
    *, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Keep only steer-token rows and exclude rows that form `</steer>`.

    Args:
        rows: Candidate token rows.

    Returns:
        Rows strictly before the first close-tag start.
    """

    stitched, spans = stitch_token_rows(rows=rows)
    boundary = steer_boundary(stitched=stitched)
    if boundary is None:
        return rows
    return [row for row, span in zip(rows, spans) if int(span[1]) <= int(boundary[0])]


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
        Mapping from zero-based step index to execution markdown text.
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


def final_answer_after_think(*, final_text: str) -> str:
    """Extract answer text that appears after first `</think>` close tag.

    Args:
        final_text: Final assistant text.

    Returns:
        Text after the first `</think>` tag, or empty string if absent.
    """

    match = THINK_CLOSE_PATTERN.search(final_text)
    if match is None:
        return ""
    return str(final_text[match.end() :]).strip()


def candidate_token_payloads(
    *, token_stats: list[dict[str, Any]]
) -> dict[tuple[int, int], dict[str, Any]]:
    """Build token payload per candidate key.

    Args:
        token_stats: Token-stat artifact rows.

    Returns:
        Mapping from candidate key to steer-only `tokens` and `full_tokens`.
    """

    grouped = candidate_token_rows(token_stats=token_stats)
    payloads: dict[tuple[int, int], dict[str, Any]] = {}
    for key, rows in grouped.items():
        payloads[key] = {
            "tokens": steer_token_rows_without_close_tag(rows=rows),
            "full_tokens": list(rows),
        }
    return payloads


def candidate_rows_by_step(
    *, candidates: list[dict[str, Any]]
) -> dict[int, list[dict[str, Any]]]:
    """Group candidate rows by step index.

    Args:
        candidates: Candidate artifact rows.

    Returns:
        Step-index grouped candidate rows.
    """

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        step = parse_optional_int(value=row.get("step_index"))
        if step is None:
            continue
        grouped[step].append(row)
    return grouped


def candidate_base_entry(
    *,
    row: dict[str, Any],
    step_index: int,
    selected_index: int,
    payloads: dict[tuple[int, int], dict[str, Any]],
    clustering: ClusteringArtifacts,
) -> dict[str, Any] | None:
    """Build one base candidate entry for dedupe + cluster grouping.

    Args:
        row: Candidate artifact row.
        step_index: Step index.
        selected_index: Selected candidate index for step.
        payloads: Candidate token payload map.
        clustering: Cluster assignments.

    Returns:
        Base candidate entry or `None`.
    """

    candidate_index = parse_optional_int(value=row.get("candidate_index"))
    if candidate_index is None:
        return None
    assignment = clustering.candidate_assignment(
        step_index=step_index, candidate_index=candidate_index
    )
    clean_text = strip_steer_suffix(text=str(row.get("text", "")))
    if assignment is not None:
        clean_text = assignment.clean_text
    cluster_id = assignment.cluster_id if assignment is not None else -1
    cluster_name = assignment.cluster_name if assignment is not None else "Unclustered"
    payload = payloads.get((step_index, candidate_index), {})
    tokens = payload.get("tokens") or row.get("tokens") or []
    full_tokens = payload.get("full_tokens") or row.get("full_tokens") or tokens
    return {
        "candidate_index": candidate_index,
        "text": clean_text,
        "selected": bool(candidate_index == selected_index),
        "cluster_id": cluster_id,
        "cluster_name": cluster_name,
        "tokens": tokens,
        "full_tokens": full_tokens,
    }


def dedupe_cluster_entries(
    *, entries: list[dict[str, Any]], token_dict: TokenDictionary
) -> list[dict[str, Any]]:
    """Deduplicate candidate entries by text and count occurrences.

    Args:
        entries: Candidate entries within one cluster.
        token_dict: Shared token dictionary.

    Returns:
        Deduplicated candidate rows with occurrence counts.
    """

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["text"])].append(entry)
    deduped: list[dict[str, Any]] = []
    for text, group in grouped.items():
        representative = representative_entry(entries=group)
        deduped.append(
            {
                "text": text,
                "count": len(group),
                "selected": any(bool(item["selected"]) for item in group),
                "tokens": compress_token_rows(
                    rows=representative["tokens"], token_dict=token_dict
                ),
                "full_tokens": compress_token_rows(
                    rows=representative.get("full_tokens", representative["tokens"]),
                    token_dict=token_dict,
                ),
                "candidate_indices": [int(item["candidate_index"]) for item in group],
            }
        )
    deduped.sort(
        key=lambda item: (
            0 if item["selected"] else 1,
            -int(item["count"]),
            item["text"],
        )
    )
    return deduped


def representative_entry(*, entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick representative entry for one deduped candidate text group.

    Args:
        entries: Candidate entries for one text value.

    Returns:
        Representative candidate entry.
    """

    selected = [entry for entry in entries if bool(entry["selected"])]
    if selected:
        return sorted(selected, key=lambda row: int(row["candidate_index"]))[0]
    return sorted(entries, key=lambda row: int(row["candidate_index"]))[0]


def cluster_rows_for_step(
    *,
    entries: list[dict[str, Any]],
    clustering: ClusteringArtifacts,
    step_index: int,
    token_dict: TokenDictionary,
) -> list[dict[str, Any]]:
    """Build cluster payload rows containing deduped candidate entries.

    Args:
        entries: Base candidate entries for one step.
        clustering: Clustering metadata.
        step_index: Step index.
        token_dict: Shared token dictionary.

    Returns:
        Cluster rows with deduped candidate items.
    """

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[int(entry["cluster_id"])].append(entry)
    name_by_id = {
        int(item["cluster_id"]): item["name"]
        for item in clustering.summary_dicts_for_step(step_index=step_index)
    }
    clusters: list[dict[str, Any]] = []
    for cluster_id in sorted(grouped):
        deduped = dedupe_cluster_entries(
            entries=grouped[cluster_id], token_dict=token_dict
        )
        clusters.append(
            {
                "cluster_id": cluster_id,
                "name": name_by_id.get(
                    cluster_id, grouped[cluster_id][0]["cluster_name"]
                ),
                "count": len(grouped[cluster_id]),
                "unique_count": len(deduped),
                "items": deduped,
            }
        )
    clusters.sort(key=lambda item: (-int(item["count"]), int(item["cluster_id"])))
    return clusters


def chosen_entry(*, clusters: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return selected deduped candidate entry from step cluster rows.

    Args:
        clusters: Cluster rows with `items`.

    Returns:
        Selected deduped row, or `None` if no selected row exists.
    """

    for cluster in clusters:
        for item in cluster["items"]:
            if bool(item["selected"]):
                return item
    return None


def step_view(
    *,
    step_index: int,
    candidates: list[dict[str, Any]],
    selected_map: dict[int, int],
    selected_text_map: dict[int, str],
    status_map: dict[int, dict[str, str | bool]],
    payloads: dict[tuple[int, int], dict[str, Any]],
    rollout_tokens_by_step: dict[int, list[dict[str, Any]]],
    execution_text_by_step: dict[int, str],
    clustering: ClusteringArtifacts,
    token_dict: TokenDictionary,
) -> dict[str, Any]:
    """Build one collapsible step payload row for report UI.

    Args:
        step_index: Step index.
        candidates: Candidate artifact rows for the step.
        selected_map: Step-to-selected candidate index mapping.
        selected_text_map: Step-to-selected cleaned text mapping.
        status_map: Step-to-terminal status metadata mapping.
        payloads: Candidate token payload map.
        rollout_tokens_by_step: Step-indexed selected-rollout token rows.
            Rollout step `i + 1` carries `<exec>...</exec>` for selected steer `i`.
        execution_text_by_step: Step-indexed execution text from `final_text`.
        clustering: Clustering metadata.
        token_dict: Shared token dictionary.

    Returns:
        One step payload mapping.
    """

    selected_index = selected_map.get(step_index, -1)
    base_entries: list[dict[str, Any]] = []
    for row in sorted(
        candidates, key=lambda item: int(item.get("candidate_index", -1))
    ):
        entry = candidate_base_entry(
            row=row,
            step_index=step_index,
            selected_index=selected_index,
            payloads=payloads,
            clustering=clustering,
        )
        if entry is not None:
            base_entries.append(entry)
    clusters = cluster_rows_for_step(
        entries=base_entries,
        clustering=clustering,
        step_index=step_index,
        token_dict=token_dict,
    )
    selected_text = selected_text_map.get(step_index, "")
    chosen = chosen_entry(clusters=clusters)
    if chosen is not None:
        selected_text = str(chosen["text"])
        chosen = {
            **chosen,
            "rollout_tokens": compress_token_rows(
                rows=rollout_tokens_by_step.get(step_index + 1, []),
                token_dict=token_dict,
            ),
            "execution_text": execution_text_by_step.get(step_index, ""),
        }
    status = status_map.get(step_index, {"terminated": False, "termination_reason": ""})
    return {
        "step_index": step_index,
        "selected_text": selected_text,
        "cluster_count": len(clusters),
        "candidate_count": len(base_entries),
        "terminated": bool(status.get("terminated", False)),
        "termination_reason": str(status.get("termination_reason", "")),
        "clusters": clusters,
        "chosen_entry": chosen,
    }


def step_indices(
    *,
    selected_map: dict[int, int],
    selected_text_map: dict[int, str],
    grouped_candidates: dict[int, list[dict[str, Any]]],
    clustering: ClusteringArtifacts,
) -> list[int]:
    """Resolve sorted unique step indices across artifact sources.

    Args:
        selected_map: Step-to-selected candidate mapping.
        selected_text_map: Step-to-selected text mapping.
        grouped_candidates: Step-grouped candidate rows.
        clustering: Cluster summary mapping by step.

    Returns:
        Sorted unique step indices.
    """

    indices = set(selected_map)
    indices.update(selected_text_map)
    indices.update(grouped_candidates)
    indices.update(clustering.summaries_by_step)
    return sorted(indices)


def is_blank_step_view(*, view: dict[str, Any]) -> bool:
    """Return whether a step view has no selected text and no candidates.

    Args:
        view: One step payload row.

    Returns:
        `True` when step is empty and should be omitted from UI.
    """

    selected_text = str(view.get("selected_text", "")).strip()
    if selected_text:
        return False
    if int(view.get("candidate_count", 0)) > 0:
        return False
    if int(view.get("cluster_count", 0)) > 0:
        return False
    return view.get("chosen_entry") is None


def build_report_payload(
    *,
    config: dict[str, Any],
    steps: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    token_stats: list[dict[str, Any]],
    final_text: str,
    clustering: ClusteringArtifacts | None = None,
) -> dict[str, Any]:
    """Build JSON payload used by the interactive HTML report.

    Args:
        config: Serialized run config.
        steps: Branch-step artifact rows.
        candidates: Candidate artifact rows.
        token_stats: Token-stat artifact rows.
        final_text: Final generated assistant text.
        clustering: Optional clustering output.

    Returns:
        JSON-serializable report payload.
    """

    resolved_clustering = clustering or empty_clustering_artifacts()
    selected_map = selected_candidates_by_step(steps=steps)
    selected_text_map = selected_text_by_step(steps=steps)
    status_map = step_status_by_index(steps=steps)
    grouped_candidates = candidate_rows_by_step(candidates=candidates)
    payloads = candidate_token_payloads(token_stats=token_stats)
    rollout_tokens = rollout_token_rows_by_step(token_stats=token_stats)
    rollout_probabilities = rollout_probability_distribution(token_stats=token_stats)
    rollout_entropies = rollout_entropy_distribution(token_stats=token_stats)
    trajectory_tokens = trajectory_token_rows(
        steps=steps,
        token_stats=token_stats,
        candidates=candidates,
        final_text=final_text,
    )
    execution_by_step = execution_text_by_step_from_final_text(
        final_text=final_text,
        selected_text_by_step_index=selected_text_map,
    )
    token_dict = TokenDictionary()
    raw_views = [
        step_view(
            step_index=step_index,
            candidates=grouped_candidates.get(step_index, []),
            selected_map=selected_map,
            selected_text_map=selected_text_map,
            status_map=status_map,
            payloads=payloads,
            rollout_tokens_by_step=rollout_tokens,
            execution_text_by_step=execution_by_step,
            clustering=resolved_clustering,
            token_dict=token_dict,
        )
        for step_index in step_indices(
            selected_map=selected_map,
            selected_text_map=selected_text_map,
            grouped_candidates=grouped_candidates,
            clustering=resolved_clustering,
        )
    ]
    views = [view for view in raw_views if not is_blank_step_view(view=view)]
    return {
        "config": config,
        "step_views": views,
        "final_text": final_text,
        "final_answer_text": final_answer_after_think(final_text=final_text),
        "rollout_probabilities": encode_float_array(
            rollout_probabilities
        ),  # Compress this too
        "rollout_entropies": encode_float_array(rollout_entropies),  # Compress this too
        "trajectory_tokens": compress_token_rows(
            rows=trajectory_tokens, token_dict=token_dict
        ),
        "trajectory_token_count": len(rollout_probabilities),
        "cluster_mode": resolved_clustering.mode,
        "cluster_warnings": list(resolved_clustering.warnings),
        "token_list": token_dict.tokens,
    }


def safe_json_for_html_script(*, payload: dict[str, Any]) -> str:
    """Serialize payload safely for inline `<script type='application/json'>`.

    Args:
        payload: JSON payload mapping.

    Returns:
        Escaped JSON string.
    """

    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c")
    encoded = encoded.replace("\u2028", "\\u2028")
    return encoded.replace("\u2029", "\\u2029")


@dataclass(frozen=True)
class ReportOutput:
    """One selectable report output in the viewer bundle.

    Args:
        output_id: Stable viewer id for this output.
        prompt: Input prompt text for display.
        run_dir: Source run directory path.
        report_payload: Serialized report payload for one run.

    Returns:
        `ReportOutput` instance.
    """

    output_id: str
    prompt: str
    run_dir: str
    report_payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize report output for the browser bundle.

        Args:
            None.

        Returns:
            JSON-safe mapping for one output entry.
        """

        prompt_text = self.prompt.strip()
        prompt_label = prompt_text or "(no prompt)"
        short_prompt = (
            prompt_label if len(prompt_label) <= 96 else f"{prompt_label[:93]}..."
        )
        return {
            "id": self.output_id,
            "prompt": prompt_text,
            "label": short_prompt,
            "run_dir": self.run_dir,
            "report": self.report_payload,
        }


def report_bundle_payload(*, outputs: list[ReportOutput]) -> dict[str, Any]:
    """Build viewer bundle payload for one or more outputs.

    Args:
        outputs: Ordered report output entries.

    Returns:
        JSON payload consumed by the browser app.
    """

    return {
        "outputs": [output.to_dict() for output in outputs],
        "algorithm_overview": (
            "Each run branches at every step, clusters candidate steers, chooses one "
            "trajectory path, and tracks rollout token metrics for uncertainty analysis."
        ),
    }


def render_report_html(
    *,
    report_bundle: dict[str, Any] | None = None,
    report_payload: dict[str, Any] | None = None,
) -> str:
    """Render static viewer shell HTML with external CSS/JS assets.

    Args:
        report_bundle: Bundle payload for multiple outputs.
        report_payload: Backward-compatible single output payload.

    Returns:
        Complete viewer HTML string.
    """

    # Note: The build_report script still writes report_data.json for large bundles.
    embedded_payload = report_bundle if report_bundle is not None else report_payload
    embedded_json = safe_json_for_html_script(payload=embedded_payload or {})

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width, initial-scale=1'>",
            "<title>Steer Branching Explorer</title>",
            "<link rel='preconnect' href='https://fonts.googleapis.com'>",
            "<link rel='preconnect' href='https://fonts.gstatic.com' crossorigin>",
            "<link href='https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:ital,wght@0,500;0,600;1,500&display=swap' rel='stylesheet'>",
            "<link href='https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,400,0,0' rel='stylesheet'>",
            "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css'>",
            "<link rel='stylesheet' href='https://cdn.jsdelivr.net/npm/markdown-it-texmath@1.0.0/css/texmath.min.css'>",
            "<link rel='stylesheet' href='report_assets/styles.css'>",
            "</head>",
            "<body>",
            "<header class='top-chrome'>",
            "<section class='top-chrome-inner'>",
            "<button type='button' class='menu-toggle chrome-icon-btn' data-sidebar-open aria-expanded='false' aria-controls='report-sidebar' aria-label='Open menu'><span class='material-symbols-rounded chrome-icon' aria-hidden='true'>menu</span></button>",
            "<button type='button' class='home-toggle chrome-icon-btn' data-show-home aria-label='Go home'><span class='material-symbols-rounded chrome-icon' aria-hidden='true'>home</span></button>",
            "<div class='top-chrome-title'>Steer Branching Explorer</div>",
            "<div id='top-controls-wrap' class='top-controls-wrap'><button type='button' id='top-settings-toggle' class='chrome-icon-btn' aria-label='Open settings' aria-expanded='false'><span class='material-symbols-rounded chrome-icon' aria-hidden='true'>tune</span></button><section id='top-controls-popover' class='top-controls-popover hidden' aria-hidden='true'><div id='top-controls' class='meta-block controls-host'></div></section></div>",
            "</section>",
            "</header>",
            "<main>",
            "<section class='workspace-grid'>",
            "<button type='button' id='sidebar-scrim' class='sidebar-scrim' aria-hidden='true' aria-label='Close menu'></button>",
            "<aside id='report-sidebar' class='panel sidebar-panel' aria-hidden='true'>",
            "<section class='sidebar-header'><h2 class='sidebar-header-title'>Steer Branching Explorer</h2><button type='button' class='sidebar-close' data-sidebar-close aria-label='Close menu'><span class='sidebar-close-icon' aria-hidden='true'>&#10005;</span></button></section>",
            "<section class='sidebar-body'><button type='button' class='sidebar-home-btn' data-show-home aria-label='Go home'><span class='material-symbols-rounded sidebar-home-icon' aria-hidden='true'>home</span><span>Home</span></button><section id='sidebar-controls' class='meta-block controls-host'></section><div id='sidebar-output-list'></div>",
            "</section>",
            "</aside>",
            "<section id='home-view' class='panel hero'></section>",
            "<section id='report-view' class='hidden-block'><section class='content-grid'><section id='final-answer' class='side-column'></section><div id='column-divider' class='column-divider' role='separator' aria-label='Resize columns' aria-orientation='vertical'></div><section id='timeline' class='timeline'></section></section></section>",
            "</section>",
            "</main>",
            "<div id='tooltip' class='tooltip hidden'></div>",
            "<div id='trajectory-overlay' class='trajectory-overlay hidden' aria-hidden='true'><section id='trajectory-overlay-panel' class='trajectory-overlay-panel panel'><button type='button' class='trajectory-overlay-close' data-trajectory-overlay-close aria-label='Close trajectory fullscreen'><span class='material-symbols-rounded' aria-hidden='true'>close</span></button><h2 class='trajectory-overlay-title' data-trajectory-overlay-title>Trajectory</h2><div class='trajectory-overlay-body' data-trajectory-overlay-body></div></section></div>",
            f"<script id='report-bundle-data' type='application/json'>{embedded_json}</script>",
            "<!-- Data is loaded externally from report_data.json and overrides embedded fallback. -->",
            "<script src='https://cdn.jsdelivr.net/npm/markdown-it@14.1.0/dist/markdown-it.min.js'></script>",
            "<script src='https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js'></script>",
            "<script src='https://cdn.jsdelivr.net/npm/markdown-it-texmath@1.0.0/texmath.min.js'></script>",
            "<script src='report_assets/app.js'></script>",
            "</body>",
            "</html>",
        ]
    )
