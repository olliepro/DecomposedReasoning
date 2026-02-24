"""Legacy steer-rollout text helpers used by branching executor."""

from __future__ import annotations

import re

THINK_CLOSE_PATTERN = re.compile(r"</think>", flags=re.IGNORECASE)
THINK_CLOSE_PARTIAL_SUFFIX_PATTERN = re.compile(
    r"</t(?:h(?:i(?:n(?:k)?)?)?)?$",
    flags=re.IGNORECASE,
)
REPAIR_TAGS = ("<steer>", "</steer>", "<exec>", "</exec>")
STEER_CLOSE_TAG = "</steer>"


def contains_think_close(*, text: str) -> bool:
    """Return whether text contains a full `</think>` marker.

    Args:
        text: Generated text chunk.

    Returns:
        `True` when a full close-think marker is present.
    """

    return THINK_CLOSE_PATTERN.search(text) is not None


def contains_think_close_or_partial(*, text: str) -> bool:
    """Return whether text contains full or trailing-partial `</think>`.

    Args:
        text: Generated text chunk.

    Returns:
        `True` when a full or partial close-think marker is present.
    """

    return contains_think_close(text=text) or (
        THINK_CLOSE_PARTIAL_SUFFIX_PATTERN.search(text) is not None
    )


def is_natural_finish_reason(
    *, finish_reason: str, stop_reason: int | str | None
) -> bool:
    """Return whether finish-reason represents natural model completion.

    Args:
        finish_reason: Parsed finish reason.
        stop_reason: Parsed stop reason.

    Returns:
        `True` for natural completion or EOS-like stop with no explicit reason.
    """

    if finish_reason == "length":
        return False
    if finish_reason != "stop":
        return True
    return stop_reason is None


def complete_trailing_partial_tag(*, text: str) -> tuple[str, str | None]:
    """Repair a trailing partial steer/exec tag using minimal suffix completion.

    Args:
        text: Assistant prefix text.

    Returns:
        Tuple `(updated_text, repaired_tag_or_none)`.
    """

    partial = trailing_partial_tag_suffix(text=text)
    if partial is None:
        return text, None
    partial_suffix, completed_tag = partial
    missing_suffix = completed_tag[len(partial_suffix) :]
    return text + missing_suffix, completed_tag


def trailing_partial_tag_suffix(*, text: str) -> tuple[str, str] | None:
    """Find the trailing repairable partial steer/exec tag suffix.

    Args:
        text: Assistant prefix text.

    Returns:
        Tuple `(partial_suffix, completed_tag)` when repairable, else `None`.
    """

    prefix_map = unique_partial_tag_prefix_map()
    max_tag_len = max(len(tag) for tag in REPAIR_TAGS)
    max_scan = min(max_tag_len - 1, len(text))
    lowered = text.lower()
    for suffix_length in range(max_scan, 0, -1):
        suffix = lowered[-suffix_length:]
        completed_tag = prefix_map.get(suffix)
        if completed_tag is None:
            continue
        return suffix, completed_tag
    return None


def unique_partial_tag_prefix_map() -> dict[str, str]:
    """Build unique lowercase partial-tag prefix map for repair.

    Args:
        None.

    Returns:
        Mapping from partial prefix to its completed tag.
    """

    prefix_map: dict[str, str] = {}
    ambiguous_prefixes: set[str] = set()
    for tag in REPAIR_TAGS:
        lowered_tag = tag.lower()
        for prefix_length in range(2, len(lowered_tag)):
            prefix = lowered_tag[:prefix_length]
            existing = prefix_map.get(prefix)
            if existing is None:
                prefix_map[prefix] = lowered_tag
                continue
            if existing != lowered_tag:
                ambiguous_prefixes.add(prefix)
    for prefix in ambiguous_prefixes:
        prefix_map.pop(prefix, None)
    prefix_map["<"] = "<steer>"
    return prefix_map


def trim_to_first_steer_close(*, text: str) -> tuple[str, bool]:
    """Trim text at the first `</steer>` close marker.

    Args:
        text: Candidate text.

    Returns:
        Tuple `(trimmed_text, closed_with_tag)`.
    """

    lowered = text.lower()
    close_start = lowered.find(STEER_CLOSE_TAG)
    if close_start < 0:
        return text, False
    close_end = close_start + len(STEER_CLOSE_TAG)
    return text[:close_end], True
