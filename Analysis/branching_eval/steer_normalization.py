"""Legacy-parity text normalization helpers for `<steer>/<exec>` flows."""

from __future__ import annotations

import re
from branching_eval.legacy_steer_rollout import complete_trailing_partial_tag

EXEC_OPEN_PATTERN = re.compile(r"<exec\b[^>]*>", flags=re.IGNORECASE)
EXEC_CLOSE_PATTERN = re.compile(r"</exec>", flags=re.IGNORECASE)
EXEC_TO_STEER_PATTERN = re.compile(r"</exec>(\s*)<steer\b", flags=re.IGNORECASE)
STEER_TO_EXEC_BOUNDARY_PATTERN = re.compile(
    r"</steer>\s*<exec>\s*",
    flags=re.IGNORECASE,
)
EXEC_TO_STEER_BOUNDARY_PATTERN = re.compile(
    r"</exec>\s*<steer>",
    flags=re.IGNORECASE,
)
STEER_ENTRY_BOUNDARY_PATTERN = re.compile(
    r"(?:<think>|</exec>)\s*<steer>$",
    flags=re.IGNORECASE,
)
STEER_CLOSE_TAG = "</steer>"


def normalize_steer_boundary_text(*, text: str) -> str:
    """Normalize text to canonical steer-open boundary form.

    Args:
        text: Assistant prefix ending at or near a steer boundary.

    Returns:
        Canonicalized prefix ending in `<steer>` with valid entry boundary.

    Example:
        >>> normalize_steer_boundary_text(text="x</exec>\\n\\n<steer>")
        'x</exec>\\n\\n<steer>'
    """

    normalized_text = text
    if is_inside_open_exec(text=normalized_text):
        normalized_text = _ensure_exec_closed_before_steer_boundary(
            text=normalized_text
        )
        return _ensure_valid_steer_entry_boundary(text=normalized_text)
    if normalized_text.endswith("<steer"):
        normalized_text += ">"
    elif not normalized_text.endswith("<steer>"):
        normalized_text += "<steer>"
    return _ensure_valid_steer_entry_boundary(text=normalized_text)


def is_inside_open_exec(*, text: str) -> bool:
    """Return whether text is currently inside an unclosed `<exec>` block.

    Args:
        text: Assistant prefix text.

    Returns:
        `True` when open `<exec>` tags outnumber close tags.
    """

    open_count = len(EXEC_OPEN_PATTERN.findall(text))
    close_count = len(EXEC_CLOSE_PATTERN.findall(text))
    return open_count > close_count


def inferred_exec_to_steer_separator(*, text: str) -> str:
    """Infer preferred separator between `</exec>` and `<steer>`.

    Args:
        text: Assistant prefix text.

    Returns:
        Observed separator or `\\n\\n` when no prior pattern exists.
    """

    matches = list(EXEC_TO_STEER_PATTERN.finditer(text))
    if not matches:
        return "\n\n"
    separator = matches[-1].group(1)
    return separator if separator is not None else "\n\n"


def forced_boundary_suffix(*, text: str) -> str:
    """Build forced suffix that moves text to a steer boundary.

    Args:
        text: Assistant prefix text.

    Returns:
        Suffix text to append.
    """

    if text.endswith("<steer>"):
        return ""
    if is_inside_open_exec(text=text):
        separator = inferred_exec_to_steer_separator(text=text)
        if text.endswith("</exec>"):
            return f"{separator}<steer>"
        newline_before_close = "" if text.endswith("\n") else "\n"
        return f"{newline_before_close}</exec>{separator}<steer>"
    return "<steer>"


def has_valid_steer_entry_boundary(*, text: str) -> bool:
    """Return whether text ends with a valid steer-entry boundary.

    Args:
        text: Assistant prefix text.

    Returns:
        `True` when trailing boundary matches `(<think>|</exec>)<steer>`.
    """

    return STEER_ENTRY_BOUNDARY_PATTERN.search(text) is not None


def _ensure_exec_closed_before_steer_boundary(*, text: str) -> str:
    normalized_text = text
    if normalized_text.endswith("<steer"):
        normalized_text += ">"
    if normalized_text.endswith("<steer>"):
        separator = inferred_exec_to_steer_separator(text=normalized_text)
        return normalized_text + f"</steer></exec>{separator}<steer>"
    return normalized_text + forced_boundary_suffix(text=normalized_text)


def _ensure_valid_steer_entry_boundary(*, text: str) -> str:
    assert text.endswith("<steer>"), "expected trailing <steer> boundary"
    if has_valid_steer_entry_boundary(text=text):
        return text
    separator = inferred_exec_to_steer_separator(text=text)
    return text + f"</steer><exec></exec>{separator}<steer>"


def selected_candidate_close_completion_suffix(*, text: str) -> tuple[str, str]:
    """Return minimal close-tag completion suffix for selected candidate text.

    Args:
        text: Selected candidate text before normalization.

    Returns:
        Tuple `(close_suffix, normalization_mode)`.
    """

    lowered = text.lower()
    lowered_close_tag = STEER_CLOSE_TAG
    if lowered.endswith(lowered_close_tag) or lowered.endswith(f"{lowered_close_tag}\n"):
        return "", "already_closed"
    for prefix_length in range(len(lowered_close_tag) - 1, 0, -1):
        if lowered.endswith(lowered_close_tag[:prefix_length]):
            return lowered_close_tag[prefix_length:], "partial_completed"
    return STEER_CLOSE_TAG, "full_close_appended"


def selected_candidate_normalization_suffix(*, text: str) -> tuple[str, str]:
    """Build minimal injected suffix so candidate ends with `</steer>\\n<exec>\\n`.

    Args:
        text: Selected candidate text before normalization.

    Returns:
        Tuple `(injected_suffix, normalization_mode)`.
    """

    close_suffix, mode = selected_candidate_close_completion_suffix(text=text)
    normalized = text + close_suffix
    newline_suffix = "" if normalized.endswith("\n") else "\n"
    normalized_with_newline = normalized + newline_suffix
    if normalized_with_newline.endswith("<exec>\n"):
        exec_suffix = ""
    elif normalized_with_newline.endswith("<exec>"):
        exec_suffix = "\n"
    else:
        exec_suffix = "<exec>\n"
    return close_suffix + newline_suffix + exec_suffix, mode


def normalize_steer_exec_chunk_text(*, text: str) -> str:
    """Canonicalize steer/exec boundaries inside one decode chunk string.

    Args:
        text: Raw decode chunk text.

    Returns:
        Chunk text with repaired trailing partial tag and canonicalized
        `</steer> -> <exec>` / `</exec> -> <steer>` separators.

    Example:
        >>> normalize_steer_exec_chunk_text(text=\"x</steer>\\n\\n<exec>\\n\")
        'x</steer>\\n<exec>\\n'
    """

    repaired_text, _ = complete_trailing_partial_tag(text=text)
    normalized_text = STEER_TO_EXEC_BOUNDARY_PATTERN.sub(
        "</steer>\n<exec>\n",
        repaired_text,
    )
    return EXEC_TO_STEER_BOUNDARY_PATTERN.sub(
        "</exec>\n\n<steer>",
        normalized_text,
    )
