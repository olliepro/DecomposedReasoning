"""Legacy-parity text normalization helpers for `<steer>/<exec>` flows."""

from __future__ import annotations

import re
from branching_eval.legacy_steer_rollout import (
    complete_trailing_partial_tag,
    contains_think_close,
)

EXEC_OPEN_PATTERN = re.compile(r"<exec\b[^>]*>", flags=re.IGNORECASE)
EXEC_CLOSE_PATTERN = re.compile(r"</exec>", flags=re.IGNORECASE)
EXEC_CLOSE_TAG = "</exec>"
EXEC_CLOSE_SUFFIX_PATTERN = re.compile(r"</exec>(?P<suffix>\n{0,2})$", flags=re.I)
EXEC_TO_STEER_SEPARATOR = "\n"
EXEC_TO_STEER_NEWLINE_COUNT = len(EXEC_TO_STEER_SEPARATOR)
STEER_TO_EXEC_BOUNDARY_PATTERN = re.compile(
    r"</steer>\s*<exec>\s*",
    flags=re.IGNORECASE,
)
EXEC_TO_STEER_BOUNDARY_PATTERN = re.compile(
    r"</exec>\s*<steer>",
    flags=re.IGNORECASE,
)
INITIAL_THINK_DECISION_PATTERN = re.compile(
    r"^\s*<think\b[^>]*>\s*$",
    flags=re.IGNORECASE,
)
STEER_ENTRY_BOUNDARY_PATTERN = re.compile(
    r"(?:<think>|</exec>)\s*<steer>$",
    flags=re.IGNORECASE,
)
STEER_CLOSE_TAG = "</steer>"
STEER_OPEN_TAG = "<steer>"
STEER_OPEN_PREFIXES = tuple(
    STEER_OPEN_TAG[:prefix_length]
    for prefix_length in range(2, len(STEER_OPEN_TAG) + 1)
)
STEER_CANDIDATE_STOP = ("</steer",)
TERMINAL_STEER_CANDIDATE_STOP = ("</steer", "</think>")


def normalize_steer_boundary_text(*, text: str) -> str:
    """Normalize text to canonical post-exec or steer-open boundary form.

    Args:
        text: Assistant prefix ending at or near a steer boundary.

    Returns:
        Canonicalized prefix using append-only suffixes. Ordinary exec prefixes
        end at `</exec>\\n`; prefixes that already contain a 2+ char
        `<steer>` suffix are completed to a valid `<steer>` entry boundary.

    Example:
        >>> normalize_steer_boundary_text(text="<exec>x")
        '<exec>x\\n</exec>\\n'
    """

    normalized_text = text
    if has_trailing_steer_open_prefix(text=normalized_text):
        normalized_text = _complete_trailing_steer_open_prefix(text=normalized_text)
        if is_inside_open_exec(text=normalized_text):
            normalized_text = _ensure_exec_closed_before_steer_boundary(
                text=normalized_text
            )
        return _ensure_valid_steer_entry_boundary(text=normalized_text)
    if is_inside_open_exec(text=normalized_text):
        return normalized_text + forced_boundary_suffix(text=normalized_text)
    if ends_at_exec_choice_boundary(text=normalized_text):
        return normalized_text + exec_choice_boundary_suffix(text=normalized_text)
    if normalized_text.endswith("<steer>"):
        return _ensure_valid_steer_entry_boundary(text=normalized_text)
    normalized_text += "<steer>"
    return _ensure_valid_steer_entry_boundary(text=normalized_text)


def has_trailing_steer_open_prefix(*, text: str) -> bool:
    """Return whether text ends with a 2+ char prefix of `<steer>`.

    Args:
        text: Assistant prefix text.

    Returns:
        `True` when the suffix is `<s`, `<st`, ..., or `<steer>`.

    Example:
        >>> has_trailing_steer_open_prefix(text="x<s")
        True
        >>> has_trailing_steer_open_prefix(text="x<")
        False
    """

    lowered = text.lower()
    return any(lowered.endswith(prefix) for prefix in STEER_OPEN_PREFIXES)


def ends_at_exec_choice_boundary(*, text: str) -> bool:
    """Return whether text can be completed to `</exec>\\n` by appending.

    Args:
        text: Assistant prefix text.

    Returns:
        `True` when text ends with `</exec>` plus zero to two newlines.
    """

    return EXEC_CLOSE_SUFFIX_PATTERN.search(text) is not None


def is_initial_steer_decision_boundary(*, text: str) -> bool:
    """Return whether text is the initial think-open steer decision point.

    Args:
        text: Assistant prefix text.

    Returns:
        `True` only for an otherwise-empty assistant prefix ending at `<think>`.
    """

    return INITIAL_THINK_DECISION_PATTERN.fullmatch(text) is not None


def is_steer_decision_boundary(*, text: str) -> bool:
    """Return whether steer mode should sample a steer candidate next.

    Args:
        text: Assistant prefix text.

    Returns:
        `True` at the initial `<think>` choice point or after `</exec>`.
    """

    return is_initial_steer_decision_boundary(
        text=text
    ) or ends_at_exec_choice_boundary(text=text)


def exec_choice_boundary_suffix(*, text: str) -> str:
    """Return suffix needed to make trailing `</exec>` boundary `\\n`.

    Args:
        text: Assistant prefix ending in `</exec>` plus up to two newlines.

    Returns:
        Empty string or `\\n`.
    """

    match = EXEC_CLOSE_SUFFIX_PATTERN.search(text)
    assert match is not None, "expected text ending at an exec choice boundary"
    suffix = match.group("suffix")
    return "\n" * max(0, EXEC_TO_STEER_NEWLINE_COUNT - len(suffix))


def explicit_exec_stop_completion_suffix(*, text: str) -> str:
    """Return the minimal suffix needed after an excluded `</exec` stop.

    Args:
        text: Assistant prefix after appending model output.

    Returns:
        Minimal completion for the terminal exec close marker.
    """

    lowered = text.lower()
    if lowered.endswith(EXEC_CLOSE_TAG):
        return ""
    for prefix_length in range(len(EXEC_CLOSE_TAG) - 1, 0, -1):
        if lowered.endswith(EXEC_CLOSE_TAG[:prefix_length]):
            return EXEC_CLOSE_TAG[prefix_length:]
    return EXEC_CLOSE_TAG


def steer_candidate_stop_markers(*, text: str) -> tuple[str, ...]:
    """Return candidate-generation stops for one normalized steer prefix.

    Args:
        text: Canonicalized assistant prefix used for steer candidate generation.

    Returns:
        `</think>` is included only when the model is allowed to choose whether
        to open another steer block after an exec boundary.
    """

    if text.lower().endswith(STEER_OPEN_TAG):
        return STEER_CANDIDATE_STOP
    return TERMINAL_STEER_CANDIDATE_STOP


def _complete_trailing_steer_open_prefix(*, text: str) -> str:
    lowered = text.lower()
    for prefix in sorted(STEER_OPEN_PREFIXES, key=len, reverse=True):
        if lowered.endswith(prefix):
            return text + STEER_OPEN_TAG[len(prefix) :]
    raise AssertionError("expected trailing steer-open prefix")


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
    """Return the canonical separator between `</exec>` and `<steer>`.

    Args:
        text: Assistant prefix text.

    Returns:
        Single newline separator used when inserting steer candidates.
    """

    _ = text
    return EXEC_TO_STEER_SEPARATOR


def forced_boundary_suffix(*, text: str) -> str:
    """Build forced suffix that moves text to the next choice boundary.

    Args:
        text: Assistant prefix text.

    Returns:
        Append-only suffix. Open exec text is closed to `</exec>\\n`
        unless a 2+ char `<steer>` prefix has already been emitted, in which
        case the canonical `</exec>\\n<steer>` boundary is preserved.
    """

    if has_trailing_steer_open_prefix(text=text):
        normalized_text = _complete_trailing_steer_open_prefix(text=text)
        if is_inside_open_exec(text=normalized_text):
            boundary_text = _ensure_exec_closed_before_steer_boundary(
                text=normalized_text
            )
        else:
            boundary_text = _ensure_valid_steer_entry_boundary(text=normalized_text)
        return boundary_text[len(text) :]
    if text.endswith("<steer>"):
        return ""
    if is_inside_open_exec(text=text):
        newline_before_close = "" if text.endswith("\n") else "\n"
        return f"{newline_before_close}</exec>{EXEC_TO_STEER_SEPARATOR}"
    if ends_at_exec_choice_boundary(text=text):
        return exec_choice_boundary_suffix(text=text)
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
    if lowered.endswith(lowered_close_tag) or lowered.endswith(
        f"{lowered_close_tag}\n"
    ):
        return "", "already_closed"
    for prefix_length in range(len(lowered_close_tag) - 1, 0, -1):
        if lowered.endswith(lowered_close_tag[:prefix_length]):
            return lowered_close_tag[prefix_length:], "partial_completed"
    return STEER_CLOSE_TAG, "full_close_appended"


def selected_candidate_normalization_suffix(*, text: str) -> tuple[str, str]:
    """Build minimal injected suffix for selected steer candidate text.

    Args:
        text: Selected candidate text before normalization.

    Returns:
        Tuple `(injected_suffix, normalization_mode)`. Terminal `</think>`
        candidates are returned without steer-section normalization.
    """

    if contains_think_close(text=text):
        return "", "think_closed"
    close_suffix, mode = selected_candidate_close_completion_suffix(text=text)
    normalized = text + close_suffix
    newline_suffix = "" if normalized.endswith("\n") else "\n"
    normalized_with_newline = normalized + newline_suffix
    if normalized_with_newline.endswith("<exec>\n"):
        exec_suffix = ""
    elif normalized_with_newline.endswith("<exec>"):
        exec_suffix = ""
    else:
        exec_suffix = "<exec>"
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
        'x</steer>\\n<exec>'
    """

    repaired_text, _ = complete_trailing_partial_tag(text=text)
    normalized_text = STEER_TO_EXEC_BOUNDARY_PATTERN.sub(
        "</steer>\n<exec>",
        repaired_text,
    )
    return EXEC_TO_STEER_BOUNDARY_PATTERN.sub(
        f"</exec>{EXEC_TO_STEER_SEPARATOR}<steer>",
        normalized_text,
    )
