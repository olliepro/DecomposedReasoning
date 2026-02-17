"""Tag boundary detection utilities for steer/exec style traces."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ScanEvent:
    """One detected structural event in generated text.

    Args:
        event_type: Event label (`branch` or `think_end`).
        start_index: Start character offset of event.
        end_index: End character offset of event.

    Returns:
        Dataclass containing one scan event.
    """

    event_type: str
    start_index: int
    end_index: int


def find_branch_event(
    *,
    text: str,
    start_index: int,
    boundary_pattern: str,
) -> ScanEvent | None:
    """Find next branch boundary matching configured steer-open pattern.

    Args:
        text: Full generated text.
        start_index: Search offset.
        boundary_pattern: Regex used for branch boundary.

    Returns:
        Branch scan event or `None`.
    """
    match = re.search(
        pattern=boundary_pattern, string=text[start_index:], flags=re.IGNORECASE
    )
    if match is None:
        return None
    event_start = start_index + match.start()
    event_end = start_index + match.end()
    return ScanEvent(event_type="branch", start_index=event_start, end_index=event_end)


def find_think_end_event(*, text: str, start_index: int) -> ScanEvent | None:
    """Find next `</think>` close tag event.

    Args:
        text: Full generated text.
        start_index: Search offset.

    Returns:
        Think-end scan event or `None`.
    """
    match = re.search(
        pattern=r"</think>", string=text[start_index:], flags=re.IGNORECASE
    )
    if match is None:
        return None
    event_start = start_index + match.start()
    event_end = start_index + match.end()
    return ScanEvent(
        event_type="think_end", start_index=event_start, end_index=event_end
    )


def choose_next_event(
    *, branch_event: ScanEvent | None, think_end_event: ScanEvent | None
) -> ScanEvent | None:
    """Select earliest event between branch and think-end markers.

    Args:
        branch_event: Optional branch event candidate.
        think_end_event: Optional think-end event candidate.

    Returns:
        Earliest event by start offset, or `None`.
    """
    if branch_event is None:
        return think_end_event
    if think_end_event is None:
        return branch_event
    if branch_event.start_index <= think_end_event.start_index:
        return branch_event
    return think_end_event


def first_steer_close_index(*, text: str) -> int | None:
    """Find first `</steer>` close tag offset in text.

    Args:
        text: Candidate steer text.

    Returns:
        Start index of first close tag or `None`.
    """
    match = re.search(pattern=r"</steer>", string=text, flags=re.IGNORECASE)
    if match is None:
        return None
    return match.start()


def trim_after_first_stop(*, text: str, stop_markers: tuple[str, ...]) -> str:
    """Trim text at earliest occurrence of any stop marker.

    Args:
        text: Generated text.
        stop_markers: Stop marker list.

    Returns:
        Trimmed text up to earliest marker.
    """
    indexes = [text.find(marker) for marker in stop_markers if marker in text]
    if not indexes:
        return text
    return text[: min(indexes)]
