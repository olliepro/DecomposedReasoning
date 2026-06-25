"""Decode/runtime helper functions shared by branching executor."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import random
import re
from typing import Callable

from branching_eval.legacy_steer_rollout import STEER_CLOSE_TAG
from branching_eval.runtime_types import DecodeOutcome
from branching_eval.selector_types import SelectionOutcome, SelectorMode
from branching_eval.tree_types import CandidatePoolRecord, CandidateRecord, TokenTrace
from token_metrics import probability_from_logprob
from vllm_client import GenerationChoice, ParsedToken

STEER_BOUNDARY_PATTERN = re.compile(r"<steer>$", flags=re.IGNORECASE)
STEER_PARTIAL_PATTERN = re.compile(r"<steer$", flags=re.IGNORECASE)
STEER_BLOCK_PATTERN = re.compile(
    r"<steer\b[^>]*>(.*?)</steer>", flags=re.IGNORECASE | re.DOTALL
)
STEER_OPEN_TAG_PATTERN = re.compile(r"<steer\b[^>]*>", flags=re.IGNORECASE)
EXEC_BLOCK_PATTERN = re.compile(
    r"<exec\b[^>]*>(.*?)</exec>", flags=re.IGNORECASE | re.DOTALL
)
EXEC_OPEN_TAG_PATTERN = re.compile(r"<exec\b[^>]*>", flags=re.IGNORECASE)
THINK_CLOSE_TAG = "</think>"
ROLLOUT_STEER_STOP = ("</exec", THINK_CLOSE_TAG)
ROLLOUT_STEER_STOP_FULL = ("</exec>", THINK_CLOSE_TAG)
DEFAULT_EXEC_REPEAT_LOOKBACK_WINDOW = 3
UNAVAILABLE_LOGPROB = -1e9
CONTROL_EDGE_TAGS = (
    "<steer>",
    "</steer>",
    "<exec>",
    "</exec>",
    "<think>",
    "</think>",
)
CONTROL_EDGE_FRAGMENTS = tuple(
    sorted(
        {
            fragment
            for tag in CONTROL_EDGE_TAGS
            for fragment in (
                *(tag[:index] for index in range(2, len(tag) + 1)),
                *(tag[index:] for index in range(0, len(tag) - 1)),
            )
            if ("<" in fragment and len(fragment) >= 2)
            or (">" in fragment and len(fragment) >= 3)
        },
        key=len,
        reverse=True,
    )
)


@dataclass(frozen=True)
class ExecRepetitionState:
    """Streaming state for repeated execution-block detection.

    Args:
        in_exec_block: Whether parser is currently inside an open `<exec>` block.
        pending_exec_text: Text buffered for the currently open exec block.
        pending_tag_text: Partial tag suffix carried across chunk boundaries.
        previous_exec_block: Most recent normalized completed exec block.
        recent_exec_blocks: Recent normalized blocks used for similarity lookback.
        repeated_exec_blocks: Count of consecutive near-duplicate exec blocks.

    Returns:
        Dataclass used for incremental exec-loop detection.

    Example:
        >>> state = initialize_exec_repetition_state(text="<exec>a</exec>")
        >>> state.repeated_exec_blocks
        1
    """

    in_exec_block: bool
    pending_exec_text: str
    pending_tag_text: str
    previous_exec_block: str | None
    recent_exec_blocks: tuple[str, ...]
    repeated_exec_blocks: int


def normalize_exec_block_text(*, text: str) -> str:
    """Normalize one execution block for fuzzy similarity checks.

    Args:
        text: Raw text inside one `<exec>...</exec>` block.

    Returns:
        Lowercased whitespace-normalized text.
    """

    return " ".join(str(text).lower().split()).strip()


def normalize_steer_block_text(*, text: str) -> str:
    """Normalize steer block text after dropping edge control tags/fragments."""

    cleaned_text = strip_control_edge_fragments(text=str(text))
    return normalize_exec_block_text(text=cleaned_text)


def strip_control_edge_fragments(*, text: str) -> str:
    """Strip complete or partial control tags from string edges only."""

    cleaned_text = str(text).strip()
    while True:
        updated_text = _strip_one_control_edge_fragment(text=cleaned_text).strip()
        if updated_text == cleaned_text:
            return updated_text
        cleaned_text = updated_text


def _strip_one_control_edge_fragment(*, text: str) -> str:
    lowered_text = text.lower()
    for fragment in CONTROL_EDGE_FRAGMENTS:
        lowered_fragment = fragment.lower()
        if lowered_text.startswith(lowered_fragment):
            return text[len(fragment) :]
        if lowered_text.endswith(lowered_fragment):
            return text[: -len(fragment)]
    return text


def exec_block_similarity_ratio(*, left_text: str, right_text: str) -> float:
    """Return fuzzy similarity ratio for two normalized exec blocks.

    Args:
        left_text: Left normalized block.
        right_text: Right normalized block.

    Returns:
        Similarity ratio in `[0.0, 1.0]`.
    """

    if not left_text or not right_text:
        return 0.0
    return float(SequenceMatcher(a=left_text, b=right_text, autojunk=False).ratio())


def initialize_exec_repetition_state(
    *,
    text: str,
    similarity_threshold: float = 0.85,
    similarity_lookback_window: int = DEFAULT_EXEC_REPEAT_LOOKBACK_WINDOW,
) -> ExecRepetitionState:
    """Seed incremental repetition state from existing assistant prefix text.

    Args:
        text: Existing assistant prefix text.
        similarity_threshold: Near-duplicate threshold for fuzzy matching.
        similarity_lookback_window: Number of recent blocks to compare against.

    Returns:
        Initial repetition-tracker state for streaming updates.
    """

    return _initialize_tag_repetition_state(
        text=text,
        block_pattern=EXEC_BLOCK_PATTERN,
        open_tag_pattern=EXEC_OPEN_TAG_PATTERN,
        close_tag="</exec>",
        block_normalizer=normalize_exec_block_text,
        similarity_threshold=similarity_threshold,
        similarity_lookback_window=similarity_lookback_window,
    )


def update_exec_repetition_state(
    *,
    state: ExecRepetitionState,
    chunk_text: str,
    similarity_threshold: float = 0.85,
    similarity_lookback_window: int = DEFAULT_EXEC_REPEAT_LOOKBACK_WINDOW,
) -> tuple[ExecRepetitionState, float | None]:
    """Advance repetition state with one appended text chunk.

    Args:
        state: Existing repetition state.
        chunk_text: Newly appended assistant text chunk.
        similarity_threshold: Near-duplicate threshold for fuzzy matching.
        similarity_lookback_window: Number of recent blocks to compare against.

    Returns:
        Tuple of `(updated_state, last_similarity_ratio)` where
        `last_similarity_ratio` is set only when at least one completed block
        is compared against a previous block.
    """

    return _update_tag_repetition_state(
        state=state,
        chunk_text=chunk_text,
        open_tag_pattern=EXEC_OPEN_TAG_PATTERN,
        close_tag="</exec>",
        block_normalizer=normalize_exec_block_text,
        similarity_threshold=similarity_threshold,
        similarity_lookback_window=similarity_lookback_window,
    )


def initialize_steer_repetition_state(
    *,
    text: str,
    similarity_threshold: float = 0.85,
    similarity_lookback_window: int = DEFAULT_EXEC_REPEAT_LOOKBACK_WINDOW,
) -> ExecRepetitionState:
    """Seed incremental repetition state from existing steer blocks."""

    return _initialize_tag_repetition_state(
        text=text,
        block_pattern=STEER_BLOCK_PATTERN,
        open_tag_pattern=STEER_OPEN_TAG_PATTERN,
        close_tag=STEER_CLOSE_TAG,
        block_normalizer=normalize_steer_block_text,
        similarity_threshold=similarity_threshold,
        similarity_lookback_window=similarity_lookback_window,
    )


def update_steer_repetition_state(
    *,
    state: ExecRepetitionState,
    chunk_text: str,
    similarity_threshold: float = 0.85,
    similarity_lookback_window: int = DEFAULT_EXEC_REPEAT_LOOKBACK_WINDOW,
) -> tuple[ExecRepetitionState, float | None]:
    """Advance repetition state with one appended chunk for steer blocks."""

    return _update_tag_repetition_state(
        state=state,
        chunk_text=chunk_text,
        open_tag_pattern=STEER_OPEN_TAG_PATTERN,
        close_tag=STEER_CLOSE_TAG,
        block_normalizer=normalize_steer_block_text,
        similarity_threshold=similarity_threshold,
        similarity_lookback_window=similarity_lookback_window,
    )


def _initialize_tag_repetition_state(
    *,
    text: str,
    block_pattern: re.Pattern[str],
    open_tag_pattern: re.Pattern[str],
    close_tag: str,
    block_normalizer: Callable[..., str],
    similarity_threshold: float,
    similarity_lookback_window: int,
) -> ExecRepetitionState:
    """Build initial repetition state for one tagged block family."""

    assert similarity_lookback_window >= 1, "similarity_lookback_window must be >= 1"
    normalized_blocks = tuple(
        normalized
        for normalized in (
            block_normalizer(text=match) for match in block_pattern.findall(text)
        )
        if normalized
    )
    trailing_repeat_count = _trailing_repeated_block_count(
        normalized_blocks=normalized_blocks,
        similarity_threshold=similarity_threshold,
        similarity_lookback_window=similarity_lookback_window,
    )
    recent_exec_blocks = tuple(normalized_blocks[-similarity_lookback_window:])
    in_exec_block, pending_exec_text, _, pending_tag_text = _consume_tag_chunk(
        in_exec_block=False,
        pending_exec_text="",
        pending_tag_text="",
        chunk_text=text,
        open_tag_pattern=open_tag_pattern,
        close_tag=close_tag,
        block_normalizer=block_normalizer,
    )
    return ExecRepetitionState(
        in_exec_block=in_exec_block,
        pending_exec_text=pending_exec_text,
        pending_tag_text=pending_tag_text,
        previous_exec_block=(recent_exec_blocks[-1] if recent_exec_blocks else None),
        recent_exec_blocks=recent_exec_blocks,
        repeated_exec_blocks=trailing_repeat_count,
    )


def _update_tag_repetition_state(
    *,
    state: ExecRepetitionState,
    chunk_text: str,
    open_tag_pattern: re.Pattern[str],
    close_tag: str,
    block_normalizer: Callable[..., str],
    similarity_threshold: float,
    similarity_lookback_window: int,
) -> tuple[ExecRepetitionState, float | None]:
    """Advance one tagged repetition state with one appended chunk."""

    assert similarity_lookback_window >= 1, "similarity_lookback_window must be >= 1"
    (
        in_exec_block,
        pending_exec_text,
        completed_exec_blocks,
        pending_tag_text,
    ) = _consume_tag_chunk(
        in_exec_block=state.in_exec_block,
        pending_exec_text=state.pending_exec_text,
        pending_tag_text=state.pending_tag_text,
        chunk_text=chunk_text,
        open_tag_pattern=open_tag_pattern,
        close_tag=close_tag,
        block_normalizer=block_normalizer,
    )
    previous_exec_block = state.previous_exec_block
    recent_exec_blocks = tuple(state.recent_exec_blocks[-similarity_lookback_window:])
    repeated_exec_blocks = state.repeated_exec_blocks
    last_similarity_ratio: float | None = None
    for exec_block in completed_exec_blocks:
        repeated_exec_blocks, similarity_ratio = _next_repeated_exec_block_count(
            recent_exec_blocks=recent_exec_blocks,
            current_exec_block=exec_block,
            repeated_exec_blocks=repeated_exec_blocks,
            similarity_threshold=similarity_threshold,
        )
        last_similarity_ratio = (
            similarity_ratio if similarity_ratio is not None else last_similarity_ratio
        )
        recent_exec_blocks = _append_exec_block_history(
            recent_exec_blocks=recent_exec_blocks,
            exec_block=exec_block,
            similarity_lookback_window=similarity_lookback_window,
        )
        previous_exec_block = exec_block
    return (
        ExecRepetitionState(
            in_exec_block=in_exec_block,
            pending_exec_text=pending_exec_text,
            pending_tag_text=pending_tag_text,
            previous_exec_block=previous_exec_block,
            recent_exec_blocks=recent_exec_blocks,
            repeated_exec_blocks=repeated_exec_blocks,
        ),
        last_similarity_ratio,
    )


def _next_repeated_exec_block_count(
    *,
    recent_exec_blocks: tuple[str, ...],
    current_exec_block: str,
    repeated_exec_blocks: int,
    similarity_threshold: float,
) -> tuple[int, float | None]:
    """Return next repeated-block count from lookback comparisons."""

    if len(recent_exec_blocks) == 0:
        return (1, None)
    similarity_ratio = max(
        exec_block_similarity_ratio(
            left_text=prior_exec_block,
            right_text=current_exec_block,
        )
        for prior_exec_block in recent_exec_blocks
    )
    next_repeated_count = (
        repeated_exec_blocks + 1 if similarity_ratio >= similarity_threshold else 1
    )
    return (next_repeated_count, similarity_ratio)


def _append_exec_block_history(
    *,
    recent_exec_blocks: tuple[str, ...],
    exec_block: str,
    similarity_lookback_window: int,
) -> tuple[str, ...]:
    """Append one block to fixed-size similarity lookback history."""

    updated_exec_blocks = (*recent_exec_blocks, exec_block)
    if len(updated_exec_blocks) <= similarity_lookback_window:
        return updated_exec_blocks
    return updated_exec_blocks[-similarity_lookback_window:]


def _consume_exec_chunk(
    *,
    in_exec_block: bool,
    pending_exec_text: str,
    chunk_text: str,
) -> tuple[bool, str, tuple[str, ...]]:
    """Parse one appended chunk and return completed normalized exec blocks."""

    return _consume_tag_chunk(
        in_exec_block=in_exec_block,
        pending_exec_text=pending_exec_text,
        pending_tag_text="",
        chunk_text=chunk_text,
        open_tag_pattern=EXEC_OPEN_TAG_PATTERN,
        close_tag="</exec>",
        block_normalizer=normalize_exec_block_text,
    )[:3]


def _consume_tag_chunk(
    *,
    in_exec_block: bool,
    pending_exec_text: str,
    pending_tag_text: str,
    chunk_text: str,
    open_tag_pattern: re.Pattern[str],
    close_tag: str,
    block_normalizer: Callable[..., str],
) -> tuple[bool, str, tuple[str, ...], str]:
    """Parse one appended chunk and return completed normalized tag blocks."""

    normalized_completed_exec_blocks: list[str] = []
    cursor = 0
    if pending_tag_text:
        chunk_text = pending_tag_text + chunk_text
        pending_tag_text = ""
    lowered_chunk_text = chunk_text.lower()
    lowered_close_tag = close_tag.lower()
    while cursor < len(chunk_text):
        if in_exec_block:
            close_index = lowered_chunk_text.find(lowered_close_tag, cursor)
            if close_index < 0:
                safe_text, pending_tag_text = _split_trailing_partial_tag(
                    text=chunk_text[cursor:],
                    tag=close_tag,
                )
                pending_exec_text += safe_text
                break
            pending_exec_text += chunk_text[cursor:close_index]
            normalized_block = block_normalizer(text=pending_exec_text)
            if normalized_block:
                normalized_completed_exec_blocks.append(normalized_block)
            pending_exec_text = ""
            in_exec_block = False
            cursor = close_index + len(close_tag)
            continue
        open_match = open_tag_pattern.search(chunk_text, pos=cursor)
        if open_match is None:
            pending_tag_text = _trailing_partial_open_tag(
                text=chunk_text[cursor:],
                open_tag_pattern=open_tag_pattern,
            )
            break
        in_exec_block = True
        cursor = open_match.end()
    return (
        in_exec_block,
        pending_exec_text,
        tuple(normalized_completed_exec_blocks),
        pending_tag_text,
    )


def _split_trailing_partial_tag(*, text: str, tag: str) -> tuple[str, str]:
    """Split a trailing partial tag prefix from ordinary block text."""

    lowered_text = text.lower()
    lowered_tag = tag.lower()
    max_scan = min(len(lowered_tag) - 1, len(lowered_text))
    for suffix_length in range(max_scan, 0, -1):
        suffix = lowered_text[-suffix_length:]
        if lowered_tag.startswith(suffix):
            return text[:-suffix_length], text[-suffix_length:]
    return text, ""


def _trailing_partial_open_tag(*, text: str, open_tag_pattern: re.Pattern[str]) -> str:
    """Return a trailing partial open tag prefix for known block tags."""

    if open_tag_pattern is EXEC_OPEN_TAG_PATTERN:
        _, partial_tag = _split_trailing_partial_tag(text=text, tag="<exec>")
        return partial_tag
    if open_tag_pattern is STEER_OPEN_TAG_PATTERN:
        _, partial_tag = _split_trailing_partial_tag(text=text, tag="<steer>")
        return partial_tag
    return ""


def _infer_open_exec_suffix(*, text: str) -> tuple[bool, str]:
    """Infer whether text currently sits inside an unfinished `<exec>` block."""

    return _infer_open_tag_suffix(
        text=text,
        open_tag_pattern=EXEC_OPEN_TAG_PATTERN,
        close_tag="</exec>",
    )


def _infer_open_tag_suffix(
    *, text: str, open_tag_pattern: re.Pattern[str], close_tag: str
) -> tuple[bool, str]:
    """Infer whether text currently sits inside an unfinished block tag."""

    open_matches = list(open_tag_pattern.finditer(text))
    if not open_matches:
        return (False, "")
    last_open_match = open_matches[-1]
    last_close_index = text.lower().rfind(close_tag.lower())
    if last_close_index >= last_open_match.start():
        return (False, "")
    return (True, text[last_open_match.end() :])


def _trailing_repeated_block_count(
    *,
    normalized_blocks: tuple[str, ...],
    similarity_threshold: float,
    similarity_lookback_window: int,
) -> int:
    """Return trailing near-duplicate block run length from completed blocks."""

    if len(normalized_blocks) == 0:
        return 0
    recent_exec_blocks: tuple[str, ...] = ()
    repeated_exec_blocks = 0
    for exec_block in normalized_blocks:
        repeated_exec_blocks, _ = _next_repeated_exec_block_count(
            recent_exec_blocks=recent_exec_blocks,
            current_exec_block=exec_block,
            repeated_exec_blocks=repeated_exec_blocks,
            similarity_threshold=similarity_threshold,
        )
        recent_exec_blocks = _append_exec_block_history(
            recent_exec_blocks=recent_exec_blocks,
            exec_block=exec_block,
            similarity_lookback_window=similarity_lookback_window,
        )
    return repeated_exec_blocks


def consume_choice_tokens(
    *,
    choice: GenerationChoice,
    assistant_prefix: str,
    token_ids: list[int],
    token_traces: list[TokenTrace],
    generated_tokens: int,
    trigger_steer: bool,
    branch_prob: float,
    rng: random.Random,
) -> DecodeOutcome:
    """Consume one completion choice and detect probabilistic trigger events.

    Args:
        choice: One completion choice.
        assistant_prefix: Existing assistant prefix text.
        token_ids: Existing generated token ids.
        token_traces: Existing token traces.
        generated_tokens: Existing generated token count.
        trigger_steer: Whether steer-boundary trigger is enabled.
        branch_prob: Branching probability at eligible trigger points.
        rng: RNG used for branch-probability sampling.

    Returns:
        Decode outcome reflecting trigger/termination after processing the choice.
    """

    choice_token_ids = tuple(choice.token_ids) if choice.token_ids is not None else ()
    if not choice.tokens:
        return _consume_choice_without_parsed_tokens(
            choice_text=str(choice.text),
            choice_token_ids=choice_token_ids,
            assistant_prefix=assistant_prefix,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            trigger_steer=trigger_steer,
            branch_prob=branch_prob,
            rng=rng,
        )

    updated_prefix = assistant_prefix
    updated_token_ids = list(token_ids)
    updated_traces = list(token_traces)
    generated = generated_tokens
    for token_index, parsed_token in enumerate(choice.tokens):
        token_id = (
            choice_token_ids[token_index]
            if token_index < len(choice_token_ids)
            else None
        )
        token_trace = trace_from_parsed_token(
            parsed_token=parsed_token,
            token_index=len(updated_traces),
            token_id=token_id,
        )
        updated_prefix += parsed_token.token
        updated_traces.append(token_trace)
        updated_token_ids.append(token_id if token_id is not None else -1)
        generated += 1
        steer_trigger = trigger_steer and has_steer_boundary(text=updated_prefix)
        if not steer_trigger:
            continue
        if rng.random() > branch_prob:
            continue
        return DecodeOutcome(
            event_type="trigger",
            trigger_type="steer_boundary",
            assistant_prefix=updated_prefix,
            prompt_token_ids=None,
            token_ids=tuple(updated_token_ids),
            token_traces=tuple(updated_traces),
            generated_tokens=generated,
            stop_reason="",
        )
    return DecodeOutcome(
        event_type="terminated",
        trigger_type=None,
        assistant_prefix=updated_prefix,
        prompt_token_ids=None,
        token_ids=tuple(updated_token_ids),
        token_traces=tuple(updated_traces),
        generated_tokens=generated,
        stop_reason="chunk_complete",
    )


def choice_has_generated_content(*, choice: GenerationChoice) -> bool:
    """Return whether a vLLM choice contains generated text or token ids.

    Args:
        choice: Parsed vLLM completion choice.

    Returns:
        `True` when the choice has output content independent of logprob traces.

    Example:
        >>> choice_has_generated_content(
        ...     choice=GenerationChoice(
        ...         index=0,
        ...         text="hello",
        ...         finish_reason="stop",
        ...         stop_reason=None,
        ...         tokens=(),
        ...         prompt_token_ids=None,
        ...         token_ids=(1,),
        ...     )
        ... )
        True
    """

    return bool(str(choice.text)) or bool(choice.token_ids)


def _consume_choice_without_parsed_tokens(
    *,
    choice_text: str,
    choice_token_ids: tuple[int, ...],
    assistant_prefix: str,
    token_ids: list[int],
    token_traces: list[TokenTrace],
    generated_tokens: int,
    trigger_steer: bool,
    branch_prob: float,
    rng: random.Random,
) -> DecodeOutcome:
    """Consume text/token-id output when vLLM logprob token traces are disabled."""

    if not choice_text and not choice_token_ids:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=None,
            token_ids=tuple(token_ids),
            token_traces=tuple(token_traces),
            generated_tokens=generated_tokens,
            stop_reason="chunk_complete",
        )
    fallback_token_ids = choice_token_ids if choice_token_ids else (-1,)
    updated_prefix = assistant_prefix + choice_text
    updated_token_ids = list(token_ids)
    updated_traces = list(token_traces)
    for offset, token_id in enumerate(fallback_token_ids):
        updated_token_ids.append(token_id)
        updated_traces.append(
            TokenTrace(
                token_index=len(updated_traces),
                token_id=token_id if token_id >= 0 else None,
                token_text=choice_text if offset == 0 else "",
                logprob=UNAVAILABLE_LOGPROB,
                probability=0.0,
            )
        )
    generated = generated_tokens + len(fallback_token_ids)
    steer_trigger = trigger_steer and has_steer_boundary(text=updated_prefix)
    if steer_trigger and rng.random() <= branch_prob:
        return DecodeOutcome(
            event_type="trigger",
            trigger_type="steer_boundary",
            assistant_prefix=updated_prefix,
            prompt_token_ids=None,
            token_ids=tuple(updated_token_ids),
            token_traces=tuple(updated_traces),
            generated_tokens=generated,
            stop_reason="",
        )
    return DecodeOutcome(
        event_type="terminated",
        trigger_type=None,
        assistant_prefix=updated_prefix,
        prompt_token_ids=None,
        token_ids=tuple(updated_token_ids),
        token_traces=tuple(updated_traces),
        generated_tokens=generated,
        stop_reason="chunk_complete",
    )


def trace_from_parsed_token(
    *, parsed_token: ParsedToken, token_index: int, token_id: int | None
) -> TokenTrace:
    """Build one token trace from parsed logprobs.

    Args:
        parsed_token: Parsed vLLM token payload.
        token_index: Global token index in sequence.
        token_id: Token id when available.

    Returns:
        Token trace row with probability estimate.
    """

    return TokenTrace(
        token_index=token_index,
        token_id=token_id,
        token_text=parsed_token.token,
        logprob=float(parsed_token.logprob),
        probability=float(probability_from_logprob(logprob=parsed_token.logprob)),
    )


def token_traces_from_choice(*, choice: GenerationChoice) -> tuple[TokenTrace, ...]:
    """Return token traces for all tokens in one completion choice.

    Args:
        choice: Completion choice.

    Returns:
        Token trace tuple for the choice.
    """

    token_ids = tuple(choice.token_ids) if choice.token_ids is not None else ()
    traces: list[TokenTrace] = []
    for token_index, parsed_token in enumerate(choice.tokens):
        token_id = token_ids[token_index] if token_index < len(token_ids) else None
        traces.append(
            trace_from_parsed_token(
                parsed_token=parsed_token,
                token_index=token_index,
                token_id=token_id,
            )
        )
    return tuple(traces)


def candidate_from_choice(
    *,
    candidate_id: int,
    choice: GenerationChoice,
    enforce_steer_stop_boundary: bool = False,
) -> CandidateRecord:
    """Build one candidate record from one completion choice.

    Args:
        candidate_id: Candidate id in its pool.
        choice: Completion choice.
        enforce_steer_stop_boundary: Drop non-whitespace suffix text after the
            first terminal `</steer>` or `</think>` marker before validation.
    Returns:
        Candidate record.
    """

    token_ids = tuple(choice.token_ids) if choice.token_ids is not None else ()
    token_traces = token_traces_from_choice(choice=choice)
    text = append_think_close_stop_text(
        text=str(choice.text),
        stop_reason=choice.stop_reason,
    )
    if enforce_steer_stop_boundary:
        trimmed_text = text_through_first_steer_terminal(text=text)
        if trimmed_text != text:
            token_ids, token_traces = _truncate_candidate_token_payload(
                trimmed_text=trimmed_text,
                token_ids=token_ids,
                token_traces=token_traces,
            )
            text = trimmed_text
        assert_no_text_after_first_steer_close(text=text)
    return CandidateRecord(
        candidate_id=candidate_id,
        text=text,
        token_ids=token_ids,
        tokens=token_traces,
        finish_reason=str(choice.finish_reason),
        stop_reason=choice.stop_reason,
    )


def is_think_close_stop_reason(*, stop_reason: int | str | None) -> bool:
    """Return whether a stop reason corresponds to `</think>`.

    Args:
        stop_reason: vLLM stop reason value.

    Returns:
        `True` for string stop reasons matching the think-close marker.
    """

    if not isinstance(stop_reason, str):
        return False
    return "</think" in stop_reason.lower()


def append_think_close_stop_text(*, text: str, stop_reason: int | str | None) -> str:
    """Append `</think>` to display text when it was the excluded stop marker.

    Args:
        text: Raw choice text.
        stop_reason: vLLM stop reason value.

    Returns:
        Text with a visible terminal `</think>` marker when appropriate.
    """

    if not is_think_close_stop_reason(stop_reason=stop_reason):
        return text
    if THINK_CLOSE_TAG in text.lower():
        return text
    return text + THINK_CLOSE_TAG


def text_before_first_think_close(*, text: str) -> str:
    """Return text preceding the first `</think>` marker.

    Args:
        text: Candidate text.

    Returns:
        Prefix before `</think>`, or the original text when absent.
    """

    close_index = text.lower().find(THINK_CLOSE_TAG)
    if close_index < 0:
        return text
    return text[:close_index]


def text_after_first_think_close(*, text: str) -> str:
    """Return text following the first `</think>` marker.

    Args:
        text: Candidate or assistant-prefix text.

    Returns:
        Suffix after `</think>`, or an empty string when absent.
    """

    close_index = text.lower().find(THINK_CLOSE_TAG)
    if close_index < 0:
        return ""
    return text[close_index + len(THINK_CLOSE_TAG) :]


def has_text_after_first_think_close(*, text: str) -> bool:
    """Return whether an assistant prefix already contains post-think answer text."""

    return bool(text_after_first_think_close(text=text).strip())


def contains_complete_boxed_answer(*, text: str) -> bool:
    """Return whether text contains a complete LaTeX boxed expression."""

    boxed_open = "\\boxed{"
    search_start = 0
    while True:
        open_index = text.find(boxed_open, search_start)
        if open_index < 0:
            return False
        depth = 1
        char_index = open_index + len(boxed_open)
        while char_index < len(text):
            char = text[char_index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return True
            char_index += 1
        search_start = open_index + len(boxed_open)


def has_boxed_answer_after_first_think_close(*, text: str) -> bool:
    """Return whether post-think text includes a complete boxed answer."""

    return contains_complete_boxed_answer(text=text_after_first_think_close(text=text))


def text_through_first_steer_terminal(*, text: str) -> str:
    """Return text through the first steer-generation terminal marker.

    Args:
        text: Candidate text generated with steer stop sequences.

    Returns:
        Original text when no terminal marker exists or only whitespace follows
        it; otherwise the prefix ending at the first `</steer>` or `</think>`.

    Example:
        >>> text_through_first_steer_terminal(text="plan</steer>answer")
        'plan</steer>'
    """

    terminal_span = _first_steer_terminal_span(text=text)
    if terminal_span is None:
        return text
    close_index, close_end = terminal_span
    marker = text[close_index:close_end].lower()
    suffix = text[close_end:]
    if suffix.strip() == "" or (
        marker == STEER_CLOSE_TAG and _is_terminal_think_close_suffix(suffix=suffix)
    ):
        return text
    return text[:close_end]


def _first_steer_terminal_span(*, text: str) -> tuple[int, int] | None:
    """Return the span of the first steer terminal marker in text."""

    lowered = text.lower()
    terminal_spans = tuple(
        (index, index + len(marker))
        for marker in (STEER_CLOSE_TAG, THINK_CLOSE_TAG)
        if (index := lowered.find(marker)) >= 0
    )
    if not terminal_spans:
        return None
    return min(terminal_spans, key=lambda span: span[0])


def _truncate_candidate_token_payload(
    *,
    trimmed_text: str,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
) -> tuple[tuple[int, ...], tuple[TokenTrace, ...]]:
    """Trim candidate token payload when a text trim lands on a token boundary."""

    if not token_traces:
        return (), ()
    accumulated_text = ""
    for token_count, token_trace in enumerate(token_traces, start=1):
        accumulated_text += token_trace.token_text
        if accumulated_text == trimmed_text:
            return tuple(token_ids[:token_count]), tuple(token_traces[:token_count])
        if len(accumulated_text) >= len(trimmed_text):
            return (), ()
    return (), ()


def _is_terminal_think_close_suffix(*, suffix: str) -> bool:
    """Return whether suffix is only an optional terminal think close."""

    return suffix.strip().lower() == THINK_CLOSE_TAG


def assert_no_text_after_first_steer_close(*, text: str) -> None:
    """Assert steer-stopped text has no suffix after its first terminal marker.

    Args:
        text: Candidate text generated with steer stop sequences.

    Returns:
        None.
    """

    terminal_span = _first_steer_terminal_span(text=text)
    if terminal_span is None:
        return
    close_index, close_end = terminal_span
    marker = text[close_index:close_end].lower()
    suffix = text[close_end:]
    if marker == STEER_CLOSE_TAG and _is_terminal_think_close_suffix(suffix=suffix):
        return
    assert (
        suffix.strip() == ""
    ), f"unexpected text after first {marker} marker: suffix={suffix!r}"


def selected_ids_for_mode(
    *, selections: tuple[SelectionOutcome, ...], selector_mode: SelectorMode
) -> tuple[int, ...]:
    """Return selected candidate ids for one selector mode.

    Args:
        selections: Selection outcomes across modes.
        selector_mode: Target selector mode.

    Returns:
        Selected candidate ids for the mode, or empty tuple.
    """

    for selection in selections:
        if selection.selector_mode != selector_mode:
            continue
        return selection.selected_candidate_ids
    return ()


def candidate_text_by_id(*, pool: CandidatePoolRecord, candidate_id: int) -> str:
    """Lookup candidate text by candidate id from a pool.

    Args:
        pool: Candidate pool.
        candidate_id: Candidate id.

    Returns:
        Candidate text or empty string when missing.
    """

    for candidate in pool.candidates:
        if candidate.candidate_id != candidate_id:
            continue
        return candidate.text
    return ""


def updated_prompt_token_ids(
    *,
    current_prompt_token_ids: tuple[int, ...] | None,
    choice: GenerationChoice,
    consumed_tokens: int,
) -> tuple[int, ...] | None:
    """Append consumed generated tokens to prompt-token chain.

    Args:
        current_prompt_token_ids: Existing prompt chain.
        choice: Completion choice with token ids.
        consumed_tokens: Count of consumed tokens from choice.

    Returns:
        Updated prompt-token chain.
    """

    if consumed_tokens <= 0:
        return current_prompt_token_ids
    choice_token_ids = tuple(choice.token_ids) if choice.token_ids is not None else ()
    if len(choice_token_ids) < consumed_tokens:
        return None
    consumed_ids = tuple(choice_token_ids[:consumed_tokens])
    if current_prompt_token_ids is None:
        if choice.prompt_token_ids is None:
            return None
        return tuple(choice.prompt_token_ids) + consumed_ids
    return tuple(current_prompt_token_ids) + consumed_ids


def append_prompt_token_ids(
    *,
    prompt_token_ids: tuple[int, ...] | None,
    continuation_token_ids: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Append continuation token ids to prompt-token chain.

    Args:
        prompt_token_ids: Existing prompt-token chain.
        continuation_token_ids: Continuation token ids to append.

    Returns:
        Updated prompt-token chain.
    """

    if prompt_token_ids is None:
        return None
    return tuple(prompt_token_ids) + tuple(continuation_token_ids)


def rollout_stop_markers(
    *, steer_enabled: bool, use_full_stop_strings: bool = False
) -> tuple[str, ...] | None:
    """Return rollout stop markers for steer-enabled decode.

    Args:
        steer_enabled: Whether steer trigger is enabled.
        use_full_stop_strings: Whether vLLM should stop on complete control
            tags rather than legacy partial prefixes.

    Returns:
        Tuple of stop markers or `None`.
    """

    if not steer_enabled:
        return None
    if use_full_stop_strings:
        return ROLLOUT_STEER_STOP_FULL
    return ROLLOUT_STEER_STOP


def is_explicit_steer_stop(*, choice: GenerationChoice, steer_enabled: bool) -> bool:
    """Return whether choice represents an explicit exec-boundary stop.

    Args:
        choice: Completion choice.
        steer_enabled: Whether steer mode is enabled.

    Returns:
        `True` when stop reason corresponds to `</exec`.
    """

    if not steer_enabled:
        return False
    if str(choice.finish_reason) != "stop":
        return False
    if choice.stop_reason is None:
        return False
    return is_steer_stop_reason(stop_reason=choice.stop_reason)


def is_steer_stop_reason(*, stop_reason: int | str) -> bool:
    """Return whether stop reason corresponds to an exec-boundary marker.

    Args:
        stop_reason: Parsed stop reason.

    Returns:
        `True` when stop reason indicates `</exec`.
    """

    if not isinstance(stop_reason, str):
        return False
    return "</exec" in stop_reason.lower()


def steer_candidate_has_decision_tag(*, prefix: str, candidate_text: str) -> bool:
    """Return whether a steer-decision candidate opened steer or closed think.

    Args:
        prefix: Normalized assistant prefix used for candidate generation.
        candidate_text: Candidate text after excluded stop-marker repair.

    Returns:
        `True` for existing open-steer prefixes, or for candidates whose first
        non-whitespace text is `<steer>` or `</think>`.
    """

    if prefix.lower().endswith("<steer>"):
        return _candidate_closes_current_steer(candidate_text=candidate_text)
    stripped_candidate = candidate_text.lstrip().lower()
    if stripped_candidate.startswith(THINK_CLOSE_TAG):
        return True
    return _candidate_opens_valid_steer(candidate_text=stripped_candidate)


def _candidate_closes_current_steer(*, candidate_text: str) -> bool:
    stripped_candidate = candidate_text.lstrip().lower()
    steer_close_index = stripped_candidate.find(STEER_CLOSE_TAG)
    forbidden_indexes = _first_control_indexes_before_steer_close(
        text=stripped_candidate,
        steer_close_index=(
            len(stripped_candidate) if steer_close_index < 0 else steer_close_index
        ),
        start_index=0,
    )
    return not forbidden_indexes


def _candidate_opens_valid_steer(*, candidate_text: str) -> bool:
    open_match = STEER_OPEN_TAG_PATTERN.match(candidate_text)
    if open_match is None:
        return False
    steer_close_index = candidate_text.find(STEER_CLOSE_TAG, open_match.end())
    forbidden_indexes = _first_control_indexes_before_steer_close(
        text=candidate_text,
        steer_close_index=(
            len(candidate_text) if steer_close_index < 0 else steer_close_index
        ),
        start_index=open_match.end(),
    )
    return not forbidden_indexes


def _first_control_indexes_before_steer_close(
    *, text: str, steer_close_index: int, start_index: int
) -> tuple[int, ...]:
    indexes: list[int] = []
    for marker in ("<steer", "<exec", THINK_CLOSE_TAG):
        marker_index = text.find(marker, start_index)
        if 0 <= marker_index < steer_close_index:
            indexes.append(marker_index)
    return tuple(indexes)


def length_tokens_exec(*, text: str) -> int | None:
    """Return token count estimate for text inside last `<exec>` block.

    Args:
        text: Full completion text.

    Returns:
        Approximate token count via whitespace splitting, or `None`.
    """

    matches = EXEC_BLOCK_PATTERN.findall(text)
    if not matches:
        return None
    last_exec = str(matches[-1]).strip()
    if not last_exec:
        return 0
    return len(last_exec.split())


def has_steer_boundary(*, text: str) -> bool:
    """Return whether text ends at full or partial steer boundary marker.

    Args:
        text: Prefix text.

    Returns:
        `True` when steer boundary marker is present at text suffix.
    """

    return (
        STEER_BOUNDARY_PATTERN.search(text) is not None
        or STEER_PARTIAL_PATTERN.search(text) is not None
    )
