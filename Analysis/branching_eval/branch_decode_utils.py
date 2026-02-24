"""Decode/runtime helper functions shared by branching executor."""

from __future__ import annotations

import random
import re

from branching_eval.legacy_steer_rollout import STEER_CLOSE_TAG
from branching_eval.runtime_types import DecodeOutcome
from branching_eval.selector_types import SelectionOutcome, SelectorMode
from branching_eval.tree_types import CandidatePoolRecord, CandidateRecord, TokenTrace
from token_metrics import approximate_entropy
from vllm_client import GenerationChoice, ParsedToken

STEER_BOUNDARY_PATTERN = re.compile(r"<steer>$", flags=re.IGNORECASE)
STEER_PARTIAL_PATTERN = re.compile(r"<steer$", flags=re.IGNORECASE)
EXEC_BLOCK_PATTERN = re.compile(
    r"<exec\b[^>]*>(.*?)</exec>", flags=re.IGNORECASE | re.DOTALL
)
ROLLOUT_STEER_STOP = ("<steer",)


def consume_choice_tokens(
    *,
    choice: GenerationChoice,
    assistant_prefix: str,
    token_ids: list[int],
    token_traces: list[TokenTrace],
    generated_tokens: int,
    trigger_steer: bool,
    trigger_entropy: bool,
    entropy_threshold: float,
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
        trigger_entropy: Whether entropy trigger is enabled.
        entropy_threshold: Entropy trigger threshold.
        branch_prob: Branching probability at eligible trigger points.
        rng: RNG used for branch-probability sampling.

    Returns:
        Decode outcome reflecting trigger/termination after processing the choice.
    """

    updated_prefix = assistant_prefix
    updated_token_ids = list(token_ids)
    updated_traces = list(token_traces)
    generated = generated_tokens
    choice_token_ids = tuple(choice.token_ids) if choice.token_ids is not None else ()
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
        entropy_trigger = trigger_entropy and token_trace.entropy > entropy_threshold
        if not (steer_trigger or entropy_trigger):
            continue
        if rng.random() > branch_prob:
            continue
        trigger_type = "steer_boundary" if steer_trigger else "high_entropy"
        entropy_value = None if steer_trigger else token_trace.entropy
        return DecodeOutcome(
            event_type="trigger",
            trigger_type=trigger_type,
            entropy_value=entropy_value,
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
        entropy_value=None,
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
        Token trace row with probability and entropy estimates.
    """

    probability, entropy, _ = approximate_entropy(
        selected_token=parsed_token.token,
        selected_logprob=parsed_token.logprob,
        top_entries=parsed_token.top_entries,
    )
    return TokenTrace(
        token_index=token_index,
        token_id=token_id,
        token_text=parsed_token.token,
        logprob=float(parsed_token.logprob),
        probability=float(probability),
        entropy=float(entropy),
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
        enforce_steer_stop_boundary: Assert no text follows first `</steer>`.
    Returns:
        Candidate record.
    """

    token_ids = tuple(choice.token_ids) if choice.token_ids is not None else ()
    token_traces = token_traces_from_choice(choice=choice)
    text = str(choice.text)
    if enforce_steer_stop_boundary:
        assert_no_text_after_first_steer_close(text=text)
    return CandidateRecord(
        candidate_id=candidate_id,
        text=text,
        token_ids=token_ids,
        tokens=token_traces,
        finish_reason=str(choice.finish_reason),
        stop_reason=choice.stop_reason,
    )


def assert_no_text_after_first_steer_close(*, text: str) -> None:
    """Assert steer-stopped text has no suffix after first `</steer>` marker.

    Args:
        text: Candidate text generated with steer stop sequence.

    Returns:
        None.
    """

    lowered = text.lower()
    close_index = lowered.find(STEER_CLOSE_TAG)
    if close_index < 0:
        return
    suffix = text[close_index + len(STEER_CLOSE_TAG) :]
    assert (
        suffix.strip() == ""
    ), f"unexpected text after first </steer> marker: suffix={suffix!r}"


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


def rollout_stop_markers(*, steer_enabled: bool) -> tuple[str, ...] | None:
    """Return rollout stop markers for steer-enabled decode.

    Args:
        steer_enabled: Whether steer trigger is enabled.

    Returns:
        Tuple of stop markers or `None`.
    """

    if not steer_enabled:
        return None
    return ROLLOUT_STEER_STOP


def is_explicit_steer_stop(*, choice: GenerationChoice, steer_enabled: bool) -> bool:
    """Return whether choice represents explicit steer stop marker hit.

    Args:
        choice: Completion choice.
        steer_enabled: Whether steer mode is enabled.

    Returns:
        `True` when stop reason corresponds to `<steer` marker.
    """

    if not steer_enabled:
        return False
    if str(choice.finish_reason) != "stop":
        return False
    if choice.stop_reason is None:
        return False
    return is_steer_stop_reason(stop_reason=choice.stop_reason)


def is_steer_stop_reason(*, stop_reason: int | str) -> bool:
    """Return whether stop reason corresponds to a steer boundary marker.

    Args:
        stop_reason: Parsed stop reason.

    Returns:
        `True` when stop reason indicates `<steer`.
    """

    if isinstance(stop_reason, int):
        return True
    return "<steer" in stop_reason.lower()


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
