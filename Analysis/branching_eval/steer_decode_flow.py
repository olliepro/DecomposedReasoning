"""Steer-specific decode flow helpers used by branch executor."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from branching_eval.branch_decode_utils import (
    append_prompt_token_ids,
    candidate_from_choice,
    consume_choice_tokens,
    updated_prompt_token_ids,
)
from branching_eval.legacy_steer_rollout import (
    complete_trailing_partial_tag,
    contains_think_close,
    contains_think_close_or_partial,
    is_natural_finish_reason,
)
from branching_eval.runtime_types import DecodeOutcome
from branching_eval.steer_normalization import forced_boundary_suffix
from branching_eval.tree_types import TokenTrace
from vllm_client import GenerationChoice

if TYPE_CHECKING:
    from branching_eval.branch_executor import BranchExecutor


def should_branch_at_trigger(*, executor: BranchExecutor) -> bool:
    """Return whether an eligible trigger should branch for this path.

    Args:
        executor: Active branch executor.

    Returns:
        `True` when random draw is within configured branch probability.
    """

    return executor.random.random() <= executor.branching.branch_prob


def continue_with_single_steer_candidate(
    *,
    executor: BranchExecutor,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    generated_tokens: int,
    request_stream_id: str,
) -> DecodeOutcome:
    """Continue via one steer candidate when an eligible steer trigger doesn't branch.

    Args:
        executor: Active branch executor.
        assistant_prefix: Current assistant prefix.
        prompt_token_ids: Current prompt token chain.
        token_ids: Generated token ids accumulated so far.
        token_traces: Generated token traces accumulated so far.
        generated_tokens: Total generated tokens accumulated so far.

    Returns:
        `DecodeOutcome` with `event_type='continued'` when continuation succeeds,
        otherwise a terminal outcome.
    """

    remaining = executor.decoding.max_gen_toks - generated_tokens
    if remaining <= 0:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="max_gen_toks_reached",
        )
    canonical_prefix, canonical_prompt_ids = (
        executor._normalize_steer_prefix_prompt_ids(
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
        )
    )
    steer_choice = executor._generate_choice(
        assistant_prefix=canonical_prefix,
        prompt_token_ids=canonical_prompt_ids,
        max_tokens=min(executor.branching.max_steer_tokens, remaining),
        stop=("</steer",),
        n=1,
        request_kind="steer_single_candidate",
        request_stream_id=request_stream_id,
        enforce_prefix_chain=True,
    )
    if not steer_choice.tokens:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=canonical_prefix,
            prompt_token_ids=canonical_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="empty_generation",
        )
    steer_candidate = candidate_from_choice(
        candidate_id=0,
        choice=steer_choice,
        enforce_steer_stop_boundary=True,
    )
    steer_text, steer_token_ids = executor._normalized_child_candidate(
        trigger_type="steer_boundary",
        candidate=steer_candidate,
    )
    consumed = len(steer_candidate.token_ids)
    updated_prompt_ids = append_prompt_token_ids(
        prompt_token_ids=canonical_prompt_ids,
        continuation_token_ids=tuple(steer_candidate.token_ids),
    )
    if updated_prompt_ids is not None and len(steer_token_ids) > len(
        steer_candidate.token_ids
    ):
        extra_ids = tuple(steer_token_ids[len(steer_candidate.token_ids) :])
        updated_prompt_ids = append_prompt_token_ids(
            prompt_token_ids=updated_prompt_ids,
            continuation_token_ids=extra_ids,
        )
    return DecodeOutcome(
        event_type="continued",
        trigger_type=None,
        entropy_value=None,
        assistant_prefix=canonical_prefix + steer_text,
        prompt_token_ids=updated_prompt_ids,
        token_ids=tuple(token_ids) + tuple(steer_token_ids),
        token_traces=tuple(token_traces) + tuple(steer_candidate.tokens),
        generated_tokens=generated_tokens + consumed,
        stop_reason="",
    )


async def continue_with_single_steer_candidate_async(
    *,
    executor: BranchExecutor,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    generated_tokens: int,
    request_stream_id: str,
) -> DecodeOutcome:
    """Async variant of single-candidate steer continuation."""

    remaining = executor.decoding.max_gen_toks - generated_tokens
    if remaining <= 0:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="max_gen_toks_reached",
        )
    canonical_prefix, canonical_prompt_ids = (
        await executor._normalize_steer_prefix_prompt_ids_async(
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
        )
    )
    steer_choice = await executor._generate_choice_async(
        assistant_prefix=canonical_prefix,
        prompt_token_ids=canonical_prompt_ids,
        max_tokens=min(executor.branching.max_steer_tokens, remaining),
        stop=("</steer",),
        n=1,
        request_kind="steer_single_candidate",
        request_stream_id=request_stream_id,
        enforce_prefix_chain=True,
    )
    if not steer_choice.tokens:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=canonical_prefix,
            prompt_token_ids=canonical_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="empty_generation",
        )
    steer_candidate = candidate_from_choice(
        candidate_id=0,
        choice=steer_choice,
        enforce_steer_stop_boundary=True,
    )
    steer_text, steer_token_ids = await executor._normalized_child_candidate_async(
        trigger_type="steer_boundary",
        candidate=steer_candidate,
    )
    consumed = len(steer_candidate.token_ids)
    updated_prompt_ids = append_prompt_token_ids(
        prompt_token_ids=canonical_prompt_ids,
        continuation_token_ids=tuple(steer_candidate.token_ids),
    )
    if updated_prompt_ids is not None and len(steer_token_ids) > len(
        steer_candidate.token_ids
    ):
        extra_ids = tuple(steer_token_ids[len(steer_candidate.token_ids) :])
        updated_prompt_ids = append_prompt_token_ids(
            prompt_token_ids=updated_prompt_ids,
            continuation_token_ids=extra_ids,
        )
    return DecodeOutcome(
        event_type="continued",
        trigger_type=None,
        entropy_value=None,
        assistant_prefix=canonical_prefix + steer_text,
        prompt_token_ids=updated_prompt_ids,
        token_ids=tuple(token_ids) + tuple(steer_token_ids),
        token_traces=tuple(token_traces) + tuple(steer_candidate.tokens),
        generated_tokens=generated_tokens + consumed,
        stop_reason="",
    )


def resolve_think_close_outcome(
    *,
    executor: BranchExecutor,
    choice: GenerationChoice,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    generated_tokens: int,
    request_stream_id: str,
) -> DecodeOutcome:
    """Resolve steer-mode behavior when a think-close marker appears.

    Args:
        executor: Active branch executor.
        choice: Current decode choice.
        assistant_prefix: Current assistant prefix.
        prompt_token_ids: Current prompt token chain.
        token_ids: Generated token ids accumulated so far.
        token_traces: Generated token traces accumulated so far.
        generated_tokens: Total generated tokens accumulated so far.

    Returns:
        Terminal think-close outcome or no-stop continuation outcome.
    """

    natural_finish = is_natural_finish_reason(
        finish_reason=str(choice.finish_reason),
        stop_reason=choice.stop_reason,
    )
    if natural_finish and contains_think_close(text=str(choice.text)):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="think_end",
        )
    return continue_after_think_close(
        executor=executor,
        assistant_prefix=assistant_prefix,
        prompt_token_ids=prompt_token_ids,
        token_ids=token_ids,
        token_traces=token_traces,
        generated_tokens=generated_tokens,
        request_stream_id=request_stream_id,
    )


async def resolve_think_close_outcome_async(
    *,
    executor: BranchExecutor,
    choice: GenerationChoice,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    generated_tokens: int,
    request_stream_id: str,
) -> DecodeOutcome:
    """Async variant of think-close handling."""

    natural_finish = is_natural_finish_reason(
        finish_reason=str(choice.finish_reason),
        stop_reason=choice.stop_reason,
    )
    if natural_finish and contains_think_close(text=str(choice.text)):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="think_end",
        )
    return await continue_after_think_close_async(
        executor=executor,
        assistant_prefix=assistant_prefix,
        prompt_token_ids=prompt_token_ids,
        token_ids=token_ids,
        token_traces=token_traces,
        generated_tokens=generated_tokens,
        request_stream_id=request_stream_id,
    )


def continue_after_think_close(
    *,
    executor: BranchExecutor,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    generated_tokens: int,
    request_stream_id: str,
) -> DecodeOutcome:
    """Continue steer decode with `stop=None` after think-close markers.

    Args:
        executor: Active branch executor.
        assistant_prefix: Current assistant prefix.
        prompt_token_ids: Current prompt token chain.
        token_ids: Generated token ids accumulated so far.
        token_traces: Generated token traces accumulated so far.
        generated_tokens: Total generated tokens accumulated so far.

    Returns:
        Terminal decode outcome for no-stop continuation.
    """

    remaining = executor.decoding.max_gen_toks - generated_tokens
    if remaining <= 0:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="max_gen_toks_reached",
        )
    continuation_choice = executor._generate_choice(
        assistant_prefix=assistant_prefix,
        prompt_token_ids=prompt_token_ids,
        max_tokens=remaining,
        stop=None,
        n=1,
        request_kind="think_close_continuation",
        request_stream_id=request_stream_id,
        enforce_prefix_chain=True,
    )
    if not continuation_choice.tokens:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="empty_generation",
        )
    continuation = consume_choice_tokens(
        choice=continuation_choice,
        assistant_prefix=assistant_prefix,
        token_ids=list(token_ids),
        token_traces=list(token_traces),
        generated_tokens=generated_tokens,
        trigger_steer=False,
        trigger_entropy=False,
        entropy_threshold=0.0,
        branch_prob=0.0,
        rng=executor.random,
    )
    consumed_tokens = continuation.generated_tokens - generated_tokens
    updated_prompt_ids = updated_prompt_token_ids(
        current_prompt_token_ids=prompt_token_ids,
        choice=continuation_choice,
        consumed_tokens=consumed_tokens,
    )
    stop_reason = (
        "max_gen_toks_reached"
        if str(continuation_choice.finish_reason) == "length"
        else "think_end"
    )
    return replace(
        continuation,
        event_type="terminated",
        prompt_token_ids=updated_prompt_ids,
        stop_reason=stop_reason,
    )


async def continue_after_think_close_async(
    *,
    executor: BranchExecutor,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    generated_tokens: int,
    request_stream_id: str,
) -> DecodeOutcome:
    """Async variant for no-stop continuation after think-close markers."""

    remaining = executor.decoding.max_gen_toks - generated_tokens
    if remaining <= 0:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="max_gen_toks_reached",
        )
    continuation_choice = await executor._generate_choice_async(
        assistant_prefix=assistant_prefix,
        prompt_token_ids=prompt_token_ids,
        max_tokens=remaining,
        stop=None,
        n=1,
        request_kind="think_close_continuation",
        request_stream_id=request_stream_id,
        enforce_prefix_chain=True,
    )
    if not continuation_choice.tokens:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            entropy_value=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="empty_generation",
        )
    continuation = consume_choice_tokens(
        choice=continuation_choice,
        assistant_prefix=assistant_prefix,
        token_ids=list(token_ids),
        token_traces=list(token_traces),
        generated_tokens=generated_tokens,
        trigger_steer=False,
        trigger_entropy=False,
        entropy_threshold=0.0,
        branch_prob=0.0,
        rng=executor.random,
    )
    consumed_tokens = continuation.generated_tokens - generated_tokens
    updated_prompt_ids = updated_prompt_token_ids(
        current_prompt_token_ids=prompt_token_ids,
        choice=continuation_choice,
        consumed_tokens=consumed_tokens,
    )
    stop_reason = (
        "max_gen_toks_reached"
        if str(continuation_choice.finish_reason) == "length"
        else "think_end"
    )
    return replace(
        continuation,
        event_type="terminated",
        prompt_token_ids=updated_prompt_ids,
        stop_reason=stop_reason,
    )


def resolve_steer_length_outcome(
    *,
    executor: BranchExecutor,
    choice: GenerationChoice,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    generated_tokens: int,
    request_stream_id: str,
) -> DecodeOutcome:
    """Resolve steer-mode behavior on length boundaries.

    Args:
        executor: Active branch executor.
        choice: Current decode choice.
        assistant_prefix: Current assistant prefix.
        prompt_token_ids: Current prompt token chain.
        token_ids: Generated token ids accumulated so far.
        token_traces: Generated token traces accumulated so far.
        generated_tokens: Total generated tokens accumulated so far.

    Returns:
        Trigger outcome at steer boundary, or terminal think-close continuation outcome.
    """

    repaired_prefix, repaired_tag = complete_trailing_partial_tag(text=assistant_prefix)
    repaired_prompt_ids = append_injected_suffix_prompt_ids(
        executor=executor,
        prompt_token_ids=prompt_token_ids,
        original_text=assistant_prefix,
        updated_text=repaired_prefix,
    )
    if repaired_tag == "<steer>":
        normalized_prefix, normalized_prompt_ids = (
            executor._normalize_steer_prefix_prompt_ids(
                assistant_prefix=repaired_prefix,
                prompt_token_ids=repaired_prompt_ids,
            )
        )
        return DecodeOutcome(
            event_type="trigger",
            trigger_type="steer_boundary",
            entropy_value=None,
            assistant_prefix=normalized_prefix,
            prompt_token_ids=normalized_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="",
        )
    if contains_think_close_or_partial(text=str(choice.text)):
        return continue_after_think_close(
            executor=executor,
            assistant_prefix=repaired_prefix,
            prompt_token_ids=repaired_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            request_stream_id=request_stream_id,
        )
    forced_suffix = forced_boundary_suffix(text=repaired_prefix)
    forced_prefix = repaired_prefix + forced_suffix
    forced_prompt_ids = append_injected_suffix_prompt_ids(
        executor=executor,
        prompt_token_ids=repaired_prompt_ids,
        original_text=repaired_prefix,
        updated_text=forced_prefix,
    )
    return DecodeOutcome(
        event_type="trigger",
        trigger_type="steer_boundary",
        entropy_value=None,
        assistant_prefix=forced_prefix,
        prompt_token_ids=forced_prompt_ids,
        token_ids=token_ids,
        token_traces=token_traces,
        generated_tokens=generated_tokens,
        stop_reason="",
    )


async def resolve_steer_length_outcome_async(
    *,
    executor: BranchExecutor,
    choice: GenerationChoice,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    generated_tokens: int,
    request_stream_id: str,
) -> DecodeOutcome:
    """Async variant of steer length-boundary handling."""

    repaired_prefix, repaired_tag = complete_trailing_partial_tag(text=assistant_prefix)
    repaired_prompt_ids = await append_injected_suffix_prompt_ids_async(
        executor=executor,
        prompt_token_ids=prompt_token_ids,
        original_text=assistant_prefix,
        updated_text=repaired_prefix,
    )
    if repaired_tag == "<steer>":
        normalized_prefix, normalized_prompt_ids = (
            await executor._normalize_steer_prefix_prompt_ids_async(
                assistant_prefix=repaired_prefix,
                prompt_token_ids=repaired_prompt_ids,
            )
        )
        return DecodeOutcome(
            event_type="trigger",
            trigger_type="steer_boundary",
            entropy_value=None,
            assistant_prefix=normalized_prefix,
            prompt_token_ids=normalized_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="",
        )
    if contains_think_close_or_partial(text=str(choice.text)):
        return await continue_after_think_close_async(
            executor=executor,
            assistant_prefix=repaired_prefix,
            prompt_token_ids=repaired_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            request_stream_id=request_stream_id,
        )
    forced_suffix = forced_boundary_suffix(text=repaired_prefix)
    forced_prefix = repaired_prefix + forced_suffix
    forced_prompt_ids = await append_injected_suffix_prompt_ids_async(
        executor=executor,
        prompt_token_ids=repaired_prompt_ids,
        original_text=repaired_prefix,
        updated_text=forced_prefix,
    )
    return DecodeOutcome(
        event_type="trigger",
        trigger_type="steer_boundary",
        entropy_value=None,
        assistant_prefix=forced_prefix,
        prompt_token_ids=forced_prompt_ids,
        token_ids=token_ids,
        token_traces=token_traces,
        generated_tokens=generated_tokens,
        stop_reason="",
    )


async def append_injected_suffix_prompt_ids_async(
    *,
    executor: BranchExecutor,
    prompt_token_ids: tuple[int, ...] | None,
    original_text: str,
    updated_text: str,
) -> tuple[int, ...] | None:
    """Async variant of injected-suffix prompt-id extension helper."""

    if updated_text == original_text:
        return prompt_token_ids
    if prompt_token_ids is None:
        return None
    injected_suffix = updated_text[len(original_text) :]
    assert injected_suffix, "injected suffix expected when text changed"
    suffix_token_ids = await executor._tokenize_text_async(text=injected_suffix)
    await executor._assert_text_token_alignment_async(
        text=injected_suffix,
        token_ids=tuple(suffix_token_ids),
        context="append_injected_suffix_prompt_ids",
    )
    return tuple(prompt_token_ids) + tuple(suffix_token_ids)


def append_injected_suffix_prompt_ids(
    *,
    executor: BranchExecutor,
    prompt_token_ids: tuple[int, ...] | None,
    original_text: str,
    updated_text: str,
) -> tuple[int, ...] | None:
    """Append tokenized injected text suffix to prompt-token chain.

    Args:
        executor: Active branch executor.
        prompt_token_ids: Existing prompt token chain.
        original_text: Previous assistant prefix text.
        updated_text: Updated assistant prefix text.

    Returns:
        Updated prompt token chain.
    """

    if updated_text == original_text:
        return prompt_token_ids
    if prompt_token_ids is None:
        return None
    injected_suffix = updated_text[len(original_text) :]
    assert injected_suffix, "injected suffix expected when text changed"
    suffix_token_ids = executor.client.tokenize(
        model=executor.model_name,
        text=injected_suffix,
        add_special_tokens=False,
    )
    executor._assert_text_token_alignment(
        text=injected_suffix,
        token_ids=tuple(suffix_token_ids),
        context="append_injected_suffix_prompt_ids",
    )
    return tuple(prompt_token_ids) + tuple(suffix_token_ids)
