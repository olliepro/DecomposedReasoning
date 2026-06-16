"""Steer-specific decode flow helpers used by branch executor."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from branching_eval.branch_decode_utils import (
    append_prompt_token_ids,
    candidate_from_choice,
    choice_has_generated_content,
    consume_choice_tokens,
    has_boxed_answer_after_first_think_close,
    is_think_close_stop_reason,
    steer_candidate_has_decision_tag,
    updated_prompt_token_ids,
)
from branching_eval.legacy_steer_rollout import (
    complete_trailing_partial_tag,
    contains_think_close_or_partial,
    is_chat_eos_stop_reason,
    is_natural_finish_reason,
)
from branching_eval.runtime_types import DecodeOutcome
from branching_eval.steer_normalization import (
    forced_boundary_suffix,
    is_initial_steer_decision_boundary,
    is_inside_open_exec,
    steer_candidate_stop_markers,
)
from branching_eval.tree_types import TokenTrace
from vllm_client import GenerationChoice, truncate_choice_at_chat_eos

if TYPE_CHECKING:
    from branching_eval.branch_executor import BranchExecutor


POST_THINK_CONTINUATION_MAX_TOKENS = 2048


def should_branch_at_trigger(*, executor: BranchExecutor) -> bool:
    """Return whether an eligible trigger should branch for this path.

    Args:
        executor: Active branch executor.

    Returns:
        `True` when random draw is within configured branch probability.
    """

    return executor.random.random() <= executor.branching.branch_prob


def _build_steer_continuation_outcome(
    *,
    executor: BranchExecutor,
    canonical_prefix: str,
    updated_prompt_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    steer_text: str,
    steer_token_ids: tuple[int, ...],
    steer_candidate_tokens: tuple[TokenTrace, ...],
    generated_tokens: int,
) -> DecodeOutcome:
    """Build one normalized steer-continuation outcome under the path token cap.

    Args:
        executor: Active branch executor.
        canonical_prefix: Canonicalized prefix at steer boundary.
        updated_prompt_ids: Prompt token chain after suffix injection.
        token_ids: Existing generated token ids.
        token_traces: Existing generated token traces.
        steer_text: Normalized steer continuation text.
        steer_token_ids: Normalized steer continuation token ids.
        steer_candidate_tokens: Raw vLLM token traces for the steer candidate.
        generated_tokens: Tokens already realized before continuation.

    Returns:
        Continued decode outcome with normalized steer suffix appended in-path.
    """

    span_start = len(token_ids)
    updated_generated_tokens = generated_tokens + len(steer_token_ids)
    span_end = span_start + len(steer_token_ids)
    return DecodeOutcome(
        event_type="continued",
        trigger_type=None,
        assistant_prefix=canonical_prefix + steer_text,
        prompt_token_ids=updated_prompt_ids,
        token_ids=tuple(token_ids) + tuple(steer_token_ids),
        token_traces=tuple(token_traces) + tuple(steer_candidate_tokens),
        generated_tokens=updated_generated_tokens,
        stop_reason="",
    )


def _build_malformed_steer_termination_outcome(
    *,
    canonical_prefix: str,
    updated_prompt_ids: tuple[int, ...] | None,
    token_ids: tuple[int, ...],
    token_traces: tuple[TokenTrace, ...],
    steer_text: str,
    steer_token_ids: tuple[int, ...],
    steer_candidate_tokens: tuple[TokenTrace, ...],
    generated_tokens: int,
) -> DecodeOutcome:
    """Build a terminal outcome that still preserves rejected steer text."""

    span_start = len(token_ids)
    span_end = span_start + len(steer_token_ids)
    updated_generated_tokens = generated_tokens + len(steer_token_ids)
    return DecodeOutcome(
        event_type="terminated",
        trigger_type=None,
        assistant_prefix=canonical_prefix + steer_text,
        prompt_token_ids=updated_prompt_ids,
        token_ids=tuple(token_ids) + tuple(steer_token_ids),
        token_traces=tuple(token_traces) + tuple(steer_candidate_tokens),
        generated_tokens=updated_generated_tokens,
        stop_reason="missing_steer_or_think_close",
        steer_phase_token_spans=((span_start, span_end),),
    )


def terminate_on_chat_eos_choice(
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
    """Consume one EOS-stopped choice as terminal without normalization."""

    if not choice_has_generated_content(choice=choice):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="model_finished",
        )
    outcome = consume_choice_tokens(
        choice=choice,
        assistant_prefix=assistant_prefix,
        token_ids=list(token_ids),
        token_traces=list(token_traces),
        generated_tokens=generated_tokens,
        trigger_steer=False,
        branch_prob=0.0,
        rng=executor.random,
    )
    consumed_tokens = outcome.generated_tokens - generated_tokens
    updated_prompt_ids = updated_prompt_token_ids(
        current_prompt_token_ids=prompt_token_ids,
        choice=choice,
        consumed_tokens=consumed_tokens,
    )
    executor._update_request_stream_state_output_ids(
        request_stream_id=request_stream_id,
        consumed_output_token_ids=tuple(choice.token_ids or ())[:consumed_tokens],
    )
    return replace(
        outcome,
        event_type="terminated",
        trigger_type=None,
        prompt_token_ids=updated_prompt_ids,
        stop_reason="model_finished",
    )


def _open_steer_generation_prefix(*, text: str) -> str | None:
    """Return the open steer prefix that should be sampled before normalization."""

    completed_text, repaired_tag = complete_trailing_partial_tag(text=text)
    if repaired_tag not in (None, "<steer>"):
        return None
    if not completed_text.lower().endswith("<steer>"):
        return None
    if is_inside_open_exec(text=completed_text):
        return None
    return completed_text


def prepare_steer_generation_prefix(
    *,
    executor: BranchExecutor,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
) -> tuple[str, tuple[int, ...] | None]:
    """Prepare the prefix used to sample steer text.

    An already-open steer tag should receive model text before structural
    normalization runs. Invalid prefixes, especially open exec blocks, still use
    full normalization before candidate sampling.
    """

    open_prefix = _open_steer_generation_prefix(text=assistant_prefix)
    if open_prefix is None:
        if is_initial_steer_decision_boundary(text=assistant_prefix):
            return assistant_prefix, prompt_token_ids
        return executor._normalize_steer_prefix_prompt_ids(
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
        )
    if prompt_token_ids is None:
        return open_prefix, None
    injected_suffix = open_prefix[len(assistant_prefix) :]
    if not injected_suffix:
        return open_prefix, prompt_token_ids
    suffix_token_ids = executor._tokenize_text(text=injected_suffix)
    executor._assert_text_token_alignment(
        text=injected_suffix,
        token_ids=tuple(suffix_token_ids),
        context="open_steer_prefix_completion",
    )
    return open_prefix, tuple(prompt_token_ids) + tuple(suffix_token_ids)


async def prepare_steer_generation_prefix_async(
    *,
    executor: BranchExecutor,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
) -> tuple[str, tuple[int, ...] | None]:
    """Async variant of `prepare_steer_generation_prefix`."""

    open_prefix = _open_steer_generation_prefix(text=assistant_prefix)
    if open_prefix is None:
        if is_initial_steer_decision_boundary(text=assistant_prefix):
            return assistant_prefix, prompt_token_ids
        return await executor._normalize_steer_prefix_prompt_ids_async(
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
        )
    if prompt_token_ids is None:
        return open_prefix, None
    injected_suffix = open_prefix[len(assistant_prefix) :]
    if not injected_suffix:
        return open_prefix, prompt_token_ids
    suffix_token_ids = await executor._tokenize_text_async(text=injected_suffix)
    await executor._assert_text_token_alignment_async(
        text=injected_suffix,
        token_ids=tuple(suffix_token_ids),
        context="open_steer_prefix_completion",
    )
    return open_prefix, tuple(prompt_token_ids) + tuple(suffix_token_ids)


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

    steer_budget = executor._candidate_token_budget(
        trigger_type="steer_boundary",
        generated_tokens=generated_tokens,
    )
    if steer_budget <= 0:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="max_gen_toks_reached",
        )
    canonical_prefix, canonical_prompt_ids = prepare_steer_generation_prefix(
        executor=executor,
        assistant_prefix=assistant_prefix,
        prompt_token_ids=prompt_token_ids,
    )
    # Steer normalization can rewrite the prompt boundary, so the next request
    # must start a fresh prefix chain for this stream.
    executor._reset_request_stream_state(request_stream_id=request_stream_id)
    steer_choice = executor._generate_choice(
        assistant_prefix=canonical_prefix,
        prompt_token_ids=canonical_prompt_ids,
        max_tokens=steer_budget,
        stop=steer_candidate_stop_markers(text=canonical_prefix),
        n=1,
        request_kind="steer_single_candidate",
        request_stream_id=request_stream_id,
        enforce_prefix_chain=True,
    )
    steer_choice = truncate_choice_at_chat_eos(choice=steer_choice)
    if is_chat_eos_stop_reason(stop_reason=steer_choice.stop_reason):
        return terminate_on_chat_eos_choice(
            executor=executor,
            choice=steer_choice,
            assistant_prefix=canonical_prefix,
            prompt_token_ids=canonical_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            request_stream_id=request_stream_id,
        )
    if not choice_has_generated_content(
        choice=steer_choice
    ) and not is_think_close_stop_reason(stop_reason=steer_choice.stop_reason):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
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
    steer_candidate_text, steer_candidate_token_ids = (
        executor._append_think_close_stop_suffix(
            candidate=steer_candidate
        )
    )
    if steer_candidate_text and not steer_candidate_token_ids:
        steer_candidate_token_ids = tuple(
            executor._tokenize_text(text=steer_candidate_text)
        )
    malformed_prompt_ids = append_prompt_token_ids(
        prompt_token_ids=canonical_prompt_ids,
        continuation_token_ids=tuple(steer_candidate_token_ids),
    )
    if not steer_candidate_has_decision_tag(
        prefix=canonical_prefix,
        candidate_text=steer_candidate_text,
    ):
        executor._update_request_stream_state_output_ids(
            request_stream_id=request_stream_id,
            consumed_output_token_ids=tuple(steer_candidate.token_ids),
        )
        return _build_malformed_steer_termination_outcome(
            canonical_prefix=canonical_prefix,
            updated_prompt_ids=malformed_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            steer_text=steer_candidate_text,
            steer_token_ids=tuple(steer_candidate_token_ids),
            steer_candidate_tokens=tuple(steer_candidate.tokens),
            generated_tokens=generated_tokens,
        )
    steer_text, steer_token_ids = executor._normalized_child_candidate(
        trigger_type="steer_boundary",
        candidate=steer_candidate
    )
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
    executor._update_request_stream_state_output_ids(
        request_stream_id=request_stream_id,
        consumed_output_token_ids=tuple(steer_candidate.token_ids),
    )
    return _build_steer_continuation_outcome(
        executor=executor,
        canonical_prefix=canonical_prefix,
        updated_prompt_ids=updated_prompt_ids,
        token_ids=token_ids,
        token_traces=token_traces,
        steer_text=steer_text,
        steer_token_ids=tuple(steer_token_ids),
        steer_candidate_tokens=tuple(steer_candidate.tokens),
        generated_tokens=generated_tokens,
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

    steer_budget = await executor._candidate_token_budget_async(
        trigger_type="steer_boundary",
        generated_tokens=generated_tokens,
    )
    if steer_budget <= 0:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="max_gen_toks_reached",
        )
    canonical_prefix, canonical_prompt_ids = (
        await prepare_steer_generation_prefix_async(
            executor=executor,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
        )
    )
    # Steer normalization can rewrite the prompt boundary, so the next request
    # must start a fresh prefix chain for this stream.
    executor._reset_request_stream_state(request_stream_id=request_stream_id)
    steer_choice = await executor._generate_choice_async(
        assistant_prefix=canonical_prefix,
        prompt_token_ids=canonical_prompt_ids,
        max_tokens=steer_budget,
        stop=steer_candidate_stop_markers(text=canonical_prefix),
        n=1,
        request_kind="steer_single_candidate",
        request_stream_id=request_stream_id,
        enforce_prefix_chain=True,
    )
    steer_choice = truncate_choice_at_chat_eos(choice=steer_choice)
    if is_chat_eos_stop_reason(stop_reason=steer_choice.stop_reason):
        return terminate_on_chat_eos_choice(
            executor=executor,
            choice=steer_choice,
            assistant_prefix=canonical_prefix,
            prompt_token_ids=canonical_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            request_stream_id=request_stream_id,
        )
    if not choice_has_generated_content(
        choice=steer_choice
    ) and not is_think_close_stop_reason(stop_reason=steer_choice.stop_reason):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
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
    steer_candidate_text, steer_candidate_token_ids = (
        await executor._append_think_close_stop_suffix_async(
            candidate=steer_candidate
        )
    )
    if steer_candidate_text and not steer_candidate_token_ids:
        steer_candidate_token_ids = tuple(
            await executor._tokenize_text_async(text=steer_candidate_text)
        )
    malformed_prompt_ids = append_prompt_token_ids(
        prompt_token_ids=canonical_prompt_ids,
        continuation_token_ids=tuple(steer_candidate_token_ids),
    )
    if not steer_candidate_has_decision_tag(
        prefix=canonical_prefix,
        candidate_text=steer_candidate_text,
    ):
        executor._update_request_stream_state_output_ids(
            request_stream_id=request_stream_id,
            consumed_output_token_ids=tuple(steer_candidate.token_ids),
        )
        return _build_malformed_steer_termination_outcome(
            canonical_prefix=canonical_prefix,
            updated_prompt_ids=malformed_prompt_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            steer_text=steer_candidate_text,
            steer_token_ids=tuple(steer_candidate_token_ids),
            steer_candidate_tokens=tuple(steer_candidate.tokens),
            generated_tokens=generated_tokens,
        )
    steer_text, steer_token_ids = await executor._normalized_child_candidate_async(
        trigger_type="steer_boundary",
        candidate=steer_candidate,
    )
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
    executor._update_request_stream_state_output_ids(
        request_stream_id=request_stream_id,
        consumed_output_token_ids=tuple(steer_candidate.token_ids),
    )
    return _build_steer_continuation_outcome(
        executor=executor,
        canonical_prefix=canonical_prefix,
        updated_prompt_ids=updated_prompt_ids,
        token_ids=token_ids,
        token_traces=token_traces,
        steer_text=steer_text,
        steer_token_ids=tuple(steer_token_ids),
        steer_candidate_tokens=tuple(steer_candidate.tokens),
        generated_tokens=generated_tokens,
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
    if is_chat_eos_stop_reason(stop_reason=choice.stop_reason):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="model_finished",
        )
    if natural_finish and has_boxed_answer_after_first_think_close(
        text=assistant_prefix
    ):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
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
    if is_chat_eos_stop_reason(stop_reason=choice.stop_reason):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason="model_finished",
        )
    if natural_finish and has_boxed_answer_after_first_think_close(
        text=assistant_prefix
    ):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
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


def _think_close_stop_reason(
    *,
    finish_reason: str,
    stop_reason: int | str | None,
    context_limited: bool = False,
    post_think_limited: bool = False,
) -> str:
    """Return a terminal stop reason for post-think answer continuation."""

    if is_chat_eos_stop_reason(stop_reason=stop_reason):
        return "model_finished"
    if str(finish_reason) != "length":
        return "think_end"
    if context_limited:
        return "max_context_length_reached"
    if post_think_limited:
        return "post_think_toks_reached"
    return "max_gen_toks_reached"


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

    request_budget = executor._request_token_budget(
        prompt_token_ids=prompt_token_ids,
        generated_tokens=generated_tokens,
    )
    if request_budget.max_tokens <= 0:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason=request_budget.exhausted_stop_reason,
        )
    continuation_max_tokens = min(
        request_budget.max_tokens,
        POST_THINK_CONTINUATION_MAX_TOKENS,
    )
    post_think_limited = continuation_max_tokens < request_budget.max_tokens
    executor._reset_request_stream_state(request_stream_id=request_stream_id)
    continuation_choice = executor._generate_choice(
        assistant_prefix=assistant_prefix,
        prompt_token_ids=prompt_token_ids,
        max_tokens=continuation_max_tokens,
        stop=None,
        n=1,
        request_kind="think_close_continuation",
        request_stream_id=request_stream_id,
        enforce_prefix_chain=True,
    )
    continuation_choice = truncate_choice_at_chat_eos(choice=continuation_choice)
    if is_chat_eos_stop_reason(stop_reason=continuation_choice.stop_reason):
        return terminate_on_chat_eos_choice(
            executor=executor,
            choice=continuation_choice,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            request_stream_id=request_stream_id,
        )
    if not choice_has_generated_content(choice=continuation_choice):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
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
        branch_prob=0.0,
        rng=executor.random,
    )
    consumed_tokens = continuation.generated_tokens - generated_tokens
    updated_prompt_ids = updated_prompt_token_ids(
        current_prompt_token_ids=prompt_token_ids,
        choice=continuation_choice,
        consumed_tokens=consumed_tokens,
    )
    executor._update_request_stream_state_output_ids(
        request_stream_id=request_stream_id,
        consumed_output_token_ids=tuple(continuation_choice.token_ids or ())[
            :consumed_tokens
        ],
    )
    stop_reason = _think_close_stop_reason(
        finish_reason=str(continuation_choice.finish_reason),
        stop_reason=continuation_choice.stop_reason,
        context_limited=request_budget.context_limited,
        post_think_limited=post_think_limited,
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

    request_budget = executor._request_token_budget(
        prompt_token_ids=prompt_token_ids,
        generated_tokens=generated_tokens,
    )
    if request_budget.max_tokens <= 0:
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            stop_reason=request_budget.exhausted_stop_reason,
        )
    continuation_max_tokens = min(
        request_budget.max_tokens,
        POST_THINK_CONTINUATION_MAX_TOKENS,
    )
    post_think_limited = continuation_max_tokens < request_budget.max_tokens
    executor._reset_request_stream_state(request_stream_id=request_stream_id)
    continuation_choice = await executor._generate_choice_async(
        assistant_prefix=assistant_prefix,
        prompt_token_ids=prompt_token_ids,
        max_tokens=continuation_max_tokens,
        stop=None,
        n=1,
        request_kind="think_close_continuation",
        request_stream_id=request_stream_id,
        enforce_prefix_chain=True,
    )
    continuation_choice = truncate_choice_at_chat_eos(choice=continuation_choice)
    if is_chat_eos_stop_reason(stop_reason=continuation_choice.stop_reason):
        return terminate_on_chat_eos_choice(
            executor=executor,
            choice=continuation_choice,
            assistant_prefix=assistant_prefix,
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_traces=token_traces,
            generated_tokens=generated_tokens,
            request_stream_id=request_stream_id,
        )
    if not choice_has_generated_content(choice=continuation_choice):
        return DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
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
        branch_prob=0.0,
        rng=executor.random,
    )
    consumed_tokens = continuation.generated_tokens - generated_tokens
    updated_prompt_ids = updated_prompt_token_ids(
        current_prompt_token_ids=prompt_token_ids,
        choice=continuation_choice,
        consumed_tokens=consumed_tokens,
    )
    executor._update_request_stream_state_output_ids(
        request_stream_id=request_stream_id,
        consumed_output_token_ids=tuple(continuation_choice.token_ids or ())[
            :consumed_tokens
        ],
    )
    stop_reason = _think_close_stop_reason(
        finish_reason=str(continuation_choice.finish_reason),
        stop_reason=continuation_choice.stop_reason,
        context_limited=request_budget.context_limited,
        post_think_limited=post_think_limited,
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
        normalized_prefix, normalized_prompt_ids = prepare_steer_generation_prefix(
            executor=executor,
            assistant_prefix=repaired_prefix,
            prompt_token_ids=repaired_prompt_ids,
        )
        return DecodeOutcome(
            event_type="trigger",
            trigger_type="steer_boundary",
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
            await prepare_steer_generation_prefix_async(
                executor=executor,
                assistant_prefix=repaired_prefix,
                prompt_token_ids=repaired_prompt_ids,
            )
        )
        return DecodeOutcome(
            event_type="trigger",
            trigger_type="steer_boundary",
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
    assert updated_text.startswith(original_text), (
        "canonical steer suffix injection must be append-only when "
        "prompt_token_ids are available"
    )
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
    assert updated_text.startswith(original_text), (
        "canonical steer suffix injection must be append-only when "
        "prompt_token_ids are available"
    )
    injected_suffix = updated_text[len(original_text) :]
    assert injected_suffix, "injected suffix expected when text changed"
    suffix_token_ids = executor._tokenize_text(text=injected_suffix)
    executor._assert_text_token_alignment(
        text=injected_suffix,
        token_ids=tuple(suffix_token_ids),
        context="append_injected_suffix_prompt_ids",
    )
    return tuple(prompt_token_ids) + tuple(suffix_token_ids)
