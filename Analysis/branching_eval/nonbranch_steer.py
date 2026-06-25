"""Helpers for resolving steer triggers without creating child branches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from branching_eval.runtime_types import DecodeOutcome, PathState
from branching_eval.steer_decode_flow import continue_with_single_steer_candidate_async
from branching_eval.tree_types import BranchTree

NonbranchSteerMode = Literal[
    "single_candidate",
    "inline_epsilon",
    "token_budget_exhausted",
]


@dataclass(frozen=True)
class NonbranchSteerContinuationResult:
    """Resolved continuation outcome plus the strategy that produced it."""

    outcome: DecodeOutcome
    mode: NonbranchSteerMode


async def resolve_nonbranch_steer_continuation_async(
    *,
    executor: Any,
    tree: BranchTree,
    state: PathState,
    trigger_outcome: DecodeOutcome,
    branching_enabled: bool,
    inline_epsilon_enabled: bool | None,
) -> NonbranchSteerContinuationResult:
    """Resolve one steer trigger without spawning true branch children."""

    use_inline_epsilon = executor._should_use_inline_epsilon(
        inline_epsilon_enabled=(
            branching_enabled
            if inline_epsilon_enabled is None
            else inline_epsilon_enabled
        )
    )
    if not use_inline_epsilon:
        outcome = await continue_with_single_steer_candidate_async(
            executor=executor,
            assistant_prefix=trigger_outcome.assistant_prefix,
            prompt_token_ids=trigger_outcome.prompt_token_ids,
            token_ids=trigger_outcome.token_ids,
            token_traces=trigger_outcome.token_traces,
            generated_tokens=trigger_outcome.generated_tokens,
            request_stream_id=f"decode:{state.node_id}",
        )
        return NonbranchSteerContinuationResult(
            outcome=outcome,
            mode="single_candidate",
        )

    candidate_token_budget = await executor._candidate_token_budget_async(
        trigger_type="steer_boundary",
        generated_tokens=trigger_outcome.generated_tokens,
    )
    if candidate_token_budget <= 0:
        outcome = DecodeOutcome(
            event_type="terminated",
            trigger_type=None,
            assistant_prefix=trigger_outcome.assistant_prefix,
            prompt_token_ids=trigger_outcome.prompt_token_ids,
            token_ids=trigger_outcome.token_ids,
            token_traces=trigger_outcome.token_traces,
            generated_tokens=trigger_outcome.generated_tokens,
            stop_reason="max_gen_toks_reached",
            branch_points_used=state.branch_points_used,
        )
        return NonbranchSteerContinuationResult(
            outcome=outcome,
            mode="token_budget_exhausted",
        )

    context = executor._require_event_context()
    assert context.doc_id is not None, "inline epsilon continuation requires doc_id"
    pool = await executor._resolve_candidate_pool_async(
        doc_id=context.doc_id,
        state=state,
        trigger_type="steer_boundary",
        assistant_prefix=trigger_outcome.assistant_prefix,
        prompt_token_ids=trigger_outcome.prompt_token_ids,
        generated_tokens=trigger_outcome.generated_tokens,
    )
    outcome = await executor._continue_triggered_state_inline_async(
        tree=tree,
        state=state,
        outcome=trigger_outcome,
        pool=pool,
    )
    return NonbranchSteerContinuationResult(
        outcome=outcome,
        mode="inline_epsilon",
    )
