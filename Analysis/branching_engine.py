"""Steer-aware branching engine using vLLM OpenAI-compatible APIs."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

from analysis_types import (
    BranchStep,
    RunArtifactsIndex,
    RunConfig,
    SteerCandidate,
    TokenStat,
)
from chat_templating import (
    build_chat_messages,
    build_chat_template_fields,
    build_raw_im_prompt,
    resolve_mode_after_error,
)
from io_utils import append_jsonl, build_artifacts_index, make_run_id, write_json
from tag_scanner import (
    choose_next_event,
    find_branch_event,
    find_think_end_event,
    first_steer_close_index,
)
from token_metrics import approximate_entropy
from vllm_client import (
    GenerationChoice,
    VllmClient,
    VllmRequestError,
    is_chat_template_error,
)


class BranchingClient(Protocol):
    """Protocol for clients exposing chat and completions methods.

    Args:
        None.

    Returns:
        Structural client protocol for branching helpers.
    """

    def completions(
        self,
        *,
        model: str,
        prompt: str,
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
    ) -> tuple[GenerationChoice, ...]:
        """Generate completion choices with OpenAI-compatible payload args."""
        ...

    def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        template_fields: dict[str, Any],
    ) -> tuple[GenerationChoice, ...]:
        """Generate chat completion choices with OpenAI-compatible payload args."""
        ...


@dataclass(frozen=True)
class RolloutOutcome:
    """Outcome of rolling out until next structural event or termination.

    Args:
        assistant_text: Updated assistant text.
        scan_index: Next scan index in assistant text.
        generated_tokens: Generated token count in this rollout.
        token_stats: Token stats generated during this rollout.
        event_type: Event type (`branch`, `think_end`, or `terminated`).
        termination_reason: Reason when event type is `terminated`.
        active_mode: Actual API mode used after fallback handling.

    Returns:
        Dataclass containing rollout outcome details.
    """

    assistant_text: str
    scan_index: int
    generated_tokens: int
    token_stats: tuple[TokenStat, ...]
    event_type: str
    termination_reason: str
    active_mode: str


@dataclass(frozen=True)
class BranchStepResult:
    """Computed outputs for one branch sampling step.

    Args:
        step_metadata: Summary metadata for current step.
        selected_candidate: Candidate selected for continuation.
        candidates: All sampled candidates.
        token_stats: Token stats for all candidate tokens.

    Returns:
        Dataclass containing outputs for one branch step.
    """

    step_metadata: BranchStep
    selected_candidate: SteerCandidate
    candidates: tuple[SteerCandidate, ...]
    token_stats: tuple[TokenStat, ...]


@dataclass(frozen=True)
class AnalyzerState:
    """Immutable snapshot of iterative run state.

    Args:
        assistant_text: Current assistant continuation text.
        scan_index: Current scan position.
        used_tokens: Total consumed generation tokens.
        active_mode: Current API mode.

    Returns:
        Dataclass for run-state transitions.
    """

    assistant_text: str
    scan_index: int
    used_tokens: int
    active_mode: str


@dataclass
class RolloutCursor:
    """Mutable state used inside one rollout loop.

    Args:
        text: Current assistant text.
        scan_index: Current scan index.
        mode: Active generation mode.
        generated_tokens: Tokens generated in this rollout.
        token_stats: Token metrics collected in this rollout.

    Returns:
        Dataclass used for local rollout updates.
    """

    text: str
    scan_index: int
    mode: str
    generated_tokens: int
    token_stats: list[TokenStat]


def run_branching_analysis(*, config: RunConfig) -> RunArtifactsIndex:
    """Run complete steer-branching workflow and persist artifacts.

    Args:
        config: Run configuration.

    Returns:
        Artifact index describing all written outputs.

    Example:
        >>> cfg = RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p")
        >>> cfg.validate()
    """
    config.validate()
    run_id = make_run_id(prefix="branch")
    artifacts = build_artifacts_index(output_root=config.output_root, run_id=run_id)
    write_json(
        path=artifacts.config_path,
        payload=serialize_config(config=config, run_id=run_id),
    )
    return execute_analysis(config=config, artifacts=artifacts)


def execute_analysis(
    *, config: RunConfig, artifacts: RunArtifactsIndex
) -> RunArtifactsIndex:
    """Execute branch loop and persist step, candidate, and token rows.

    Args:
        config: Runtime configuration.
        artifacts: Canonical artifact paths.

    Returns:
        Same artifact index after run completion.
    """
    client = VllmClient(base_url=config.base_url)
    rng = random.Random(config.seed)
    state = AnalyzerState("", 0, 0, config.api_mode_config.default_mode)
    for step_index in range(config.max_steps):
        state, should_stop = run_step(
            client=client,
            config=config,
            rng=rng,
            artifacts=artifacts,
            state=state,
            step_index=step_index,
        )
        if should_stop:
            break
    write_final_text(
        path=artifacts.run_dir / "final_text.json", assistant_text=state.assistant_text
    )
    return artifacts


def run_step(
    *,
    client: BranchingClient,
    config: RunConfig,
    rng: random.Random,
    artifacts: RunArtifactsIndex,
    state: AnalyzerState,
    step_index: int,
) -> tuple[AnalyzerState, bool]:
    """Run one rollout+branch step and persist outputs.

    Args:
        client: vLLM client.
        config: Runtime config.
        rng: RNG for random branch selection.
        artifacts: Output artifacts.
        state: Current analyzer state.
        step_index: Step index.

    Returns:
        Tuple `(new_state, should_stop)`.
    """
    outcome = rollout_for_step(
        client=client, config=config, state=state, step_index=step_index
    )
    append_token_stats(path=artifacts.token_stats_path, token_stats=outcome.token_stats)
    state = apply_rollout_state(state=state, outcome=outcome)
    if outcome.event_type != "branch":
        return stop_after_rollout(
            artifacts=artifacts,
            step_index=step_index,
            scan_index=state.scan_index,
            outcome=outcome,
            state=state,
        )
    state = run_branch_sampling(
        client=client,
        config=config,
        rng=rng,
        artifacts=artifacts,
        state=state,
        step_index=step_index,
    )
    return stop_if_token_limit(
        artifacts=artifacts,
        step_index=step_index,
        state=state,
        max_total_tokens=config.max_total_tokens,
    )


def rollout_for_step(
    *, client: BranchingClient, config: RunConfig, state: AnalyzerState, step_index: int
) -> RolloutOutcome:
    """Run rollout stage for one step.

    Args:
        client: vLLM API client.
        config: Runtime config.
        state: Current state.
        step_index: Step index.

    Returns:
        Rollout outcome.
    """
    return rollout_until_event(
        client=client, config=config, state=state, step_index=step_index
    )


def stop_after_rollout(
    *,
    artifacts: RunArtifactsIndex,
    step_index: int,
    scan_index: int,
    outcome: RolloutOutcome,
    state: AnalyzerState,
) -> tuple[AnalyzerState, bool]:
    """Record terminal rollout outcome and stop execution.

    Args:
        artifacts: Artifact index.
        step_index: Branch step index.
        scan_index: Current scan index.
        outcome: Rollout outcome.
        state: Current analyzer state.

    Returns:
        Tuple `(state, True)`.
    """
    reason = outcome.termination_reason or outcome.event_type
    record_terminal_step(
        artifacts=artifacts, step_index=step_index, scan_index=scan_index, reason=reason
    )
    return state, True


def run_branch_sampling(
    *,
    client: BranchingClient,
    config: RunConfig,
    rng: random.Random,
    artifacts: RunArtifactsIndex,
    state: AnalyzerState,
    step_index: int,
) -> AnalyzerState:
    """Run branch candidate sampling and update state.

    Args:
        client: vLLM client.
        config: Runtime config.
        rng: RNG for branch selection.
        artifacts: Artifact index.
        state: Current analyzer state.
        step_index: Branch step index.

    Returns:
        Updated analyzer state after selected continuation.
    """
    branch_result = process_branch_step(
        client=client,
        config=config,
        rng=rng,
        assistant_text=state.assistant_text,
        step_index=step_index,
        active_mode=state.active_mode,
    )
    persist_branch_result(artifacts=artifacts, branch_result=branch_result)
    return apply_branch_state(
        state=state, selected_candidate=branch_result.selected_candidate
    )


def stop_if_token_limit(
    *,
    artifacts: RunArtifactsIndex,
    step_index: int,
    state: AnalyzerState,
    max_total_tokens: int,
) -> tuple[AnalyzerState, bool]:
    """Stop run when accumulated token budget is exhausted.

    Args:
        artifacts: Artifact index.
        step_index: Current step index.
        state: Current analyzer state.
        max_total_tokens: Max token budget.

    Returns:
        Tuple `(state, should_stop)`.
    """
    if state.used_tokens < max_total_tokens:
        return state, False
    record_terminal_step(
        artifacts=artifacts,
        step_index=step_index,
        scan_index=state.scan_index,
        reason="max_total_tokens_reached",
    )
    return state, True


def serialize_config(*, config: RunConfig, run_id: str) -> dict[str, object]:
    """Serialize run config into JSON-friendly mapping.

    Args:
        config: Runtime config.
        run_id: Stable run identifier.

    Returns:
        Serialized config mapping.
    """
    payload = asdict(config)
    payload["output_root"] = str(config.output_root)
    payload["run_id"] = run_id
    payload["effective_top_logprobs"] = config.capped_top_logprobs()
    return payload


def rollout_until_event(
    *, client: BranchingClient, config: RunConfig, state: AnalyzerState, step_index: int
) -> RolloutOutcome:
    """Roll out one branch path until next boundary or terminal signal.

    Args:
        client: vLLM API client.
        config: Runtime config.
        state: Current analyzer state.
        step_index: Current branch step index.

    Returns:
        Rollout outcome at next structural event or termination.
    """
    cursor = RolloutCursor(
        text=state.assistant_text,
        scan_index=state.scan_index,
        mode=state.active_mode,
        generated_tokens=0,
        token_stats=[],
    )
    while True:
        outcome = rollout_chunk(
            client=client,
            config=config,
            state=state,
            step_index=step_index,
            cursor=cursor,
        )
        if outcome is not None:
            return outcome


def rollout_chunk(
    *,
    client: BranchingClient,
    config: RunConfig,
    state: AnalyzerState,
    step_index: int,
    cursor: RolloutCursor,
) -> RolloutOutcome | None:
    """Run one rollout chunk and emit outcome when event appears.

    Args:
        client: vLLM API client.
        config: Runtime config.
        state: Persistent analyzer state.
        step_index: Branch step index.
        cursor: Mutable rollout cursor state.

    Returns:
        Rollout outcome for terminal/event states, otherwise `None`.
    """
    budget_outcome = budget_termination_outcome(
        config=config, state=state, cursor=cursor
    )
    if budget_outcome is not None:
        return budget_outcome
    choice, cursor.mode = generate_rollout_choice(
        client=client,
        config=config,
        state=state,
        step_index=step_index,
        cursor=cursor,
    )
    if is_empty_choice(choice=choice):
        return empty_choice_termination(cursor=cursor)
    apply_choice_to_cursor(cursor=cursor, step_index=step_index, choice=choice)
    return scan_cursor_for_event(config=config, cursor=cursor)


def budget_termination_outcome(
    *, config: RunConfig, state: AnalyzerState, cursor: RolloutCursor
) -> RolloutOutcome | None:
    """Return termination outcome when rollout has no remaining token budget.

    Args:
        config: Runtime config.
        state: Persistent analyzer state.
        cursor: Mutable rollout cursor.

    Returns:
        Termination outcome or `None`.
    """
    chunk_tokens = rollout_chunk_budget(
        config=config, state=state, generated_tokens=cursor.generated_tokens
    )
    if chunk_tokens > 0:
        return None
    return terminated_rollout(
        text=cursor.text,
        scan_index=cursor.scan_index,
        generated_tokens=cursor.generated_tokens,
        token_stats=cursor.token_stats,
        active_mode=cursor.mode,
        reason="max_total_tokens_reached",
    )


def generate_rollout_choice(
    *,
    client: BranchingClient,
    config: RunConfig,
    state: AnalyzerState,
    step_index: int,
    cursor: RolloutCursor,
) -> tuple[GenerationChoice, str]:
    """Generate one rollout chunk choice.

    Args:
        client: vLLM API client.
        config: Runtime config.
        state: Persistent analyzer state.
        step_index: Branch step index.
        cursor: Mutable rollout cursor.

    Returns:
        Tuple `(choice, mode)`.
    """
    chunk_tokens = rollout_chunk_budget(
        config=config,
        state=state,
        generated_tokens=cursor.generated_tokens,
    )
    return generate_one(
        client=client,
        config=config,
        assistant_prefix=cursor.text,
        max_tokens=chunk_tokens,
        step_index=step_index,
        candidate_index=-1,
        active_mode=cursor.mode,
    )


def empty_choice_termination(*, cursor: RolloutCursor) -> RolloutOutcome:
    """Build termination outcome for empty rollout generations.

    Args:
        cursor: Mutable rollout cursor.

    Returns:
        Terminated rollout outcome.
    """
    return terminated_rollout(
        text=cursor.text,
        scan_index=cursor.scan_index,
        generated_tokens=cursor.generated_tokens,
        token_stats=cursor.token_stats,
        active_mode=cursor.mode,
        reason="empty_generation",
    )


def apply_choice_to_cursor(
    *, cursor: RolloutCursor, step_index: int, choice: GenerationChoice
) -> None:
    """Apply one generated choice to rollout cursor state.

    Args:
        cursor: Mutable rollout cursor.
        step_index: Branch step index.
        choice: Generated choice.

    Returns:
        None.
    """
    cursor.token_stats.extend(
        choice_token_stats(
            source="rollout",
            step_index=step_index,
            candidate_index=-1,
            choice=choice,
        )
    )
    cursor.text += choice.text
    cursor.generated_tokens += len(choice.tokens)


def scan_cursor_for_event(
    *, config: RunConfig, cursor: RolloutCursor
) -> RolloutOutcome | None:
    """Scan cursor text for branch/terminal tags.

    Args:
        config: Runtime config.
        cursor: Mutable rollout cursor.

    Returns:
        Event outcome or `None`.
    """
    event = find_next_structural_event(
        text=cursor.text,
        scan_index=cursor.scan_index,
        boundary_pattern=config.boundary_pattern,
    )
    if event is None:
        cursor.scan_index = len(cursor.text)
        return None
    return event_rollout(
        text=cursor.text,
        event_end=event.end_index,
        generated_tokens=cursor.generated_tokens,
        token_stats=cursor.token_stats,
        active_mode=cursor.mode,
        event_type=event.event_type,
    )


def rollout_chunk_budget(
    *, config: RunConfig, state: AnalyzerState, generated_tokens: int
) -> int:
    """Compute next rollout chunk token budget.

    Args:
        config: Runtime config.
        state: Current run state.
        generated_tokens: Tokens generated in current rollout.

    Returns:
        Chunk budget for next generation request.
    """
    remaining_tokens = config.max_total_tokens - state.used_tokens - generated_tokens
    if remaining_tokens <= 0:
        return 0
    return min(config.rollout_chunk_tokens, remaining_tokens)


def is_empty_choice(*, choice: GenerationChoice) -> bool:
    """Check whether a generation choice has no useful output.

    Args:
        choice: Parsed generation choice.

    Returns:
        `True` when both text and tokens are empty.
    """
    return not choice.text and not choice.tokens


def event_rollout(
    *,
    text: str,
    event_end: int,
    generated_tokens: int,
    token_stats: list[TokenStat],
    active_mode: str,
    event_type: str,
) -> RolloutOutcome:
    """Build rollout outcome for structural event completion.

    Args:
        text: Current text.
        event_end: Event end index.
        generated_tokens: Generated token count.
        token_stats: Collected token stats.
        active_mode: Active API mode.
        event_type: Event type (`branch` or `think_end`).

    Returns:
        Rollout outcome object.
    """
    # Keep post-</think> text so final-answer content is preserved.
    if event_type == "think_end":
        resolved_text = text
    else:
        resolved_text = text[:event_end]
    return RolloutOutcome(
        resolved_text,
        len(resolved_text),
        generated_tokens,
        tuple(token_stats),
        event_type,
        "",
        active_mode,
    )


def terminated_rollout(
    *,
    text: str,
    scan_index: int,
    generated_tokens: int,
    token_stats: list[TokenStat],
    active_mode: str,
    reason: str,
) -> RolloutOutcome:
    """Build rollout outcome for termination without structural event.

    Args:
        text: Current assistant text.
        scan_index: Current scan index.
        generated_tokens: Generated token count.
        token_stats: Collected token stats.
        active_mode: Active mode.
        reason: Termination reason.

    Returns:
        Terminated rollout outcome.
    """
    return RolloutOutcome(
        text,
        scan_index,
        generated_tokens,
        tuple(token_stats),
        "terminated",
        reason,
        active_mode,
    )


def apply_rollout_state(
    *, state: AnalyzerState, outcome: RolloutOutcome
) -> AnalyzerState:
    """Apply rollout outcome onto persistent analyzer state.

    Args:
        state: Current state.
        outcome: Rollout outcome.

    Returns:
        Updated analyzer state.
    """
    return AnalyzerState(
        assistant_text=outcome.assistant_text,
        scan_index=outcome.scan_index,
        used_tokens=state.used_tokens + outcome.generated_tokens,
        active_mode=outcome.active_mode,
    )


def process_branch_step(
    *,
    client: BranchingClient,
    config: RunConfig,
    rng: random.Random,
    assistant_text: str,
    step_index: int,
    active_mode: str,
) -> BranchStepResult:
    """Sample steer candidates at one branch point and select continuation.

    Args:
        client: vLLM API client.
        config: Runtime config.
        rng: Random number generator.
        assistant_text: Text prefix ending at `<steer>` open tag.
        step_index: Current branch step.
        active_mode: API mode to use.

    Returns:
        Branch-step result with selected candidate and metadata.
    """
    choices, _ = generate_many(
        client=client,
        config=config,
        assistant_prefix=assistant_text,
        max_tokens=config.max_steer_tokens,
        n=config.branch_factor,
        step_index=step_index,
        active_mode=active_mode,
    )
    candidates = tuple(
        make_candidate(step_index=step_index, choice=choice) for choice in choices
    )
    selected_index = rng.randrange(len(candidates))
    selected_candidate = candidates[selected_index]
    metadata = build_branch_step_metadata(
        step_index=step_index,
        assistant_text=assistant_text,
        candidates=candidates,
        selected_index=selected_index,
    )
    token_stats = flatten_candidate_token_stats(step_index=step_index, choices=choices)
    return BranchStepResult(metadata, selected_candidate, candidates, token_stats)


def build_branch_step_metadata(
    *,
    step_index: int,
    assistant_text: str,
    candidates: tuple[SteerCandidate, ...],
    selected_index: int,
) -> BranchStep:
    """Build branch-step summary metadata.

    Args:
        step_index: Branch step index.
        assistant_text: Current assistant prefix.
        candidates: All candidates.
        selected_index: Selected candidate index.

    Returns:
        Branch-step metadata row.
    """
    selected_text = candidates[selected_index].text
    unique_count = len({candidate.text for candidate in candidates})
    return BranchStep(
        step_index=step_index,
        prefix_char_end=len(assistant_text),
        selected_candidate_index=selected_index,
        selected_text=selected_text,
        total_candidates=len(candidates),
        unique_candidate_count=unique_count,
        terminated=False,
        termination_reason="",
    )


def apply_branch_state(
    *, state: AnalyzerState, selected_candidate: SteerCandidate
) -> AnalyzerState:
    """Apply selected candidate continuation to analyzer state.

    Args:
        state: Current state.
        selected_candidate: Selected candidate.

    Returns:
        Updated analyzer state.
    """
    assistant_text = state.assistant_text + selected_candidate.text
    return AnalyzerState(
        assistant_text=assistant_text,
        scan_index=len(assistant_text),
        used_tokens=state.used_tokens + selected_candidate.token_count,
        active_mode=state.active_mode,
    )


def persist_branch_result(
    *, artifacts: RunArtifactsIndex, branch_result: BranchStepResult
) -> None:
    """Persist one branch-step outputs to artifact files.

    Args:
        artifacts: Artifact index.
        branch_result: Branch step outputs.

    Returns:
        None.
    """
    append_jsonl(path=artifacts.steps_path, payload=asdict(branch_result.step_metadata))
    append_candidates(
        path=artifacts.candidates_path, candidates=branch_result.candidates
    )
    append_token_stats(
        path=artifacts.token_stats_path, token_stats=branch_result.token_stats
    )


def write_final_text(*, path: Path, assistant_text: str) -> None:
    """Write final assistant text artifact.

    Args:
        path: Output path.
        assistant_text: Final assistant completion text.

    Returns:
        None.
    """
    write_json(path=path, payload={"assistant_text": assistant_text})


def make_candidate(*, step_index: int, choice: GenerationChoice) -> SteerCandidate:
    """Build `SteerCandidate` from one generation choice.

    Args:
        step_index: Branch step index.
        choice: Parsed generation choice.

    Returns:
        Candidate summary dataclass.
    """
    text, is_closed = trim_to_steer_close(text=choice.text)
    token_count = len(choice.tokens)
    cumulative_logprob = sum(token.logprob for token in choice.tokens)
    average_logprob = cumulative_logprob / token_count if token_count else 0.0
    return SteerCandidate(
        step_index=step_index,
        candidate_index=choice.index,
        text=text,
        token_count=token_count,
        closed_with_tag=is_closed,
        finish_reason=choice.finish_reason,
        cumulative_logprob=cumulative_logprob,
        average_logprob=average_logprob,
    )


def trim_to_steer_close(*, text: str) -> tuple[str, bool]:
    """Trim candidate text after first `</steer>` close tag.

    Args:
        text: Candidate text.

    Returns:
        Tuple `(trimmed_text, closed_with_tag)`.
    """
    close_start = first_steer_close_index(text=text)
    if close_start is None:
        return text, False
    close_end = close_start + len("</steer>")
    return text[:close_end], True


def find_next_structural_event(*, text: str, scan_index: int, boundary_pattern: str):
    """Find next branch or terminal event from scan position.

    Args:
        text: Current assistant text.
        scan_index: Scan offset.
        boundary_pattern: Branch boundary regex.

    Returns:
        Earliest structural scan event or `None`.
    """
    branch_event = find_branch_event(
        text=text, start_index=scan_index, boundary_pattern=boundary_pattern
    )
    think_end_event = find_think_end_event(text=text, start_index=scan_index)
    return choose_next_event(branch_event=branch_event, think_end_event=think_end_event)


def generate_one(
    *,
    client: BranchingClient,
    config: RunConfig,
    assistant_prefix: str,
    max_tokens: int,
    step_index: int,
    candidate_index: int,
    active_mode: str,
) -> tuple[GenerationChoice, str]:
    """Generate a single continuation choice with mode fallback support.

    Args:
        client: vLLM API client.
        config: Runtime config.
        assistant_prefix: Prefix passed as context.
        max_tokens: Max generation tokens.
        step_index: Step index for deterministic seed offset.
        candidate_index: Candidate index for deterministic seed offset.
        active_mode: Preferred API mode.

    Returns:
        Tuple `(choice, resolved_mode)`.
    """
    choices, resolved_mode = generate_many(
        client=client,
        config=config,
        assistant_prefix=assistant_prefix,
        max_tokens=max_tokens,
        n=1,
        step_index=step_index,
        active_mode=active_mode,
        candidate_index_offset=candidate_index,
    )
    return choices[0], resolved_mode


def generate_many(
    *,
    client: BranchingClient,
    config: RunConfig,
    assistant_prefix: str,
    max_tokens: int,
    n: int,
    step_index: int,
    active_mode: str,
    candidate_index_offset: int = 0,
) -> tuple[tuple[GenerationChoice, ...], str]:
    """Generate one or more choices from selected API mode.

    Args:
        client: vLLM API client.
        config: Runtime config.
        assistant_prefix: Prefix passed as context.
        max_tokens: Max generation tokens.
        n: Number of choices.
        step_index: Step index for deterministic seed offset.
        active_mode: Preferred API mode.
        candidate_index_offset: Additional seed offset.

    Returns:
        Tuple `(choices, resolved_mode)`.
    """
    request_seed = build_request_seed(
        config=config,
        step_index=step_index,
        candidate_index_offset=candidate_index_offset,
    )
    if active_mode == "completions":
        choices = call_completions(
            client=client,
            config=config,
            assistant_prefix=assistant_prefix,
            max_tokens=max_tokens,
            n=n,
            request_seed=request_seed,
        )
        return choices, "completions"
    return call_chat_with_fallback(
        client=client,
        config=config,
        assistant_prefix=assistant_prefix,
        max_tokens=max_tokens,
        n=n,
        request_seed=request_seed,
    )


def build_request_seed(
    *, config: RunConfig, step_index: int, candidate_index_offset: int
) -> int:
    """Build deterministic request seed.

    Args:
        config: Runtime config.
        step_index: Step index.
        candidate_index_offset: Candidate offset.

    Returns:
        Deterministic request seed.
    """
    return config.seed + step_index * 997 + candidate_index_offset


def call_chat_with_fallback(
    *,
    client: BranchingClient,
    config: RunConfig,
    assistant_prefix: str,
    max_tokens: int,
    n: int,
    request_seed: int,
) -> tuple[tuple[GenerationChoice, ...], str]:
    """Call chat endpoint and fallback to completions on template errors.

    Args:
        client: vLLM API client.
        config: Runtime config.
        assistant_prefix: Prefix context.
        max_tokens: Max generation tokens.
        n: Number of choices.
        request_seed: Request seed.

    Returns:
        Tuple `(choices, resolved_mode)`.
    """
    try:
        choices = call_chat(
            client=client,
            config=config,
            assistant_prefix=assistant_prefix,
            max_tokens=max_tokens,
            n=n,
            request_seed=request_seed,
        )
        return choices, "chat"
    except VllmRequestError as request_error:
        if not should_fallback_to_completions(
            config=config, error_text=str(request_error)
        ):
            raise
    choices = call_completions(
        client=client,
        config=config,
        assistant_prefix=assistant_prefix,
        max_tokens=max_tokens,
        n=n,
        request_seed=request_seed,
    )
    return choices, "completions"


def should_fallback_to_completions(*, config: RunConfig, error_text: str) -> bool:
    """Decide whether chat errors should trigger completions fallback.

    Args:
        config: Runtime config.
        error_text: Request error body text.

    Returns:
        `True` when fallback should be applied.
    """
    resolved_mode = resolve_mode_after_error(
        preferred_mode="chat",
        allow_fallback=config.api_mode_config.allow_fallback,
        had_chat_template_error=is_chat_template_error(error_text=error_text),
    )
    return resolved_mode == "completions"


def call_completions(
    *,
    client: BranchingClient,
    config: RunConfig,
    assistant_prefix: str,
    max_tokens: int,
    n: int,
    request_seed: int,
) -> tuple[GenerationChoice, ...]:
    """Call completions endpoint with raw `<|im_start|>` prompt template.

    Args:
        client: vLLM API client.
        config: Runtime config.
        assistant_prefix: Assistant prefix context.
        max_tokens: Max generation tokens.
        n: Number of choices.
        request_seed: Request seed.

    Returns:
        Parsed generation choices.
    """
    prompt = build_raw_im_prompt(
        prompt=config.prompt, assistant_prefix=assistant_prefix
    )
    return client.completions(
        model=config.model,
        prompt=prompt,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=max_tokens,
        n=n,
        seed=request_seed,
        stop=None,
        top_logprobs=config.capped_top_logprobs(),
    )


def call_chat(
    *,
    client: BranchingClient,
    config: RunConfig,
    assistant_prefix: str,
    max_tokens: int,
    n: int,
    request_seed: int,
) -> tuple[GenerationChoice, ...]:
    """Call chat endpoint with configured templating fields.

    Args:
        client: vLLM API client.
        config: Runtime config.
        assistant_prefix: Assistant prefix context.
        max_tokens: Max generation tokens.
        n: Number of choices.
        request_seed: Request seed.

    Returns:
        Parsed generation choices.
    """
    template_fields = build_chat_template_fields(template_config=config.template_config)
    messages = build_chat_messages(
        prompt=config.prompt,
        assistant_prefix=assistant_prefix,
        content_format=config.template_config.content_format,
    )
    return client.chat(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=max_tokens,
        n=n,
        seed=request_seed,
        stop=None,
        top_logprobs=config.capped_top_logprobs(),
        template_fields=template_fields,
    )


def flatten_candidate_token_stats(
    *, step_index: int, choices: Iterable[GenerationChoice]
) -> tuple[TokenStat, ...]:
    """Build token stats for all candidate choices in one branch step.

    Args:
        step_index: Branch step index.
        choices: Candidate generation choices.

    Returns:
        Flattened token stats.
    """
    token_stats: list[TokenStat] = []
    for choice in choices:
        token_stats.extend(
            choice_token_stats(
                source="candidate",
                step_index=step_index,
                candidate_index=choice.index,
                choice=choice,
            )
        )
    return tuple(token_stats)


def choice_token_stats(
    *,
    source: str,
    step_index: int,
    candidate_index: int,
    choice: GenerationChoice,
) -> list[TokenStat]:
    """Build token statistics for one generation choice.

    Args:
        source: Source label (`rollout` or `candidate`).
        step_index: Branch step index.
        candidate_index: Candidate index or -1.
        choice: Generation choice.

    Returns:
        Token statistics list.
    """
    token_stats: list[TokenStat] = []
    for token_index, token in enumerate(choice.tokens):
        probability, entropy, alternatives = approximate_entropy(
            selected_token=token.token,
            selected_logprob=token.logprob,
            top_entries=token.top_entries,
        )
        token_stats.append(
            TokenStat(
                source=source,
                step_index=step_index,
                candidate_index=candidate_index,
                token_index=token_index,
                token=token.token,
                logprob=token.logprob,
                probability=probability,
                entropy=entropy,
                alternatives=alternatives,
            )
        )
    return token_stats


def append_candidates(*, path: Path, candidates: Iterable[SteerCandidate]) -> None:
    """Append candidate rows to JSONL path.

    Args:
        path: Output path.
        candidates: Candidate rows.

    Returns:
        None.
    """
    for candidate in candidates:
        append_jsonl(path=path, payload=asdict(candidate))


def append_token_stats(*, path: Path, token_stats: Iterable[TokenStat]) -> None:
    """Append token-stat rows to JSONL path.

    Args:
        path: Output path.
        token_stats: Token statistic rows.

    Returns:
        None.
    """
    for token_stat in token_stats:
        append_jsonl(path=path, payload=serialize_token_stat(token_stat=token_stat))


def serialize_token_stat(*, token_stat: TokenStat) -> dict[str, object]:
    """Serialize token stat with nested alternatives.

    Args:
        token_stat: Token-stat dataclass.

    Returns:
        JSON-friendly token stat mapping.
    """
    payload = asdict(token_stat)
    payload["alternatives"] = [
        asdict(alternative) for alternative in token_stat.alternatives
    ]
    return payload


def record_terminal_step(
    *, artifacts: RunArtifactsIndex, step_index: int, scan_index: int, reason: str
) -> None:
    """Append terminal step row for early exit conditions.

    Args:
        artifacts: Artifact index.
        step_index: Branch step index.
        scan_index: Current scan index.
        reason: Termination reason.

    Returns:
        None.
    """
    terminal_step = BranchStep(
        step_index=step_index,
        prefix_char_end=scan_index,
        selected_candidate_index=-1,
        selected_text="",
        total_candidates=0,
        unique_candidate_count=0,
        terminated=True,
        termination_reason=reason,
    )
    append_jsonl(path=artifacts.steps_path, payload=asdict(terminal_step))
