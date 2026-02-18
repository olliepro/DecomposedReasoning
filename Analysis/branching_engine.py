"""Steer-aware branching engine using completions-only vLLM APIs."""

from __future__ import annotations

import logging
import random
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Protocol

from analysis_types import (
    BranchStep,
    RunArtifactsIndex,
    RunConfig,
    SteerCandidate,
    TokenStat,
)
from chat_templating import build_raw_im_prompt
from io_utils import (
    append_jsonl,
    append_text,
    build_artifacts_index,
    make_run_id,
    write_json,
)
from tag_scanner import first_steer_close_index
from token_metrics import approximate_entropy
from vllm_client import GenerationChoice, VllmClient, VllmRequestError

ROLLOUT_STEER_STOP = ("<steer",)
BRANCH_STEER_STOP = ("</steer>",)
STEER_CLOSE_TAG = "</steer>"
THINK_CLOSE_PATTERN = re.compile(r"</think>", flags=re.IGNORECASE)
THINK_CLOSE_PARTIAL_SUFFIX_PATTERN = re.compile(
    r"</t(?:h(?:i(?:n(?:k)?)?)?)?$",
    flags=re.IGNORECASE,
)
EXEC_OPEN_PATTERN = re.compile(r"<exec\b[^>]*>", flags=re.IGNORECASE)
EXEC_CLOSE_PATTERN = re.compile(r"</exec>", flags=re.IGNORECASE)
EXEC_TO_STEER_PATTERN = re.compile(r"</exec>(\s*)<steer\b", flags=re.IGNORECASE)
STEER_ENTRY_BOUNDARY_PATTERN = re.compile(
    r"(?:<think>|</exec>)\s*<steer>$",
    flags=re.IGNORECASE,
)
REPAIR_TAGS = ("<steer>", "</steer>", "<exec>", "</exec>")
LOGGER = logging.getLogger(__name__)


class BranchingClient(Protocol):
    """Protocol for completions and tokenizer endpoints used by branching.

    Args:
        None.

    Returns:
        Structural protocol for branching helpers.
    """

    def completions(
        self,
        *,
        model: str,
        prompt: str | None,
        prompt_token_ids: tuple[int, ...] | None,
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

    def tokenize(
        self,
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        """Tokenize one text fragment using server tokenizer."""
        ...


@dataclass(frozen=True)
class RolloutOutcome:
    """Outcome of rolling out until a branch boundary or termination.

    Args:
        assistant_text: Updated assistant text.
        scan_index: Next scan index in assistant text.
        generated_tokens: Generated token count in this rollout.
        token_stats: Token stats generated during this rollout.
        event_type: Event type (`branch` or `terminated`).
        termination_reason: Reason when event type is `terminated`.
        prompt_token_ids: Updated prompt token chain.

    Returns:
        Dataclass containing rollout outcome details.
    """

    assistant_text: str
    scan_index: int
    generated_tokens: int
    token_stats: tuple[TokenStat, ...]
    event_type: str
    termination_reason: str
    prompt_token_ids: tuple[int, ...] | None


@dataclass(frozen=True)
class BranchStepResult:
    """Computed outputs for one branch sampling step.

    Args:
        step_metadata: Summary metadata for current step.
        selected_candidate: Candidate selected for continuation.
        selected_token_ids: Token IDs for selected candidate output.
        candidates: All sampled candidates.
        token_stats: Token stats for all candidate tokens.

    Returns:
        Dataclass containing outputs for one branch step.
    """

    step_metadata: BranchStep
    selected_candidate: SteerCandidate
    selected_token_ids: tuple[int, ...]
    candidates: tuple[SteerCandidate, ...]
    token_stats: tuple[TokenStat, ...]


@dataclass(frozen=True)
class AnalyzerState:
    """Immutable snapshot of iterative run state.

    Args:
        assistant_text: Current assistant continuation text.
        scan_index: Current scan position.
        used_tokens: Total consumed generation tokens.
        prompt_token_ids: Prompt token chain used for token-space continuation.

    Returns:
        Dataclass for run-state transitions.
    """

    assistant_text: str
    scan_index: int
    used_tokens: int
    prompt_token_ids: tuple[int, ...] | None


@dataclass
class RolloutCursor:
    """Mutable state used inside one rollout loop.

    Args:
        text: Current assistant text.
        scan_index: Current scan index.
        generated_tokens: Tokens generated in this rollout.
        token_stats: Token metrics collected in this rollout.
        prompt_token_ids: Prompt token chain for token-space requests.

    Returns:
        Dataclass used for local rollout updates.
    """

    text: str
    scan_index: int
    generated_tokens: int
    token_stats: list[TokenStat]
    prompt_token_ids: tuple[int, ...] | None


def log_stage(*, stage: str, **fields: object) -> None:
    """Emit one structured stage log event for rollout tracing.

    Args:
        stage: Stable stage identifier.
        **fields: Structured event fields.

    Returns:
        None.
    """
    if not LOGGER.isEnabledFor(logging.INFO):
        return
    if not fields:
        LOGGER.info("%s", stage)
        return
    parts = [f"{key}={format_log_value(value=value)}" for key, value in fields.items()]
    LOGGER.info("%s %s", stage, " ".join(parts))


def format_log_value(*, value: object) -> str:
    """Format one log field value for compact one-line output.

    Args:
        value: Log value.

    Returns:
        Formatted text suitable for single-line logs.
    """
    if value is None:
        return "None"
    if isinstance(value, str):
        compact = value.replace("\n", "\\n")
        if len(compact) > 80:
            compact = compact[:80] + "..."
        return repr(compact)
    return str(value)


def append_decoded_path_snapshot(
    *,
    path: Path,
    step_index: int,
    phase: str,
    state: AnalyzerState,
) -> None:
    """Append decoded chosen-path state snapshot to a run log file.

    Args:
        path: Chosen-path log file path.
        step_index: Branch step index for this snapshot.
        phase: Stable lifecycle phase label.
        state: Current decoded chosen-path state.

    Returns:
        None.
    """
    header = (
        f"=== step={step_index} phase={phase} used_tokens={state.used_tokens} "
        f"scan_index={state.scan_index} chars={len(state.assistant_text)} ===\n"
    )
    body = state.assistant_text
    if body and not body.endswith("\n"):
        body += "\n"
    append_text(path=path, content=header + body + "\n")


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
    log_stage(
        stage="run.start",
        run_id=run_id,
        model=config.model,
        max_steps=config.max_steps,
        max_total_tokens=config.max_total_tokens,
        rollout_chunk_tokens=config.rollout_chunk_tokens,
        branch_factor=config.branch_factor,
        output_dir=str(artifacts.run_dir),
        chosen_path_log_path=str(artifacts.chosen_path_log_path),
    )
    completed_artifacts = execute_analysis(config=config, artifacts=artifacts)
    log_stage(
        stage="run.complete",
        run_id=run_id,
        final_text_path=str(completed_artifacts.run_dir / "final_text.json"),
    )
    return completed_artifacts


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
    state = AnalyzerState(
        assistant_text="",
        scan_index=0,
        used_tokens=0,
        prompt_token_ids=None,
    )
    append_decoded_path_snapshot(
        path=artifacts.chosen_path_log_path,
        step_index=-1,
        phase="initial",
        state=state,
    )
    for step_index in range(config.max_steps):
        log_stage(
            stage="step.start",
            step_index=step_index,
            used_tokens=state.used_tokens,
            scan_index=state.scan_index,
            prompt_token_chain_len=(
                len(state.prompt_token_ids) if state.prompt_token_ids is not None else 0
            ),
        )
        state, should_stop = run_step(
            client=client,
            config=config,
            rng=rng,
            artifacts=artifacts,
            state=state,
            step_index=step_index,
        )
        if should_stop:
            log_stage(
                stage="step.stop",
                step_index=step_index,
                used_tokens=state.used_tokens,
            )
            break
    write_final_text(
        path=artifacts.run_dir / "final_text.json",
        assistant_text=state.assistant_text,
    )
    log_stage(
        stage="analysis.complete",
        used_tokens=state.used_tokens,
        final_characters=len(state.assistant_text),
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
    log_stage(
        stage="rollout.start",
        step_index=step_index,
        used_tokens=state.used_tokens,
    )
    outcome = rollout_for_step(
        client=client,
        config=config,
        state=state,
        step_index=step_index,
    )
    log_stage(
        stage="rollout.result",
        step_index=step_index,
        event_type=outcome.event_type,
        termination_reason=outcome.termination_reason or "none",
        generated_tokens=outcome.generated_tokens,
    )
    append_token_stats(path=artifacts.token_stats_path, token_stats=outcome.token_stats)
    state = apply_rollout_state(state=state, outcome=outcome)
    append_decoded_path_snapshot(
        path=artifacts.chosen_path_log_path,
        step_index=step_index,
        phase="after_rollout",
        state=state,
    )
    if outcome.event_type != "branch":
        return stop_after_rollout(
            artifacts=artifacts,
            step_index=step_index,
            scan_index=state.scan_index,
            reason=outcome.termination_reason,
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
    append_decoded_path_snapshot(
        path=artifacts.chosen_path_log_path,
        step_index=step_index,
        phase="after_branch_selection",
        state=state,
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
        client=client,
        config=config,
        state=state,
        step_index=step_index,
    )


def stop_after_rollout(
    *,
    artifacts: RunArtifactsIndex,
    step_index: int,
    scan_index: int,
    reason: str,
    state: AnalyzerState,
) -> tuple[AnalyzerState, bool]:
    """Record terminal rollout outcome and stop execution.

    Args:
        artifacts: Artifact index.
        step_index: Branch step index.
        scan_index: Current scan index.
        reason: Termination reason.
        state: Current analyzer state.

    Returns:
        Tuple `(state, True)`.
    """
    resolved_reason = reason or "terminated"
    log_stage(
        stage="analysis.terminate",
        step_index=step_index,
        reason=resolved_reason,
        scan_index=scan_index,
    )
    record_terminal_step(
        artifacts=artifacts,
        step_index=step_index,
        scan_index=scan_index,
        reason=resolved_reason,
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
        prompt_token_ids=state.prompt_token_ids,
        step_index=step_index,
    )
    branch_result = normalize_selected_candidate_for_execution(
        client=client,
        config=config,
        branch_result=branch_result,
    )
    log_stage(
        stage="branch.selected",
        step_index=step_index,
        selected_index=branch_result.step_metadata.selected_candidate_index,
        total_candidates=branch_result.step_metadata.total_candidates,
        unique_candidates=branch_result.step_metadata.unique_candidate_count,
        selected_preview=branch_result.selected_candidate.preview(limit=100),
    )
    persist_branch_result(artifacts=artifacts, branch_result=branch_result)
    return apply_branch_state(
        state=state,
        selected_candidate=branch_result.selected_candidate,
        selected_token_ids=branch_result.selected_token_ids,
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
    log_stage(
        stage="analysis.terminate",
        step_index=step_index,
        reason="max_total_tokens_reached",
        used_tokens=state.used_tokens,
    )
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
    payload["mode"] = "completions"
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
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=state.prompt_token_ids,
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
        config=config,
        state=state,
        cursor=cursor,
    )
    if budget_outcome is not None:
        log_stage(
            stage="rollout.chunk.terminate",
            step_index=step_index,
            reason=budget_outcome.termination_reason,
        )
        return budget_outcome
    chunk_tokens = rollout_chunk_budget(
        config=config,
        state=state,
        generated_tokens=cursor.generated_tokens,
    )
    remaining_budget = (
        config.max_total_tokens - state.used_tokens - cursor.generated_tokens
    )
    log_stage(
        stage="rollout.chunk.request",
        step_index=step_index,
        max_tokens=chunk_tokens,
        remaining_budget=remaining_budget,
    )
    choice = generate_rollout_choice(
        client=client,
        config=config,
        cursor=cursor,
        max_tokens=chunk_tokens,
        step_index=step_index,
    )
    log_stage(
        stage="rollout.chunk.response",
        step_index=step_index,
        finish_reason=choice.finish_reason,
        stop_reason=choice.stop_reason,
        output_tokens=len(choice.tokens),
        output_chars=len(choice.text),
        text=choice.text,
    )
    if is_empty_choice(choice=choice):
        return empty_choice_termination(cursor=cursor)
    apply_choice_to_cursor(cursor=cursor, step_index=step_index, choice=choice)
    if contains_think_close_or_partial(text=choice.text):
        if is_natural_finish_reason(
            finish_reason=choice.finish_reason,
            stop_reason=choice.stop_reason,
        ) and contains_think_close(text=choice.text):
            return terminated_rollout(
                text=cursor.text,
                scan_index=len(cursor.text),
                generated_tokens=cursor.generated_tokens,
                token_stats=cursor.token_stats,
                prompt_token_ids=cursor.prompt_token_ids,
                reason="think_end",
            )
        return continue_after_think_close(
            client=client,
            config=config,
            state=state,
            step_index=step_index,
            cursor=cursor,
        )
    if choice.finish_reason == "stop":
        if is_natural_finish_reason(
            finish_reason=choice.finish_reason,
            stop_reason=choice.stop_reason,
        ):
            return terminated_rollout(
                text=cursor.text,
                scan_index=len(cursor.text),
                generated_tokens=cursor.generated_tokens,
                token_stats=cursor.token_stats,
                prompt_token_ids=cursor.prompt_token_ids,
                reason="model_finished",
            )
        return stop_finished_outcome(
            client=client,
            config=config,
            cursor=cursor,
            choice=choice,
        )
    if choice.finish_reason == "length":
        return length_finished_outcome(
            client=client,
            config=config,
            state=state,
            step_index=step_index,
            cursor=cursor,
            choice=choice,
        )
    return terminated_rollout(
        text=cursor.text,
        scan_index=len(cursor.text),
        generated_tokens=cursor.generated_tokens,
        token_stats=cursor.token_stats,
        prompt_token_ids=cursor.prompt_token_ids,
        reason="model_finished",
    )


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
        config=config,
        state=state,
        generated_tokens=cursor.generated_tokens,
    )
    if chunk_tokens > 0:
        return None
    return terminated_rollout(
        text=cursor.text,
        scan_index=cursor.scan_index,
        generated_tokens=cursor.generated_tokens,
        token_stats=cursor.token_stats,
        prompt_token_ids=cursor.prompt_token_ids,
        reason="max_total_tokens_reached",
    )


def generate_rollout_choice(
    *,
    client: BranchingClient,
    config: RunConfig,
    cursor: RolloutCursor,
    max_tokens: int,
    step_index: int,
) -> GenerationChoice:
    """Generate one rollout chunk choice.

    Args:
        client: vLLM API client.
        config: Runtime config.
        cursor: Mutable rollout cursor.
        max_tokens: Chunk token budget.
        step_index: Branch step index.

    Returns:
        One generated choice.
    """
    return generate_one(
        client=client,
        config=config,
        assistant_prefix=cursor.text,
        prompt_token_ids=cursor.prompt_token_ids,
        max_tokens=max_tokens,
        step_index=step_index,
        candidate_index=-1,
        stop=ROLLOUT_STEER_STOP,
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
        prompt_token_ids=cursor.prompt_token_ids,
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
    cursor.scan_index = len(cursor.text)
    cursor.prompt_token_ids = updated_prompt_token_ids(
        current_prompt_token_ids=cursor.prompt_token_ids,
        choice=choice,
    )


def updated_prompt_token_ids(
    *,
    current_prompt_token_ids: tuple[int, ...] | None,
    choice: GenerationChoice,
) -> tuple[int, ...]:
    """Append generated token IDs to current token-chain prompt state.

    Args:
        current_prompt_token_ids: Existing prompt token chain.
        choice: Completion choice containing prompt/output token IDs.

    Returns:
        Updated prompt token chain.
    """
    base_prompt_token_ids = current_prompt_token_ids
    if base_prompt_token_ids is None:
        assert (
            choice.prompt_token_ids is not None
        ), "missing prompt_token_ids in response"
        base_prompt_token_ids = choice.prompt_token_ids
    assert choice.token_ids is not None, "missing token_ids in response"
    return tuple(base_prompt_token_ids) + tuple(choice.token_ids)


def stop_finished_outcome(
    *,
    client: BranchingClient,
    config: RunConfig,
    cursor: RolloutCursor,
    choice: GenerationChoice,
) -> RolloutOutcome:
    """Resolve rollout outcome for stop-sequence finishes.

    Args:
        client: vLLM client.
        config: Runtime config.
        cursor: Mutable rollout cursor.
        choice: Generated choice.

    Returns:
        Branch rollout outcome.
    """
    log_stage(
        stage="rollout.chunk.stop",
        stop_reason=choice.stop_reason,
    )
    assert choice.stop_reason is not None, "explicit stop boundary requires stop_reason"
    assert is_steer_stop_reason(
        stop_reason=choice.stop_reason
    ), "unexpected stop_reason"
    ensure_canonical_steer_open(client=client, config=config, cursor=cursor)
    return branch_rollout(cursor=cursor)


def is_steer_stop_reason(*, stop_reason: int | str) -> bool:
    """Return whether stop reason corresponds to steer-stop boundary.

    Args:
        stop_reason: Parsed stop reason from vLLM.

    Returns:
        `True` when stop reason maps to `<steer` stop condition.
    """
    if isinstance(stop_reason, int):
        return True
    return "<steer" in stop_reason.lower()


def ensure_canonical_steer_open(
    *, client: BranchingClient, config: RunConfig, cursor: RolloutCursor
) -> None:
    """Ensure cursor ends at canonical `<steer>` boundary with closed `<exec>`.

    Args:
        client: vLLM client.
        config: Runtime config.
        cursor: Mutable rollout cursor.

    Returns:
        None.
    """
    if is_inside_open_exec(text=cursor.text):
        ensure_exec_closed_before_steer_boundary(
            client=client,
            config=config,
            cursor=cursor,
        )
        return
    if cursor.text.endswith("<steer"):
        log_stage(stage="rollout.boundary.canonical_steer", action="complete_partial")
        append_injected_text_to_cursor(
            client=client,
            config=config,
            cursor=cursor,
            text=">",
        )
    elif not cursor.text.endswith("<steer>"):
        log_stage(stage="rollout.boundary.canonical_steer", action="inject_full_tag")
        append_injected_text_to_cursor(
            client=client,
            config=config,
            cursor=cursor,
            text="<steer>",
        )
    else:
        log_stage(stage="rollout.boundary.canonical_steer", action="already_complete")
    ensure_valid_steer_entry_boundary(client=client, config=config, cursor=cursor)


def ensure_exec_closed_before_steer_boundary(
    *, client: BranchingClient, config: RunConfig, cursor: RolloutCursor
) -> None:
    """Ensure open `<exec>` is closed before final boundary `<steer>` tag.

    Args:
        client: vLLM client.
        config: Runtime config.
        cursor: Mutable rollout cursor.

    Returns:
        None.
    """
    if cursor.text.endswith("<steer"):
        append_injected_text_to_cursor(
            client=client,
            config=config,
            cursor=cursor,
            text=">",
        )
    if cursor.text.endswith("<steer>"):
        separator = inferred_exec_to_steer_separator(text=cursor.text)
        suffix = f"</steer></exec>{separator}<steer>"
        log_stage(
            stage="rollout.boundary.exec_close",
            action="close_exec_after_existing_steer_open",
            suffix=suffix,
        )
        append_injected_text_to_cursor(
            client=client,
            config=config,
            cursor=cursor,
            text=suffix,
        )
        return
    suffix = forced_boundary_suffix(text=cursor.text)
    log_stage(
        stage="rollout.boundary.exec_close",
        action="force_exec_close_then_steer",
        suffix=suffix,
    )
    append_injected_text_to_cursor(
        client=client,
        config=config,
        cursor=cursor,
        text=suffix,
    )


def ensure_valid_steer_entry_boundary(
    *, client: BranchingClient, config: RunConfig, cursor: RolloutCursor
) -> None:
    """Ensure trailing steer-entry is `(<think>|</exec>)<steer>`.

    Args:
        client: vLLM client.
        config: Runtime config.
        cursor: Mutable rollout cursor.

    Returns:
        None.
    """
    assert cursor.text.endswith(
        "<steer>"
    ), "expected trailing <steer> for boundary check"
    if has_valid_steer_entry_boundary(text=cursor.text):
        return
    separator = inferred_exec_to_steer_separator(text=cursor.text)
    suffix = f"</steer><exec></exec>{separator}<steer>"
    log_stage(
        stage="rollout.boundary.repair_entry",
        suffix=suffix,
    )
    append_injected_text_to_cursor(
        client=client,
        config=config,
        cursor=cursor,
        text=suffix,
    )


def has_valid_steer_entry_boundary(*, text: str) -> bool:
    """Return whether text ends with a valid steer-entry boundary.

    Args:
        text: Current assistant text.

    Returns:
        `True` when trailing boundary matches `(<think>|</exec>)<steer>`.
    """
    return STEER_ENTRY_BOUNDARY_PATTERN.search(text) is not None


def length_finished_outcome(
    *,
    client: BranchingClient,
    config: RunConfig,
    state: AnalyzerState,
    step_index: int,
    cursor: RolloutCursor,
    choice: GenerationChoice,
) -> RolloutOutcome:
    """Resolve rollout outcome for max-token chunk boundaries.

    Args:
        client: vLLM client.
        config: Runtime config.
        state: Persistent analyzer state.
        step_index: Branch step index.
        cursor: Mutable rollout cursor.
        choice: Generated chunk choice.

    Returns:
        Rollout outcome for this boundary condition.
    """
    repaired_tag = repair_trailing_partial_tag(
        client=client,
        config=config,
        cursor=cursor,
    )
    log_stage(
        stage="rollout.length",
        repaired_tag=repaired_tag,
        has_think_marker=contains_think_close_or_partial(text=choice.text),
    )
    if repaired_tag == "<steer>":
        ensure_canonical_steer_open(client=client, config=config, cursor=cursor)
        log_stage(stage="rollout.length.branch", reason="repaired_steer_tag")
        return branch_rollout(cursor=cursor)
    if contains_think_close_or_partial(text=choice.text):
        return continue_after_think_close(
            client=client,
            config=config,
            state=state,
            step_index=step_index,
            cursor=cursor,
        )
    force_text = forced_boundary_suffix(text=cursor.text)
    log_stage(
        stage="rollout.length.force_boundary",
        force_text=force_text,
    )
    append_injected_text_to_cursor(
        client=client,
        config=config,
        cursor=cursor,
        text=force_text,
    )
    return branch_rollout(cursor=cursor)


def repair_trailing_partial_tag(
    *, client: BranchingClient, config: RunConfig, cursor: RolloutCursor
) -> str | None:
    """Repair trailing partial steer/exec tags by minimal suffix completion.

    Args:
        client: vLLM client.
        config: Runtime config.
        cursor: Mutable rollout cursor.

    Returns:
        Completed tag string when repaired, else `None`.
    """
    partial = trailing_partial_tag_suffix(text=cursor.text)
    if partial is None:
        return None
    suffix_text, completed_tag = partial
    missing_suffix = completed_tag[len(suffix_text) :]
    log_stage(
        stage="rollout.length.repair_tag",
        partial_suffix=suffix_text,
        completed_tag=completed_tag,
        injected_suffix=missing_suffix,
    )
    append_injected_text_to_cursor(
        client=client,
        config=config,
        cursor=cursor,
        text=missing_suffix,
    )
    return completed_tag


def trailing_partial_tag_suffix(*, text: str) -> tuple[str, str] | None:
    """Find a trailing partial steer/exec tag suffix requiring completion.

    Args:
        text: Current assistant text.

    Returns:
        Tuple `(suffix, completed_tag)` when repairable, else `None`.
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
    """Build unique partial-prefix mapping for steer/exec tags.

    Args:
        None.

    Returns:
        Mapping from unique partial lowercase prefix to completed lowercase tag.
    """
    prefix_map: dict[str, str] = {}
    ambiguous_prefixes: set[str] = set()
    for tag in REPAIR_TAGS:
        lower_tag = tag.lower()
        for prefix_length in range(2, len(lower_tag)):
            prefix = lower_tag[:prefix_length]
            existing = prefix_map.get(prefix)
            if existing is None:
                prefix_map[prefix] = lower_tag
                continue
            if existing != lower_tag:
                ambiguous_prefixes.add(prefix)
    for prefix in ambiguous_prefixes:
        prefix_map.pop(prefix, None)
    # Single "<" is ambiguous by construction; prefer canonical steer-open repair.
    prefix_map["<"] = "<steer>"
    return prefix_map


def contains_think_close(*, text: str) -> bool:
    """Return whether text chunk contains a closing `</think>` tag.

    Args:
        text: Text chunk.

    Returns:
        `True` when text includes a close-think marker.
    """
    return THINK_CLOSE_PATTERN.search(text) is not None


def contains_think_close_or_partial(*, text: str) -> bool:
    """Return whether a text chunk contains complete or partial think-close marker.

    Args:
        text: Text chunk.

    Returns:
        `True` when chunk has `</think>` or ends with a partial close-think prefix.
    """
    if contains_think_close(text=text):
        return True
    return THINK_CLOSE_PARTIAL_SUFFIX_PATTERN.search(text) is not None


def is_natural_finish_reason(
    *, finish_reason: str, stop_reason: int | str | None
) -> bool:
    """Return whether finish reason represents natural model completion.

    Args:
        finish_reason: Parsed finish reason string.
        stop_reason: Parsed stop reason field from vLLM response.

    Returns:
        `True` for natural completion, including EOS-style `"stop"` with no reason.
    """
    if finish_reason == "length":
        return False
    if finish_reason != "stop":
        return True
    return stop_reason is None


def continue_after_think_close(
    *,
    client: BranchingClient,
    config: RunConfig,
    state: AnalyzerState,
    step_index: int,
    cursor: RolloutCursor,
) -> RolloutOutcome:
    """After `</think>`, continue with no stops until natural completion/budget end.

    Args:
        client: vLLM client.
        config: Runtime config.
        state: Persistent analyzer state.
        step_index: Branch step index.
        cursor: Mutable rollout cursor.

    Returns:
        Terminated rollout outcome after no-stop continuation.
    """
    remaining_budget = (
        config.max_total_tokens - state.used_tokens - cursor.generated_tokens
    )
    if remaining_budget <= 0:
        return terminated_rollout(
            text=cursor.text,
            scan_index=len(cursor.text),
            generated_tokens=cursor.generated_tokens,
            token_stats=cursor.token_stats,
            prompt_token_ids=cursor.prompt_token_ids,
            reason="max_total_tokens_reached",
        )
    log_stage(
        stage="rollout.think_close.continue",
        remaining_budget=remaining_budget,
    )
    continuation_choice = generate_one(
        client=client,
        config=config,
        assistant_prefix=cursor.text,
        prompt_token_ids=cursor.prompt_token_ids,
        max_tokens=remaining_budget,
        step_index=step_index,
        candidate_index=-2,
        stop=None,
    )
    if is_empty_choice(choice=continuation_choice):
        return terminated_rollout(
            text=cursor.text,
            scan_index=len(cursor.text),
            generated_tokens=cursor.generated_tokens,
            token_stats=cursor.token_stats,
            prompt_token_ids=cursor.prompt_token_ids,
            reason="empty_generation",
        )
    apply_choice_to_cursor(
        cursor=cursor,
        step_index=step_index,
        choice=continuation_choice,
    )
    termination_reason = (
        "max_total_tokens_reached"
        if continuation_choice.finish_reason == "length"
        else "think_end"
    )
    return terminated_rollout(
        text=cursor.text,
        scan_index=len(cursor.text),
        generated_tokens=cursor.generated_tokens,
        token_stats=cursor.token_stats,
        prompt_token_ids=cursor.prompt_token_ids,
        reason=termination_reason,
    )


def forced_boundary_suffix(*, text: str) -> str:
    """Build forced branch boundary suffix at a length boundary.

    Args:
        text: Current assistant text.

    Returns:
        Text to append for forcing the next `<steer>` boundary.
    """
    if text.endswith("<steer>"):
        return ""
    if is_inside_open_exec(text=text):
        separator = inferred_exec_to_steer_separator(text=text)
        if text.endswith("</exec>"):
            return f"{separator}<steer>"
        return f"</exec>{separator}<steer>"
    return "<steer>"


def is_inside_open_exec(*, text: str) -> bool:
    """Return whether text is currently inside an unclosed `<exec>` block.

    Args:
        text: Current assistant text.

    Returns:
        `True` when open `<exec>` tags outnumber close tags.
    """
    open_count = len(EXEC_OPEN_PATTERN.findall(text))
    close_count = len(EXEC_CLOSE_PATTERN.findall(text))
    return open_count > close_count


def inferred_exec_to_steer_separator(*, text: str) -> str:
    """Infer preferred whitespace separator between `</exec>` and `<steer>`.

    Args:
        text: Current assistant text.

    Returns:
        Separator string. Defaults to `\n\n` when not observed yet.
    """
    matches = list(EXEC_TO_STEER_PATTERN.finditer(text))
    if not matches:
        return "\n\n"
    separator = matches[-1].group(1)
    return separator if separator is not None else "\n\n"


def append_injected_text_to_cursor(
    *,
    client: BranchingClient,
    config: RunConfig,
    cursor: RolloutCursor,
    text: str,
) -> None:
    """Append forced text and corresponding token IDs to cursor state.

    Args:
        client: vLLM client.
        config: Runtime config.
        cursor: Mutable rollout cursor.
        text: Injected text.

    Returns:
        None.
    """
    if not text:
        return
    assert (
        cursor.prompt_token_ids is not None
    ), "prompt_token_ids missing before injection"
    injected_token_ids = client.tokenize(
        model=config.model,
        text=text,
        add_special_tokens=False,
    )
    log_stage(
        stage="rollout.inject",
        text=text,
        token_count=len(injected_token_ids),
    )
    cursor.text += text
    cursor.scan_index = len(cursor.text)
    cursor.prompt_token_ids = tuple(cursor.prompt_token_ids) + tuple(injected_token_ids)


def branch_rollout(*, cursor: RolloutCursor) -> RolloutOutcome:
    """Build rollout outcome for a branch boundary event.

    Args:
        cursor: Mutable rollout cursor.

    Returns:
        Branch rollout outcome.
    """
    return RolloutOutcome(
        assistant_text=cursor.text,
        scan_index=len(cursor.text),
        generated_tokens=cursor.generated_tokens,
        token_stats=tuple(cursor.token_stats),
        event_type="branch",
        termination_reason="",
        prompt_token_ids=cursor.prompt_token_ids,
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


def terminated_rollout(
    *,
    text: str,
    scan_index: int,
    generated_tokens: int,
    token_stats: list[TokenStat],
    prompt_token_ids: tuple[int, ...] | None,
    reason: str,
) -> RolloutOutcome:
    """Build rollout outcome for termination without a branch event.

    Args:
        text: Current assistant text.
        scan_index: Current scan index.
        generated_tokens: Generated token count.
        token_stats: Collected token stats.
        prompt_token_ids: Prompt token chain.
        reason: Termination reason.

    Returns:
        Terminated rollout outcome.
    """
    return RolloutOutcome(
        assistant_text=text,
        scan_index=scan_index,
        generated_tokens=generated_tokens,
        token_stats=tuple(token_stats),
        event_type="terminated",
        termination_reason=reason,
        prompt_token_ids=prompt_token_ids,
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
        prompt_token_ids=outcome.prompt_token_ids,
    )


def process_branch_step(
    *,
    client: BranchingClient,
    config: RunConfig,
    rng: random.Random,
    assistant_text: str,
    prompt_token_ids: tuple[int, ...] | None,
    step_index: int,
) -> BranchStepResult:
    """Sample steer candidates at one branch point and select continuation.

    Args:
        client: vLLM API client.
        config: Runtime config.
        rng: Random number generator.
        assistant_text: Text prefix ending at `<steer>` open tag.
        prompt_token_ids: Prefix token IDs matching `assistant_text`.
        step_index: Current branch step.

    Returns:
        Branch-step result with selected candidate and metadata.
    """
    choices = generate_many(
        client=client,
        config=config,
        assistant_prefix=assistant_text,
        prompt_token_ids=prompt_token_ids,
        max_tokens=config.max_steer_tokens,
        n=config.branch_factor,
        step_index=step_index,
        stop=BRANCH_STEER_STOP,
    )
    candidates = tuple(
        make_candidate(step_index=step_index, choice=choice) for choice in choices
    )
    selected_index = rng.randrange(len(candidates))
    selected_candidate = candidates[selected_index]
    selected_choice = choices[selected_index]
    assert (
        selected_choice.token_ids is not None
    ), "missing token_ids for selected choice"
    metadata = build_branch_step_metadata(
        step_index=step_index,
        assistant_text=assistant_text,
        candidates=candidates,
        selected_index=selected_index,
    )
    token_stats = flatten_candidate_token_stats(step_index=step_index, choices=choices)
    return BranchStepResult(
        step_metadata=metadata,
        selected_candidate=selected_candidate,
        selected_token_ids=selected_choice.token_ids,
        candidates=candidates,
        token_stats=token_stats,
    )


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
    *,
    state: AnalyzerState,
    selected_candidate: SteerCandidate,
    selected_token_ids: tuple[int, ...],
) -> AnalyzerState:
    """Apply selected candidate continuation to analyzer state.

    Args:
        state: Current state.
        selected_candidate: Selected candidate.
        selected_token_ids: Selected candidate output token IDs.

    Returns:
        Updated analyzer state.
    """
    assistant_text = state.assistant_text + selected_candidate.text
    assert (
        state.prompt_token_ids is not None
    ), "missing prompt_token_ids before branch apply"
    prompt_token_ids = tuple(state.prompt_token_ids) + tuple(selected_token_ids)
    return AnalyzerState(
        assistant_text=assistant_text,
        scan_index=len(assistant_text),
        used_tokens=state.used_tokens + selected_candidate.token_count,
        prompt_token_ids=prompt_token_ids,
    )


def normalize_selected_candidate_for_execution(
    *,
    client: BranchingClient,
    config: RunConfig,
    branch_result: BranchStepResult,
) -> BranchStepResult:
    """Ensure selected candidate ends with canonical `</steer>\n`.

    Args:
        client: vLLM client.
        config: Runtime config.
        branch_result: Raw branch-step result.

    Returns:
        Branch result with selected candidate guaranteed closed with `</steer>\n`.
    """
    selected_candidate = branch_result.selected_candidate
    selected_index = branch_result.step_metadata.selected_candidate_index
    injected_text, normalization_mode = selected_candidate_normalization_suffix(
        text=selected_candidate.text
    )
    requires_update = bool(injected_text) or not selected_candidate.closed_with_tag
    if not requires_update:
        log_stage(
            stage="branch.normalize_selected",
            selected_index=selected_index,
            normalization_mode=normalization_mode,
            injected_suffix=injected_text,
            injected_token_count=0,
        )
        return branch_result
    suffix_token_ids: tuple[int, ...] = ()
    if injected_text:
        suffix_token_ids = client.tokenize(
            model=config.model,
            text=injected_text,
            add_special_tokens=False,
        )
    updated_selected = replace(
        selected_candidate,
        text=selected_candidate.text + injected_text,
        closed_with_tag=True,
    )
    updated_candidates = replace_candidate_at_index(
        candidates=branch_result.candidates,
        index=selected_index,
        candidate=updated_selected,
    )
    updated_metadata = replace(
        branch_result.step_metadata,
        selected_text=updated_selected.text,
    )
    updated_token_ids = tuple(branch_result.selected_token_ids) + suffix_token_ids
    log_stage(
        stage="branch.normalize_selected",
        selected_index=selected_index,
        normalization_mode=normalization_mode,
        injected_suffix=injected_text,
        injected_token_count=len(suffix_token_ids),
    )
    return replace(
        branch_result,
        step_metadata=updated_metadata,
        selected_candidate=updated_selected,
        selected_token_ids=updated_token_ids,
        candidates=updated_candidates,
    )


def selected_candidate_normalization_suffix(*, text: str) -> tuple[str, str]:
    """Build minimal injected suffix so selected candidate ends with `</steer>\n`.

    Args:
        text: Selected candidate text before normalization.

    Returns:
        Tuple `(injected_text, normalization_mode)`.
    """
    close_suffix, normalization_mode = selected_candidate_close_completion_suffix(
        text=text
    )
    normalized_text = text + close_suffix
    newline_suffix = "" if normalized_text.endswith("\n") else "\n"
    return close_suffix + newline_suffix, normalization_mode


def selected_candidate_close_completion_suffix(*, text: str) -> tuple[str, str]:
    """Return minimal close-tag completion suffix for selected candidate text.

    Args:
        text: Selected candidate text before normalization.

    Returns:
        Tuple `(close_suffix, normalization_mode)`.
    """
    if text.endswith(STEER_CLOSE_TAG) or text.endswith(f"{STEER_CLOSE_TAG}\n"):
        return "", "already_closed"
    lowered = text.lower()
    lowered_close_tag = STEER_CLOSE_TAG
    for prefix_length in range(len(lowered_close_tag) - 1, 0, -1):
        if lowered.endswith(lowered_close_tag[:prefix_length]):
            return lowered_close_tag[prefix_length:], "partial_completed"
    return STEER_CLOSE_TAG, "full_close_appended"


def replace_candidate_at_index(
    *,
    candidates: tuple[SteerCandidate, ...],
    index: int,
    candidate: SteerCandidate,
) -> tuple[SteerCandidate, ...]:
    """Replace one candidate in tuple while preserving order.

    Args:
        candidates: Candidate tuple.
        index: Candidate index to replace.
        candidate: Replacement candidate.

    Returns:
        Updated candidate tuple.
    """
    updated: list[SteerCandidate] = list(candidates)
    updated[index] = candidate
    return tuple(updated)


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
        path=artifacts.token_stats_path,
        token_stats=branch_result.token_stats,
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


def generate_one(
    *,
    client: BranchingClient,
    config: RunConfig,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    max_tokens: int,
    step_index: int,
    candidate_index: int,
    stop: tuple[str, ...] | None,
) -> GenerationChoice:
    """Generate a single completion choice.

    Args:
        client: vLLM API client.
        config: Runtime config.
        assistant_prefix: Prefix passed as context.
        prompt_token_ids: Prompt token chain for token-space requests.
        max_tokens: Max generation tokens.
        step_index: Step index for deterministic seed offset.
        candidate_index: Candidate index for deterministic seed offset.
        stop: Optional stop sequences for generation termination.

    Returns:
        One generation choice.
    """
    choices = generate_many(
        client=client,
        config=config,
        assistant_prefix=assistant_prefix,
        prompt_token_ids=prompt_token_ids,
        max_tokens=max_tokens,
        n=1,
        step_index=step_index,
        candidate_index_offset=candidate_index,
        stop=stop,
    )
    return choices[0]


def generate_many(
    *,
    client: BranchingClient,
    config: RunConfig,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    max_tokens: int,
    n: int,
    step_index: int,
    candidate_index_offset: int = 0,
    stop: tuple[str, ...] | None = None,
) -> tuple[GenerationChoice, ...]:
    """Generate one or more choices from completions endpoint.

    Args:
        client: vLLM API client.
        config: Runtime config.
        assistant_prefix: Prefix passed as context.
        prompt_token_ids: Prompt token chain for token-space requests.
        max_tokens: Max generation tokens.
        n: Number of choices.
        step_index: Step index for deterministic seed offset.
        candidate_index_offset: Additional seed offset.
        stop: Optional stop sequences for generation termination.

    Returns:
        Generated choices.
    """
    request_seed = build_request_seed(
        config=config,
        step_index=step_index,
        candidate_index_offset=candidate_index_offset,
    )
    log_stage(
        stage="generation.request",
        step_index=step_index,
        n=n,
        max_tokens=max_tokens,
        stop=stop,
        request_seed=request_seed,
        prompt_mode="token_ids" if prompt_token_ids is not None else "text",
    )
    choices = call_completions(
        client=client,
        config=config,
        assistant_prefix=assistant_prefix,
        prompt_token_ids=prompt_token_ids,
        max_tokens=max_tokens,
        n=n,
        request_seed=request_seed,
        stop=stop,
    )
    finish_reasons = ",".join(choice.finish_reason for choice in choices)
    log_stage(
        stage="generation.response",
        step_index=step_index,
        choice_count=len(choices),
        finish_reasons=finish_reasons,
    )
    validate_stop_reason_presence(choices=choices, stop=stop)
    return choices


def validate_stop_reason_presence(
    *, choices: tuple[GenerationChoice, ...], stop: tuple[str, ...] | None
) -> None:
    """Assert stop-reason presence for stop-triggered completions requests.

    Args:
        choices: Parsed generation choices.
        stop: Stop markers used for the request.

    Returns:
        None.
    """
    if stop is None:
        return
    # Compatibility mode: some vLLM builds omit `stop_reason` even for stop finishes.
    for choice in choices:
        if choice.finish_reason != "stop":
            continue


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


def call_completions(
    *,
    client: BranchingClient,
    config: RunConfig,
    assistant_prefix: str,
    prompt_token_ids: tuple[int, ...] | None,
    max_tokens: int,
    n: int,
    request_seed: int,
    stop: tuple[str, ...] | None,
) -> tuple[GenerationChoice, ...]:
    """Call completions endpoint with either text or token-ID prompt mode.

    Args:
        client: vLLM API client.
        config: Runtime config.
        assistant_prefix: Assistant prefix context.
        prompt_token_ids: Prompt token IDs for token-space continuation.
        max_tokens: Max generation tokens.
        n: Number of choices.
        request_seed: Request seed.
        stop: Optional stop sequences for generation termination.

    Returns:
        Parsed generation choices.
    """
    prompt_text = build_raw_im_prompt(
        prompt=config.prompt,
        assistant_prefix=assistant_prefix,
    )
    if prompt_token_ids is None:
        return client.completions(
            model=config.model,
            prompt=prompt_text,
            prompt_token_ids=None,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=max_tokens,
            n=n,
            seed=request_seed,
            stop=stop,
            top_logprobs=config.capped_top_logprobs(),
        )
    cached_support = read_prompt_token_ids_support(client=client)
    if cached_support is False:
        return client.completions(
            model=config.model,
            prompt=prompt_text,
            prompt_token_ids=None,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=max_tokens,
            n=n,
            seed=request_seed,
            stop=stop,
            top_logprobs=config.capped_top_logprobs(),
        )
    try:
        choices = client.completions(
            model=config.model,
            prompt=None,
            prompt_token_ids=prompt_token_ids,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=max_tokens,
            n=n,
            seed=request_seed,
            stop=stop,
            top_logprobs=config.capped_top_logprobs(),
        )
        write_prompt_token_ids_support(client=client, supported=True)
        return choices
    except VllmRequestError as request_error:
        if "Either prompt or prompt_embeds must be provided and non-empty." not in str(
            request_error
        ):
            raise
        write_prompt_token_ids_support(client=client, supported=False)
        log_stage(
            stage="generation.fallback",
            reason="token_prompt_rejected_by_server",
        )
    return client.completions(
        model=config.model,
        prompt=prompt_text,
        prompt_token_ids=None,
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=max_tokens,
        n=n,
        seed=request_seed,
        stop=stop,
        top_logprobs=config.capped_top_logprobs(),
    )


def read_prompt_token_ids_support(*, client: BranchingClient) -> bool | None:
    """Read cached client capability for `prompt_token_ids` completions mode.

    Args:
        client: vLLM API client.

    Returns:
        `True` or `False` when cached, else `None`.
    """
    cached = getattr(client, "supports_prompt_token_ids", None)
    if cached is None:
        return None
    assert isinstance(cached, bool), "supports_prompt_token_ids must be bool when set"
    return cached


def write_prompt_token_ids_support(*, client: BranchingClient, supported: bool) -> None:
    """Write cached client capability for `prompt_token_ids` mode.

    Args:
        client: vLLM API client.
        supported: Capability value to cache.

    Returns:
        None.
    """
    setattr(client, "supports_prompt_token_ids", supported)


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
