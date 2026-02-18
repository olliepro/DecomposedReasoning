"""Tests for completions-only branching behavior and token-space boundaries."""

from __future__ import annotations

import random
from pathlib import Path

from analysis_types import BranchStep, RunConfig, SteerCandidate
from branching_engine import (
    AnalyzerState,
    BranchStepResult,
    RolloutCursor,
    append_decoded_path_snapshot,
    apply_choice_to_cursor,
    call_completions,
    ensure_canonical_steer_open,
    forced_boundary_suffix,
    length_finished_outcome,
    make_candidate,
    normalize_selected_candidate_for_execution,
    process_branch_step,
    rollout_chunk,
    rollout_chunk_budget,
    stop_finished_outcome,
    trailing_partial_tag_suffix,
)
from vllm_client import GenerationChoice, ParsedToken, VllmRequestError


class FakeClient:
    """Minimal client stub for deterministic branching tests."""

    def __init__(
        self,
        *,
        responses: list[tuple[GenerationChoice, ...]],
        tokenize_base_id: int = 1000,
    ) -> None:
        self.responses = list(responses)
        self.completion_calls: list[dict[str, object]] = []
        self.tokenize_calls: list[str] = []
        self._tokenize_base_id = tokenize_base_id
        self.supports_prompt_token_ids: bool | None = None

    def completions(  # noqa: D401
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
        self.completion_calls.append(
            {
                "model": model,
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "n": n,
                "seed": seed,
                "stop": stop,
                "top_logprobs": top_logprobs,
            }
        )
        assert self.responses, "no fake response queued"
        return self.responses.pop(0)

    def tokenize(
        self,
        *,
        model: str,
        text: str,
        add_special_tokens: bool = False,
    ) -> tuple[int, ...]:
        _ = model, add_special_tokens
        self.tokenize_calls.append(text)
        start = self._tokenize_base_id
        self._tokenize_base_id += len(text)
        return tuple(range(start, start + len(text)))


class TokenPromptRejectingClient(FakeClient):
    """Client stub that rejects token-id prompt mode like unsupported vLLM servers."""

    def completions(  # noqa: D401
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
        if prompt_token_ids is not None:
            raise VllmRequestError(
                "Either prompt or prompt_embeds must be provided and non-empty."
            )
        return super().completions(
            model=model,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            n=n,
            seed=seed,
            stop=stop,
            top_logprobs=top_logprobs,
        )


def build_choice(
    *,
    index: int,
    text: str,
    token_logprob: float,
    finish_reason: str = "length",
    stop_reason: int | str | None = None,
    prompt_token_ids: tuple[int, ...] = (11, 22, 33),
    token_ids: tuple[int, ...] = (44, 55),
) -> GenerationChoice:
    """Build deterministic generation choice for testing.

    Args:
        index: Choice index.
        text: Generated text.
        token_logprob: Logprob for each synthetic token.
        finish_reason: Choice finish reason.
        stop_reason: Optional stop reason.
        prompt_token_ids: Prompt token ID chain used by request.
        token_ids: Generated output token IDs.

    Returns:
        Generation choice dataclass.
    """
    tokens = (
        ParsedToken(
            token="a", logprob=token_logprob, top_entries=(("a", token_logprob),)
        ),
        ParsedToken(
            token="b", logprob=token_logprob, top_entries=(("b", token_logprob),)
        ),
    )
    return GenerationChoice(
        index=index,
        text=text,
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        tokens=tokens,
        prompt_token_ids=prompt_token_ids,
        token_ids=token_ids,
    )


def test_seeded_random_branch_selection_reproducible() -> None:
    """Random keep-1 policy should be reproducible for same seed."""
    choices = tuple(
        build_choice(index=index, text=f"choice-{index}", token_logprob=-0.1)
        for index in range(5)
    )
    fake_client = FakeClient(responses=[choices, choices])
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        branch_factor=5,
    )
    step_a = process_branch_step(
        client=fake_client,
        config=config,
        rng=random.Random(123),
        assistant_text="prefix<steer>",
        prompt_token_ids=(1, 2, 3),
        step_index=0,
    )
    step_b = process_branch_step(
        client=fake_client,
        config=config,
        rng=random.Random(123),
        assistant_text="prefix<steer>",
        prompt_token_ids=(1, 2, 3),
        step_index=0,
    )
    assert (
        step_a.step_metadata.selected_candidate_index
        == step_b.step_metadata.selected_candidate_index
    )


def test_call_completions_caches_prompt_token_ids_support() -> None:
    """After one rejection, calls should stop retrying token-id prompt mode."""
    responses = [
        (
            build_choice(
                index=0, text="first", token_logprob=-0.1, finish_reason="stop"
            ),
        ),
        (
            build_choice(
                index=0, text="second", token_logprob=-0.1, finish_reason="stop"
            ),
        ),
    ]
    client = TokenPromptRejectingClient(responses=responses)
    config = RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p")
    first = call_completions(
        client=client,
        config=config,
        assistant_prefix="<think><steer>",
        prompt_token_ids=(1, 2, 3),
        max_tokens=8,
        n=1,
        request_seed=11,
        stop=("</steer>",),
    )
    second = call_completions(
        client=client,
        config=config,
        assistant_prefix="<think><steer>first</steer><exec>x</exec><steer>",
        prompt_token_ids=(1, 2, 3, 4),
        max_tokens=8,
        n=1,
        request_seed=12,
        stop=("</steer>",),
    )
    assert first[0].text == "first"
    assert second[0].text == "second"
    assert client.supports_prompt_token_ids is False
    assert len(client.completion_calls) == 2
    assert all(call["prompt_token_ids"] is None for call in client.completion_calls)


def test_make_candidate_trims_after_steer_close() -> None:
    """Candidate text should trim after first `</steer>` marker."""
    choice = build_choice(index=0, text="abc</steer>extra", token_logprob=-0.2)
    candidate = make_candidate(step_index=1, choice=choice)
    assert candidate.text == "abc</steer>"
    assert candidate.closed_with_tag is True


def test_trailing_partial_tag_suffix_detects_known_prefixes() -> None:
    """Partial steer/exec suffixes should map to canonical completion tags."""
    steer_partial = trailing_partial_tag_suffix(text="prefix<ste")
    exec_partial = trailing_partial_tag_suffix(text="prefix</ex")
    single_angle_partial = trailing_partial_tag_suffix(text="prefix<")
    assert steer_partial == ("<ste", "<steer>")
    assert exec_partial == ("</ex", "</exec>")
    assert single_angle_partial == ("<", "<steer>")


def test_forced_boundary_suffix_reuses_observed_separator() -> None:
    """Forced boundary should reuse `</exec>`-to-`<steer>` whitespace when available."""
    text = "<exec>a</exec>\n\n<steer>x</steer><exec>y"
    suffix = forced_boundary_suffix(text=text)
    assert suffix == "</exec>\n\n<steer>"


def test_stop_finished_outcome_injects_canonical_steer() -> None:
    """Stop-reason steer hits should inject canonical `<steer>` before branching."""
    fake_client = FakeClient(responses=[])
    cursor = RolloutCursor(
        text="<think><exec>work</exec>",
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=(1, 2, 3),
    )
    choice = build_choice(
        index=0,
        text="",
        token_logprob=-0.1,
        finish_reason="stop",
        stop_reason="<steer",
    )
    outcome = stop_finished_outcome(
        client=fake_client,
        config=RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p"),
        cursor=cursor,
        choice=choice,
    )
    assert outcome.event_type == "branch"
    assert outcome.assistant_text.endswith("<steer>")


def test_stop_finished_outcome_closes_exec_before_steer_boundary() -> None:
    """Stop boundary inside `<exec>` should close exec before next branch steer."""
    fake_client = FakeClient(responses=[])
    cursor = RolloutCursor(
        text="<think><exec>work<steer",
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=(1, 2, 3),
    )
    choice = build_choice(
        index=0,
        text="",
        token_logprob=-0.1,
        finish_reason="stop",
        stop_reason="<steer",
    )
    outcome = stop_finished_outcome(
        client=fake_client,
        config=RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p"),
        cursor=cursor,
        choice=choice,
    )
    assert outcome.event_type == "branch"
    assert outcome.assistant_text.endswith("</steer></exec>\n\n<steer>")


def test_length_boundary_think_edge_runs_one_continuation_without_stop() -> None:
    """Length+think-end case should issue one no-stop continuation with remaining budget."""
    first_choice = build_choice(
        index=0,
        text="<think>trace</think>",
        token_logprob=-0.1,
        finish_reason="length",
    )
    continuation_choice = build_choice(
        index=0,
        text="\nFinal answer.",
        token_logprob=-0.2,
        finish_reason="stop",
        stop_reason=None,
        prompt_token_ids=(11, 22, 33, 44, 55),
        token_ids=(66, 77),
    )
    fake_client = FakeClient(responses=[(continuation_choice,)])
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        max_total_tokens=20,
    )
    state = AnalyzerState(
        assistant_text="",
        scan_index=0,
        used_tokens=0,
        prompt_token_ids=None,
    )
    cursor = RolloutCursor(
        text="",
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=None,
    )
    apply_choice_to_cursor(cursor=cursor, step_index=0, choice=first_choice)
    remaining_budget = rollout_chunk_budget(
        config=config,
        state=state,
        generated_tokens=cursor.generated_tokens,
    )
    outcome = length_finished_outcome(
        client=fake_client,
        config=config,
        state=state,
        step_index=0,
        cursor=cursor,
        choice=first_choice,
    )
    assert fake_client.completion_calls[-1]["stop"] is None
    assert fake_client.completion_calls[-1]["max_tokens"] == remaining_budget
    assert outcome.event_type == "terminated"
    assert outcome.termination_reason == "think_end"


def test_stop_chunk_with_think_close_continues_without_stop() -> None:
    """Stop-triggered think close should continue with `stop=None` and terminate."""
    first_choice = build_choice(
        index=0,
        text="</think><steer",
        token_logprob=-0.1,
        finish_reason="stop",
        stop_reason="<steer",
    )
    continuation_choice = build_choice(
        index=0,
        text="\nDone.",
        token_logprob=-0.2,
        finish_reason="stop",
        stop_reason=None,
        prompt_token_ids=(11, 22, 33, 44, 55),
        token_ids=(66, 77),
    )
    fake_client = FakeClient(responses=[(first_choice,), (continuation_choice,)])
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        max_total_tokens=20,
    )
    state = AnalyzerState(
        assistant_text="<think>\n<steer>p</steer>\n<exec>x</exec>\n<steer>q</steer>\n<exec>y",
        scan_index=0,
        used_tokens=0,
        prompt_token_ids=(1, 2, 3),
    )
    cursor = RolloutCursor(
        text=state.assistant_text,
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=(1, 2, 3),
    )
    outcome = rollout_chunk(
        client=fake_client,
        config=config,
        state=state,
        step_index=0,
        cursor=cursor,
    )
    assert outcome is not None
    assert outcome.event_type == "terminated"
    assert outcome.termination_reason == "think_end"
    assert fake_client.completion_calls[0]["stop"] == ("<steer",)
    assert fake_client.completion_calls[1]["stop"] is None
    assert fake_client.tokenize_calls == []


def test_length_chunk_with_partial_think_continues_without_stop() -> None:
    """Length chunk ending in partial `</think` should trigger no-stop continuation."""
    first_choice = build_choice(
        index=0,
        text="</thi",
        token_logprob=-0.1,
        finish_reason="length",
    )
    continuation_choice = build_choice(
        index=0,
        text="nk>\nFinal answer.",
        token_logprob=-0.2,
        finish_reason="stop",
        stop_reason=None,
        prompt_token_ids=(11, 22, 33, 44, 55),
        token_ids=(66, 77),
    )
    fake_client = FakeClient(responses=[(first_choice,), (continuation_choice,)])
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        max_total_tokens=20,
    )
    state = AnalyzerState(
        assistant_text="<think>\n<steer>p</steer>\n<exec>x</exec>\n<steer>q</steer>\n<exec>y",
        scan_index=0,
        used_tokens=0,
        prompt_token_ids=(1, 2, 3),
    )
    cursor = RolloutCursor(
        text=state.assistant_text,
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=(1, 2, 3),
    )
    outcome = rollout_chunk(
        client=fake_client,
        config=config,
        state=state,
        step_index=0,
        cursor=cursor,
    )
    assert outcome is not None
    assert outcome.event_type == "terminated"
    assert outcome.termination_reason == "think_end"
    assert fake_client.completion_calls[0]["stop"] == ("<steer",)
    assert fake_client.completion_calls[1]["stop"] is None
    assert fake_client.tokenize_calls == []


def test_natural_finish_with_think_close_does_not_force_continuation() -> None:
    """Natural finish with complete `</think>` should terminate without extra call."""
    first_choice = build_choice(
        index=0,
        text="</think>",
        token_logprob=-0.1,
        finish_reason="eos",
    )
    fake_client = FakeClient(responses=[(first_choice,)])
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        max_total_tokens=20,
    )
    state = AnalyzerState(
        assistant_text="<think>\n<steer>p</steer>\n<exec>x</exec>\n<steer>q</steer>\n<exec>y",
        scan_index=0,
        used_tokens=0,
        prompt_token_ids=(1, 2, 3),
    )
    cursor = RolloutCursor(
        text=state.assistant_text,
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=(1, 2, 3),
    )
    outcome = rollout_chunk(
        client=fake_client,
        config=config,
        state=state,
        step_index=0,
        cursor=cursor,
    )
    assert outcome is not None
    assert outcome.event_type == "terminated"
    assert outcome.termination_reason == "think_end"
    assert len(fake_client.completion_calls) == 1
    assert fake_client.completion_calls[0]["stop"] == ("<steer",)


def test_stop_without_reason_terminates_naturally() -> None:
    """`finish_reason=stop` with `stop_reason=None` should terminate, not branch."""
    first_choice = build_choice(
        index=0,
        text="Final answer text.",
        token_logprob=-0.1,
        finish_reason="stop",
        stop_reason=None,
    )
    fake_client = FakeClient(responses=[(first_choice,)])
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        max_total_tokens=20,
    )
    state = AnalyzerState(
        assistant_text="<think>\n<steer>p</steer>\n<exec>y",
        scan_index=0,
        used_tokens=0,
        prompt_token_ids=(1, 2, 3),
    )
    cursor = RolloutCursor(
        text=state.assistant_text,
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=(1, 2, 3),
    )
    outcome = rollout_chunk(
        client=fake_client,
        config=config,
        state=state,
        step_index=0,
        cursor=cursor,
    )
    assert outcome is not None
    assert outcome.event_type == "terminated"
    assert outcome.termination_reason == "model_finished"
    assert len(fake_client.completion_calls) == 1
    assert fake_client.tokenize_calls == []


def test_stop_without_reason_with_think_close_does_not_continue() -> None:
    """`</think>` plus `stop_reason=None` should terminate without continuation."""
    first_choice = build_choice(
        index=0,
        text="</think>\nDone.",
        token_logprob=-0.1,
        finish_reason="stop",
        stop_reason=None,
    )
    fake_client = FakeClient(responses=[(first_choice,)])
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        max_total_tokens=20,
    )
    state = AnalyzerState(
        assistant_text="<think>\n<steer>p</steer>\n<exec>y",
        scan_index=0,
        used_tokens=0,
        prompt_token_ids=(1, 2, 3),
    )
    cursor = RolloutCursor(
        text=state.assistant_text,
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=(1, 2, 3),
    )
    outcome = rollout_chunk(
        client=fake_client,
        config=config,
        state=state,
        step_index=0,
        cursor=cursor,
    )
    assert outcome is not None
    assert outcome.event_type == "terminated"
    assert outcome.termination_reason == "think_end"
    assert len(fake_client.completion_calls) == 1
    assert fake_client.completion_calls[0]["stop"] == ("<steer",)
    assert fake_client.tokenize_calls == []


def test_length_repaired_steer_closes_exec_before_branch_boundary() -> None:
    """Length repair to `<steer>` inside `<exec>` should close exec before branching."""
    fake_client = FakeClient(responses=[])
    config = RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p")
    state = AnalyzerState(
        assistant_text="",
        scan_index=0,
        used_tokens=0,
        prompt_token_ids=(1, 2, 3),
    )
    cursor = RolloutCursor(
        text="<think><exec>work<",
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=(1, 2, 3),
    )
    choice = build_choice(
        index=0,
        text="<",
        token_logprob=-0.1,
        finish_reason="length",
    )
    outcome = length_finished_outcome(
        client=fake_client,
        config=config,
        state=state,
        step_index=0,
        cursor=cursor,
        choice=choice,
    )
    assert outcome.event_type == "branch"
    assert outcome.assistant_text.endswith("</steer></exec>\n\n<steer>")


def test_ensure_canonical_steer_open_repairs_invalid_entry_context() -> None:
    """Canonical steer boundary should repair invalid entry context via token suffixes."""
    fake_client = FakeClient(responses=[])
    cursor = RolloutCursor(
        text="<think>\n<steer></steer><steer",
        scan_index=0,
        generated_tokens=0,
        token_stats=[],
        prompt_token_ids=(1, 2, 3),
    )
    ensure_canonical_steer_open(
        client=fake_client,
        config=RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p"),
        cursor=cursor,
    )
    assert cursor.text.endswith("</exec>\n\n<steer>")
    assert fake_client.tokenize_calls[0] == ">"
    assert fake_client.tokenize_calls[1] == "</steer><exec></exec>\n\n<steer>"


def test_normalize_selected_candidate_for_execution_appends_full_steer_close() -> None:
    """Selected candidate without close suffix should append full `</steer>\n`."""
    fake_client = FakeClient(responses=[])
    selected = SteerCandidate(
        step_index=0,
        candidate_index=0,
        text="Plan next step",
        token_count=2,
        closed_with_tag=False,
        finish_reason="length",
        cumulative_logprob=-0.2,
        average_logprob=-0.1,
    )
    branch_result = BranchStepResult(
        step_metadata=BranchStep(
            step_index=0,
            prefix_char_end=10,
            selected_candidate_index=0,
            selected_text=selected.text,
            total_candidates=1,
            unique_candidate_count=1,
            terminated=False,
            termination_reason="",
        ),
        selected_candidate=selected,
        selected_token_ids=(44, 55),
        candidates=(selected,),
        token_stats=(),
    )
    normalized = normalize_selected_candidate_for_execution(
        client=fake_client,
        config=RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p"),
        branch_result=branch_result,
    )
    assert normalized.selected_candidate.closed_with_tag is True
    assert normalized.selected_candidate.text.endswith("</steer>\n")
    assert normalized.step_metadata.selected_text.endswith("</steer>\n")
    assert normalized.candidates[0].text.endswith("</steer>\n")
    assert normalized.selected_token_ids == (
        44,
        55,
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
        1008,
    )
    assert fake_client.tokenize_calls == ["</steer>\n"]


def test_normalize_selected_candidate_for_execution_completes_partial_close() -> None:
    """Selected candidate ending with partial `</steer` should append only `>\n`."""
    fake_client = FakeClient(responses=[])
    selected = SteerCandidate(
        step_index=0,
        candidate_index=0,
        text="Calculate walking time</steer",
        token_count=2,
        closed_with_tag=False,
        finish_reason="length",
        cumulative_logprob=-0.2,
        average_logprob=-0.1,
    )
    branch_result = BranchStepResult(
        step_metadata=BranchStep(
            step_index=0,
            prefix_char_end=10,
            selected_candidate_index=0,
            selected_text=selected.text,
            total_candidates=1,
            unique_candidate_count=1,
            terminated=False,
            termination_reason="",
        ),
        selected_candidate=selected,
        selected_token_ids=(44, 55),
        candidates=(selected,),
        token_stats=(),
    )
    normalized = normalize_selected_candidate_for_execution(
        client=fake_client,
        config=RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p"),
        branch_result=branch_result,
    )
    assert "</steer</steer>" not in normalized.selected_candidate.text
    assert normalized.selected_candidate.text == "Calculate walking time</steer>\n"
    assert normalized.selected_token_ids == (44, 55, 1000, 1001)
    assert fake_client.tokenize_calls == [">\n"]


def test_normalize_selected_candidate_for_execution_completes_single_lt() -> None:
    """Selected candidate ending with `<` should append `/steer>\n`."""
    fake_client = FakeClient(responses=[])
    selected = SteerCandidate(
        step_index=0,
        candidate_index=0,
        text="Calculate walking time<",
        token_count=2,
        closed_with_tag=False,
        finish_reason="length",
        cumulative_logprob=-0.2,
        average_logprob=-0.1,
    )
    branch_result = BranchStepResult(
        step_metadata=BranchStep(
            step_index=0,
            prefix_char_end=10,
            selected_candidate_index=0,
            selected_text=selected.text,
            total_candidates=2,
            unique_candidate_count=2,
            terminated=False,
            termination_reason="",
        ),
        selected_candidate=selected,
        selected_token_ids=(44, 55),
        candidates=(
            selected,
            SteerCandidate(
                step_index=0,
                candidate_index=1,
                text="Unselected text",
                token_count=2,
                closed_with_tag=False,
                finish_reason="length",
                cumulative_logprob=-0.2,
                average_logprob=-0.1,
            ),
        ),
        token_stats=(),
    )
    normalized = normalize_selected_candidate_for_execution(
        client=fake_client,
        config=RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p"),
        branch_result=branch_result,
    )
    assert normalized.selected_candidate.text == "Calculate walking time</steer>\n"
    assert normalized.candidates[1].text == "Unselected text"
    assert normalized.selected_token_ids == (
        44,
        55,
        1000,
        1001,
        1002,
        1003,
        1004,
        1005,
        1006,
        1007,
    )
    assert fake_client.tokenize_calls == ["/steer>\n"]


def test_normalize_selected_candidate_for_execution_keeps_canonical_text() -> None:
    """Already-canonical selected candidate should not inject extra suffix tokens."""
    fake_client = FakeClient(responses=[])
    selected = SteerCandidate(
        step_index=0,
        candidate_index=0,
        text="Plan next step</steer>\n",
        token_count=2,
        closed_with_tag=True,
        finish_reason="stop",
        cumulative_logprob=-0.2,
        average_logprob=-0.1,
    )
    branch_result = BranchStepResult(
        step_metadata=BranchStep(
            step_index=0,
            prefix_char_end=10,
            selected_candidate_index=0,
            selected_text=selected.text,
            total_candidates=1,
            unique_candidate_count=1,
            terminated=False,
            termination_reason="",
        ),
        selected_candidate=selected,
        selected_token_ids=(44, 55),
        candidates=(selected,),
        token_stats=(),
    )
    normalized = normalize_selected_candidate_for_execution(
        client=fake_client,
        config=RunConfig(base_url="http://127.0.0.1:8000/v1", model="m", prompt="p"),
        branch_result=branch_result,
    )
    assert normalized is branch_result
    assert normalized.selected_token_ids == (44, 55)
    assert fake_client.tokenize_calls == []


def test_append_decoded_path_snapshot_writes_chosen_path_log(tmp_path: Path) -> None:
    """Chosen-path snapshot logging should persist decoded state text."""
    state = AnalyzerState(
        assistant_text="<think>\nwalk 200 feet</think>",
        scan_index=29,
        used_tokens=8,
        prompt_token_ids=(1, 2, 3),
    )
    log_path = tmp_path / "chosen_path.log"
    append_decoded_path_snapshot(
        path=log_path,
        step_index=3,
        phase="after_rollout",
        state=state,
    )
    content = log_path.read_text(encoding="utf-8")
    assert "step=3" in content
    assert "phase=after_rollout" in content
    assert "used_tokens=8" in content
    assert "<think>\nwalk 200 feet</think>" in content
