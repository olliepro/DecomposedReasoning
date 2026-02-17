"""Tests for branching behavior and fallback logic."""

from __future__ import annotations

import random

from analysis_types import ApiModeConfig, RunConfig
from branching_engine import (
    event_rollout,
    make_candidate,
    process_branch_step,
    should_fallback_to_completions,
)
from vllm_client import GenerationChoice, ParsedToken


class FakeClient:
    """Minimal client stub for deterministic branching tests."""

    def __init__(self, *, choices: tuple[GenerationChoice, ...]) -> None:
        self.choices = choices

    def completions(  # noqa: D401
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
        _ = (
            model,
            prompt,
            temperature,
            top_p,
            max_tokens,
            n,
            seed,
            stop,
            top_logprobs,
        )
        return self.choices

    def chat(  # noqa: D401
        self,
        *,
        model: str,
        messages: list[dict[str, object]],
        temperature: float,
        top_p: float,
        max_tokens: int,
        n: int,
        seed: int,
        stop: tuple[str, ...] | None,
        top_logprobs: int,
        template_fields: dict[str, object],
    ) -> tuple[GenerationChoice, ...]:
        _ = (
            model,
            messages,
            temperature,
            top_p,
            max_tokens,
            n,
            seed,
            stop,
            top_logprobs,
            template_fields,
        )
        return self.choices


def build_choice(*, index: int, text: str, token_logprob: float) -> GenerationChoice:
    """Build deterministic generation choice for testing.

    Args:
        index: Choice index.
        text: Generated text.
        token_logprob: Logprob for each synthetic token.

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
        index=index, text=text, finish_reason="length", tokens=tokens
    )


def test_seeded_random_branch_selection_reproducible() -> None:
    """Random keep-1 policy should be reproducible for same seed."""
    choices = tuple(
        build_choice(index=index, text=f"choice-{index}", token_logprob=-0.1)
        for index in range(5)
    )
    fake_client = FakeClient(choices=choices)
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1", model="m", prompt="p", branch_factor=5
    )
    step_a = process_branch_step(
        client=fake_client,
        config=config,
        rng=random.Random(123),
        assistant_text="prefix<steer>",
        step_index=0,
        active_mode="completions",
    )
    step_b = process_branch_step(
        client=fake_client,
        config=config,
        rng=random.Random(123),
        assistant_text="prefix<steer>",
        step_index=0,
        active_mode="completions",
    )
    assert (
        step_a.step_metadata.selected_candidate_index
        == step_b.step_metadata.selected_candidate_index
    )


def test_make_candidate_trims_after_steer_close() -> None:
    """Candidate text should trim after first `</steer>` marker."""
    choice = build_choice(index=0, text="abc</steer>extra", token_logprob=-0.2)
    candidate = make_candidate(step_index=1, choice=choice)
    assert candidate.text == "abc</steer>"
    assert candidate.closed_with_tag is True


def test_missing_chat_template_fallback_strategy() -> None:
    """Fallback should trigger only for chat-template-style errors."""
    config = RunConfig(
        base_url="http://127.0.0.1:8000/v1",
        model="m",
        prompt="p",
        api_mode_config=ApiModeConfig(default_mode="chat", allow_fallback=True),
    )
    should_fallback = should_fallback_to_completions(
        config=config,
        error_text="chat template error: missing conversation template",
    )
    should_not_fallback = should_fallback_to_completions(
        config=config,
        error_text="some unrelated error",
    )
    assert should_fallback is True
    assert should_not_fallback is False


def test_event_rollout_keeps_post_think_text() -> None:
    """Think-end rollout should preserve text generated after `</think>`."""
    text = "<think>trace</think>\nFinal answer."
    event_end = text.index("</think>") + len("</think>")
    outcome = event_rollout(
        text=text,
        event_end=event_end,
        generated_tokens=3,
        token_stats=[],
        active_mode="completions",
        event_type="think_end",
    )
    assert outcome.assistant_text == text
    assert outcome.scan_index == len(text)


def test_event_rollout_trims_branch_text_at_boundary() -> None:
    """Branch rollout should still trim text at branch boundary end."""
    text = "<think><steer>after"
    boundary_end = len("<think><steer>")
    outcome = event_rollout(
        text=text,
        event_end=boundary_end,
        generated_tokens=2,
        token_stats=[],
        active_mode="completions",
        event_type="branch",
    )
    assert outcome.assistant_text == "<think><steer>"
    assert outcome.scan_index == boundary_end
