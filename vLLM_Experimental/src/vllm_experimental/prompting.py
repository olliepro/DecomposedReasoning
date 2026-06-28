"""Prompt formatting and tokenizer payload helpers for native benchmarks."""

from __future__ import annotations

import os
from typing import Any

from vllm_experimental.verbalized import (
    continue_exec_prefill,
    continue_steer,
    enumeration_prompt,
)

DEFAULT_MATH_SYSTEM_PROMPT = (
    "Solve the math problem. Put your reasoning in one <think>...</think> "
    "block made of alternating non-empty <steer>...</steer> and "
    "<exec>...</exec> blocks, starting with <steer>. Use <steer> blocks for "
    "guidance: guide thinking, make executive decisions, choose subproblems, "
    "slow down, enumerate, verify, or backtrack. Examples: Guide thinking: "
    '"Try applying ___." Make decisions: "Name the dog \'___\'." Choose '
    'subproblems: "Consider a<=3." Slow down: "Use a more precise method." '
    'Enumerate: "List 5 options and choose one." Verify: '
    '"Double Check that calculation." Backtrack: "Abandon this approach." '
    "Use <exec> blocks to precisely carry out the chosen guidance with "
    "calculations and deductions. After </think>, give the final answer as "
    "\\boxed{...} with no extra prose."
)
ASSISTANT_PREFILL = "<think>\n"


def ensure_assistant_prefill(*, rendered_prompt: str) -> str:
    """Return a rendered prompt ending with the assistant `<think>\n` seed."""

    if rendered_prompt.endswith(ASSISTANT_PREFILL):
        return rendered_prompt
    if rendered_prompt.endswith("<think>"):
        return rendered_prompt + "\n"
    return rendered_prompt + ASSISTANT_PREFILL


def chat_prompts(*, prompts: list[str], tokenizer: Any) -> list[str]:
    """Format raw task prompts with the model chat template."""

    assert hasattr(tokenizer, "apply_chat_template"), "tokenizer needs chat template"
    rendered_prompts: list[str] = []
    for prompt in prompts:
        rendered = str(
            tokenizer.apply_chat_template(
                conversation=[
                    {"role": "system", "content": DEFAULT_MATH_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        rendered_prompts.append(ensure_assistant_prefill(rendered_prompt=rendered))
    return rendered_prompts


def maybe_pad_raw_prompts(*, raw_prompts: list[str]) -> list[str]:
    """Optionally pad user prompts before chat-template formatting."""

    repeat_count = int(os.environ.get("PROMPT_PAD_REPEATS", "0"))
    if repeat_count <= 0:
        return raw_prompts
    pad = "\n\n" + ("filler " * repeat_count).strip()
    return [prompt + pad for prompt in raw_prompts]


def single_token_id(*, tokenizer: Any, text: str) -> int:
    """Return the token id for an atomic control token."""

    token_ids = tokenizer.encode(text=text, add_special_tokens=False)
    assert len(token_ids) == 1, f"{text!r} must be one added token, got {token_ids}"
    return int(token_ids[0])


def control_token_payload(*, tokenizer: Any) -> dict[str, int | None]:
    """Return control token IDs for SamplingParams.extra_args."""

    return {
        "think_open": single_token_id(tokenizer=tokenizer, text="<think>"),
        "think_close": single_token_id(tokenizer=tokenizer, text="</think>"),
        "steer_open": single_token_id(tokenizer=tokenizer, text="<steer>"),
        "steer_close": single_token_id(tokenizer=tokenizer, text="</steer>"),
        "exec_open": single_token_id(tokenizer=tokenizer, text="<exec>"),
        "exec_close": single_token_id(tokenizer=tokenizer, text="</exec>"),
        "newline": single_token_id(tokenizer=tokenizer, text="\n"),
        "eos": getattr(tokenizer, "eos_token_id", None),
    }


def tokenized_text(*, tokenizer: Any, text: str) -> list[int]:
    """Return token ids for deterministic grammar-forced text."""

    token_ids = tokenizer.encode(text=text, add_special_tokens=False)
    assert token_ids, f"script text tokenized empty: {text!r}"
    return [int(token_id) for token_id in token_ids]


def attach_prompt_prefill(*, tree_search: dict[str, object], tokenizer: Any) -> None:
    """Attach assistant-prefill tokens already present in every rendered prompt."""

    tree_search["prefix_output_token_ids"] = tokenized_text(
        tokenizer=tokenizer,
        text=ASSISTANT_PREFILL,
    )


def verbalized_token_scripts(*, tokenizer: Any) -> dict[str, dict[str, list[int]]]:
    """Return tokenized off-policy verbalized grammar scripts."""

    enumerate_scripts: dict[str, list[int]] = {}
    continue_scripts: dict[str, list[int]] = {}
    for candidate_count in range(3, 11):
        text = (
            f"{enumeration_prompt(candidate_count=candidate_count)}" "</steer>\n<exec>"
        )
        enumerate_scripts[str(candidate_count)] = tokenized_text(
            tokenizer=tokenizer,
            text=text,
        )
    for option_number in range(1, 11):
        text = (
            f"{continue_steer(option_number=option_number)}</steer>"
            f"\n<exec>{continue_exec_prefill(option_number=option_number)}"
        )
        continue_scripts[str(option_number)] = tokenized_text(
            tokenizer=tokenizer,
            text=text,
        )
    return {"enumerate": enumerate_scripts, "continue": continue_scripts}


def attach_token_payloads(*, tree_search: dict[str, object], tokenizer: Any) -> None:
    """Attach tokenizer-derived payloads required by the vLLM plugin."""

    tree_search["control_token_ids"] = control_token_payload(tokenizer=tokenizer)
    attach_prompt_prefill(tree_search=tree_search, tokenizer=tokenizer)
    tree_search["verbalized_token_scripts"] = verbalized_token_scripts(
        tokenizer=tokenizer
    )
