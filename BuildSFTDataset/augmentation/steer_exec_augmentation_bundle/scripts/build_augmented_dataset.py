from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trace_augmentor import (  # noqa: E402
    OpenRouterAsyncClient,
    TokenCounter,
    TraceBlock,
    apply_sampling_defaults_to_spec,
    build_intervention_schema,
    build_prompt_values,
    choose_pairs_to_generate,
    choose_prompt_path,
    choose_variant,
    dump_jsonl,
    extract_trace_text,
    get_next_original_context,
    load_json,
    load_jsonl,
    maybe_call_bridge_judge,
    mock_intervention_window,
    parse_trace_text,
    render_prompt,
    render_trace,
    render_window,
    slice_pairs,
    splice_blocks,
    validate_generated_window,
)

DEFAULT_SOURCE_PATH = (
    ROOT.parent.parent
    / "output_transform_async_16384"
    / "transformed_subset_analysis"
    / "transformed_output_think_ratio_0.8_1.3_valid_le_16384_no_extract_anomalies.jsonl"
)


@dataclass(frozen=True)
class StepPlan:
    """One intervention step applied within a multi-step augmentation plan.

    Args:
        intervention_name: Human-readable intervention label.
        mode: Prompt mode to use for this step.
        variant: Exact first-steer text for the generated window.
        pairs_generated: Number of new `<steer>/<exec>` pairs to insert.
        cut_after_pairs: Pair boundary in the original trace before this step.
        post_splice_policy: Suffix-handling policy from `interventions.json`.
        category: Intervention category for reporting.
        steer_token_limit: Inclusive token cap for generated steer blocks.

    Example:
        >>> StepPlan(
        ...     intervention_name="State known constraints",
        ...     mode="insert",
        ...     variant="state the constraints explicitly",
        ...     pairs_generated=1,
        ...     cut_after_pairs=3,
        ...     post_splice_policy="keep_original_suffix",
        ...     category="local_compressive",
        ...     steer_token_limit=15,
        ... ).mode
        'insert'
    """

    intervention_name: str
    mode: Literal["insert", "bridge", "redirect"]
    variant: str
    pairs_generated: int
    cut_after_pairs: int
    post_splice_policy: str
    category: str
    steer_token_limit: int | None

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of this step plan."""
        return asdict(self)


@dataclass(frozen=True)
class RowPlan:
    """Multi-step augmentation plan for one source row.

    Args:
        source_row_index: Zero-based row index in the source JSONL.
        source_row_id: Stable source row identifier.
        plan_type: Requested composition for this augmented row.
        seed: Deterministic seed for per-row planning and generation.
        steps: Ordered intervention steps for this row.
    """

    source_row_index: int
    source_row_id: str
    plan_type: Literal["insert_only", "bridge_or_redirect"]
    seed: int
    steps: tuple[StepPlan, ...]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of this row plan."""
        payload = asdict(self)
        payload["steps"] = [step.to_json() for step in self.steps]
        return payload


@dataclass(frozen=True)
class ExecutedStep:
    """One completed intervention step and its suffix decision."""

    plan: StepPlan
    cut_after_pairs_current: int
    suffix_decision: Literal["keep_suffix", "regenerate_suffix"]
    bridge_judge_reason: str
    generated_blocks: tuple[dict[str, str], ...]

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of this executed step."""
        return {
            "plan": self.plan.to_json(),
            "cut_after_pairs_current": self.cut_after_pairs_current,
            "suffix_decision": self.suffix_decision,
            "bridge_judge_reason": self.bridge_judge_reason,
            "generated_blocks": list(self.generated_blocks),
        }


@dataclass(frozen=True)
class RowResult:
    """Result payload for one processed source row."""

    ok: bool
    row: dict[str, Any] | None
    failure: dict[str, Any] | None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the multi-step augmentation job."""
    parser = argparse.ArgumentParser(
        description="Build a 400-row multi-step augmentation set in the transformed dataset format."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument(
        "--output-dir", default=str(ROOT / "out" / "multi_step_augmented")
    )
    parser.add_argument("--interventions", default=str(ROOT / "interventions.json"))
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument("--env-file", default=str(ROOT.parent.parent / ".env"))
    parser.add_argument("--target-rows", type=int, default=400)
    parser.add_argument("--bridge-redirect-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-concurrency", type=int, default=6)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--style-window-pairs", type=int, default=2)
    parser.add_argument("--exec-token-limit", type=int, default=512)
    parser.add_argument("--length-tokenizer", default=None)
    parser.add_argument("--mock-intervention", action="store_true")
    parser.add_argument("--run-bridge-judge", action="store_true", default=True)
    parser.add_argument("--openrouter-model", default="openai/gpt-oss-20b")
    parser.add_argument("--openrouter-timeout-s", type=int, default=180)
    parser.add_argument("--openrouter-temperature", type=float, default=0.4)
    parser.add_argument("--openrouter-max-tokens", type=int, default=1200)
    parser.add_argument("--openrouter-site-url", default=None)
    parser.add_argument("--openrouter-site-name", default="steer-exec-multi-augmentor")
    parser.add_argument("--provider-data-collection", default="deny")
    parser.add_argument("--use-response-healing", action="store_true")
    parser.add_argument("--vllm-base-url", default="http://localhost:8000/v1")
    parser.add_argument("--vllm-model", default="")
    return parser


def load_env_values(path: str | Path | None) -> dict[str, str]:
    """Load simple `KEY=VALUE` pairs from a dotenv-style file.

    Args:
        path: Optional dotenv file path.

    Returns:
        Parsed key/value mapping. Missing files return an empty mapping.
    """

    if path is None:
        return {}
    env_path = Path(path)
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def resolve_openrouter_key(args: argparse.Namespace) -> str | None:
    """Resolve the OpenRouter API key from environment or the provided env file."""
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    if os.environ.get("OPEN_ROUTER_KEY"):
        os.environ["OPENROUTER_API_KEY"] = os.environ["OPEN_ROUTER_KEY"]
        return os.environ["OPENROUTER_API_KEY"]

    env_values = load_env_values(path=args.env_file)
    api_key = env_values.get("OPENROUTER_API_KEY") or env_values.get("OPEN_ROUTER_KEY")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    return api_key


def coalesce_user_prompt(record: Mapping[str, Any]) -> str:
    """Extract the display user prompt from a transformed dataset row."""
    if isinstance(record.get("user_prompt"), str):
        return str(record["user_prompt"])

    messages = record.get("messages")
    if not isinstance(messages, list):
        return ""

    for message in messages:
        if message.get("role") == "user" and isinstance(message.get("content"), str):
            return str(message["content"])
    return ""


def split_assistant_content(content: str) -> tuple[str, str, str]:
    """Split assistant content into prefix, think body, and suffix.

    Args:
        content: Assistant message content containing exactly one `<think>` block.

    Returns:
        Tuple of prefix text, inner think text, and suffix text.
    """

    assert content.count("<think>") == 1, "Expected exactly one opening <think> tag."
    assert content.count("</think>") == 1, "Expected exactly one closing </think> tag."
    prefix, remainder = content.split("<think>", 1)
    think_text, suffix = remainder.split("</think>", 1)
    return prefix, think_text.strip(), suffix


def capitalize_first_word(text: str) -> str:
    """Uppercase the first alphabetical character in the first word.

    Args:
        text: Steer text to normalize.

    Returns:
        Text with the first word capitalized when possible.

    Example:
        >>> capitalize_first_word("state the constraints explicitly")
        'State the constraints explicitly'
    """
    chars = list(text)
    seen_non_whitespace = False
    for index, char in enumerate(chars):
        if not char.isspace():
            seen_non_whitespace = True
        if seen_non_whitespace and char.isalpha():
            chars[index] = char.upper()
            return "".join(chars)
    return text


def normalize_steer_blocks(blocks: Sequence[TraceBlock]) -> list[TraceBlock]:
    """Return blocks with steer texts normalized to capitalized first words."""
    normalized_blocks: list[TraceBlock] = []
    for block in blocks:
        if block.type == "steer":
            normalized_blocks.append(
                TraceBlock(
                    type=block.type,
                    text=capitalize_first_word(block.text),
                )
            )
            continue
        normalized_blocks.append(block)
    return normalized_blocks


def render_dataset_trace(blocks: Sequence[TraceBlock]) -> str:
    """Render blocks with the whitespace pattern used in clean transformed rows.

    Args:
        blocks: Interleaved steer/exec blocks.

    Returns:
        Canonically formatted inner `<think>` content.

    Example:
        >>> render_dataset_trace(
        ...     blocks=[
        ...         TraceBlock(type="steer", text="rest"),
        ...         TraceBlock(type="exec", text="work"),
        ...     ]
        ... )
        '<steer>rest</steer>\\n<exec>\\nwork\\n</exec>'
    """

    normalized_blocks = normalize_steer_blocks(blocks=blocks)
    assert (
        len(normalized_blocks) % 2 == 0
    ), "Expected an even number of interleaved blocks."
    sections: list[str] = []
    for pair_start in range(0, len(normalized_blocks), 2):
        steer_block = normalized_blocks[pair_start]
        exec_block = normalized_blocks[pair_start + 1]
        sections.append(
            f"<steer>{steer_block.text}</steer>\n" f"<exec>\n{exec_block.text}\n</exec>"
        )
    return "\n\n".join(sections)


def render_assistant_content(
    *, prefix: str, blocks: Sequence[TraceBlock], suffix: str
) -> str:
    """Render full assistant content while preserving non-think suffix text."""
    think_text = render_dataset_trace(blocks=blocks)
    return f"{prefix}<think>\n{think_text}\n</think>{suffix}"


def reopen_final_exec(*, think_text: str) -> str:
    """Return think text with the last `</exec>` removed for continuation.

    Args:
        think_text: Closed interleaved steer/exec text from inside `<think>`.

    Returns:
        Think text ending inside the final `<exec>` block.

    Example:
        >>> reopen_final_exec(think_text="<steer>x</steer>\\n<exec>\\ny\\n</exec>")
        '<steer>x</steer>\\n<exec>\\ny\\n'
    """

    closing_tag = "\n</exec>"
    assert think_text.endswith(closing_tag), "regen prefill must end with a closed exec"
    return think_text[: -len(closing_tag)]


def build_open_exec_prefill(*, assistant_content: str) -> str:
    """Return assistant prefill ending inside the final `<exec>` block.

    Args:
        assistant_content: Closed assistant content containing exactly one think block.

    Returns:
        Assistant prefill with open `<think>` and open final `<exec>`.
    """

    prefix, think_text, suffix = split_assistant_content(content=assistant_content)
    assert not suffix.strip(), "regen prefill should not include post-think answer text"
    return f"{prefix}<think>\n{reopen_final_exec(think_text=think_text)}"


def replace_last_assistant_content(
    *, messages: Sequence[Mapping[str, Any]], new_content: str
) -> list[dict[str, Any]]:
    """Replace the last assistant message content in a messages list."""
    updated_messages = [dict(message) for message in messages]
    for index in range(len(updated_messages) - 1, -1, -1):
        if updated_messages[index].get("role") == "assistant":
            updated_messages[index]["content"] = new_content
            return updated_messages
    raise ValueError("Could not find an assistant message to replace.")


def compute_think_token_count(
    *, encoding: tiktoken.Encoding, assistant_content: str
) -> int:
    """Count tokens in the inner `<think>` block for one assistant message."""
    _, think_text, _ = split_assistant_content(content=assistant_content)
    return len(encoding.encode(think_text))


def build_intervention_pools(
    interventions_obj: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split interventions into insert-capable and bridge/redirect-capable pools."""
    insert_specs: list[dict[str, Any]] = []
    late_specs: list[dict[str, Any]] = []
    for raw_spec in interventions_obj["interventions"]:
        spec = apply_sampling_defaults_to_spec(
            interventions_obj=interventions_obj,
            raw_spec=raw_spec,
        )
        allowed_modes = set(spec.get("allowed_editor_modes", []))
        if "insert" in allowed_modes:
            insert_specs.append(spec)
        if allowed_modes.intersection({"bridge", "redirect"}):
            late_specs.append(spec)
    return insert_specs, late_specs


def make_slot_segments(total_pairs: int, segment_count: int) -> list[list[int]]:
    """Partition candidate cut slots into ordered contiguous segments."""
    candidates = list(range(1, total_pairs))
    assert (
        len(candidates) >= segment_count
    ), "Not enough pair boundaries for the requested segments."

    segments: list[list[int]] = []
    for segment_index in range(segment_count):
        start = len(candidates) * segment_index // segment_count
        end = len(candidates) * (segment_index + 1) // segment_count
        segment = candidates[start:end]
        assert segment, "Encountered an empty slot segment."
        segments.append(segment)
    return segments


def choose_insert_slots(
    total_pairs: int, insert_count: int, rng: random.Random
) -> list[int]:
    """Choose well-separated insert slots across the original trace."""
    segments = make_slot_segments(total_pairs=total_pairs, segment_count=insert_count)
    return sorted(rng.choice(segment) for segment in segments)


def choose_bridge_redirect_slots(
    total_pairs: int, rng: random.Random
) -> tuple[int, int]:
    """Choose an early insert slot and a later bridge/redirect slot."""
    segments = make_slot_segments(total_pairs=total_pairs, segment_count=3)
    prefix_slot = rng.choice(segments[0])
    late_slot = rng.choice(segments[-1])
    return prefix_slot, late_slot


def choose_distinct_specs(
    *, pool: Sequence[Mapping[str, Any]], count: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Sample intervention specs with no duplicates when the pool is large enough."""
    if len(pool) >= count:
        return [dict(spec) for spec in rng.sample(list(pool), count)]
    return [dict(rng.choice(list(pool))) for _ in range(count)]


def choose_restricted_mode(
    *, spec: Mapping[str, Any], allowed_modes: set[str], rng: random.Random
) -> Literal["bridge", "redirect"]:
    """Choose a bridge/redirect mode using the spec's mode weights."""
    choices = [
        (mode, float(weight))
        for mode, weight in spec.get("mode_weight_hint", {}).items()
        if mode in allowed_modes and weight > 0
    ]
    if not choices:
        recommended = str(spec.get("recommended_editor_mode", "bridge"))
        if recommended in allowed_modes:
            return recommended  # type: ignore[return-value]
        return sorted(allowed_modes)[0]  # type: ignore[return-value]

    total_weight = sum(weight for _, weight in choices)
    sample = rng.random() * total_weight
    cumulative = 0.0
    for mode, weight in choices:
        cumulative += weight
        if sample <= cumulative:
            return mode  # type: ignore[return-value]
    return choices[-1][0]  # type: ignore[return-value]


def build_step_plan(
    *,
    spec: Mapping[str, Any],
    mode: Literal["insert", "bridge", "redirect"],
    cut_after_pairs: int,
    rng: random.Random,
) -> StepPlan:
    """Construct a `StepPlan` from one intervention spec."""
    return StepPlan(
        intervention_name=str(spec["name"]),
        mode=mode,
        variant=capitalize_first_word(
            choose_variant(intervention_spec=dict(spec), rng=rng)
        ),
        pairs_generated=choose_pairs_to_generate(intervention_spec=dict(spec), rng=rng),
        cut_after_pairs=cut_after_pairs,
        post_splice_policy=str(spec["post_splice_policy"]),
        category=str(spec.get("category", "")),
        steer_token_limit=(
            int(spec["steer_token_limit"])
            if spec.get("steer_token_limit") is not None
            else None
        ),
    )


def build_insert_only_plan(
    *,
    record_index: int,
    record: Mapping[str, Any],
    seed: int,
    insert_specs: Sequence[Mapping[str, Any]],
) -> RowPlan:
    """Build a spaced 2-3 insert-only plan for one source row."""
    rng = random.Random(seed)
    total_pairs = len(parse_trace_text(extract_trace_text(dict(record)))) // 2
    insert_count = 2 if rng.random() < 0.5 else 3
    selected_slots = choose_insert_slots(
        total_pairs=total_pairs,
        insert_count=insert_count,
        rng=rng,
    )
    selected_specs = choose_distinct_specs(
        pool=insert_specs, count=insert_count, rng=rng
    )
    steps = tuple(
        build_step_plan(spec=spec, mode="insert", cut_after_pairs=slot, rng=rng)
        for slot, spec in zip(selected_slots, selected_specs)
    )
    return RowPlan(
        source_row_index=record_index,
        source_row_id=str(record["id"]),
        plan_type="insert_only",
        seed=seed,
        steps=steps,
    )


def build_bridge_redirect_plan(
    *,
    record_index: int,
    record: Mapping[str, Any],
    seed: int,
    insert_specs: Sequence[Mapping[str, Any]],
    late_specs: Sequence[Mapping[str, Any]],
) -> RowPlan:
    """Build a plan with one early insert and one later bridge/redirect step."""
    rng = random.Random(seed)
    total_pairs = len(parse_trace_text(extract_trace_text(dict(record)))) // 2
    prefix_slot, late_slot = choose_bridge_redirect_slots(
        total_pairs=total_pairs, rng=rng
    )
    insert_spec = dict(rng.choice(list(insert_specs)))
    late_spec = dict(rng.choice(list(late_specs)))
    late_mode = choose_restricted_mode(
        spec=late_spec,
        allowed_modes=set(late_spec.get("allowed_editor_modes", [])).intersection(
            {"bridge", "redirect"}
        ),
        rng=rng,
    )
    steps = (
        build_step_plan(
            spec=insert_spec, mode="insert", cut_after_pairs=prefix_slot, rng=rng
        ),
        build_step_plan(
            spec=late_spec, mode=late_mode, cut_after_pairs=late_slot, rng=rng
        ),
    )
    return RowPlan(
        source_row_index=record_index,
        source_row_id=str(record["id"]),
        plan_type="bridge_or_redirect",
        seed=seed,
        steps=steps,
    )


async def request_window_object(
    *,
    openrouter: OpenRouterAsyncClient | None,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any],
    mock_intervention: bool,
    required_first_steer: str,
    pairs_generated: int,
    request_retries: int,
) -> dict[str, Any]:
    """Request one structured intervention window with network retries."""
    if mock_intervention:
        return mock_intervention_window(required_first_steer, pairs_generated)

    assert openrouter is not None, "OpenRouter client is required when not mocking."
    last_error: Exception | None = None
    for attempt in range(1, request_retries + 1):
        try:
            return await openrouter.chat_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                json_schema=schema,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < request_retries:
                await asyncio.sleep(float(attempt))
    raise RuntimeError(
        f"OpenRouter request failed after {request_retries} tries: {last_error}"
    )


def build_validation_feedback(errors: Sequence[str]) -> str:
    """Format generation validation errors for the next prompt attempt."""
    return (
        "\n## Validation feedback from the previous attempt\n"
        "Revise the intervention window so that it satisfies all constraints.\n"
        + "\n".join(f"- {error}" for error in errors)
    )


async def generate_validated_window(
    *,
    record: Mapping[str, Any],
    current_blocks: Sequence[TraceBlock],
    prefix_blocks: Sequence[TraceBlock],
    step: StepPlan,
    prompts_dir: str | Path,
    token_counter: TokenCounter,
    exec_token_limit: int,
    style_window_pairs: int,
    openrouter: OpenRouterAsyncClient | None,
    mock_intervention: bool,
    max_attempts: int,
    request_retries: int,
) -> list[TraceBlock]:
    """Generate and validate one intervention window for a planned step."""
    system_prompt = Path(prompts_dir, "system.md").read_text(encoding="utf-8")
    prompt_path = choose_prompt_path(prompts_dir=prompts_dir, mode=step.mode)
    validation_feedback = ""
    generation_errors: list[str] = []

    for _ in range(max_attempts):
        prompt_values = build_prompt_values(
            record=dict(record),
            all_blocks=current_blocks,
            prefix_blocks=prefix_blocks,
            intervention_spec={
                "name": step.intervention_name,
                "category": step.category,
                "steer_token_limit": step.steer_token_limit,
            },
            intervention_variant=step.variant,
            pairs_to_generate_k=step.pairs_generated,
            exec_token_limit=exec_token_limit,
            style_window_pairs=style_window_pairs,
            validation_feedback=validation_feedback,
        )
        user_prompt = render_prompt(path=prompt_path, values=prompt_values)
        raw_window = await request_window_object(
            openrouter=openrouter,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=build_intervention_schema(step.pairs_generated),
            mock_intervention=mock_intervention,
            required_first_steer=step.variant,
            pairs_generated=step.pairs_generated,
            request_retries=request_retries,
        )
        generated_blocks, generation_errors = validate_generated_window(
            obj=raw_window,
            requested_pairs=step.pairs_generated,
            required_first_steer=step.variant,
            token_counter=token_counter,
            exec_token_limit=exec_token_limit,
            steer_token_limit=step.steer_token_limit,
        )
        if not generation_errors:
            return normalize_steer_blocks(blocks=generated_blocks)
        validation_feedback = build_validation_feedback(errors=generation_errors)

    raise ValueError(
        "Generated intervention failed validation: " + "; ".join(generation_errors)
    )


async def decide_suffix_action(
    *,
    step: StepPlan,
    prompts_dir: str | Path,
    current_blocks: Sequence[TraceBlock],
    prefix_blocks: Sequence[TraceBlock],
    generated_blocks: Sequence[TraceBlock],
    openrouter: OpenRouterAsyncClient | None,
    mock_intervention: bool,
    run_bridge_judge: bool,
) -> tuple[Literal["keep_suffix", "regenerate_suffix"], str]:
    """Decide whether the original suffix should remain after a generated step."""
    if step.mode == "insert":
        return "keep_suffix", "insert mode chosen; keep suffix by construction"
    if step.post_splice_policy == "keep_original_suffix":
        return "keep_suffix", "post_splice_policy=keep_original_suffix"
    if step.mode == "redirect":
        return "regenerate_suffix", "redirect mode chosen; suffix regeneration required"
    if not run_bridge_judge:
        return "regenerate_suffix", "bridge judge disabled"

    cut_after_pairs = len(prefix_blocks) // 2
    next_original_steer, next_original_exec_preview = get_next_original_context(
        all_blocks=current_blocks,
        cut_after_pairs=cut_after_pairs,
    )
    try:
        judge = await maybe_call_bridge_judge(
            openrouter=openrouter,
            prompts_dir=prompts_dir,
            trace_prefix=render_trace(prefix_blocks, wrap_think=False),
            inserted_window=render_window(generated_blocks),
            next_original_steer=next_original_steer,
            next_original_exec_preview=next_original_exec_preview,
            mock=mock_intervention,
        )
        return judge.decision, judge.reason
    except Exception as exc:  # noqa: BLE001
        return "regenerate_suffix", f"bridge judge error; falling back to regen: {exc}"


async def apply_step(
    *,
    record: Mapping[str, Any],
    current_blocks: Sequence[TraceBlock],
    step: StepPlan,
    cut_after_pairs_current: int,
    prompts_dir: str | Path,
    token_counter: TokenCounter,
    exec_token_limit: int,
    style_window_pairs: int,
    openrouter: OpenRouterAsyncClient | None,
    mock_intervention: bool,
    max_attempts: int,
    request_retries: int,
    run_bridge_judge: bool,
) -> tuple[ExecutedStep, list[TraceBlock], bool]:
    """Apply one planned step to the current trace blocks."""
    prefix_blocks = slice_pairs(current_blocks, cut_after_pairs_current)
    generated_blocks = await generate_validated_window(
        record=record,
        current_blocks=current_blocks,
        prefix_blocks=prefix_blocks,
        step=step,
        prompts_dir=prompts_dir,
        token_counter=token_counter,
        exec_token_limit=exec_token_limit,
        style_window_pairs=style_window_pairs,
        openrouter=openrouter,
        mock_intervention=mock_intervention,
        max_attempts=max_attempts,
        request_retries=request_retries,
    )
    suffix_decision, bridge_judge_reason = await decide_suffix_action(
        step=step,
        prompts_dir=prompts_dir,
        current_blocks=current_blocks,
        prefix_blocks=prefix_blocks,
        generated_blocks=generated_blocks,
        openrouter=openrouter,
        mock_intervention=mock_intervention,
        run_bridge_judge=run_bridge_judge,
    )
    keep_suffix = suffix_decision == "keep_suffix"
    augmented_prefix, augmented_full = splice_blocks(
        all_blocks=current_blocks,
        prefix_blocks=prefix_blocks,
        new_window_blocks=generated_blocks,
        keep_suffix=keep_suffix,
    )
    next_blocks = augmented_full if keep_suffix else augmented_prefix
    executed_step = ExecutedStep(
        plan=step,
        cut_after_pairs_current=cut_after_pairs_current,
        suffix_decision=suffix_decision,
        bridge_judge_reason=bridge_judge_reason,
        generated_blocks=tuple(
            {"type": block.type, "text": block.text} for block in generated_blocks
        ),
    )
    return executed_step, next_blocks, keep_suffix


def build_augmented_id(*, source_id: str, plan: RowPlan) -> str:
    """Build a deterministic augmented row id from the source id and plan."""
    plan_bytes = json.dumps(plan.to_json(), sort_keys=True).encode("utf-8")
    digest = hashlib.sha1(plan_bytes).hexdigest()[:12]
    return f"{source_id}__aug__{plan.plan_type}__{digest}"


def build_regen_seed(
    *,
    record: Mapping[str, Any],
    assistant_prefill: str,
    vllm_base_url: str,
    vllm_model: str,
) -> dict[str, Any]:
    """Build the continuation seed payload for later tail regeneration."""
    return {
        "user_prompt": coalesce_user_prompt(record=record),
        "assistant_prefill": build_open_exec_prefill(
            assistant_content=assistant_prefill
        ),
        "vllm_base_url": vllm_base_url,
        "vllm_model": vllm_model,
        "note": (
            "This row needs continuation from inside the open <think> prefix. "
            "Do not merge it directly into the final SFT dataset until the tail is regenerated."
        ),
    }


def build_output_row(
    *,
    record: Mapping[str, Any],
    plan: RowPlan,
    executed_steps: Sequence[ExecutedStep],
    assistant_content: str,
    status: Literal["merge_ready", "needs_regen"],
    encoding: tiktoken.Encoding,
    vllm_base_url: str,
    vllm_model: str,
) -> dict[str, Any]:
    """Build one final dataset-format row from a processed plan."""
    output_row = copy.deepcopy(dict(record))
    output_row["id"] = build_augmented_id(source_id=str(record["id"]), plan=plan)
    output_row["messages"] = replace_last_assistant_content(
        messages=record["messages"],
        new_content=assistant_content,
    )
    output_row["think_token_count"] = compute_think_token_count(
        encoding=encoding,
        assistant_content=assistant_content,
    )
    output_row["augmentation_meta"] = {
        "status": status,
        "source_row_id": str(record["id"]),
        "source_row_index": plan.source_row_index,
        "source_dataset_source": str(record.get("dataset_source", "")),
        "plan_type": plan.plan_type,
        "seed": plan.seed,
        "steps": [step.to_json() for step in executed_steps],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if status == "needs_regen":
        output_row["regen_seed"] = build_regen_seed(
            record=record,
            assistant_prefill=assistant_content,
            vllm_base_url=vllm_base_url,
            vllm_model=vllm_model,
        )
    return output_row


async def process_plan(
    *,
    plan: RowPlan,
    record: Mapping[str, Any],
    prompts_dir: str | Path,
    token_counter: TokenCounter,
    exec_token_limit: int,
    style_window_pairs: int,
    openrouter: OpenRouterAsyncClient | None,
    mock_intervention: bool,
    max_attempts: int,
    request_retries: int,
    run_bridge_judge: bool,
    encoding: tiktoken.Encoding,
    vllm_base_url: str,
    vllm_model: str,
) -> RowResult:
    """Apply a full multi-step plan to a source row and build the output row."""
    try:
        assistant_messages = [
            message
            for message in record["messages"]
            if message.get("role") == "assistant"
            and isinstance(message.get("content"), str)
        ]
        assert assistant_messages, "Missing assistant message content."
        assistant_prefix, _, assistant_suffix = split_assistant_content(
            content=str(assistant_messages[-1]["content"])
        )
        current_blocks = parse_trace_text(extract_trace_text(dict(record)))
        executed_steps: list[ExecutedStep] = []
        inserted_pairs_so_far = 0
        status: Literal["merge_ready", "needs_regen"] = "merge_ready"

        for step in plan.steps:
            current_cut = step.cut_after_pairs + inserted_pairs_so_far
            executed_step, next_blocks, keep_suffix = await apply_step(
                record={**dict(record), "task_id": str(record["id"])},
                current_blocks=current_blocks,
                step=step,
                cut_after_pairs_current=current_cut,
                prompts_dir=prompts_dir,
                token_counter=token_counter,
                exec_token_limit=exec_token_limit,
                style_window_pairs=style_window_pairs,
                openrouter=openrouter,
                mock_intervention=mock_intervention,
                max_attempts=max_attempts,
                request_retries=request_retries,
                run_bridge_judge=run_bridge_judge,
            )
            current_blocks = next_blocks
            inserted_pairs_so_far += step.pairs_generated
            executed_steps.append(executed_step)
            if not keep_suffix:
                status = "needs_regen"
                break

        final_suffix = assistant_suffix if status == "merge_ready" else ""
        assistant_content = render_assistant_content(
            prefix=assistant_prefix,
            blocks=current_blocks,
            suffix=final_suffix,
        )
        output_row = build_output_row(
            record=record,
            plan=plan,
            executed_steps=executed_steps,
            assistant_content=assistant_content,
            status=status,
            encoding=encoding,
            vllm_base_url=vllm_base_url,
            vllm_model=vllm_model,
        )
        return RowResult(ok=True, row=output_row, failure=None)
    except Exception as exc:  # noqa: BLE001
        failure = {
            "source_row_index": plan.source_row_index,
            "source_row_id": plan.source_row_id,
            "plan_type": plan.plan_type,
            "plan": plan.to_json(),
            "error": str(exc),
        }
        return RowResult(ok=False, row=None, failure=failure)


async def collect_group_results(
    *,
    group_name: str,
    target_count: int,
    record_indices: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    plan_builder: Any,
    plan_seed_offset: int,
    prompts_dir: str | Path,
    token_counter: TokenCounter,
    exec_token_limit: int,
    style_window_pairs: int,
    openrouter: OpenRouterAsyncClient | None,
    mock_intervention: bool,
    max_attempts: int,
    request_retries: int,
    run_bridge_judge: bool,
    encoding: tiktoken.Encoding,
    vllm_base_url: str,
    vllm_model: str,
    max_concurrency: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Process one plan group until the requested number of successes is reached."""
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scheduled = 0
    active_tasks: set[asyncio.Task[RowResult]] = set()
    group_start = time.monotonic()

    log_group_progress(
        group_name=group_name,
        successes=0,
        failures=0,
        target_count=target_count,
        scheduled=0,
        total_candidates=len(record_indices),
        active_tasks=0,
        elapsed_s=0.0,
        suffix="starting",
    )

    def schedule_next() -> bool:
        nonlocal scheduled
        if scheduled >= len(record_indices):
            return False
        row_index = record_indices[scheduled]
        scheduled += 1
        plan_seed = plan_seed_offset + row_index * 1009
        plan = plan_builder(
            record_index=row_index, record=rows[row_index], seed=plan_seed
        )
        task = asyncio.create_task(
            process_plan(
                plan=plan,
                record=rows[row_index],
                prompts_dir=prompts_dir,
                token_counter=token_counter,
                exec_token_limit=exec_token_limit,
                style_window_pairs=style_window_pairs,
                openrouter=openrouter,
                mock_intervention=mock_intervention,
                max_attempts=max_attempts,
                request_retries=request_retries,
                run_bridge_judge=run_bridge_judge,
                encoding=encoding,
                vllm_base_url=vllm_base_url,
                vllm_model=vllm_model,
            )
        )
        active_tasks.add(task)
        return True

    while len(successes) < target_count:
        while (
            len(active_tasks) < max_concurrency
            and len(successes) + len(active_tasks) < target_count
            and schedule_next()
        ):
            continue
        if not active_tasks:
            break

        done_tasks, _ = await asyncio.wait(
            active_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in done_tasks:
            active_tasks.remove(task)
            result = await task
            if result.ok and result.row is not None:
                successes.append(result.row)
            elif result.failure is not None:
                failures.append(result.failure)
            should_log = (
                bool(result.failure)
                or len(successes) == target_count
                or (len(successes) > 0 and len(successes) % 10 == 0)
            )
            if should_log:
                log_group_progress(
                    group_name=group_name,
                    successes=len(successes),
                    failures=len(failures),
                    target_count=target_count,
                    scheduled=scheduled,
                    total_candidates=len(record_indices),
                    active_tasks=len(active_tasks),
                    elapsed_s=time.monotonic() - group_start,
                )

    if len(successes) < target_count:
        raise RuntimeError(
            f"Only produced {len(successes)} successful rows out of the requested {target_count}."
        )
    log_group_progress(
        group_name=group_name,
        successes=len(successes),
        failures=len(failures),
        target_count=target_count,
        scheduled=scheduled,
        total_candidates=len(record_indices),
        active_tasks=len(active_tasks),
        elapsed_s=time.monotonic() - group_start,
        suffix="finished",
    )
    return successes, failures


def summarize_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Build a compact output summary for the generated rows."""
    status_counts = Counter(
        str(row.get("augmentation_meta", {}).get("status", "unknown")) for row in rows
    )
    plan_type_counts = Counter(
        str(row.get("augmentation_meta", {}).get("plan_type", "unknown"))
        for row in rows
    )
    mode_counts: Counter[str] = Counter()
    for row in rows:
        for step in row.get("augmentation_meta", {}).get("steps", []):
            mode = step.get("plan", {}).get("mode", "")
            if mode:
                mode_counts[str(mode)] += 1
    return {
        "row_count": len(rows),
        "status_counts": dict(status_counts),
        "plan_type_counts": dict(plan_type_counts),
        "step_mode_counts": dict(mode_counts),
    }


def log_group_progress(
    *,
    group_name: str,
    successes: int,
    failures: int,
    target_count: int,
    scheduled: int,
    total_candidates: int,
    active_tasks: int,
    elapsed_s: float,
    suffix: str = "",
) -> None:
    """Print one progress line for a running augmentation group."""
    message = (
        f"[{group_name}] successes={successes}/{target_count} "
        f"failures={failures} scheduled={scheduled}/{total_candidates} "
        f"active={active_tasks} elapsed_s={elapsed_s:.1f}"
    )
    if suffix:
        message = f"{message} {suffix}"
    print(message, flush=True)


async def async_main() -> None:
    """Run the multi-step augmentation job end-to-end."""
    args = build_arg_parser().parse_args()
    target_rows = int(args.target_rows)
    bridge_redirect_target = int(
        round(target_rows * float(args.bridge_redirect_fraction))
    )
    insert_only_target = target_rows - bridge_redirect_target
    assert insert_only_target > 0, "Need at least one insert-only target row."
    assert bridge_redirect_target > 0, "Need at least one bridge/redirect target row."

    rows = load_jsonl(args.source)
    assert (
        len(rows) >= target_rows
    ), "Source dataset is smaller than the requested output size."

    interventions_obj = load_json(args.interventions)
    insert_specs, late_specs = build_intervention_pools(
        interventions_obj=interventions_obj
    )
    assert insert_specs, "Could not find any insert-capable interventions."
    assert late_specs, "Could not find any bridge/redirect-capable interventions."

    token_counter = TokenCounter(args.length_tokenizer)
    encoding = tiktoken.get_encoding("cl100k_base")
    row_order = list(range(len(rows)))
    random.Random(args.seed).shuffle(row_order)
    insert_candidate_count = min(len(rows), insert_only_target + 40)
    bridge_candidate_start = insert_candidate_count
    assert (
        len(row_order) - bridge_candidate_start >= bridge_redirect_target
    ), "Not enough reserved rows for the bridge/redirect target count."

    api_key = None if args.mock_intervention else resolve_openrouter_key(args=args)
    openrouter: OpenRouterAsyncClient | None = None
    if not args.mock_intervention:
        if not api_key:
            raise SystemExit(
                "OpenRouter API key is required unless --mock-intervention is used."
            )
        provider_data_collection = (
            None
            if args.provider_data_collection == "none"
            else args.provider_data_collection
        )
        openrouter = OpenRouterAsyncClient(
            api_key=api_key,
            model=args.openrouter_model,
            site_url=args.openrouter_site_url,
            site_name=args.openrouter_site_name,
            timeout_s=args.openrouter_timeout_s,
            use_response_healing=args.use_response_healing,
            provider_data_collection=provider_data_collection,
            temperature=args.openrouter_temperature,
            max_tokens=args.openrouter_max_tokens,
        )

    try:
        insert_candidates = row_order[:insert_candidate_count]
        bridge_candidates = row_order[bridge_candidate_start:]
        insert_rows, insert_failures = await collect_group_results(
            group_name="insert_only",
            target_count=insert_only_target,
            record_indices=insert_candidates,
            rows=rows,
            plan_builder=lambda **kwargs: build_insert_only_plan(
                insert_specs=insert_specs,
                **kwargs,
            ),
            plan_seed_offset=args.seed * 10_000,
            prompts_dir=args.prompts_dir,
            token_counter=token_counter,
            exec_token_limit=args.exec_token_limit,
            style_window_pairs=args.style_window_pairs,
            openrouter=openrouter,
            mock_intervention=args.mock_intervention,
            max_attempts=args.max_attempts,
            request_retries=args.request_retries,
            run_bridge_judge=args.run_bridge_judge,
            encoding=encoding,
            vllm_base_url=args.vllm_base_url,
            vllm_model=args.vllm_model,
            max_concurrency=max(1, int(args.max_concurrency)),
        )
        bridge_rows, bridge_failures = await collect_group_results(
            group_name="bridge_or_redirect",
            target_count=bridge_redirect_target,
            record_indices=bridge_candidates,
            rows=rows,
            plan_builder=lambda **kwargs: build_bridge_redirect_plan(
                insert_specs=insert_specs,
                late_specs=late_specs,
                **kwargs,
            ),
            plan_seed_offset=args.seed * 20_000,
            prompts_dir=args.prompts_dir,
            token_counter=token_counter,
            exec_token_limit=args.exec_token_limit,
            style_window_pairs=args.style_window_pairs,
            openrouter=openrouter,
            mock_intervention=args.mock_intervention,
            max_attempts=args.max_attempts,
            request_retries=args.request_retries,
            run_bridge_judge=args.run_bridge_judge,
            encoding=encoding,
            vllm_base_url=args.vllm_base_url,
            vllm_model=args.vllm_model,
            max_concurrency=max(1, int(args.max_concurrency)),
        )
    finally:
        if openrouter is not None:
            await openrouter.aclose()

    all_rows = insert_rows + bridge_rows
    all_rows.sort(key=lambda row: str(row["id"]))
    failures = insert_failures + bridge_failures
    failures.sort(
        key=lambda failure: (failure["source_row_index"], failure["source_row_id"])
    )

    merge_ready_rows = [
        row
        for row in all_rows
        if row.get("augmentation_meta", {}).get("status") == "merge_ready"
    ]
    regen_rows = [
        row
        for row in all_rows
        if row.get("augmentation_meta", {}).get("status") == "needs_regen"
    ]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_stem = f"augmented_{target_rows}"
    all_path = output_dir / f"{file_stem}_all.jsonl"
    merge_ready_path = output_dir / f"{file_stem}_merge_ready.jsonl"
    regen_path = output_dir / f"{file_stem}_needs_regen.jsonl"
    rejected_path = output_dir / f"{file_stem}.rejected.jsonl"
    summary_path = output_dir / f"{file_stem}.summary.json"

    dump_jsonl(all_rows, all_path)
    dump_jsonl(merge_ready_rows, merge_ready_path)
    dump_jsonl(regen_rows, regen_path)
    dump_jsonl(failures, rejected_path)

    summary = {
        "source": str(Path(args.source)),
        "target_rows": target_rows,
        "insert_only_target": insert_only_target,
        "bridge_redirect_target": bridge_redirect_target,
        "token_counter_method": token_counter.method,
        "mock_intervention": args.mock_intervention,
        "summary_all": summarize_rows(rows=all_rows),
        "summary_merge_ready": summarize_rows(rows=merge_ready_rows),
        "summary_needs_regen": summarize_rows(rows=regen_rows),
        "rejected_count": len(failures),
        "all_output": str(all_path),
        "merge_ready_output": str(merge_ready_path),
        "needs_regen_output": str(regen_path),
        "rejected_output": str(rejected_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    """CLI entrypoint."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
