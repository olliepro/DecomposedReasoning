from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import tiktoken

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.non_sequitur_builder import (  # noqa: E402
    NonSequiturPlan,
    build_non_sequitur_plan,
    build_output_row_multi,
    choose_insertion_count,
    choose_uniform_cut_after_pairs_batch,
)
from src.non_sequitur_source import (  # noqa: E402
    PreparedRow,
    SourcePrepSummary,
    prepare_source_rows,
)
from src.row_utils import normalize_steer_blocks  # noqa: E402
from src.trace_augmentor import (  # noqa: E402
    OpenRouterAsyncClient,
    TokenCounter,
    TraceBlock,
    build_intervention_schema,
    build_prompt_values,
    choose_intervention,
    choose_prompt_path,
    dump_jsonl,
    load_json,
    load_jsonl,
    mock_intervention_window,
    render_prompt,
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
class RowResult:
    """Result payload for one processed non-sequitur source row.

    Args:
        ok: Whether processing succeeded.
        row: Final dataset-format row when successful.
        failure: Failure payload when generation or validation failed.
    """

    ok: bool
    row: dict[str, Any] | None
    failure: dict[str, Any] | None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for non-sequitur dataset generation.

    Returns:
        Configured argument parser.

    Example:
        >>> parser = build_arg_parser()
        >>> parser.prog
        'build_non_sequitur_dataset.py'
    """

    parser = argparse.ArgumentParser(
        description="Build an insert-only non-sequitur augmentation dataset."
    )
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "out" / "non_sequitur_insert_only"),
    )
    parser.add_argument("--interventions", default=str(ROOT / "non-sequiturs.json"))
    parser.add_argument("--prompts-dir", default=str(ROOT / "prompts"))
    parser.add_argument("--env-file", default=str(ROOT.parent.parent / ".env"))
    parser.add_argument("--target-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--min-insertions-per-row", type=int, default=1)
    parser.add_argument("--max-insertions-per-row", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=20)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--request-retries", type=int, default=3)
    parser.add_argument("--style-window-pairs", type=int, default=2)
    parser.add_argument("--exec-token-limit", type=int, default=512)
    parser.add_argument("--max-source-tokens", type=int, default=14000)
    parser.add_argument("--length-tokenizer", default=None)
    parser.add_argument("--mock-intervention", action="store_true")
    parser.add_argument("--openrouter-model", default="openai/gpt-oss-20b")
    parser.add_argument("--openrouter-timeout-s", type=int, default=180)
    parser.add_argument("--openrouter-temperature", type=float, default=0.4)
    parser.add_argument("--openrouter-max-tokens", type=int, default=1200)
    parser.add_argument("--openrouter-site-url", default=None)
    parser.add_argument(
        "--openrouter-site-name",
        default="steer-exec-non-sequitur-augmentor",
    )
    parser.add_argument("--provider-data-collection", default="deny")
    parser.add_argument("--use-response-healing", action="store_true")
    return parser


def load_env_values(path: str | Path | None) -> dict[str, str]:
    """Load dotenv-style values from disk.

    Args:
        path: Optional dotenv path.

    Returns:
        Parsed environment values, or an empty mapping when absent.
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


def resolve_openrouter_key(*, env_file: str | Path | None) -> str | None:
    """Resolve the OpenRouter API key from process env or dotenv file.

    Args:
        env_file: Optional dotenv path to inspect.

    Returns:
        API key string when found, otherwise None.
    """

    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPEN_ROUTER_KEY")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
        return api_key
    env_values = load_env_values(path=env_file)
    api_key = env_values.get("OPENROUTER_API_KEY") or env_values.get("OPEN_ROUTER_KEY")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
    return api_key


def validate_interventions_obj(interventions_obj: Mapping[str, Any]) -> None:
    """Assert that the non-sequitur catalog matches runner expectations.

    Args:
        interventions_obj: Parsed intervention catalog.

    Returns:
        None. Raises on incompatible catalog shape.
    """

    interventions = interventions_obj.get("interventions")
    assert isinstance(interventions, list), "Expected interventions list."
    assert len(interventions) == 25, "Expected exactly 25 non-sequitur themes."
    sampling_defaults = interventions_obj.get("sampling_defaults", {})
    assert (
        sampling_defaults.get("first_new_steer_should_match_variant_exactly") is False
    ), "Non-sequitur catalog should not require exact first-steer matching."
    steer_token_limit = int(sampling_defaults.get("steer_token_limit", 15))
    assert steer_token_limit == 15, "Expected a 15-token steer cap."
    encoding = tiktoken.get_encoding("cl100k_base")
    for spec in interventions:
        assert spec["recommended_editor_mode"] == "insert"
        assert spec["allowed_editor_modes"] == ["insert"]
        assert spec["post_splice_policy"] == "keep_original_suffix"
        assert spec["suffix_handling_hint"] == "keep_suffix_without_judge"
        assert spec["preferred_slots"] == []
        assert spec["prompt_template"] == "non_sequitur_insert.md"
        for variant in spec["variants"]:
            assert (
                not variant.strip().lower().startswith("count")
            ), f"Count nonsequitur variant remains: {variant}"
            assert (
                len(encoding.encode(variant)) <= steer_token_limit
            ), f"Variant exceeds {steer_token_limit} tokens: {variant}"


def validate_insertion_range(*, min_insertions: int, max_insertions: int) -> None:
    """Assert that the requested insertion-count range is valid.

    Args:
        min_insertions: Minimum insertions per row.
        max_insertions: Maximum insertions per row.

    Returns:
        None. Raises on invalid ranges.
    """

    assert min_insertions >= 1, "Need at least one insertion per row."
    assert max_insertions >= min_insertions, "Insertion max must be >= min."


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
    """Request one structured intervention window with retries.

    Args:
        openrouter: OpenRouter client when live generation is enabled.
        system_prompt: Shared system prompt.
        user_prompt: Rendered user prompt for this insertion.
        schema: Structured output schema.
        mock_intervention: Whether to bypass the network call.
        required_first_steer: Reference steer text used by the mock path.
        pairs_generated: Number of steer/exec pairs to request.
        request_retries: Maximum API retry count.

    Returns:
        Structured intervention window payload.
    """

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


def build_validation_feedback(*, errors: Sequence[str]) -> str:
    """Format validator errors for a follow-up generation attempt.

    Args:
        errors: Validation error strings from the previous attempt.

    Returns:
        Prompt-ready validation feedback block.
    """

    return (
        "\n## Validation feedback from the previous attempt\n"
        "Revise the intervention window so that it satisfies all constraints.\n"
        + "\n".join(f"- {error}" for error in errors)
    )


async def generate_validated_window(
    *,
    row: Mapping[str, Any],
    all_blocks: Sequence[TraceBlock],
    plan: NonSequiturPlan,
    intervention_spec: Mapping[str, Any],
    prompts_dir: str | Path,
    token_counter: TokenCounter,
    exec_token_limit: int,
    style_window_pairs: int,
    openrouter: OpenRouterAsyncClient | None,
    mock_intervention: bool,
    max_attempts: int,
    request_retries: int,
) -> list[TraceBlock]:
    """Generate and validate one non-sequitur intervention window.

    Args:
        row: Source dataset row.
        all_blocks: Current trace blocks before this insertion.
        plan: Chosen insertion plan for this row.
        intervention_spec: Full intervention spec from `non-sequiturs.json`.
        prompts_dir: Prompt template directory.
        token_counter: Exec token counter.
        exec_token_limit: Exclusive per-exec token ceiling.
        style_window_pairs: Number of prefix pairs used for style matching.
        openrouter: Optional OpenRouter client.
        mock_intervention: Whether to use local mock output.
        max_attempts: Maximum structured-generation attempts.
        request_retries: Maximum API retry count per attempt.

    Returns:
        Normalized generated blocks.
    """

    system_prompt = Path(prompts_dir, "system.md").read_text(encoding="utf-8")
    prompt_path = choose_prompt_path(
        prompts_dir=prompts_dir,
        mode="insert",
        intervention_spec=intervention_spec,
    )
    prefix_blocks = slice_pairs(all_blocks, plan.cut_after_pairs)
    validation_feedback = ""
    generation_errors: list[str] = []
    for _ in range(max_attempts):
        prompt_values = build_prompt_values(
            record=dict(row),
            all_blocks=all_blocks,
            prefix_blocks=prefix_blocks,
            intervention_spec=dict(intervention_spec),
            intervention_variant=plan.variant,
            pairs_to_generate_k=plan.pairs_generated,
            exec_token_limit=exec_token_limit,
            style_window_pairs=style_window_pairs,
            validation_feedback=validation_feedback,
        )
        user_prompt = render_prompt(path=prompt_path, values=prompt_values)
        raw_window = await request_window_object(
            openrouter=openrouter,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=build_intervention_schema(plan.pairs_generated),
            mock_intervention=mock_intervention,
            required_first_steer=plan.variant,
            pairs_generated=plan.pairs_generated,
            request_retries=request_retries,
        )
        generated_blocks, generation_errors = validate_generated_window(
            obj=raw_window,
            requested_pairs=plan.pairs_generated,
            required_first_steer=plan.variant,
            enforce_first_steer_exact=False,
            token_counter=token_counter,
            exec_token_limit=exec_token_limit,
            steer_token_limit=intervention_spec.get("steer_token_limit"),
        )
        if not generation_errors:
            return normalize_steer_blocks(blocks=generated_blocks)
        validation_feedback = build_validation_feedback(errors=generation_errors)
    raise ValueError(
        "Generated intervention failed validation: " + "; ".join(generation_errors)
    )


async def process_prepared_row(
    *,
    prepared_row: PreparedRow,
    interventions_obj: Mapping[str, Any],
    seed: int,
    min_insertions_per_row: int,
    max_insertions_per_row: int,
    prompts_dir: str | Path,
    token_counter: TokenCounter,
    exec_token_limit: int,
    style_window_pairs: int,
    openrouter: OpenRouterAsyncClient | None,
    mock_intervention: bool,
    max_attempts: int,
    request_retries: int,
    encoding: tiktoken.Encoding,
) -> RowResult:
    """Process one filtered source row into a merge-ready augmented row.

    Args:
        prepared_row: Filtered source row and parsed trace blocks.
        interventions_obj: Parsed non-sequitur catalog.
        seed: Deterministic per-row seed.
        min_insertions_per_row: Minimum insertions per row.
        max_insertions_per_row: Maximum insertions per row.
        prompts_dir: Prompt directory.
        token_counter: Exec token counter.
        exec_token_limit: Exclusive per-exec token ceiling.
        style_window_pairs: Local style window length in pairs.
        openrouter: Optional OpenRouter client.
        mock_intervention: Whether to use mock generation.
        max_attempts: Maximum structured-generation attempts.
        request_retries: Maximum API retries per attempt.
        encoding: `cl100k_base` encoding for output row token counts.

    Returns:
        Success row or failure payload.
    """

    plans: list[NonSequiturPlan] = []
    try:
        row_rng = random.Random(seed)
        current_blocks = list(prepared_row.parsed_blocks)
        total_pairs = len(prepared_row.parsed_blocks) // 2
        insertion_count = choose_insertion_count(
            min_insertions=min_insertions_per_row,
            max_insertions=max_insertions_per_row,
            total_pairs=total_pairs,
            rng=row_rng,
        )
        original_cuts = choose_uniform_cut_after_pairs_batch(
            total_pairs=total_pairs,
            insertion_count=insertion_count,
            rng=row_rng,
        )
        generated_windows: list[list[TraceBlock]] = []
        for step_index, original_cut in enumerate(original_cuts):
            current_cut = original_cut + sum(
                prior_plan.pairs_generated for prior_plan in plans
            )
            intervention_spec = dict(
                choose_intervention(dict(interventions_obj), row_rng)
            )
            plan = build_non_sequitur_plan(
                record_index=prepared_row.row_index,
                record=prepared_row.row,
                parsed_blocks=current_blocks,
                intervention_spec=intervention_spec,
                seed=seed * 100 + step_index,
                cut_after_pairs=current_cut,
            )
            generated_blocks = await generate_validated_window(
                row=prepared_row.row,
                all_blocks=current_blocks,
                plan=plan,
                intervention_spec=intervention_spec,
                prompts_dir=prompts_dir,
                token_counter=token_counter,
                exec_token_limit=exec_token_limit,
                style_window_pairs=style_window_pairs,
                openrouter=openrouter,
                mock_intervention=mock_intervention,
                max_attempts=max_attempts,
                request_retries=request_retries,
            )
            prefix_blocks = slice_pairs(current_blocks, plan.cut_after_pairs)
            _, current_blocks = splice_blocks(
                all_blocks=current_blocks,
                prefix_blocks=prefix_blocks,
                new_window_blocks=generated_blocks,
                keep_suffix=True,
            )
            plans.append(plan)
            generated_windows.append(generated_blocks)
        output_row = build_output_row_multi(
            record=prepared_row.row,
            plans=plans,
            generated_windows=generated_windows,
            augmented_blocks=current_blocks,
            encoding=encoding,
        )
        return RowResult(ok=True, row=output_row, failure=None)
    except Exception as exc:  # noqa: BLE001
        failure = {
            "source_row_index": prepared_row.row_index,
            "source_row_id": str(prepared_row.row.get("id", "")),
            "plan": None if not plans else [plan.to_plan_json() for plan in plans],
            "error": str(exc),
        }
        return RowResult(ok=False, row=None, failure=failure)


def log_progress(
    *,
    successes: int,
    failures: int,
    target_count: int,
    scheduled: int,
    total_candidates: int,
    active_tasks: int,
    elapsed_s: float,
    suffix: str = "",
) -> None:
    """Print one progress line for the running non-sequitur job.

    Args:
        successes: Successful rows produced so far.
        failures: Failed rows so far.
        target_count: Requested number of successes.
        scheduled: Number of rows scheduled so far.
        total_candidates: Size of the eligible candidate pool.
        active_tasks: In-flight task count.
        elapsed_s: Elapsed wall-clock seconds.
        suffix: Optional trailing status note.

    Returns:
        None.
    """

    message = (
        f"[non_sequitur] successes={successes}/{target_count} "
        f"failures={failures} scheduled={scheduled}/{total_candidates} "
        f"active={active_tasks} elapsed_s={elapsed_s:.1f}"
    )
    if suffix:
        message = f"{message} {suffix}"
    print(message, flush=True)


async def collect_results(
    *,
    prepared_rows: Sequence[PreparedRow],
    target_count: int,
    interventions_obj: Mapping[str, Any],
    prompts_dir: str | Path,
    token_counter: TokenCounter,
    exec_token_limit: int,
    style_window_pairs: int,
    openrouter: OpenRouterAsyncClient | None,
    mock_intervention: bool,
    max_attempts: int,
    request_retries: int,
    encoding: tiktoken.Encoding,
    max_concurrency: int,
    seed: int,
    min_insertions_per_row: int,
    max_insertions_per_row: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run non-sequitur augmentation until the target success count is reached.

    Args:
        prepared_rows: Eligible source rows in sampling order.
        target_count: Requested number of successful outputs.
        interventions_obj: Parsed non-sequitur catalog.
        prompts_dir: Prompt template directory.
        token_counter: Exec token counter.
        exec_token_limit: Exclusive per-exec token ceiling.
        style_window_pairs: Local style window length in pairs.
        openrouter: Optional OpenRouter client.
        mock_intervention: Whether to use mock generation.
        max_attempts: Maximum structured-generation attempts.
        request_retries: Maximum API retries per attempt.
        encoding: `cl100k_base` encoding for output rows.
        max_concurrency: Maximum in-flight rows.
        seed: Global deterministic seed.
        min_insertions_per_row: Minimum insertions per row.
        max_insertions_per_row: Maximum insertions per row.

    Returns:
        Successful output rows and failure payloads.
    """

    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scheduled = 0
    active_tasks: set[asyncio.Task[RowResult]] = set()
    start_time = time.monotonic()
    log_progress(
        successes=0,
        failures=0,
        target_count=target_count,
        scheduled=0,
        total_candidates=len(prepared_rows),
        active_tasks=0,
        elapsed_s=0.0,
        suffix="starting",
    )

    def schedule_next() -> bool:
        nonlocal scheduled
        if scheduled >= len(prepared_rows):
            return False
        prepared_row = prepared_rows[scheduled]
        scheduled += 1
        row_seed = seed * 10_000 + prepared_row.row_index * 1009
        task = asyncio.create_task(
            process_prepared_row(
                prepared_row=prepared_row,
                interventions_obj=interventions_obj,
                seed=row_seed,
                min_insertions_per_row=min_insertions_per_row,
                max_insertions_per_row=max_insertions_per_row,
                prompts_dir=prompts_dir,
                token_counter=token_counter,
                exec_token_limit=exec_token_limit,
                style_window_pairs=style_window_pairs,
                openrouter=openrouter,
                mock_intervention=mock_intervention,
                max_attempts=max_attempts,
                request_retries=request_retries,
                encoding=encoding,
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
                log_progress(
                    successes=len(successes),
                    failures=len(failures),
                    target_count=target_count,
                    scheduled=scheduled,
                    total_candidates=len(prepared_rows),
                    active_tasks=len(active_tasks),
                    elapsed_s=time.monotonic() - start_time,
                )

    if len(successes) < target_count:
        raise RuntimeError(
            f"Only produced {len(successes)} rows out of the requested {target_count}."
        )
    log_progress(
        successes=len(successes),
        failures=len(failures),
        target_count=target_count,
        scheduled=scheduled,
        total_candidates=len(prepared_rows),
        active_tasks=len(active_tasks),
        elapsed_s=time.monotonic() - start_time,
        suffix="finished",
    )
    return successes, failures


def summarize_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    prep_summary: SourcePrepSummary,
    target_rows: int,
    max_source_tokens: int,
    failures: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a compact summary for the non-sequitur output dataset.

    Args:
        rows: Successful merge-ready augmented rows.
        prep_summary: Source filtering summary.
        target_rows: Requested number of output rows.
        max_source_tokens: Strict source filter threshold.
        failures: Failure payloads recorded during processing.

    Returns:
        JSON-friendly dataset summary.
    """

    status_counts = Counter(
        str(row.get("augmentation_meta", {}).get("status", "unknown")) for row in rows
    )
    theme_counts = Counter(
        str(step["plan"]["intervention_name"])
        for row in rows
        for step in row["augmentation_meta"]["steps"]
    )
    insertion_counts = Counter(len(row["augmentation_meta"]["steps"]) for row in rows)
    pair_counts = Counter(
        str(step["plan"]["pairs_generated"])
        for row in rows
        for step in row["augmentation_meta"]["steps"]
    )
    steer_counts = Counter(
        len(row["augmentation_meta"]["mask_targets"]["generated_steer_texts"])
        for row in rows
    )
    return {
        "row_count": len(rows),
        "target_rows": target_rows,
        "max_source_tokens": max_source_tokens,
        "source_prep": prep_summary.to_json(),
        "status_counts": dict(status_counts),
        "insertion_count_distribution": {
            str(k): v for k, v in insertion_counts.items()
        },
        "theme_counts": dict(theme_counts),
        "pair_count_distribution": dict(pair_counts),
        "masked_steer_count_distribution": {str(k): v for k, v in steer_counts.items()},
        "failure_count": len(failures),
    }


async def async_main() -> None:
    """Run the non-sequitur augmentation job end-to-end.

    Returns:
        None. Writes JSONL artifacts and prints progress lines.
    """

    args = build_arg_parser().parse_args()
    validate_insertion_range(
        min_insertions=int(args.min_insertions_per_row),
        max_insertions=int(args.max_insertions_per_row),
    )
    rows = load_jsonl(args.source)
    interventions_obj = load_json(args.interventions)
    validate_interventions_obj(interventions_obj=interventions_obj)

    encoding = tiktoken.get_encoding("cl100k_base")
    token_counter = TokenCounter(args.length_tokenizer)
    prepared_rows, prep_summary = prepare_source_rows(
        rows=rows,
        encoding=encoding,
        max_source_tokens=int(args.max_source_tokens),
    )
    if prep_summary.eligible_rows < int(args.target_rows):
        raise SystemExit(
            f"Only {prep_summary.eligible_rows} rows are eligible under the "
            f"< {args.max_source_tokens} token filter, but {args.target_rows} were requested."
        )
    random.Random(args.seed).shuffle(prepared_rows)

    openrouter: OpenRouterAsyncClient | None = None
    if not args.mock_intervention:
        api_key = resolve_openrouter_key(env_file=args.env_file)
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
        success_rows, failures = await collect_results(
            prepared_rows=prepared_rows,
            target_count=int(args.target_rows),
            interventions_obj=interventions_obj,
            prompts_dir=args.prompts_dir,
            token_counter=token_counter,
            exec_token_limit=int(args.exec_token_limit),
            style_window_pairs=int(args.style_window_pairs),
            openrouter=openrouter,
            mock_intervention=bool(args.mock_intervention),
            max_attempts=int(args.max_attempts),
            request_retries=int(args.request_retries),
            encoding=encoding,
            max_concurrency=max(1, int(args.max_concurrency)),
            seed=int(args.seed),
            min_insertions_per_row=int(args.min_insertions_per_row),
            max_insertions_per_row=int(args.max_insertions_per_row),
        )
    finally:
        if openrouter is not None:
            await openrouter.aclose()

    success_rows.sort(key=lambda row: str(row.get("id", "")))
    summary = summarize_rows(
        rows=success_rows,
        prep_summary=prep_summary,
        target_rows=int(args.target_rows),
        max_source_tokens=int(args.max_source_tokens),
        failures=failures,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_rows = int(args.target_rows)
    all_path = output_dir / f"augmented_{target_rows}_all.jsonl"
    merge_ready_path = output_dir / f"augmented_{target_rows}_merge_ready.jsonl"
    rejected_path = output_dir / f"augmented_{target_rows}.rejected.jsonl"
    summary_path = output_dir / f"augmented_{target_rows}.summary.json"
    dump_jsonl(success_rows, all_path)
    dump_jsonl(success_rows, merge_ready_path)
    dump_jsonl(failures, rejected_path)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {len(success_rows)} rows to {all_path}", flush=True)
    print(f"wrote summary to {summary_path}", flush=True)


def main() -> None:
    """Run the async non-sequitur dataset builder from a sync entrypoint.

    Returns:
        None.
    """

    asyncio.run(async_main())


if __name__ == "__main__":
    main()
