from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.trace_augmentor import (
    OpenRouterAsyncClient,
    TokenCounter,
    augment_record,
    choose_intervention,
    dump_jsonl,
    load_json,
    load_jsonl,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Augment interleaved <steer>/<exec> traces with intervention windows.")
    p.add_argument("--input", required=True, help="Input JSONL file.")
    p.add_argument("--output", required=True, help="Output JSONL file.")
    p.add_argument("--interventions", required=True, help="Path to interventions.json.")
    p.add_argument("--prompts-dir", required=True, help="Directory containing prompt templates.")
    p.add_argument("--augmentations-per-record", type=int, default=1, help="How many augmented variants to produce per input record.")
    p.add_argument("--intervention-name", default=None, help="Optional fixed intervention name. If omitted, sample randomly.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-records", type=int, default=0, help="If > 0, process only the first N records.")
    p.add_argument("--max-attempts", type=int, default=3, help="Retries for intervention generation when validation fails.")
    p.add_argument("--max-concurrency", type=int, default=8, help="Maximum number of concurrent OpenRouter augmentation tasks.")
    p.add_argument("--style-window-pairs", type=int, default=2)
    p.add_argument("--exec-token-limit", type=int, default=512)
    p.add_argument("--length-tokenizer", default=None, help="Optional tokenizer name/path for exact-ish exec length checks.")
    p.add_argument("--wrap-think", action="store_true", help="Wrap emitted augmented traces in <think>...</think>.")
    p.add_argument("--run-bridge-judge", action="store_true", help="Use the judge prompt for bridge-mode suffix decisions.")
    p.add_argument("--mock-intervention", action="store_true", help="Do not call OpenRouter; emit deterministic mock interventions.")
    p.add_argument("--use-response-healing", action="store_true", help="Enable OpenRouter's response-healing plugin.")
    p.add_argument("--openrouter-api-key", default=os.environ.get("OPENROUTER_API_KEY"), help="OpenRouter API key, or set OPENROUTER_API_KEY.")
    p.add_argument("--openrouter-model", default="openai/gpt-oss-20b")
    p.add_argument("--openrouter-site-url", default=None)
    p.add_argument("--openrouter-site-name", default="steer-exec-augmentor")
    p.add_argument("--openrouter-timeout-s", type=int, default=120)
    p.add_argument("--openrouter-temperature", type=float, default=0.4)
    p.add_argument("--openrouter-max-tokens", type=int, default=1200)
    p.add_argument("--provider-data-collection", default="deny", choices=["allow", "deny", "none"], help="OpenRouter provider.data_collection. Use 'none' to omit.")
    p.add_argument("--vllm-base-url", default="http://localhost:8000/v1", help="Only written into regen_seed metadata.")
    p.add_argument("--vllm-model", default="", help="Only written into regen_seed metadata.")
    return p


async def process_one(
    *,
    record_idx: int,
    aug_idx: int,
    record: Dict[str, Any],
    task_seed: int,
    args: argparse.Namespace,
    interventions_obj: Dict[str, Any],
    token_counter: TokenCounter,
    openrouter: OpenRouterAsyncClient | None,
) -> Tuple[bool, Dict[str, Any]]:
    local_rng = random.Random(task_seed)
    try:
        intervention_spec = choose_intervention(interventions_obj, local_rng, name=args.intervention_name)
        augmented = await augment_record(
            record=record,
            intervention_spec=intervention_spec,
            prompts_dir=args.prompts_dir,
            rng=local_rng,
            token_counter=token_counter,
            exec_token_limit=args.exec_token_limit,
            style_window_pairs=args.style_window_pairs,
            openrouter=openrouter,
            mock_intervention=args.mock_intervention,
            wrap_think=args.wrap_think,
            vllm_base_url=args.vllm_base_url,
            vllm_model=args.vllm_model,
            max_attempts=args.max_attempts,
            run_bridge_judge=args.run_bridge_judge,
        )
        augmented["source_record_index"] = record_idx
        augmented["augmentation_index"] = aug_idx
        return True, augmented
    except Exception as e:
        rejected = {
            "source_record_index": record_idx,
            "augmentation_index": aug_idx,
            "task_id": record.get("task_id", ""),
            "error": str(e),
        }
        return False, rejected


async def async_main() -> None:
    args = build_arg_parser().parse_args()
    seed_rng = random.Random(args.seed)

    interventions_obj = load_json(args.interventions)
    rows = load_jsonl(args.input)
    if args.max_records > 0:
        rows = rows[: args.max_records]

    token_counter = TokenCounter(args.length_tokenizer)

    openrouter = None
    if not args.mock_intervention:
        if not args.openrouter_api_key:
            raise SystemExit("OpenRouter API key is required unless --mock-intervention is used.")
        provider_data_collection = None if args.provider_data_collection == "none" else args.provider_data_collection
        openrouter = OpenRouterAsyncClient(
            api_key=args.openrouter_api_key,
            model=args.openrouter_model,
            site_url=args.openrouter_site_url,
            site_name=args.openrouter_site_name,
            timeout_s=args.openrouter_timeout_s,
            use_response_healing=args.use_response_healing,
            provider_data_collection=provider_data_collection,
            temperature=args.openrouter_temperature,
            max_tokens=args.openrouter_max_tokens,
        )

    task_specs: List[Tuple[int, int, Dict[str, Any], int]] = []
    for record_idx, record in enumerate(rows):
        for aug_idx in range(args.augmentations_per_record):
            task_seed = seed_rng.randrange(0, 2**63)
            task_specs.append((record_idx, aug_idx, record, task_seed))

    output_rows: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    semaphore = asyncio.Semaphore(max(1, args.max_concurrency))

    async def bounded_process(spec: Tuple[int, int, Dict[str, Any], int]) -> Tuple[bool, Dict[str, Any]]:
        record_idx, aug_idx, record, task_seed = spec
        async with semaphore:
            return await process_one(
                record_idx=record_idx,
                aug_idx=aug_idx,
                record=record,
                task_seed=task_seed,
                args=args,
                interventions_obj=interventions_obj,
                token_counter=token_counter,
                openrouter=openrouter,
            )

    try:
        tasks = [asyncio.create_task(bounded_process(spec)) for spec in task_specs]
        for future in asyncio.as_completed(tasks):
            ok, payload = await future
            if ok:
                output_rows.append(payload)
            else:
                rejected.append(payload)
    finally:
        if openrouter is not None:
            await openrouter.aclose()

    output_rows.sort(key=lambda row: (row.get("source_record_index", -1), row.get("augmentation_index", -1)))
    rejected.sort(key=lambda row: (row.get("source_record_index", -1), row.get("augmentation_index", -1)))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(output_rows, args.output)

    rejected_path = Path(args.output).with_suffix(Path(args.output).suffix + ".rejected.jsonl")
    dump_jsonl(rejected, rejected_path)

    summary = {
        "input_records": len(rows),
        "augmentations_requested": len(rows) * args.augmentations_per_record,
        "augmentations_written": len(output_rows),
        "augmentations_rejected": len(rejected),
        "output": str(Path(args.output)),
        "rejected_output": str(rejected_path),
        "token_counter_method": token_counter.method,
        "max_concurrency": max(1, args.max_concurrency),
        "async_api_calls": not args.mock_intervention,
    }
    print(json.dumps(summary, indent=2))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
