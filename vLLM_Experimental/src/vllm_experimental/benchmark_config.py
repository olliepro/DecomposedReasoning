"""Generate native vLLM benchmark manifest rows."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import cast

from vllm_experimental.types import (
    BenchmarkConfig,
    BenchmarkMode,
    DEFAULT_SCRATCH_ROOT,
    DiversityVectorSource,
    TreeSearchParams,
)


def int_tuple_from_env(*, name: str, default: tuple[int, ...]) -> tuple[int, ...]:
    """Read a comma-separated integer tuple from the environment."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    assert values, f"{name} did not contain any integers"
    return values


def int_from_env(*, name: str, default: int) -> int:
    """Read one integer from the environment."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw)


def float_from_env(*, name: str, default: float) -> float:
    """Read one float from the environment."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw)


def bool_from_env(*, name: str, default: bool) -> bool:
    """Read one boolean from the environment."""

    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def diversity_source_from_env() -> DiversityVectorSource:
    """Read the candidate diversity vector source."""

    raw = os.environ.get("DIVERSITY_VECTOR_SOURCE", "lexical").strip() or "lexical"
    assert raw in {"lexical", "model_hidden_state"}, "invalid diversity source"
    return cast(DiversityVectorSource, raw)


def build_config(*, run_name: str, mode: BenchmarkMode) -> BenchmarkConfig:
    """Build the default AIME benchmark config for one mode."""

    max_model_len = int_from_env(name="MAX_MODEL_LEN", default=17_408)
    max_num_batched_tokens = int_from_env(
        name="MAX_NUM_BATCHED_TOKENS",
        default=65_536,
    )
    params = TreeSearchParams(
        mode=mode,
        fire_rate=float_from_env(name="FIRE_RATE", default=0.10),
        candidate_count=int_from_env(name="CANDIDATE_COUNT", default=50),
        branch_fanout=int_from_env(name="BRANCH_FANOUT", default=2),
        branch_depth=int_from_env(name="BRANCH_DEPTH", default=4),
        off_policy_min_candidates=int_from_env(
            name="OFF_POLICY_MIN_CANDIDATES",
            default=3,
        ),
        off_policy_max_candidates=int_from_env(
            name="OFF_POLICY_MAX_CANDIDATES",
            default=10,
        ),
        branch_max_tokens=int_from_env(name="BRANCH_MAX_TOKENS", default=700),
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_steer_tokens=int_from_env(name="MAX_STEER_TOKENS", default=30),
        max_exec_tokens=int_from_env(name="MAX_EXEC_TOKENS", default=512),
        diversity_vector_source=diversity_source_from_env(),
        native_scheduler_kv_fork=bool_from_env(
            name="NATIVE_SCHEDULER_KV_FORK",
            default=mode != "grammar_temp",
        ),
        native_branch_wave_size=int_from_env(
            name="NATIVE_BRANCH_WAVE_SIZE",
            default=50,
        ),
        native_branch_dynamic_admission=bool_from_env(
            name="NATIVE_BRANCH_DYNAMIC_ADMISSION",
            default=True,
        ),
        native_branch_min_free_blocks=int_from_env(
            name="NATIVE_BRANCH_MIN_FREE_BLOCKS",
            default=256,
        ),
        native_branch_free_block_fraction=float_from_env(
            name="NATIVE_BRANCH_FREE_BLOCK_FRACTION",
            default=0.05,
        ),
        native_branch_seq_reserve=int_from_env(
            name="NATIVE_BRANCH_SEQ_RESERVE",
            default=8,
        ),
        native_branch_priority_boost=int_from_env(
            name="NATIVE_BRANCH_PRIORITY_BOOST",
            default=1000,
        ),
        native_branch_block_safety_multiplier=float_from_env(
            name="NATIVE_BRANCH_BLOCK_SAFETY_MULTIPLIER",
            default=1.25,
        ),
        native_branch_blocked_log_interval_s=float_from_env(
            name="NATIVE_BRANCH_BLOCKED_LOG_INTERVAL_S",
            default=5.0,
        ),
        native_branch_max_live_pools=int_from_env(
            name="NATIVE_BRANCH_MAX_LIVE_POOLS",
            default=2,
        ),
        native_branch_max_queued_pools=int_from_env(
            name="NATIVE_BRANCH_MAX_QUEUED_POOLS",
            default=8,
        ),
    )
    return BenchmarkConfig(
        run_name=run_name,
        mode=mode,
        doc_ids=int_tuple_from_env(name="DOC_IDS", default=(0, 1, 2, 3, 4, 5, 6, 7)),
        prompt_concurrency=int_tuple_from_env(
            name="PROMPT_CONCURRENCY",
            default=(1, 2, 4, 8, 16),
        ),
        request_prompt_batch_size=int_tuple_from_env(
            name="REQUEST_PROMPT_BATCH_SIZE",
            default=(1, 4, 8),
        ),
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=int_from_env(name="MAX_NUM_SEQS", default=384),
        params=params,
    )


def write_rows(*, rows: list[dict[str, object]], output_dir: Path) -> None:
    """Write JSONL and compact CSV benchmark manifests."""

    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "benchmark_sweep.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    csv_path = output_dir / "benchmark_sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "mode",
                "task_name",
                "prompt_concurrency",
                "request_prompt_batch_size",
                "max_model_len",
                "max_num_batched_tokens",
                "model_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames if key in row})


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for benchmark manifest generation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", default="vllm-experimental-aime25")
    parser.add_argument(
        "--mode",
        choices=[
            "grammar_temp",
            "eps_on_policy_diverse",
            "eps_off_policy_verbalized",
        ],
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_SCRATCH_ROOT / "benchmarks",
    )
    args = parser.parse_args(argv)
    config = build_config(run_name=args.run_name, mode=args.mode)
    rows = config.sweep_rows()
    write_rows(rows=rows, output_dir=args.output_dir / args.run_name / args.mode)
    print(json.dumps({"rows": len(rows), "mode": args.mode}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
