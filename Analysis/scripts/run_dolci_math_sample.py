"""Run one-rollout epsilon-greedy generation on sampled Dolci math prompts.

This script samples prompt/answer rows from the Hugging Face
`allenai/Dolci-RL-Zero-Math-7B` dataset, executes one epsilon-greedy rollout per
sampled prompt using the existing branching runtime, and writes standard event
artifacts plus compact JSON summaries.

Example:
    cd Analysis
    .venv/bin/python scripts/run_dolci_math_sample.py \
      --base-url http://127.0.0.1:8020/v1 \
      --sample-size 32 \
      --sample-seed 1234 \
      --epsilon-greedy-prob 0.5 \
      --max-gen-toks 32768 \
      --decode-chunk-tokens 512
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, cast
from urllib import request as urllib_request

from datasets import Dataset, load_dataset

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_ROOT_TEXT = str(ANALYSIS_ROOT)
if ANALYSIS_ROOT_TEXT not in sys.path:
    sys.path.insert(0, ANALYSIS_ROOT_TEXT)

from branching_eval.artifact_store import ArtifactStore, build_run_manifest_payload
from branching_eval.config_types import (
    ArtifactConfig,
    BranchingConfig,
    BranchingEvalConfig,
    DecodingConfig,
    ExperimentSpec,
    ModelSpec,
    RunMatrixConfig,
    ServeConfig,
    TaskConfig,
)
from branching_eval.lm_eval_adapter import DocRecord
from branching_eval.run_matrix import (
    DocExecutionPlan,
    build_config_snapshot,
    build_run_dir,
    doc_event_context,
    read_git_commit_hash,
    recompute_outputs_from_events,
    run_docs_with_limit_async,
    run_event_context,
    selector_mode_for_spec,
)
from io_utils import append_jsonl, write_json
from vllm_client import VllmClient


@dataclass(frozen=True)
class SampledDolciDoc:
    """One sampled Dolci dataset row.

    Args:
        row_id: Original dataset row index.
        prompt: User prompt text.
        ground_truth: Canonical reference answer string.
        dataset_name: Source subset label from the dataset payload.

    Returns:
        Dataclass containing one sampled prompt/answer pair.
    """

    row_id: int
    prompt: str
    ground_truth: str
    dataset_name: str

    def to_doc_record(self) -> DocRecord:
        """Convert this sample into the runtime doc payload shape.

        Args:
            None.

        Returns:
            `DocRecord` keyed by the original dataset row id.
        """

        return DocRecord(
            doc_id=self.row_id,
            doc_payload={
                "prompt": self.prompt,
                "ground_truth": self.ground_truth,
                "dataset": self.dataset_name,
                "row_id": self.row_id,
            },
            prompt_text=self.prompt,
        )


class DolciMathAdapter:
    """Duck-typed scoring adapter compatible with `run_matrix` helpers.

    Args:
        None.

    Returns:
        Adapter exposing the subset of the lm-eval adapter interface that the
        runtime scoring path needs.
    """

    def score_response(
        self, *, doc: dict[str, Any], response_text: str
    ) -> dict[str, Any]:
        """Score one response with exact-match style answer verification.

        Args:
            doc: Sampled Dolci payload containing `ground_truth`.
            response_text: Model response text.

        Returns:
            Metric mapping containing `exact_match`.
        """

        exact_match = self.verification(doc=doc, response_text=response_text)
        return {"exact_match": exact_match}

    def verification(self, *, doc: dict[str, Any], response_text: str) -> int:
        """Return binary correctness using the shared boxed-answer parser.

        Args:
            doc: Sampled Dolci payload containing `ground_truth`.
            response_text: Model response text.

        Returns:
            `1` when the extracted answer matches `ground_truth`, else `0`.
        """

        target = _normalize_answer(value=str(doc["ground_truth"]))
        candidate = self.extract_answer(response_text=response_text)
        return int(candidate == target)

    def aggregate_doc_metrics(
        self, *, rollout_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reduce rollout metrics for one document.

        Args:
            rollout_metrics: Per-rollout metrics for a single document.

        Returns:
            Mean-reduced metric mapping.
        """

        return _mean_numeric_metrics(metric_rows=rollout_metrics)

    def aggregate_task_metrics(
        self, *, per_doc_metrics: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Reduce metrics across all sampled documents.

        Args:
            per_doc_metrics: Reduced metrics per document.

        Returns:
            Mean-reduced task summary mapping.
        """

        return _mean_numeric_metrics(metric_rows=per_doc_metrics)

    def extract_answer(self, *, response_text: str) -> str:
        """Extract and normalize one candidate answer from model text.

        Args:
            response_text: Raw model response.

        Returns:
            Canonical normalized answer string.
        """

        module = _load_eval_answer_module()
        extracted = module.extract_candidate_answer(response=response_text)
        return _normalize_answer(value=str(extracted))


@dataclass(frozen=True)
class DolciRunSummary:
    """Compact summary written after one sampled Dolci run finishes.

    Args:
        run_dir: Artifact directory for this run.
        sample_size: Number of sampled documents executed.
        sample_seed: RNG seed used for dataset sampling.
        rollout_seed: RNG seed used for runtime branching.
        correct_count: Number of exact-match correct documents.
        accuracy: Exact-match accuracy over sampled documents.
        completed_doc_count: Completed document count from the event log.

    Returns:
        Dataclass containing top-level run summary fields.
    """

    run_dir: str
    sample_size: int
    sample_seed: int
    rollout_seed: int
    correct_count: int
    accuracy: float
    completed_doc_count: int


def parse_args() -> argparse.Namespace:
    """Parse CLI args for the Dolci sampling runner.

    Args:
        None.

    Returns:
        Parsed namespace with sampling and runtime overrides.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument("--sample-size", type=int, default=32)
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--rollout-seed", type=int, default=1234)
    parser.add_argument("--epsilon-greedy-prob", type=float, default=0.5)
    parser.add_argument("--max-gen-toks", type=int, default=32768)
    parser.add_argument("--decode-chunk-tokens", type=int, default=512)
    parser.add_argument("--max-concurrent-docs", type=int, default=2)
    parser.add_argument("--num-candidates", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Run the sampled Dolci epsilon-greedy generation workflow.

    Args:
        None.

    Returns:
        None.
    """

    args = parse_args()
    sampled_docs = sample_dolci_docs(
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    model_name = resolve_served_model_name(base_url=args.base_url)
    config = build_runtime_config(
        base_url=args.base_url,
        model_name=model_name,
        output_root=resolve_output_root(output_root=args.output_root),
        epsilon_greedy_prob=args.epsilon_greedy_prob,
        max_gen_toks=args.max_gen_toks,
        decode_chunk_tokens=args.decode_chunk_tokens,
        max_concurrent_docs=args.max_concurrent_docs,
        num_candidates=args.num_candidates,
        rollout_seed=args.rollout_seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    spec = build_experiment_spec(rollout_seed=args.rollout_seed)
    run_dir = run_sampled_dolci_generation(
        config=config,
        spec=spec,
        sampled_docs=sampled_docs,
        sample_seed=args.sample_seed,
    )
    print(str(run_dir))


def sample_dolci_docs(*, sample_size: int, sample_seed: int) -> list[SampledDolciDoc]:
    """Load the Dolci dataset and sample stable random rows.

    Args:
        sample_size: Number of rows to sample without replacement.
        sample_seed: RNG seed used for reproducible sampling.

    Returns:
        Sampled docs keyed by their original dataset row ids.
    """

    dataset = cast(
        Dataset,
        load_dataset("allenai/Dolci-RL-Zero-Math-7B", split="train"),
    )
    assert sample_size >= 1, "sample_size must be >= 1"
    assert sample_size <= len(dataset), "sample_size exceeds dataset length"
    row_ids = sorted(Random(sample_seed).sample(range(len(dataset)), k=sample_size))
    sampled_docs: list[SampledDolciDoc] = []
    for row_id in row_ids:
        row = dataset[int(row_id)]
        sampled_docs.append(
            SampledDolciDoc(
                row_id=int(row_id),
                prompt=str(row["prompt"]),
                ground_truth=str(row["ground_truth"]),
                dataset_name=str(row["dataset"]),
            )
        )
    return sampled_docs


def resolve_served_model_name(*, base_url: str) -> str:
    """Resolve the request-time served model name from `/v1/models`.

    Args:
        base_url: OpenAI-compatible base URL ending in `/v1`.

    Returns:
        Served model identifier accepted by the running vLLM server.
    """

    response = urllib_request.urlopen(f"{base_url.rstrip('/')}/models")
    payload = json.load(response)
    data = payload.get("data", [])
    assert isinstance(data, list) and data, "No models returned by vLLM server"
    first_entry = data[0]
    assert isinstance(first_entry, dict), "Unexpected /models payload"
    model_name = str(first_entry["id"])
    assert model_name, "Served model id must be non-empty"
    return model_name


def resolve_output_root(*, output_root: Path | None) -> Path:
    """Resolve the output root used for sampled Dolci artifacts.

    Args:
        output_root: Optional explicit output root override.

    Returns:
        Absolute output root directory.
    """

    if output_root is not None:
        return output_root.resolve()
    return Path(
        "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/"
        "profile_runs/20260420_dolci_math_eps50_sample32"
    ).resolve()


def build_runtime_config(
    *,
    base_url: str,
    model_name: str,
    output_root: Path,
    epsilon_greedy_prob: float,
    max_gen_toks: int,
    decode_chunk_tokens: int,
    max_concurrent_docs: int,
    num_candidates: int,
    rollout_seed: int,
    temperature: float,
    top_p: float,
) -> BranchingEvalConfig:
    """Build a typed runtime config for sampled Dolci execution.

    Args:
        base_url: External vLLM base URL ending in `/v1`.
        model_name: Served model id accepted by the live server.
        output_root: Artifact root directory.
        epsilon_greedy_prob: Epsilon-greedy intervention probability.
        max_gen_toks: Max completion token budget.
        decode_chunk_tokens: Decode chunk size.
        max_concurrent_docs: Maximum in-flight sampled docs.
        num_candidates: Candidate count per epsilon trigger.
        rollout_seed: Experiment-level seed used for executor seeding.
        temperature: Sampling temperature.
        top_p: Nucleus cutoff.

    Returns:
        Fully typed branching runtime config.
    """

    config = BranchingEvalConfig(
        tasks=TaskConfig(task_names=("dolci_rl_zero_math_7b_sample32",)),
        models=(
            ModelSpec(
                model_id="sft",
                base_url=base_url,
                served_model_name=model_name,
                trigger_steer_default=True,
                trigger_entropy_default=False,
            ),
        ),
        serve=ServeConfig(
            host="127.0.0.1",
            base_port=8020,
            tensor_parallel_size=2,
            gpu_memory_utilization=0.85,
            dtype="auto",
            scheduling_policy="priority",
            kv_offloading_size_gb=0.0,
            trust_remote_code=True,
            max_logprobs=20,
            startup_timeout_seconds=30.0,
            poll_interval_seconds=1.0,
        ),
        decoding=DecodingConfig(
            temperature=temperature,
            top_p=top_p,
            max_gen_toks=max_gen_toks,
            top_logprobs=20,
            decode_chunk_tokens=decode_chunk_tokens,
            debug_assert_text_token_alignment=False,
        ),
        branching=BranchingConfig(
            branch_prob=0.0,
            max_branch_points_per_rollout=4,
            max_concurrent_branches=20,
            num_candidates=num_candidates,
            branch_fanout=2,
            max_clusters=4,
            candidate_span_tokens=15,
            max_steer_tokens=15,
            steer_repetition_penalty=1.01,
            epsilon_greedy_prob=epsilon_greedy_prob,
        ),
        artifacts=ArtifactConfig(output_root=output_root),
        run_matrix=RunMatrixConfig(
            include_baselines=False,
            include_structured_baselines=False,
            baseline_rollouts=1,
            include_branching=False,
            include_epsilon_greedy=True,
            selectors=("embed_diverse_topk_random",),
            seed_values=(rollout_seed,),
            default_limit=None,
            max_concurrent_docs=max_concurrent_docs,
        ),
    )
    config.validate()
    return config


def build_experiment_spec(*, rollout_seed: int) -> ExperimentSpec:
    """Build the one concrete experiment spec used for sampled Dolci docs.

    Args:
        rollout_seed: Experiment-level seed used for executor seeding.

    Returns:
        Epsilon-greedy experiment spec with one rollout per document.
    """

    return ExperimentSpec(
        task_name="dolci_rl_zero_math_7b_sample32",
        model_id="sft",
        mode="epsilon_greedy",
        selector="embed_diverse_topk_random",
        seed=rollout_seed,
        baseline_rollouts=1,
        trigger_steer=True,
    )


def run_sampled_dolci_generation(
    *,
    config: BranchingEvalConfig,
    spec: ExperimentSpec,
    sampled_docs: list[SampledDolciDoc],
    sample_seed: int,
) -> Path:
    """Execute sampled Dolci docs and write run artifacts.

    Args:
        config: Typed runtime config.
        spec: Concrete experiment spec.
        sampled_docs: Sampled prompt/answer rows.
        sample_seed: RNG seed used for dataset sampling.

    Returns:
        Completed run directory path.
    """

    run_dir = build_run_dir(config=config, spec=spec)
    store = ArtifactStore(run_dir=run_dir, run_id=run_dir.name)
    try:
        adapter: Any = DolciMathAdapter()
        doc_records = [sampled_doc.to_doc_record() for sampled_doc in sampled_docs]
        doc_map = {sampled_doc.row_id: sampled_doc for sampled_doc in sampled_docs}
        model_spec = config.models[0]
        assert model_spec.base_url is not None, "external base_url required"
        client = VllmClient(base_url=model_spec.base_url, timeout_seconds=None)
        config_snapshot = build_config_snapshot(
            config=config,
            spec=spec,
            model_spec=model_spec,
        )
        store.write_config_snapshot(payload=config_snapshot)
        store.write_run_manifest(
            payload=build_run_manifest_payload(
                run_id=store.run_id,
                task_name=spec.task_name,
                model_id=spec.model_id,
                selector_mode=selector_mode_for_spec(spec=spec),
                config_snapshot=config_snapshot,
                git_commit=read_git_commit_hash(),
            )
        )
        write_sample_manifest(
            run_dir=run_dir,
            sampled_docs=sampled_docs,
            sample_seed=sample_seed,
        )
        run_context = run_event_context(spec=spec, run_id=store.run_id)
        store.append_event(
            context=run_context,
            event_type="run_started",
            payload={
                "resume": False,
                "doc_count_total": len(doc_records),
                "doc_count_scheduled": len(doc_records),
                "doc_count_skipped": 0,
                "mode": spec.mode,
                "dataset_name": "allenai/Dolci-RL-Zero-Math-7B",
                "sample_seed": sample_seed,
                "doc_ids": [doc.doc_id for doc in doc_records],
            },
        )
        plans = build_execution_plans(doc_records=doc_records)
        results = asyncio.run(
            run_docs_with_limit_async(
                plans=plans,
                max_concurrent_docs=config.run_matrix.max_concurrent_docs,
                config=config,
                spec=spec,
                client=client,
                cluster_client=None,
                model_name_for_generation=model_spec.served_model_name or "",
                cluster_model_name_for_generation=None,
                task_name=spec.task_name,
                model_id=spec.model_id,
                resolved_branching=config.branching,
                store=store,
                adapter=cast(Any, adapter),
                events_snapshot=[],
            )
        )
        summary_payload = recompute_outputs_from_events(
            store=store,
            adapter=cast(Any, adapter),
            selector_mode=selector_mode_for_spec(spec=spec),
        )
        result_rows = build_result_rows(
            doc_results=results,
            doc_map=doc_map,
            adapter=adapter,
        )
        write_result_files(run_dir=run_dir, result_rows=result_rows)
        store.append_event(
            context=run_context,
            event_type="run_finished",
            payload={
                "resume": False,
                "doc_count_total": len(doc_records),
                "doc_count_scheduled": len(doc_records),
                "doc_count_skipped": 0,
                **summary_payload,
            },
        )
        summary = build_summary(
            run_dir=run_dir,
            result_rows=result_rows,
            sample_seed=sample_seed,
            rollout_seed=spec.seed,
            completed_doc_count=int(summary_payload["doc_count_completed"]),
        )
        write_json(path=run_dir / "dolci_summary.json", payload=asdict(summary))
        return run_dir
    finally:
        store.close()


def build_execution_plans(*, doc_records: list[DocRecord]) -> list[DocExecutionPlan]:
    """Build one execution plan per sampled document.

    Args:
        doc_records: Sampled prompt records.

    Returns:
        Ordered zero-attempt execution plans.
    """

    return [
        DocExecutionPlan(
            doc_record=doc_record,
            doc_attempt=0,
            resumed_reason=None,
            resume_in_place=False,
            has_started_event=False,
        )
        for doc_record in doc_records
    ]


def write_sample_manifest(
    *,
    run_dir: Path,
    sampled_docs: list[SampledDolciDoc],
    sample_seed: int,
) -> None:
    """Write sampled dataset rows and sampling metadata.

    Args:
        run_dir: Output directory for this run.
        sampled_docs: Sampled docs written to disk.
        sample_seed: RNG seed used for sampling.

    Returns:
        None.
    """

    write_json(
        path=run_dir / "sample_manifest.json",
        payload={
            "dataset_name": "allenai/Dolci-RL-Zero-Math-7B",
            "sample_seed": sample_seed,
            "sample_size": len(sampled_docs),
            "row_ids": [doc.row_id for doc in sampled_docs],
        },
    )
    for sampled_doc in sampled_docs:
        append_jsonl(
            path=run_dir / "sampled_docs.jsonl",
            payload=asdict(sampled_doc),
        )


def build_result_rows(
    *,
    doc_results: list[Any],
    doc_map: dict[int, SampledDolciDoc],
    adapter: DolciMathAdapter,
) -> list[dict[str, Any]]:
    """Build compact result rows from completed sampled-doc executions.

    Args:
        doc_results: Completed doc results from the runtime.
        doc_map: Sampled doc mapping keyed by dataset row id.
        adapter: Scoring adapter used for answer extraction.

    Returns:
        One JSON-ready summary row per sampled document.
    """

    result_rows: list[dict[str, Any]] = []
    for doc_result in sorted(doc_results, key=lambda item: item.doc_id):
        sampled_doc = doc_map[doc_result.doc_id]
        leaf = doc_result.scored_leaves[0]
        result_rows.append(
            {
                "row_id": sampled_doc.row_id,
                "dataset": sampled_doc.dataset_name,
                "ground_truth": sampled_doc.ground_truth,
                "verification": int(leaf.verification),
                "stop_reason": leaf.stop_reason,
                "length_tokens_total": int(leaf.length_tokens_total),
                "task_metrics": dict(leaf.task_metrics),
                "extracted_answer": adapter.extract_answer(response_text=leaf.text),
                "response_text": leaf.text,
                "prompt": sampled_doc.prompt,
            }
        )
    return result_rows


def write_result_files(*, run_dir: Path, result_rows: list[dict[str, Any]]) -> None:
    """Persist sampled-doc result rows in JSON and JSONL formats.

    Args:
        run_dir: Output directory for this run.
        result_rows: Result rows to persist.

    Returns:
        None.
    """

    (run_dir / "dolci_results.json").write_text(
        json.dumps(result_rows, indent=2),
        encoding="utf-8",
    )
    for result_row in result_rows:
        append_jsonl(path=run_dir / "dolci_results.jsonl", payload=result_row)


def build_summary(
    *,
    run_dir: Path,
    result_rows: list[dict[str, Any]],
    sample_seed: int,
    rollout_seed: int,
    completed_doc_count: int,
) -> DolciRunSummary:
    """Build the top-level sampled Dolci run summary.

    Args:
        run_dir: Output directory for this run.
        result_rows: Compact per-document result rows.
        sample_seed: RNG seed used for dataset sampling.
        rollout_seed: Runtime seed used by the executor.
        completed_doc_count: Completed doc count from event aggregation.

    Returns:
        Top-level run summary dataclass.
    """

    correct_count = sum(int(row["verification"]) for row in result_rows)
    sample_size = len(result_rows)
    accuracy = float(correct_count / sample_size) if sample_size else 0.0
    return DolciRunSummary(
        run_dir=str(run_dir),
        sample_size=sample_size,
        sample_seed=sample_seed,
        rollout_seed=rollout_seed,
        correct_count=correct_count,
        accuracy=accuracy,
        completed_doc_count=completed_doc_count,
    )


def _mean_numeric_metrics(*, metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean-reduce numeric metrics from per-rollout or per-doc rows.

    Args:
        metric_rows: Metric rows to reduce.

    Returns:
        Mean-reduced metric mapping.
    """

    if not metric_rows:
        return {}
    keys = sorted({key for row in metric_rows for key in row})
    reduced: dict[str, Any] = {}
    for key in keys:
        numeric_values = [
            float(row[key]) for row in metric_rows if isinstance(row.get(key), (int, float))
        ]
        if numeric_values:
            reduced[key] = sum(numeric_values) / len(numeric_values)
            continue
        first_present = next((row[key] for row in metric_rows if key in row), None)
        if first_present is not None:
            reduced[key] = first_present
    return reduced


def _normalize_answer(*, value: str) -> str:
    """Normalize one candidate or target answer for exact-match scoring.

    Args:
        value: Raw answer string.

    Returns:
        Normalized answer string used for equality comparison.
    """

    module = _load_eval_answer_module()
    return str(module.normalize_answer_text(value=str(value)))


def _load_eval_answer_module() -> Any:
    """Load the shared math answer utility module from `Eval/eval_runner`.

    Args:
        None.

    Returns:
        Imported module object exposing answer extraction helpers.
    """

    repo_root = Path(__file__).resolve().parent.parent.parent
    eval_dir = repo_root / "Eval"
    eval_dir_text = str(eval_dir)
    if eval_dir_text not in sys.path:
        sys.path.insert(0, eval_dir_text)
    return importlib.import_module("eval_runner.aime_avgk")


if __name__ == "__main__":
    main()
