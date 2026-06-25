# Analysis

SQLite-backed branching evaluation and visualization for vLLM
OpenAI-compatible serving.

## Setup

```bash
cd Analysis
uv sync --extra dev
```

## Run branching lm_eval framework (AIME24 default)

```bash
cd Analysis
uv run python run_branching_lm_eval.py \
  --config branching_eval/example_aime24.yaml
```

The top-level `branching_eval/*.yaml` files are examples only. Historical
checkpoint-specific OLMo/Qwen launch snapshots live under
`branching_eval/archive/checkpoint_run_configs/`; do not copy them for new
runs.

For real checkpoint evals, generate an immutable run spec:

```bash
cd Analysis
RUN_NAME=qwen3-8b-doc6-12-structured \
MODEL_PATH=/path/to/checkpoint-or-hf-repo \
TASK_NAME=aime25 \
EVAL_MODE=structured \
DOC_IDS=6,12 \
BASELINE_ROLLOUTS=48 \
MAX_GEN_TOKS=32768 \
uv run python -m branching_eval.run_specs dry-run
```

This writes `branching_eval/generated_run_specs/<run-name>/config.yaml` and
`run_spec.json`. Submit only after checking the generated config and Slurm
forecast:

```bash
uv run python -m branching_eval.run_specs test-only
uv run python -m branching_eval.run_specs submit
```

Useful knobs:

- `EVAL_MODE`: `baseline`, `structured`, `branching`, `epsilon`, or `all`.
- `SELECTOR`: `embed_diverse_topk_random`, `cluster_across`, `within_cluster`,
  or `random`.
- `DOC_IDS`: comma/space separated lm_eval doc ids passed as repeated
  `--doc-id` flags.
- `PARTITION`, `GPU_COUNT`, `GPU_TYPE`, `TIME_LIMIT`, `MEMORY`, `CPU_COUNT`:
  Slurm resource request fields.

## Run NoveltyBench generation

The NoveltyBench path uses the same vLLM/branching runtime, but writes the
official `generations.jsonl` shape consumed by the upstream NoveltyBench
partition and scoring scripts.

```bash
cd Analysis
uv run python run_branching_novelty_bench.py \
  --config branching_eval/example_novelty_bench_curated.yaml \
  --model sft \
  --seed 1234
```

The example config is a 5-prompt smoke run on the curated split. For the full
curated split, pass `--limit 100` or set `run_matrix.default_limit: null`; for
WildChat, add `--dataset-split wildchat`. NoveltyBench standard metrics use
`num_generations: 10`, so each run directory contains exactly 10 generations per
completed prompt unless true tree branching fails to produce enough leaves.

Outputs are written under `artifacts.output_root`:
- `generations.jsonl`: official-compatible NoveltyBench generation rows.
- `generation_metadata.jsonl`: raw/cleaned text length sidecar.
- `tree_events.sqlite`: branching runtime request/tree events.
- `run_manifest.json` and `config_snapshot.json`: reproducibility metadata.

Score a completed run with the upstream NoveltyBench repo:

```bash
cd /path/to/novelty-bench
uv run python src/partition.py --eval-dir /path/to/run-dir --alg classifier
uv run python src/score.py --eval-dir /path/to/run-dir --patience 0.8
uv run python src/summarize.py --eval-dir /path/to/run-dir
```

Optional overrides:

```bash
cd Analysis
uv run python run_branching_lm_eval.py \
  --config branching_eval/example_aime24.yaml \
  --limit 2 \
  --seed 1234 \
  --selector random \
  --model non_sft
```

## Serve a dynamic branching viewer

```bash
cd Analysis
uv run python scripts/serve_branching_viz.py \
  --run-dir path/to/run-dir
```

The viewer reads `tree_events.sqlite` directly. Pass repeated `--run-dir`
arguments or `--run-root path/to/artifact-root` to serve a run picker.
Per-document diagnostics are emitted as `doc_diagnostics_recorded` rows in the
same SQLite event stream. `clustering_debug.jsonl` remains an intentional raw
provider trace for selector retries and is not part of canonical tree replay.

## Validate

```bash
cd Analysis
uv run pytest
uv run pyright
```

## vLLM Compatibility References

- OpenAI-compatible server overview:
  - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
- Completions API (`n`, `stop`, `logprobs`):
  - https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/completions.html
- Chat Completions API (`logprobs`, `top_logprobs`):
  - https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat/completions.html
- Protocol fields:
  - https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/protocol.html
- Engine args (`max_logprobs`):
  - https://docs.vllm.ai/en/latest/configuration/engine_args.html
