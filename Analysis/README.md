# Analysis

Steer-aware branching sampler and modular report viewer builder for vLLM OpenAI-compatible serving.

## Setup

```bash
cd Analysis
uv sync --extra dev
```

## Run sampler + report

```bash
cd Analysis
uv run python run_steer_branching.py \
  --model qwen3_8b_to_think \
  --prompt "Find all a which satisfy a^4 + a^3 + a^2 = 100" \
  --base-url http://127.0.0.1:8000/v1
```

## Run branching lm_eval framework (AIME24 default)

```bash
cd Analysis
uv run python run_branching_lm_eval.py \
  --config branching_eval/example_aime24.yaml
```

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
- `tree_events.jsonl`: branching runtime request/tree events.
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

## Run a baseline-vs-branching TOK/sec comparison

Use the dedicated comparison config to launch a matched baseline `n` rollout and
one branching run:

```bash
cd Analysis
uv run python run_branching_lm_eval.py \
  --config branching_eval/example_tok_sec.yaml \
  --limit 1 \
  --model non_sft \
  --seed 1234
```

The baseline `n` value is `run_matrix.baseline_rollouts` in
`branching_eval/example_tok_sec.yaml`.

After runs finish, compare token throughput from `tree_events.jsonl`:

```bash
cd Analysis
uv run python compare_tok_sec.py \
  --output-root output/branching_eval \
  --model non_sft \
  --seed 1234 \
  --show-request-kinds
```

Notes:
- `req_tok/s` is computed from total output tokens divided by summed
  `vllm_response.latency_seconds`.
- `wall_tok/s` is computed from total output tokens divided by elapsed wall time
  between the first `vllm_request` and the last `vllm_response`.
- `--all-runs` disables latest-run deduplication when you want every historical
  run under `output/branching_eval`.

One-time entropy calibration helper:

```bash
cd /users/PAA0201/ollieproudman/work/DecomposedReasoning
python Analysis/scripts/calibrate_entropy_threshold.py
```

Outputs are written under `output/<run_id>/`:
- `config.json`
- `steps.jsonl`
- `steer_candidates.jsonl`
- `token_stats.jsonl`
- `report.html`
- `final_text.json`
- `report_assets/`

## Rebuild report from artifacts

```bash
cd Analysis
uv run python build_report.py --run-dir output/<run_id>
```

Bundle multiple outputs in one viewer:

```bash
cd Analysis
uv run python build_report.py \
  --run-dir output/<run_id_a> \
  --run-dir output/<run_id_b> \
  --output output/report_bundle.html
```

To tune clustering behavior:

```bash
cd Analysis
uv run python build_report.py \
  --run-dir output/<run_id> \
  --gemini-model gemini-3-flash-preview \
  --gemini-temperature 0.2 \
  --previous-steps-window 5 \
  --cluster-max-concurrency 50 \
  --cluster-cache output/<run_id>/cluster_prompt_cache.json \
  --env-file .env \
  --env-file BuildSFTDataset/.env
```

Clustering uses Gemini structured-output prompting on deduplicated steer strings.
When previous selected steps exist, up to 5 are included as context with `>>` separators.
Prompt responses are cached by default at `<run_dir>/cluster_prompt_cache.json`.

Viewer behavior:
- Home page explains the generation algorithm and lists available outputs.
- Sidebar includes Home plus per-output selection named by input prompt.
- Steps are collapsible and labeled with selected steer text.
- Clusters are single-click selectable.
- Chosen candidate is always visible.
- Unchosen variants appear only after selecting a cluster.
- Variants are deduplicated with occurrence counts.
- Execution text is in a dropdown and rendered as markdown.

By default, `build_report.py` now checks dotenv files in this order:
- repo root `.env` (for `VERTEX_KEY` / `GEMINI_API_KEY`)
- `BuildSFTDataset/.env`
- current working directory `.env`
- `<run_dir>/.env`

To disable semantic clustering and force fallback grouping:

```bash
cd Analysis
uv run python build_report.py --run-dir output/<run_id> --disable-clustering
```

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
