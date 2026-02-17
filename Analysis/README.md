# Analysis

Steer-aware branching sampler and static report builder for vLLM OpenAI-compatible serving.

## Setup

```bash
cd Analysis
uv sync --extra dev
```

## Run sampler + report

```bash
cd Analysis
uv run python run_steer_branching.py \
  --model qwen3_8b_lora \
  --prompt "Find all a which satisfy a^4 + a^3 + a^2 = 100" \
  --base-url http://127.0.0.1:8000/v1
```

Outputs are written under `output/<run_id>/`:
- `config.json`
- `steps.jsonl`
- `steer_candidates.jsonl`
- `token_stats.jsonl`
- `report.html`
- `final_text.json`

## Rebuild report from artifacts

```bash
cd Analysis
uv run python build_report.py --run-dir output/<run_id>
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

Interactive report behavior:
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
