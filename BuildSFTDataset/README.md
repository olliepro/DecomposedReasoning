# BuildSFTDataset

Staged CLI for building and transforming a sampled subset of `allenai/Dolci-Think-SFT-7B`.

Pipeline stages:
- `sample`: sample random shards and rows into `output/raw_sample.jsonl`
- `filter`: keep rows whose assistant `<think>` token count is within range
- `stratify`: balanced sampling by `dataset_source` into final subset
- `transform`: rewrite `<think>` blocks using Gemini/Vertex prompts

## Setup

```bash
cd "/Users/olliepro/Code/School/DecomposedReasoning/BuildSFTDataset"
uv sync
```

## Default behavior

When you run `build_sft_dataset.py` with no stage:
- It reads `output/pipeline_state.json`
- It resumes from the next incomplete stage
- It defaults to batch transform mode

Current key defaults:
- `--mode gemini`
- `--model gemini-3-flash-preview`
- `--thinking-level low`
- `--max-output-tokens 20000`
- `--batch`

## Common commands

Run next stage automatically:

```bash
uv run python build_sft_dataset.py --yes
```

Run only transform stage:

```bash
uv run python build_sft_dataset.py --stage transform --yes
```

Recover and process a previously succeeded batch job:

```bash
uv run python build_sft_dataset.py --stage transform --batch-job batches/<job_id> --yes
```

Dry-run transform without API calls:

```bash
uv run python build_sft_dataset.py --stage transform --dry-run --max-rows 20 --yes
```

Run local dry tests:

```bash
uv run python tests/test_cli_dry_run.py
uv run python tests/test_batch_recovery_dry_run.py
```

Force skipping the >100 rows confirmation gate:

```bash
uv run python build_sft_dataset.py --stage transform --max-rows 2000 --yes
```

Run the Streamlit steering viewer:

```bash
uv run streamlit run streamlit_app.py --server.port 8501
```

Streamlit row sorting now supports:
- Dataset order
- Lowest think/steer ratio
- Highest think/steer ratio
- Most steer instances

Ratio definition in the UI:
- `think_to_steer_instances_ratio = think_tokens / number_of_<steer_or_steering>_instances`
- Uses exact token counts from `output/cluster_analysis/think_token_stats.parquet` when available.
- Falls back to a whitespace token estimate when token stats are unavailable.

## Steering Cluster Analysis

Run the full two-pass steering clustering and naming pipeline:

```bash
uv run python analyze_steering_clusters.py --stage all
```

Run one stage only:

```bash
uv run python analyze_steering_clusters.py --stage cluster1
```

If you hit API quota errors, throttle request rate and resume:

```bash
uv run python analyze_steering_clusters.py \
  --stage all \
  --token-requests-per-minute 1800 \
  --embed-requests-per-minute 900 \
  --naming-requests-per-minute 600
```

Embedding optimization defaults:
- `--embed-batch-size 100` (hard-capped to 100 per request)
- Batched embedding responses are mapped back by input index order.

Default analysis inputs and outputs:
- Transformed input: `/Users/olliepro/Code/School/DecomposedReasoning/BuildSFTDataset/output/transformed_output.jsonl`
- OG reference: `/Users/olliepro/Code/School/DecomposedReasoning/BuildSFTDataset/output/stratified_sample.jsonl`
- Output root: `/Users/olliepro/Code/School/DecomposedReasoning/BuildSFTDataset/output/cluster_analysis`

Run dry integration tests for the analysis script:

```bash
uv run python tests/test_analyze_steering_clusters_dry.py
```

Run strict `<think>` structure validation on the cleaned transformed dataset:

```bash
uv run python tests/test_clean_think_structure.py \
  --dataset-path output/transformed_output.jsonl \
  --max-issue-examples 20
```

Validation rules:
- The **last assistant message** must contain exactly one `<think>...</think>` block.
- The block must parse into matched steering/execution pairs.
- No non-whitespace residual text is allowed outside steer/exec sections.

## Paths

- Env vars: `.env`
- System prompt: `system_prompt.md`
- User prompt template: `user_prompt.md` (must include `{think_text}`)
- Outputs: `output/`
- State file: `output/pipeline_state.json`

## Notes

- Transform resume is id-based. Existing transformed row IDs are skipped.
- `--max-rows` is intentionally not a default override; set it per run.
- Use `--no-batch` to switch to synchronous per-request calls.
