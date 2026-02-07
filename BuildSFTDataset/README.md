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

Dry-run transform without API calls:

```bash
uv run python build_sft_dataset.py --stage transform --dry-run --max-rows 20 --yes
```

Force skipping the >100 rows confirmation gate:

```bash
uv run python build_sft_dataset.py --stage transform --max-rows 2000 --yes
```

Run the Streamlit steering viewer:

```bash
uv run streamlit run streamlit_app.py --server.port 8501
```

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
