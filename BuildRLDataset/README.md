# BuildRLDataset

Staged CLI for building a math-only RL dataset from `allenai/Dolci-Think-RL-7B`.

Pipeline stages:
- `sample`: stream a subset into `output/raw_sample.jsonl`
- `filter`: keep rows where `dataset` contains `math` and required fields are present
- `stratify`: draw a balanced 10k train subset by normalized source family
- `export`: write `train.parquet`, `manifest.json`, and `source_audit.json`

## Setup

```bash
cd /users/PAA0201/ollieproudman/work/DecomposedReasoning/BuildRLDataset
uv sync
```

## Common commands

Run the next incomplete stage:

```bash
uv run python build_rl_dataset.py --yes
```

Run the full pipeline end-to-end:

```bash
uv run python build_rl_dataset.py --stage all --yes
```

Change the sampling budget while keeping the default 10k train export:

```bash
uv run python build_rl_dataset.py --stage all --sample-rows 75000 --yes
```

Dry-run the current stage selection:

```bash
uv run python build_rl_dataset.py --dry-run
```

## Outputs

- Raw sample: `output/raw_sample.jsonl`
- Filtered candidates: `output/filtered_candidates.jsonl`
- Stratified sample: `output/stratified_sample.jsonl`
- Final train parquet: `output/train.parquet`
- Source audit: `output/source_audit.json`
- Manifest: `output/manifest.json`
- Pipeline state: `output/pipeline_state.json`

## Notes

- The exported `prompt` column is converted into chat-message format for `RLHFDataset`.
- The original prompt string is preserved in `source_prompt_text`.
- Original token columns and source metadata are retained in the final parquet rows.
