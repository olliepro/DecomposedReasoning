# DecomposedReasoning

Tools for building and transforming a sampled SFT dataset from `allenai/Dolci-Think-SFT-7B`.

## Project layout

- `BuildSFTDataset/`: staged pipeline CLI and Streamlit viewer.

## Quick start

```bash
cd BuildSFTDataset
uv sync
uv run python build_sft_dataset.py --yes
```

## Notes

- Generated outputs are ignored by git (`BuildSFTDataset/output/`, `BuildSFTDataset/output_nonbatch5/`).
- Secrets are ignored by git (`.env`, `*.env`).
