# Legacy Checkpoint Run Configs

These YAML files are old one-off launch snapshots for specific OLMo/Qwen
checkpoints and AIME document slices. They are kept here only as historical
evidence for prior runs.

Do not add new checkpoint-specific YAMLs here. For new branching-eval launches,
generate a typed run spec instead:

```bash
cd /users/PAA0201/ollieproudman/work/DecomposedReasoning/Analysis
RUN_NAME=my-eval \
MODEL_PATH=/path/to/checkpoint \
EVAL_MODE=structured \
DOC_IDS=6,12 \
uv run python -m branching_eval.run_specs dry-run
```

The generator writes:

- `branching_eval/generated_run_specs/<run-name>/config.yaml`
- `branching_eval/generated_run_specs/<run-name>/run_spec.json`

The generated YAML is the immutable config passed to
`run_branching_lm_eval.py`; `run_spec.json` records the launch intent and
Slurm resource request.
