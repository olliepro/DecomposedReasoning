# SFTTraining

Canonical checkpoint-based SFT training pipeline for:
- `Qwen/Qwen3-8B` (used in two run configs with different learning rates)
- `allenai/Olmo-3-7B-Think`
- `allenai/Olmo-3-7B-Instruct`

Training behavior:
- Full SFT on `BuildSFTDataset/output/transformed_output.jsonl`
- Supervision only on final assistant turn per row
- `<think>...</think>` preserved in targets
- `num_train_epochs=8`
- Train and save `final_model`
- Run standalone blocking `lm-eval` after training process exits (automatic in `slurm/train.sbatch`) using the separate `../Eval` environment on:
  - `minerva_math500`
  - `aime24`
  - `aime25`
- Evaluations run via Python API (`lm_eval.simple_evaluate`) rather than subprocess CLI
- Benchmark parsing logs Minerva Math500 `math_verify` only (`exact_match` ignored)
- AIME24/25 use a custom sampled `avg@k` metric (`mean@k`) with configurable `lm_eval.aime_avg_k` (default `32`)
- Flattened AIME metric keys are dynamic by configured `k`:
  - `bench/aime24/avg_at_<k>`
  - `bench/aime25/avg_at_<k>`
- Log benchmark metrics in a separate eval W&B run, grouped with the train run
- Checkpoints are configured to save under `/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs`
- SLURM launcher places HF/Torch/W&B/uv caches under `/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/cache`
- Launchers default to Accelerate + DeepSpeed ZeRO-2 (`configs/accelerate/deepspeed_zero2.yaml`)
- DeepSpeed ZeRO-3 is available via `configs/accelerate/deepspeed_zero3.yaml`
- Run configs keep `deepspeed_config_path: null` so DeepSpeed is controlled by Accelerate
- SLURM launcher emits timestamped W&B names per phase:
  - `<config_stem>_train_YYYY-MM-DD_HH-MM-SS`
  - `<config_stem>_eval_YYYY-MM-DD_HH-MM-SS`
  - shared W&B group `<config_stem>_YYYY-MM-DD_HH-MM-SS`

## Setup

```bash
cd SFTTraining
uv sync --extra dev --extra train

cd ../Eval
uv sync --extra dev
```

## Run one config locally

```bash
uv run --extra train accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  --num_processes 1 \
  -m sft_training.train \
  --config configs/runs/olmo3_7b_instruct_to_think.yaml

cd ../Eval
uv run python -m eval_runner.standalone \
  --checkpoint /fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_instruct_to_think/final_model \
  --config ../SFTTraining/configs/runs/olmo3_7b_instruct_to_think.yaml
```

To save per-example model outputs for debugging:

```bash
cd ../Eval
uv run python -m eval_runner.standalone \
  --checkpoint /fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_instruct_to_think/final_model \
  --config ../SFTTraining/configs/runs/olmo3_7b_instruct_to_think.yaml \
  --log-samples
```

For a short smoke run without changing YAML:

```bash
uv run --extra train accelerate launch \
  --config_file configs/accelerate/deepspeed_zero2.yaml \
  --num_processes 2 \
  -m sft_training.train \
  --config configs/runs/olmo3_7b_instruct_to_think.yaml \
  --max-train-samples 64 \
  --override-num-epochs 1 \
  --override-max-seq-length 1024

cd ../Eval
uv run python -m eval_runner.standalone \
  --checkpoint /fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_instruct_to_think/final_model \
  --config ../SFTTraining/configs/runs/olmo3_7b_instruct_to_think.yaml
```

## Submit one SLURM job

```bash
sbatch slurm/train.sbatch configs/runs/olmo3_7b_instruct_to_think.yaml
```

To use a different Accelerate config:

```bash
ACCELERATE_CONFIG_FILE=configs/accelerate/deepspeed_zero3.yaml \
  sbatch slurm/train.sbatch configs/runs/olmo3_7b_instruct_to_think.yaml
```

## Submit the full 4-run matrix

```bash
scripts/submit_matrix.sh
```

## Testing and type checking

```bash
cd SFTTraining
uv run pytest
uv run pyright

cd ../Eval
uv run pytest
uv run pyright
```
