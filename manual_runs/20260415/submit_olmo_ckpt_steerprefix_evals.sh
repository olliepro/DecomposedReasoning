#!/bin/bash
set -euo pipefail

REPO_ROOT=/users/PAA0201/ollieproudman/work/DecomposedReasoning
SBATCH_SCRIPT="$REPO_ROOT/manual_runs/20260415/run_olmo_ckpt_steerprefix_eval.sbatch"
LOG_DIR="$REPO_ROOT/manual_runs/20260415/logs"
mkdir -p "$LOG_DIR" "$REPO_ROOT/Eval/output/manual_20260415"

echo "Test-only AIME24"
sbatch --test-only \
  --time=06:00:00 \
  --job-name=a24-olmo-ckpt32-32k-steer \
  --export=ALL,TASK_NAME=aime24,SOURCE_SAMPLES="$REPO_ROOT/Eval/output/manual_20260413/samples_aime24_2026-04-14T00-32-35.203153.jsonl",PORT=8040 \
  "$SBATCH_SCRIPT"

echo "Test-only AIME25"
sbatch --test-only \
  --time=07:00:00 \
  --job-name=a25-olmo-ckpt32-32k-steer \
  --export=ALL,TASK_NAME=aime25,SOURCE_SAMPLES="$REPO_ROOT/Eval/output/manual_20260413/samples_aime25_2026-04-14T10-34-48.125475.jsonl",PORT=8041 \
  "$SBATCH_SCRIPT"

echo "Submit AIME24"
sbatch --parsable \
  --time=06:00:00 \
  --job-name=a24-olmo-ckpt32-32k-steer \
  --export=ALL,TASK_NAME=aime24,SOURCE_SAMPLES="$REPO_ROOT/Eval/output/manual_20260413/samples_aime24_2026-04-14T00-32-35.203153.jsonl",PORT=8040 \
  "$SBATCH_SCRIPT"

echo "Submit AIME25"
sbatch --parsable \
  --time=07:00:00 \
  --job-name=a25-olmo-ckpt32-32k-steer \
  --export=ALL,TASK_NAME=aime25,SOURCE_SAMPLES="$REPO_ROOT/Eval/output/manual_20260413/samples_aime25_2026-04-14T10-34-48.125475.jsonl",PORT=8041 \
  "$SBATCH_SCRIPT"
