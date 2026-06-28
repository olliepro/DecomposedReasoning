#!/bin/bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPT="${REPO_ROOT}/vLLM_Experimental/slurm/native_benchmark_debug.sbatch"

sbatch \
  --parsable \
  --account="${ACCOUNT:-PAA0201}" \
  --partition="${PARTITION:-debug}" \
  --gres="gpu:${GPU_TYPE:-a100}:${GPU_COUNT:-1}" \
  --cpus-per-task="${CPU_COUNT:-16}" \
  --mem="${MEMORY:-128G}" \
  --time="${TIME_LIMIT:-00:30:00}" \
  --job-name="${JOB_NAME:-vllm-exp-test}" \
  --export=ALL \
  --test-only \
  "${SCRIPT}"
