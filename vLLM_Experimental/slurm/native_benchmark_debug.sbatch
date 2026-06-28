#!/bin/bash
#SBATCH --job-name=vllm-exp
#SBATCH --account=PAA0201
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=00:30:00
#SBATCH --output=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/vLLM_Experimental/slurm_logs/%x-%j.out
#SBATCH --error=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/vLLM_Experimental/slurm_logs/%x-%j.err

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
REPO_ROOT="${REPO_ROOT:-${SUBMIT_DIR}}"
EXPERIMENTAL_DIR="${EXPERIMENTAL_DIR:-${VLLM_EXP_DIR:-${REPO_ROOT}/vLLM_Experimental}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning}"
PYTHON_BIN="${PYTHON_BIN:-${SCRATCH_ROOT}/RLTraining/envs/qwen35-rl-cu129/bin/python}"
RUN_NAME="${RUN_NAME:-vllm-exp-aime25-debug}"
MODE="${MODE:-grammar_temp}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/vLLM_Experimental/benchmarks}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
ANALYSIS_PYTHON="${ANALYSIS_PYTHON:-/users/PAA0201/ollieproudman/work/DecomposedReasoning/Analysis/.venv/bin/python}"

if ! command -v module >/dev/null 2>&1; then
  for module_init in /etc/profile.d/lmod.sh /apps/lmod/lmod/init/bash; do
    if [[ -r "${module_init}" ]]; then
      # shellcheck disable=SC1090
      source "${module_init}"
      break
    fi
  done
fi

if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE:-python/3.12}"
  module load "${CUDA_MODULE:-cuda/13.2.1}"
fi

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export HF_HOME="${SCRATCH_ROOT}/Analysis/cache/hf_home"
export HF_HUB_CACHE="${SCRATCH_ROOT}/Analysis/cache/hf_hub"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_ROOT}/Analysis/cache/hf_hub"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/Analysis/cache/datasets"
export HF_ASSETS_CACHE="${SCRATCH_ROOT}/Analysis/cache/hf_assets"
export TORCH_HOME="${SCRATCH_ROOT}/Analysis/cache/torch"
export UV_CACHE_DIR="${SCRATCH_ROOT}/Analysis/cache/uv"
export TMPDIR="/tmp/vllm_exp_${SLURM_JOB_ID:-local}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/torchinductor"
export TRITON_CACHE_DIR="${TMPDIR}/triton"
export CUDA_CACHE_PATH="${TMPDIR}/cuda_cache"
export XDG_CACHE_HOME="${TMPDIR}/xdg_cache"
export VLLM_CACHE_ROOT="${TMPDIR}/vllm"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="${TMPDIR}/flashinfer"
unset VLLM_EXP_DIR
unset VLLM_EXPERIMENTAL_DIR
mkdir -p \
  "${OUTPUT_DIR}" \
  "${SCRATCH_ROOT}/vLLM_Experimental/slurm_logs" \
  "${TMPDIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}" \
  "${XDG_CACHE_HOME}" \
  "${VLLM_CACHE_ROOT}" \
  "${VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR}"

cd "${REPO_ROOT}"
runtime_manifest="$(
  PYTHONPATH="${EXPERIMENTAL_DIR}/src:${PYTHONPATH:-}" \
    "${PYTHON_BIN}" -m vllm_experimental.materialize_runtime \
      --runtime-parent "${SCRATCH_ROOT}/vLLM_Experimental/runtimes" \
      --patches-dir "${EXPERIMENTAL_DIR}/patches"
)"
runtime_root="$(
  RUNTIME_MANIFEST="${runtime_manifest}" "${PYTHON_BIN}" - <<'PY'
import json
import os
print(json.loads(os.environ["RUNTIME_MANIFEST"])["runtime_root"])
PY
)"

export PYTHONPATH="${runtime_root}:${EXPERIMENTAL_DIR}/src:${REPO_ROOT}/Analysis:${PYTHONPATH:-}"
export ANALYSIS_PYTHON

echo "node=$(hostname)"
echo "MODE=${MODE}"
echo "RUN_NAME=${RUN_NAME}"
echo "MAX_TOKENS=${MAX_TOKENS}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS:-384}"
echo "NATIVE_BRANCH_WAVE_SIZE=${NATIVE_BRANCH_WAVE_SIZE:-50}"
echo "NATIVE_BRANCH_DYNAMIC_ADMISSION=${NATIVE_BRANCH_DYNAMIC_ADMISSION:-true}"
echo "NATIVE_BRANCH_MIN_FREE_BLOCKS=${NATIVE_BRANCH_MIN_FREE_BLOCKS:-256}"
echo "NATIVE_BRANCH_FREE_BLOCK_FRACTION=${NATIVE_BRANCH_FREE_BLOCK_FRACTION:-0.05}"
echo "NATIVE_BRANCH_SEQ_RESERVE=${NATIVE_BRANCH_SEQ_RESERVE:-8}"
echo "NATIVE_BRANCH_PRIORITY_BOOST=${NATIVE_BRANCH_PRIORITY_BOOST:-1000}"
echo "NATIVE_BRANCH_BLOCK_SAFETY_MULTIPLIER=${NATIVE_BRANCH_BLOCK_SAFETY_MULTIPLIER:-1.25}"
echo "NATIVE_BRANCH_MAX_LIVE_POOLS=${NATIVE_BRANCH_MAX_LIVE_POOLS:-2}"
echo "NATIVE_BRANCH_MAX_QUEUED_POOLS=${NATIVE_BRANCH_MAX_QUEUED_POOLS:-8}"
echo "runtime_root=${runtime_root}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits

exec "${PYTHON_BIN}" -m vllm_experimental.run_benchmark_job \
  --run-name "${RUN_NAME}" \
  --mode "${MODE}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-tokens "${MAX_TOKENS}"
