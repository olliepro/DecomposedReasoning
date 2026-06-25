#!/bin/bash
#SBATCH --job-name=sft-vllm
#SBATCH --account=PAA0201
#SBATCH --partition=nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=05:00:00
#SBATCH --output=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/slurm_logs/%x-%j.out
#SBATCH --error=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/slurm_logs/%x-%j.err

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/users/PAA0201/ollieproudman/work/DecomposedReasoning}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning}"
PYTHON_BIN="${PYTHON_BIN:-${SCRATCH_ROOT}/RLTraining/envs/qwen35-rl-cu129/bin/python}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH is required}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:?SERVED_MODEL_NAME is required}"
PORT="${PORT:?PORT is required}"

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
export HF_HOME="${SCRATCH_ROOT}/Analysis/cache/hf_home"
export HF_HUB_CACHE="${SCRATCH_ROOT}/Analysis/cache/hf_hub"
export HUGGINGFACE_HUB_CACHE="${SCRATCH_ROOT}/Analysis/cache/hf_hub"
export HF_DATASETS_CACHE="${SCRATCH_ROOT}/Analysis/cache/datasets"
export HF_ASSETS_CACHE="${SCRATCH_ROOT}/Analysis/cache/hf_assets"
export TORCH_HOME="${SCRATCH_ROOT}/Analysis/cache/torch"
export UV_CACHE_DIR="${SCRATCH_ROOT}/Analysis/cache/uv"
export VLLM_ASSETS_CACHE="${SCRATCH_ROOT}/Analysis/cache/vllm_assets"
export TMPDIR="/tmp/sft_vllm_${SLURM_JOB_ID:-local}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

gpu_cache_slug="$(
  nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader 2>/dev/null \
    | head -n 1 \
    | tr '[:upper:] ' '[:lower:]_' \
    | tr -cd 'a-z0-9._-'
)"
gpu_cache_slug="${gpu_cache_slug:-unknown_gpu}"
vllm_version="$("${PYTHON_BIN}" - <<'PY'
from importlib.metadata import version

print(version("vllm"))
PY
)"
COMPILE_CACHE_ROOT="${SCRATCH_ROOT}/Analysis/cache/compiler/ascend/cuda_13.2.1/${gpu_cache_slug}/${vllm_version}/sft_vllm"
export TORCHINDUCTOR_CACHE_DIR="${COMPILE_CACHE_ROOT}/torch_inductor"
export TRITON_CACHE_DIR="${COMPILE_CACHE_ROOT}/triton"
export VLLM_CACHE_ROOT="${COMPILE_CACHE_ROOT}/vllm"
export FLASHINFER_WORKSPACE_BASE="${COMPILE_CACHE_ROOT}/flashinfer_workspace"

mkdir -p \
  "${SCRATCH_ROOT}/SFTTraining/slurm_logs" \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_ASSETS_CACHE}" \
  "${TORCH_HOME}" \
  "${UV_CACHE_DIR}" \
  "${VLLM_ASSETS_CACHE}" \
  "${TMPDIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${TRITON_CACHE_DIR}" \
  "${VLLM_CACHE_ROOT}" \
  "${FLASHINFER_WORKSPACE_BASE}"

cd "${REPO_ROOT}"
echo "node=$(hostname)"
echo "MODEL_PATH=${MODEL_PATH}"
echo "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}"
echo "PORT=${PORT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi --query-gpu=index,name,compute_cap,memory.total --format=csv,noheader

exec "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server \
  --model "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.82}" \
  --max-model-len "${MAX_MODEL_LEN:-17408}" \
  --max-num-seqs "${MAX_NUM_SEQS:-64}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-65536}" \
  --dtype bfloat16 \
  --trust-remote-code \
  --language-model-only \
  --skip-mm-profiling \
  --no-enable-log-requests
