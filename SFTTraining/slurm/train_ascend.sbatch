#!/bin/bash
#SBATCH --job-name=sft-q35-4b
#SBATCH --account=PAA0201
#SBATCH --partition=quad
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/slurm_logs/%x-%j.out
#SBATCH --error=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/slurm_logs/%x-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/train_ascend.sbatch <run-config-yaml> [extra-train-args...]"
  exit 1
fi

RUN_CONFIG="$(realpath -m "$1")"
shift
RUN_CONFIG_NAME="$(basename "${RUN_CONFIG}")"
RUN_CONFIG_STEM="${RUN_CONFIG_NAME%.*}"
PROJECT_DIR="${SFT_PROJECT_DIR:-}"
if [[ -z "${PROJECT_DIR}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}/sft_training" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  elif [[ "${RUN_CONFIG}" == */SFTTraining/configs/runs/* ]]; then
    PROJECT_DIR="${RUN_CONFIG%%/configs/runs/*}"
  else
    PROJECT_DIR="/users/PAA0201/ollieproudman/work/DecomposedReasoning/SFTTraining"
  fi
fi
PROJECT_DIR="$(realpath -m "${PROJECT_DIR}")"
REPO_ROOT="${REPO_ROOT:-$(cd "${PROJECT_DIR}/.." && pwd)}"
EVAL_PROJECT_DIR="${EVAL_PROJECT_DIR:-${REPO_ROOT}/Eval}"
if [[ -z "${ACCELERATE_CONFIG_FILE:-}" ]]; then
  if grep -Eq 'deepspeed_config_path:.*zero3|deepspeed/zero3' "${RUN_CONFIG}"; then
    ACCELERATE_CONFIG_FILE="${PROJECT_DIR}/configs/accelerate/deepspeed_zero3.yaml"
  else
    ACCELERATE_CONFIG_FILE="${PROJECT_DIR}/configs/accelerate/deepspeed_zero2.yaml"
  fi
fi
ACCELERATE_CONFIG_FILE="$(realpath -m "${ACCELERATE_CONFIG_FILE}")"

if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "Run config not found: ${RUN_CONFIG}"
  exit 1
fi
if [[ ! -f "${ACCELERATE_CONFIG_FILE}" ]]; then
  echo "Accelerate config not found: ${ACCELERATE_CONFIG_FILE}"
  exit 1
fi

SKIP_EVAL="${SKIP_EVAL:-1}"
if [[ "${SKIP_EVAL}" != "1" ]]; then
  EVAL_STANDALONE_SCRIPT="${EVAL_PROJECT_DIR}/scripts/run_standalone_eval.sh"
  if [[ ! -x "${EVAL_STANDALONE_SCRIPT}" ]]; then
    echo "Eval standalone script not found or not executable: ${EVAL_STANDALONE_SCRIPT}"
    exit 1
  fi
fi

GPU_COUNT_RAW="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ -z "${GPU_COUNT_RAW}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT_RAW="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi
NUM_PROCESSES="$(echo "${GPU_COUNT_RAW:-}" | grep -oE '[0-9]+' | head -n 1)"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_MACHINES="${SLURM_JOB_NUM_NODES:-1}"
MACHINE_RANK="${SLURM_NODEID:-0}"

cd "${PROJECT_DIR}"

if ! command -v module >/dev/null 2>&1; then
  for MODULE_INIT in /etc/profile.d/lmod.sh /apps/lmod/lmod/init/bash; do
    if [[ -f "${MODULE_INIT}" ]]; then
      # shellcheck source=/dev/null
      source "${MODULE_INIT}"
      break
    fi
  done
fi

module load python/3.12
module load cuda/12.4.1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SCRATCH_ROOT="/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining"
export CACHE_ROOT="${SCRATCH_ROOT}/cache"
mkdir -p "${CACHE_ROOT}"/{uv,tmp,xdg,wandb,torch,transformers,datasets,hf_home,hf_hub,triton}
mkdir -p "${SCRATCH_ROOT}/slurm_logs"
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
TMPDIR_BASENAME="sft_${SLURM_JOB_ID:-local}_${USER}"
export TMPDIR="/tmp/${TMPDIR_BASENAME}"
mkdir -p "${TMPDIR}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export WANDB_DIR="${CACHE_ROOT}/wandb"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-decomposed_reasoning_sft}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export HF_HOME="${CACHE_ROOT}/hf_home"
export HF_HUB_CACHE="${CACHE_ROOT}/hf_hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_hub"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"

uv sync --frozen --extra train
TRAIN_ENV_DIR="${UV_PROJECT_ENVIRONMENT:-${PROJECT_DIR}/.venv}"
TRAIN_PYTHON_BIN="${TRAIN_ENV_DIR}/bin/python"
ACCELERATE_BIN="${TRAIN_ENV_DIR}/bin/accelerate"
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "Accelerate binary not found in project venv: ${ACCELERATE_BIN}"
  exit 1
fi

JOB_TIMESTAMP="$(date -u +"%Y-%m-%d_%H-%M-%S")"
WANDB_GROUP_NAME="${RUN_CONFIG_STEM}_${JOB_TIMESTAMP}"
TRAIN_WANDB_RUN_NAME="${RUN_CONFIG_STEM}_train_${JOB_TIMESTAMP}"
WANDB_RUN_ID="$("${TRAIN_PYTHON_BIN}" -c "import uuid; print(uuid.uuid4().hex)")"
GRADIENT_ACCUMULATION_STEPS="$(
  RUN_CONFIG_PATH="${RUN_CONFIG}" "${TRAIN_PYTHON_BIN}" - <<'PY'
import os
from pathlib import Path

from sft_training.config_types import RunConfig

config = RunConfig.from_yaml(yaml_path=Path(os.environ["RUN_CONFIG_PATH"]))
print(config.gradient_accumulation_steps)
PY
)"
export SFT_JOB_TIMESTAMP="${JOB_TIMESTAMP}"
export SFT_WANDB_GROUP="${WANDB_GROUP_NAME}"
export SFT_WANDB_RUN_NAME="${TRAIN_WANDB_RUN_NAME}"
export SFT_WANDB_RUN_ID="${WANDB_RUN_ID}"

echo "Run config: ${RUN_CONFIG}"
echo "Accelerate config: ${ACCELERATE_CONFIG_FILE}"
echo "Processes: ${NUM_PROCESSES}"
echo "W&B mode: ${WANDB_MODE}"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B train run name: ${TRAIN_WANDB_RUN_NAME}"
echo "W&B group: ${WANDB_GROUP_NAME}"
echo "W&B run id: ${WANDB_RUN_ID}"
echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"

"${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  -m sft_training.train \
  --config "${RUN_CONFIG}" \
  "$@"

if [[ "${SKIP_EVAL}" == "1" ]]; then
  echo "SKIP_EVAL=1, skipping post-training standalone benchmark eval."
  exit 0
fi

echo "Training complete. Use slurm/train.sbatch if post-training eval is needed."
