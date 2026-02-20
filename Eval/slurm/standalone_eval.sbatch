#!/bin/bash
#SBATCH --job-name=standalone-eval
#SBATCH --account=PAS3268
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch slurm/standalone_eval.sbatch <checkpoint-path-or-repo> <config-yaml> [extra-standalone-args...]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${EVAL_PROJECT_DIR:-}"
if [[ -z "${PROJECT_DIR}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}/eval_runner" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}/Eval/eval_runner" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}/Eval"
  else
    PROJECT_DIR="${DEFAULT_PROJECT_DIR}"
  fi
fi
CHECKPOINT_PATH="$1"
CONFIG_PATH="$2"
shift 2

CHECKPOINT_PATH="$(realpath -m "${CHECKPOINT_PATH}")"
CONFIG_PATH="$(realpath -m "${CONFIG_PATH}")"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}"
  exit 1
fi

LOG_ROOT_DIR="${SLURM_SUBMIT_DIR:-${PROJECT_DIR}}"
mkdir -p "${LOG_ROOT_DIR}/logs"
cd "${PROJECT_DIR}"

module load python/3.12
module load cuda/12.4.1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SCRATCH_ROOT="/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/Eval"
export CACHE_ROOT="${SCRATCH_ROOT}/cache"
mkdir -p "${CACHE_ROOT}"/{uv,tmp,xdg,wandb,torch,transformers,datasets,hf_home,hf_hub,triton}
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
TMPDIR_BASENAME="eval_${SLURM_JOB_ID:-local}_${USER}"
export TMPDIR="/tmp/${TMPDIR_BASENAME}"
mkdir -p "${TMPDIR}"
export XDG_CACHE_HOME="${CACHE_ROOT}/xdg"
export WANDB_DIR="${CACHE_ROOT}/wandb"
export TORCH_HOME="${CACHE_ROOT}/torch"
export HF_DATASETS_CACHE="${CACHE_ROOT}/datasets"
export HF_HOME="${CACHE_ROOT}/hf_home"
export HF_HUB_CACHE="${CACHE_ROOT}/hf_hub"
export HUGGINGFACE_HUB_CACHE="${CACHE_ROOT}/hf_hub"
export TRITON_CACHE_DIR="${CACHE_ROOT}/triton"

uv sync --frozen
EVAL_PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${EVAL_PYTHON_BIN}" ]]; then
  echo "Eval python binary not found: ${EVAL_PYTHON_BIN}"
  exit 1
fi
EVAL_STANDALONE_SCRIPT="${PROJECT_DIR}/scripts/run_standalone_eval.sh"
if [[ ! -x "${EVAL_STANDALONE_SCRIPT}" ]]; then
  echo "Eval standalone script not found or not executable: ${EVAL_STANDALONE_SCRIPT}"
  exit 1
fi

echo "Running standalone eval"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Config: ${CONFIG_PATH}"

EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN}" "${EVAL_STANDALONE_SCRIPT}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --config "${CONFIG_PATH}" \
  "$@"
