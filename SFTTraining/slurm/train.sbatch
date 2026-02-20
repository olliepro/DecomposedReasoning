#!/bin/bash
#SBATCH --job-name=sft-train
#SBATCH --account=PAS3268
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch slurm/train.sbatch <run-config-yaml> [extra-train-args...]"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_DIR="${SFT_PROJECT_DIR:-}"
if [[ -z "${PROJECT_DIR}" ]]; then
  if [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}/sft_training" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}"
  elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && [[ -d "${SLURM_SUBMIT_DIR}/SFTTraining/sft_training" ]]; then
    PROJECT_DIR="${SLURM_SUBMIT_DIR}/SFTTraining"
  else
    PROJECT_DIR="${DEFAULT_PROJECT_DIR}"
  fi
fi
REPO_ROOT="${REPO_ROOT:-$(cd "${PROJECT_DIR}/.." && pwd)}"
EVAL_PROJECT_DIR="${EVAL_PROJECT_DIR:-${REPO_ROOT}/Eval}"
RUN_CONFIG="$1"
shift
RUN_CONFIG="$(realpath -m "${RUN_CONFIG}")"
RUN_CONFIG_NAME="$(basename "${RUN_CONFIG}")"
RUN_CONFIG_STEM="${RUN_CONFIG_NAME%.*}"
ACCELERATE_CONFIG_FILE="${ACCELERATE_CONFIG_FILE:-${PROJECT_DIR}/configs/accelerate/deepspeed_zero2.yaml}"
ACCELERATE_CONFIG_FILE="$(realpath -m "${ACCELERATE_CONFIG_FILE}")"
GPU_COUNT_RAW="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-}}"
if [[ -z "${GPU_COUNT_RAW}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT_RAW="$(nvidia-smi -L | wc -l | tr -d ' ')"
fi
NUM_PROCESSES="$(echo "${GPU_COUNT_RAW:-}" | grep -oE '[0-9]+' | head -n 1)"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
NUM_MACHINES="${SLURM_JOB_NUM_NODES:-1}"
MACHINE_RANK="${SLURM_NODEID:-0}"

# Mitigate a reproducible NCCL barrier deadlock for 2-GPU ZeRO-2 runs.
# Users can override any of these by pre-setting env vars at launch time.
# if [[ "${NUM_PROCESSES}" -eq 2 ]]; then
#   export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
#   export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
#   export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-0}"
#   export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
#   echo "Applied 2-GPU NCCL stability defaults (override via env):"
#   echo "  NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE}"
#   echo "  NCCL_IB_DISABLE=${NCCL_IB_DISABLE}"
#   echo "  NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE}"
#   echo "  TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING}"
# fi

if [[ ! -f "${RUN_CONFIG}" ]]; then
  echo "Run config not found: ${RUN_CONFIG}"
  exit 1
fi
if [[ ! -f "${ACCELERATE_CONFIG_FILE}" ]]; then
  echo "Accelerate config not found: ${ACCELERATE_CONFIG_FILE}"
  exit 1
fi
if [[ ! -d "${EVAL_PROJECT_DIR}" ]]; then
  echo "Eval project directory not found: ${EVAL_PROJECT_DIR}"
  exit 1
fi
EVAL_STANDALONE_SCRIPT="${EVAL_PROJECT_DIR}/scripts/run_standalone_eval.sh"
if [[ ! -x "${EVAL_STANDALONE_SCRIPT}" ]]; then
  echo "Eval standalone script not found or not executable: ${EVAL_STANDALONE_SCRIPT}"
  exit 1
fi

LOG_ROOT_DIR="${SLURM_SUBMIT_DIR:-${PROJECT_DIR}}"
mkdir -p "${LOG_ROOT_DIR}/logs"

cd "${PROJECT_DIR}"

module load python/3.12
module load cuda/12.4.1

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export SCRATCH_ROOT="/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining"
export CACHE_ROOT="${SCRATCH_ROOT}/cache"
mkdir -p "${CACHE_ROOT}"/{uv,tmp,xdg,wandb,torch,transformers,datasets,hf_home,hf_hub,triton}
export UV_CACHE_DIR="${CACHE_ROOT}/uv"
TMPDIR_BASENAME="sft_${SLURM_JOB_ID:-local}_${USER}"
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

uv sync --frozen --extra train
TRAIN_PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python"
ACCELERATE_BIN="${PROJECT_DIR}/.venv/bin/accelerate"
if [[ ! -x "${ACCELERATE_BIN}" ]]; then
  echo "Accelerate binary not found in project venv: ${ACCELERATE_BIN}"
  exit 1
fi
(
  cd "${EVAL_PROJECT_DIR}"
  uv sync --frozen
)
EVAL_PYTHON_BIN="${EVAL_PROJECT_DIR}/.venv/bin/python"
if [[ ! -x "${EVAL_PYTHON_BIN}" ]]; then
  echo "Eval python binary not found: ${EVAL_PYTHON_BIN}"
  exit 1
fi
JOB_TIMESTAMP="$(date -u +"%Y-%m-%d_%H-%M-%S")"
WANDB_GROUP_NAME="${RUN_CONFIG_STEM}_${JOB_TIMESTAMP}"
TRAIN_WANDB_RUN_NAME="${RUN_CONFIG_STEM}_train_${JOB_TIMESTAMP}"
WANDB_RUN_ID="$("${TRAIN_PYTHON_BIN}" -c "import uuid; print(uuid.uuid4().hex)")"
export SFT_JOB_TIMESTAMP="${JOB_TIMESTAMP}"
export SFT_WANDB_GROUP="${WANDB_GROUP_NAME}"
export SFT_WANDB_RUN_NAME="${TRAIN_WANDB_RUN_NAME}"
export SFT_WANDB_RUN_ID="${WANDB_RUN_ID}"

echo "W&B train run name: ${TRAIN_WANDB_RUN_NAME}"
echo "W&B group: ${WANDB_GROUP_NAME}"
echo "W&B run id: ${WANDB_RUN_ID}"

"${ACCELERATE_BIN}" launch \
  --config_file "${ACCELERATE_CONFIG_FILE}" \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  -m sft_training.train \
  --config "${RUN_CONFIG}" \
  "$@"

RUN_OUTPUT_DIR="$(
  RUN_CONFIG_PATH="${RUN_CONFIG}" "${TRAIN_PYTHON_BIN}" -c "import os; from pathlib import Path; from sft_training.config_types import RunConfig; config = RunConfig.from_yaml(yaml_path=Path(os.environ['RUN_CONFIG_PATH'])); print(config.output_dir)"
)"
EVAL_OUTPUT_DIR="${RUN_OUTPUT_DIR}/benchmark_evals"
mkdir -p "${EVAL_OUTPUT_DIR}"
EVAL_LIMIT="${EVAL_LIMIT:-}"
EVAL_LIMIT_ARGS=()
if [[ -n "${EVAL_LIMIT}" ]]; then
  if ! [[ "${EVAL_LIMIT}" =~ ^[0-9]+$ ]] || [[ "${EVAL_LIMIT}" -lt 1 ]]; then
    echo "EVAL_LIMIT must be a positive integer when set."
    exit 1
  fi
  EVAL_LIMIT_ARGS=(--limit "${EVAL_LIMIT}")
fi

run_standalone_eval() {
  local checkpoint_dir="$1"
  local output_json_path="$2"

  echo "Running standalone benchmark eval from: ${checkpoint_dir}"
  echo "W&B shared run name: ${SFT_WANDB_RUN_NAME}"
  EVAL_PYTHON_BIN="${EVAL_PYTHON_BIN}" "${EVAL_STANDALONE_SCRIPT}" \
    --checkpoint "${checkpoint_dir}" \
    --config "${RUN_CONFIG}" \
    --output "${output_json_path}" \
    "${EVAL_LIMIT_ARGS[@]}"
}

prepare_hf_checkpoint_for_eval() {
  local checkpoint_dir="$1"
  local output_dir="${checkpoint_dir}-hf"
  local converter_script="${checkpoint_dir}/zero_to_fp32.py"

  if [[ ! -f "${converter_script}" ]]; then
    echo "Expected conversion script not found: ${converter_script}"
    exit 1
  fi

  if [[ ! -f "${output_dir}/model.safetensors.index.json" ]]; then
    echo "Converting checkpoint to HF format: ${checkpoint_dir} -> ${output_dir}"
    mkdir -p "${output_dir}"
    rsync -a \
      --exclude 'global_step*' \
      --exclude 'latest' \
      --exclude 'zero_to_fp32.py' \
      --exclude 'trainer_state.json' \
      --exclude 'training_args.bin' \
      --exclude 'pytorch_model*' \
      --exclude 'model*.safetensors*' \
      "${checkpoint_dir}/" "${output_dir}/"
    "${TRAIN_PYTHON_BIN}" "${converter_script}" "${checkpoint_dir}" "${output_dir}" \
      --safe_serialization \
      --max_shard_size 5GB
  else
    echo "HF checkpoint already exists, skipping conversion: ${output_dir}"
  fi

  echo "${output_dir}"
}

mapfile -t CHECKPOINT_DIRS < <(
  find "${RUN_OUTPUT_DIR}" -maxdepth 1 -mindepth 1 -type d -name "checkpoint-*" \
    | grep -E '/checkpoint-[0-9]+$' \
    | sort -V
)

echo "Training complete. Converting checkpoints to HF format and running standalone benchmark eval."
for checkpoint_dir in "${CHECKPOINT_DIRS[@]}"; do
  checkpoint_label="$(basename "${checkpoint_dir}")-hf"
  checkpoint_output_json="${EVAL_OUTPUT_DIR}/${checkpoint_label}.json"
  hf_checkpoint_dir="$(prepare_hf_checkpoint_for_eval "${checkpoint_dir}")"
  run_standalone_eval \
    "${hf_checkpoint_dir}" \
    "${checkpoint_output_json}"
done
