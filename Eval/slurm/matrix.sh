#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MATRIX_DRY_RUN="${MATRIX_DRY_RUN:-0}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DEFAULT_JOB_TIMESTAMP="$(date -u +"%Y-%m-%dT%H-%M-%SZ")"
MATRIX_JOB_TIMESTAMP="${SFT_JOB_TIMESTAMP:-${DEFAULT_JOB_TIMESTAMP}}"

# Backward-compatible behavior:
# 1) RUN_DIRS (space-delimited) if provided
# 2) RUN_DIR (single path) if provided
# 3) Default single run dir
if [[ -n "${RUN_DIRS:-}" ]]; then
  RUN_DIRS_TEXT="${RUN_DIRS}"
elif [[ -n "${RUN_DIR:-}" ]]; then
  RUN_DIRS_TEXT="${RUN_DIR}"
else
  RUN_DIRS_TEXT="/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/qwen3_8b_to_think_aimek4_b1_ga12"
fi
read -r -a RUN_DIR_PATHS <<< "${RUN_DIRS_TEXT}"

CONFIG="${CONFIG:-/users/PAA0201/ollieproudman/work/DecomposedReasoning/SFTTraining/configs/runs/qwen3_8b.yaml}"
SBATCH_SCRIPT="${SBATCH_SCRIPT:-${SCRIPT_DIR}/standalone_eval.sbatch}"
INCLUDE_BASELINE_MODEL="${INCLUDE_BASELINE_MODEL:-1}"
BASELINE_MODEL="${BASELINE_MODEL:-Qwen/Qwen3-8B}"
BASELINE_LABEL="${BASELINE_LABEL:-qwen3_8b_base_hf}"

# Space-delimited task list. Example override:
# TASKS="aime24 aime25" bash Eval/slurm/matrix.sh
TASKS_TEXT="${TASKS:-minerva_math500 aime24 aime25}"
read -r -a TASK_NAMES <<< "${TASKS_TEXT}"

if [[ ! -f "${CONFIG}" ]]; then
  echo "Config not found: ${CONFIG}"
  exit 1
fi
if [[ ! -f "${SBATCH_SCRIPT}" ]]; then
  echo "SBATCH script not found: ${SBATCH_SCRIPT}"
  exit 1
fi
if [[ ${#TASK_NAMES[@]} -eq 0 ]]; then
  echo "No tasks configured. Set TASKS to a space-delimited task list."
  exit 1
fi

if [[ ${#RUN_DIR_PATHS[@]} -eq 0 ]]; then
  echo "No run directories configured. Set RUN_DIRS or RUN_DIR."
  exit 1
fi

discover_checkpoint_paths() {
  local run_dir_path="$1"
  PYTHONPATH="${PROJECT_DIR}" "${PYTHON_BIN}" -m eval_runner.checkpoint_sweep \
    discover \
    --run-dir "${run_dir_path}"
}

submit_eval_job() {
  local export_spec="$1"
  local job_name="$2"
  shift 2
  if [[ "${MATRIX_DRY_RUN}" == "1" ]]; then
    printf 'DRY_RUN %q --export=%q --job-name %q' "${SBATCH_BIN}" "${export_spec}" "${job_name}"
    for arg in "$@"; do
      printf ' %q' "${arg}"
    done
    printf '\n'
    return
  fi
  "${SBATCH_BIN}" --export="${export_spec}" --job-name "${job_name}" "$@"
}

submitted_job_count=0
for run_dir_path in "${RUN_DIR_PATHS[@]}"; do
  if [[ ! -d "${run_dir_path}" ]]; then
    echo "Skipping missing run directory: ${run_dir_path}"
    continue
  fi

  mkdir -p "${run_dir_path}/benchmark_evals"
  mapfile -t checkpoint_paths < <(discover_checkpoint_paths "${run_dir_path}")
  run_dir_name="$(basename "${run_dir_path}")"
  run_group_name="${SFT_WANDB_GROUP:-${run_dir_name}_checkpoint_eval_${MATRIX_JOB_TIMESTAMP}}"
  export_spec="ALL,SFT_JOB_TIMESTAMP=${MATRIX_JOB_TIMESTAMP},SFT_WANDB_GROUP=${run_group_name},SFT_WANDB_RUN_ID="

  valid_checkpoint_count=0
  for checkpoint_path in "${checkpoint_paths[@]}"; do
    [[ -d "${checkpoint_path}" ]] || continue
    valid_checkpoint_count=$((valid_checkpoint_count + 1))
    checkpoint_label="$(basename "${checkpoint_path}")"
    for task_name in "${TASK_NAMES[@]}"; do
      output_json="${run_dir_path}/benchmark_evals/${checkpoint_label}_${task_name}.json"
      submit_eval_job \
        "${export_spec}" \
        "eval-${checkpoint_label}-${task_name}" \
        "${SBATCH_SCRIPT}" \
        "${checkpoint_path}" \
        "${CONFIG}" \
        --tasks "${task_name}" \
        --output "${output_json}"
      submitted_job_count=$((submitted_job_count + 1))
    done
  done

  if [[ "${valid_checkpoint_count}" -eq 0 ]]; then
    echo "No raw checkpoint directories found under: ${run_dir_path}"
    continue
  fi

  echo "Ranking command after jobs finish:"
  echo "PYTHONPATH=\"${PROJECT_DIR}\" ${PYTHON_BIN} -m eval_runner.checkpoint_sweep rank --run-dir \"${run_dir_path}\""
done

if [[ "${INCLUDE_BASELINE_MODEL}" == "1" ]]; then
  baseline_output_parent=""
  for run_dir_path in "${RUN_DIR_PATHS[@]}"; do
    if [[ -d "${run_dir_path}" ]]; then
      baseline_output_parent="${run_dir_path}"
      break
    fi
  done
  if [[ -z "${baseline_output_parent}" ]]; then
    baseline_output_parent="${RUN_DIR_PATHS[0]}"
  fi
  mkdir -p "${baseline_output_parent}/benchmark_evals_base"
  baseline_group_name="${SFT_WANDB_GROUP:-$(basename "${baseline_output_parent}")_checkpoint_eval_${MATRIX_JOB_TIMESTAMP}}"
  baseline_export_spec="ALL,SFT_JOB_TIMESTAMP=${MATRIX_JOB_TIMESTAMP},SFT_WANDB_GROUP=${baseline_group_name},SFT_WANDB_RUN_ID="
  for task_name in "${TASK_NAMES[@]}"; do
    output_json="${baseline_output_parent}/benchmark_evals_base/${BASELINE_LABEL}_${task_name}.json"
    submit_eval_job \
      "${baseline_export_spec}" \
      "eval-${BASELINE_LABEL}-${task_name}" \
      "${SBATCH_SCRIPT}" \
      "${BASELINE_MODEL}" \
      "${CONFIG}" \
      --tasks "${task_name}" \
      --output "${output_json}"
    submitted_job_count=$((submitted_job_count + 1))
  done
fi

if [[ "${submitted_job_count}" -eq 0 ]]; then
  echo "No jobs submitted."
  exit 1
fi

echo "Submitted ${submitted_job_count} eval jobs."
