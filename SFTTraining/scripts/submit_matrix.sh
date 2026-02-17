#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_SCRIPT="${PROJECT_DIR}/slurm/train.sbatch"

declare -a RUN_CONFIGS=(
  "${PROJECT_DIR}/configs/runs/olmo3_7b_think_to_think.yaml"
  "${PROJECT_DIR}/configs/runs/olmo3_7b_instruct_to_think.yaml"
  "${PROJECT_DIR}/configs/runs/qwen3_8b_to_think_low_lr.yaml"
  "${PROJECT_DIR}/configs/runs/qwen3_8b_to_think.yaml"
)

previous_job_id=""
for run_config in "${RUN_CONFIGS[@]}"; do
  run_config_name="$(basename "${run_config}")"
  run_config_stem="${run_config_name%.*}"
  job_timestamp="$(date -u +"%Y%m%dT%H%M%SZ")"
  slurm_job_name="sft-${run_config_stem}-${job_timestamp}"
  submit_command=(sbatch)
  if [[ -n "${previous_job_id}" ]]; then
    submit_command+=(--dependency="afterok:${previous_job_id}")
  fi
  submit_command+=(--job-name="${slurm_job_name}")
  submit_command+=("${SBATCH_SCRIPT}" "${run_config}")
  submit_output="$("${submit_command[@]}")"
  previous_job_id="$(echo "${submit_output}" | awk '{print $4}')"
  echo "Submitted ${run_config} as job ${previous_job_id} (${slurm_job_name})"
done
