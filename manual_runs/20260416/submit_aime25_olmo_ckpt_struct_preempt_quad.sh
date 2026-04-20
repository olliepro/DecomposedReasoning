#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
RUN_DIR="${REPO_ROOT}/manual_runs/20260416"
LOG_DIR="${RUN_DIR}/logs"
MODEL_ID="sft"
SEED_VALUE="1234"
TRAIN_JOB_ID="4940062"
MODEL_ROOT="/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_think_sft_to_think_merged_2213"
SOURCE_SAMPLES="${REPO_ROOT}/Eval/output/manual_20260413/samples_aime25_2026-04-14T10-34-48.125475.jsonl"
STANDALONE_SCRIPT="${RUN_DIR}/run_olmo_ckpt_steerprefix_eval_preempt_quad.sbatch"
ANALYSIS_SCRIPT="${RUN_DIR}/run_analysis_aime_preempt_quad.sbatch"
STRUCT_512_CONFIG="${RUN_DIR}/structured_aime25_olmo3_7b_checkpoint448_n32_32k_exec512_preemptquad.yaml"
STRUCT_1024_CONFIG="${RUN_DIR}/structured_aime25_olmo3_7b_checkpoint448_n32_32k_exec1024_preemptquad.yaml"

mkdir -p "${LOG_DIR}"

validate_analysis_config() {
  local config_path="$1"
  (
    cd "${REPO_ROOT}/Analysis"
    "${HOME}/.local/bin/uv" run python -c '
from pathlib import Path
import sys
from branching_eval.config_types import load_branching_eval_config

config_path = Path(sys.argv[1])
config = load_branching_eval_config(config_path=config_path)
assert config.tasks.task_names == ("aime25",), config.tasks
assert config.decoding.max_gen_toks == 32768, config.decoding
assert config.branching.max_concurrent_branches == 24, config.branching
assert config.branching.branch_prob == 0.0, config.branching
print(config_path)
' "${config_path}"
  )
}

validate_standalone_prereqs() {
  (
    cd "${REPO_ROOT}"
    python -m py_compile "${RUN_DIR}/run_olmo_ckpt_steerprefix_eval_preempt_quad.sbatch" >/dev/null 2>&1 || true
    python -m py_compile "${REPO_ROOT}/manual_runs/20260415/olmo_steerprefix_eval.py"
    test -f "${SOURCE_SAMPLES}"
  )
}

submit_checked_job() {
  local job_name="$1"
  local time_limit="$2"
  shift 2
  sbatch --test-only \
    --dependency="afterok:${TRAIN_JOB_ID}" \
    --job-name "${job_name}" \
    --time "${time_limit}" \
    "$@" >/dev/null
  sbatch --parsable \
    --dependency="afterok:${TRAIN_JOB_ID}" \
    --job-name "${job_name}" \
    --time "${time_limit}" \
    "$@"
}

validate_standalone_prereqs
validate_analysis_config "${STRUCT_512_CONFIG}"
validate_analysis_config "${STRUCT_1024_CONFIG}"

declare -a submitted_job_ids=()

submit_and_record() {
  local job_name="$1"
  local time_limit="$2"
  shift 2
  local job_id
  echo "Submitting ${job_name}"
  job_id="$(
    submit_checked_job \
      "${job_name}" \
      "${time_limit}" \
      "$@"
  )"
  submitted_job_ids+=("${job_id}")
  echo "${job_name}=${job_id}"
}

submit_and_record \
  "a25-olmo-ckpt32-steer-pquad-r2" \
  "07:30:00" \
  --export=ALL,TASK_NAME=aime25,SOURCE_SAMPLES="${SOURCE_SAMPLES}",MODEL_ROOT="${MODEL_ROOT}" \
  "${STANDALONE_SCRIPT}"

submit_and_record \
  "a25-olmo-struct32-exec512-pquad-r2" \
  "07:00:00" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${STRUCT_512_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a25-olmo-struct32-exec1024-pquad-r2" \
  "07:00:00" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${STRUCT_1024_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

printf 'Submitted job IDs: %s\n' "${submitted_job_ids[*]}"
