#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
RUN_DIR="${REPO_ROOT}/manual_runs/20260417"
LOG_DIR="${RUN_DIR}/logs"
MODEL_ID="sft"
SEED_VALUE="1234"
MODEL_ROOT="/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_think_sft_to_think_merged_2213/checkpoint-432"
SOURCE_AIME24="${REPO_ROOT}/Eval/output/manual_20260413/samples_aime24_2026-04-14T00-32-35.203153.jsonl"

STANDALONE_SCRIPT="${RUN_DIR}/run_olmo_ckpt_steerprefix_eval_preempt_quad.sbatch"
ANALYSIS_SCRIPT="${RUN_DIR}/run_analysis_aime_preempt_quad.sbatch"
STRUCT_AIME24_CONFIG="${RUN_DIR}/structured_aime24_olmo3_7b_checkpoint432_n32_32k_exec1024_preemptquad.yaml"
RAND_AIME24_CONFIG="${RUN_DIR}/branching_aime24_olmo3_7b_checkpoint432_random_bp010_32k_exec1024_preemptquad.yaml"
CLUS_AIME24_CONFIG="${RUN_DIR}/branching_aime24_olmo3_7b_checkpoint432_cluster_across_bp010_32k_exec1024_preemptquad.yaml"
RAND_AIME25_CONFIG="${RUN_DIR}/branching_aime25_olmo3_7b_checkpoint432_random_bp010_32k_exec1024_preemptquad.yaml"
CLUS_AIME25_CONFIG="${RUN_DIR}/branching_aime25_olmo3_7b_checkpoint432_cluster_across_bp010_32k_exec1024_preemptquad.yaml"

CKPT_TIME="07:30:00"
STRUCT_TIME="07:00:00"
RAND_TIME="07:30:00"
CLUS_TIME="08:00:00"
CPUS_PER_TASK="32"
MEMORY_PER_NODE="256G"

mkdir -p "${LOG_DIR}"

validate_analysis_config() {
  local config_path="$1"
  local expected_task="$2"
  local expected_branch_prob="$3"
  (
    cd "${REPO_ROOT}/Analysis"
    "${HOME}/.local/bin/uv" run python -c '
from pathlib import Path
import sys
from branching_eval.config_types import load_branching_eval_config

config_path = Path(sys.argv[1])
expected_task = sys.argv[2]
expected_branch_prob = float(sys.argv[3])
config = load_branching_eval_config(config_path=config_path)
assert config.tasks.task_names == (expected_task,), config.tasks
assert config.decoding.max_gen_toks == 32768, config.decoding
assert config.decoding.decode_chunk_tokens == 1024, config.decoding
assert config.branching.max_concurrent_branches == 24, config.branching
assert abs(config.branching.branch_prob - expected_branch_prob) < 1e-9, config.branching
print(config_path)
' "${config_path}" "${expected_task}" "${expected_branch_prob}"
  )
}

validate_standalone_prereqs() {
  (
    cd "${REPO_ROOT}"
    bash -n "${STANDALONE_SCRIPT}"
    python -m py_compile "${REPO_ROOT}/manual_runs/20260415/olmo_steerprefix_eval.py"
    test -f "${SOURCE_AIME24}"
  )
}

submit_checked_job() {
  local job_name="$1"
  local time_limit="$2"
  shift 2
  sbatch --test-only \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${time_limit}" \
    --job-name "${job_name}" \
    "$@" >/dev/null
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${time_limit}" \
    --job-name "${job_name}" \
    "$@"
}

validate_standalone_prereqs
validate_analysis_config "${STRUCT_AIME24_CONFIG}" "aime24" "0.0"
validate_analysis_config "${RAND_AIME24_CONFIG}" "aime24" "0.10"
validate_analysis_config "${CLUS_AIME24_CONFIG}" "aime24" "0.10"
validate_analysis_config "${RAND_AIME25_CONFIG}" "aime25" "0.10"
validate_analysis_config "${CLUS_AIME25_CONFIG}" "aime25" "0.10"

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
  "a25-olmo-rand10-exec1024-pq-c432" \
  "${RAND_TIME}" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${RAND_AIME25_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a25-olmo-clus10-exec1024-pq-c432" \
  "${CLUS_TIME}" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${CLUS_AIME25_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a24-olmo-ckpt32-steer-pq-c432" \
  "${CKPT_TIME}" \
  --export=ALL,TASK_NAME=aime24,SOURCE_SAMPLES="${SOURCE_AIME24}",MODEL_ROOT="${MODEL_ROOT}" \
  "${STANDALONE_SCRIPT}"

submit_and_record \
  "a24-olmo-struct32-exec1024-pq-c432" \
  "${STRUCT_TIME}" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${STRUCT_AIME24_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a24-olmo-rand10-exec1024-pq-c432" \
  "${RAND_TIME}" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${RAND_AIME24_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a24-olmo-clus10-exec1024-pq-c432" \
  "${CLUS_TIME}" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${CLUS_AIME24_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

printf 'Submitted job IDs: %s\n' "${submitted_job_ids[*]}"
