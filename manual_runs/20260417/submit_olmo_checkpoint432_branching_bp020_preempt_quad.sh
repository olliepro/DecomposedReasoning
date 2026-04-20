#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
RUN_DIR="${REPO_ROOT}/manual_runs/20260417"
LOG_DIR="${RUN_DIR}/logs"
MODEL_ID="sft"
SEED_VALUE="1234"
MODEL_ROOT="/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_think_sft_to_think_merged_2213/checkpoint-432"

ANALYSIS_SCRIPT="${RUN_DIR}/run_analysis_aime_preempt_quad.sbatch"
RAND_AIME24_CONFIG="${RUN_DIR}/branching_aime24_olmo3_7b_checkpoint432_random_bp020_32k_exec1024_preemptquad.yaml"
CLUS_AIME24_CONFIG="${RUN_DIR}/branching_aime24_olmo3_7b_checkpoint432_cluster_across_bp020_32k_exec1024_preemptquad.yaml"
RAND_AIME25_CONFIG="${RUN_DIR}/branching_aime25_olmo3_7b_checkpoint432_random_bp020_32k_exec1024_preemptquad.yaml"
CLUS_AIME25_CONFIG="${RUN_DIR}/branching_aime25_olmo3_7b_checkpoint432_cluster_across_bp020_32k_exec1024_preemptquad.yaml"

RAND_TIME="07:30:00"
CLUS_TIME="08:00:00"
CPUS_PER_TASK="32"
MEMORY_PER_NODE="256G"

mkdir -p "${LOG_DIR}"

validate_analysis_config() {
  local config_path="$1"
  local expected_task="$2"
  (
    cd "${REPO_ROOT}/Analysis"
    "${HOME}/.local/bin/uv" run python -c '
from pathlib import Path
import sys
from branching_eval.config_types import load_branching_eval_config

config_path = Path(sys.argv[1])
expected_task = sys.argv[2]
config = load_branching_eval_config(config_path=config_path)
assert config.tasks.task_names == (expected_task,), config.tasks
assert config.decoding.max_gen_toks == 32768, config.decoding
assert config.decoding.decode_chunk_tokens == 1024, config.decoding
assert config.branching.max_concurrent_branches == 48, config.branching
assert abs(config.branching.branch_prob - 0.20) < 1e-9, config.branching
assert abs(config.branching.epsilon_greedy_prob - 0.0) < 1e-9, config.branching
assert config.run_matrix.max_concurrent_docs == 4, config.run_matrix
print(config_path)
' "${config_path}" "${expected_task}"
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

validate_analysis_config "${RAND_AIME24_CONFIG}" "aime24"
validate_analysis_config "${CLUS_AIME24_CONFIG}" "aime24"
validate_analysis_config "${RAND_AIME25_CONFIG}" "aime25"
validate_analysis_config "${CLUS_AIME25_CONFIG}" "aime25"

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
  "a25-olmo-rand20-exec1024-pq-c432" \
  "${RAND_TIME}" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${RAND_AIME25_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a25-olmo-clus20-exec1024-pq-c432" \
  "${CLUS_TIME}" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${CLUS_AIME25_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a24-olmo-rand20-exec1024-pq-c432" \
  "${RAND_TIME}" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${RAND_AIME24_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a24-olmo-clus20-exec1024-pq-c432" \
  "${CLUS_TIME}" \
  --export=ALL,MODEL_ROOT="${MODEL_ROOT}" \
  "${ANALYSIS_SCRIPT}" \
  "${CLUS_AIME24_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

printf 'Submitted job IDs: %s\n' "${submitted_job_ids[*]}"
