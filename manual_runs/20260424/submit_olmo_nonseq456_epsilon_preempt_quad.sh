#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
RUN_DIR="${REPO_ROOT}/manual_runs/20260424"
LOG_DIR="${RUN_DIR}/logs"
MODEL_ID="sft"
SEED_VALUE="1234"

ANALYSIS_SCRIPT="${RUN_DIR}/run_analysis_aime_nonseq456_preempt_quad.sbatch"
EPS_AIME24_CONFIG="${RUN_DIR}/epsilon_aime24_olmo3_7b_nonseq456_eps033_n32_32k_exec1024_preemptquad.yaml"
EPS_AIME25_CONFIG="${RUN_DIR}/epsilon_aime25_olmo3_7b_nonseq456_eps033_n32_32k_exec1024_preemptquad.yaml"

EPS_TIME="07:30:00"
CPUS_PER_TASK="32"
MEMORY_PER_NODE="256G"

mkdir -p "${LOG_DIR}"

validate_epsilon_config() {
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
expected_model = "/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_think_sft_to_think_merged_2213_non_sequitur_masked_rerun_20260423/checkpoint-456"
config = load_branching_eval_config(config_path=config_path)
assert config.tasks.task_names == (expected_task,), config.tasks
assert str(config.models[0].checkpoint_or_repo) == expected_model, config.models
assert config.decoding.max_gen_toks == 32768, config.decoding
assert config.decoding.decode_chunk_tokens == 1024, config.decoding
assert abs(config.serve.gpu_memory_utilization - 0.90) < 1e-9, config.serve
assert config.branching.max_concurrent_branches == 32, config.branching
assert abs(config.branching.branch_prob - 0.0) < 1e-9, config.branching
assert abs(config.branching.epsilon_greedy_prob - 0.33) < 1e-9, config.branching
assert config.run_matrix.include_branching is False, config.run_matrix
assert config.run_matrix.include_epsilon_greedy is True, config.run_matrix
assert config.run_matrix.baseline_rollouts == 32, config.run_matrix
assert config.run_matrix.max_concurrent_docs == 4, config.run_matrix
assert config.run_matrix.selectors == ("embed_diverse_topk_random",), config.run_matrix
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

validate_epsilon_config "${EPS_AIME24_CONFIG}" "aime24"
validate_epsilon_config "${EPS_AIME25_CONFIG}" "aime25"

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
  "a24-olmo456nseq-eps033-mc32-gm90" \
  "${EPS_TIME}" \
  "${ANALYSIS_SCRIPT}" \
  "${EPS_AIME24_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a25-olmo456nseq-eps033-mc32-gm90" \
  "${EPS_TIME}" \
  "${ANALYSIS_SCRIPT}" \
  "${EPS_AIME25_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

printf 'Submitted job IDs: %s\n' "${submitted_job_ids[*]}"
