#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
RUN_DIR="${REPO_ROOT}/manual_runs/20260413"
LOG_DIR="${RUN_DIR}/logs"
STANDALONE_SCRIPT="${RUN_DIR}/run_standalone_aime_ascend_quad.sbatch"
ANALYSIS_SCRIPT="${RUN_DIR}/run_analysis_aime_ascend_quad.sbatch"
MODEL_ID="sft"
SEED_VALUE="1234"

QWEN_BASE_REFERENCE="Qwen/Qwen3-8B"
QWEN_CHECKPOINT_REFERENCE="/users/PAA0201/ollieproudman/work/DecomposedReasoning/scratch_checkpoints/qwen3_8b_to_think_merged_2213_zero3/checkpoint-456"
OLMO_BASE_REFERENCE="allenai/Olmo-3-7B-Think-SFT"
OLMO_CHECKPOINT_REFERENCE="/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_think_sft_to_think_merged_2213/checkpoint-448"

A25_QWEN_BASE_TIME="05:15:00"
A25_QWEN_CKPT_TIME="05:45:00"
A25_QWEN_STRUCT_TIME="06:00:00"
A24_OLMO_BASE_TIME="05:30:00"
A24_OLMO_CKPT_TIME="06:15:00"
A25_OLMO_BASE_TIME="06:30:00"
A25_OLMO_CKPT_TIME="07:15:00"
A25_OLMO_RAND_TIME="05:30:00"
CPUS_PER_TASK="32"
MEMORY_PER_NODE="256G"

mkdir -p "${LOG_DIR}"

validate_standalone_config() {
  local config_path="$1"
  (
    cd "${REPO_ROOT}/Eval"
    uv run python -c '
from pathlib import Path
import sys
from eval_runner.config_types import parse_eval_config

config_path = Path(sys.argv[1])
config, _ = parse_eval_config(config_path=config_path)
assert config.max_gen_toks == 32768, config
assert config.think_end_token is None, config
assert config.log_samples, config
print(config_path)
' "${config_path}"
  )
}

validate_analysis_config() {
  local config_path="$1"
  (
    cd "${REPO_ROOT}/Analysis"
    uv run python -c '
from pathlib import Path
import sys
from branching_eval.config_types import load_branching_eval_config

config_path = Path(sys.argv[1])
config = load_branching_eval_config(config_path=config_path)
assert config.decoding.max_gen_toks == 32768, config
assert config.branching.max_concurrent_branches == 24, config
print(config_path)
' "${config_path}"
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

STANDALONE_CONFIGS=(
  "${REPO_ROOT}/manual_runs/20260410/eval_aime24_32k_rawthink_samples.yaml"
  "${REPO_ROOT}/manual_runs/20260410/eval_aime25_32k_rawthink_samples.yaml"
)
ANALYSIS_CONFIGS=(
  "${RUN_DIR}/structured_aime25_qwen3_8b_checkpoint456_n32_32k_mc24_rerun.yaml"
  "${RUN_DIR}/branching_aime25_olmo3_7b_checkpoint448_random_bp010_32k_mc24_rerun.yaml"
)

for config_path in "${STANDALONE_CONFIGS[@]}"; do
  validate_standalone_config "${config_path}"
done

for config_path in "${ANALYSIS_CONFIGS[@]}"; do
  validate_analysis_config "${config_path}"
done

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
  "a25-qwen-base32-32k-r2" \
  "${A25_QWEN_BASE_TIME}" \
  "${STANDALONE_SCRIPT}" \
  "${QWEN_BASE_REFERENCE}" \
  "${REPO_ROOT}/manual_runs/20260410/eval_aime25_32k_rawthink_samples.yaml" \
  "${REPO_ROOT}/Eval/output/manual_20260413/aime25_qwen3_8b_hf_tp2_avg32_32k_rawthink_samples_r2.json"

submit_and_record \
  "a25-qwen-ckpt32-32k-r2" \
  "${A25_QWEN_CKPT_TIME}" \
  "${STANDALONE_SCRIPT}" \
  "${QWEN_CHECKPOINT_REFERENCE}" \
  "${REPO_ROOT}/manual_runs/20260410/eval_aime25_32k_rawthink_samples.yaml" \
  "${REPO_ROOT}/Eval/output/manual_20260413/aime25_qwen3_8b_checkpoint456_tp2_avg32_32k_rawthink_samples_r2.json"

submit_and_record \
  "a25-qwen-struct32-32k-r2" \
  "${A25_QWEN_STRUCT_TIME}" \
  "${ANALYSIS_SCRIPT}" \
  "${RUN_DIR}/structured_aime25_qwen3_8b_checkpoint456_n32_32k_mc24_rerun.yaml" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

submit_and_record \
  "a24-olmo-base32-32k-r2" \
  "${A24_OLMO_BASE_TIME}" \
  "${STANDALONE_SCRIPT}" \
  "${OLMO_BASE_REFERENCE}" \
  "${REPO_ROOT}/manual_runs/20260410/eval_aime24_32k_rawthink_samples.yaml" \
  "${REPO_ROOT}/Eval/output/manual_20260413/aime24_olmo3_7b_think_sft_hf_tp2_avg32_32k_rawthink_samples_r2.json"

submit_and_record \
  "a24-olmo-ckpt32-32k-r2" \
  "${A24_OLMO_CKPT_TIME}" \
  "${STANDALONE_SCRIPT}" \
  "${OLMO_CHECKPOINT_REFERENCE}" \
  "${REPO_ROOT}/manual_runs/20260410/eval_aime24_32k_rawthink_samples.yaml" \
  "${REPO_ROOT}/Eval/output/manual_20260413/aime24_olmo3_7b_checkpoint448_tp2_avg32_32k_rawthink_samples_r2.json"

submit_and_record \
  "a25-olmo-base32-32k-r2" \
  "${A25_OLMO_BASE_TIME}" \
  "${STANDALONE_SCRIPT}" \
  "${OLMO_BASE_REFERENCE}" \
  "${REPO_ROOT}/manual_runs/20260410/eval_aime25_32k_rawthink_samples.yaml" \
  "${REPO_ROOT}/Eval/output/manual_20260413/aime25_olmo3_7b_think_sft_hf_tp2_avg32_32k_rawthink_samples_r2.json"

submit_and_record \
  "a25-olmo-ckpt32-32k-r2" \
  "${A25_OLMO_CKPT_TIME}" \
  "${STANDALONE_SCRIPT}" \
  "${OLMO_CHECKPOINT_REFERENCE}" \
  "${REPO_ROOT}/manual_runs/20260410/eval_aime25_32k_rawthink_samples.yaml" \
  "${REPO_ROOT}/Eval/output/manual_20260413/aime25_olmo3_7b_checkpoint448_tp2_avg32_32k_rawthink_samples_r2.json"

submit_and_record \
  "a25-olmo-rand10-32k-r2" \
  "${A25_OLMO_RAND_TIME}" \
  "${ANALYSIS_SCRIPT}" \
  "${RUN_DIR}/branching_aime25_olmo3_7b_checkpoint448_random_bp010_32k_mc24_rerun.yaml" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

printf 'Submitted rerun job IDs: %s\n' "${submitted_job_ids[*]}"
