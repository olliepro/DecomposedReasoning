#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
ANALYSIS_DIR="${REPO_ROOT}/Analysis"
RUN_DIR="${REPO_ROOT}/manual_runs/20260414"
LOG_DIR="${RUN_DIR}/logs"
SBATCH_SCRIPT="${RUN_DIR}/run_branching_eval_debug_h100_1gpu.sbatch"
STRUCTURED_CONFIG="${ANALYSIS_DIR}/branching_eval/olmo3_checkpoint448_aime25_docs6_12_structured_only_32k_n48_mc24_grid.yaml"
EPS_CONFIG="${ANALYSIS_DIR}/branching_eval/olmo3_checkpoint448_aime25_docs6_12_eps_only_32k_eps033_n48_mc24_grid.yaml"

mkdir -p "${LOG_DIR}"

validate_config() {
  local config_path="$1"
  (
    cd "${ANALYSIS_DIR}"
    uv run python -c '
from pathlib import Path
import sys
from branching_eval.config_types import load_branching_eval_config

config_path = Path(sys.argv[1])
config = load_branching_eval_config(config_path=config_path)
assert config.decoding.max_gen_toks == 32768, config
assert config.branching.max_concurrent_branches == 24, config
assert config.run_matrix.baseline_rollouts == 48, config
print(config_path)
' "${config_path}"
  )
}

submit_checked_job() {
  local dependency_arg="$1"
  local job_name="$2"
  shift 2
  local cmd=(sbatch --parsable)
  if [[ -n "${dependency_arg}" ]]; then
    cmd+=("${dependency_arg}")
  fi
  cmd+=(--job-name "${job_name}" "$@")
  sbatch --test-only \
    ${dependency_arg:+${dependency_arg}} \
    --job-name "${job_name}" \
    "$@" >/dev/null
  "${cmd[@]}"
}

join_job_ids() {
  local result=""
  local job_id
  for job_id in "$@"; do
    if [[ -z "${result}" ]]; then
      result="${job_id}"
    else
      result="${result}:${job_id}"
    fi
  done
  printf '%s' "${result}"
}

submit_wave_job() {
  local dependency_arg="$1"
  local wave_index="$2"
  local strategy="$3"
  local doc_id="$4"
  local seed_value="$5"
  local base_port="$6"
  local config_path
  if [[ "${strategy}" == "structured" ]]; then
    config_path="${STRUCTURED_CONFIG}"
  else
    config_path="${EPS_CONFIG}"
  fi
  local job_name="a25-olmo-${strategy}-d${doc_id}-w${wave_index}-s${seed_value}"
  submit_checked_job \
    "${dependency_arg}" \
    "${job_name}" \
    "${SBATCH_SCRIPT}" \
    "${config_path}" \
    "${doc_id}" \
    "${seed_value}" \
    "${base_port}"
}

validate_config "${STRUCTURED_CONFIG}"
validate_config "${EPS_CONFIG}"

declare -a all_job_ids=()
declare -a seeds=(1234 1235 1236 1237)
declare -A previous_job_id_by_combo=()

submit_combo_job() {
  local wave_index="$1"
  local strategy="$2"
  local doc_id="$3"
  local seed_value="$4"
  local base_port="$5"
  local combo_key="${strategy}_d${doc_id}"
  local dependency_arg=""
  local previous_job_id="${previous_job_id_by_combo[${combo_key}]:-}"
  if [[ -n "${previous_job_id}" ]]; then
    dependency_arg="--dependency=afterany:${previous_job_id}"
  fi
  local job_id
  job_id="$(
    submit_wave_job \
      "${dependency_arg}" \
      "${wave_index}" \
      "${strategy}" \
      "${doc_id}" \
      "${seed_value}" \
      "${base_port}"
  )"
  previous_job_id_by_combo["${combo_key}"]="${job_id}"
  printf '%s' "${job_id}"
}

for wave_index in 0 1 2 3; do
  seed_value="${seeds[wave_index]}"
  echo "Submitting wave $((wave_index + 1)) with seed ${seed_value}"
  structured_doc6_job_id="$(
    submit_combo_job \
      "$((wave_index + 1))" \
      "structured" \
      "6" \
      "${seed_value}" \
      "8120"
  )"
  structured_doc12_job_id="$(
    submit_combo_job \
      "$((wave_index + 1))" \
      "structured" \
      "12" \
      "${seed_value}" \
      "8130"
  )"
  eps_doc6_job_id="$(
    submit_combo_job \
      "$((wave_index + 1))" \
      "eps" \
      "6" \
      "${seed_value}" \
      "8140"
  )"
  eps_doc12_job_id="$(
    submit_combo_job \
      "$((wave_index + 1))" \
      "eps" \
      "12" \
      "${seed_value}" \
      "8150"
  )"
  all_job_ids+=(
    "${structured_doc6_job_id}"
    "${structured_doc12_job_id}"
    "${eps_doc6_job_id}"
    "${eps_doc12_job_id}"
  )
  printf 'Wave %d job IDs: %s\n' \
    "$((wave_index + 1))" \
    "${structured_doc6_job_id} ${structured_doc12_job_id} ${eps_doc6_job_id} ${eps_doc12_job_id}"
done

printf 'Submitted all job IDs: %s\n' "${all_job_ids[*]}"
