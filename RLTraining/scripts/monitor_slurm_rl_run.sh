#!/usr/bin/env bash
set -euo pipefail

JOB_ID="${1:-${JOB_ID:-}}"
if [[ -z "${JOB_ID}" ]]; then
  echo "Usage: $0 <job-id>" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLTRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOG_DIR="${MONITOR_LOG_DIR:-${RLTRAINING_DIR}/logs}"
MONITOR_LOG="${MONITOR_LOG:-${LOG_DIR}/monitor_${JOB_ID}.log}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-300}"
MAX_HOURS="${MAX_HOURS:-30}"
JOB_LOG_PREFIX="${JOB_LOG_PREFIX:-q35rl-branching}"

mkdir -p "${LOG_DIR}"

end_epoch="$(($(date +%s) + MAX_HOURS * 3600))"
out_file="${LOG_DIR}/${JOB_LOG_PREFIX}-${JOB_ID}.out"
err_file="${LOG_DIR}/${JOB_LOG_PREFIX}-${JOB_ID}.err"

append_latest_metrics() {
  if [[ ! -f "${out_file}" ]]; then
    echo "trainer_metrics: no stdout log yet"
    return
  fi
  local latest_step
  latest_step="$(rg "training/global_step" "${out_file}" | tail -n 1 || true)"
  if [[ -z "${latest_step}" ]]; then
    echo "trainer_metrics: no trainer step line yet"
    return
  fi
  echo "trainer_metrics: $(summarize_metric_line "${latest_step}")"
}

metric_value() {
  local metric_line="$1"
  local metric_key="$2"
  printf '%s\n' "${metric_line}" \
    | sed -E 's/\x1B\[[0-9;]*[mK]//g' \
    | awk -v key="${metric_key}" 'BEGIN { RS = " - "; FS = ":" } $1 == key { print $2; exit }'
}

summarize_metric_line() {
  local metric_line="$1"
  local metric_keys=(
    "training/global_step"
    "critic/score/mean"
    "critic/rewards/mean"
    "critic/advantages/mean"
    "critic/advantages/max"
    "response_length/clip_ratio"
    "response/aborted_ratio"
    "branching/realized_leaf_count"
    "branching/realization_rate"
    "branching/adv/final_abs_mean"
    "timing_s/gen"
    "timing_s/old_log_prob"
    "timing_s/update_actor"
    "timing_s/step"
    "perf/throughput"
    "perf/total_num_tokens"
  )
  local parts=()
  local metric_key
  for metric_key in "${metric_keys[@]}"; do
    local value
    value="$(metric_value "${metric_line}" "${metric_key}")"
    if [[ -n "${value}" ]]; then
      parts+=("${metric_key}=${value}")
    fi
  done
  if [[ "${#parts[@]}" -eq 0 ]]; then
    echo "${metric_line}"
    return
  fi
  local IFS=' '
  echo "${parts[*]}"
}

append_recent_errors() {
  if [[ ! -f "${err_file}" ]]; then
    echo "recent_errors: no stderr log yet"
    return
  fi
  local recent
  recent="$(rg -n "Traceback|Error|CUDA out of memory|out of memory|RuntimeError|Exception" "${err_file}" | tail -n 5 || true)"
  if [[ -z "${recent}" ]]; then
    echo "recent_errors: none"
    return
  fi
  echo "recent_errors:"
  echo "${recent}"
}

append_wandb_status() {
  if [[ ! -f "${err_file}" ]]; then
    echo "wandb: no stderr log yet"
    return
  fi
  local wandb_line
  wandb_line="$(rg -n "wandb: Run data is saved locally|wandb: Tracking run" "${err_file}" | tail -n 1 || true)"
  if [[ -z "${wandb_line}" ]]; then
    echo "wandb: no run line yet"
    return
  fi
  echo "wandb: ${wandb_line}"
}

job_terminal_state() {
  local state
  state="$(sacct -j "${JOB_ID}" --format=State -n -P 2>/dev/null | head -n 1 | cut -d'|' -f1 | tr -d ' ' || true)"
  case "${state}" in
    COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|PREEMPTED)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

while (( $(date +%s) <= end_epoch )); do
  {
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') job ${JOB_ID} ====="
    squeue -j "${JOB_ID}" -o '%.18i %.9P %.24j %.8T %.12M %.12l %.6D %.20S %.50R' || true
    sacct -j "${JOB_ID}" --format=JobID,JobName%24,State,Elapsed,ExitCode,NodeList%20,Start,End -P || true
    append_wandb_status
    append_latest_metrics
    append_recent_errors
    echo
  } >>"${MONITOR_LOG}" 2>&1

  if [[ -z "$(squeue -j "${JOB_ID}" -h 2>/dev/null)" ]] && job_terminal_state; then
    break
  fi
  sleep "${INTERVAL_SECONDS}"
done
