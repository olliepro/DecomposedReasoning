#!/usr/bin/env bash
#SBATCH --job-name=branching_eval_debug
#SBATCH --account=PAS2138
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=01:00:00
#SBATCH --output=/users/PAA0201/ollieproudman/work/DecomposedReasoning/Analysis/branching_eval/output/slurm_logs/branching_eval_debug_%j.out
#SBATCH --error=/users/PAA0201/ollieproudman/work/DecomposedReasoning/Analysis/branching_eval/output/slurm_logs/branching_eval_debug_%j.err

set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
ANALYSIS_DIR="${REPO_ROOT}/Analysis"
LOG_DIR="${ANALYSIS_DIR}/branching_eval/output/slurm_logs"

mkdir -p "${LOG_DIR}"
cd "${ANALYSIS_DIR}"

BRANCH_CONFIG="${BRANCH_CONFIG:-branching_eval/debug_aime24_1gpu_sft_steer.yaml}"
BRANCH_LIMIT="${BRANCH_LIMIT:-}"
BRANCH_DOC_ID="${BRANCH_DOC_ID:-}"
RESUME_RUN_DIR="${RESUME_RUN_DIR:-}"
BRANCH_FANOUT_OVERRIDE="${BRANCH_FANOUT_OVERRIDE:-}"

resolved_config_path="${BRANCH_CONFIG}"
temp_config_path=""

detect_gpu_count() {
  if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
    local slurm_gpu_value
    slurm_gpu_value="$(printf "%s" "${SLURM_GPUS_ON_NODE}" | tr -d ' ')"
    slurm_gpu_value="${slurm_gpu_value%%\(*}"
    if [[ "${slurm_gpu_value}" =~ ^[0-9]+$ ]]; then
      printf "%s" "${slurm_gpu_value}"
      return
    fi
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    IFS=',' read -r -a cuda_devices <<< "${CUDA_VISIBLE_DEVICES}"
    printf "%s" "${#cuda_devices[@]}"
    return
  fi
  printf "0"
}

allocated_gpu_count="$(detect_gpu_count)"
tp_override=""
if [[ "${allocated_gpu_count}" =~ ^[0-9]+$ ]]; then
  if (( allocated_gpu_count >= 4 )); then
    tp_override="4"
  elif (( allocated_gpu_count >= 2 )); then
    tp_override="2"
  fi
fi

if [[ -n "${tp_override}" || -n "${BRANCH_FANOUT_OVERRIDE}" ]]; then
  temp_config_path="$(mktemp "${ANALYSIS_DIR}/branching_eval/tmp_runtime_config_XXXXXX.yaml")"
  ANALYSIS_DIR="${ANALYSIS_DIR}" \
  BRANCH_CONFIG="${BRANCH_CONFIG}" \
  TP_OVERRIDE="${tp_override}" \
  BRANCH_FANOUT_OVERRIDE="${BRANCH_FANOUT_OVERRIDE}" \
  TEMP_CONFIG_PATH="${temp_config_path}" \
  python -c '
from pathlib import Path
import os
import yaml

source_path = Path(os.environ["BRANCH_CONFIG"])
resolved_source_path = source_path if source_path.is_absolute() else Path(os.environ["ANALYSIS_DIR"]) / source_path
payload = yaml.safe_load(resolved_source_path.read_text(encoding="utf-8")) or {}
assert isinstance(payload, dict), "config root must be a mapping"
serve_block = payload.setdefault("serve", {})
assert isinstance(serve_block, dict), "serve block must be a mapping"
tp_override = os.environ.get("TP_OVERRIDE", "")
if tp_override:
    serve_block["tensor_parallel_size"] = int(tp_override)
branch_fanout_override = os.environ.get("BRANCH_FANOUT_OVERRIDE", "")
if branch_fanout_override:
    branching_block = payload.setdefault("branching", {})
    assert isinstance(branching_block, dict), "branching block must be a mapping"
    branching_block["branch_fanout"] = int(branch_fanout_override)
Path(os.environ["TEMP_CONFIG_PATH"]).write_text(
    yaml.safe_dump(payload, sort_keys=False),
    encoding="utf-8",
)
'
  resolved_config_path="${temp_config_path}"
fi

cleanup_temp_config() {
  if [[ -n "${temp_config_path}" && -f "${temp_config_path}" ]]; then
    rm -f "${temp_config_path}"
  fi
}
trap cleanup_temp_config EXIT

CMD=(uv run python run_branching_lm_eval.py --config "${resolved_config_path}")
if [[ -n "${BRANCH_LIMIT}" ]]; then
  CMD+=(--limit "${BRANCH_LIMIT}")
fi
if [[ -n "${BRANCH_DOC_ID}" ]]; then
  CMD+=(--doc-id "${BRANCH_DOC_ID}")
fi
if [[ -n "${RESUME_RUN_DIR}" ]]; then
  CMD+=(--resume-run-dir "${RESUME_RUN_DIR}")
fi

echo "[branching_eval_debug] host=$(hostname)"
echo "[branching_eval_debug] cwd=${ANALYSIS_DIR}"
echo "[branching_eval_debug] cmd=${CMD[*]}"
"${CMD[@]}"
