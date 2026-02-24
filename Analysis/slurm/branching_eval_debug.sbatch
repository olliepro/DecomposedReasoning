#!/usr/bin/env bash
#SBATCH --job-name=branching_eval_debug
#SBATCH --account=PAS2138
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
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

CMD=(uv run python run_branching_lm_eval.py --config "${BRANCH_CONFIG}")
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
