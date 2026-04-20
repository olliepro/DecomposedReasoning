#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
RUN_DIR="${REPO_ROOT}/manual_runs/20260402"
STANDALONE_SCRIPT="${RUN_DIR}/run_standalone_aime25_2gpu.sbatch"
BRANCHING_SCRIPT="${RUN_DIR}/run_branching_aime25_checkpoint456_2gpu.sbatch"
STANDALONE_CONFIG="${RUN_DIR}/eval_aime25_qwen3_8b_16k.yaml"
BRANCHING_RANDOM_CONFIG="${RUN_DIR}/branching_aime25_qwen3_8b_checkpoint456_random.yaml"
BRANCHING_CLUSTER_CONFIG="${RUN_DIR}/branching_aime25_qwen3_8b_checkpoint456_cluster_across.yaml"
BASE_OUTPUT="${REPO_ROOT}/Eval/output/manual_20260402/aime25_qwen3_8b_hf_tp2_avg32_16k.json"
CHECKPOINT_OUTPUT="${REPO_ROOT}/Eval/output/manual_20260402/aime25_qwen3_8b_checkpoint456_tp2_avg32_16k.json"
CHECKPOINT_REFERENCE="${REPO_ROOT}/scratch_checkpoints/qwen3_8b_to_think_merged_2213_zero3/checkpoint-456"
STANDALONE_TIME="04:00:00"
BRANCHING_RANDOM_TIME="04:00:00"
BRANCHING_CLUSTER_TIME="08:00:00"
CPUS_PER_TASK="32"
MEMORY_PER_NODE="256G"

mkdir -p "${RUN_DIR}/logs"

echo "Test-only: base standalone"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${STANDALONE_TIME}" \
  --job-name aime25-qwen3-8b-base \
  "${STANDALONE_SCRIPT}" \
  "Qwen/Qwen3-8B" \
  "${STANDALONE_CONFIG}" \
  "${BASE_OUTPUT}"

echo "Test-only: checkpoint standalone"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${STANDALONE_TIME}" \
  --job-name aime25-qwen3-8b-ckpt456 \
  "${STANDALONE_SCRIPT}" \
  "${CHECKPOINT_REFERENCE}" \
  "${STANDALONE_CONFIG}" \
  "${CHECKPOINT_OUTPUT}"

echo "Test-only: checkpoint branching random"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${BRANCHING_RANDOM_TIME}" \
  --job-name aime25-qwen3-branch-rand \
  "${BRANCHING_SCRIPT}" \
  "${BRANCHING_RANDOM_CONFIG}"

echo "Test-only: checkpoint branching cluster_across"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${BRANCHING_CLUSTER_TIME}" \
  --job-name aime25-qwen3-branch-clust \
  "${BRANCHING_SCRIPT}" \
  "${BRANCHING_CLUSTER_CONFIG}"

echo "Submitting: base standalone"
BASE_JOB_ID="$(sbatch --parsable --cpus-per-task "${CPUS_PER_TASK}" --mem "${MEMORY_PER_NODE}" --time "${STANDALONE_TIME}" --job-name aime25-qwen3-8b-base "${STANDALONE_SCRIPT}" "Qwen/Qwen3-8B" "${STANDALONE_CONFIG}" "${BASE_OUTPUT}")"
echo "BASE_JOB_ID=${BASE_JOB_ID}"

echo "Submitting: checkpoint standalone"
CHECKPOINT_JOB_ID="$(sbatch --parsable --cpus-per-task "${CPUS_PER_TASK}" --mem "${MEMORY_PER_NODE}" --time "${STANDALONE_TIME}" --job-name aime25-qwen3-8b-ckpt456 "${STANDALONE_SCRIPT}" "${CHECKPOINT_REFERENCE}" "${STANDALONE_CONFIG}" "${CHECKPOINT_OUTPUT}")"
echo "CHECKPOINT_JOB_ID=${CHECKPOINT_JOB_ID}"

echo "Submitting: checkpoint branching random"
BRANCHING_RANDOM_JOB_ID="$(sbatch --parsable --cpus-per-task "${CPUS_PER_TASK}" --mem "${MEMORY_PER_NODE}" --time "${BRANCHING_RANDOM_TIME}" --job-name aime25-qwen3-branch-rand "${BRANCHING_SCRIPT}" "${BRANCHING_RANDOM_CONFIG}")"
echo "BRANCHING_RANDOM_JOB_ID=${BRANCHING_RANDOM_JOB_ID}"

echo "Submitting: checkpoint branching cluster_across"
BRANCHING_CLUSTER_JOB_ID="$(sbatch --parsable --cpus-per-task "${CPUS_PER_TASK}" --mem "${MEMORY_PER_NODE}" --time "${BRANCHING_CLUSTER_TIME}" --job-name aime25-qwen3-branch-clust "${BRANCHING_SCRIPT}" "${BRANCHING_CLUSTER_CONFIG}")"
echo "BRANCHING_CLUSTER_JOB_ID=${BRANCHING_CLUSTER_JOB_ID}"
