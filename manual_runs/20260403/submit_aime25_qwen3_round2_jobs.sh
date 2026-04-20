#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
RUN_DIR="${REPO_ROOT}/manual_runs/20260403"
STANDALONE_SCRIPT="${RUN_DIR}/run_standalone_aime25_2gpu.sbatch"
ANALYSIS_SCRIPT="${RUN_DIR}/run_analysis_aime25_2gpu.sbatch"
STANDALONE_CONFIG="${RUN_DIR}/eval_aime25_qwen3_8b_16k_samples.yaml"
BRANCH_RANDOM_CONFIG="${RUN_DIR}/branching_aime25_qwen3_8b_checkpoint456_random_bp014.yaml"
BRANCH_RANDOM_SMOKE_CONFIG="${RUN_DIR}/branching_aime25_qwen3_8b_checkpoint456_random_bp014_smoke.yaml"
BRANCH_CLUSTER_CONFIG="${RUN_DIR}/branching_aime25_qwen3_8b_checkpoint456_cluster_across_bp014.yaml"
BRANCH_CLUSTER_SMOKE_CONFIG="${RUN_DIR}/branching_aime25_qwen3_8b_checkpoint456_cluster_across_bp014_smoke.yaml"
STRUCTURED_CONFIG="${RUN_DIR}/structured_aime25_qwen3_8b_checkpoint456_n32.yaml"
STRUCTURED_SMOKE_CONFIG="${RUN_DIR}/structured_aime25_qwen3_8b_checkpoint456_n32_smoke.yaml"
CHECKPOINT_REFERENCE="${REPO_ROOT}/scratch_checkpoints/qwen3_8b_to_think_merged_2213_zero3/checkpoint-456"

BASE_OUTPUT="${REPO_ROOT}/Eval/output/manual_20260403/aime25_qwen3_8b_hf_tp2_avg32_16k_samples.json"
BASE_SMOKE_OUTPUT="${REPO_ROOT}/Eval/output/manual_20260403/smoke/aime25_qwen3_8b_hf_tp2_avg32_16k_samples_limit1.json"
CHECKPOINT_OUTPUT="${REPO_ROOT}/Eval/output/manual_20260403/aime25_qwen3_8b_checkpoint456_tp2_avg32_16k_samples.json"
CHECKPOINT_SMOKE_OUTPUT="${REPO_ROOT}/Eval/output/manual_20260403/smoke/aime25_qwen3_8b_checkpoint456_tp2_avg32_16k_samples_limit1.json"

MODEL_ID="sft"
SEED_VALUE="1234"
CPUS_PER_TASK="32"
MEMORY_PER_NODE="256G"

SMOKE_STANDALONE_TIME="00:45:00"
SMOKE_RANDOM_TIME="01:00:00"
SMOKE_CLUSTER_TIME="00:45:00"
SMOKE_STRUCTURED_TIME="01:00:00"
STANDALONE_TIME="03:00:00"
BRANCH_RANDOM_TIME="03:00:00"
BRANCH_CLUSTER_TIME="04:00:00"
STRUCTURED_TIME="04:00:00"

mkdir -p "${RUN_DIR}/logs"

echo "Test-only: base standalone samples"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${STANDALONE_TIME}" \
  --job-name aime25-qwen3-base-samp \
  "${STANDALONE_SCRIPT}" \
  "Qwen/Qwen3-8B" \
  "${STANDALONE_CONFIG}" \
  "${BASE_OUTPUT}"

echo "Test-only: checkpoint standalone samples"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${STANDALONE_TIME}" \
  --job-name aime25-qwen3-ckpt-samp \
  "${STANDALONE_SCRIPT}" \
  "${CHECKPOINT_REFERENCE}" \
  "${STANDALONE_CONFIG}" \
  "${CHECKPOINT_OUTPUT}"

echo "Test-only: checkpoint branching random bp014"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${BRANCH_RANDOM_TIME}" \
  --job-name aime25-qwen3-rand014 \
  "${ANALYSIS_SCRIPT}" \
  "${BRANCH_RANDOM_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

echo "Test-only: checkpoint branching cluster_across bp014"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${BRANCH_CLUSTER_TIME}" \
  --job-name aime25-qwen3-clust014 \
  "${ANALYSIS_SCRIPT}" \
  "${BRANCH_CLUSTER_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

echo "Test-only: checkpoint structured-baseline n32"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${STRUCTURED_TIME}" \
  --job-name aime25-qwen3-struct32 \
  "${ANALYSIS_SCRIPT}" \
  "${STRUCTURED_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

echo "Submitting smoke: base standalone samples"
BASE_SMOKE_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${SMOKE_STANDALONE_TIME}" \
    --job-name aime25-qwen3-base-s1 \
    "${STANDALONE_SCRIPT}" \
    "Qwen/Qwen3-8B" \
    "${STANDALONE_CONFIG}" \
    "${BASE_SMOKE_OUTPUT}" \
    --limit 1
)"
echo "BASE_SMOKE_JOB_ID=${BASE_SMOKE_JOB_ID}"

echo "Submitting smoke: checkpoint standalone samples"
CHECKPOINT_SMOKE_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${SMOKE_STANDALONE_TIME}" \
    --job-name aime25-qwen3-ckpt-s1 \
    "${STANDALONE_SCRIPT}" \
    "${CHECKPOINT_REFERENCE}" \
    "${STANDALONE_CONFIG}" \
    "${CHECKPOINT_SMOKE_OUTPUT}" \
    --limit 1
)"
echo "CHECKPOINT_SMOKE_JOB_ID=${CHECKPOINT_SMOKE_JOB_ID}"

echo "Submitting smoke: checkpoint branching random bp014"
BRANCH_RANDOM_SMOKE_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${SMOKE_RANDOM_TIME}" \
    --job-name aime25-qwen3-rand014-s1 \
    "${ANALYSIS_SCRIPT}" \
    "${BRANCH_RANDOM_SMOKE_CONFIG}" \
    "${MODEL_ID}" \
    "${SEED_VALUE}" \
    1
)"
echo "BRANCH_RANDOM_SMOKE_JOB_ID=${BRANCH_RANDOM_SMOKE_JOB_ID}"

echo "Submitting smoke: checkpoint branching cluster_across bp014"
BRANCH_CLUSTER_SMOKE_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${SMOKE_CLUSTER_TIME}" \
    --job-name aime25-qwen3-clust014-s1 \
    "${ANALYSIS_SCRIPT}" \
    "${BRANCH_CLUSTER_SMOKE_CONFIG}" \
    "${MODEL_ID}" \
    "${SEED_VALUE}" \
    1
)"
echo "BRANCH_CLUSTER_SMOKE_JOB_ID=${BRANCH_CLUSTER_SMOKE_JOB_ID}"

echo "Submitting smoke: checkpoint structured-baseline n32"
STRUCTURED_SMOKE_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${SMOKE_STRUCTURED_TIME}" \
    --job-name aime25-qwen3-struct32-s1 \
    "${ANALYSIS_SCRIPT}" \
    "${STRUCTURED_SMOKE_CONFIG}" \
    "${MODEL_ID}" \
    "${SEED_VALUE}" \
    1
)"
echo "STRUCTURED_SMOKE_JOB_ID=${STRUCTURED_SMOKE_JOB_ID}"

echo "Submitting full: base standalone samples"
BASE_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${BASE_SMOKE_JOB_ID}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${STANDALONE_TIME}" \
    --job-name aime25-qwen3-base-samp \
    "${STANDALONE_SCRIPT}" \
    "Qwen/Qwen3-8B" \
    "${STANDALONE_CONFIG}" \
    "${BASE_OUTPUT}"
)"
echo "BASE_JOB_ID=${BASE_JOB_ID}"

echo "Submitting full: checkpoint standalone samples"
CHECKPOINT_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${CHECKPOINT_SMOKE_JOB_ID}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${STANDALONE_TIME}" \
    --job-name aime25-qwen3-ckpt-samp \
    "${STANDALONE_SCRIPT}" \
    "${CHECKPOINT_REFERENCE}" \
    "${STANDALONE_CONFIG}" \
    "${CHECKPOINT_OUTPUT}"
)"
echo "CHECKPOINT_JOB_ID=${CHECKPOINT_JOB_ID}"

echo "Submitting full: checkpoint branching random bp014"
BRANCH_RANDOM_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${BRANCH_RANDOM_SMOKE_JOB_ID}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${BRANCH_RANDOM_TIME}" \
    --job-name aime25-qwen3-rand014 \
    "${ANALYSIS_SCRIPT}" \
    "${BRANCH_RANDOM_CONFIG}" \
    "${MODEL_ID}" \
    "${SEED_VALUE}"
)"
echo "BRANCH_RANDOM_JOB_ID=${BRANCH_RANDOM_JOB_ID}"

echo "Submitting full: checkpoint branching cluster_across bp014"
BRANCH_CLUSTER_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${BRANCH_CLUSTER_SMOKE_JOB_ID}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${BRANCH_CLUSTER_TIME}" \
    --job-name aime25-qwen3-clust014 \
    "${ANALYSIS_SCRIPT}" \
    "${BRANCH_CLUSTER_CONFIG}" \
    "${MODEL_ID}" \
    "${SEED_VALUE}"
)"
echo "BRANCH_CLUSTER_JOB_ID=${BRANCH_CLUSTER_JOB_ID}"

echo "Submitting full: checkpoint structured-baseline n32"
STRUCTURED_JOB_ID="$(
  sbatch --parsable \
    --dependency="afterok:${STRUCTURED_SMOKE_JOB_ID}" \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${STRUCTURED_TIME}" \
    --job-name aime25-qwen3-struct32 \
    "${ANALYSIS_SCRIPT}" \
    "${STRUCTURED_CONFIG}" \
    "${MODEL_ID}" \
    "${SEED_VALUE}"
)"
echo "STRUCTURED_JOB_ID=${STRUCTURED_JOB_ID}"
