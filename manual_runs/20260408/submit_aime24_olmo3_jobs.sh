#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
RUN_DIR="${REPO_ROOT}/manual_runs/20260408"
STANDALONE_SCRIPT="${REPO_ROOT}/manual_runs/20260406/run_standalone_aime_2gpu.sbatch"
ANALYSIS_SCRIPT="${REPO_ROOT}/manual_runs/20260406/run_analysis_aime_2gpu.sbatch"
STANDALONE_CONFIG="${RUN_DIR}/eval_aime24_olmo3_7b_16k_samples.yaml"
STRUCTURED_CONFIG="${RUN_DIR}/structured_aime24_olmo3_7b_checkpoint448_n32.yaml"
BRANCH_RANDOM_CONFIG="${RUN_DIR}/branching_aime24_olmo3_7b_checkpoint448_random_bp014.yaml"
BRANCH_CLUSTER_CONFIG="${RUN_DIR}/branching_aime24_olmo3_7b_checkpoint448_cluster_across_bp014.yaml"

BASE_MODEL_REFERENCE="allenai/Olmo-3-7B-Think-SFT"
CHECKPOINT_REFERENCE="/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_think_sft_to_think_merged_2213/checkpoint-448"
MODEL_ID="sft"
SEED_VALUE="1234"

BASE_OUTPUT="${REPO_ROOT}/Eval/output/manual_20260408/aime24_olmo3_7b_think_sft_hf_tp2_avg32_16k_samples.json"
CHECKPOINT_OUTPUT="${REPO_ROOT}/Eval/output/manual_20260408/aime24_olmo3_7b_checkpoint448_tp2_avg32_16k_samples.json"

CPUS_PER_TASK="32"
MEMORY_PER_NODE="256G"
BASE_STANDALONE_TIME="02:30:00"
CHECKPOINT_STANDALONE_TIME="03:00:00"
BRANCH_RANDOM_TIME="02:30:00"
BRANCH_CLUSTER_TIME="03:30:00"
STRUCTURED_TIME="03:30:00"

mkdir -p "${RUN_DIR}/logs"

echo "Test-only: base standalone samples"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${BASE_STANDALONE_TIME}" \
  --job-name aime24-olmo3-base-samp \
  "${STANDALONE_SCRIPT}" \
  "${BASE_MODEL_REFERENCE}" \
  "${STANDALONE_CONFIG}" \
  "${BASE_OUTPUT}"

echo "Test-only: checkpoint standalone samples"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${CHECKPOINT_STANDALONE_TIME}" \
  --job-name aime24-olmo3-ckpt-samp \
  "${STANDALONE_SCRIPT}" \
  "${CHECKPOINT_REFERENCE}" \
  "${STANDALONE_CONFIG}" \
  "${CHECKPOINT_OUTPUT}"

echo "Test-only: checkpoint branching random bp014"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${BRANCH_RANDOM_TIME}" \
  --job-name aime24-olmo3-rand014 \
  "${ANALYSIS_SCRIPT}" \
  "${BRANCH_RANDOM_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

echo "Test-only: checkpoint branching cluster_across bp014"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${BRANCH_CLUSTER_TIME}" \
  --job-name aime24-olmo3-clust014 \
  "${ANALYSIS_SCRIPT}" \
  "${BRANCH_CLUSTER_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

echo "Test-only: checkpoint structured-baseline n32"
sbatch --test-only \
  --cpus-per-task "${CPUS_PER_TASK}" \
  --mem "${MEMORY_PER_NODE}" \
  --time "${STRUCTURED_TIME}" \
  --job-name aime24-olmo3-struct32 \
  "${ANALYSIS_SCRIPT}" \
  "${STRUCTURED_CONFIG}" \
  "${MODEL_ID}" \
  "${SEED_VALUE}"

echo "Submitting full: base standalone samples"
BASE_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${BASE_STANDALONE_TIME}" \
    --job-name aime24-olmo3-base-samp \
    "${STANDALONE_SCRIPT}" \
    "${BASE_MODEL_REFERENCE}" \
    "${STANDALONE_CONFIG}" \
    "${BASE_OUTPUT}"
)"
echo "BASE_JOB_ID=${BASE_JOB_ID}"

echo "Submitting full: checkpoint standalone samples"
CHECKPOINT_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${CHECKPOINT_STANDALONE_TIME}" \
    --job-name aime24-olmo3-ckpt-samp \
    "${STANDALONE_SCRIPT}" \
    "${CHECKPOINT_REFERENCE}" \
    "${STANDALONE_CONFIG}" \
    "${CHECKPOINT_OUTPUT}"
)"
echo "CHECKPOINT_JOB_ID=${CHECKPOINT_JOB_ID}"

echo "Submitting full: checkpoint branching random bp014"
BRANCH_RANDOM_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${BRANCH_RANDOM_TIME}" \
    --job-name aime24-olmo3-rand014 \
    "${ANALYSIS_SCRIPT}" \
    "${BRANCH_RANDOM_CONFIG}" \
    "${MODEL_ID}" \
    "${SEED_VALUE}"
)"
echo "BRANCH_RANDOM_JOB_ID=${BRANCH_RANDOM_JOB_ID}"

echo "Submitting full: checkpoint branching cluster_across bp014"
BRANCH_CLUSTER_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${BRANCH_CLUSTER_TIME}" \
    --job-name aime24-olmo3-clust014 \
    "${ANALYSIS_SCRIPT}" \
    "${BRANCH_CLUSTER_CONFIG}" \
    "${MODEL_ID}" \
    "${SEED_VALUE}"
)"
echo "BRANCH_CLUSTER_JOB_ID=${BRANCH_CLUSTER_JOB_ID}"

echo "Submitting full: checkpoint structured-baseline n32"
STRUCTURED_JOB_ID="$(
  sbatch --parsable \
    --cpus-per-task "${CPUS_PER_TASK}" \
    --mem "${MEMORY_PER_NODE}" \
    --time "${STRUCTURED_TIME}" \
    --job-name aime24-olmo3-struct32 \
    "${ANALYSIS_SCRIPT}" \
    "${STRUCTURED_CONFIG}" \
    "${MODEL_ID}" \
    "${SEED_VALUE}"
)"
echo "STRUCTURED_JOB_ID=${STRUCTURED_JOB_ID}"
