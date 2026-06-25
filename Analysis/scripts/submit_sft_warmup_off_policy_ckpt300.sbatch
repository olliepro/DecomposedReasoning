#!/usr/bin/env bash
#SBATCH -A PAA0201
#SBATCH -p preemptible-nextgen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --job-name=sft-warmup-offpolicy
#SBATCH --output=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/Analysis/branching_eval/sft_warmup_off_policy/slurm/sft_warmup_offpolicy_%j.out
#SBATCH --error=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/Analysis/branching_eval/sft_warmup_off_policy/slurm/sft_warmup_offpolicy_%j.err

set -euo pipefail

REPO=/users/PAA0201/ollieproudman/work/DecomposedReasoning
RL_REPO="${REPO}/RLTraining"
ANALYSIS_REPO="${REPO}/Analysis"
MODEL=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/qwen35_branching_dapo/served_models/qwen35_4b_branch_gs50_structured_baseline_lr2e6_execT1_5611097_global_step_300_hf
SFT_JSONL="${REPO}/BuildSFTDataset/output_transform_async_16384/transformed_subset_analysis/merged_with_output_transformed_output_aug390_t1p0_pruned_top10_max_exec_truncated_seed42_plus_non_sequitur500_1to2_v2.jsonl"
MODEL_NAME=qwen35-4b-5611097-step300
PORT="${PORT:-18001}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-${VLLM_TENSOR_PARALLEL_SIZE:-1}}"
unset VLLM_TENSOR_PARALLEL_SIZE
BASE_URL="http://127.0.0.1:${PORT}/v1"
OUTPUT_ROOT=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/Analysis/branching_eval/sft_warmup_off_policy
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_LABEL="${RUN_LABEL:-sft_warmup_final_sft2000_ckpt300_offpolicy_p02_nobranch}"
RUN_DIR="${RUN_DIR:-${OUTPUT_ROOT}/${RUN_LABEL}_${SLURM_JOB_ID}_${RUN_STAMP}}"
FINAL_DIR="${FINAL_DIR:-${OUTPUT_ROOT}/${RUN_LABEL}_complete_2000_${SLURM_JOB_ID}_${RUN_STAMP}}"
GEN_LIMIT="${GEN_LIMIT:-2000}"
SOURCE_INDICES_FILE="${SOURCE_INDICES_FILE:-}"
FORCE_INITIAL_STEER_OPEN="${FORCE_INITIAL_STEER_OPEN:-0}"
NON_NVLINK_VLLM="${NON_NVLINK_VLLM:-0}"

export HF_HOME=/fs/scratch/PAA0201/ollieproudman/hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export VLLM_CACHE_ROOT=/fs/scratch/PAA0201/ollieproudman/vllm_cache
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONPATH=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/qwen35_branching_dapo/serve_jobs/torchvision_text_stub:${PYTHONPATH:-}
if [[ "${NON_NVLINK_VLLM}" == "1" ]]; then
  export NCCL_P2P_DISABLE=1
  export VLLM_DISABLE_PYNCCL=1
  export VLLM_SKIP_P2P_CHECK=0
fi

mkdir -p "${OUTPUT_ROOT}/slurm" "${RUN_DIR}" "${RUN_DIR}/logs"

echo "node=$(hostname -f)"
echo "run_dir=${RUN_DIR}"
echo "final_dir=${FINAL_DIR}"
echo "model=${MODEL}"
echo "base_url=${BASE_URL}"
echo "tensor_parallel_size=${TENSOR_PARALLEL_SIZE}"
echo "input_jsonl=${SFT_JSONL}"
echo "source_indices_file=${SOURCE_INDICES_FILE}"
echo "force_initial_steer_open=${FORCE_INITIAL_STEER_OPEN}"
echo "non_nvlink_vllm=${NON_NVLINK_VLLM}"
if [[ "${NON_NVLINK_VLLM}" == "1" ]]; then
  echo "non_nvlink_env=NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE} VLLM_DISABLE_PYNCCL=${VLLM_DISABLE_PYNCCL} VLLM_SKIP_P2P_CHECK=${VLLM_SKIP_P2P_CHECK}"
fi
nvidia-smi

cd "${RL_REPO}"
uv run python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --served-model-name "${MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --dtype bfloat16 \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len 17408 \
  --gpu-memory-utilization 0.85 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 65536 \
  --trust-remote-code \
  --language-model-only \
  --skip-mm-profiling \
  --no-enable-log-requests \
  >"${RUN_DIR}/logs/vllm.out" \
  2>"${RUN_DIR}/logs/vllm.err" &
VLLM_PID=$!

cleanup() {
  if kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    kill "${VLLM_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

for _ in $(seq 1 900); do
  if curl -fsS "${BASE_URL}/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
    echo "vLLM exited before readiness" >&2
    wait "${VLLM_PID}" || true
    exit 1
  fi
  sleep 2
done
curl -fsS "${BASE_URL}/models" >"${RUN_DIR}/served_models.json"

cd "${ANALYSIS_REPO}"
GENERATOR_ARGS=(
  --input-jsonl "${SFT_JSONL}"
  --limit "${GEN_LIMIT}"
  --base-url "${BASE_URL}"
  --model "${MODEL_NAME}"
  --run-dir "${RUN_DIR}"
  --max-concurrent-docs 24
  --epsilon-greedy-prob 0.02
  --off-policy-min-candidates 3
  --off-policy-max-candidates 10
  --max-gen-toks 16384
  --max-model-len 17408
  --decode-chunk-tokens 512
  --temperature 0.7
  --top-p 0.95
  --top-logprobs 5
  --startup-timeout-seconds 5
)
if [[ -n "${SOURCE_INDICES_FILE}" ]]; then
  GENERATOR_ARGS+=(--source-indices-file "${SOURCE_INDICES_FILE}")
fi
if [[ "${FORCE_INITIAL_STEER_OPEN}" == "1" ]]; then
  GENERATOR_ARGS+=(--force-initial-steer-open)
fi
uv run python scripts/generate_sft_warmup_off_policy.py \
  "${GENERATOR_ARGS[@]}"

uv run python scripts/finalize_sft_warmup_off_policy.py \
  --base-run-dir "${RUN_DIR}" \
  --output-dir "${FINAL_DIR}" \
  --target-count 2000 \
  --spotcheck-count 50

echo "completed run_dir=${RUN_DIR}"
echo "completed final_dir=${FINAL_DIR}"
