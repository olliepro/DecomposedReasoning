#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLTRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SBATCH_SCRIPT="${RLTRAINING_DIR}/slurm/branching_dapo_qwen35_smoke.sbatch"
MODES=(${SMOKE_ROLLOUT_MODES:-branching no_branching structured_baseline epsilon_greedy})

PARTITION="${PARTITION:-preemptible-quad}"
GPU_COUNT="${GPU_COUNT:-4}"
GPU_TYPE="${GPU_TYPE:-a100}"
CPU_COUNT="${CPU_COUNT:-48}"
MEMORY="${MEMORY:-256G}"
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"
RUN_LABEL="${RUN_LABEL:-smoke}"
EXCLUSIVE="${EXCLUSIVE:-false}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"
TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-${GPU_COUNT}}"
TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-${GPU_COUNT}}"
TRAIN_PROMPT_MICRO_BSZ_PER_GPU="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU:-1}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_STEER_TOKENS="${MAX_STEER_TOKENS:-15}"
EXEC_TEMPERATURE="${EXEC_TEMPERATURE:-0.7}"
STEER_TEMPERATURE="${STEER_TEMPERATURE:-1.0}"
EXEC_TOP_P="${EXEC_TOP_P:-0.95}"
STEER_TOP_P="${STEER_TOP_P:-0.95}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
STEER_REPETITION_PENALTY="${STEER_REPETITION_PENALTY:-${REPETITION_PENALTY}}"
ROLLOUT_TOP_LOGPROBS="${ROLLOUT_TOP_LOGPROBS:-1}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
SAVE_FREQ="${SAVE_FREQ:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-524288}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4096}"
ROLLOUT_GDN_PREFILL_BACKEND="${ROLLOUT_GDN_PREFILL_BACKEND:-}"
ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
ACTOR_USE_REMOVE_PADDING="${ACTOR_USE_REMOVE_PADDING:-true}"
ACTOR_USE_FUSED_KERNELS="${ACTOR_USE_FUSED_KERNELS:-true}"
ACTOR_FUSED_KERNEL_BACKEND="${ACTOR_FUSED_KERNEL_BACKEND:-triton}"
ACTOR_ATTN_IMPLEMENTATION="${ACTOR_ATTN_IMPLEMENTATION:-sdpa}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"console\",\"wandb\"]}"
PROJECT_NAME="${PROJECT_NAME:-branching_dapo_qwen35_smoke}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-}"
TRAIN_FILE="${TRAIN_FILE:-}"
VAL_FILE="${VAL_FILE:-}"
REWARD_REQUIRE_STEER_EXEC="${REWARD_REQUIRE_STEER_EXEC:-True}"
BRANCHING_ALPHA="${BRANCHING_ALPHA:-0.5}"
TOPOLOGY_POLICY="${TOPOLOGY_POLICY:-auto}"
HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE_ROOT:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/Analysis/cache/datasets/submissions}"
SUBMISSION_ID="${SUBMISSION_ID:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ -n "${GPU_GRES:-}" ]]; then
  gpu_gres="${GPU_GRES}"
elif [[ -n "${GPU_TYPE}" ]]; then
  gpu_gres="gpu:${GPU_TYPE}:${GPU_COUNT}"
else
  gpu_gres="gpu:${GPU_COUNT}"
fi

for mode in "${MODES[@]}"; do
  mode_slug="${mode//_/-}"
  selector_mode="${SELECTOR_MODE:-random}"
  branch_prob="${BRANCH_PROB:-0.0}"
  branch_fanout="${BRANCH_FANOUT:-2}"
  max_branch_points="${MAX_BRANCH_POINTS_PER_ROLLOUT:-1}"
  num_candidates="${NUM_CANDIDATES:-4}"
  epsilon_greedy_prob="${EPSILON_GREEDY_PROB:-0.0}"
  n_resp_per_prompt="${N_RESP_PER_PROMPT:-16}"
  algorithm_adv_estimator="${ALGORITHM_ADV_ESTIMATOR:-grpo}"
  rollout_gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  hf_datasets_cache="${HF_DATASETS_CACHE:-${HF_DATASETS_CACHE_ROOT}/${SUBMISSION_ID}/${mode}}"

  case "${mode}" in
    branching)
      n_resp_per_prompt="${N_RESP_PER_PROMPT:-1}"
      algorithm_adv_estimator="${ALGORITHM_ADV_ESTIMATOR:-branch_interpolated_grpo}"
      selector_mode="${BRANCHING_SELECTOR_MODE:-${SELECTOR_MODE:-cluster_across}}"
      branch_prob="${BRANCHING_BRANCH_PROB:-${BRANCH_PROB:-0.10}}"
      branch_fanout="${BRANCHING_BRANCH_FANOUT:-${BRANCH_FANOUT:-2}}"
      max_branch_points="${BRANCHING_MAX_BRANCH_POINTS_PER_ROLLOUT:-${MAX_BRANCH_POINTS_PER_ROLLOUT:-4}}"
      num_candidates="${BRANCHING_NUM_CANDIDATES:-${NUM_CANDIDATES:-50}}"
      epsilon_greedy_prob="${BRANCHING_EPSILON_GREEDY_PROB:-${EPSILON_GREEDY_PROB:-0.1}}"
      rollout_gpu_memory_utilization="${BRANCHING_ROLLOUT_GPU_MEMORY_UTILIZATION:-${ROLLOUT_GPU_MEMORY_UTILIZATION}}"
      ;;
    baseline|no_branching|structured_baseline)
      n_resp_per_prompt="${N_RESP_PER_PROMPT:-16}"
      algorithm_adv_estimator="${ALGORITHM_ADV_ESTIMATOR:-grpo}"
      selector_mode="${BASELINE_SELECTOR_MODE:-random}"
      branch_prob=0.0
      epsilon_greedy_prob=0.0
      rollout_gpu_memory_utilization="${BASELINE_ROLLOUT_GPU_MEMORY_UTILIZATION:-${ROLLOUT_GPU_MEMORY_UTILIZATION}}"
      ;;
    epsilon_greedy)
      n_resp_per_prompt="${N_RESP_PER_PROMPT:-16}"
      algorithm_adv_estimator="${ALGORITHM_ADV_ESTIMATOR:-grpo}"
      selector_mode="${EPSILON_SELECTOR_MODE:-embed_diverse_topk_random}"
      branch_prob=0.0
      branch_fanout="${EPSILON_BRANCH_FANOUT:-1}"
      max_branch_points="${EPSILON_MAX_BRANCH_POINTS_PER_ROLLOUT:-${MAX_BRANCH_POINTS_PER_ROLLOUT:-1}}"
      num_candidates="${EPSILON_NUM_CANDIDATES:-${NUM_CANDIDATES:-4}}"
      epsilon_greedy_prob="${EPSILON_GREEDY_PROB:-0.1}"
      rollout_gpu_memory_utilization="${EPSILON_ROLLOUT_GPU_MEMORY_UTILIZATION:-${ROLLOUT_GPU_MEMORY_UTILIZATION}}"
      ;;
    *)
      echo "Unsupported mode: ${mode}" >&2
      exit 2
      ;;
  esac

  job_id="$(
    sbatch_args=()
    if [[ "${EXCLUSIVE}" == "true" ]]; then
      sbatch_args+=(--exclusive)
    fi
    if [[ -n "${EXCLUDE_NODES}" ]]; then
      sbatch_args+=(--exclude="${EXCLUDE_NODES}")
    fi
    sbatch_env=(
      env
      SMOKE_ROLLOUT_MODE="${mode}"
      SMOKE_RUN_LABEL="${RUN_LABEL}"
      NGPUS_PER_NODE="${GPU_COUNT}"
      GEN_TP="${GPU_COUNT}"
      FSDP_SIZE="${GPU_COUNT}"
      RAY_NUM_CPUS="${CPU_COUNT}"
      TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ}"
      TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ}"
      TRAIN_PROMPT_MICRO_BSZ_PER_GPU="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU}"
      N_RESP_PER_PROMPT="${n_resp_per_prompt}"
      MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH}"
      MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH}"
      MAX_MODEL_LEN="${MAX_MODEL_LEN}"
      MAX_STEER_TOKENS="${MAX_STEER_TOKENS}"
      EXEC_TEMPERATURE="${EXEC_TEMPERATURE}"
      STEER_TEMPERATURE="${STEER_TEMPERATURE}"
      EXEC_TOP_P="${EXEC_TOP_P}"
      STEER_TOP_P="${STEER_TOP_P}"
      PRESENCE_PENALTY="${PRESENCE_PENALTY}"
      REPETITION_PENALTY="${REPETITION_PENALTY}"
      STEER_REPETITION_PENALTY="${STEER_REPETITION_PENALTY}"
      ROLLOUT_TOP_LOGPROBS="${ROLLOUT_TOP_LOGPROBS}"
      CLIP_RATIO_LOW="${CLIP_RATIO_LOW}"
      CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH}"
      ACTOR_LR="${ACTOR_LR}"
      TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS}"
      SAVE_FREQ="${SAVE_FREQ}"
      ROLLOUT_GPU_MEMORY_UTILIZATION="${rollout_gpu_memory_utilization}"
      ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
      ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS}"
      ROLLOUT_GDN_PREFILL_BACKEND="${ROLLOUT_GDN_PREFILL_BACKEND}"
      ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE}"
      ACTOR_USE_REMOVE_PADDING="${ACTOR_USE_REMOVE_PADDING}"
      ACTOR_USE_FUSED_KERNELS="${ACTOR_USE_FUSED_KERNELS}"
      ACTOR_FUSED_KERNEL_BACKEND="${ACTOR_FUSED_KERNEL_BACKEND}"
      ACTOR_ATTN_IMPLEMENTATION="${ACTOR_ATTN_IMPLEMENTATION}"
      TRAINER_LOGGERS="${TRAINER_LOGGERS}"
      PROJECT_NAME="${PROJECT_NAME}"
      EXPERIMENT_NAME="${EXPERIMENT_NAME}"
      HF_DATASETS_CACHE="${hf_datasets_cache}"
      ALGORITHM_ADV_ESTIMATOR="${algorithm_adv_estimator}"
      TRAIN_FILE="${TRAIN_FILE}"
      VAL_FILE="${VAL_FILE}"
      REWARD_REQUIRE_STEER_EXEC="${REWARD_REQUIRE_STEER_EXEC}"
      BRANCHING_ALPHA="${BRANCHING_ALPHA}"
      TOPOLOGY_POLICY="${TOPOLOGY_POLICY}"
      SELECTOR_MODE="${selector_mode}"
      BRANCH_PROB="${branch_prob}"
      BRANCH_FANOUT="${branch_fanout}"
      MAX_BRANCH_POINTS_PER_ROLLOUT="${max_branch_points}"
      NUM_CANDIDATES="${num_candidates}"
      EPSILON_GREEDY_PROB="${epsilon_greedy_prob}"
    )

    "${sbatch_env[@]}" sbatch --parsable \
      --partition="${PARTITION}" \
      --gres="${gpu_gres}" \
      --cpus-per-task="${CPU_COUNT}" \
      --mem="${MEMORY}" \
      --time="${TIME_LIMIT}" \
      --job-name="q35rl-${mode_slug}" \
      "${sbatch_args[@]}" \
      --export=ALL \
      "${SBATCH_SCRIPT}"
  )"
  echo "${mode}: ${job_id}"
done
