#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLTRAINING_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${RLTRAINING_DIR}/.." && pwd)"

BEST_CHECKPOINT_FILE="${BEST_CHECKPOINT_FILE:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_think_sft_to_think_merged_2213/benchmark_evals/best_checkpoint.txt}"
if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_PATH_VALUE="${MODEL_PATH}"
else
  if [[ ! -f "${BEST_CHECKPOINT_FILE}" ]]; then
    echo "Best-checkpoint file not found: ${BEST_CHECKPOINT_FILE}"
    exit 1
  fi
  MODEL_PATH_VALUE="$(<"${BEST_CHECKPOINT_FILE}")"
fi

TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/BuildRLDataset/output/train.parquet}"
VAL_FILE="${VAL_FILE:-${TRAIN_FILE}}"
PROJECT_NAME="${PROJECT_NAME:-branching_dapo}"
ROLLOUT_MODE="${ROLLOUT_MODE:-branching}"
RL_UPDATE_MODE="${RL_UPDATE_MODE:-all}"
SELECTOR_MODE="${SELECTOR_MODE:-cluster_across}"
if [[ -z "${ALGORITHM_ADV_ESTIMATOR:-}" ]]; then
  case "${ROLLOUT_MODE}" in
    branching)
      ALGORITHM_ADV_ESTIMATOR=branch_interpolated_grpo
      ;;
    baseline|no_branching|structured_baseline|epsilon_greedy)
      ALGORITHM_ADV_ESTIMATOR=grpo
      ;;
    *)
      echo "Unsupported ROLLOUT_MODE for adv estimator default: ${ROLLOUT_MODE}" >&2
      exit 2
      ;;
  esac
fi
BRANCHING_ALPHA="${BRANCHING_ALPHA:-0.5}"
EPSILON_GREEDY_PROB="${EPSILON_GREEDY_PROB:-0.05}"
BRANCH_FANOUT="${BRANCH_FANOUT:-4}"
MAX_BRANCH_POINTS_PER_ROLLOUT="${MAX_BRANCH_POINTS_PER_ROLLOUT:-2}"
BRANCH_PROB="${BRANCH_PROB:-0.05}"
NUM_CANDIDATES="${NUM_CANDIDATES:-100}"
MAX_CLUSTERS="${MAX_CLUSTERS:-4}"
MAX_CONCURRENT_BRANCHES="${MAX_CONCURRENT_BRANCHES:-512}"
EXEC_TEMPERATURE="${EXEC_TEMPERATURE:-0.7}"
STEER_TEMPERATURE="${STEER_TEMPERATURE:-1.0}"
EXEC_TOP_P="${EXEC_TOP_P:-0.95}"
STEER_TOP_P="${STEER_TOP_P:-0.95}"
ROLLOUT_TOP_LOGPROBS="${ROLLOUT_TOP_LOGPROBS:-1}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
STEER_REPETITION_PENALTY="${STEER_REPETITION_PENALTY:-${REPETITION_PENALTY}}"
REPETITION_CHECKING_ENABLED="${REPETITION_CHECKING_ENABLED:-True}"
USE_FULL_STOP_STRINGS="${USE_FULL_STOP_STRINGS:-False}"
INITIAL_ASSISTANT_PREFIX="${INITIAL_ASSISTANT_PREFIX:-}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
TRIGGER_STEER_ENABLED="${TRIGGER_STEER_ENABLED:-True}"
REWARD_REQUIRE_STEER_EXEC="${REWARD_REQUIRE_STEER_EXEC:-True}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-32}"
TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
TRAIN_PROMPT_MICRO_BSZ_PER_GPU="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU:-1}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
MAX_STEER_TOKENS="${MAX_STEER_TOKENS:-15}"
GEN_TP="${GEN_TP:-4}"
SP_SIZE="${SP_SIZE:-1}"
OFFLOAD="${OFFLOAD:-true}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"console\",\"wandb\"]}"
SAVE_FREQ="${SAVE_FREQ:-10}"
PERSISTENT_LOG_INTERVAL_STEPS="${PERSISTENT_LOG_INTERVAL_STEPS:-10}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-200}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-olmo3_7b_branching_dapo_${SELECTOR_MODE}_a${BRANCHING_ALPHA}}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning}"
CACHE_ROOT="${CACHE_ROOT:-${SCRATCH_ROOT}/RLTraining/branching_dapo}"
TOPOLOGY_POLICY="${TOPOLOGY_POLICY:-disabled}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-False}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-16}}"
CLEANUP_LOCAL_RAY="${CLEANUP_LOCAL_RAY:-1}"
DEFAULT_PYTHON_BIN="${RLTRAINING_DIR}/.venv/bin/python"
if [[ -n "${RL_PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${RL_PYTHON_BIN}"
elif [[ -x "${DEFAULT_PYTHON_BIN}" ]]; then
  PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="python3"
fi

export HF_HOME="${HF_HOME:-${SCRATCH_ROOT}/Analysis/cache/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${SCRATCH_ROOT}/Analysis/cache/hf_hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HUB_CACHE}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${SCRATCH_ROOT}/Analysis/cache/datasets}"
export HF_ASSETS_CACHE="${HF_ASSETS_CACHE:-${SCRATCH_ROOT}/Analysis/cache/hf_assets}"
export PYTHONPATH="${RLTRAINING_DIR}:${RLTRAINING_DIR}/verl:${REPO_ROOT}/Analysis:${PYTHONPATH:-}"
export VLLM_LOGGING_LEVEL

mkdir -p \
  "${HF_HOME}" \
  "${HF_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_ASSETS_CACHE}"

if ! "${PYTHON_BIN}" -c "import hydra, ray" >/dev/null 2>&1; then
  echo "RL_PYTHON_BIN must point to a Python environment with hydra and ray installed: ${PYTHON_BIN}"
  exit 1
fi

RAY_BIN="$(dirname "${PYTHON_BIN}")/ray"
LOCAL_RAY_MANAGED=0

cleanup_local_ray() {
  if [[ "${LOCAL_RAY_MANAGED}" -ne 1 ]]; then
    return
  fi
  if [[ ! -x "${RAY_BIN}" ]]; then
    return
  fi
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
}

if [[ "${CLEANUP_LOCAL_RAY}" == "1" ]] && [[ -z "${RAY_ADDRESS:-}" ]] && [[ -x "${RAY_BIN}" ]]; then
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
  LOCAL_RAY_MANAGED=1
  trap cleanup_local_ray EXIT
fi

if [[ "${TOPOLOGY_POLICY}" == "auto" ]]; then
  TOPOLOGY_EXPORTS="$(
    "${PYTHON_BIN}" -c "import sys; sys.path.insert(0, '${REPO_ROOT}/Eval'); from eval_runner.topology_env import maybe_set_cross_numa_vllm_env; applied, reason = maybe_set_cross_numa_vllm_env(model_type='vllm'); print('export NCCL_P2P_DISABLE=1 VLLM_DISABLE_PYNCCL=1 VLLM_SKIP_P2P_CHECK=0' if applied else f'# {reason}')"
  )"
  if [[ "${TOPOLOGY_EXPORTS}" == export* ]]; then
    eval "${TOPOLOGY_EXPORTS}"
    echo "Applied cross-NUMA env: ${TOPOLOGY_EXPORTS#export }"
  else
    echo "Cross-NUMA env inactive: ${TOPOLOGY_EXPORTS#\# }"
  fi
fi

RAY_INIT_ARGS=(
  +ray_kwargs.ray_init.include_dashboard="${RAY_INCLUDE_DASHBOARD}"
)

if [[ -n "${RAY_NUM_CPUS}" ]]; then
  RAY_INIT_ARGS+=(
    ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
  )
fi

ROLLOUT_BATCHING_OVERRIDES=()
if [[ "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}" == "none" || "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}" == "null" ]]; then
  ROLLOUT_BATCHING_OVERRIDES+=(
    actor_rollout_ref.rollout.max_num_batched_tokens=null
  )
elif [[ -n "${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  ROLLOUT_BATCHING_OVERRIDES+=(
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  )
fi

HYDRA_OVERRIDES=(
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.prompt_key=prompt
  data.truncation=left
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.train_batch_size="${TRAIN_PROMPT_BSZ}"
  actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.mode=async
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.temperature="${EXEC_TEMPERATURE}"
  actor_rollout_ref.rollout.top_p="${EXEC_TOP_P}"
  actor_rollout_ref.rollout.top_k=-1
  +actor_rollout_ref.rollout.presence_penalty="${PRESENCE_PENALTY}"
  +actor_rollout_ref.rollout.repetition_penalty="${REPETITION_PENALTY}"
  actor_rollout_ref.rollout.disable_log_stats=False
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=branching_dapo.agent_loop_manager.BranchingAgentLoopManager
  +actor_rollout_ref.rollout.custom.branching_dapo.rollout_mode="${ROLLOUT_MODE}"
  +actor_rollout_ref.rollout.custom.branching_dapo.selector_mode="${SELECTOR_MODE}"
  +actor_rollout_ref.rollout.custom.branching_dapo.branch_fanout="${BRANCH_FANOUT}"
  +actor_rollout_ref.rollout.custom.branching_dapo.max_branch_points_per_rollout="${MAX_BRANCH_POINTS_PER_ROLLOUT}"
  +actor_rollout_ref.rollout.custom.branching_dapo.branch_prob="${BRANCH_PROB}"
  +actor_rollout_ref.rollout.custom.branching_dapo.epsilon_greedy_prob="${EPSILON_GREEDY_PROB}"
  +actor_rollout_ref.rollout.custom.branching_dapo.num_candidates="${NUM_CANDIDATES}"
  +actor_rollout_ref.rollout.custom.branching_dapo.off_policy_min_candidates="${OFF_POLICY_MIN_CANDIDATES:-3}"
  +actor_rollout_ref.rollout.custom.branching_dapo.off_policy_max_candidates="${OFF_POLICY_MAX_CANDIDATES:-10}"
  +actor_rollout_ref.rollout.custom.branching_dapo.max_steer_tokens="${MAX_STEER_TOKENS}"
  +actor_rollout_ref.rollout.custom.branching_dapo.max_clusters="${MAX_CLUSTERS}"
  +actor_rollout_ref.rollout.custom.branching_dapo.max_concurrent_branches="${MAX_CONCURRENT_BRANCHES}"
  +actor_rollout_ref.rollout.custom.branching_dapo.steer_temperature="${STEER_TEMPERATURE}"
  +actor_rollout_ref.rollout.custom.branching_dapo.steer_top_p="${STEER_TOP_P}"
  +actor_rollout_ref.rollout.custom.branching_dapo.steer_repetition_penalty="${STEER_REPETITION_PENALTY}"
  +actor_rollout_ref.rollout.custom.branching_dapo.repetition_checking_enabled="${REPETITION_CHECKING_ENABLED}"
  +actor_rollout_ref.rollout.custom.branching_dapo.use_full_stop_strings="${USE_FULL_STOP_STRINGS}"
  +actor_rollout_ref.rollout.custom.branching_dapo.top_logprobs="${ROLLOUT_TOP_LOGPROBS}"
  +actor_rollout_ref.rollout.custom.branching_dapo.trigger_steer_enabled="${TRIGGER_STEER_ENABLED}"
  +actor_rollout_ref.rollout.custom.branching_dapo.update_mode="${RL_UPDATE_MODE}"
  +actor_rollout_ref.rollout.custom.branching_dapo.cache_root="${CACHE_ROOT}"
  +actor_rollout_ref.rollout.custom.branching_dapo.persistent_log_interval_steps="${PERSISTENT_LOG_INTERVAL_STEPS}"
  actor_rollout_ref.model.path="${MODEL_PATH_VALUE}"
  actor_rollout_ref.model.use_remove_padding=True
  +actor_rollout_ref.model.override_config.max_position_embeddings="${MAX_MODEL_LEN}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2))"
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))"
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3))"
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}"
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}"
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}"
  actor_rollout_ref.actor.optim.lr_warmup_steps=10
  actor_rollout_ref.actor.optim.weight_decay=0.1
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_PROMPT_MINI_BSZ}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU}"
  actor_rollout_ref.actor.fsdp_config.param_offload="${OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.actor.entropy_coeff=0.0
  actor_rollout_ref.actor.grad_clip=1.0
  actor_rollout_ref.actor.loss_agg_mode=token-mean
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU}"
  actor_rollout_ref.ref.fsdp_config.param_offload="${OFFLOAD}"
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SP_SIZE}"
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU}"
  reward.reward_manager.name=dapo
  reward.custom_reward_function.path="${RLTRAINING_DIR}/branching_dapo/reward_fn.py"
  reward.custom_reward_function.name=compute_score_branching_dapo
  +reward.custom_reward_function.reward_kwargs.require_steer_exec="${REWARD_REQUIRE_STEER_EXEC}"
  algorithm.adv_estimator="${ALGORITHM_ADV_ESTIMATOR}"
  +algorithm.branching_alpha="${BRANCHING_ALPHA}"
  +algorithm.branching_intra_norm_by_std=True
  +algorithm.branching_epsilon=1e-6
  trainer.logger="${TRAINER_LOGGERS}"
  trainer.project_name="${PROJECT_NAME}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
  trainer.nnodes="${NNODES}"
  trainer.val_before_train=False
  trainer.test_freq=0
  trainer.save_freq="${SAVE_FREQ}"
  trainer.total_epochs=1
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.default_local_dir="${CACHE_ROOT}/checkpoints/${EXPERIMENT_NAME}"
  trainer.resume_mode=auto
)

HYDRA_OVERRIDES+=("${ROLLOUT_BATCHING_OVERRIDES[@]}")
HYDRA_OVERRIDES+=("${RAY_INIT_ARGS[@]}")
HYDRA_OVERRIDES+=("$@")

exec "${PYTHON_BIN}" -m branching_dapo.main_ppo_branching "${HYDRA_OVERRIDES[@]}"
