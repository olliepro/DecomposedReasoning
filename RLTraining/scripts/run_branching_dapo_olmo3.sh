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
SELECTOR_MODE="${SELECTOR_MODE:-cluster_across}"
BRANCHING_ALPHA="${BRANCHING_ALPHA:-0.5}"
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-32}"
TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-16}"
TRAIN_PROMPT_MICRO_BSZ_PER_GPU="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU:-1}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
GEN_TP="${GEN_TP:-4}"
SP_SIZE="${SP_SIZE:-1}"
OFFLOAD="${OFFLOAD:-true}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-true}"
VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-olmo3_7b_branching_dapo_${SELECTOR_MODE}_a${BRANCHING_ALPHA}}"
CACHE_ROOT="${CACHE_ROOT:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/RLTraining/branching_dapo}"
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

export PYTHONPATH="${RLTRAINING_DIR}:${RLTRAINING_DIR}/verl:${REPO_ROOT}/Analysis:${PYTHONPATH:-}"
export VLLM_LOGGING_LEVEL

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

"${PYTHON_BIN}" -m branching_dapo.main_ppo_branching \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  data.truncation='left' \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
  data.val_batch_size="${TRAIN_PROMPT_BSZ}" \
  actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.disable_log_stats=False \
  +actor_rollout_ref.rollout.agent.agent_loop_manager_class=branching_dapo.agent_loop_manager.BranchingAgentLoopManager \
  +actor_rollout_ref.rollout.custom.branching_dapo.selector_mode="${SELECTOR_MODE}" \
  +actor_rollout_ref.rollout.custom.branching_dapo.branch_fanout=4 \
  +actor_rollout_ref.rollout.custom.branching_dapo.max_branch_points_per_rollout=2 \
  +actor_rollout_ref.rollout.custom.branching_dapo.branch_prob=0.05 \
  +actor_rollout_ref.rollout.custom.branching_dapo.num_candidates=100 \
  +actor_rollout_ref.rollout.custom.branching_dapo.max_clusters=4 \
  +actor_rollout_ref.rollout.custom.branching_dapo.max_concurrent_branches=64 \
  +actor_rollout_ref.rollout.custom.branching_dapo.trigger_steer_enabled=True \
  +actor_rollout_ref.rollout.custom.branching_dapo.trigger_entropy_enabled=False \
  +actor_rollout_ref.rollout.custom.branching_dapo.cache_root="${CACHE_ROOT}" \
  actor_rollout_ref.model.path="${MODEL_PATH_VALUE}" \
  actor_rollout_ref.model.use_remove_padding=True \
  +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2)) \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3)) \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 3)) \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
  actor_rollout_ref.actor.optim.weight_decay=0.1 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_PROMPT_MINI_BSZ}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.loss_agg_mode=token-mean \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${OFFLOAD}" \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SP_SIZE}" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU}" \
  reward.reward_manager.name=dapo \
  reward.custom_reward_function.path="${RLTRAINING_DIR}/branching_dapo/reward_fn.py" \
  reward.custom_reward_function.name=compute_score_branching_dapo \
  algorithm.adv_estimator=branch_interpolated_grpo \
  +algorithm.branching_alpha="${BRANCHING_ALPHA}" \
  +algorithm.branching_intra_norm_by_std=True \
  +algorithm.branching_epsilon=1e-6 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.val_before_train=False \
  trainer.test_freq=0 \
  trainer.save_freq=10 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=200 \
  trainer.default_local_dir="${CACHE_ROOT}/checkpoints/${EXPERIMENT_NAME}" \
  trainer.resume_mode=auto \
  "${RAY_INIT_ARGS[@]}" \
  "$@"
