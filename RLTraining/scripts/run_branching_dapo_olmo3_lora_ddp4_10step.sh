#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export GEN_TP="${GEN_TP:-4}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-8}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-8}"
export TRAIN_PROMPT_MICRO_BSZ_PER_GPU="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU:-1}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-false}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
export PROJECT_NAME="${PROJECT_NAME:-branching_dapo_smoke}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-lora_ddp4_ten_step_smoke_cluster_openrouter_mem75}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-32}"
export MODEL_PATH="${MODEL_PATH:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/olmo3_7b_think_sft_to_think_merged_2213/checkpoint-448}"

exec bash "${SCRIPT_DIR}/run_branching_dapo_olmo3.sh" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
  actor_rollout_ref.ref.fsdp_config.fsdp_size=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
  +actor_rollout_ref.rollout.custom.branching_dapo.env_paths="[\"${REPO_ROOT}/.env\",\"${REPO_ROOT}/BuildSFTDataset/.env\"]" \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  reward.num_workers=2 \
  data.dataloader_num_workers=2 \
  trainer.total_training_steps=10 \
  trainer.save_freq=10 \
  "$@"
