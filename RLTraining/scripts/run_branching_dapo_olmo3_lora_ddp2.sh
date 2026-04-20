#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# In verl FSDP, fsdp_size=1 on 2 ranks yields a 2-way ddp x 1-way fsdp mesh.
export NGPUS_PER_NODE="${NGPUS_PER_NODE:-2}"
export GEN_TP="${GEN_TP:-2}"

exec bash "${SCRIPT_DIR}/run_branching_dapo_olmo3.sh" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.ref.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
  actor_rollout_ref.ref.fsdp_config.param_offload=false \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.70 \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  "$@"
