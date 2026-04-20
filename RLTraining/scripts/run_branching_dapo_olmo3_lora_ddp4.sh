#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# In verl FSDP, fsdp_size=1 on 4 ranks yields a 4-way ddp x 1-way fsdp mesh.
exec bash "${SCRIPT_DIR}/run_branching_dapo_olmo3.sh" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.ref.fsdp_config.fsdp_size=1 \
  actor_rollout_ref.model.lora_rank=32 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules='["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]' \
  actor_rollout_ref.actor.optim.lr=1e-4 \
  "$@"
