#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

set_default() {
  local name="$1"
  local value="$2"
  if [[ -z "${!name:-}" ]]; then
    export "${name}=${value}"
  fi
}

export NGPUS_PER_NODE="${NGPUS_PER_NODE:-4}"
export GEN_TP="${GEN_TP:-4}"
export ROLLOUT_DATA_PARALLEL_SIZE="${ROLLOUT_DATA_PARALLEL_SIZE:-1}"
export FSDP_SIZE="${FSDP_SIZE:-${NGPUS_PER_NODE}}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-${NGPUS_PER_NODE}}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-${NGPUS_PER_NODE}}"
export TRAIN_PROMPT_MICRO_BSZ_PER_GPU="${TRAIN_PROMPT_MICRO_BSZ_PER_GPU:-1}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-false}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
export MAX_STEER_TOKENS="${MAX_STEER_TOKENS:-15}"
export EXEC_TEMPERATURE="${EXEC_TEMPERATURE:-0.7}"
export STEER_TEMPERATURE="${STEER_TEMPERATURE:-1.0}"
export EXEC_TOP_P="${EXEC_TOP_P:-0.95}"
export STEER_TOP_P="${STEER_TOP_P:-0.95}"
export ROLLOUT_TOP_LOGPROBS="${ROLLOUT_TOP_LOGPROBS:-1}"
export RL_UPDATE_MODE="${RL_UPDATE_MODE:-all}"
export PRESENCE_PENALTY="${PRESENCE_PENALTY:-1.5}"
export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
export STEER_REPETITION_PENALTY="${STEER_REPETITION_PENALTY:-${REPETITION_PENALTY}}"
export CLIP_RATIO_LOW="${CLIP_RATIO_LOW:-0.2}"
export CLIP_RATIO_HIGH="${CLIP_RATIO_HIGH:-0.28}"
export ACTOR_LR="${ACTOR_LR:-1e-6}"
export ROLLOUT_MODE="${SMOKE_ROLLOUT_MODE:-${ROLLOUT_MODE:-branching}}"
case "${ROLLOUT_MODE}" in
  branching)
    set_default N_RESP_PER_PROMPT 1
    set_default ALGORITHM_ADV_ESTIMATOR branch_interpolated_grpo
    set_default SELECTOR_MODE cluster_across
    set_default BRANCH_PROB 0.10
    set_default BRANCH_FANOUT 2
    set_default MAX_BRANCH_POINTS_PER_ROLLOUT 4
    set_default NUM_CANDIDATES 50
    set_default EPSILON_GREEDY_PROB 0.1
    ;;
  baseline|no_branching)
    set_default N_RESP_PER_PROMPT 16
    set_default ALGORITHM_ADV_ESTIMATOR grpo
    set_default SELECTOR_MODE random
    set_default BRANCH_PROB 0.0
    set_default EPSILON_GREEDY_PROB 0.0
    ;;
  structured_baseline)
    set_default N_RESP_PER_PROMPT 16
    set_default ALGORITHM_ADV_ESTIMATOR grpo
    set_default SELECTOR_MODE random
    set_default BRANCH_PROB 0.0
    set_default EPSILON_GREEDY_PROB 0.0
    ;;
  epsilon_greedy)
    set_default N_RESP_PER_PROMPT 16
    set_default ALGORITHM_ADV_ESTIMATOR grpo
    set_default SELECTOR_MODE embed_diverse_topk_random
    set_default BRANCH_FANOUT 1
    set_default MAX_BRANCH_POINTS_PER_ROLLOUT 1
    set_default NUM_CANDIDATES 4
    set_default EPSILON_GREEDY_PROB 0.1
    ;;
  *)
    echo "Unsupported ROLLOUT_MODE: ${ROLLOUT_MODE}" >&2
    exit 2
    ;;
esac
export PROJECT_NAME="${PROJECT_NAME:-branching_dapo_qwen35_smoke}"
export MODEL_NAME_SLUG="${MODEL_NAME_SLUG:-qwen35_2b}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_NAME_SLUG}_${ROLLOUT_MODE}_smoke}"
export TRAINER_LOGGERS="${TRAINER_LOGGERS:-[\"console\",\"wandb\"]}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-2}"
export SAVE_FREQ="${SAVE_FREQ:-2}"
export RAY_NUM_CPUS="${RAY_NUM_CPUS:-${SLURM_CPUS_PER_TASK:-24}}"
export OFFLOAD="${OFFLOAD:-false}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-524288}"
export ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-4096}"
export ROLLOUT_CUDAGRAPH_CAPTURE_SIZES="${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES:-}"
export ROLLOUT_GDN_PREFILL_BACKEND="${ROLLOUT_GDN_PREFILL_BACKEND:-}"
export ROLLOUT_ASYNC_SCHEDULING="${ROLLOUT_ASYNC_SCHEDULING:-}"
export ROLLOUT_LONG_PREFILL_TOKEN_THRESHOLD="${ROLLOUT_LONG_PREFILL_TOKEN_THRESHOLD:-}"
export ROLLOUT_MAX_NUM_PARTIAL_PREFILLS="${ROLLOUT_MAX_NUM_PARTIAL_PREFILLS:-}"
export ROLLOUT_MAX_LONG_PARTIAL_PREFILLS="${ROLLOUT_MAX_LONG_PARTIAL_PREFILLS:-}"
export ROLLOUT_STREAM_INTERVAL="${ROLLOUT_STREAM_INTERVAL:-}"
export ROLLOUT_DECODE_CONTEXT_PARALLEL_SIZE="${ROLLOUT_DECODE_CONTEXT_PARALLEL_SIZE:-1}"
export ROLLOUT_DCP_COMM_BACKEND="${ROLLOUT_DCP_COMM_BACKEND:-}"
export ROLLOUT_ENABLE_DBO="${ROLLOUT_ENABLE_DBO:-false}"
export ROLLOUT_DBO_DECODE_TOKEN_THRESHOLD="${ROLLOUT_DBO_DECODE_TOKEN_THRESHOLD:-}"
export ROLLOUT_DBO_PREFILL_TOKEN_THRESHOLD="${ROLLOUT_DBO_PREFILL_TOKEN_THRESHOLD:-}"
export ROLLOUT_DISABLE_HYBRID_KV_CACHE_MANAGER="${ROLLOUT_DISABLE_HYBRID_KV_CACHE_MANAGER:-false}"
export ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE="${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE:-false}"
export ROLLOUT_REASONING_PARSER="${ROLLOUT_REASONING_PARSER-qwen3}"
export TOPOLOGY_POLICY="${TOPOLOGY_POLICY:-auto}"
export ACTOR_USE_REMOVE_PADDING="${ACTOR_USE_REMOVE_PADDING:-true}"
export ACTOR_USE_FUSED_KERNELS="${ACTOR_USE_FUSED_KERNELS:-true}"
export ACTOR_FUSED_KERNEL_BACKEND="${ACTOR_FUSED_KERNEL_BACKEND:-triton}"
export ACTOR_ATTN_IMPLEMENTATION="${ACTOR_ATTN_IMPLEMENTATION:-sdpa}"
export ACTOR_FSDP_SYNC_MODULE_STATES="${ACTOR_FSDP_SYNC_MODULE_STATES:-true}"
export ACTOR_FSDP_USE_META_INIT="${ACTOR_FSDP_USE_META_INIT:-true}"
export MODEL_LAYER_CLS_TO_WRAP="${MODEL_LAYER_CLS_TO_WRAP:-Qwen3_5DecoderLayer}"
export MODEL_PATH="${MODEL_PATH:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/vllm_exports/qwen35_2b_lr7e6_checkpoint-480_official_layout}"

ROLLOUT_BATCHING_OVERRIDES=()
if [[ "${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" == "none" || "${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" == "null" ]]; then
  ROLLOUT_BATCHING_OVERRIDES+=(
    actor_rollout_ref.rollout.max_num_batched_tokens=null
  )
elif [[ -n "${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" ]]; then
  ROLLOUT_BATCHING_OVERRIDES+=(
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  )
fi
if [[ "${ROLLOUT_MAX_NUM_SEQS}" == "none" || "${ROLLOUT_MAX_NUM_SEQS}" == "null" ]]; then
  ROLLOUT_BATCHING_OVERRIDES+=(
    actor_rollout_ref.rollout.max_num_seqs=null
  )
elif [[ -n "${ROLLOUT_MAX_NUM_SEQS}" ]]; then
  ROLLOUT_BATCHING_OVERRIDES+=(
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}"
  )
fi

ROLLOUT_GRAPH_OVERRIDES=()
if [[ -n "${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES}" ]]; then
  ROLLOUT_GRAPH_OVERRIDES+=(
    actor_rollout_ref.rollout.cudagraph_capture_sizes="${ROLLOUT_CUDAGRAPH_CAPTURE_SIZES}"
  )
fi

ROLLOUT_ENGINE_OVERRIDES=()
if [[ -n "${ROLLOUT_GDN_PREFILL_BACKEND}" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.gdn-prefill-backend="${ROLLOUT_GDN_PREFILL_BACKEND}"
  )
fi
if [[ -n "${ROLLOUT_ASYNC_SCHEDULING}" ]]; then
  case "${ROLLOUT_ASYNC_SCHEDULING}" in
    true|True|1)
      ROLLOUT_ENGINE_OVERRIDES+=(
        +actor_rollout_ref.rollout.engine_kwargs.vllm.async-scheduling=True
      )
      ;;
    false|False|0)
      ROLLOUT_ENGINE_OVERRIDES+=(
        +actor_rollout_ref.rollout.engine_kwargs.vllm.no-async-scheduling=True
      )
      ;;
    *)
      echo "ROLLOUT_ASYNC_SCHEDULING must be true or false, got: ${ROLLOUT_ASYNC_SCHEDULING}" >&2
      exit 2
      ;;
  esac
fi
if [[ -n "${ROLLOUT_LONG_PREFILL_TOKEN_THRESHOLD}" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.long-prefill-token-threshold="${ROLLOUT_LONG_PREFILL_TOKEN_THRESHOLD}"
  )
fi
if [[ -n "${ROLLOUT_MAX_NUM_PARTIAL_PREFILLS}" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max-num-partial-prefills="${ROLLOUT_MAX_NUM_PARTIAL_PREFILLS}"
  )
fi
if [[ -n "${ROLLOUT_MAX_LONG_PARTIAL_PREFILLS}" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.max-long-partial-prefills="${ROLLOUT_MAX_LONG_PARTIAL_PREFILLS}"
  )
fi
if [[ -n "${ROLLOUT_STREAM_INTERVAL}" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.stream-interval="${ROLLOUT_STREAM_INTERVAL}"
  )
fi
if [[ "${ROLLOUT_DECODE_CONTEXT_PARALLEL_SIZE}" != "1" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.decode-context-parallel-size="${ROLLOUT_DECODE_CONTEXT_PARALLEL_SIZE}"
  )
fi
if [[ -n "${ROLLOUT_DCP_COMM_BACKEND}" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.dcp-comm-backend="${ROLLOUT_DCP_COMM_BACKEND}"
  )
fi
if [[ "${ROLLOUT_ENABLE_DBO}" == "true" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable-dbo=True
  )
fi
if [[ -n "${ROLLOUT_DBO_DECODE_TOKEN_THRESHOLD}" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.dbo-decode-token-threshold="${ROLLOUT_DBO_DECODE_TOKEN_THRESHOLD}"
  )
fi
if [[ -n "${ROLLOUT_DBO_PREFILL_TOKEN_THRESHOLD}" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.dbo-prefill-token-threshold="${ROLLOUT_DBO_PREFILL_TOKEN_THRESHOLD}"
  )
fi
if [[ "${ROLLOUT_DISABLE_HYBRID_KV_CACHE_MANAGER}" == "true" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable-hybrid-kv-cache-manager=True
  )
fi
if [[ "${ROLLOUT_DISABLE_CUSTOM_ALL_REDUCE}" == "true" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable-custom-all-reduce=True
  )
fi
if [[ -n "${ROLLOUT_REASONING_PARSER}" ]]; then
  ROLLOUT_ENGINE_OVERRIDES+=(
    +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning-parser="${ROLLOUT_REASONING_PARSER}"
  )
fi

exec bash "${SCRIPT_DIR}/run_branching_dapo_olmo3.sh" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  actor_rollout_ref.ref.fsdp_config.fsdp_size="${FSDP_SIZE}" \
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="[\"${MODEL_LAYER_CLS_TO_WRAP}\"]" \
  +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="[\"${MODEL_LAYER_CLS_TO_WRAP}\"]" \
  actor_rollout_ref.actor.fsdp_config.param_offload="${OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OFFLOAD}" \
  actor_rollout_ref.ref.fsdp_config.param_offload="${OFFLOAD}" \
  actor_rollout_ref.actor.fsdp_config.use_torch_compile=False \
  actor_rollout_ref.ref.fsdp_config.use_torch_compile=False \
  +actor_rollout_ref.actor.fsdp_config.sync_module_states="${ACTOR_FSDP_SYNC_MODULE_STATES}" \
  +actor_rollout_ref.ref.fsdp_config.sync_module_states="${ACTOR_FSDP_SYNC_MODULE_STATES}" \
  +actor_rollout_ref.actor.fsdp_config.use_meta_init="${ACTOR_FSDP_USE_META_INIT}" \
  +actor_rollout_ref.ref.fsdp_config.use_meta_init="${ACTOR_FSDP_USE_META_INIT}" \
  actor_rollout_ref.actor.use_torch_compile=False \
  actor_rollout_ref.ref.use_torch_compile=False \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  "${ROLLOUT_BATCHING_OVERRIDES[@]}" \
  "${ROLLOUT_GRAPH_OVERRIDES[@]}" \
  actor_rollout_ref.rollout.data_parallel_size="${ROLLOUT_DATA_PARALLEL_SIZE}" \
  actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING:-true}" \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.scheduling_policy="${ROLLOUT_SCHEDULING_POLICY:-priority}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.language-model-only=True \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.max-logprobs="${ROLLOUT_TOP_LOGPROBS}" \
  "${ROLLOUT_ENGINE_OVERRIDES[@]}" \
  +actor_rollout_ref.rollout.custom.branching_dapo.env_paths="[\"${REPO_ROOT}/.env\",\"${REPO_ROOT}/BuildSFTDataset/.env\"]" \
  +actor_rollout_ref.model.override_config.attn_implementation="${ACTOR_ATTN_IMPLEMENTATION}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.use_remove_padding="${ACTOR_USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.use_fused_kernels="${ACTOR_USE_FUSED_KERNELS}" \
  actor_rollout_ref.model.fused_kernel_options.impl_backend="${ACTOR_FUSED_KERNEL_BACKEND}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR}" \
  reward.num_workers=2 \
  data.dataloader_num_workers=0 \
  trainer.logger="${TRAINER_LOGGERS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.save_freq="${SAVE_FREQ}" \
  "$@"
