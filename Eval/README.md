# Eval

Standalone benchmark evaluation project for checkpoint scoring with `lm-eval`.

## Setup

```bash
cd Eval
uv sync --extra dev
```

## Run standalone eval

```bash
cd Eval
uv run python -m eval_runner.standalone \
  --checkpoint /path/to/checkpoint-or-repo \
  --config ../SFTTraining/configs/runs/olmo3_7b_instruct_to_think.yaml
```

To run a quick subset eval:

```bash
cd Eval
uv run python -m eval_runner.standalone \
  --checkpoint Qwen/Qwen3-8B \
  --config configs/lm_eval_vllm.yaml \
  --limit 5
```

To evaluate with an eval-only config file:

```bash
cd Eval
uv run python -m eval_runner.standalone \
  --checkpoint /path/to/checkpoint-or-repo \
  --config configs/lm_eval_vllm.yaml
```

## Cleanup LoRA checkpoints

Create vLLM-friendly adapter checkpoints by stripping non-LoRA tensors from
`adapter_model.safetensors` (dry run):

```bash
cd Eval
uv run python cleanup_lora_checkpoints.py \
  --run-dir /fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/qwen3_8b_to_think_aimek4_lora_b1_ga12 \
  --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

Write cleaned adapters to sibling `checkpoint-<step>-vllm` directories:

```bash
cd Eval
uv run python cleanup_lora_checkpoints.py \
  --run-dir /fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/qwen3_8b_to_think_aimek4_lora_b1_ga12 \
  --target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj \
  --apply \
  --overwrite
```

## Serve LoRA checkpoint with vLLM

Serve a cleaned LoRA checkpoint on top of the base model with cross-NUMA env
auto-detection:

```bash
REPO_ROOT=/users/PAA0201/ollieproudman/work/DecomposedReasoning
LORA_CKPT=/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning/SFTTraining/outputs/qwen3_8b_to_think_aimek4_lora_b1_ga12/checkpoint-102-vllm

eval "$(python -c 'import sys; sys.path.insert(0,"'"${REPO_ROOT}"'/Eval"); from eval_runner.topology_env import read_topology_output, is_cross_numa_topology; topology_output = read_topology_output(); print("export NCCL_P2P_DISABLE=1 VLLM_DISABLE_PYNCCL=1 VLLM_SKIP_P2P_CHECK=0" if topology_output and is_cross_numa_topology(topology_text=topology_output) else "true")')"

uv run --project "${REPO_ROOT}/Eval" vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --dtype auto \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8 \
  --enable-lora \
  --lora-modules "qwen3_8b_lora=${LORA_CKPT}" \
  --port 8000
```

## Metrics

- Minerva Math500 logs only `math_verify`.
- AIME24/25 use custom sampled `avg@k` (`mean@k`) with `aime_avg_k` (default `32`).
- Flattened keys are dynamic:
  - `bench/aime24/avg_at_<k>`
  - `bench/aime25/avg_at_<k>`

## Cross-NUMA policy

For `model_type: vllm`, eval inspects `nvidia-smi topo -m`. When GPU links are cross-NUMA (`SYS`), eval sets:

```bash
export NCCL_P2P_DISABLE=1
export VLLM_DISABLE_PYNCCL=1
export VLLM_SKIP_P2P_CHECK=0
```

If topology probing fails or topology is not cross-NUMA, eval proceeds without forcing those overrides.

## Validation

```bash
cd Eval
uv run pytest
uv run pyright
```
