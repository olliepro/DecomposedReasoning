# RLTraining

`RLTraining` is the repo's reinforcement-learning area. It has two layers:

1. An upstream [`verl`](./verl) submodule that provides the PPO/DAPO trainer, worker model, and Hydra config base.
2. A repo-local [`branching_dapo`](./branching_dapo) package that plugs branching rollout behavior into `verl` without modifying the submodule.

The intent is to keep upstream `verl` updateable while putting project-specific rollout logic, reward shaping, and launch scripts in a small local surface.

## Directory Layout

```text
RLTraining/
├── branching_dapo/      # Repo-local PPO/DAPO extensions layered on top of verl
├── scripts/             # Shell entrypoints for local runs and alpha sweeps
├── slurm/               # Cluster launch wrappers
├── tests/               # Regression coverage for branching behavior
└── verl/                # Upstream submodule; treat as vendor code
```

## What Lives Where

### `branching_dapo/`

This is the local integration layer. It does not replace `verl`; it wraps the points where `verl` already allows customization.

- [`main_ppo_branching.py`](./branching_dapo/main_ppo_branching.py)
  - Hydra entrypoint for RL runs.
  - Registers the repo-local advantage estimator and launches `verl.trainer.main_ppo.run_ppo`.
- [`task_runner.py`](./branching_dapo/task_runner.py)
  - Reuses upstream dataset and worker setup.
  - Swaps in the local trainer wrapper.
- [`trainer.py`](./branching_dapo/trainer.py)
  - Extends the upstream Ray PPO trainer.
  - Preserves the normal PPO update flow and merges rollout-side and advantage-side branching metrics into logging.
- [`agent_loop_manager.py`](./branching_dapo/agent_loop_manager.py)
  - Main rollout customization point.
  - Groups repeated prompts, builds one shared branch tree per prompt group, and returns one leaf per repeated sample so `verl` batch shapes stay compatible.
  - Uses the `Analysis/branching_eval` machinery for actual branch execution.
- [`advantage.py`](./branching_dapo/advantage.py)
  - Implements `branch_interpolated_grpo`.
  - Computes prompt-level inter-branch GRPO-style advantages plus recursive intra-branch advantages, then interpolates them with `alpha`.
- [`reward_fn.py`](./branching_dapo/reward_fn.py)
  - Adapts math reward scoring into the shape expected by the DAPO reward manager.
  - Injects serialized branch metadata into `uid` so the custom advantage estimator can reconstruct the tree.
- [`config_types.py`](./branching_dapo/config_types.py)
  - Typed helpers for rollout settings and advantage settings.
  - Defines the supported selector modes: `cluster_across` and `random`.
- [`runtime_metrics.py`](./branching_dapo/runtime_metrics.py)
  - Small in-process metric stash used to pass branching metrics from rollout and advantage code into the trainer logger.
- [`bootstrap.py`](./branching_dapo/bootstrap.py)
  - Adds repo-local import roots for `RLTraining`, `Analysis`, and the `verl` submodule.
- [`rollout_utils.py`](./branching_dapo/rollout_utils.py)
  - Shared helpers for prompt grouping, leaf metadata, rollout summaries, and reward score packaging.
- [`verl_compat.py`](./branching_dapo/verl_compat.py)
  - Thin compatibility layer for local registration points into `verl`.

### `scripts/`

These are the operator-facing shell entrypoints.

- [`run_branching_dapo_olmo3.sh`](./scripts/run_branching_dapo_olmo3.sh)
  - Main launch script for OLMo-3 branching DAPO.
  - Resolves the SFT checkpoint, RL dataset path, rollout parameters, and Hydra overrides.
- [`run_branching_alpha_sweep.sh`](./scripts/run_branching_alpha_sweep.sh)
  - Runs the same launcher three times with `alpha in {0.25, 0.5, 0.75}`.

### `slurm/`

- [`branching_dapo_train.sbatch`](./slurm/branching_dapo_train.sbatch)
  - Two-node, four-GPU-per-node cluster wrapper.
  - Validates `RL_PYTHON_BIN` or `RL_ENV_SETUP_CMD` before delegating to the shell launcher.

### `tests/`

- [`test_branching_dapo.py`](./tests/test_branching_dapo.py)
  - Covers the local estimator, selector-mode behavior, branch metadata, and rollout-side expectations.

### `verl/`

This is upstream code. Treat it as vendor code.

- Keep local project behavior in `branching_dapo/` whenever possible.
- Prefer using existing seams such as:
  - custom reward functions
  - custom advantage estimators
  - custom task runners
  - custom agent-loop managers

## Control Flow

The branching RL stack is intentionally flat:

1. `sbatch RLTraining/slurm/branching_dapo_train.sbatch`
2. [`run_branching_dapo_olmo3.sh`](./scripts/run_branching_dapo_olmo3.sh)
3. `python -m branching_dapo.main_ppo_branching`
4. [`BranchingTaskRunner`](./branching_dapo/task_runner.py)
5. [`BranchingRayPPOTrainer`](./branching_dapo/trainer.py)
6. [`BranchingAgentLoopManager`](./branching_dapo/agent_loop_manager.py)
7. `Analysis/branching_eval` branch executor and selector logic

The key design choice is that rollout branching happens before PPO sees the batch, but the output shape is kept compatible with standard `verl` repeated-prompt rollouts.

## Current Branching Behavior

The local code currently supports two selector modes:

- `cluster_across`
  - Default mode.
  - Uses semantic grouping through the `Analysis/branching_eval` selector path and chooses branches across clusters.
- `random`
  - No-cluster fallback and ablation path.
  - Keeps the same branching code path without semantic clustering.

The advantage estimator mixes:

- inter-branch reward centering across all leaves for the same prompt
- recursive intra-branch deltas along the selected path

Final scalar advantage:

```text
A = alpha * A_intra + (1 - alpha) * A_inter
```

## Inputs and Outputs

### Expected inputs

- RL dataset from [`BuildRLDataset/output/train.parquet`](../BuildRLDataset/output/train.parquet)
- An SFT checkpoint path
  - Usually resolved from `BEST_CHECKPOINT_FILE`
  - Can be overridden directly with `MODEL_PATH`
- A Python environment with `hydra`, `ray`, and `verl` dependencies
- For clustering mode, selector credentials available through the configured environment files or env vars used by `Analysis/branching_eval`

### Main outputs

- Checkpoints under `CACHE_ROOT/checkpoints/<experiment_name>`
- Branching artifacts and selector caches under `CACHE_ROOT`
- Standard trainer logs plus branching-specific runtime metrics in W&B and console output

## Main Environment Knobs

The shell launcher is parameterized through environment variables.

Common ones:

- `RL_PYTHON_BIN`
  - Python executable with the RL dependencies installed.
- `BEST_CHECKPOINT_FILE`
  - File containing the selected SFT checkpoint path.
- `MODEL_PATH`
  - Direct checkpoint override; bypasses `BEST_CHECKPOINT_FILE`.
- `TRAIN_FILE`
  - RL parquet input. Defaults to `BuildRLDataset/output/train.parquet`.
- `SELECTOR_MODE`
  - `cluster_across` or `random`.
- `BRANCHING_ALPHA`
  - Interpolation weight for intra-branch advantage.
- `EPSILON_GREEDY_PROB`
  - Probability that an eligible steer trigger which does not create true
    branches still uses one-candidate epsilon-greedy selector exploration.
  - Compatible with branching; set to `0` to disable inline exploration.
- `EXPERIMENT_NAME`
  - W&B and checkpoint naming.
- `CACHE_ROOT`
  - Root for caches, branch artifacts, and trainer outputs.

## Common Edit Points

If you need to change RL behavior, start here:

- Change rollout grouping or leaf metadata:
  - [`agent_loop_manager.py`](./branching_dapo/agent_loop_manager.py)
- Change recursive advantage logic:
  - [`advantage.py`](./branching_dapo/advantage.py)
- Change reward normalization or math scoring payloads:
  - [`reward_fn.py`](./branching_dapo/reward_fn.py)
- Change launch defaults for OLMo-3:
  - [`run_branching_dapo_olmo3.sh`](./scripts/run_branching_dapo_olmo3.sh)
- Change cluster resource shape:
  - [`branching_dapo_train.sbatch`](./slurm/branching_dapo_train.sbatch)
- Change selector or branching execution semantics:
  - `Analysis/branching_eval/*`

## Operational Notes

- `val_before_train` is disabled in the launcher right now. Model selection is expected to happen through external eval jobs rather than an internal RL validation split.
- `VAL_FILE` currently defaults to the train parquet so the data interface stays satisfied even when the RL dataset is train-only.
- The local code is written to keep `verl` unmodified. If a change can be done in `branching_dapo/`, prefer that over editing the submodule.
