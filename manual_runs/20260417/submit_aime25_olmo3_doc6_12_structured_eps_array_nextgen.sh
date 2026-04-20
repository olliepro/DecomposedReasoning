#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/users/PAA0201/ollieproudman/work/DecomposedReasoning"
ANALYSIS_DIR="${REPO_ROOT}/Analysis"
RUN_DIR="${REPO_ROOT}/manual_runs/20260417"
LOG_DIR="${RUN_DIR}/logs"
SBATCH_SCRIPT="${RUN_DIR}/run_branching_eval_nextgen_a100_1gpu_array.sbatch"
STRUCTURED_CONFIG="${ANALYSIS_DIR}/branching_eval/olmo3_checkpoint432_aime25_docs6_12_structured_only_32k_n48_mc24_grid.yaml"
EPS_CONFIG="${ANALYSIS_DIR}/branching_eval/olmo3_checkpoint432_aime25_docs6_12_eps_only_32k_eps033_n48_mc24_grid.yaml"

mkdir -p "${LOG_DIR}"

validate_config() {
  local config_path="$1"
  (
    cd "${ANALYSIS_DIR}"
    uv run python -c '
from pathlib import Path
import sys
from branching_eval.config_types import load_branching_eval_config

config_path = Path(sys.argv[1])
config = load_branching_eval_config(config_path=config_path)
assert config.decoding.max_gen_toks == 32768, config
assert config.branching.max_concurrent_branches == 24, config
assert config.run_matrix.baseline_rollouts == 48, config
print(config_path)
' "${config_path}"
  )
}

validate_config "${STRUCTURED_CONFIG}"
validate_config "${EPS_CONFIG}"

sbatch --test-only "${SBATCH_SCRIPT}" >/dev/null
sbatch --parsable "${SBATCH_SCRIPT}"
