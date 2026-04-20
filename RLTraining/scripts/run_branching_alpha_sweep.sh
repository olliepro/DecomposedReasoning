#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for alpha in 0.25 0.5 0.75; do
  BRANCHING_ALPHA="${alpha}" \
  EXPERIMENT_NAME="olmo3_7b_branching_dapo_alpha_${alpha}" \
  "${SCRIPT_DIR}/run_branching_dapo_olmo3.sh"
done
