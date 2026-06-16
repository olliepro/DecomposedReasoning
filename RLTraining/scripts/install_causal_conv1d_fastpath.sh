#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-${RL_PYTHON_BIN:-python3.12}}"
PACKAGE_VERSION="${CAUSAL_CONV1D_VERSION:-1.6.2.post1}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/fs/scratch/PAA0201/ollieproudman/DecomposedReasoning}"
BUILD_ROOT="${CAUSAL_CONV1D_BUILD_ROOT:-${SCRATCH_ROOT}/RLTraining/build/causal_conv1d_fastpath}"
MAX_JOBS="${MAX_JOBS:-12}"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc is required to build causal-conv1d against the active torch ABI." >&2
  echo "Load a CUDA module first, e.g. module load cuda/12.9.1." >&2
  exit 1
fi

CUDA_HOME="${CUDA_HOME:-$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)}"
export CUDA_HOME
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAX_JOBS

mkdir -p "${BUILD_ROOT}"
rm -rf "${BUILD_ROOT:?}/causal_conv1d-${PACKAGE_VERSION}" "${BUILD_ROOT:?}/download"
mkdir -p "${BUILD_ROOT}/download"

"${PYTHON_BIN}" -m pip download \
  --no-cache-dir \
  --no-binary=:all: \
  --no-deps \
  --no-build-isolation \
  --dest "${BUILD_ROOT}/download" \
  "causal-conv1d==${PACKAGE_VERSION}"

tar -xzf "${BUILD_ROOT}/download/causal_conv1d-${PACKAGE_VERSION}.tar.gz" -C "${BUILD_ROOT}"

setup_py="${BUILD_ROOT}/causal_conv1d-${PACKAGE_VERSION}/setup.py"
perl -0pi -e 's{        cc_flag\.append\("-gencode"\)\n        cc_flag\.append\("arch=compute_75,code=sm_75"\)\n        cc_flag\.append\("-gencode"\)\n        cc_flag\.append\("arch=compute_80,code=sm_80"\)\n        cc_flag\.append\("-gencode"\)\n        cc_flag\.append\("arch=compute_87,code=sm_87"\)\n        if bare_metal_version >= Version\("11\.8"\):\n            cc_flag\.append\("-gencode"\)\n            cc_flag\.append\("arch=compute_90,code=sm_90"\)\n        if bare_metal_version >= Version\("12\.8"\):\n            cc_flag\.append\("-gencode"\)\n            cc_flag\.append\("arch=compute_100,code=sm_100"\)\n            cc_flag\.append\("-gencode"\)\n            cc_flag\.append\("arch=compute_120,code=sm_120"\)\n        if bare_metal_version >= Version\("13\.0"\):\n            cc_flag\.append\("-gencode"\)\n            cc_flag\.append\("arch=compute_103,code=sm_103"\)\n            cc_flag\.append\("-gencode"\)\n            cc_flag\.append\("arch=compute_110,code=sm_110"\)\n            cc_flag\.append\("-gencode"\)\n            cc_flag\.append\("arch=compute_121,code=sm_121"\)\n}{        cc_flag.extend(["-gencode", "arch=compute_80,code=sm_80"])\n        if bare_metal_version >= Version("11.8"):\n            cc_flag.extend(["-gencode", "arch=compute_90,code=sm_90"])\n}s' "${setup_py}"

grep -q 'arch=compute_80,code=sm_80' "${setup_py}"
grep -q 'arch=compute_90,code=sm_90' "${setup_py}"
if grep -Eq 'arch=compute_(75|87|100|103|110|120|121)' "${setup_py}"; then
  echo "Unexpected causal-conv1d CUDA arch target remained in ${setup_py}" >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip install \
  --no-cache-dir \
  --no-deps \
  --no-build-isolation \
  --force-reinstall \
  -v \
  "${BUILD_ROOT}/causal_conv1d-${PACKAGE_VERSION}"

"${PYTHON_BIN}" - <<'PY'
import torch
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from fla.modules import FusedRMSNormGated
from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule

print("torch:", torch.__version__)
print(
    "fast path imports:",
    causal_conv1d_fn is not None,
    causal_conv1d_update is not None,
    FusedRMSNormGated is not None,
    chunk_gated_delta_rule is not None,
    fused_recurrent_gated_delta_rule is not None,
)
PY
