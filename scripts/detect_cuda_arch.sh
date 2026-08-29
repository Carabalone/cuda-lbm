#!/usr/bin/env bash
# Emit nvcc -arch= value for the primary CUDA device.
# Override: CUDA_ARCH=sm_XX in the environment or make GPU_ARCH=sm_XX
set -euo pipefail

if [[ -n "${CUDA_ARCH:-}" ]]; then
  case "${CUDA_ARCH}" in
    sm_*) echo "${CUDA_ARCH}" ;;
    *)    echo "sm_${CUDA_ARCH}" ;;
  esac
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "sm_61" >&2
  echo "sm_61"
  exit 0
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
compute_cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)"

case "${gpu_name}" in
  *1050*|*1050Ti*|*GTX1050*)
    echo "sm_61"
    exit 0
    ;;
  *2080*|*RTX2080*)
    echo "sm_75"
    exit 0
    ;;
esac

if [[ -n "${compute_cap}" ]]; then
  major="${compute_cap%%.*}"
  minor="${compute_cap#*.}"
  echo "sm_${major}${minor}"
  exit 0
fi

echo "sm_61" >&2
echo "sm_61"
