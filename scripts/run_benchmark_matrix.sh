#!/usr/bin/env bash
# Full TGV benchmark matrix: 100 warmup + 250 measured steps, cold-GPU pauses.
set -euo pipefail
cd "$(dirname "$0")/.."

DEFINES=src/defines.hpp
BACKUP=$(mktemp)
cp "$DEFINES" "$BACKUP"
trap 'cp "$BACKUP" "$DEFINES"' EXIT

RUN_TIMEOUT="${RUN_TIMEOUT:-120}"
RUN_TIMEOUT_MRT_OPT_AOS="${RUN_TIMEOUT_MRT_OPT_AOS:-120}"
COOLDOWN_RUN="${COOLDOWN_RUN:-15}"
COOLDOWN_CASE="${COOLDOWN_CASE:-30}"
RESULTS="${RESULTS:-benchmark_results_long.txt}"

case_run_timeout() {
  local label="$1"
  if [[ "$label" == "MRT_opt_AoS" ]]; then
    echo "$RUN_TIMEOUT_MRT_OPT_AOS"
  else
    echo "$RUN_TIMEOUT"
  fi
}

median_of_three() {
  printf '%s\n' "$@" | sort -n | awk 'NR==2 {print $1}'
}

gpu_cooldown() {
  local sec="$1"
  local label="${2:-GPU cooldown}"
  echo "[$label] sleeping ${sec}s..." >&2
  sleep "$sec"
}

mlups_one_run() {
  local timeout_sec="$1"
  local log
  log=$(mktemp)
  timeout "${timeout_sec}s" ./bin/lbm_solver --benchmark >"$log" 2>&1 || true
  if grep -q '\[WARNING\]' "$log"; then
    echo "WARN"
    cat "$log" >&2
    rm -f "$log"
    return
  fi
  if ! grep -q 'Performance:' "$log"; then
    echo "TIMEOUT"
    tail -20 "$log" >&2
    rm -f "$log"
    return
  fi
  awk '/Performance:/ {print $2; exit}' "$log"
  rm -f "$log"
}

write_defines() {
  local collision="$1"
  local layout="$2"

  {
    echo '#ifndef DEFINES_H'
    echo '#define DEFINES_H'
    echo '#define D3Q27'
    echo ' #define USE_TAYLOR_GREEN_3D'
    echo ''
    case "$layout" in
      AOS)  echo '#define LBM_LAYOUT_AOS' ;;
      CSOA) echo '#define LBM_LAYOUT_CSOA' ;;
    esac
    echo ''
    case "$collision" in
      BGK)          echo '#define LBM_COLLISION_BGK' ;;
      MRT)          echo '#define LBM_COLLISION_MRT' ;;
      MRT_BASELINE) echo '#define LBM_COLLISION_MRT_BASELINE' ;;
    esac
    cat <<'EOF'
#if (defined(LBM_COLLISION_BGK) + defined(LBM_COLLISION_MRT) + defined(LBM_COLLISION_MRT_BASELINE)) > 1
#error "Pick at most one collision operator define"
#endif
#ifdef USE_TAYLOR_GREEN_3D
    #define SCALE 2
    #define NX (64 * SCALE)
    #define NY (64 * SCALE)
    #define NZ (64 * SCALE)
#endif
#ifndef NZ
    #define NZ 1
#endif
#define BLOCK_SIZE 16
#ifndef DEBUG_NODE
  #define DEBUG_NODE 5
#endif
#ifndef VALUE_THRESHOLD
  #define VALUE_THRESHOLD 5.0f
#endif
#endif
EOF
  } >"$DEFINES"
}

run_case() {
  local label="$1"
  local collision="$2"
  local layout="$3"

  echo "=== $label ($collision / $layout) ===" >&2
  write_defines "$collision" "$layout"
  make clean >/dev/null 2>&1
  if ! make -j4 >/dev/null 2>&1; then
    echo "$label,BUILD_FAIL,,,,"
    return
  fi

  local tsec
  tsec=$(case_run_timeout "$label")
  echo "Run timeout for $label: ${tsec}s" >&2

  gpu_cooldown "$COOLDOWN_CASE" "before $label runs"

  local r1 r2 r3
  r1=$(mlups_one_run "$tsec")
  gpu_cooldown "$COOLDOWN_RUN" "between run 1 and 2"
  r2=$(mlups_one_run "$tsec")
  gpu_cooldown "$COOLDOWN_RUN" "between run 2 and 3"
  r3=$(mlups_one_run "$tsec")

  if [[ "$r1" == "WARN" || "$r2" == "WARN" || "$r3" == "WARN" ]]; then
    echo "$label,WARNING,,,,"
    return
  fi
  if [[ "$r1" == "TIMEOUT" || "$r2" == "TIMEOUT" || "$r3" == "TIMEOUT" ]]; then
    echo "$label,TIMEOUT,,,,"
    return
  fi
  local med
  med=$(median_of_three "$r1" "$r2" "$r3")
  echo "$label,$collision,$layout,$r1,$r2,$r3,$med"
}

{
  echo "# TGV benchmark matrix"
  echo "# Date: $(date -Iseconds)"
  echo "# Window: 100 warmup + 250 measured (350 total timesteps)"
  echo "# Per-run timeout: ${RUN_TIMEOUT}s | cooldown run: ${COOLDOWN_RUN}s | cooldown case: ${COOLDOWN_CASE}s"
  echo "# Grid: 128^3 | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
  echo ""
  echo "label,collision,layout,run1,run2,run3,median_mlups"
} >"$RESULTS"

run_case "unop_MRT"      "MRT_BASELINE" "AOS"  | tee -a "$RESULTS"
gpu_cooldown "$COOLDOWN_CASE" "after unop_MRT"
run_case "BGK_SoA"       "BGK"          "SOA"  | tee -a "$RESULTS"
gpu_cooldown "$COOLDOWN_CASE" "after BGK_SoA"
run_case "BGK_AoS"       "BGK"          "AOS"  | tee -a "$RESULTS"
gpu_cooldown "$COOLDOWN_CASE" "after BGK_AoS"
run_case "BGK_CSoA"      "BGK"          "CSOA" | tee -a "$RESULTS"
gpu_cooldown "$COOLDOWN_CASE" "after BGK_CSoA"
run_case "MRT_opt_SoA"   "MRT"          "SOA"  | tee -a "$RESULTS"
gpu_cooldown "$COOLDOWN_CASE" "after MRT_opt_SoA"
run_case "MRT_opt_AoS"   "MRT"          "AOS"  | tee -a "$RESULTS"
gpu_cooldown "$COOLDOWN_CASE" "after MRT_opt_AoS"
run_case "MRT_opt_CSoA"  "MRT"          "CSOA" | tee -a "$RESULTS"
gpu_cooldown "$COOLDOWN_CASE" "after MRT_opt_CSoA"
run_case "CM_SoA"        "CM"           "SOA"  | tee -a "$RESULTS"
gpu_cooldown "$COOLDOWN_CASE" "after CM_SoA"
run_case "CM_AoS"        "CM"           "AOS"  | tee -a "$RESULTS"
gpu_cooldown "$COOLDOWN_CASE" "after CM_AoS"
run_case "CM_CSoA"       "CM"           "CSOA" | tee -a "$RESULTS"

echo "Wrote $RESULTS" >&2
