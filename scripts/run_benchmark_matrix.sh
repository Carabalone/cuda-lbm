#!/usr/bin/env bash
# Legacy compatibility entry point; prefer: python3 benchmark.py
set -euo pipefail
cd "$(dirname "$0")/.."
exec python3 benchmark.py --grid "${GRID:-128}" "$@"
