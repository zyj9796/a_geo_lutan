#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/env.sh"
export SKIP_POLISHED_GEOCODE_FIGURES=1
mkdir -p results/logs/main
"$SAR_GEOCODE_PYTHON" \
  "$SAR_GEOCODE_SHARED_CODE_DIR/run_tongji_gamma_geocode.py" \
  --dates "$SAR_GEOCODE_DATE" --max-buildings 8 --max-points-per-building 120 \
  "$@" 2>&1 | tee results/logs/main/run.log
