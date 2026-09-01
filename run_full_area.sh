#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/env.sh"
mkdir -p results/tables/full_area results/logs/full_area
"$SAR_GEOCODE_PYTHON" \
  "$SAR_GEOCODE_SHARED_CODE_DIR/run_full_area_geocode.py" \
  --date "$SAR_GEOCODE_DATE" --max-buildings 0 --max-points-per-building 60 \
  "$@" 2>&1 | tee results/logs/full_area/run_full_area.log
