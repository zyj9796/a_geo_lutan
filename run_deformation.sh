#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/env.sh"
mkdir -p results/tables/full_area results/images/lutan_deformation results/summaries results/logs
octave --quiet code/attach_lutan_deformation.m \
  "$SAR_GEOCODE_DEFORMATION_MAT" \
  "results/tables/full_area/${SAR_GEOCODE_DATE}_all_buildings_method_vs_gamma_points.csv" \
  "$SAR_GEOCODE_DEFORMATION_POINTS_CSV" \
  2>&1 | tee results/logs/attach_lutan_deformation.log
"$SAR_GEOCODE_PYTHON" code/make_lutan_deformation_maps.py \
  2>&1 | tee results/logs/make_lutan_deformation_maps.log
