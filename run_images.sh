#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/env.sh"

PY="$SAR_GEOCODE_PYTHON"
SHARED="$SAR_GEOCODE_SHARED_CODE_DIR"

"$PY" "$SHARED/make_full_area_planar_comparison.py"
"$PY" "$SHARED/make_additional_full_area_ppt_figures.py"
"$PY" code/make_lutan_coordinate_maps.py
"$PY" code/make_lutan_deformation_maps.py
"$PY" "$SHARED/make_huajiachi_full_area_geobc_3d.py"
"$PY" "$SHARED/make_huajiachi_hotspot_3d_zoom.py"
"$PY" code/mark_lutan_3d_extent_on_planar_map.py
"$PY" "$SHARED/make_huajiachi_interpolated_building_surfaces_3d.py"
"$PY" code/collect_all_images.py
"$PY" code/collect_all_images.py --check
