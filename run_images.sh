#!/usr/bin/env bash
set -euo pipefail
cd /home/u/geocoding/geo_hangzhou/geo_bc/a_geo_lutan
source env.sh

PY=/home/u/miniconda3/envs/sar-geocode/bin/python
SHARED=../a_geo_huajiachi/code

"$PY" "$SHARED/make_full_area_planar_comparison.py"
"$PY" "$SHARED/make_additional_full_area_ppt_figures.py"
"$PY" code/make_lutan_coordinate_maps.py
"$PY" code/make_lutan_deformation_maps.py
"$PY" "$SHARED/make_huajiachi_full_area_geobc_3d.py"
"$PY" "$SHARED/make_huajiachi_hotspot_3d_zoom.py"
"$PY" "$SHARED/mark_3d_extent_on_planar_geobc_map.py"
"$PY" "$SHARED/make_huajiachi_interpolated_building_surfaces_3d.py"
