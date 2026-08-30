#!/usr/bin/env bash
set -euo pipefail
cd /home/u/geocoding/geo_hangzhou/geo_bc/a_geo_lutan
mkdir -p results/tables/full_area results/images/lutan_deformation results/summaries results/logs
octave --quiet code/attach_lutan_deformation.m \
  /home/u/geocoding/geo_hangzhou/Lutan/defo_ssa.mat \
  results/tables/full_area/20250124_all_buildings_method_vs_gamma_points.csv \
  results/tables/full_area/20250124_all_buildings_method_vs_gamma_with_lutan_deformation.csv \
  2>&1 | tee results/logs/attach_lutan_deformation.log
/home/u/miniconda3/envs/sar-geocode/bin/python code/make_lutan_deformation_maps.py \
  2>&1 | tee results/logs/make_lutan_deformation_maps.log
