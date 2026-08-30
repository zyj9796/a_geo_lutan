#!/usr/bin/env bash
set -euo pipefail
cd /home/u/geocoding/geo_hangzhou/geo_bc/a_geo_lutan
source env.sh
mkdir -p results/tables/full_area results/logs/full_area
/home/u/miniconda3/envs/sar-geocode/bin/python \
  ../a_geo_huajiachi/code/run_full_area_geocode.py \
  --date 20250124 --max-buildings 0 --max-points-per-building 60 \
  "$@" 2>&1 | tee results/logs/full_area/run_full_area.log
