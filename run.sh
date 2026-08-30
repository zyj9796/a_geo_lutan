#!/usr/bin/env bash
set -euo pipefail
cd /home/u/geocoding/geo_hangzhou/geo_bc/a_geo_lutan
source env.sh
export SKIP_POLISHED_GEOCODE_FIGURES=1
mkdir -p results/logs/main
/home/u/miniconda3/envs/sar-geocode/bin/python \
  ../a_geo_huajiachi/code/run_tongji_gamma_geocode.py \
  --dates 20250124 --max-buildings 8 --max-points-per-building 120 \
  "$@" 2>&1 | tee results/logs/main/run.log
