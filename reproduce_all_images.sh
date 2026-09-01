#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
source "$PROJECT_DIR/env.sh"

required_files=(
  "$SAR_GEOCODE_RSLC_DIR/${SAR_GEOCODE_DATE}.rslc"
  "$SAR_GEOCODE_RSLC_DIR/${SAR_GEOCODE_DATE}.rslc.par"
  "$SAR_GEOCODE_DEFORMATION_MAT"
  "$SAR_GEOCODE_BUILDINGS_SHP"
  "${SAR_GEOCODE_BUILDINGS_SHP%.shp}.dbf"
  "${SAR_GEOCODE_BUILDINGS_SHP%.shp}.shx"
  "$SAR_GEOCODE_DSM_SAR_EXTENT_TIF"
  "$SAR_GEOCODE_SHARED_CODE_DIR/run_tongji_gamma_geocode.py"
  "$SAR_GEOCODE_SHARED_CODE_DIR/run_full_area_geocode.py"
)

missing=0
for required in "${required_files[@]}"; do
  if [[ ! -f "$required" ]]; then
    printf 'Missing required input: %s\n' "$required" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 2
fi

"$SAR_GEOCODE_PYTHON" -c 'import matplotlib, numpy, scipy, shapely; from osgeo import gdal, ogr'
command -v octave >/dev/null

printf '[1/5] Small-batch method figures\n'
bash "$PROJECT_DIR/run.sh"
printf '[2/5] Full-area geocoding and diagnostics\n'
bash "$PROJECT_DIR/run_full_area.sh"
printf '[3/5] Strict same-pixel deformation attachment\n'
bash "$PROJECT_DIR/run_deformation.sh"
printf '[4/5] Coordinate, deformation, and 3D figures\n'
bash "$PROJECT_DIR/run_images.sh"
printf '[5/5] Final package verification\n'
"$SAR_GEOCODE_PYTHON" "$PROJECT_DIR/code/collect_all_images.py" --check

printf 'Reproduction complete: %s\n' "$PROJECT_DIR/results/all_images"
