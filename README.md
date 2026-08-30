# a_geo_lutan: LuTan-1 building-constrained SAR geocoding

This directory is an independent rerun workspace for the LuTan-1A/LuTan-1B
Huajiachi dataset. It reuses the validated building footprint and DSM inputs from
`a_geo_huajiachi`, but all generated tables, rasters, figures, logs, and summaries
are written below this directory.

## Inputs

- RSLC time series: `/home/u/geocoding/geo_hangzhou/Lutan/RE_SLAVES`
- Deformation matrix: `/home/u/geocoding/geo_hangzhou/Lutan/defo_ssa.mat`
- Master geometry used for the rerun: `20250124`
- Building footprints: `../a_geo_huajiachi/data/shp/huajiachi_clip.shp`
- DSM: `../a_geo_huajiachi/data/huajiachi_dsm_sar_extent.tif`

The LuTan RSLC files are GAMMA `FCOMPLEX` (big-endian interleaved float32),
1000 range samples by 2000 azimuth lines. The deformation matrix contains
1,346,701 rows and 25 columns: one-based range-column/azimuth-row, longitude,
latitude, height, followed by 20 displacement epochs.

## Run

```bash
cd /home/u/geocoding/geo_hangzhou/geo_bc/a_geo_lutan
bash run.sh
bash run_full_area.sh
bash run_deformation.sh
bash run_images.sh
```

`run.sh` is the small-batch validation run. `run_full_area.sh` processes all
buildings intersecting the LuTan master scene.

## Current results

- Candidate buildings in the master-scene footprint: 2,869
- Valid geocoded buildings: 2,490
- Skipped buildings: 379
- Building-constrained scatter points: 102,332
- Mean footprint-boundary distance: Geo-BC 0.43 m; GAMMA/DSM 6.22 m
- Strict same-pixel LuTan deformation matches: 87,127
- Unmatched Geo-BC points (invalid/missing deformation pixels): 15,205
- Median linear deformation rate: -0.51 mm/yr
- Rate 5th/95th percentiles: -10.25 / 13.34 mm/yr

Key outputs:

- `results/tables/full_area/20250124_all_buildings_method_vs_gamma_points.csv`
- `results/tables/full_area/20250124_all_buildings_method_vs_gamma_with_lutan_deformation.csv`
- `results/summaries/20250124_full_area_summary.json`
- `results/summaries/lutan_deformation_summary.json`
- `results/images/full_area/`
- `results/images/lutan_deformation/`
- `results/images/full_area_geobc_ps/`
- `results/pic_all/3d/`

Run `bash run_deformation.sh` after the full-area run to repeat the strict
same-pixel deformation join and deformation maps.

Run `bash run_images.sh` to regenerate the complete LuTan image package:

- 8 small-batch projection, mask, geocoding, error, and 3D figures;
- 11 full-area geocoding and diagnostic figures;
- 6 Geo-BC/LuTan coordinate maps;
- 6 deformation-rate and cumulative-deformation maps;
- 4 full-area, locator, hotspot, and interpolated-surface 3D figures.
