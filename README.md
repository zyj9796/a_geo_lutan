# a_geo_lutan：LuTan-1 建筑约束 SAR 精细地理编码

本目录是 LuTan-1A/LuTan-1B 华家池数据的独立复现工程。最终 35 张核心结果图已汇总到
[`results/all_images`](results/all_images/)，完整方法、输入、代码路径、逐阶段命令和图件对应
关系见 [`REPRODUCE_ALL_IMAGES.md`](REPRODUCE_ALL_IMAGES.md)。

## 快速复现

准备 `20250124.rslc`、`20250124.rslc.par` 和 `defo_ssa.mat` 后执行：

```bash
cd a_geo_lutan
mamba env create -f environment.yml
conda activate sar-geocode
export SAR_GEOCODE_PYTHON="$(command -v python)"
bash reproduce_all_images.sh
```

脚本自动执行主流程、全区域、严格同像素形变连接、坐标/形变/三维制图，并校验 35 张图。

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

默认输入位置可通过 `env.sh` 中列出的环境变量覆盖。原始 RSLC 和形变矩阵体积较大，不在
Git 仓库中；建筑轮廓、裁剪 DSM、代码和最终图片包随仓库提供。

## 分阶段运行

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

最终执行 `code/collect_all_images.py --check`，应报告 `checked_images=35`。每张图的来源、
SHA-256、文件大小和像素尺寸见 `results/all_images/MANIFEST.json`。
