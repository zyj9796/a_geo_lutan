# LuTan-1 全部 35 张结果图复现说明

本文档说明如何从 LuTan-1 主影像、建筑轮廓、DSM 和 `defo_ssa.mat` 出发，复现
[`results/all_images`](results/all_images/) 中的 35 张 PNG。复现入口是
[`reproduce_all_images.sh`](reproduce_all_images.sh)，最终汇总和逐文件校验由
[`code/collect_all_images.py`](code/collect_all_images.py) 完成。

## 1. 复现范围与基准结果

本次固定使用主几何日期 `20250124`。完整结果包包含：

| 分组 | 数量 | 内容 |
|---|---:|---|
| `main__` | 8 | 小批量建筑投影、掩膜精炼、Geo-BC、误差和三维点 |
| `full_area__` | 11 | 全区域 Geo-BC/GAMMA 对比及诊断统计 |
| `coordinates__` | 6 | 严格同 SAR 像素下的 Geo-BC/LuTan 坐标图 |
| `deformation__` | 6 | 形变速率和累计形变图 |
| `3d__` | 4 | 全区域、热点和逐建筑表面插值三维图 |

当前基准运行得到：

- 候选建筑 2,869 栋，有效建筑 2,490 栋，跳过 379 栋；
- Geo-BC 散射点 102,332 个；
- Geo-BC 与 LuTan 形变矩阵严格同像素匹配 87,127 个，未匹配 15,205 个；
- 建筑级平均轮廓边界距离：Geo-BC 0.4273546 m，GAMMA/DSM 6.2199079 m；
- 形变速率中位数 -0.5083603 mm/yr，5%/95% 分位数为 -10.2451583/13.3398703 mm/yr。

这些数值分别记录在：

- `results/summaries/20250124_full_area_summary.json`
- `results/summaries/lutan_deformation_summary.json`
- `results/summaries/main_summary.json`

`results/` 中除 `all_images` 外均为可重新生成的中间结果，因此不推送到 GitHub。

## 2. 输入数据

### 2.1 GitHub 仓库中包含的输入

- 建筑轮廓：`data/buildings/huajiachi_clip.{shp,shx,dbf,prj}`；
- SAR 覆盖范围 DSM：`data/lutan_dsm_sar_extent.tif`；
- DSM 元数据：`data/lutan_dsm_sar_extent.json`；
- 鹿潭入口和制图代码：`code/`；
- 随仓库发布的共用算法和绘图代码：`code/shared/`；
- 核心距离—多普勒、建筑投影和反算代码：`src/`。

### 2.2 需要单独准备的大文件

原始 SAR 和形变矩阵体积较大，不放入 Git。默认目录结构为：

```text
<仓库父目录>/Lutan/
├── RE_SLAVES/
│   ├── 20250124.rslc
│   └── 20250124.rslc.par
└── defo_ssa.mat
```

精确复现 35 张图只使用 `20250124.rslc`、`20250124.rslc.par` 和
`defo_ssa.mat`。形变矩阵内部包含 20 个历元的位移列。

RSLC 必须与参数文件一致。本次数据为 GAMMA `FCOMPLEX`：大端、实部/虚部交错的
`float32`，尺寸为 2,000 个方位行乘 1,000 个距离采样。读取代码也兼容 GAMMA
`SCOMPLEX` 大端交错 `int16`。

如果数据不在默认位置，可在运行前覆盖变量：

```bash
export SAR_GEOCODE_RSLC_DIR=/path/to/RE_SLAVES
export SAR_GEOCODE_DEFORMATION_MAT=/path/to/defo_ssa.mat
```

建筑和 DSM 也可替换：

```bash
export SAR_GEOCODE_BUILDINGS_SHP=/path/to/buildings.shp
export SAR_GEOCODE_DSM_SAR_EXTENT_TIF=/path/to/dsm_clip.tif
```

建筑矢量必须使用经纬度坐标，包含 `height` 或 `Floor` 字段；缺少 `height` 时按
`Floor × 3 m` 估算建筑高度。

## 3. 软件环境

已验证环境如下：Python 3.10.20、NumPy 2.2.6、SciPy 1.15.2、Matplotlib
3.10.9、Shapely 2.1.2、Python GDAL 3.11.4、GNU Octave 6.4.0。

推荐从仓库提供的环境文件创建环境：

```bash
cd <仓库根目录>/a_geo_lutan
mamba env create -f environment.yml
conda activate sar-geocode
export SAR_GEOCODE_PYTHON="$(command -v python)"
```

系统还需安装可显示中文的 Noto CJK 字体。没有该字体不会改变计算结果，但文字宽度和
PNG 的 SHA-256 会不同。

依赖检查命令：

```bash
source env.sh
"$SAR_GEOCODE_PYTHON" -c \
  'import matplotlib,numpy,scipy,shapely; from osgeo import gdal,ogr; print("Python dependencies OK")'
octave --version
```

## 4. 一条命令完整复现

```bash
cd <仓库根目录>/a_geo_lutan
bash reproduce_all_images.sh
```

脚本会先检查输入和依赖，然后严格按以下顺序执行：

```bash
bash run.sh
bash run_full_area.sh
bash run_deformation.sh
bash run_images.sh
```

最后两次校验应输出：

```text
collected_images=35
image_package_check=ok
checked_images=35
```

只想利用已经存在的中间结果重新生成图件和汇总目录时，执行：

```bash
bash run_images.sh
```

只想重新汇总或校验 `all_images` 时，执行：

```bash
source env.sh
"$SAR_GEOCODE_PYTHON" code/collect_all_images.py
"$SAR_GEOCODE_PYTHON" code/collect_all_images.py --check
```

## 5. 方法原理

### 5.1 RSLC 幅度和轨道模型

RSLC 每个像素由实部 `I` 和虚部 `Q` 组成，幅度为：

```text
A = sqrt(I² + Q²)
```

显示时使用有效幅度的 2% 和 98% 分位数做拉伸。`.rslc.par` 提供近距、距离采样间隔、
方位起始时刻、方位行时间、雷达频率、多普勒多项式和轨道状态向量。代码对状态向量做
三次样条插值，获得任意方位时刻的卫星位置与速度。

主要代码：

- `src/geocode_gamma_rslc_with_buildings.py::parse_gamma_par`
- `src/geocode_gamma_rslc_with_buildings.py::read_rslc_amplitude`
- `src/geocode_gamma_rslc_with_buildings.py::make_orbit`

### 5.2 三维建筑模型投影到 SAR 坐标

对每栋建筑，在轮廓内部从 DSM 取建筑顶面高程 `H_top`，再计算：

```text
H_base = max(0, H_top - H_building)
```

建筑轮廓分别放到 `H_base` 和 `H_top`，生成底部和顶部顶点；每条边拆成两个立面三角形，
屋顶用三角扇连接。每个三维顶点从 WGS84 经纬高转为 ECEF，再通过距离—多普勒方程投影：

```text
R = ||X_ground - X_sat(t)||
f_d = -2 · (X_ground - X_sat(t)) · V_sat(t) / (lambda · R)
row = (t - t_start) / azimuth_line_time
col = (R - near_range) / range_pixel_spacing
```

其中方位时刻 `t` 通过多普勒残差最小化求解。投影后三角面栅格化得到初始建筑掩膜
`mask0`。

主要代码：

- `src/reproduce_thesis_tongji_tsx.py::build_model`
- `src/reproduce_thesis_tongji_tsx.py::project_llh_to_radar`
- `src/reproduce_thesis_tongji_tsx.py::rasterize_building`

### 5.3 强散射掩膜精炼

只在 `mask0` 内统计幅度，阈值为：

```text
threshold = max(P65(A), mean(A) + 0.25 × std(A))
```

保留高于阈值的像素，再膨胀 1 次，并限制在初始掩膜膨胀 2 次的范围内。全区域运行要求
初始掩膜至少 4 像素、精炼掩膜至少 2 像素；每栋建筑最多等间隔保留 60 个点。

主要代码：`src/reproduce_thesis_tongji_tsx.py::refine_mask`。

### 5.4 Geo-BC 三维坐标反算

每个精炼像素都有一个所属投影三角面。代码在 SAR 行列平面计算该像素相对于三角形三个
顶点的重心坐标 `(u, v, w)`，然后用同一组权重在 ECEF 三维顶点上插值：

```text
X = u · X0 + v · X1 + w · X2,  u + v + w = 1
```

最后把 `X` 转回 WGS84 经纬高。这一步把散射点约束到相应建筑的屋顶或立面，而不是把
它放到统一的地面高程面。

主要代码：

- `src/reproduce_thesis_tongji_tsx.py::barycentric`
- `src/reproduce_thesis_tongji_tsx.py::scatter_points_from_mask`

### 5.5 GAMMA/DSM 对照和误差统计

对照组使用完全相同的 SAR `row/col`，但固定在该位置的 DSM 高度，再联立斜距与多普勒
方程求经纬度。这样比较的是“建筑表面约束”和“DSM 高程面反算”的差别，不混入不同
SAR 像素的影响。

水平误差指标是点到所属建筑轮廓边界的最短距离，先在建筑中心附近把经纬度转换为局部
东—北坐标，再计算点到多边形边界的距离。它衡量结果对建筑轮廓的贴合程度，不是独立
控制点精度，也不应解释为绝对定位精度。

主要代码：

- `src/geocode_tongji_all_buildings_compare_gamma.py::solve_pixel_llh_at_height`
- `src/geocode_tongji_all_buildings_compare_gamma.py::gamma_dsm_height_points`
- `src/geocode_tongji_all_buildings_compare_gamma.py::boundary_distances`

### 5.6 形变矩阵严格同像素连接

`defo_ssa.mat` 的 `defo` 矩阵约定为：

- 第 1 列：从 1 开始的距离列；
- 第 2 列：从 1 开始的方位行；
- 第 3–5 列：LuTan 经度、纬度、高程；
- 第 6–25 列：20 个历元的位移。

Geo-BC CSV 则是从 0 开始的方位行、距离列。因此连接键为：

```text
defo(range_col, azimuth_row) = Geo-BC(col + 1, row + 1)
```

只保留键完全相同的记录。对 20 个历元，以距首历元的年数 `t` 做带截距线性回归；代码
先中心化时间，斜率为：

```text
velocity = displacement · (t - mean(t)) / sum((t - mean(t))²)
cumulative = displacement_last - displacement_first
```

本项目负值表示沉降。主要代码：`code/attach_lutan_deformation.m`。

### 5.7 坐标、形变和三维制图

- 坐标图使用 87,127 个严格同像素点；为控制 PNG 体积，显示时按固定索引均匀抽取最多
  70,000 个点，计算统计仍使用全量点。
- 形变图固定使用 `[-30, 30]` mm/yr 或 mm 的发散色标，负值为沉降。
- 全区域三维图把绝对点高程减去对应建筑基底高程，得到相对建筑高度。
- 热点图在 100 m 步长网格上搜索 450 m 窗口，综合极端形变点、非稳定点、总点数和
  形变等级数评分，选择得分最高且不在排除区内的窗口。
- 表面图以 5 m 网格构建每栋建筑屋顶和立面，只使用该建筑自身的散射点做三维 IDW：
  `weight = 1 / max(distance, 0.75)²`，绝不跨建筑插值。

主要代码：

- `code/make_lutan_coordinate_maps.py`
- `code/make_lutan_deformation_maps.py`
- `code/shared/make_huajiachi_full_area_geobc_3d.py`
- `code/shared/make_huajiachi_hotspot_3d_zoom.py`
- `code/mark_lutan_3d_extent_on_planar_map.py`
- `code/shared/make_huajiachi_interpolated_building_surfaces_3d.py`

## 6. 各阶段代码、输入和输出

### 阶段 A：8 张主流程图

命令：`bash run.sh`

入口：`code/shared/run_tongji_gamma_geocode.py`，核心算法位于
`src/run_result_tongji_geocoding.py` 和 `src/reproduce_thesis_tongji_tsx.py`。

输入是主 RSLC、参数文件、建筑轮廓和 DSM。该阶段选取 8 栋建筑，每栋最多 120 个点，
输出 `results/images/main/fig_01...fig_08`，汇总后对应 8 个 `main__` 文件。

### 阶段 B：11 张全区域图

命令：`bash run_full_area.sh`，随后由 `run_images.sh` 生成补充诊断图。

`code/shared/run_full_area_geocode.py` 调用
`src/geocode_tongji_all_buildings_compare_gamma.py`，处理 SAR 覆盖范围内全部建筑。关键中间表：

- `results/tables/full_area/20250124_all_buildings_method_vs_gamma_points.csv`
- `results/tables/full_area/20250124_all_buildings_fig5_4_like_stats.csv`
- `results/tables/full_area/20250124_all_buildings_skipped.csv`
- `results/geodata/full_area/20250124_all_valid_geocoded_buildings.geojson`
- `results/rasters/main/20250124_gamma_dem_geocoded_wgs84.tif`
- `results/tables/full_area/20250124_building_aligned_gamma_dsm_geocoded_wgs84.tif`

图件对应关系：

| `all_images` 文件 | 生成代码 |
|---|---|
| `full_area__20250124_fig5_4_like_all_buildings_error_scatter.png` | `run_full_area_geocode.py` |
| `full_area__20250124_fig5_4_like_all_buildings_map.png` | `run_full_area_geocode.py` |
| `full_area__20250124_fig_full_area_error_statistics.png` | `run_full_area_geocode.py` |
| `full_area__20250124_fig_full_area_gamma_vs_proposed.png` | `run_full_area_geocode.py` |
| `full_area__20250124_fig_full_area_planar_method_vs_gamma.png` | `make_full_area_planar_comparison.py` |
| `full_area__fig_10...fig_15` | `make_additional_full_area_ppt_figures.py` |

### 阶段 C：同像素形变表和 6 张形变图

命令：`bash run_deformation.sh`

Octave 脚本把全区域点表与 `defo_ssa.mat` 严格按 SAR 像素连接，输出：

`results/tables/full_area/20250124_all_buildings_method_vs_gamma_with_lutan_deformation.csv`

随后 `code/make_lutan_deformation_maps.py` 生成 Geo-BC、LuTan 源坐标以及二者对照的形变
速率/累计形变图，对应 6 个 `deformation__` 文件。

### 阶段 D：6 张坐标图

`code/make_lutan_coordinate_maps.py` 读取阶段 C 的同像素表和建筑轮廓，生成无底图与 SAR
底图版本各 3 张，对应 `coordinates__fig_01...fig_06`。

### 阶段 E：4 张三维图

`run_images.sh` 依次调用四个三维脚本：

| `all_images` 文件 | 生成代码 |
|---|---|
| `3d__fig_09_lutan_full_area_geobc_3d.png` | `make_huajiachi_full_area_geobc_3d.py` |
| `3d__fig_10_lutan_planar_geobc_3d_extent.png` | `mark_lutan_3d_extent_on_planar_map.py` |
| `3d__fig_11_lutan_hotspot_3d_zoom.png` | `make_huajiachi_hotspot_3d_zoom.py` |
| `3d__fig_12_lutan_interpolated_building_surfaces_3d.png` | `make_huajiachi_interpolated_building_surfaces_3d.py` |

### 阶段 F：汇总和完整性校验

`code/collect_all_images.py` 使用显式白名单从各输出目录复制 35 张核心图，增加分组前缀以避免
重名，并写入 `results/all_images/MANIFEST.json`。清单记录每张图的来源、SHA-256、字节数和
像素尺寸。`--check` 会验证：

1. 35 个源文件全部存在；
2. 35 个汇总文件全部存在且具有合法 PNG 头；
3. 汇总文件与对应源文件 SHA-256 完全一致；
4. 清单文件存在。

## 7. 中断后继续与常见问题

完整流程不会自动删除已有结果。某阶段失败后，修复输入或环境，再从该阶段重新运行即可。

- 报 `Missing GAMMA geocoded TIF`：先执行 `bash run.sh`。
- 报全区域点表不存在：执行 `bash run_full_area.sh`。
- 报带形变点表不存在：执行 `bash run_deformation.sh`。
- 报 `No module named osgeo`：GDAL 的 Python 绑定与当前解释器不是同一个环境，重新激活
  `sar-geocode` 并设置 `SAR_GEOCODE_PYTHON`。
- 中文显示为方框：安装 Noto Sans CJK 字体；统计值不受影响，但图片哈希会改变。
- 图像内容一致但 SHA-256 不同：检查 Matplotlib、字体、GDAL 版本和输入文件是否与上述
  环境一致。数值复现应优先比较摘要 JSON、CSV 行数和统计值。
- `matched_points` 不是 87,127：优先检查 `defo_ssa.mat` 的行列顺序、1/0 基索引约定，
  以及主影像日期是否为 `20250124`。

## 8. 结果解释限制

Geo-BC 点由建筑三角面约束，因此轮廓边界距离主要反映约束后的建筑贴合程度。它适合比较
同一 SAR 像素在建筑表面与 DSM 高程面反算后的几何差异，但不能替代外部控制点、独立测量
或绝对定位精度验证。三维表面形变图是逐建筑 IDW 可视化，不应解释为新增观测或跨建筑的
连续形变场。
