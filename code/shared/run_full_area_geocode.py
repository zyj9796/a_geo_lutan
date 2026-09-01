from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
os.environ.setdefault("FONTCONFIG_PATH", "/etc/fonts")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from osgeo import gdal

from io_paths import (
    BUILDINGS_SHP,
    DATA_DIR,
    DSM_TIF,
    FULL_AREA_DIR as FULL_DIR,
    FULL_AREA_GEOJSON_DIR,
    FULL_AREA_IMAGE_DIR,
    FULL_AREA_LOG_DIR,
    PROJECT_DIR,
    REPO_ROOT,
    RESULTS_DIR,
    RSLC_DIR,
    SUMMARY_DIR,
    TIF_DIR,
    ensure_core_output_dirs,
)

sys.path.insert(0, str(REPO_ROOT / "src"))

import geocode_tongji_all_buildings_compare_gamma as full_core

AREA_LABEL = os.environ.get("SAR_GEOCODE_AREA_LABEL", "华家池")


def read_tif(path: Path) -> tuple[np.ndarray, list[float]]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    max_pixels = int(os.environ.get("MAX_PLOT_GEOTIFF_PIXELS", "4000000"))
    width = int(ds.RasterXSize)
    height = int(ds.RasterYSize)
    if width * height > max_pixels:
        scale = float(np.sqrt((width * height) / max_pixels))
        arr = ds.ReadAsArray(buf_xsize=max(1, int(width / scale)), buf_ysize=max(1, int(height / scale))).astype(np.float32)
    else:
        arr = ds.ReadAsArray().astype(np.float32)
    gt = ds.GetGeoTransform()
    extent = [gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3]]
    ds = None
    valid = arr[arr > 0]
    if valid.size:
        lo, hi = np.percentile(valid, [1, 99.5])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return arr, extent


def read_buildings(path: Path) -> list[np.ndarray]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rings = []
    for feat in data.get("features", []):
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        rings.append(ring[:, :2])
    return rings


def read_points(path: Path, amp: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    method = []
    gamma = []
    intensity = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method.append((float(row["method_lon"]), float(row["method_lat"]), float(row["method_height_m"])))
            gamma.append((float(row["gamma_dsm_lon"]), float(row["gamma_dsm_lat"])))
            if amp is not None:
                rr = int(round(float(row["row"])))
                cc = int(round(float(row["col"])))
                if 0 <= rr < amp.shape[0] and 0 <= cc < amp.shape[1]:
                    intensity.append(float(amp[rr, cc]) / 255.0)
                else:
                    intensity.append(0.0)
    return np.asarray(method, dtype=np.float64), np.asarray(gamma, dtype=np.float64), np.asarray(intensity, dtype=np.float64)


def write_points_geojson(points_csv: Path, out_geojson: Path) -> None:
    features = []
    with points_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            features.append(
                {
                    "type": "Feature",
                    "properties": {
                        "fid": int(row["fid"]),
                        "row": float(row["row"]),
                        "col": float(row["col"]),
                        "height_m": float(row["method_height_m"]),
                        "gamma_lon": float(row["gamma_dsm_lon"]),
                        "gamma_lat": float(row["gamma_dsm_lat"]),
                        "gamma_height_m": float(row["gamma_dsm_height_m"]),
                        "gamma_ok": int(row["gamma_dsm_ok"]),
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(row["method_lon"]), float(row["method_lat"]), float(row["method_height_m"])],
                    },
                }
            )
    out_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")


def safe_apply_dsm_heights(buildings: list[dict], dsm) -> list[dict]:
    out = []
    for b in buildings:
        try:
            top_h = dsm.building_surface_height(b["ring_lonlat"])
        except Exception as exc:
            print(f"skip building {b.get('fid')}: DSM sample failed: {exc}", flush=True)
            continue
        item = dict(b)
        item["top_height_m"] = top_h
        item["base_height_m"] = max(0.0, top_h - float(b["height_m"]))
        out.append(item)
    return out


def make_full_area_figures(date: str, gamma_tif: Path, stats_csv: Path, points_csv: Path, buildings_geojson: Path) -> None:
    bg, extent = read_tif(gamma_tif)
    rings = read_buildings(buildings_geojson)
    par = full_core.parse_gamma_par(RSLC_DIR / f"{date}.rslc.par")
    amp = full_core.read_rslc_amplitude(RSLC_DIR / f"{date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))
    method, gamma, intensity = read_points(points_csv, amp)
    all_rings = np.vstack(rings)
    xpad = max(float(np.ptp(all_rings[:, 0])) * 0.035, 0.00015)
    ypad = max(float(np.ptp(all_rings[:, 1])) * 0.035, 0.00015)
    xlim = (float(np.min(all_rings[:, 0]) - xpad), float(np.max(all_rings[:, 0]) + xpad))
    ylim = (float(np.min(all_rings[:, 1]) - ypad), float(np.max(all_rings[:, 1]) + ypad))

    plt.rcParams.update(
        {
            "font.family": "Noto Sans CJK JP",
            "axes.unicode_minus": False,
            "font.size": 8,
            "axes.titlesize": 10,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    )

    fig, ax = plt.subplots(figsize=(7.0, 6.3), dpi=450)
    if intensity.size == method.shape[0]:
        ax.set_facecolor("#050505")
        order = np.argsort(intensity)
        step_sar = max(1, order.size // 90000)
        keep = order[::step_sar]
        ax.scatter(method[keep, 0], method[keep, 1], s=0.65, c=intensity[keep], cmap="gray", vmin=0, vmax=1, alpha=0.58, linewidths=0, zorder=1)
    else:
        ax.imshow(bg, cmap="gray", extent=extent, origin="upper", interpolation="nearest", alpha=0.35)
    step_ring = max(1, len(rings) // 1000)
    for ring in rings[::step_ring]:
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="white", linewidth=0.32, alpha=0.62, zorder=3))
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="#222222", linewidth=0.16, alpha=0.7, zorder=4))
    step_gamma = max(1, gamma.shape[0] // 60000)
    step_method = max(1, method.shape[0] // 60000)
    ax.scatter(gamma[::step_gamma, 0], gamma[::step_gamma, 1], s=0.55, c="#f28e2b", alpha=0.24, linewidths=0, label="传统 GAMMA/DEM", zorder=5)
    sc = ax.scatter(
        method[::step_method, 0],
        method[::step_method, 1],
        s=0.8,
        c=method[::step_method, 2],
        cmap="viridis",
        alpha=0.72,
        linewidths=0,
        label="建筑约束方法",
        zorder=6,
    )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("经度 / 度")
    ax.set_ylabel("纬度 / 度")
    ax.set_title(f"{AREA_LABEL} 全区域建筑约束地理编码（{date}）")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="white", linewidth=0.2, alpha=0.2)
    ax.legend(loc="upper right", frameon=True, framealpha=0.92)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.018)
    cbar.set_label("高度 / 米")
    fig.tight_layout(pad=0.25)
    fig.savefig(FULL_AREA_IMAGE_DIR / f"{date}_fig_full_area_gamma_vs_proposed.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    rows = list(csv.DictReader(stats_csv.open(encoding="utf-8")))
    method_mean = np.asarray([float(r["method_mean_boundary_distance_m"]) for r in rows], dtype=np.float64)
    gamma_mean = np.asarray([float(r["gamma_dsm_mean_boundary_distance_m"]) for r in rows], dtype=np.float64)
    order = np.argsort(gamma_mean)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.9), dpi=450)
    x = np.arange(len(rows))
    axes[0].scatter(x, gamma_mean[order], s=3.0, c="#f28e2b", alpha=0.62, linewidths=0, label="传统 GAMMA/DEM")
    axes[0].scatter(x, np.maximum(method_mean[order], 1e-6), s=3.0, c="#2c7fb8", alpha=0.68, linewidths=0, label="建筑约束方法")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("按 GAMMA/DEM 误差排序的建筑")
    axes[0].set_ylabel("平均边界距离 / 米")
    axes[0].grid(axis="y", which="both", color="#dddddd", linewidth=0.28)
    axes[0].legend(frameon=False, loc="upper left")
    axes[1].boxplot([gamma_mean, np.maximum(method_mean, 1e-6)], tick_labels=["GAMMA/DEM", "建筑约束\n方法"], showfliers=True)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("平均边界距离 / 米")
    axes[1].grid(axis="y", which="both", color="#dddddd", linewidth=0.28)
    fig.suptitle("全区域水平误差统计")
    fig.tight_layout(pad=0.35)
    fig.savefig(FULL_AREA_IMAGE_DIR / f"{date}_fig_full_area_error_statistics.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    # 兼容旧文件名的两张图也在本地重绘，避免继续沿用上游英文标注版本。
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), dpi=360, sharex=True, sharey=True)
    for ax, points, title, color in [
        (axes[0], method, "建筑约束方法", "#2563eb"),
        (axes[1], gamma, "传统 GAMMA/DEM", "#f28e2b"),
    ]:
        ax.imshow(bg, cmap="gray", extent=extent, origin="upper", interpolation="nearest", alpha=0.38)
        for ring in rings[::step_ring]:
            ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="white", linewidth=0.34, alpha=0.65))
        step = max(1, points.shape[0] // 70000)
        ax.scatter(points[::step, 0], points[::step, 1], s=0.65, c=color, alpha=0.52, linewidths=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("经度 / 度")
        ax.set_title(title)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.grid(color="white", linewidth=0.2, alpha=0.18)
    axes[0].set_ylabel("纬度 / 度")
    fig.suptitle(f"{AREA_LABEL}全区域建筑地理编码结果对比（{date}）")
    fig.tight_layout()
    fig.savefig(FULL_AREA_IMAGE_DIR / f"{date}_fig5_4_like_all_buildings_map.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.6), dpi=360)
    positive = (gamma_mean > 0) & (method_mean > 0)
    ax.scatter(gamma_mean[positive], method_mean[positive], s=8, c="#2563eb", alpha=0.52, linewidths=0)
    lo = max(1e-4, float(min(gamma_mean[positive].min(), method_mean[positive].min())))
    hi = float(max(gamma_mean[positive].max(), method_mean[positive].max()))
    ax.plot([lo, hi], [lo, hi], color="#d62728", linestyle="--", linewidth=1.0, label="一比一参考线")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("传统 GAMMA/DEM 平均边界距离 / 米")
    ax.set_ylabel("建筑约束方法平均边界距离 / 米")
    ax.set_title("逐建筑边界误差对比")
    ax.grid(which="both", color="#d8d8d8", linewidth=0.3, alpha=0.65)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FULL_AREA_IMAGE_DIR / f"{date}_fig5_4_like_all_buildings_error_scatter.png", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def collect_full_area_images() -> None:
    FULL_AREA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FULL_DIR.glob("*.png"):
        path.replace(FULL_AREA_IMAGE_DIR / path.name)


def collect_full_area_data() -> None:
    FULL_AREA_GEOJSON_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    for path in FULL_DIR.glob("*.geojson"):
        path.replace(FULL_AREA_GEOJSON_DIR / path.name)
    for path in FULL_DIR.glob("*.json"):
        path.replace(SUMMARY_DIR / path.name)


def summarize(date: str, stats_csv: Path, points_csv: Path, skipped_csv: Path) -> dict:
    rows = list(csv.DictReader(stats_csv.open(encoding="utf-8")))
    skipped = list(csv.DictReader(skipped_csv.open(encoding="utf-8"))) if skipped_csv.exists() else []
    point_count = sum(1 for _ in points_csv.open(encoding="utf-8")) - 1
    method_mean = [float(r["method_mean_boundary_distance_m"]) for r in rows]
    gamma_mean = [float(r["gamma_dsm_mean_boundary_distance_m"]) for r in rows]
    out = {
        "date": date,
        "valid_buildings": len(rows),
        "skipped_buildings": len(skipped),
        "scatter_points": point_count,
        "method_mean_boundary_distance_m": float(np.mean(method_mean)),
        "method_median_boundary_distance_m": float(statistics.median(method_mean)),
        "method_p90_boundary_distance_m": float(np.percentile(method_mean, 90)),
        "gamma_dem_mean_boundary_distance_m": float(np.mean(gamma_mean)),
        "gamma_dem_median_boundary_distance_m": float(statistics.median(gamma_mean)),
        "gamma_dem_p90_boundary_distance_m": float(np.percentile(gamma_mean, 90)),
    }
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / f"{date}_full_area_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    return out


def update_markdown(summary: dict) -> None:
    import update_markdown as md_updater

    md_updater.main()


def write_runner() -> None:
    if PROJECT_DIR != Path(__file__).resolve().parent.parent:
        return
    runner = PROJECT_DIR / "run_full_area.sh"
    shared_code_dir = Path(__file__).resolve().parent
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"cd {PROJECT_DIR}\n"
        f"export SAR_GEOCODE_PROJECT_DIR={shlex.quote(str(PROJECT_DIR))}\n"
        f"export SAR_GEOCODE_RSLC_DIR={shlex.quote(str(RSLC_DIR))}\n"
        f"export SAR_GEOCODE_BUILDINGS_SHP={shlex.quote(str(BUILDINGS_SHP))}\n"
        f"export SAR_GEOCODE_DSM_TIF={shlex.quote(str(DSM_TIF))}\n"
        f"export SAR_GEOCODE_DSM_SAR_EXTENT_TIF={shlex.quote(str(DSM_TIF))}\n"
        "export PROJ_LIB=/home/u/miniconda3/envs/sar-geocode/share/proj\n"
        "export PROJ_DATA=/home/u/miniconda3/envs/sar-geocode/share/proj\n"
        "export MAX_PLOT_GEOTIFF_PIXELS=3000000\n"
        "mkdir -p results/tables/full_area results/logs/full_area\n"
        f"/home/u/miniconda3/envs/sar-geocode/bin/python {shared_code_dir / 'prepare_dsm_for_sar.py'}\n"
        f"/home/u/miniconda3/envs/sar-geocode/bin/python {shared_code_dir / 'run_full_area_geocode.py'} \"$@\" 2>&1 | tee results/logs/full_area/run_full_area.log\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20250109")
    parser.add_argument("--max-points-per-building", type=int, default=60)
    parser.add_argument("--max-buildings", type=int, default=0, help="0 means all buildings in SAR coverage")
    args = parser.parse_args()

    ensure_core_output_dirs()
    FULL_DIR.mkdir(parents=True, exist_ok=True)
    FULL_AREA_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    write_runner()
    gamma_tif = TIF_DIR / f"{args.date}_gamma_dem_geocoded_wgs84.tif"
    if not gamma_tif.exists():
        raise FileNotFoundError(f"Missing GAMMA geocoded TIF: {gamma_tif}. Run bash run.sh first.")

    full_core.DEFAULT_RSLC_DIR = RSLC_DIR
    full_core.apply_dsm_heights = safe_apply_dsm_heights
    ns = argparse.Namespace(
        date=args.date,
        buildings_shp=str(BUILDINGS_SHP),
        gamma_tif=str(gamma_tif),
        dsm=str(DSM_TIF),
        out_dir=str(FULL_DIR),
        max_buildings=args.max_buildings,
        max_points_per_building=args.max_points_per_building,
        min_mask0_pixels=4,
        min_mask_pixels=2,
    )
    full_core.run(ns)
    collect_full_area_images()
    collect_full_area_data()

    stats_csv = FULL_DIR / f"{args.date}_all_buildings_fig5_4_like_stats.csv"
    points_csv = FULL_DIR / f"{args.date}_all_buildings_method_vs_gamma_points.csv"
    skipped_csv = FULL_DIR / f"{args.date}_all_buildings_skipped.csv"
    buildings_geojson = FULL_AREA_GEOJSON_DIR / f"{args.date}_all_valid_geocoded_buildings.geojson"
    write_points_geojson(points_csv, FULL_AREA_GEOJSON_DIR / f"{args.date}_all_buildings_proposed_points.geojson")
    make_full_area_figures(args.date, gamma_tif, stats_csv, points_csv, buildings_geojson)
    collect_full_area_images()
    collect_full_area_data()
    summary = summarize(args.date, stats_csv, points_csv, skipped_csv)
    update_markdown(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
