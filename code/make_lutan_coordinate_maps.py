from __future__ import annotations

import csv
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from osgeo import gdal, ogr


ROOT = Path(__file__).resolve().parents[1]
DATE = "20250124"
POINTS = ROOT / "results" / "tables" / "full_area" / f"{DATE}_all_buildings_method_vs_gamma_with_lutan_deformation.csv"
BUILDINGS = Path("/home/u/geocoding/geo_hangzhou/geo_bc/a_geo_huajiachi/data/shp/huajiachi_clip.shp")
SAR = ROOT / "results" / "tables" / "full_area" / f"{DATE}_building_aligned_gamma_dsm_geocoded_wgs84.tif"
OUT_DIR = ROOT / "results" / "images" / "full_area_geobc_ps"
PIC_DIR = ROOT / "results" / "pic_all" / "full_area_geobc_ps"


def read_points() -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(POINTS.open(encoding="utf-8")))
    return {key: np.asarray([float(row[key]) for row in rows], dtype=float) for key in rows[0]}


def read_building_segments() -> np.ndarray:
    ds = ogr.Open(str(BUILDINGS))
    if ds is None:
        raise FileNotFoundError(BUILDINGS)
    segments = []
    for feature in ds.GetLayer(0):
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        poly = geom.GetGeometryRef(0) if geom.GetGeometryName().upper() == "MULTIPOLYGON" else geom
        ring = poly.GetGeometryRef(0) if poly is not None else None
        if ring is None:
            continue
        xy = np.asarray([ring.GetPoint(i)[:2] for i in range(ring.GetPointCount())], dtype=float)
        if xy.shape[0] > 1:
            segments.extend(np.stack([xy[:-1], xy[1:]], axis=1))
    return np.asarray(segments, dtype=float)


def read_sar() -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ds = gdal.Open(str(SAR), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(SAR)
    scale = max(1.0, math.sqrt(ds.RasterXSize * ds.RasterYSize / 3_000_000))
    arr = ds.ReadAsArray(
        buf_xsize=max(1, int(ds.RasterXSize / scale)),
        buf_ysize=max(1, int(ds.RasterYSize / scale)),
    ).astype(np.float32)
    gt = ds.GetGeoTransform()
    extent = (gt[0], gt[0] + ds.RasterXSize * gt[1], gt[3] + ds.RasterYSize * gt[5], gt[3])
    valid = arr[arr > 0]
    if valid.size:
        lo, hi = np.percentile(valid, [1, 99.5])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    return arr, extent


def style_axis(ax, bounds: tuple[float, float, float, float]) -> None:
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_aspect(1.0 / math.cos(math.radians((bounds[2] + bounds[3]) / 2)), adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.ticklabel_format(useOffset=False, style="plain")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PIC_DIR.mkdir(parents=True, exist_ok=True)
    data = read_points()
    segments = read_building_segments()
    sar, sar_extent = read_sar()
    all_xy = np.vstack([segments.reshape(-1, 2), np.column_stack([data["method_lon"], data["method_lat"]])])
    bounds = (
        float(np.min(all_xy[:, 0])), float(np.max(all_xy[:, 0])),
        float(np.min(all_xy[:, 1])), float(np.max(all_xy[:, 1])),
    )
    method = np.column_stack([data["method_lon"], data["method_lat"]])
    raw = np.column_stack([data["lutan_lon"], data["lutan_lat"]])
    specs = [
        ("fig_01_full_area_geobc_buildings.png", "Geo-BC coordinates", False, "method"),
        ("fig_02_full_area_lutan_coordinates_buildings.png", "LuTan source coordinates", False, "raw"),
        ("fig_03_full_area_geobc_vs_lutan_buildings.png", "Geo-BC vs LuTan source coordinates", False, "both"),
        ("fig_04_full_area_geobc_sar_buildings.png", "Geo-BC coordinates on SAR", True, "method"),
        ("fig_05_full_area_lutan_coordinates_sar_buildings.png", "LuTan source coordinates on SAR", True, "raw"),
        ("fig_06_full_area_geobc_vs_lutan_sar_buildings.png", "Geo-BC vs LuTan source coordinates on SAR", True, "both"),
    ]
    keep = np.linspace(0, len(method) - 1, min(len(method), 70000), dtype=int)
    for filename, title, with_sar, mode in specs:
        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=350)
        if with_sar:
            ax.imshow(sar, cmap="gray", extent=sar_extent, origin="upper", interpolation="nearest", alpha=0.90)
            line_color, line_alpha = "#f4f4f4", 0.62
        else:
            ax.set_facecolor("#fafafa")
            line_color, line_alpha = "#8b9298", 0.48
        ax.add_collection(LineCollection(segments, colors=line_color, linewidths=0.22, alpha=line_alpha, zorder=2))
        if mode in {"raw", "both"}:
            ax.scatter(raw[keep, 0], raw[keep, 1], s=0.7, c="#f28e2b", alpha=0.42, linewidths=0, label="LuTan source", zorder=3)
        if mode in {"method", "both"}:
            ax.scatter(method[keep, 0], method[keep, 1], s=0.7, c="#2563eb", alpha=0.58, linewidths=0, label="Geo-BC", zorder=4)
        if mode == "both":
            ax.legend(loc="upper right", frameon=False, markerscale=5)
        style_axis(ax, bounds)
        ax.set_title(f"LuTan-1 {title} ({DATE})")
        fig.tight_layout()
        out = OUT_DIR / filename
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        shutil.copy2(out, PIC_DIR / filename)
    print(f"generated={len(specs)} matched_points={len(method)}")


if __name__ == "__main__":
    main()
