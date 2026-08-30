from __future__ import annotations

import csv
import json
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon as MplPolygon
from osgeo import ogr


ROOT = Path(__file__).resolve().parents[1]
DATE = "20250124"
POINTS = ROOT / "results" / "tables" / "full_area" / f"{DATE}_all_buildings_method_vs_gamma_with_lutan_deformation.csv"
BUILDINGS = Path("/home/u/geocoding/geo_hangzhou/geo_bc/a_geo_huajiachi/data/shp/huajiachi_clip.shp")
OUT_DIR = ROOT / "results" / "images" / "lutan_deformation"
REFERENCE_DIR = ROOT / "results" / "images" / "full_area_geobc_ps" / "defo"
PIC_DIR = ROOT / "results" / "pic_all" / "full_area_geobc_ps" / "defo"
SUMMARY = ROOT / "results" / "summaries" / "lutan_deformation_summary.json"


def read_rings(path: Path) -> list[np.ndarray]:
    ds = ogr.Open(str(path))
    if ds is None:
        raise FileNotFoundError(path)
    layer = ds.GetLayer(0)
    rings: list[np.ndarray] = []
    for feature in layer:
        geom = feature.GetGeometryRef()
        if geom is None:
            continue
        poly = geom.GetGeometryRef(0) if geom.GetGeometryName().upper() == "MULTIPOLYGON" else geom
        ring = poly.GetGeometryRef(0) if poly is not None else None
        if ring is None:
            continue
        xy = np.asarray([ring.GetPoint(i)[:2] for i in range(ring.GetPointCount())], dtype=float)
        if xy.shape[0] > 1:
            rings.append(xy)
    return rings


def read_points(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    keys = rows[0].keys()
    return {key: np.asarray([float(row[key]) for row in rows], dtype=float) for key in keys}


def draw_buildings(ax, rings: list[np.ndarray]) -> None:
    for ring in rings:
        ax.add_patch(MplPolygon(ring, closed=True, facecolor="#eeeeee", edgecolor="#777777", linewidth=0.22, alpha=0.55))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    PIC_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    data = read_points(POINTS)
    rings = read_rings(BUILDINGS)
    cmap = LinearSegmentedColormap.from_list("subsidence_zero_uplift", ["#b2182b", "#f7f7f7", "#2166ac"])
    norm = TwoSlopeNorm(vmin=-30.0, vcenter=0.0, vmax=30.0)
    specs = [
        ("method_lon", "method_lat", "deformation_rate_mm_yr", "Geo-BC deformation rate", "fig_01_lutan_geobc_deformation_rate.png", "mm/yr"),
        ("gamma_dsm_lon", "gamma_dsm_lat", "deformation_rate_mm_yr", "GAMMA/DSM deformation rate", "fig_02_lutan_gamma_deformation_rate.png", "mm/yr"),
        ("method_lon", "method_lat", "cumulative_deformation_mm", "Geo-BC cumulative deformation", "fig_03_lutan_geobc_cumulative_deformation.png", "mm"),
    ]
    all_xy = np.vstack([np.column_stack([data["method_lon"], data["method_lat"]]), np.vstack(rings)])
    xlim = (float(np.nanmin(all_xy[:, 0])), float(np.nanmax(all_xy[:, 0])))
    ylim = (float(np.nanmin(all_xy[:, 1])), float(np.nanmax(all_xy[:, 1])))
    for xkey, ykey, value_key, title, filename, unit in specs:
        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=350)
        draw_buildings(ax, rings)
        values = data[value_key]
        sc = ax.scatter(data[xkey], data[ykey], c=values, s=1.0, cmap=cmap, norm=norm, linewidths=0, alpha=0.82)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"LuTan-1 {title} ({DATE})")
        ax.ticklabel_format(useOffset=False, style="plain")
        cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.018)
        cb.set_label(unit)
        fig.tight_layout()
        fig.savefig(OUT_DIR / filename, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)

    reference_specs = [
        ("method_lon", "method_lat", "deformation_rate_mm_yr", "Geo-BC deformation rate", "fig_07_full_area_geobc_deformation_rate_buildings.png", "mm/yr"),
        ("lutan_lon", "lutan_lat", "deformation_rate_mm_yr", "LuTan source deformation rate", "fig_08_full_area_lutan_deformation_rate_buildings.png", "mm/yr"),
        ("method_lon", "method_lat", "cumulative_deformation_mm", "Geo-BC cumulative deformation", "fig_13_full_area_geobc_cumulative_deformation_buildings.png", "mm"),
        ("lutan_lon", "lutan_lat", "cumulative_deformation_mm", "LuTan source cumulative deformation", "fig_14_full_area_lutan_cumulative_deformation_buildings.png", "mm"),
    ]
    for xkey, ykey, value_key, title, filename, unit in reference_specs:
        fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=350)
        draw_buildings(ax, rings)
        sc = ax.scatter(data[xkey], data[ykey], c=data[value_key], s=1.0, cmap=cmap, norm=norm, linewidths=0, alpha=0.82)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"LuTan-1 {title} ({DATE})")
        ax.ticklabel_format(useOffset=False, style="plain")
        cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.018)
        cb.set_label(unit)
        fig.tight_layout()
        out = REFERENCE_DIR / filename
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        shutil.copy2(out, PIC_DIR / filename)

    comparison_specs = [
        ("deformation_rate_mm_yr", "fig_09_full_area_geobc_vs_lutan_deformation_rate_buildings.png", "Deformation rate", "mm/yr"),
        ("cumulative_deformation_mm", "fig_15_full_area_geobc_vs_lutan_cumulative_deformation_buildings.png", "Cumulative deformation", "mm"),
    ]
    for value_key, filename, title, unit in comparison_specs:
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.8), dpi=350, sharex=True, sharey=True)
        for ax, xkey, ykey, subtitle in [
            (axes[0], "method_lon", "method_lat", "Geo-BC"),
            (axes[1], "lutan_lon", "lutan_lat", "LuTan source"),
        ]:
            draw_buildings(ax, rings)
            sc = ax.scatter(data[xkey], data[ykey], c=data[value_key], s=0.9, cmap=cmap, norm=norm, linewidths=0, alpha=0.82)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(subtitle)
            ax.set_xlabel("Longitude")
            ax.ticklabel_format(useOffset=False, style="plain")
        axes[0].set_ylabel("Latitude")
        fig.subplots_adjust(left=0.06, right=0.86, bottom=0.10, top=0.90, wspace=0.08)
        cax = fig.add_axes([0.885, 0.20, 0.018, 0.60])
        cb = fig.colorbar(sc, cax=cax)
        cb.set_label(unit)
        fig.suptitle(f"LuTan-1 Geo-BC vs source {title.lower()} ({DATE})")
        out = REFERENCE_DIR / filename
        fig.savefig(out, bbox_inches="tight", pad_inches=0.04)
        plt.close(fig)
        shutil.copy2(out, PIC_DIR / filename)

    velocity = data["deformation_rate_mm_yr"]
    cumulative = data["cumulative_deformation_mm"]
    summary = {
        "date": DATE,
        "matched_points": int(velocity.size),
        "velocity_median_mm_yr": float(np.median(velocity)),
        "velocity_p05_mm_yr": float(np.percentile(velocity, 5)),
        "velocity_p95_mm_yr": float(np.percentile(velocity, 95)),
        "cumulative_median_mm": float(np.median(cumulative)),
        "cumulative_p05_mm": float(np.percentile(cumulative, 5)),
        "cumulative_p95_mm": float(np.percentile(cumulative, 95)),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
