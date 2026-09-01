from __future__ import annotations

import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from make_full_area_geobc_discrete_3d import (
    CLASS_BOUNDS,
    CLASS_CMAP,
    CLASS_NORM,
    add_discrete_colorbar,
)
from make_publication_3d_geobc import OUT_DIR, TEXT, plot_scene


PROJECT_DIR = Path(os.environ.get("SAR_GEOCODE_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
DATE = os.environ.get("SAR_GEOCODE_DATE", "20250109")
AREA_LABEL = os.environ.get("SAR_GEOCODE_AREA_LABEL", "华家池")
BUILDINGS_GEOJSON = (
    PROJECT_DIR / "results" / "geodata" / "full_area" / f"{DATE}_all_valid_geocoded_buildings.geojson"
)
POINTS_CSV = Path(
    os.environ.get(
        "SAR_GEOCODE_DEFORMATION_POINTS_CSV",
        PROJECT_DIR / "results" / "tables" / "full_area" / f"{DATE}_all_buildings_method_vs_gamma_points.csv",
    )
).resolve()
DEFORMATION_RATE = Path(
    "/home/u/geocoding/geo_hangzhou/Hangzhou_Huajiachi/postprocessing/defo_rate"
)
OUT_PNG = OUT_DIR / f"fig_09_{os.environ.get('SAR_GEOCODE_FIGURE_PREFIX', 'huajiachi')}_full_area_geobc_3d.png"

POINT_SIZE = 0.90
BUILDING_ALPHA = 0.10
VIEW_ELEV = 42
VIEW_AZIM = 45
VIEW_ROLL = 0


def load_full_area_buildings() -> dict[int, dict]:
    data = json.loads(BUILDINGS_GEOJSON.read_text(encoding="utf-8"))
    buildings: dict[int, dict] = {}
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        fid = int(props["fid"])
        ring = np.asarray(feature["geometry"]["coordinates"][0], dtype=float)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        buildings[fid] = {
            "fid": fid,
            "floor": int(props.get("floor", 0)),
            "height_m": float(props.get("height_m", 0.0)),
            "base_height_m": float(props.get("base_height_m", 0.0)),
            "top_height_m": float(props.get("top_height_m", 0.0)),
            "ring_lonlat": ring[:, :2],
        }
    return buildings


def read_velocity_by_pixel() -> dict[tuple[int, int], float]:
    if not DEFORMATION_RATE.exists():
        return {}
    grouped: dict[tuple[int, int], list[float]] = {}
    with DEFORMATION_RATE.open(encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                pixel_row = int(round(float(parts[0])))
                pixel_col = int(round(float(parts[1])))
                velocity = float(parts[5])
            except ValueError:
                continue
            grouped.setdefault((pixel_col - 1, pixel_row - 1), []).append(velocity)
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def load_full_area_points(
    buildings: dict[int, dict], velocity_by_pixel: dict[tuple[int, int], float]
) -> list[dict]:
    rows: list[dict] = []
    with POINTS_CSV.open(newline="", encoding="utf-8") as fh:
        for source in csv.DictReader(fh):
            try:
                fid = int(float(source["fid"]))
                sar_row = int(round(float(source["row"])))
                sar_col = int(round(float(source["col"])))
                lon = float(source["method_lon"])
                lat = float(source["method_lat"])
                absolute_height = float(source["method_height_m"])
            except (KeyError, TypeError, ValueError):
                continue
            building = buildings.get(fid)
            velocity_text = source.get("deformation_rate_mm_yr", "")
            velocity = float(velocity_text) if velocity_text else velocity_by_pixel.get((sar_row, sar_col))
            if building is None or velocity is None:
                continue
            rows.append(
                {
                    "fid": fid,
                    "lon": lon,
                    "lat": lat,
                    "z": absolute_height - float(building["base_height_m"]),
                    "velocity": velocity,
                }
            )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings = load_full_area_buildings()
    rows = load_full_area_points(buildings, read_velocity_by_pixel())
    velocity = np.asarray([row["velocity"] for row in rows], dtype=float)
    counts = np.histogram(np.clip(velocity, CLASS_BOUNDS[0], CLASS_BOUNDS[-1]), bins=CLASS_BOUNDS)[0]

    rings = np.vstack([building["ring_lonlat"] for building in buildings.values()])
    lon_span = float(np.ptp(rings[:, 0]))
    lat_span = float(np.ptp(rings[:, 1]))

    fig = plt.figure(figsize=(10.6, 8.2), dpi=300)
    ax = fig.add_axes([0.015, 0.18, 0.97, 0.76], projection="3d")
    scatter = plot_scene(
        ax,
        buildings,
        rows,
        set(),
        detail=False,
        show_guides=False,
        point_size=POINT_SIZE,
        point_cmap=CLASS_CMAP,
        point_norm=CLASS_NORM,
        building_alpha=BUILDING_ALPHA,
        point_edgecolor="#37474f",
        point_linewidth=0.08,
    )
    ax.set_box_aspect((1.18, 1.0, 0.20), zoom=1.48)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM, roll=VIEW_ROLL)

    fig.text(
        0.025,
        0.965,
        f"{AREA_LABEL}全区域 Geo-BC 形变等级",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    fig.text(
        0.025,
        0.925,
        "透明三维建筑模型与同像素形变速率",
        fontsize=7.5,
        color="#555b61",
        va="top",
    )
    fig.text(
        0.975,
        0.165,
        f"{len(buildings)} 栋建筑｜{len(rows)} 个点｜标记面积 {POINT_SIZE:g} 平方磅｜建筑透明度 {BUILDING_ALPHA:.2f}",
        fontsize=6.4,
        color="#666b70",
        ha="right",
    )
    add_discrete_colorbar(fig, scatter, counts.astype(int).tolist())
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    print(f"buildings={len(buildings)}")
    print(f"points={len(rows)}")
    print(f"class_counts={counts.astype(int).tolist()}")
    print(f"building_extent_lonlat=({rings[:, 0].min()}, {rings[:, 1].min()}, {rings[:, 0].max()}, {rings[:, 1].max()})")
    print(f"building_span_deg=({lon_span}, {lat_span})")
    print(f"view=elev_{VIEW_ELEV}_azim_{VIEW_AZIM}_roll_{VIEW_ROLL}")
    print(f"output={OUT_PNG}")


if __name__ == "__main__":
    main()
