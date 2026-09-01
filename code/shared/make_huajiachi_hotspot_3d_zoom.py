from __future__ import annotations

import math
import os

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
from make_huajiachi_full_area_geobc_3d import (
    OUT_DIR,
    load_full_area_buildings,
    load_full_area_points,
    read_velocity_by_pixel,
)
from make_publication_3d_geobc import TEXT, plot_scene
from make_reference_area_selected_building_deformation import local_en
from search_geobc_3d_view_angles import projected_visibility_scores, view_direction


AREA_LABEL = os.environ.get("SAR_GEOCODE_AREA_LABEL", "华家池")
OUT_PNG = OUT_DIR / f"fig_11_{os.environ.get('SAR_GEOCODE_FIGURE_PREFIX', 'huajiachi')}_hotspot_3d_zoom.png"
WINDOW_M = 450.0
GRID_STEP_M = 100.0
POINT_SIZE = 3.0
BUILDING_ALPHA = 0.16
REF_BOUNDS = (120.1850, 30.2625, 120.1900, 30.2825)
LOCAL_BOX_ASPECT = (1.05, 1.0, 0.36)


def select_hotspot(
    buildings: dict[int, dict], rows: list[dict]
) -> tuple[dict[int, dict], list[dict], tuple[float, float, float, float], dict[str, float | int]]:
    rings = np.vstack([building["ring_lonlat"] for building in buildings.values()])
    lon0 = float(np.mean(rings[:, 0]))
    lat0 = float(np.mean(rings[:, 1]))
    point_lon = np.asarray([row["lon"] for row in rows], dtype=float)
    point_lat = np.asarray([row["lat"] for row in rows], dtype=float)
    velocity = np.asarray([row["velocity"] for row in rows], dtype=float)
    east, north = local_en(point_lon, point_lat, lon0, lat0)
    half = WINDOW_M / 2.0

    candidates = []
    for cx in np.arange(float(np.min(east) + half), float(np.max(east) - half + 1), GRID_STEP_M):
        for cy in np.arange(float(np.min(north) + half), float(np.max(north) - half + 1), GRID_STEP_M):
            center_lon = lon0 + cx / (6378137.0 * math.cos(math.radians(lat0))) * 180.0 / math.pi
            center_lat = lat0 + cy / 6378137.0 * 180.0 / math.pi
            if REF_BOUNDS[0] <= center_lon <= REF_BOUNDS[2] and REF_BOUNDS[1] <= center_lat <= REF_BOUNDS[3]:
                continue
            mask = (np.abs(east - cx) <= half) & (np.abs(north - cy) <= half)
            count = int(np.count_nonzero(mask))
            if count < 150:
                continue
            local_velocity = velocity[mask]
            extreme = int(np.count_nonzero(np.abs(local_velocity) > 5.0))
            nonstable = int(np.count_nonzero(np.abs(local_velocity) > 2.0))
            classes = int(
                np.unique(np.digitize(np.clip(local_velocity, -30, 30), [-10, -5, -2, 2, 5, 10])).size
            )
            score = 2.0 * extreme + nonstable + 0.15 * count + 12.0 * classes
            candidates.append((score, cx, cy, center_lon, center_lat, count, extreme, nonstable, classes))
    if not candidates:
        raise RuntimeError("No hotspot candidate satisfied the selection constraints")
    best = max(candidates)
    _, cx, cy, center_lon, center_lat, count, extreme, nonstable, classes = best
    en_bounds = (cx - half, cy - half, cx + half, cy + half)

    selected_rows = [
        row
        for row, ex, ny in zip(rows, east, north, strict=True)
        if en_bounds[0] <= ex <= en_bounds[2] and en_bounds[1] <= ny <= en_bounds[3]
    ]
    selected_buildings: dict[int, dict] = {}
    for fid, building in buildings.items():
        ring = building["ring_lonlat"]
        be, bn = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
        if (
            float(np.max(be)) >= en_bounds[0]
            and float(np.min(be)) <= en_bounds[2]
            and float(np.max(bn)) >= en_bounds[1]
            and float(np.min(bn)) <= en_bounds[3]
        ):
            selected_buildings[fid] = building

    lon_bounds = (
        lon0 + en_bounds[0] / (6378137.0 * math.cos(math.radians(lat0))) * 180.0 / math.pi,
        lat0 + en_bounds[1] / 6378137.0 * 180.0 / math.pi,
        lon0 + en_bounds[2] / (6378137.0 * math.cos(math.radians(lat0))) * 180.0 / math.pi,
        lat0 + en_bounds[3] / 6378137.0 * 180.0 / math.pi,
    )
    stats = {
        "score": float(best[0]),
        "center_lon": float(center_lon),
        "center_lat": float(center_lat),
        "points": int(count),
        "extreme_points": int(extreme),
        "nonstable_points": int(nonstable),
        "classes": int(classes),
    }
    return selected_buildings, selected_rows, lon_bounds, stats


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings_all = load_full_area_buildings()
    rows_all = load_full_area_points(buildings_all, read_velocity_by_pixel())
    buildings, rows, lonlat_bounds, stats = select_hotspot(buildings_all, rows_all)
    velocity = np.asarray([row["velocity"] for row in rows], dtype=float)
    counts = np.histogram(np.clip(velocity, CLASS_BOUNDS[0], CLASS_BOUNDS[-1]), bins=CLASS_BOUNDS)[0]

    scores = projected_visibility_scores(
        buildings,
        rows,
        box_aspect=LOCAL_BOX_ASPECT,
        cell_size_px=2.5,
    )
    view = max(
        (score for score in scores if 25 <= int(score["elev_deg"]) <= 55),
        key=lambda score: int(score["visible_proxy"]),
    )
    elev = int(view["elev_deg"])
    azim = int(view["azim_deg"])

    fig = plt.figure(figsize=(9.2, 7.4), dpi=300)
    ax = fig.add_axes([0.015, 0.17, 0.97, 0.77], projection="3d")
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
        point_edgecolor="#263238",
        point_linewidth=0.16,
    )
    ax.set_box_aspect(LOCAL_BOX_ASPECT, zoom=1.30)
    ax.view_init(elev=elev, azim=azim, roll=0)
    fig.text(
        0.025,
        0.965,
        f"{AREA_LABEL}选定形变热点：三维放大图",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    fig.text(
        0.025,
        0.925,
        f"450 米窗口，中心为东经 {stats['center_lon']:.5f}°、北纬 {stats['center_lat']:.5f}°｜从{view_direction(azim)}观察",
        fontsize=7.5,
        color="#555b61",
        va="top",
    )
    fig.text(
        0.975,
        0.155,
        f"{len(buildings)} 栋建筑｜{len(rows)} 个点｜|形变速率| > 5 毫米/年：{stats['extreme_points']} 个｜标记面积 {POINT_SIZE:g} 平方磅",
        fontsize=6.4,
        color="#666b70",
        ha="right",
    )
    add_discrete_colorbar(fig, scatter, counts.astype(int).tolist())
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    print(f"bounds_lonlat={lonlat_bounds}")
    print(f"selection_stats={stats}")
    print(f"buildings={len(buildings)}")
    print(f"points={len(rows)}")
    print(f"class_counts={counts.astype(int).tolist()}")
    print(f"view=elev_{elev}_azim_{azim}_{view_direction(azim)}")
    print(f"visible_proxy={view['visible_proxy']}/{len(rows)}")
    print(f"output={OUT_PNG}")


if __name__ == "__main__":
    main()
