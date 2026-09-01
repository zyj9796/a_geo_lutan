from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import proj3d

from export_reference_area_3d_white_model import (
    BUILDINGS_GEOJSON,
    POINTS_CSV,
    PS_DEFORMATION_RATE,
    REF_BOUNDS,
    building_intersects_bounds,
    load_buildings,
    read_points,
    read_ps_velocity_by_pixel,
    reference_method_rows,
    select_fids,
)
from make_publication_3d_geobc import (
    OUT_DIR,
    TEXT,
    add_colorbar,
    plot_scene,
    point_arrays,
)


ANGLE_DIR = OUT_DIR / "angle_search"
SCORES_CSV = ANGLE_DIR / "angle_visibility_scores.csv"
BEST_PNG = OUT_DIR / "fig_04_geobc_3d_best_visibility.png"
TOP_ANGLES_PNG = OUT_DIR / "fig_05_geobc_3d_top_angles.png"
SEARCH_ELEVATIONS = range(15, 76, 10)
SEARCH_AZIMUTHS = range(0, 360, 10)
CELL_SIZE_PX = 3.5


def prepare_data() -> tuple[dict[int, dict], list[dict], set[int]]:
    buildings_all = load_buildings(BUILDINGS_GEOJSON)
    velocity_by_pixel = read_ps_velocity_by_pixel(PS_DEFORMATION_RATE)
    source_rows = read_points(POINTS_CSV, velocity_by_pixel)
    selected_fids = set(select_fids(source_rows, buildings_all))
    rows = reference_method_rows(source_rows, buildings_all, list(selected_fids))
    buildings = {
        fid: building
        for fid, building in buildings_all.items()
        if building_intersects_bounds(building, REF_BOUNDS)
    }
    return buildings, rows, selected_fids


def scene_geometry(buildings: dict[int, dict], rows: list[dict]) -> tuple[np.ndarray, ...]:
    rings = np.vstack([building["ring_lonlat"] for building in buildings.values()])
    lon0 = float(np.mean(rings[:, 0]))
    lat0 = float(np.mean(rings[:, 1]))
    south, west, height, _ = point_arrays(rows, lon0, lat0)

    from export_reference_area_3d_white_model import local_en

    east_all, north_all = local_en(rings[:, 0], rings[:, 1], lon0, lat0)
    south_all, west_all = -north_all, -east_all
    pad_x = max(float(np.ptp(south_all)) * 0.035, 12.0)
    pad_y = max(float(np.ptp(west_all)) * 0.055, 12.0)
    xlim = (float(np.min(south_all) - pad_x), float(np.max(south_all) + pad_x))
    ylim = (float(np.min(west_all) - pad_y), float(np.max(west_all) + pad_y))
    zlim = (0.0, max(float(building["height_m"]) for building in buildings.values()) + 14.0)
    return south, west, height, xlim, ylim, zlim


def projected_visibility_scores(
    buildings: dict[int, dict],
    rows: list[dict],
    box_aspect: tuple[float, float, float] = (2.55, 0.68, 0.34),
    cell_size_px: float = CELL_SIZE_PX,
) -> list[dict[str, float | int]]:
    south, west, height, xlim, ylim, zlim = scene_geometry(buildings, rows)
    fig = plt.figure(figsize=(6.4, 4.0), dpi=100)
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.96], projection="3d")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(*zlim)
    ax.set_box_aspect(box_aspect, zoom=1.0)
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    fig.canvas.draw()

    results: list[dict[str, float | int]] = []
    for elev in SEARCH_ELEVATIONS:
        for azim in SEARCH_AZIMUTHS:
            ax.view_init(elev=elev, azim=azim, roll=0)
            xp, yp, _ = proj3d.proj_transform(south, west, height, ax.get_proj())
            pixels = ax.transData.transform(np.column_stack([xp, yp]))
            inside = (
                (pixels[:, 0] >= ax.bbox.x0)
                & (pixels[:, 0] <= ax.bbox.x1)
                & (pixels[:, 1] >= ax.bbox.y0)
                & (pixels[:, 1] <= ax.bbox.y1)
            )
            px = pixels[inside]
            cells = np.floor((px - np.asarray([ax.bbox.x0, ax.bbox.y0])) / cell_size_px).astype(int)
            unique_cells = np.unique(cells, axis=0)
            spread_x = float(np.ptp(px[:, 0]))
            spread_y = float(np.ptp(px[:, 1]))
            results.append(
                {
                    "elev_deg": elev,
                    "azim_deg": azim,
                    "visible_proxy": int(len(unique_cells)),
                    "visible_fraction": float(len(unique_cells) / len(rows)),
                    "projected_width_px": spread_x,
                    "projected_height_px": spread_y,
                }
            )
    plt.close(fig)
    return sorted(
        results,
        key=lambda item: (
            int(item["visible_proxy"]),
            float(item["projected_width_px"]) * float(item["projected_height_px"]),
        ),
        reverse=True,
    )


def write_scores(rows: list[dict[str, float | int]]) -> None:
    with SCORES_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def view_direction(azim: int) -> str:
    labels = ["南侧", "西南侧", "西侧", "西北侧", "北侧", "东北侧", "东侧", "东南侧"]
    return labels[int(((azim + 22.5) % 360) // 45)]


def render_view(
    path: Path,
    buildings: dict[int, dict],
    rows: list[dict],
    selected_fids: set[int],
    score: dict[str, float | int],
) -> None:
    elev = int(score["elev_deg"])
    azim = int(score["azim_deg"])
    fig = plt.figure(figsize=(9.2, 5.7), dpi=300)
    ax = fig.add_axes([0.01, 0.13, 0.98, 0.84], projection="3d")
    scatter = plot_scene(ax, buildings, rows, selected_fids, detail=False, show_guides=False)
    ax.set_box_aspect((2.55, 0.68, 0.34), zoom=1.05)
    ax.view_init(elev=elev, azim=azim, roll=0)
    ax.text2D(0.018, 0.97, "点分离度最大视角", transform=ax.transAxes, fontsize=11, fontweight="bold", color=TEXT, va="top")
    ax.text2D(
        0.018,
        0.915,
        f"View from {view_direction(azim)} | elevation {elev} deg | azimuth {azim} deg",
        transform=ax.transAxes,
        fontsize=7.5,
        color="#555b61",
        va="top",
    )
    ax.text2D(
        0.98,
        0.025,
        f"Distinct projected point cells: {int(score['visible_proxy'])}/{len(rows)}",
        transform=ax.transAxes,
        fontsize=6.5,
        color="#666b70",
        ha="right",
    )
    add_colorbar(fig, scatter, [0.32, 0.055, 0.36, 0.017])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def render_top_angles(
    path: Path,
    buildings: dict[int, dict],
    rows: list[dict],
    selected_fids: set[int],
    top_scores: list[dict[str, float | int]],
) -> None:
    fig = plt.figure(figsize=(13.0, 8.2), dpi=240)
    for index, score in enumerate(top_scores, start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        plot_scene(ax, buildings, rows, selected_fids, detail=False, show_guides=False)
        ax.set_box_aspect((2.55, 0.68, 0.34), zoom=0.86)
        elev = int(score["elev_deg"])
        azim = int(score["azim_deg"])
        ax.view_init(elev=elev, azim=azim, roll=0)
        ax.text2D(
            0.02,
            0.98,
            f"#{index}  elev {elev} deg, azim {azim} deg",
            transform=ax.transAxes,
            fontsize=8,
            fontweight="bold",
            va="top",
        )
        ax.text2D(
            0.02,
            0.92,
            f"{view_direction(azim)} | score {int(score['visible_proxy'])}",
            transform=ax.transAxes,
            fontsize=6.5,
            color="#555b61",
            va="top",
        )
    fig.suptitle("按投影点分离度排序的前六个视角", fontsize=13, fontweight="bold", y=0.985)
    fig.text(
        0.5,
        0.018,
        f"Search: {len(SEARCH_ELEVATIONS) * len(SEARCH_AZIMUTHS)} angles; {len(rows)} Geo-BC points; {CELL_SIZE_PX:g}-pixel occupancy cells",
        fontsize=7,
        color="#666b70",
        ha="center",
    )
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.045, top=0.94, wspace=0.01, hspace=0.02)
    fig.savefig(path, dpi=240)
    plt.close(fig)


def main() -> None:
    ANGLE_DIR.mkdir(parents=True, exist_ok=True)
    buildings, rows, selected_fids = prepare_data()
    scores = projected_visibility_scores(buildings, rows)
    write_scores(scores)
    render_view(BEST_PNG, buildings, rows, selected_fids, scores[0])
    render_top_angles(TOP_ANGLES_PNG, buildings, rows, selected_fids, scores[:6])
    for rank, score in enumerate(scores[:6], start=1):
        path = ANGLE_DIR / (
            f"rank_{rank:02d}_elev_{int(score['elev_deg']):02d}_azim_{int(score['azim_deg']):03d}.png"
        )
        render_view(path, buildings, rows, selected_fids, score)
    best = scores[0]
    print(f"tested_angles={len(scores)}")
    print(f"best_elev_deg={best['elev_deg']}")
    print(f"best_azim_deg={best['azim_deg']}")
    print(f"best_direction={view_direction(int(best['azim_deg']))}")
    print(f"visible_proxy={best['visible_proxy']}/{len(rows)}")
    print(f"scores={SCORES_CSV}")
    print(f"best_figure={BEST_PNG}")
    print(f"top_angles_figure={TOP_ANGLES_PNG}")


if __name__ == "__main__":
    main()
