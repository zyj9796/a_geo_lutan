from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from export_reference_area_3d_white_model import (
    BUILDINGS_GEOJSON,
    DEFORMATION_CMAP,
    DEFORMATION_NORM,
    FOCUS_BOUNDS,
    POINTS_CSV,
    PS_DEFORMATION_RATE,
    REF_BOUNDS,
    building_intersects_bounds,
    building_intersects_focus,
    extruded_faces,
    load_buildings,
    local_en,
    read_points,
    read_ps_velocity_by_pixel,
    reference_method_rows,
    select_fids,
)


PROJECT_DIR = Path(os.environ.get("SAR_GEOCODE_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
OUT_DIR = PROJECT_DIR / "results" / "pic_all"
OVERVIEW_PNG = OUT_DIR / "fig_01_geobc_3d_overview.png"
DETAIL_PNG = OUT_DIR / "fig_02_geobc_3d_settlement_detail.png"
COMPOSITE_PNG = OUT_DIR / "fig_03_geobc_3d_overview_detail.png"
COMPOSITE_PDF = OUT_DIR / "fig_03_geobc_3d_overview_detail.pdf"

TEXT = "#202124"
EDGE = "#92979c"
SELECTED_EDGE = "#25282b"
GROUND = "#f1f0ec"
BUILDING = "#f8f8f6"
SELECTED_BUILDING = "#fff1c7"

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans", "sans-serif"],
        "axes.unicode_minus": False,
        "font.size": 8,
        "axes.linewidth": 0.7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def to_south_west(face: np.ndarray) -> np.ndarray:
    """Convert local East/North geometry to South/West display coordinates."""
    out = np.asarray(face, dtype=float).copy()
    east = out[:, 0].copy()
    north = out[:, 1].copy()
    out[:, 0] = -north
    out[:, 1] = -east
    return out


def collect_faces(
    buildings: dict[int, dict], lon0: float, lat0: float, selected: set[int]
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    roofs, walls, selected_roofs, selected_walls = [], [], [], []
    for fid, building in buildings.items():
        faces = [to_south_west(face) for face in extruded_faces(building, lon0, lat0)]
        if fid in selected:
            selected_roofs.append(faces[0])
            selected_walls.extend(faces[2:])
        else:
            roofs.append(faces[0])
            walls.extend(faces[2:])
    return roofs, walls, selected_roofs, selected_walls


def point_arrays(rows: list[dict], lon0: float, lat0: float) -> tuple[np.ndarray, ...]:
    lon = np.asarray([row["lon"] for row in rows], dtype=float)
    lat = np.asarray([row["lat"] for row in rows], dtype=float)
    east, north = local_en(lon, lat, lon0, lat0)
    south = -north
    west = -east
    height = np.asarray([row["z"] for row in rows], dtype=float) + 0.55
    velocity = np.asarray([row["velocity"] for row in rows], dtype=float)
    return south, west, height, velocity


def add_collection(
    ax,
    faces: list[np.ndarray],
    facecolor: str,
    edgecolor: str,
    linewidth: float,
    alpha: float,
) -> None:
    if not faces:
        return
    collection = Poly3DCollection(
        faces,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        rasterized=True,
    )
    ax.add_collection3d(collection)


def add_ground(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    xmin, xmax = xlim
    ymin, ymax = ylim
    ground = np.asarray(
        [[xmin, ymin, 0], [xmax, ymin, 0], [xmax, ymax, 0], [xmin, ymax, 0]],
        dtype=float,
    )
    add_collection(ax, [ground], GROUND, "none", 0, 1.0)


def add_site_boundary(ax, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    xmin, xmax = xlim
    ymin, ymax = ylim
    ax.plot(
        [xmin, xmax, xmax, xmin, xmin],
        [ymin, ymin, ymax, ymax, ymin],
        [-0.8] * 5,
        color="#d6d3cc",
        linewidth=0.55,
        zorder=0,
    )


def add_orientation(ax, xlim: tuple[float, float], ylim: tuple[float, float], scale: float) -> None:
    x0 = xlim[0] + 0.09 * (xlim[1] - xlim[0])
    y0 = ylim[0] + 0.10 * (ylim[1] - ylim[0])
    z0 = 1.0
    ax.quiver(x0, y0, z0, scale, 0, 0, color=TEXT, linewidth=0.8, arrow_length_ratio=0.13)
    ax.quiver(x0, y0, z0, 0, scale, 0, color=TEXT, linewidth=0.8, arrow_length_ratio=0.13)
    ax.quiver(x0, y0, z0, 0, 0, scale * 0.45, color=TEXT, linewidth=0.8, arrow_length_ratio=0.22)
    ax.text(x0 + scale * 1.08, y0, z0, "南", color=TEXT, fontsize=7, ha="center", va="center")
    ax.text(x0, y0 + scale * 1.08, z0, "西", color=TEXT, fontsize=7, ha="center", va="center")
    ax.text(x0, y0, z0 + scale * 0.52, "高度", color=TEXT, fontsize=6.5, ha="center", va="bottom")


def add_scale_bar(ax, xlim: tuple[float, float], ylim: tuple[float, float], length: float) -> None:
    x0 = xlim[1] - 0.10 * (xlim[1] - xlim[0]) - length
    y0 = ylim[0] + 0.08 * (ylim[1] - ylim[0])
    ax.plot([x0, x0 + length], [y0, y0], [0.8, 0.8], color=TEXT, lw=1.5, solid_capstyle="butt")
    ax.text(x0 + length / 2, y0, 2.2, f"{int(length)} m", fontsize=6.5, color=TEXT, ha="center", va="bottom")


def style_3d_axis(ax) -> None:
    ax.set_proj_type("ortho")
    ax.set_axis_off()
    ax.set_facecolor("white")
    ax.grid(False)


def plot_scene(
    ax,
    buildings: dict[int, dict],
    rows: list[dict],
    selected_fids: set[int],
    *,
    detail: bool,
    show_guides: bool = True,
    point_size: float | None = None,
    point_cmap=None,
    point_norm=None,
    building_alpha: float | None = None,
    point_edgecolor: str | None = None,
    point_linewidth: float | None = None,
) -> ScalarMappable:
    rings = np.vstack([building["ring_lonlat"] for building in buildings.values()])
    lon0 = float(np.mean(rings[:, 0]))
    lat0 = float(np.mean(rings[:, 1]))
    roofs, walls, selected_roofs, selected_walls = collect_faces(buildings, lon0, lat0, selected_fids)

    east, north = local_en(rings[:, 0], rings[:, 1], lon0, lat0)
    south_all, west_all = -north, -east
    pad_x = max(float(np.ptp(south_all)) * (0.055 if detail else 0.035), 12.0)
    pad_y = max(float(np.ptp(west_all)) * (0.075 if detail else 0.055), 12.0)
    xlim = (float(np.min(south_all) - pad_x), float(np.max(south_all) + pad_x))
    ylim = (float(np.min(west_all) - pad_y), float(np.max(west_all) + pad_y))
    max_height = max(float(building["height_m"]) for building in buildings.values())

    if detail:
        add_ground(ax, xlim, ylim)
    else:
        add_site_boundary(ax, xlim, ylim)
    wall_alpha = building_alpha if building_alpha is not None else 0.97
    roof_alpha = min(1.0, wall_alpha + 0.08) if building_alpha is not None else 1.0
    selected_wall_alpha = min(1.0, wall_alpha + 0.13) if building_alpha is not None else 1.0
    selected_roof_alpha = min(1.0, roof_alpha + 0.13) if building_alpha is not None else 1.0
    add_collection(ax, walls, BUILDING, EDGE, 0.12 if detail else 0.08, wall_alpha)
    add_collection(ax, roofs, "#ffffff", EDGE, 0.16 if detail else 0.10, roof_alpha)
    add_collection(ax, selected_walls, SELECTED_BUILDING, SELECTED_EDGE, 0.34 if detail else 0.22, selected_wall_alpha)
    add_collection(ax, selected_roofs, "#ffe5a0", SELECTED_EDGE, 0.42 if detail else 0.28, selected_roof_alpha)

    south, west, height, velocity = point_arrays(rows, lon0, lat0)
    marker_size = point_size if point_size is not None else (13 if detail else 5.2)
    scatter = ax.scatter(
        south,
        west,
        height,
        c=velocity,
        cmap=point_cmap if point_cmap is not None else DEFORMATION_CMAP,
        norm=point_norm if point_norm is not None else DEFORMATION_NORM,
        s=marker_size,
        alpha=0.98,
        depthshade=False,
        linewidths=(
            point_linewidth
            if point_linewidth is not None
            else (0.08 if detail and marker_size < 8 else (0.16 if detail else 0))
        ),
        edgecolors=(
            point_edgecolor
            if point_edgecolor is not None
            else ("#303030" if detail and marker_size >= 4 else "none")
        ),
        rasterized=True,
        zorder=8,
    )

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(0, max_height + (10 if detail else 14))
    if detail:
        ax.set_box_aspect((1.55, 1.0, 0.78), zoom=1.16)
        ax.view_init(elev=27, azim=35, roll=0)
        if show_guides:
            add_orientation(ax, xlim, ylim, 24)
            add_scale_bar(ax, xlim, ylim, 50)
    else:
        ax.set_box_aspect((2.55, 0.68, 0.34), zoom=1.28)
        ax.view_init(elev=29, azim=70, roll=0)
        if show_guides:
            add_orientation(ax, xlim, ylim, 135)
            add_scale_bar(ax, xlim, ylim, 500)
            selected_rings = np.vstack(
                [buildings[fid]["ring_lonlat"] for fid in selected_fids if fid in buildings]
            )
            selected_e, selected_n = local_en(selected_rings[:, 0], selected_rings[:, 1], lon0, lat0)
            sx, wy = -selected_n, -selected_e
            bx = (float(np.min(sx) - 25), float(np.max(sx) + 25))
            by = (float(np.min(wy) - 20), float(np.max(wy) + 20))
            ax.plot(
                [bx[0], bx[1], bx[1], bx[0], bx[0]],
                [by[0], by[0], by[1], by[1], by[0]],
                [2.0] * 5,
                color=TEXT,
                lw=0.9,
                zorder=12,
            )
            ax.text(bx[1], by[1], 4.0, "详图 b", fontsize=7, color=TEXT, ha="left", va="bottom")
    style_3d_axis(ax)
    return scatter


def add_colorbar(fig, mappable, rect: list[float]) -> None:
    cax = fig.add_axes(rect)
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    cbar.set_ticks([-30, -20, -10, 0, 10, 20, 30])
    cbar.set_label(r"形变速率（毫米·年$^{-1}$）", fontsize=7, labelpad=3)
    cbar.ax.tick_params(labelsize=6.5, length=2.5, pad=2)
    cbar.outline.set_linewidth(0.6)
    cbar.ax.text(0.0, 1.75, "沉降", transform=cbar.ax.transAxes, fontsize=6.5, color="#b40d18", ha="left")
    cbar.ax.text(0.5, 1.75, "接近零", transform=cbar.ax.transAxes, fontsize=6.5, color="#138a36", ha="center")
    cbar.ax.text(1.0, 1.75, "正形变", transform=cbar.ax.transAxes, fontsize=6.5, color="#08519c", ha="right")


def save_single(
    path: Path,
    buildings: dict[int, dict],
    rows: list[dict],
    selected_fids: set[int],
    *,
    detail: bool,
) -> None:
    fig = plt.figure(figsize=(8.0, 5.2), dpi=300)
    ax = fig.add_axes([0.01, 0.11, 0.98, 0.86], projection="3d")
    scatter = plot_scene(ax, buildings, rows, selected_fids, detail=detail)
    ax.text2D(0.02, 0.965, "沉降建筑簇详图" if detail else "参考区域概览", transform=ax.transAxes, fontsize=10, fontweight="bold", color=TEXT, va="top")
    ax.text2D(0.02, 0.915, "三维建筑模型上的 Geo-BC 点", transform=ax.transAxes, fontsize=7.5, color="#555b61", va="top")
    ax.text2D(0.98, 0.025, "垂直夸张约 3 倍", transform=ax.transAxes, fontsize=6.2, color="#666b70", ha="right")
    add_colorbar(fig, scatter, [0.29, 0.045, 0.42, 0.018])
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_composite(
    buildings: dict[int, dict],
    overview_rows: list[dict],
    detail_buildings: dict[int, dict],
    detail_rows: list[dict],
    selected_fids: set[int],
) -> None:
    fig = plt.figure(figsize=(12.0, 6.5), dpi=300)
    ax_main = fig.add_axes([0.015, 0.13, 0.66, 0.82], projection="3d")
    ax_detail = fig.add_axes([0.68, 0.20, 0.305, 0.68], projection="3d")
    scatter = plot_scene(ax_main, buildings, overview_rows, selected_fids, detail=False)
    plot_scene(ax_detail, detail_buildings, detail_rows, selected_fids, detail=True)

    ax_main.text2D(0.015, 0.97, "a", transform=ax_main.transAxes, fontsize=11, fontweight="bold", va="top")
    ax_main.text2D(0.065, 0.97, "参考区域概览", transform=ax_main.transAxes, fontsize=10, fontweight="bold", color=TEXT, va="top")
    ax_detail.text2D(0.00, 1.00, "b", transform=ax_detail.transAxes, fontsize=11, fontweight="bold", va="top")
    ax_detail.text2D(0.12, 1.00, "沉降建筑簇详图", transform=ax_detail.transAxes, fontsize=9, fontweight="bold", color=TEXT, va="top")
    fig.text(0.022, 0.955, "约束于三维建筑模型的 Geo-BC 形变点", fontsize=12, fontweight="bold", color=TEXT, va="top")
    fig.text(0.985, 0.095, "正交投影；垂直夸张约 3 倍", fontsize=6.5, color="#666b70", ha="right")
    add_colorbar(fig, scatter, [0.36, 0.065, 0.30, 0.014])
    fig.savefig(COMPOSITE_PNG, dpi=300)
    fig.savefig(COMPOSITE_PDF, dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings_all = load_buildings(BUILDINGS_GEOJSON)
    velocity_by_pixel = read_ps_velocity_by_pixel(PS_DEFORMATION_RATE)
    source_rows = read_points(POINTS_CSV, velocity_by_pixel)
    selected_fids = set(select_fids(source_rows, buildings_all))
    overview_rows = reference_method_rows(source_rows, buildings_all, list(selected_fids))
    overview_buildings = {
        fid: building
        for fid, building in buildings_all.items()
        if building_intersects_bounds(building, REF_BOUNDS)
    }
    detail_buildings = {
        fid: building for fid, building in buildings_all.items() if building_intersects_focus(building)
    }
    detail_rows = [
        row
        for row in overview_rows
        if FOCUS_BOUNDS[0] <= row["lon"] <= FOCUS_BOUNDS[2]
        and FOCUS_BOUNDS[1] <= row["lat"] <= FOCUS_BOUNDS[3]
    ]

    save_single(OVERVIEW_PNG, overview_buildings, overview_rows, selected_fids, detail=False)
    save_single(DETAIL_PNG, detail_buildings, detail_rows, selected_fids, detail=True)
    save_composite(overview_buildings, overview_rows, detail_buildings, detail_rows, selected_fids)
    print(f"overview={OVERVIEW_PNG}")
    print(f"detail={DETAIL_PNG}")
    print(f"composite_png={COMPOSITE_PNG}")
    print(f"composite_pdf={COMPOSITE_PDF}")
    print(f"overview_buildings={len(overview_buildings)} overview_points={len(overview_rows)}")
    print(f"detail_buildings={len(detail_buildings)} detail_points={len(detail_rows)}")


if __name__ == "__main__":
    main()
