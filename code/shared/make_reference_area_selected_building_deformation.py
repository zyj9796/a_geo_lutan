from __future__ import annotations

import csv
import json
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
from matplotlib import colors
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from io_paths import FULL_AREA_GEOJSON_DIR, IMAGE_DIR, PIC_ALL_DIR, TABLE_ROOT


DATE = "20250109"
POINTS_CSV = TABLE_ROOT / "full_area" / f"{DATE}_all_buildings_method_vs_gamma_points.csv"
BUILDINGS_GEOJSON = FULL_AREA_GEOJSON_DIR / f"{DATE}_all_valid_geocoded_buildings.geojson"
PS_DEFORMATION_RATE = Path("/home/u/geocoding/geo_hangzhou/Hangzhou_Huajiachi/postprocessing/defo_rate")

OUT_IMAGES = IMAGE_DIR / "full_area_geobc_ps" / "defo"
OUT_PIC_ALL = PIC_ALL_DIR

REF_BOUNDS = (120.1850, 30.2625, 120.1900, 30.2825)
FOCUS_BOUNDS = (120.1857, 30.2680, 120.1869, 30.2703)
EARTH_R = 6378137.0
SELECT_COUNT = 8
MIN_POINTS_PER_BUILDING = 24
SELECTED_RED_FIDS = [1494, 1487, 1534, 1486, 1485, 1484]
RANDOM_OTHER_FIDS = [1539, 1926, 1187, 1407, 1838, 1772]

DEFORMATION_BOUNDS = np.asarray([-30.0, -20.0, -6.0, -4.0, -2.0, 2.0, 4.0, 6.0, 10.0, 30.0], dtype=float)
DEFORMATION_TICKS = [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0]
DEFORMATION_CMAP = colors.LinearSegmentedColormap.from_list(
    "deformation_rate_continuous",
    [
        (0.0, "#8b0000"),
        (1.0 / 6.0, "#ff0000"),
        (0.40, "#ff1a1a"),
        (0.4667, "#42d915"),
        (0.5667, "#00c853"),
        (0.6667, "#00c9a7"),
        (1.0, "#08519c"),
    ],
    N=256,
)
DEFORMATION_NORM = colors.Normalize(vmin=float(DEFORMATION_BOUNDS[0]), vmax=float(DEFORMATION_BOUNDS[-1]), clip=True)


def local_en(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    east = (lon - lon0) * math.pi / 180.0 * EARTH_R * math.cos(math.radians(lat0))
    north = (lat - lat0) * math.pi / 180.0 * EARTH_R
    return east, north


def load_buildings(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, dict] = {}
    for feat in data.get("features", []):
        props = feat.get("properties", {})
        fid = int(props["fid"])
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        out[fid] = {
            "fid": fid,
            "floor": int(props.get("floor", 0)),
            "height_m": float(props.get("height_m", 0.0)),
            "base_height_m": float(props.get("base_height_m", 0.0)),
            "top_height_m": float(props.get("top_height_m", 0.0)),
            "ring_lonlat": ring[:, :2],
        }
    return out


def read_ps_velocity_by_pixel(path: Path) -> dict[tuple[int, int], list[float]]:
    by_pixel: dict[tuple[int, int], list[float]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                ps_row = int(round(float(parts[0])))
                ps_col = int(round(float(parts[1])))
                velocity = float(parts[5])
            except ValueError:
                continue
            by_pixel.setdefault((ps_col - 1, ps_row - 1), []).append(velocity)
    return by_pixel


def in_reference(lon: float, lat: float) -> bool:
    minx, miny, maxx, maxy = REF_BOUNDS
    return minx <= lon <= maxx and miny <= lat <= maxy


def read_points(path: Path, velocity_by_pixel: dict[tuple[int, int], list[float]]) -> list[dict]:
    mean_velocity = {key: float(np.mean(values)) for key, values in velocity_by_pixel.items()}
    rows: list[dict] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                sar_row = int(round(float(row["row"])))
                sar_col = int(round(float(row["col"])))
                method_lon = float(row["method_lon"])
                method_lat = float(row["method_lat"])
                method_h = float(row["method_height_m"])
                gamma_lon = float(row["gamma_dsm_lon"])
                gamma_lat = float(row["gamma_dsm_lat"])
                gamma_h = float(row["gamma_dsm_height_m"])
                gamma_ok = int(float(row["gamma_dsm_ok"]))
                fid = int(float(row["fid"]))
            except (KeyError, TypeError, ValueError):
                continue
            velocity = mean_velocity.get((sar_row, sar_col))
            if velocity is None or gamma_ok != 1:
                continue
            if not (in_reference(method_lon, method_lat) or in_reference(gamma_lon, gamma_lat)):
                continue
            rows.append(
                {
                    "fid": fid,
                    "row": sar_row,
                    "col": sar_col,
                    "method_lon": method_lon,
                    "method_lat": method_lat,
                    "method_h": method_h,
                    "gamma_lon": gamma_lon,
                    "gamma_lat": gamma_lat,
                    "gamma_h": gamma_h,
                    "velocity": velocity,
                }
            )
    return rows


def select_fids(rows: list[dict], buildings: dict[int, dict]) -> list[int]:
    available = {int(row["fid"]) for row in rows}
    selected = [fid for fid in SELECTED_RED_FIDS if fid in buildings and fid in available]
    if selected:
        return selected

    grouped: dict[int, list[float]] = {}
    for row in rows:
        fid = int(row["fid"])
        if fid in buildings:
            grouped.setdefault(fid, []).append(float(row["velocity"]))
    candidates = []
    for fid, values in grouped.items():
        if len(values) < MIN_POINTS_PER_BUILDING:
            continue
        velocities = np.asarray(values, dtype=float)
        score = float(np.median(np.abs(velocities)))
        candidates.append((fid, score, len(values)))
    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [fid for fid, _, _ in candidates[:SELECT_COUNT]]


def draw_building_outlines(ax, buildings: dict[int, dict], fids: set[int] | None = None) -> None:
    for fid, building in buildings.items():
        ring = building["ring_lonlat"]
        if not (
            np.any((FOCUS_BOUNDS[0] <= ring[:, 0]) & (ring[:, 0] <= FOCUS_BOUNDS[2]))
            and np.any((FOCUS_BOUNDS[1] <= ring[:, 1]) & (ring[:, 1] <= FOCUS_BOUNDS[3]))
        ):
            continue
        is_selected = fids is not None and fid in fids
        ax.plot(
            ring[:, 0],
            ring[:, 1],
            color="#ffcc00" if is_selected else "#e8e8e8",
            linewidth=0.72 if is_selected else 0.22,
            alpha=0.96 if is_selected else 0.54,
            zorder=5 if is_selected else 2,
        )


def annotate_selected_buildings(ax, buildings: dict[int, dict], fids: list[int]) -> None:
    offsets = [(-13, 8), (12, 8), (-13, -8), (12, -8), (-16, 0), (15, 0), (0, 12), (0, -12)]
    for i, fid in enumerate(fids):
        building = buildings.get(fid)
        if not building:
            continue
        ring = building["ring_lonlat"]
        cx = float(np.mean(ring[:, 0]))
        cy = float(np.mean(ring[:, 1]))
        ox, oy = offsets[i % len(offsets)]
        ax.annotate(
            str(fid),
            xy=(cx, cy),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=4.4,
            color="#ffcc00",
            ha="center",
            va="center",
            arrowprops={"arrowstyle": "-", "color": "#ffcc00", "linewidth": 0.32, "alpha": 0.84},
            zorder=8,
        )


def extruded_faces(building: dict, lon0: float, lat0: float) -> list[np.ndarray]:
    ring = building["ring_lonlat"]
    east, north = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
    bottom_z = np.zeros_like(east)
    roof_z = np.full_like(east, float(building["height_m"]))
    bottom = np.column_stack([east, north, bottom_z])
    roof = np.column_stack([east, north, roof_z])
    faces: list[np.ndarray] = [roof, bottom[::-1]]
    for i in range(ring.shape[0]):
        j = (i + 1) % ring.shape[0]
        faces.append(np.asarray([bottom[i], bottom[j], roof[j], roof[i]], dtype=np.float64))
    return faces


def plot_selected_buildings_3d(out_png: Path, rows: list[dict], buildings: dict[int, dict], fids: list[int]) -> None:
    ncols = 4
    nrows = int(math.ceil(len(fids) / ncols))
    fig = plt.figure(figsize=(4.3 * ncols, 4.25 * nrows), dpi=300)
    scatter = None
    fid_set = set(fids)
    for i, fid in enumerate(fids):
        ax = fig.add_subplot(nrows, ncols, i + 1, projection="3d")
        b = buildings[fid]
        fid_rows = [r for r in rows if int(r["fid"]) == fid]
        lon0 = float(np.mean(b["ring_lonlat"][:, 0]))
        lat0 = float(np.mean(b["ring_lonlat"][:, 1]))
        faces = extruded_faces(b, lon0, lat0)
        ax.add_collection3d(Poly3DCollection(faces, facecolor="#7dd3fc", edgecolor="#111111", linewidth=0.35, alpha=0.16))

        method_lon = np.asarray([r["method_lon"] for r in fid_rows], dtype=float)
        method_lat = np.asarray([r["method_lat"] for r in fid_rows], dtype=float)
        gamma_lon = np.asarray([r["gamma_lon"] for r in fid_rows], dtype=float)
        gamma_lat = np.asarray([r["gamma_lat"] for r in fid_rows], dtype=float)
        method_e, method_n = local_en(method_lon, method_lat, lon0, lat0)
        gamma_e, gamma_n = local_en(gamma_lon, gamma_lat, lon0, lat0)
        method_z = np.asarray([r["method_h"] - b["base_height_m"] for r in fid_rows], dtype=float)
        gamma_z = np.asarray([r["gamma_h"] - b["base_height_m"] for r in fid_rows], dtype=float)
        velocity = np.asarray([r["velocity"] for r in fid_rows], dtype=float)

        scatter = ax.scatter(
            method_e,
            method_n,
            method_z,
            c=velocity,
            cmap=DEFORMATION_CMAP,
            norm=DEFORMATION_NORM,
            marker="o",
            s=10,
            alpha=0.92,
            depthshade=False,
            linewidths=0,
        )
        ax.scatter(
            gamma_e,
            gamma_n,
            gamma_z,
            c=velocity,
            cmap=DEFORMATION_CMAP,
            norm=DEFORMATION_NORM,
            marker="^",
            s=9,
            alpha=0.62,
            depthshade=False,
            linewidths=0,
        )
        line_step = max(1, len(fid_rows) // 36)
        for k in range(0, len(fid_rows), line_step):
            ax.plot(
                [method_e[k], gamma_e[k]],
                [method_n[k], gamma_n[k]],
                [method_z[k], gamma_z[k]],
                color="#a3a3a3",
                linewidth=0.32,
                alpha=0.36,
            )

        all_x = np.concatenate([method_e, gamma_e, *[face[:, 0] for face in faces]])
        all_y = np.concatenate([method_n, gamma_n, *[face[:, 1] for face in faces]])
        radius = max(float(np.ptp(all_x)), float(np.ptp(all_y)), 20.0) * 0.62
        cx, cy = float(np.mean(all_x)), float(np.mean(all_y))
        ax.set_xlim(cx - radius, cx + radius)
        ax.set_ylim(cy - radius, cy + radius)
        ax.set_zlim(-4.0, max(float(b["height_m"]) + 5.0, float(np.nanmax([method_z.max(), gamma_z.max()])) + 3.0))
        ax.set_xlabel("东向 / 米", fontsize=7)
        ax.set_ylabel("北向 / 米", fontsize=7)
        ax.set_zlabel("高度 / 米", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.view_init(elev=25, azim=-52)
        ax.set_title(f"建筑编号 {fid}｜点数={len(fid_rows)}｜{b['floor']} 层", fontsize=8.5)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d9d9d9", markersize=5, label="建筑约束方法"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d9d9d9", markersize=5, label="传统 GAMMA/DSM"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, fontsize=9, bbox_to_anchor=(0.49, 0.94))
    if scatter is not None:
        cax = fig.add_axes([0.925, 0.17, 0.012, 0.62])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_ticks(DEFORMATION_TICKS)
        cbar.set_label("形变速率 /（毫米·年$^{-1}$）")
    fig.suptitle("参考区域选定建筑：按 PS 形变着色的两种方法对比", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.025, right=0.90, bottom=0.035, top=0.88, wspace=0.02, hspace=0.23)
    fig.savefig(out_png)
    plt.close(fig)


def plot_facade_panels(out_png: Path, rows: list[dict], buildings: dict[int, dict], fids: list[int]) -> None:
    ncols = 4
    nrows = int(math.ceil(len(fids) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.35 * ncols, 3.2 * nrows), dpi=300, squeeze=False)
    scatter = None
    for i, fid in enumerate(fids):
        ax = axes.ravel()[i]
        b = buildings[fid]
        fid_rows = [r for r in rows if int(r["fid"]) == fid]
        method_x = np.asarray([r["method_lon"] for r in fid_rows], dtype=float)
        gamma_x = np.asarray([r["gamma_lon"] for r in fid_rows], dtype=float)
        method_z = np.asarray([r["method_h"] - b["base_height_m"] for r in fid_rows], dtype=float)
        gamma_z = np.asarray([r["gamma_h"] - b["base_height_m"] for r in fid_rows], dtype=float)
        velocity = np.asarray([r["velocity"] for r in fid_rows], dtype=float)
        ax.set_facecolor("#101214")
        scatter = ax.scatter(method_x, method_z, c=velocity, cmap=DEFORMATION_CMAP, norm=DEFORMATION_NORM, s=8, marker="o", alpha=0.88, linewidths=0)
        ax.scatter(gamma_x, gamma_z, c=velocity, cmap=DEFORMATION_CMAP, norm=DEFORMATION_NORM, s=7, marker="^", alpha=0.60, linewidths=0)
        ax.set_title(f"建筑编号 {fid}｜点数={len(fid_rows)}", fontsize=8.5)
        ax.set_xlabel("经度 / 度", fontsize=7)
        ax.set_ylabel("高度 / 米", fontsize=7)
        ax.grid(color="white", linewidth=0.22, alpha=0.20)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(fids) :]:
        ax.axis("off")
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d9d9d9", markersize=5, label="建筑约束方法"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d9d9d9", markersize=5, label="传统 GAMMA/DSM"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, fontsize=9, bbox_to_anchor=(0.49, 0.94))
    if scatter is not None:
        cax = fig.add_axes([0.925, 0.17, 0.012, 0.62])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_ticks(DEFORMATION_TICKS)
        cbar.set_label("形变速率 /（毫米·年$^{-1}$）")
    fig.suptitle("参考区域建筑立面面板：按 PS 形变着色", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.045, right=0.90, bottom=0.06, top=0.86, wspace=0.22, hspace=0.42)
    fig.savefig(out_png)
    plt.close(fig)


def plot_focused_plan_compare(out_png: Path, rows: list[dict], buildings: dict[int, dict], fids: list[int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 8.0), dpi=300, sharex=True, sharey=True)
    selected = set(fids)
    specs = [
        (axes[0], "建筑约束方法", "method_lon", "method_lat", "o", 7.0, 0.88),
        (axes[1], "传统 GAMMA/DSM", "gamma_lon", "gamma_lat", "^", 7.0, 0.66),
    ]
    scatter = None
    for ax, title, lon_key, lat_key, marker, size, alpha in specs:
        ax.set_facecolor("#101214")
        draw_building_outlines(ax, buildings, selected)
        lon = np.asarray([r[lon_key] for r in rows], dtype=float)
        lat = np.asarray([r[lat_key] for r in rows], dtype=float)
        velocity = np.asarray([r["velocity"] for r in rows], dtype=float)
        in_focus = (
            (FOCUS_BOUNDS[0] <= lon)
            & (lon <= FOCUS_BOUNDS[2])
            & (FOCUS_BOUNDS[1] <= lat)
            & (lat <= FOCUS_BOUNDS[3])
        )
        if np.any(in_focus):
            scatter = ax.scatter(
                lon[in_focus],
                lat[in_focus],
                c=velocity[in_focus],
                cmap=DEFORMATION_CMAP,
                norm=DEFORMATION_NORM,
                marker=marker,
                s=size,
                alpha=alpha,
                linewidths=0,
                zorder=4,
            )
        annotate_selected_buildings(ax, buildings, fids)
        ax.set_xlim(FOCUS_BOUNDS[0], FOCUS_BOUNDS[2])
        ax.set_ylim(FOCUS_BOUNDS[1], FOCUS_BOUNDS[3])
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("经度 / 度", fontsize=8)
        ax.grid(color="white", linewidth=0.20, alpha=0.18)
        ax.ticklabel_format(useOffset=False, style="plain")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("纬度 / 度", fontsize=8)
    handles = [
        Line2D([0], [0], color="#e8e8e8", linewidth=0.6, label="建筑轮廓"),
        Line2D([0], [0], color="#ffcc00", linewidth=0.9, label="选定建筑"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d9d9d9", markersize=4, label="建筑约束点"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d9d9d9", markersize=4, label="GAMMA/DSM 点"),
    ]
    axes[1].legend(handles=handles, loc="upper right", fontsize=7, frameon=True)
    if scatter is not None:
        cax = fig.add_axes([0.905, 0.20, 0.014, 0.54])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_ticks(DEFORMATION_TICKS)
        cbar.set_label("形变速率 /（毫米·年$^{-1}$）", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("参考区域聚焦建筑：两种方法平面对比", fontsize=12, y=0.98)
    fig.subplots_adjust(left=0.055, right=0.88, bottom=0.09, top=0.90, wspace=0.06)
    fig.savefig(out_png)
    plt.close(fig)


def plot_random_other_regions_compare(out_png: Path, rows: list[dict], buildings: dict[int, dict], fids: list[int]) -> None:
    nrows = len(fids)
    fig, axes = plt.subplots(nrows, 2, figsize=(8.4, 2.45 * nrows), dpi=300, squeeze=False)
    scatter = None
    for i, fid in enumerate(fids):
        building = buildings[fid]
        fid_rows = [r for r in rows if int(r["fid"]) == fid]
        ring = building["ring_lonlat"]
        minx, miny = np.min(ring[:, 0]), np.min(ring[:, 1])
        maxx, maxy = np.max(ring[:, 0]), np.max(ring[:, 1])
        pad_x = max((maxx - minx) * 1.15, 0.00022)
        pad_y = max((maxy - miny) * 1.15, 0.00020)
        bounds = (minx - pad_x, miny - pad_y, maxx + pad_x, maxy + pad_y)
        specs = [
            (axes[i, 0], "建筑约束方法", "method_lon", "method_lat", "o", 9.0, 0.90),
            (axes[i, 1], "GAMMA/DSM", "gamma_lon", "gamma_lat", "^", 8.0, 0.72),
        ]
        for ax, label, lon_key, lat_key, marker, size, alpha in specs:
            ax.set_facecolor("#101214")
            ax.plot(ring[:, 0], ring[:, 1], color="#ffcc00", linewidth=0.75, alpha=0.96, zorder=3)
            lon = np.asarray([r[lon_key] for r in fid_rows], dtype=float)
            lat = np.asarray([r[lat_key] for r in fid_rows], dtype=float)
            velocity = np.asarray([r["velocity"] for r in fid_rows], dtype=float)
            scatter = ax.scatter(
                lon,
                lat,
                c=velocity,
                cmap=DEFORMATION_CMAP,
                norm=DEFORMATION_NORM,
                marker=marker,
                s=size,
                alpha=alpha,
                linewidths=0,
                zorder=4,
            )
            ax.set_xlim(bounds[0], bounds[2])
            ax.set_ylim(bounds[1], bounds[3])
            ax.set_aspect("equal", adjustable="box")
            ax.grid(color="white", linewidth=0.18, alpha=0.18)
            ax.ticklabel_format(useOffset=False, style="plain")
            ax.tick_params(labelsize=5.5)
            ax.set_title(f"建筑编号 {fid}｜{label}｜点数={len(fid_rows)}", fontsize=7.2)
            if i == nrows - 1:
                ax.set_xlabel("经度 / 度", fontsize=6.5)
            if label == "建筑约束方法":
                ax.set_ylabel("纬度 / 度", fontsize=6.5)
    handles = [
        Line2D([0], [0], color="#ffcc00", linewidth=0.8, label="建筑轮廓"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d9d9d9", markersize=4, label="建筑约束点"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#d9d9d9", markersize=4, label="GAMMA/DSM 点"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=True, bbox_to_anchor=(0.47, 0.975))
    if scatter is not None:
        cax = fig.add_axes([0.91, 0.16, 0.016, 0.70])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_ticks(DEFORMATION_TICKS)
        cbar.set_label("形变速率 /（毫米·年$^{-1}$）", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    fig.suptitle("参考区域其他随机建筑：两种方法平面对比", fontsize=11.5, y=0.997)
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.055, top=0.94, wspace=0.16, hspace=0.42)
    fig.savefig(out_png)
    plt.close(fig)


def building_intersects_focus(building: dict) -> bool:
    ring = building["ring_lonlat"]
    return bool(
        np.any((FOCUS_BOUNDS[0] <= ring[:, 0]) & (ring[:, 0] <= FOCUS_BOUNDS[2]))
        and np.any((FOCUS_BOUNDS[1] <= ring[:, 1]) & (ring[:, 1] <= FOCUS_BOUNDS[3]))
    )


def plot_focused_white_model_3d(out_png: Path, rows: list[dict], buildings: dict[int, dict], fids: list[int]) -> None:
    focus_buildings = {fid: b for fid, b in buildings.items() if building_intersects_focus(b)}
    if not focus_buildings:
        focus_buildings = {fid: buildings[fid] for fid in fids if fid in buildings}
    selected = set(fids)
    all_rings = np.vstack([b["ring_lonlat"] for b in focus_buildings.values()])
    lon0 = float(np.mean(all_rings[:, 0]))
    lat0 = float(np.mean(all_rings[:, 1]))

    fig = plt.figure(figsize=(12.6, 9.2), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    all_faces = []
    selected_faces = []
    for fid, building in focus_buildings.items():
        faces = extruded_faces(building, lon0, lat0)
        if fid in selected:
            selected_faces.extend(faces)
        else:
            all_faces.extend(faces)
    if all_faces:
        ax.add_collection3d(
            Poly3DCollection(
                all_faces,
                facecolor="#ffffff",
                edgecolor="#d4d4d4",
                linewidth=0.16,
                alpha=0.24,
            )
        )
    if selected_faces:
        ax.add_collection3d(
            Poly3DCollection(
                selected_faces,
                facecolor="#ffffff",
                edgecolor="#4a4a4a",
                linewidth=0.40,
                alpha=0.42,
            )
        )

    plot_rows = [r for r in rows if int(r["fid"]) in selected]
    scatter = None
    if plot_rows:
        method_lon = np.asarray([r["method_lon"] for r in plot_rows], dtype=float)
        method_lat = np.asarray([r["method_lat"] for r in plot_rows], dtype=float)
        method_e, method_n = local_en(method_lon, method_lat, lon0, lat0)
        method_z = np.asarray([r["method_h"] - buildings[int(r["fid"])]["base_height_m"] for r in plot_rows], dtype=float) + 0.35
        velocity = np.asarray([r["velocity"] for r in plot_rows], dtype=float)
        scatter = ax.scatter(
            method_e,
            method_n,
            method_z,
            c=velocity,
            cmap=DEFORMATION_CMAP,
            norm=DEFORMATION_NORM,
            marker="o",
            s=46,
            alpha=0.96,
            depthshade=False,
            linewidths=0.42,
            edgecolors="#111111",
        )

    for fid in fids:
        building = buildings.get(fid)
        if building is None:
            continue
        ring = building["ring_lonlat"]
        east, north = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
        ax.text(
            float(np.mean(east)),
            float(np.mean(north)),
            float(building["height_m"]) + 2.5,
            str(fid),
            fontsize=7,
            color="#111111",
            ha="center",
            va="bottom",
        )

    selected_rings = np.vstack([buildings[fid]["ring_lonlat"] for fid in fids if fid in buildings])
    selected_e, selected_n = local_en(selected_rings[:, 0], selected_rings[:, 1], lon0, lat0)
    radius = max(float(np.ptp(selected_e)), float(np.ptp(selected_n)), 80.0) * 0.72
    cx, cy = float(np.mean(selected_e)), float(np.mean(selected_n))
    max_height = max(float(b["height_m"]) for b in focus_buildings.values())
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_zlim(0.0, max_height + 12.0)
    ax.set_xlabel("东向 / 米", fontsize=9)
    ax.set_ylabel("北向 / 米", fontsize=9)
    ax.set_zlabel("Relative height / m", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.view_init(elev=28, azim=-48)
    ax.set_box_aspect((1, 1, 0.55))
    ax.set_facecolor("#f3f4f6")
    ax.xaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
    ax.yaxis.pane.set_facecolor((0.97, 0.97, 0.97, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.grid(color="#d9d9d9", linewidth=0.3, alpha=0.55)
    if scatter is not None:
        cax = fig.add_axes([0.90, 0.19, 0.014, 0.58])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_ticks(DEFORMATION_TICKS)
        cbar.set_label("形变速率 /（毫米·年$^{-1}$）", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
    ax.set_title("参考区域聚焦建筑白模与 Geo-BC 点", fontsize=13, pad=14)
    fig.subplots_adjust(left=0.02, right=0.87, bottom=0.03, top=0.94)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_PIC_ALL.mkdir(parents=True, exist_ok=True)
    buildings = load_buildings(BUILDINGS_GEOJSON)
    velocity_by_pixel = read_ps_velocity_by_pixel(PS_DEFORMATION_RATE)
    rows = read_points(POINTS_CSV, velocity_by_pixel)
    fids = select_fids(rows, buildings)
    if not fids:
        raise RuntimeError("No selected buildings with enough deformation-matched points in reference area")
    selected_rows = [r for r in rows if int(r["fid"]) in set(fids)]
    random_other_fids = [fid for fid in RANDOM_OTHER_FIDS if fid in buildings and any(int(r["fid"]) == fid for r in rows)]
    random_other_rows = [r for r in rows if int(r["fid"]) in set(random_other_fids)]

    outputs = [
        OUT_IMAGES / "fig_21_reference_area_selected_buildings_3d_method_vs_gamma_deformation_rate.png",
        OUT_IMAGES / "fig_22_reference_area_selected_building_facade_panels_deformation_rate.png",
        OUT_IMAGES / "fig_24_reference_area_focused_plan_method_vs_gamma_selected_buildings.png",
        OUT_IMAGES / "fig_25_reference_area_random_other_buildings_plan_method_vs_gamma.png",
        OUT_IMAGES / "fig_26_reference_area_building_white_model_geobc_points.png",
    ]
    plot_selected_buildings_3d(outputs[0], selected_rows, buildings, fids)
    plot_facade_panels(outputs[1], selected_rows, buildings, fids)
    plot_focused_plan_compare(outputs[2], selected_rows, buildings, fids)
    plot_random_other_regions_compare(outputs[3], random_other_rows, buildings, random_other_fids)
    plot_focused_white_model_3d(outputs[4], selected_rows, buildings, fids)
    for out in outputs:
        shutil.copy2(out, OUT_PIC_ALL / out.name)

    stats = []
    for fid in fids:
        fid_rows = [r for r in selected_rows if int(r["fid"]) == fid]
        velocities = np.asarray([r["velocity"] for r in fid_rows], dtype=float)
        stats.append(
            {
                "fid": fid,
                "matched_points": len(fid_rows),
                "velocity_median_mm_yr": float(np.median(velocities)),
                "velocity_median_abs_mm_yr": float(np.median(np.abs(velocities))),
                "velocity_p10_mm_yr": float(np.percentile(velocities, 10)),
                "velocity_p90_mm_yr": float(np.percentile(velocities, 90)),
            }
        )
    random_stats = []
    for fid in random_other_fids:
        fid_rows = [r for r in random_other_rows if int(r["fid"]) == fid]
        velocities = np.asarray([r["velocity"] for r in fid_rows], dtype=float)
        random_stats.append(
            {
                "fid": fid,
                "matched_points": len(fid_rows),
                "velocity_median_mm_yr": float(np.median(velocities)),
                "velocity_median_abs_mm_yr": float(np.median(np.abs(velocities))),
            }
        )

    readme = OUT_IMAGES / "README_reference_area_selected_buildings.md"
    readme.write_text(
        "\n".join(
            [
                "# Reference-area selected building deformation comparison",
                "",
                f"- Bounds: lon `{REF_BOUNDS[0]}` to `{REF_BOUNDS[2]}`, lat `{REF_BOUNDS[1]}` to `{REF_BOUNDS[3]}`.",
                f"- Focused plan-comparison bounds: lon `{FOCUS_BOUNDS[0]}` to `{FOCUS_BOUNDS[2]}`, lat `{FOCUS_BOUNDS[1]}` to `{FOCUS_BOUNDS[3]}`.",
                f"- Selected FIDs: {', '.join(str(fid) for fid in fids)}",
                f"- Random other-area comparison FIDs: {', '.join(str(fid) for fid in random_other_fids)}",
                f"- Source point table: `{POINTS_CSV}`.",
                f"- PS deformation rate: `{PS_DEFORMATION_RATE}` column 6, matched by SAR pixel.",
                "- Marker shapes: circles are building-constrained points; triangles are traditional GAMMA/DSM points.",
                "- White-model view: building footprints in the focused area are extruded by `height_m`; geo_bc points are placed by relative height `method_h - base_height_m`.",
                "- Building labels in the focused plan map are small offset labels with leader lines to avoid covering building footprints.",
                "- Color ramp: continuous, negative red, near-zero green, positive blue.",
                "- Selection rule: red negative-deformation building cluster to the lower-left of FID `1495`.",
                "- Random other-area rule: fixed-seed spatially spread sample from other reference-area buildings with at least 18 matched points.",
                "",
                "## Per-building counts",
                "",
                *[
                    f"- FID `{s['fid']}`: {s['matched_points']} points, median velocity {s['velocity_median_mm_yr']:.2f} mm/yr, median |velocity| {s['velocity_median_abs_mm_yr']:.2f} mm/yr"
                    for s in stats
                ],
                "",
                "## Random other-area counts",
                "",
                *[
                    f"- FID `{s['fid']}`: {s['matched_points']} points, median velocity {s['velocity_median_mm_yr']:.2f} mm/yr, median |velocity| {s['velocity_median_abs_mm_yr']:.2f} mm/yr"
                    for s in random_stats
                ],
                "",
                *[f"- `{p.name}`" for p in outputs],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"selected_fids={fids}")
    print(f"selected_points={len(selected_rows)}")
    print(f"out_dir={OUT_IMAGES}")
    print(f"pic_all_dir={OUT_PIC_ALL}")


if __name__ == "__main__":
    main()
