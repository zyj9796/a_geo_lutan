from __future__ import annotations

import os
from collections import defaultdict

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay, QhullError
from shapely.geometry import Point, Polygon
from shapely.ops import triangulate

from make_huajiachi_full_area_geobc_3d import (
    OUT_DIR,
    VIEW_AZIM,
    VIEW_ELEV,
    VIEW_ROLL,
    load_full_area_buildings,
    load_full_area_points,
    read_velocity_by_pixel,
)
from make_publication_3d_geobc import TEXT, add_site_boundary, style_3d_axis
from make_reference_area_selected_building_deformation import (
    DEFORMATION_CMAP,
    DEFORMATION_NORM,
    DEFORMATION_TICKS,
    local_en,
)


AREA_LABEL = os.environ.get("SAR_GEOCODE_AREA_LABEL", "华家池")
OUT_PNG = OUT_DIR / f"fig_12_{os.environ.get('SAR_GEOCODE_FIGURE_PREFIX', 'huajiachi')}_interpolated_building_surfaces_3d.png"
SURFACE_STEP_M = 5.0
IDW_POWER = 2.0
SURFACE_ALPHA = 0.88
NO_DATA_COLOR = (0.91, 0.92, 0.92, 0.10)


def to_display(xyz: np.ndarray) -> np.ndarray:
    result = np.asarray(xyz, dtype=float).copy()
    east = result[:, 0].copy()
    north = result[:, 1].copy()
    result[:, 0] = -north
    result[:, 1] = -east
    return result


def idw(query_xyz: np.ndarray, sample_xyz: np.ndarray, values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        return np.full(query_xyz.shape[0], values[0], dtype=float)
    distance = np.linalg.norm(query_xyz[:, None, :] - sample_xyz[None, :, :], axis=2)
    exact = distance < 1e-7
    weights = 1.0 / np.maximum(distance, 0.75) ** IDW_POWER
    result = np.sum(weights * values[None, :], axis=1) / np.sum(weights, axis=1)
    exact_rows = np.any(exact, axis=1)
    if np.any(exact_rows):
        result[exact_rows] = values[np.argmax(exact[exact_rows], axis=1)]
    return result


def roof_triangles(ring_en: np.ndarray, height: float) -> list[np.ndarray]:
    polygon = Polygon(ring_en)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    boundary_points: list[np.ndarray] = []
    for index in range(ring_en.shape[0]):
        start = ring_en[index]
        end = ring_en[(index + 1) % ring_en.shape[0]]
        count = max(1, int(np.ceil(np.linalg.norm(end - start) / SURFACE_STEP_M)))
        boundary_points.extend(
            start + fraction * (end - start) for fraction in np.linspace(0.0, 1.0, count, endpoint=False)
        )
    min_x, min_y, max_x, max_y = polygon.bounds
    interior_points = [
        np.asarray([x, y], dtype=float)
        for x in np.arange(min_x + SURFACE_STEP_M, max_x, SURFACE_STEP_M)
        for y in np.arange(min_y + SURFACE_STEP_M, max_y, SURFACE_STEP_M)
        if polygon.covers(Point(x, y))
    ]
    points = np.unique(np.asarray(boundary_points + interior_points, dtype=float), axis=0)
    faces: list[np.ndarray] = []
    if points.shape[0] >= 3:
        try:
            simplices = Delaunay(points).simplices
            for simplex in simplices:
                coords = points[simplex]
                if polygon.covers(Polygon(coords)):
                    faces.append(np.column_stack([coords, np.full(3, height)]))
        except QhullError:
            pass
    if not faces:
        for triangle in triangulate(polygon):
            if polygon.covers(triangle):
                coords = np.asarray(triangle.exterior.coords[:-1], dtype=float)
                faces.append(np.column_stack([coords, np.full(coords.shape[0], height)]))
    return faces


def wall_quads(ring_en: np.ndarray, height: float) -> list[np.ndarray]:
    faces: list[np.ndarray] = []
    vertical_count = max(1, int(np.ceil(height / SURFACE_STEP_M)))
    z_edges = np.linspace(0.0, height, vertical_count + 1)
    for index in range(ring_en.shape[0]):
        start = ring_en[index]
        end = ring_en[(index + 1) % ring_en.shape[0]]
        horizontal_count = max(1, int(np.ceil(np.linalg.norm(end - start) / SURFACE_STEP_M)))
        xy_edges = start[None, :] + np.linspace(0.0, 1.0, horizontal_count + 1)[:, None] * (end - start)
        for h_index in range(horizontal_count):
            for z_index in range(vertical_count):
                faces.append(
                    np.asarray(
                        [
                            [*xy_edges[h_index], z_edges[z_index]],
                            [*xy_edges[h_index + 1], z_edges[z_index]],
                            [*xy_edges[h_index + 1], z_edges[z_index + 1]],
                            [*xy_edges[h_index], z_edges[z_index + 1]],
                        ],
                        dtype=float,
                    )
                )
    return faces


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings = load_full_area_buildings()
    rows = load_full_area_points(buildings, read_velocity_by_pixel())
    grouped_rows: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[int(row["fid"])].append(row)

    all_rings = np.vstack([building["ring_lonlat"] for building in buildings.values()])
    lon0 = float(np.mean(all_rings[:, 0]))
    lat0 = float(np.mean(all_rings[:, 1]))
    east_all, north_all = local_en(all_rings[:, 0], all_rings[:, 1], lon0, lat0)
    south_all, west_all = -north_all, -east_all
    pad_south = max(float(np.ptp(south_all)) * 0.035, 12.0)
    pad_west = max(float(np.ptp(west_all)) * 0.055, 12.0)
    xlim = (float(np.min(south_all) - pad_south), float(np.max(south_all) + pad_south))
    ylim = (float(np.min(west_all) - pad_west), float(np.max(west_all) + pad_west))

    colored_faces: list[np.ndarray] = []
    colored_values: list[float] = []
    no_data_faces: list[np.ndarray] = []
    for fid, building in buildings.items():
        ring = building["ring_lonlat"]
        east, north = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
        ring_en = np.column_stack([east, north])
        height = max(float(building["height_m"]), 0.5)
        faces = roof_triangles(ring_en, height) + wall_quads(ring_en, height)
        building_rows = grouped_rows.get(fid, [])
        if not building_rows:
            no_data_faces.extend(to_display(face) for face in faces)
            continue

        point_lon = np.asarray([row["lon"] for row in building_rows], dtype=float)
        point_lat = np.asarray([row["lat"] for row in building_rows], dtype=float)
        point_east, point_north = local_en(point_lon, point_lat, lon0, lat0)
        point_height = np.clip(
            np.asarray([row["z"] for row in building_rows], dtype=float), 0.0, height
        )
        samples = np.column_stack([point_east, point_north, point_height])
        velocity = np.asarray([row["velocity"] for row in building_rows], dtype=float)
        centers = np.asarray([np.mean(face, axis=0) for face in faces], dtype=float)
        estimates = idw(centers, samples, velocity)
        colored_faces.extend(to_display(face) for face in faces)
        colored_values.extend(estimates.tolist())

    values_array = np.asarray(colored_values, dtype=float)
    facecolors = DEFORMATION_CMAP(DEFORMATION_NORM(values_array))
    facecolors[:, 3] = SURFACE_ALPHA

    fig = plt.figure(figsize=(10.6, 8.2), dpi=300)
    ax = fig.add_axes([0.015, 0.18, 0.97, 0.76], projection="3d")
    ax.add_collection3d(
        Poly3DCollection(
            colored_faces,
            facecolors=facecolors,
            edgecolors="none",
            linewidths=0,
            rasterized=True,
        )
    )
    if no_data_faces:
        ax.add_collection3d(
            Poly3DCollection(
                no_data_faces,
                facecolors=NO_DATA_COLOR,
                edgecolors=(0.45, 0.47, 0.48, 0.16),
                linewidths=0.025,
                rasterized=True,
            )
        )
    add_site_boundary(ax, xlim, ylim)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_zlim(-1.5, max(float(building["height_m"]) for building in buildings.values()) * 1.04)
    style_3d_axis(ax)
    ax.set_box_aspect((1.18, 1.0, 0.20), zoom=1.48)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM, roll=VIEW_ROLL)

    mappable = ScalarMappable(norm=DEFORMATION_NORM, cmap=DEFORMATION_CMAP)
    mappable.set_array(values_array)
    fig.text(
        0.025,
        0.965,
        f"{AREA_LABEL}全区域建筑表面插值形变",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    fig.text(
        0.025,
        0.925,
        "逐建筑三维反距离插值；不跨建筑插值",
        fontsize=7.5,
        color="#555b61",
        va="top",
    )
    fig.text(
        0.975,
        0.165,
        f"{len(buildings)} 栋建筑｜{len(rows)} 个表面样本｜{len(grouped_rows)} 栋参与插值｜5 米表面网格",
        fontsize=6.4,
        color="#666b70",
        ha="right",
    )
    cax = fig.add_axes([0.135, 0.075, 0.73, 0.032])
    colorbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    colorbar.set_ticks(DEFORMATION_TICKS)
    colorbar.ax.tick_params(labelsize=6.4, length=2.5, pad=3)
    colorbar.outline.set_linewidth(0.6)
    colorbar.set_label(r"插值形变速率（毫米·年$^{-1}$）", fontsize=7.5, labelpad=5)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    print(f"buildings={len(buildings)}")
    print(f"surface_samples={len(rows)}")
    print(f"interpolated_buildings={len(grouped_rows)}")
    print(f"colored_surface_cells={len(colored_faces)}")
    print(f"no_data_surface_cells={len(no_data_faces)}")
    print(f"output={OUT_PNG}")


if __name__ == "__main__":
    main()
