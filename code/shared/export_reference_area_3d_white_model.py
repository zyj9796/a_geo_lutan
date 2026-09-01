from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
from osgeo import ogr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from io_paths import GEOJSON_ROOT
from make_reference_area_selected_building_deformation import (
    BUILDINGS_GEOJSON,
    DATE,
    FOCUS_BOUNDS,
    REF_BOUNDS,
    OUT_IMAGES,
    OUT_PIC_ALL,
    POINTS_CSV,
    PS_DEFORMATION_RATE,
    building_intersects_focus,
    DEFORMATION_CMAP,
    DEFORMATION_NORM,
    DEFORMATION_TICKS,
    extruded_faces,
    load_buildings,
    local_en,
    plot_focused_white_model_3d,
    read_points,
    read_ps_velocity_by_pixel,
    select_fids,
)


OUT_DIR = GEOJSON_ROOT / "full_area_geobc_ps" / "defo_3d" / "fig20_reference_area"
BUILDING_SHP = OUT_DIR / "fig20_reference_area_building_white_model_3d.shp"
GEObc_POINT_SHP = OUT_DIR / "fig20_reference_area_geobc_points_3d.shp"
GAMMA_POINT_SHP = OUT_DIR / "fig20_reference_area_gamma_dsm_points_3d.shp"
GEObc_POINT_CSV = OUT_DIR / "fig20_reference_area_geobc_points_3d.csv"
GAMMA_POINT_CSV = OUT_DIR / "fig20_reference_area_gamma_dsm_points_3d.csv"
VIS_PNG = OUT_IMAGES / "fig_26_reference_area_building_white_model_geobc_points.png"
REF_VIS_PNG = OUT_IMAGES / "fig_27_fig20_reference_area_white_model_3d_points.png"
REF_VIS_BUILDING_PNG = OUT_IMAGES / "fig_28_fig20_reference_area_white_model_building_emphasis.png"
WEST_TO_EAST_VIS_PNG = OUT_IMAGES / "fig_31_fig20_reference_area_west_to_east_high_angle.png"
GEObc_ONLY_WEST_TO_EAST_VIS_PNG = OUT_IMAGES / "fig_32_fig20_reference_area_geobc_only_west_to_east_high_angle.png"
GEObc_ONLY_SOUTHWEST_VIS_PNG = OUT_IMAGES / "fig_33_fig20_reference_area_geobc_only_southwest_high_angle.png"
ANGLE_VIEW_SPECS = [
    ("fig_29a_fig20_reference_area_building_emphasis_view_ne.png", 34, -58, "东北斜视"),
    ("fig_29b_fig20_reference_area_building_emphasis_view_nw.png", 34, -122, "西北斜视"),
    ("fig_29c_fig20_reference_area_building_emphasis_view_se.png", 32, 35, "东南斜视"),
    ("fig_29d_fig20_reference_area_building_emphasis_view_sw.png", 32, 138, "西南斜视"),
    ("fig_29e_fig20_reference_area_building_emphasis_low_angle.png", 18, -64, "低角度斜视"),
    ("fig_29f_fig20_reference_area_building_emphasis_high_angle.png", 58, -62, "高角度斜视"),
]
WEST_VIEW_SPECS = [
    ("fig_30a_fig20_reference_area_building_emphasis_west_view.png", 34, 180, "西向视角"),
    ("fig_30b_fig20_reference_area_building_emphasis_west_low_angle.png", 18, 180, "西向低角度视角"),
    ("fig_30c_fig20_reference_area_building_emphasis_west_high_angle.png", 58, 180, "西向高角度视角"),
]
WGS84_PRJ = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137,298.257223563]],'
    'PRIMEM["Greenwich",0],UNIT["Degree",0.017453292519943295]]'
)


def write_prj(path: Path) -> None:
    path.with_suffix(".prj").write_text(WGS84_PRJ + "\n", encoding="utf-8")


def recreate_layer(path: Path, geom_type: int, layer_name: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    driver = ogr.GetDriverByName("ESRI Shapefile")
    if path.exists():
        driver.DeleteDataSource(str(path))
    ds = driver.CreateDataSource(str(path))
    if ds is None:
        raise RuntimeError(f"Could not create {path}")
    layer = ds.CreateLayer(layer_name, srs=None, geom_type=geom_type)
    if layer is None:
        raise RuntimeError(f"Could not create layer {layer_name}")
    return ds, layer


def add_field(layer, name: str, field_type: int, width: int | None = None, precision: int | None = None) -> None:
    field = ogr.FieldDefn(name, field_type)
    if width is not None:
        field.SetWidth(width)
    if precision is not None:
        field.SetPrecision(precision)
    layer.CreateField(field)


def polygon_z(points: list[tuple[float, float, float]]) -> ogr.Geometry:
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for lon, lat, z in points:
        ring.AddPoint(float(lon), float(lat), float(z))
    if points[0] != points[-1]:
        lon, lat, z = points[0]
        ring.AddPoint(float(lon), float(lat), float(z))
    geom = ogr.Geometry(ogr.wkbPolygon25D)
    geom.AddGeometry(ring)
    return geom


def building_faces_lonlat(building: dict) -> list[tuple[str, list[tuple[float, float, float]]]]:
    ring = building["ring_lonlat"]
    height = float(building["height_m"])
    faces: list[tuple[str, list[tuple[float, float, float]]]] = []
    roof = [(float(lon), float(lat), height) for lon, lat in ring]
    base = [(float(lon), float(lat), 0.0) for lon, lat in ring[::-1]]
    faces.append(("roof", roof))
    faces.append(("base", base))
    for i in range(ring.shape[0]):
        j = (i + 1) % ring.shape[0]
        lon0, lat0 = ring[i]
        lon1, lat1 = ring[j]
        faces.append(
            (
                "wall",
                [
                    (float(lon0), float(lat0), 0.0),
                    (float(lon1), float(lat1), 0.0),
                    (float(lon1), float(lat1), height),
                    (float(lon0), float(lat0), height),
                ],
            )
        )
    return faces


def write_building_white_model(path: Path, buildings: dict[int, dict], selected_fids: list[int]) -> int:
    ds, layer = recreate_layer(path, ogr.wkbPolygon25D, "white_model")
    add_field(layer, "fid", ogr.OFTInteger)
    add_field(layer, "face_id", ogr.OFTInteger)
    add_field(layer, "face_type", ogr.OFTString, width=10)
    add_field(layer, "floor", ogr.OFTInteger)
    add_field(layer, "height_m", ogr.OFTReal, width=12, precision=3)
    add_field(layer, "selected", ogr.OFTInteger)

    selected = set(selected_fids)
    layer_defn = layer.GetLayerDefn()
    feature_count = 0
    for fid, building in sorted(buildings.items()):
        if not building_intersects_bounds(building, REF_BOUNDS):
            continue
        for face_id, (face_type, coords) in enumerate(building_faces_lonlat(building), start=1):
            feat = ogr.Feature(layer_defn)
            feat.SetField("fid", int(fid))
            feat.SetField("face_id", int(face_id))
            feat.SetField("face_type", face_type)
            feat.SetField("floor", int(building["floor"]))
            feat.SetField("height_m", float(building["height_m"]))
            feat.SetField("selected", 1 if fid in selected else 0)
            feat.SetGeometry(polygon_z(coords))
            layer.CreateFeature(feat)
            feat = None
            feature_count += 1
    ds = None
    write_prj(path)
    return feature_count


def building_intersects_bounds(building: dict, bounds: tuple[float, float, float, float]) -> bool:
    minx, miny, maxx, maxy = bounds
    ring = building["ring_lonlat"]
    return bool(
        np.any((minx <= ring[:, 0]) & (ring[:, 0] <= maxx))
        and np.any((miny <= ring[:, 1]) & (ring[:, 1] <= maxy))
    )


def point_in_bounds(row: dict, lon_key: str, lat_key: str, bounds: tuple[float, float, float, float]) -> bool:
    lon = float(row[lon_key])
    lat = float(row[lat_key])
    return bounds[0] <= lon <= bounds[2] and bounds[1] <= lat <= bounds[3]


def selected_point_rows(rows: list[dict], buildings: dict[int, dict], fids: list[int]) -> list[dict]:
    selected = set(fids)
    out = []
    for row in rows:
        fid = int(row["fid"])
        if fid not in selected or fid not in buildings:
            continue
        item = dict(row)
        item["rel_h"] = float(row["method_h"]) - float(buildings[fid]["base_height_m"])
        item["selected"] = 1
        out.append(item)
    return out


def reference_method_rows(rows: list[dict], buildings: dict[int, dict], selected_fids: list[int]) -> list[dict]:
    selected = set(selected_fids)
    out = []
    for row in rows:
        fid = int(row["fid"])
        if fid not in buildings or not point_in_bounds(row, "method_lon", "method_lat", REF_BOUNDS):
            continue
        item = dict(row)
        item["lon"] = float(row["method_lon"])
        item["lat"] = float(row["method_lat"])
        item["z"] = float(row["method_h"]) - float(buildings[fid]["base_height_m"])
        item["abs_h"] = float(row["method_h"])
        item["selected"] = 1 if fid in selected else 0
        out.append(item)
    return out


def reference_gamma_rows(rows: list[dict], buildings: dict[int, dict], selected_fids: list[int]) -> list[dict]:
    selected = set(selected_fids)
    out = []
    for row in rows:
        fid = int(row["fid"])
        if fid not in buildings or not point_in_bounds(row, "gamma_lon", "gamma_lat", REF_BOUNDS):
            continue
        item = dict(row)
        item["lon"] = float(row["gamma_lon"])
        item["lat"] = float(row["gamma_lat"])
        item["z"] = float(row["gamma_h"]) - float(buildings[fid]["base_height_m"])
        item["abs_h"] = float(row["gamma_h"])
        item["selected"] = 1 if fid in selected else 0
        out.append(item)
    return out


def write_point_csv(path: Path, rows: list[dict]) -> None:
    fields = ["fid", "row", "col", "lon", "lat", "z", "abs_h", "velocity", "selected"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def write_points(path: Path, rows: list[dict]) -> int:
    ds, layer = recreate_layer(path, ogr.wkbPoint25D, "geobc_points")
    add_field(layer, "fid", ogr.OFTInteger)
    add_field(layer, "sar_row", ogr.OFTInteger)
    add_field(layer, "sar_col", ogr.OFTInteger)
    add_field(layer, "rel_h", ogr.OFTReal, width=12, precision=3)
    add_field(layer, "method_h", ogr.OFTReal, width=12, precision=3)
    add_field(layer, "vel_mm_yr", ogr.OFTReal, width=12, precision=3)
    add_field(layer, "selected", ogr.OFTInteger)

    layer_defn = layer.GetLayerDefn()
    for row in rows:
        geom = ogr.Geometry(ogr.wkbPoint25D)
        geom.AddPoint(float(row["lon"]), float(row["lat"]), float(row["z"]))
        feat = ogr.Feature(layer_defn)
        feat.SetField("fid", int(row["fid"]))
        feat.SetField("sar_row", int(row["row"]))
        feat.SetField("sar_col", int(row["col"]))
        feat.SetField("rel_h", float(row["z"]))
        feat.SetField("method_h", float(row["abs_h"]))
        feat.SetField("vel_mm_yr", float(row["velocity"]))
        feat.SetField("selected", int(row["selected"]))
        feat.SetGeometry(geom)
        layer.CreateFeature(feat)
        feat = None
    ds = None
    write_prj(path)
    return len(rows)


def plot_reference_area_white_model_3d(out_png: Path, buildings: dict[int, dict], selected_fids: list[int], geobc_rows: list[dict], gamma_rows: list[dict]) -> None:
    ref_buildings = {fid: b for fid, b in buildings.items() if building_intersects_bounds(b, REF_BOUNDS)}
    all_rings = np.vstack([b["ring_lonlat"] for b in ref_buildings.values()])
    lon0 = float(np.mean(all_rings[:, 0]))
    lat0 = float(np.mean(all_rings[:, 1]))
    selected = set(selected_fids)

    all_faces = []
    selected_faces = []
    for fid, building in ref_buildings.items():
        faces = extruded_faces(building, lon0, lat0)
        if fid in selected:
            selected_faces.extend(faces)
        else:
            all_faces.extend(faces)

    fig = plt.figure(figsize=(16.0, 8.8), dpi=260)
    specs = [
        ("建筑约束方法", geobc_rows, "o", 5.5, 0.86),
        ("传统 GAMMA/DSM", gamma_rows, "^", 5.0, 0.62),
    ]
    scatter = None
    for idx, (title, rows, marker, size, alpha) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, 2, idx, projection="3d")
        if all_faces:
            ax.add_collection3d(Poly3DCollection(all_faces, facecolor="#ffffff", edgecolor="#d6d6d6", linewidth=0.08, alpha=0.15))
        if selected_faces:
            ax.add_collection3d(Poly3DCollection(selected_faces, facecolor="#ffffff", edgecolor="#3f3f3f", linewidth=0.24, alpha=0.34))
        if rows:
            lon = np.asarray([r["lon"] for r in rows], dtype=float)
            lat = np.asarray([r["lat"] for r in rows], dtype=float)
            east, north = local_en(lon, lat, lon0, lat0)
            z = np.asarray([r["z"] for r in rows], dtype=float) + 0.25
            velocity = np.asarray([r["velocity"] for r in rows], dtype=float)
            scatter = ax.scatter(
                east,
                north,
                z,
                c=velocity,
                cmap=DEFORMATION_CMAP,
                norm=DEFORMATION_NORM,
                marker=marker,
                s=size,
                alpha=alpha,
                depthshade=False,
                linewidths=0,
            )
        east_all, north_all = local_en(all_rings[:, 0], all_rings[:, 1], lon0, lat0)
        xmid = float((np.min(east_all) + np.max(east_all)) / 2.0)
        ymid = float((np.min(north_all) + np.max(north_all)) / 2.0)
        radius_x = float(np.ptp(east_all)) * 0.54
        radius_y = float(np.ptp(north_all)) * 0.54
        max_height = max(float(b["height_m"]) for b in ref_buildings.values())
        ax.set_xlim(xmid - radius_x, xmid + radius_x)
        ax.set_ylim(ymid - radius_y, ymid + radius_y)
        ax.set_zlim(0.0, max_height + 12.0)
        ax.set_xlabel("东向 / 米", fontsize=8)
        ax.set_ylabel("北向 / 米", fontsize=8)
        ax.set_zlabel("相对高度 / 米", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=33, azim=-62)
        ax.set_box_aspect((0.42, 1.0, 0.22))
        ax.set_facecolor("#f3f4f6")
        ax.xaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        ax.yaxis.pane.set_facecolor((0.98, 0.98, 0.98, 1.0))
        ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        ax.grid(color="#d0d0d0", linewidth=0.22, alpha=0.50)
        ax.set_title(title, fontsize=11, pad=10)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#777777", markersize=5, label="Geo-BC 点"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#777777", markersize=5, label="GAMMA/DSM 点"),
        Line2D([0], [0], color="#bdbdbd", linewidth=1.0, label="三维建筑白模"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=8, frameon=True, bbox_to_anchor=(0.48, 0.945))
    if scatter is not None:
        cax = fig.add_axes([0.91, 0.18, 0.012, 0.60])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_ticks(DEFORMATION_TICKS)
        cbar.set_label("形变速率 /（毫米·年$^{-1}$）", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        cbar.ax.text(1.95, 0.08, "负形变\n沉降", transform=cbar.ax.transAxes, fontsize=7, color="#8b0000", ha="left", va="center")
        cbar.ax.text(1.95, 0.50, "接近零", transform=cbar.ax.transAxes, fontsize=7, color="#008b2f", ha="left", va="center")
        cbar.ax.text(1.95, 0.92, "正形变", transform=cbar.ax.transAxes, fontsize=7, color="#08519c", ha="left", va="center")
    fig.suptitle("图20参考区域：三维建筑白模与形变点", fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.02, right=0.89, bottom=0.04, top=0.88, wspace=0.02)
    fig.savefig(out_png)
    plt.close(fig)


def ground_plane_faces(
    all_rings: np.ndarray,
    lon0: float,
    lat0: float,
    horizontal_signs: tuple[float, float] = (1.0, 1.0),
) -> list[np.ndarray]:
    east, north = local_en(all_rings[:, 0], all_rings[:, 1], lon0, lat0)
    east *= horizontal_signs[0]
    north *= horizontal_signs[1]
    pad_x = max(float(np.ptp(east)) * 0.025, 12.0)
    pad_y = max(float(np.ptp(north)) * 0.025, 12.0)
    xmin, xmax = float(np.min(east) - pad_x), float(np.max(east) + pad_x)
    ymin, ymax = float(np.min(north) - pad_y), float(np.max(north) + pad_y)
    return [
        np.asarray(
            [
                [xmin, ymin, 0.0],
                [xmax, ymin, 0.0],
                [xmax, ymax, 0.0],
                [xmin, ymax, 0.0],
            ],
            dtype=float,
        )
    ]


def draw_ground_footprints(
    ax,
    buildings: dict[int, dict],
    lon0: float,
    lat0: float,
    horizontal_signs: tuple[float, float] = (1.0, 1.0),
) -> None:
    for building in buildings.values():
        ring = building["ring_lonlat"]
        east, north = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
        east *= horizontal_signs[0]
        north *= horizontal_signs[1]
        east = np.r_[east, east[0]]
        north = np.r_[north, north[0]]
        ax.plot(east, north, np.zeros_like(east), color="#5f5f5a", linewidth=0.20, alpha=0.70, zorder=2)


def plot_reference_area_building_emphasis_3d(
    out_png: Path,
    buildings: dict[int, dict],
    selected_fids: list[int],
    geobc_rows: list[dict],
    gamma_rows: list[dict],
    elev: float = 36.0,
    azim: float = -58.0,
    view_label: str = "",
    geobc_only: bool = False,
    horizontal_directions: tuple[str, str] = ("East", "North"),
) -> None:
    ref_buildings = {fid: b for fid, b in buildings.items() if building_intersects_bounds(b, REF_BOUNDS)}
    all_rings = np.vstack([b["ring_lonlat"] for b in ref_buildings.values()])
    lon0 = float(np.mean(all_rings[:, 0]))
    lat0 = float(np.mean(all_rings[:, 1]))
    selected = set(selected_fids)
    horizontal_signs = (
        -1.0 if horizontal_directions[0] == "West" else 1.0,
        -1.0 if horizontal_directions[1] == "South" else 1.0,
    )

    def orient_faces(faces: list[np.ndarray]) -> list[np.ndarray]:
        oriented = []
        for face in faces:
            coords = np.asarray(face, dtype=float).copy()
            coords[:, 0] *= horizontal_signs[0]
            coords[:, 1] *= horizontal_signs[1]
            oriented.append(coords)
        return oriented

    all_faces = []
    selected_faces = []
    for fid, building in ref_buildings.items():
        faces = orient_faces(extruded_faces(building, lon0, lat0))
        if fid in selected:
            selected_faces.extend(faces)
        else:
            all_faces.extend(faces)

    fig = plt.figure(figsize=(9.2, 8.8) if geobc_only else (16.0, 8.8), dpi=260)
    specs = [
        ("建筑约束方法", geobc_rows, "o", 4.2, 0.96),
    ]
    if not geobc_only:
        specs.append(("传统 GAMMA/DSM", gamma_rows, "^", 3.8, 0.84))
    scatter = None
    for idx, (title, rows, marker, size, alpha) in enumerate(specs, start=1):
        ax = fig.add_subplot(1, len(specs), idx, projection="3d")
        ax.add_collection3d(
            Poly3DCollection(
                ground_plane_faces(all_rings, lon0, lat0, horizontal_signs),
                facecolor="#e7e7df",
                edgecolor="#a7a79f",
                linewidth=0.28,
                alpha=0.35,
            )
        )
        if all_faces:
            ax.add_collection3d(
                Poly3DCollection(
                    all_faces,
                    facecolor="#f7f7f2",
                    edgecolor="#7a7a72",
                    linewidth=0.18,
                    alpha=0.58,
                )
            )
        if selected_faces:
            ax.add_collection3d(
                Poly3DCollection(
                    selected_faces,
                    facecolor="#fff8d8",
                    edgecolor="#1f1f1f",
                    linewidth=0.44,
                    alpha=0.72,
                )
            )
        draw_ground_footprints(ax, ref_buildings, lon0, lat0, horizontal_signs)
        if rows:
            lon = np.asarray([r["lon"] for r in rows], dtype=float)
            lat = np.asarray([r["lat"] for r in rows], dtype=float)
            east, north = local_en(lon, lat, lon0, lat0)
            east *= horizontal_signs[0]
            north *= horizontal_signs[1]
            z = np.asarray([r["z"] for r in rows], dtype=float) + 0.40
            velocity = np.asarray([r["velocity"] for r in rows], dtype=float)
            scatter = ax.scatter(
                east,
                north,
                z,
                c=velocity,
                cmap=DEFORMATION_CMAP,
                norm=DEFORMATION_NORM,
                marker=marker,
                s=size,
                alpha=alpha,
                depthshade=False,
                linewidths=0.10,
                edgecolors="#1f1f1f",
            )
        east_all, north_all = local_en(all_rings[:, 0], all_rings[:, 1], lon0, lat0)
        east_all *= horizontal_signs[0]
        north_all *= horizontal_signs[1]
        xmid = float((np.min(east_all) + np.max(east_all)) / 2.0)
        ymid = float((np.min(north_all) + np.max(north_all)) / 2.0)
        radius_x = float(np.ptp(east_all)) * 0.54
        radius_y = float(np.ptp(north_all)) * 0.54
        max_height = max(float(b["height_m"]) for b in ref_buildings.values())
        ax.set_xlim(xmid - radius_x, xmid + radius_x)
        ax.set_ylim(ymid - radius_y, ymid + radius_y)
        ax.set_zlim(0.0, max_height + 12.0)
        direction_labels = {"East": "东向", "West": "西向", "North": "北向", "South": "南向"}
        ax.set_xlabel(f"{direction_labels[horizontal_directions[0]]} / 米", fontsize=8)
        ax.set_ylabel(f"{direction_labels[horizontal_directions[1]]} / 米", fontsize=8)
        ax.set_zlabel("相对高度 / 米", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((0.42, 1.0, 0.24))
        ax.set_facecolor("#f1f1ed")
        ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.92, 1.0))
        ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.92, 1.0))
        ax.zaxis.pane.set_facecolor((0.99, 0.99, 0.97, 1.0))
        ax.grid(color="#9f9f98", linewidth=0.25, alpha=0.45)
        ax.set_title(title, fontsize=11, pad=10)
    handles = [
        Line2D([0], [0], color="#7a7a72", linewidth=1.5, label="三维建筑白模"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555555", markersize=3, label="Geo-BC 点"),
    ]
    if not geobc_only:
        handles.append(Line2D([0], [0], marker="^", color="none", markerfacecolor="#555555", markersize=3, label="GAMMA/DSM 点"))
    color_handles = [
        Line2D([0], [0], marker="o", color="#1f1f1f", markerfacecolor="#ff1a1a", markeredgewidth=0.25, markersize=4, linewidth=0, label="点颜色：负形变 / 沉降"),
        Line2D([0], [0], marker="o", color="#1f1f1f", markerfacecolor="#42d915", markeredgewidth=0.25, markersize=4, linewidth=0, label="点颜色：接近零"),
        Line2D([0], [0], marker="o", color="#1f1f1f", markerfacecolor="#08519c", markeredgewidth=0.25, markersize=4, linewidth=0, label="点颜色：正形变"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), fontsize=8, frameon=True, bbox_to_anchor=(0.48, 0.952))
    fig.legend(handles=color_handles, loc="upper center", ncol=3, fontsize=7.2, frameon=True, bbox_to_anchor=(0.48, 0.918))
    if scatter is not None:
        cax = fig.add_axes([0.91, 0.18, 0.012, 0.60])
        cbar = fig.colorbar(scatter, cax=cax)
        cbar.set_ticks(DEFORMATION_TICKS)
        cbar.set_label("形变速率 /（毫米·年$^{-1}$）", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    suffix = f" | {view_label}" if view_label else ""
    if geobc_only and horizontal_directions == ("West", "South"):
        title = "图20参考区域：Geo-BC 西南侧三维视图"
    elif geobc_only:
        title = "图20参考区域：Geo-BC 由西向东三维视图"
    else:
        title = f"图20参考区域：建筑强调三维白模{suffix}"
    fig.suptitle(title, fontsize=13, y=0.985)
    fig.subplots_adjust(left=0.02, right=0.86 if geobc_only else 0.89, bottom=0.04, top=0.84, wspace=0.02)
    fig.savefig(out_png)
    plt.close(fig)


def write_readme(building_faces: int, geobc_count: int, gamma_count: int, fids: list[int]) -> None:
    readme = OUT_DIR / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Reference-area 3D white model export",
                "",
                f"- Date: `{DATE}`",
                f"- Fig. 20 reference bounds: lon `{REF_BOUNDS[0]}` to `{REF_BOUNDS[2]}`, lat `{REF_BOUNDS[1]}` to `{REF_BOUNDS[3]}`.",
                f"- Selected FIDs: {', '.join(str(fid) for fid in fids)}",
                f"- Building white model: `{BUILDING_SHP.name}`",
                f"- Geo-BC 3D points: `{GEObc_POINT_SHP.name}`",
                f"- GAMMA/DSM 3D points: `{GAMMA_POINT_SHP.name}`",
                f"- Geo-BC CSV mirror: `{GEObc_POINT_CSV.name}`",
                f"- GAMMA/DSM CSV mirror: `{GAMMA_POINT_CSV.name}`",
                f"- Focused 3D visualization: `{VIS_PNG}`",
                f"- Fig. 20 extent 3D visualization: `{REF_VIS_PNG}`",
                f"- Fig. 20 building-emphasis visualization: `{REF_VIS_BUILDING_PNG}`",
                "- Multi-angle building-emphasis visualizations: `fig_29a` to `fig_29f`.",
                "- West-facing building-emphasis visualizations: `fig_30a` to `fig_30c`.",
                f"- West-to-east high-angle visualization: `{WEST_TO_EAST_VIS_PNG.name}`.",
                f"- Geo-BC-only west-to-east high-angle visualization: `{GEObc_ONLY_WEST_TO_EAST_VIS_PNG.name}`.",
                f"- Geo-BC-only southwest high-angle visualization: `{GEObc_ONLY_SOUTHWEST_VIS_PNG.name}`.",
                f"- Building face features: {building_faces}",
                f"- Geo-BC point features: {geobc_count}",
                f"- GAMMA/DSM point features: {gamma_count}",
                "",
                "The building layer is an EPSG:4326 `POLYGONZ` Shapefile. Each roof, base, or wall face is one feature; base faces and wall bottom edges are fixed at Z=0, and roof faces are at `height_m`.",
                "The point layers are EPSG:4326 `POINTZ` Shapefiles. Geo-BC point Z is `method_h - base_height_m`; GAMMA/DSM point Z is `gamma_h - base_height_m`; `vel_mm_yr` stores the matched PS deformation rate.",
                "The 3D visualizations include an explicit Z=0 ground plane and ground-footprint outlines to avoid the visual impression of floating buildings.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def sync_visualizations_to_geodata(paths: list[Path]) -> None:
    for path in paths:
        if path.exists():
            shutil.copy2(path, OUT_DIR / path.name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_PIC_ALL.mkdir(parents=True, exist_ok=True)

    buildings = load_buildings(BUILDINGS_GEOJSON)
    velocity_by_pixel = read_ps_velocity_by_pixel(PS_DEFORMATION_RATE)
    rows = read_points(POINTS_CSV, velocity_by_pixel)
    fids = select_fids(rows, buildings)
    selected_rows = selected_point_rows(rows, buildings, fids)
    geobc_rows = reference_method_rows(rows, buildings, fids)
    gamma_rows = reference_gamma_rows(rows, buildings, fids)

    building_faces = write_building_white_model(BUILDING_SHP, buildings, fids)
    geobc_count = write_points(GEObc_POINT_SHP, geobc_rows)
    gamma_count = write_points(GAMMA_POINT_SHP, gamma_rows)
    write_point_csv(GEObc_POINT_CSV, geobc_rows)
    write_point_csv(GAMMA_POINT_CSV, gamma_rows)
    plot_focused_white_model_3d(VIS_PNG, selected_rows, buildings, fids)
    plot_reference_area_white_model_3d(REF_VIS_PNG, buildings, fids, geobc_rows, gamma_rows)
    plot_reference_area_building_emphasis_3d(REF_VIS_BUILDING_PNG, buildings, fids, geobc_rows, gamma_rows)
    plot_reference_area_building_emphasis_3d(
        WEST_TO_EAST_VIS_PNG,
        buildings,
        fids,
        geobc_rows,
        gamma_rows,
        elev=58,
        azim=180,
        view_label="由西向东高角度视图",
    )
    plot_reference_area_building_emphasis_3d(
        GEObc_ONLY_WEST_TO_EAST_VIS_PNG,
        buildings,
        fids,
        geobc_rows,
        gamma_rows,
        elev=58,
        azim=180,
        view_label="仅 Geo-BC｜由西向东高角度视图",
        geobc_only=True,
    )
    plot_reference_area_building_emphasis_3d(
        GEObc_ONLY_SOUTHWEST_VIS_PNG,
        buildings,
        fids,
        geobc_rows,
        gamma_rows,
        elev=58,
        azim=45,
        view_label="仅 Geo-BC｜西南侧高角度视图",
        geobc_only=True,
        horizontal_directions=("West", "South"),
    )
    angle_outputs = []
    for filename, elev, azim, view_label in ANGLE_VIEW_SPECS + WEST_VIEW_SPECS:
        out_png = OUT_IMAGES / filename
        plot_reference_area_building_emphasis_3d(out_png, buildings, fids, geobc_rows, gamma_rows, elev=elev, azim=azim, view_label=view_label)
        angle_outputs.append(out_png)
    shutil.copy2(VIS_PNG, OUT_PIC_ALL / VIS_PNG.name)
    shutil.copy2(REF_VIS_PNG, OUT_PIC_ALL / REF_VIS_PNG.name)
    shutil.copy2(REF_VIS_BUILDING_PNG, OUT_PIC_ALL / REF_VIS_BUILDING_PNG.name)
    shutil.copy2(WEST_TO_EAST_VIS_PNG, OUT_PIC_ALL / WEST_TO_EAST_VIS_PNG.name)
    shutil.copy2(GEObc_ONLY_WEST_TO_EAST_VIS_PNG, OUT_PIC_ALL / GEObc_ONLY_WEST_TO_EAST_VIS_PNG.name)
    shutil.copy2(GEObc_ONLY_SOUTHWEST_VIS_PNG, OUT_PIC_ALL / GEObc_ONLY_SOUTHWEST_VIS_PNG.name)
    for out_png in angle_outputs:
        shutil.copy2(out_png, OUT_PIC_ALL / out_png.name)
    sync_visualizations_to_geodata([VIS_PNG, REF_VIS_PNG, REF_VIS_BUILDING_PNG, WEST_TO_EAST_VIS_PNG, GEObc_ONLY_WEST_TO_EAST_VIS_PNG, GEObc_ONLY_SOUTHWEST_VIS_PNG, *angle_outputs])
    write_readme(building_faces, geobc_count, gamma_count, fids)

    print(f"selected_fids={fids}")
    print(f"building_faces={building_faces}")
    print(f"geobc_points={geobc_count}")
    print(f"gamma_dsm_points={gamma_count}")
    print(f"building_shp={BUILDING_SHP}")
    print(f"geobc_point_shp={GEObc_POINT_SHP}")
    print(f"gamma_point_shp={GAMMA_POINT_SHP}")
    print(f"geobc_point_csv={GEObc_POINT_CSV}")
    print(f"gamma_point_csv={GAMMA_POINT_CSV}")
    print(f"visualization={VIS_PNG}")
    print(f"reference_area_visualization={REF_VIS_PNG}")
    print(f"building_emphasis_visualization={REF_VIS_BUILDING_PNG}")
    print(f"west_to_east_visualization={WEST_TO_EAST_VIS_PNG}")
    print(f"geobc_only_west_to_east_visualization={GEObc_ONLY_WEST_TO_EAST_VIS_PNG}")
    print(f"geobc_only_southwest_visualization={GEObc_ONLY_SOUTHWEST_VIS_PNG}")
    for out_png in angle_outputs:
        print(f"angle_visualization={out_png}")


if __name__ == "__main__":
    main()
