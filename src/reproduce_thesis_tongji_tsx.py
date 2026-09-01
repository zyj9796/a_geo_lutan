from __future__ import annotations

import csv
import json
import math
import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath
from matplotlib.patches import Polygon as MplPolygon
from osgeo import ogr
from scipy.ndimage import binary_dilation
from scipy.optimize import least_squares

from geocode_gamma_rslc_with_buildings import (
    SPEED_OF_LIGHT,
    WGS84_A,
    WGS84_E2,
    doppler_hz,
    llh_to_ecef,
    make_orbit,
    parse_gamma_par,
    read_rslc_amplitude,
    solve_pixel_llh,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_RSLC_DIR = Path(r"E:\all_data\TSX\Tongji\RE_SLAVES")
WINDOWS_BUILDINGS_VECTOR = Path(
    r"E:\all_data\建筑矢量\2025年全国城市建筑房屋轮廓矢量数据shp带层数高程2000坐标建模"
    r"\截至2025-全国城市建筑房屋轮廓矢量数据\上海建筑轮廓数据\shanghai.shp"
)


def first_existing_path(candidates: list[str | Path | None]) -> Path:
    fallback: Path | None = None
    for item in candidates:
        if not item:
            continue
        path = Path(item)
        if fallback is None:
            fallback = path
        if path.exists():
            return path
    if fallback is None:
        raise ValueError("No path candidates were provided")
    return fallback


DEFAULT_RSLC_DIR = first_existing_path(
    [
        os.environ.get("TSX_RSLC_DIR"),
        REPO_ROOT.parent / "tsx_tongji" / "RE_SLAVES",
        REPO_ROOT / "tsx_tongji" / "RE_SLAVES",
        WINDOWS_RSLC_DIR,
    ]
)
DEFAULT_BUILDINGS_SHP = first_existing_path(
    [
        os.environ.get("TSX_BUILDINGS_VECTOR"),
        os.environ.get("TSX_BUILDINGS_SHP"),
        REPO_ROOT.parent.parent / "tongji_ps" / "tongji2" / "shp" / "tongji_clip.shp",
        REPO_ROOT.parent.parent / "tongji_ps" / "tongji3" / "shp" / "tongji_clip.shp",
        REPO_ROOT / "tsx_tongji_geocode" / "shanghai_buildings_subset.geojson",
        WINDOWS_BUILDINGS_VECTOR,
    ]
)
OUT_ROOT = Path("thesis_reproduction_tongji_tsx")


def ecef_to_llh(xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = map(float, xyz)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    for _ in range(8):
        sin_lat = math.sin(lat)
        n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        h = p / math.cos(lat) - n
        lat = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + h)))
    sin_lat = math.sin(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    h = p / math.cos(lat) - n
    return math.degrees(lon), math.degrees(lat), h


def local_en(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray]:
    east = (lon - lon0) * math.pi / 180.0 * WGS84_A * math.cos(math.radians(lat0))
    north = (lat - lat0) * math.pi / 180.0 * WGS84_A
    return east, north


def point_to_polygon_boundary_distance(points_en: np.ndarray, ring_en: np.ndarray) -> np.ndarray:
    seg_a = ring_en
    seg_b = np.roll(ring_en, -1, axis=0)
    out = np.full((points_en.shape[0],), np.inf, dtype=np.float64)
    for a, b in zip(seg_a, seg_b):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 0:
            continue
        ap = points_en - a
        t = np.clip((ap @ ab) / denom, 0.0, 1.0)
        proj = a + t[:, None] * ab
        out = np.minimum(out, np.linalg.norm(points_en - proj, axis=1))
    return out


def first_exterior_ring(geom) -> np.ndarray | None:
    if geom is None:
        return None
    if geom.GetGeometryName().upper() == "MULTIPOLYGON":
        best = None
        best_area = -1.0
        for i in range(geom.GetGeometryCount()):
            g = geom.GetGeometryRef(i)
            area = g.GetArea()
            if area > best_area:
                best = g
                best_area = area
        geom = best
    if geom is None or geom.GetGeometryName().upper() != "POLYGON":
        return None
    ring = geom.GetGeometryRef(0)
    pts = np.asarray([[ring.GetX(i), ring.GetY(i)] for i in range(ring.GetPointCount())], dtype=np.float64)
    if pts.shape[0] < 4:
        return None
    if np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def select_buildings(shp_path: Path, bounds: tuple[float, float, float, float], n: int = 4) -> list[dict]:
    ds = ogr.Open(str(shp_path))
    if ds is None:
        raise ValueError(f"Failed to open shapefile: {shp_path}")
    lyr = ds.GetLayer(0)
    min_lon, min_lat, max_lon, max_lat = bounds
    lyr.SetSpatialFilterRect(min_lon, min_lat, max_lon, max_lat)
    candidates = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        ring = first_exterior_ring(geom)
        if ring is None:
            continue
        height = float(feat.GetField("height") or 0.0)
        floor = float(feat.GetField("Floor") or 0.0)
        if height <= 0 and floor > 0:
            height = floor * 3.0
        if height <= 0:
            continue
        area = float(geom.GetArea())
        candidates.append(
            {
                "fid": int(feat.GetFID()),
                "floor": int(floor),
                "height_m": float(height),
                "area_deg2": area,
                "ring_lonlat": ring,
            }
        )
    candidates.sort(key=lambda b: (b["height_m"], b["area_deg2"]), reverse=True)
    picked = candidates[:n]
    if len(picked) < n:
        raise ValueError(f"Only found {len(picked)} buildings in TSX scene bounds")
    return picked


def project_llh_to_radar(lon: float, lat: float, h: float, par: dict, orbit) -> tuple[float, float]:
    pos_spline, vel_spline = orbit
    x = llh_to_ecef(lon, lat, h)
    wavelength = SPEED_OF_LIGHT / float(par["radar_frequency"])
    t_center = float(par["start_time"]) + 0.5 * float(par["azimuth_lines"]) * float(par["azimuth_line_time"])
    t_min = float(par["start_time"]) - 2.0
    t_max = float(par["start_time"]) + float(par["azimuth_lines"]) * float(par["azimuth_line_time"]) + 2.0

    def residual(t_arr: np.ndarray) -> np.ndarray:
        t = float(t_arr[0])
        sat = np.asarray(pos_spline(t), dtype=np.float64)
        vel = np.asarray(vel_spline(t), dtype=np.float64)
        los = x - sat
        rng = float(np.linalg.norm(los))
        fd = -2.0 * float(np.dot(los, vel)) / (wavelength * rng)
        return np.array([(fd - doppler_hz(par, rng)) / 20.0])

    res = least_squares(residual, np.array([t_center]), bounds=(t_min, t_max), max_nfev=80)
    t = float(res.x[0])
    sat = np.asarray(pos_spline(t), dtype=np.float64)
    rng = float(np.linalg.norm(x - sat))
    row = (t - float(par["start_time"])) / float(par["azimuth_line_time"])
    col = (rng - float(par["near_range_slc"])) / float(par["range_pixel_spacing"])
    return row, col


def build_model(building: dict) -> tuple[np.ndarray, np.ndarray]:
    ring = building["ring_lonlat"]
    bottom_h = float(building.get("base_height_m", 0.0))
    top_h = float(building.get("top_height_m", bottom_h + float(building["height_m"])))
    n = ring.shape[0]
    vertices = []
    for lon, lat in ring:
        vertices.append(llh_to_ecef(float(lon), float(lat), bottom_h))
    for lon, lat in ring:
        vertices.append(llh_to_ecef(float(lon), float(lat), top_h))
    vertices = np.asarray(vertices, dtype=np.float64)
    tris = []
    for i in range(n):
        j = (i + 1) % n
        tris.append((i, j, n + j))
        tris.append((i, n + j, n + i))
    for i in range(1, n - 1):
        tris.append((n, n + i, n + i + 1))
    return vertices, np.asarray(tris, dtype=np.int32)


def barycentric(p: np.ndarray, tri: np.ndarray) -> tuple[float, float, float] | None:
    a, b, c = tri
    v0 = b - a
    v1 = c - a
    v2 = p - a
    d00 = float(np.dot(v0, v0))
    d01 = float(np.dot(v0, v1))
    d11 = float(np.dot(v1, v1))
    d20 = float(np.dot(v2, v0))
    d21 = float(np.dot(v2, v1))
    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-12:
        return None
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    return u, v, w


def rasterize_building(building: dict, par: dict, orbit, image_shape: tuple[int, int]) -> dict:
    rows, cols = image_shape
    vertices_ecef, tris = build_model(building)
    proj = []
    for xyz in vertices_ecef:
        lon, lat, h = ecef_to_llh(xyz)
        row, col = project_llh_to_radar(lon, lat, h, par, orbit)
        proj.append((row, col))
    proj = np.asarray(proj, dtype=np.float64)

    mask = np.zeros((rows, cols), dtype=bool)
    tri_idx = np.full((rows, cols), -1, dtype=np.int32)
    for ti, tri in enumerate(tris):
        pts_rc = proj[tri]
        pts_xy = np.column_stack([pts_rc[:, 1], pts_rc[:, 0]])
        if not np.all(np.isfinite(pts_xy)):
            continue
        c0 = max(0, int(math.floor(np.min(pts_xy[:, 0]))) - 1)
        c1 = min(cols - 1, int(math.ceil(np.max(pts_xy[:, 0]))) + 1)
        r0 = max(0, int(math.floor(np.min(pts_xy[:, 1]))) - 1)
        r1 = min(rows - 1, int(math.ceil(np.max(pts_xy[:, 1]))) + 1)
        if c1 < c0 or r1 < r0:
            continue
        yy, xx = np.mgrid[r0 : r1 + 1, c0 : c1 + 1]
        inside = MplPath(pts_xy).contains_points(np.column_stack([xx.ravel(), yy.ravel()])).reshape(yy.shape)
        sub_mask = mask[r0 : r1 + 1, c0 : c1 + 1]
        sub_tri = tri_idx[r0 : r1 + 1, c0 : c1 + 1]
        new = inside & ~sub_mask
        sub_mask[new] = True
        sub_tri[new] = ti
    return {"vertices_ecef": vertices_ecef, "triangles": tris, "projected_rc": proj, "mask0": mask, "tri_idx": tri_idx}


def refine_mask(mask0: np.ndarray, amp: np.ndarray) -> np.ndarray:
    vals = amp[mask0]
    if vals.size == 0:
        return mask0.copy()
    thr = max(float(np.percentile(vals, 65)), float(vals.mean() + 0.25 * vals.std()))
    refined = mask0 & (amp >= thr)
    refined = binary_dilation(refined, iterations=1) & binary_dilation(mask0, iterations=2)
    return refined


def scatter_points_from_mask(model: dict, mask: np.ndarray, max_points: int = 2500) -> np.ndarray:
    rr, cc = np.nonzero(mask)
    if rr.size == 0:
        return np.zeros((0, 6), dtype=np.float64)
    step = max(1, int(math.ceil(rr.size / max_points)))
    rows = rr[::step]
    cols = cc[::step]
    out = []
    proj_xy = np.column_stack([model["projected_rc"][:, 1], model["projected_rc"][:, 0]])
    for r, c in zip(rows, cols):
        ti = int(model["tri_idx"][r, c])
        if ti < 0:
            continue
        tri = model["triangles"][ti]
        bc = barycentric(np.array([float(c), float(r)]), proj_xy[tri])
        if bc is None:
            continue
        u, v, w = bc
        if min(u, v, w) < -0.05:
            continue
        xyz = u * model["vertices_ecef"][tri[0]] + v * model["vertices_ecef"][tri[1]] + w * model["vertices_ecef"][tri[2]]
        lon, lat, h = ecef_to_llh(xyz)
        out.append((float(r), float(c), lon, lat, h, float(ti)))
    return np.asarray(out, dtype=np.float64)


def save_mask_plot(amp: np.ndarray, buildings: list[dict], models: list[dict], out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 7), dpi=180)
    ax.imshow(amp, cmap="gray")
    colors = ["#00ff66", "#ffcc00", "#2dd4ff", "#ff66cc"]
    for i, (building, model) in enumerate(zip(buildings, models), start=1):
        color = colors[(i - 1) % len(colors)]
        for tri in model["triangles"]:
            pts = np.column_stack([model["projected_rc"][tri, 1], model["projected_rc"][tri, 0]])
            ax.add_patch(MplPolygon(pts, fill=False, edgecolor=color, linewidth=0.35, alpha=0.55))
        rr, cc = np.nonzero(model["mask"])
        if rr.size:
            ax.scatter(cc[::20], rr[::20], s=1, c=color, alpha=0.55)
        cx = float(np.nanmean(model["projected_rc"][:, 1]))
        cy = float(np.nanmean(model["projected_rc"][:, 0]))
        ax.text(cx, cy, f"B{i}\\n{building['height_m']:.0f}m", color=color, fontsize=8, weight="bold")
    ax.set_xlim(0, amp.shape[1])
    ax.set_ylim(amp.shape[0], 0)
    ax.set_title("Initial model projection and refined masks")
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def save_pointcloud_plot(buildings: list[dict], points_by_building: list[np.ndarray], out_png: Path) -> None:
    fig = plt.figure(figsize=(9, 7), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#0ea5e9", "#f59e0b", "#22c55e", "#ec4899"]
    for i, (building, pts) in enumerate(zip(buildings, points_by_building), start=1):
        if pts.size == 0:
            continue
        ring = building["ring_lonlat"]
        lon0 = float(np.mean(ring[:, 0]))
        lat0 = float(np.mean(ring[:, 1]))
        e, n = local_en(pts[:, 2], pts[:, 3], lon0, lat0)
        ax.scatter(e, n, pts[:, 4], s=1.5, c=colors[(i - 1) % len(colors)], label=f"B{i}")
    ax.set_xlabel("East / m")
    ax.set_ylabel("North / m")
    ax.set_zlabel("Height / m")
    ax.set_title("Building-constrained 3D scatter points")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def zero_height_baseline(points: np.ndarray, par: dict, orbit, building: dict, max_points: int = 900) -> dict:
    if points.size == 0:
        return {}
    step = max(1, int(math.ceil(points.shape[0] / max_points)))
    sample = points[::step]
    ring = building["ring_lonlat"]
    lon0 = float(np.mean(ring[:, 0]))
    lat0 = float(np.mean(ring[:, 1]))
    re, rn = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
    out_lon = []
    out_lat = []
    errs = []
    last_xy = None
    for row, col in sample[:, :2]:
        xy0 = last_xy if last_xy is not None else (0.0, 0.0)
        lon, lat, ok, _err = solve_pixel_llh(float(row), float(col), par, orbit, xy0)
        last_xy = ((lon - float(par["center_longitude"])) * math.pi / 180.0 * WGS84_A * math.cos(math.radians(float(par["center_latitude"]))),
                   (lat - float(par["center_latitude"])) * math.pi / 180.0 * WGS84_A)
        out_lon.append(lon)
        out_lat.append(lat)
    if out_lon:
        pe, pn = local_en(np.asarray(out_lon), np.asarray(out_lat), lon0, lat0)
        d = point_to_polygon_boundary_distance(np.column_stack([pe, pn]), np.column_stack([re, rn]))
        return {
            "traditional_points": int(len(out_lon)),
            "traditional_mean_boundary_distance_m": float(np.mean(d)),
            "traditional_median_boundary_distance_m": float(np.median(d)),
            "traditional_p90_boundary_distance_m": float(np.percentile(d, 90)),
            "traditional_max_boundary_distance_m": float(np.max(d)),
        }
    return {}


def load_or_select_buildings(reference_date: str, bounds: tuple[float, float, float, float]) -> list[dict]:
    selected_path = OUT_ROOT / "selected_buildings_master.geojson"
    if selected_path.exists():
        data = json.loads(selected_path.read_text(encoding="utf-8"))
        out = []
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
            if np.allclose(ring[0], ring[-1]):
                ring = ring[:-1]
            out.append(
                {
                    "fid": int(props["fid"]),
                    "floor": int(props["floor"]),
                    "height_m": float(props["height_m"]),
                    "area_deg2": float(props.get("area_deg2", 0.0)),
                    "ring_lonlat": ring,
                }
            )
        if len(out) == 4:
            return out

    par = parse_gamma_par(DEFAULT_RSLC_DIR / f"{reference_date}.rslc.par")
    orbit = make_orbit(par)
    rows = int(par["azimuth_lines"])
    cols = int(par["range_samples"])
    amp = read_rslc_amplitude(DEFAULT_RSLC_DIR / f"{reference_date}.rslc", rows, cols)
    candidates = select_buildings(DEFAULT_BUILDINGS_SHP, bounds, n=30)
    buildings = []
    for candidate in candidates:
        model = rasterize_building(candidate, par, orbit, amp.shape)
        if int(model["mask0"].sum()) < 100:
            continue
        model["mask"] = refine_mask(model["mask0"], amp)
        if int(model["mask"].sum()) < 50:
            continue
        buildings.append(candidate)
        if len(buildings) == 4:
            break
    if len(buildings) < 4:
        raise ValueError(f"Only found {len(buildings)} valid projected buildings")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {k: b[k] for k in ("fid", "floor", "height_m", "area_deg2")},
                        "geometry": {"type": "Polygon", "coordinates": [b["ring_lonlat"].tolist() + [b["ring_lonlat"][0].tolist()]]},
                    }
                    for b in buildings
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return buildings


def run_scene(date: str, buildings: list[dict]) -> list[dict]:
    out_dir = OUT_ROOT / date
    out_dir.mkdir(parents=True, exist_ok=True)
    par = parse_gamma_par(DEFAULT_RSLC_DIR / f"{date}.rslc.par")
    orbit = make_orbit(par)
    rows = int(par["azimuth_lines"])
    cols = int(par["range_samples"])
    amp = read_rslc_amplitude(DEFAULT_RSLC_DIR / f"{date}.rslc", rows, cols)

    models = []
    valid_buildings = []
    for candidate in buildings:
        model = rasterize_building(candidate, par, orbit, amp.shape)
        model["mask"] = refine_mask(model["mask0"], amp)
        valid_buildings.append(candidate)
        models.append(model)

    points_by_building = []
    metrics = []
    for i, (building, model) in enumerate(zip(valid_buildings, models), start=1):
        pts = scatter_points_from_mask(model, model["mask"])
        points_by_building.append(pts)

        bdir = out_dir / f"building_{i}"
        bdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(bdir / "masks_and_projection.npz", mask0=model["mask0"], mask=model["mask"], projected_rc=model["projected_rc"])
        with (bdir / "scatter_points_wgs84.csv").open("w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["row", "col", "lon", "lat", "height_m", "triangle_index"])
            wr.writerows(pts.tolist())

        ring = building["ring_lonlat"]
        lon0 = float(np.mean(ring[:, 0]))
        lat0 = float(np.mean(ring[:, 1]))
        if pts.size:
            pe, pn = local_en(pts[:, 2], pts[:, 3], lon0, lat0)
            re, rn = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
            d = point_to_polygon_boundary_distance(np.column_stack([pe, pn]), np.column_stack([re, rn]))
            metric = {
                "scene": date,
                "building": i,
                "source_fid": building["fid"],
                "floor": building["floor"],
                "height_m": building["height_m"],
                "mask0_pixels": int(model["mask0"].sum()),
                "mask_pixels": int(model["mask"].sum()),
                "scatter_points": int(pts.shape[0]),
                "mean_boundary_distance_m": float(np.mean(d)),
                "median_boundary_distance_m": float(np.median(d)),
                "p90_boundary_distance_m": float(np.percentile(d, 90)),
                "max_boundary_distance_m": float(np.max(d)),
            }
            metric.update(zero_height_baseline(pts, par, orbit, building))
        else:
            metric = {
                "scene": date,
                "building": i,
                "source_fid": building["fid"],
                "floor": building["floor"],
                "height_m": building["height_m"],
                "mask0_pixels": int(model["mask0"].sum()),
                "mask_pixels": int(model["mask"].sum()),
                "scatter_points": 0,
            }
        metrics.append(metric)

    save_mask_plot(amp, valid_buildings, models, out_dir / "fig_initial_projection_and_refined_masks.png")
    save_pointcloud_plot(valid_buildings, points_by_building, out_dir / "fig_3d_scatter_points.png")
    (out_dir / "selected_buildings.geojson").write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {k: b[k] for k in ("fid", "floor", "height_m", "area_deg2")},
                        "geometry": {"type": "Polygon", "coordinates": [b["ring_lonlat"].tolist() + [b["ring_lonlat"][0].tolist()]]},
                    }
                    for b in valid_buildings
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for m in metrics for k in m})
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(metrics)
    print(f"Reproduction outputs: {out_dir}")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics


def write_summary(all_metrics: list[dict]) -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUT_ROOT / "all_scene_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for m in all_metrics for k in m})
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        wr.writerows(all_metrics)
    (OUT_ROOT / "all_scene_metrics.json").write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    scenes = sorted({m["scene"] for m in all_metrics})
    buildings = sorted({int(m["building"]) for m in all_metrics})
    mean_matrix = np.array(
        [[next(m for m in all_metrics if m["scene"] == s and int(m["building"]) == b).get("mean_boundary_distance_m", np.nan) for b in buildings] for s in scenes],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    x = np.arange(len(scenes))
    width = 0.18
    for j, b in enumerate(buildings):
        ax.bar(x + (j - 1.5) * width, mean_matrix[:, j], width=width, label=f"Building {b}")
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel("Mean boundary distance / m")
    ax.set_title("Multi-scene TSX building-constrained geocoding statistics")
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_ROOT / "fig_table_5_2_like_multiscene_mean_error.png")
    plt.close(fig)

    trad = [m for m in all_metrics if "traditional_mean_boundary_distance_m" in m]
    if trad:
        labels = [f"{m['scene']} B{m['building']}" for m in trad]
        method = [m["mean_boundary_distance_m"] for m in trad]
        base = [m["traditional_mean_boundary_distance_m"] for m in trad]
        fig, ax = plt.subplots(figsize=(11, 4.8), dpi=180)
        xx = np.arange(len(labels))
        ax.bar(xx - 0.2, base, width=0.4, label="Zero-height baseline")
        ax.bar(xx + 0.2, method, width=0.4, label="Building surface method")
        ax.set_xticks(xx)
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_ylabel("Mean boundary distance / m")
        ax.set_title("Traditional zero-height baseline vs building-constrained method")
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_ROOT / "fig_table_5_3_like_baseline_comparison.png")
        plt.close(fig)


def run(dates: list[str]) -> None:
    bounds = (121.48953764945898, 31.275678992265323, 121.50432264945898, 31.289738992265324)
    buildings = load_or_select_buildings(dates[0], bounds)
    all_metrics = []
    for date in dates:
        all_metrics.extend(run_scene(date, buildings))
    write_summary(all_metrics)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", default=["20200708", "20200730", "20200821"])
    args = ap.parse_args()
    run(args.dates)
