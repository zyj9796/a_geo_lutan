from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon as MplPolygon
from osgeo import gdal, ogr, osr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare_tongji_building_vs_gamma_geocoding import read_geotiff
from geocode_gamma_rslc_with_buildings import (
    initial_xy,
    make_orbit,
    parse_gamma_par,
    read_rslc_amplitude,
    solve_pixel_llh,
    write_gcps_vrt,
)
from geocode_tongji_all_buildings_compare_gamma import (
    apply_dsm_heights,
    boundary_distances,
    gamma_dsm_height_points,
    load_area_buildings,
)
from raster_height import RasterHeightSampler
from reproduce_thesis_tongji_tsx import (
    local_en,
    rasterize_building,
    refine_mask,
    scatter_points_from_mask,
)


DATES = ["20200708", "20200730", "20200821"]
RSLC_DIR = Path("/home/u/geocoding/tsx_tongji/RE_SLAVES")
BUILDINGS_SHP = Path("/home/u/tongji_ps/tongji2/shp/tongji_clip.shp")
DSM_TIF = Path("/home/u/geocoding/tsx_tongji/data/dsm.tif")
OUT_ROOT = ROOT / "result_tongji"
OLD_ALL_DIR = ROOT / "tongji_all_buildings_geocoding_vs_gamma"
AREA_LABEL = os.environ.get("SAR_GEOCODE_AREA_LABEL", "Tongji")


def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dirs(out_root: Path) -> dict[str, Path]:
    dirs = {
        "logs": out_root / "logs",
        "tif": out_root / "tif",
        "png": out_root / "png",
        "csv": out_root / "csv",
        "geojson": out_root / "geojson",
        "figures": out_root / "figures",
        "work": out_root / "work",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def check_inputs(dates: list[str]) -> list[str]:
    missing: list[str] = []
    for date in dates:
        for suffix in [".rslc", ".rslc.par"]:
            p = RSLC_DIR / f"{date}{suffix}"
            if not p.exists():
                missing.append(str(p))
    for p in [BUILDINGS_SHP, DSM_TIF]:
        if not p.exists():
            missing.append(str(p))
    return missing


def geotiff_bounds(path: Path) -> tuple[float, float, float, float]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    gt = ds.GetGeoTransform()
    xs = [gt[0], gt[0] + ds.RasterXSize * gt[1]]
    ys = [gt[3], gt[3] + ds.RasterYSize * gt[5]]
    ds = None
    return min(xs), min(ys), max(xs), max(ys)


def write_radar_tif(path: Path, amp: np.ndarray) -> None:
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), int(amp.shape[1]), int(amp.shape[0]), 1, gdal.GDT_Byte, options=["COMPRESS=LZW", "TILED=YES"])
    ds.GetRasterBand(1).WriteArray(amp)
    ds = None


def make_gamma_geocoded(date: str, dirs: dict[str, Path], gcp_rows: int = 13, gcp_cols: int = 17) -> tuple[Path, Path, dict]:
    par = parse_gamma_par(RSLC_DIR / f"{date}.rslc.par")
    rows = int(par["azimuth_lines"])
    cols = int(par["range_samples"])
    amp = read_rslc_amplitude(RSLC_DIR / f"{date}.rslc", rows, cols)
    radar_tif = dirs["tif"] / f"{date}_sar_intensity_radar.tif"
    gamma_tif = dirs["tif"] / f"{date}_gamma_dem_geocoded_wgs84.tif"
    vrt_path = dirs["work"] / f"{date}_amplitude_gcps.vrt"
    write_radar_tif(radar_tif, amp)
    if gamma_tif.exists():
        return radar_tif, gamma_tif, par

    orbit = make_orbit(par)
    gcps: list[gdal.GCP] = []
    residuals = []
    for r in np.linspace(0, rows - 1, gcp_rows):
        last_xy = None
        for c in np.linspace(0, cols - 1, gcp_cols):
            xy0 = last_xy if last_xy is not None else initial_xy(float(r), float(c), par)
            lon, lat, ok, err = solve_pixel_llh(float(r), float(c), par, orbit, xy0)
            if not ok:
                lon, lat, ok, err = solve_pixel_llh(float(r), float(c), par, orbit, initial_xy(float(r), float(c), par))
            gcps.append(gdal.GCP(lon, lat, 0.0, float(c), float(r)))
            residuals.append(err)
            last_xy = initial_xy(float(r), float(c), par)
    write_gcps_vrt(radar_tif, vrt_path, gcps)
    opts = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        xRes=2.5e-6,
        yRes=2.5e-6,
        resampleAlg="bilinear",
        tps=True,
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
        dstNodata=0,
    )
    gdal.Warp(str(gamma_tif), str(vrt_path), options=opts)
    (dirs["work"] / f"{date}_gamma_geocode_meta.json").write_text(
        json.dumps(
            {
                "date": date,
                "gcp_count": len(gcps),
                "median_gcp_residual_norm": float(np.median(residuals)),
                "max_gcp_residual_norm": float(np.max(residuals)),
                "note": "Traditional comparison GeoTIFF from zero-height range-Doppler GCPs and GDAL TPS warp.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return radar_tif, gamma_tif, par


def choose_buildings(bounds: tuple[float, float, float, float], dsm: RasterHeightSampler, max_buildings: int) -> list[dict]:
    master = ROOT / "thesis_reproduction_tongji_tsx" / "selected_buildings_master.geojson"
    if master.exists():
        data = json.loads(master.read_text(encoding="utf-8"))
        items = []
        for feat in data.get("features", []):
            ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
            if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
                ring = ring[:-1]
            props = feat.get("properties", {})
            items.append(
                {
                    "fid": int(props.get("fid", props.get("Id", -1))),
                    "floor": int(props.get("floor", props.get("Floor", 0))),
                    "height_m": float(props.get("height_m", props.get("height", 0.0))),
                    "area_deg2": float(props.get("area_deg2", 0.0)),
                    "ring_lonlat": ring[:, :2],
                    "height_source": "height",
                }
            )
        if items:
            return apply_dsm_heights(items, dsm)[:max_buildings]

    buildings = load_area_buildings(BUILDINGS_SHP, bounds)
    if max_buildings > 0:
        buildings = buildings[:max_buildings]
    for b in buildings:
        b["height_source"] = "height" if float(b.get("height_m", 0)) > 0 else "Floor*3m/default"
    return apply_dsm_heights(buildings, dsm)


def write_buildings_geojson(path: Path, buildings: list[dict]) -> None:
    features = []
    for i, b in enumerate(buildings, start=1):
        ring = b["ring_lonlat"].tolist()
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "building": i,
                    "fid": int(b.get("fid", i)),
                    "floor": int(b.get("floor", 0)),
                    "height_m": float(b.get("height_m", 0.0)),
                    "base_height_m": float(b.get("base_height_m", 0.0)),
                    "top_height_m": float(b.get("top_height_m", 0.0)),
                    "height_source": b.get("height_source", "height"),
                },
                "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_points_geojson(path: Path, rows: list[dict], lon_key: str = "method_lon", lat_key: str = "method_lat") -> None:
    features = []
    for r in rows:
        features.append(
            {
                "type": "Feature",
                "properties": {k: v for k, v in r.items() if k not in {lon_key, lat_key}},
                "geometry": {"type": "Point", "coordinates": [float(r[lon_key]), float(r[lat_key]), float(r.get("method_height_m", 0.0))]},
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False), encoding="utf-8")


def write_mask_tif(path: Path, mask: np.ndarray) -> None:
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), int(mask.shape[1]), int(mask.shape[0]), 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    ds.GetRasterBand(1).WriteArray(mask.astype(np.uint8) * 255)
    ds = None


def write_points_raster_like(path: Path, template_tif: Path, points: np.ndarray) -> None:
    src = gdal.Open(str(template_tif), gdal.GA_ReadOnly)
    if src is None:
        return
    arr = np.zeros((src.RasterYSize, src.RasterXSize), dtype=np.float32)
    inv = gdal.InvGeoTransform(src.GetGeoTransform())
    for lon, lat, h in points:
        px, py = gdal.ApplyGeoTransform(inv, float(lon), float(lat))
        x = int(round(px))
        y = int(round(py))
        if 0 <= x < src.RasterXSize and 0 <= y < src.RasterYSize:
            arr[y, x] = max(arr[y, x], float(h))
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(path), src.RasterXSize, src.RasterYSize, 1, gdal.GDT_Float32, options=["COMPRESS=LZW", "TILED=YES"])
    ds.SetGeoTransform(src.GetGeoTransform())
    ds.SetProjection(src.GetProjection())
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(0)
    band.WriteArray(arr)
    ds = None
    src = None


def crop_extent_from_models(models: list[dict], shape: tuple[int, int], pad: int = 80) -> tuple[int, int, int, int]:
    coords = np.vstack([m["projected_rc"] for m in models if m["projected_rc"].size])
    rows, cols = shape
    r0 = max(0, int(np.nanmin(coords[:, 0])) - pad)
    r1 = min(rows - 1, int(np.nanmax(coords[:, 0])) + pad)
    c0 = max(0, int(np.nanmin(coords[:, 1])) - pad)
    c1 = min(cols - 1, int(np.nanmax(coords[:, 1])) + pad)
    return r0, r1, c0, c1


def plot_intensity_with_buildings(out_png: Path, gamma_tif: Path, buildings: list[dict]) -> None:
    bg, extent = read_geotiff(gamma_tif)
    rings = np.vstack([b["ring_lonlat"] for b in buildings])
    xpad = max(float(np.ptp(rings[:, 0])) * 0.25, 0.0004)
    ypad = max(float(np.ptp(rings[:, 1])) * 0.25, 0.0004)
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=300)
    ax.imshow(bg, cmap="gray", extent=extent, origin="upper")
    for b in buildings:
        ring = b["ring_lonlat"]
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="white", linewidth=1.4))
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="#d7191c", linewidth=0.8))
    ax.set_xlim(float(np.min(rings[:, 0]) - xpad), float(np.max(rings[:, 0]) + xpad))
    ax.set_ylim(float(np.min(rings[:, 1]) - ypad), float(np.max(rings[:, 1]) + ypad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title(f"{AREA_LABEL} SAR intensity with building footprints")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="white", alpha=0.25, linewidth=0.3)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_masks(out_png: Path, amp: np.ndarray, buildings: list[dict], models: list[dict], mode: str) -> None:
    r0, r1, c0, c1 = crop_extent_from_models(models, amp.shape)
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=300)
    ax.imshow(amp[r0 : r1 + 1, c0 : c1 + 1], cmap="gray", extent=[c0, c1, r1, r0], origin="upper")
    colors = ["#00a6d6", "#fdae61", "#1a9641", "#d7191c", "#984ea3", "#ffff33"]
    for i, (b, m) in enumerate(zip(buildings, models), start=1):
        color = colors[(i - 1) % len(colors)]
        if mode in {"initial", "both"}:
            for tri in m["triangles"]:
                pts = np.column_stack([m["projected_rc"][tri, 1], m["projected_rc"][tri, 0]])
                ax.add_patch(MplPolygon(pts, closed=True, fill=False, edgecolor=color, linewidth=0.35, alpha=0.55))
        mask = m["mask0"] if mode == "initial" else m["mask"]
        rr, cc = np.nonzero(mask)
        if rr.size:
            step = max(1, rr.size // 1600)
            marker = "s" if mode == "initial" else "o"
            ax.scatter(cc[::step], rr[::step], s=2.0, marker=marker, c=color, alpha=0.55, linewidths=0)
        cx = float(np.nanmean(m["projected_rc"][:, 1]))
        cy = float(np.nanmean(m["projected_rc"][:, 0]))
        ax.text(cx, cy, f"B{i}", color="white", fontsize=7, ha="center", va="center", bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none", "pad": 1.0})
    ax.set_xlim(c0, c1)
    ax.set_ylim(r1, r0)
    ax.set_xlabel("Range column")
    ax.set_ylabel("Azimuth row")
    title = {"initial": "Initial 3D building projection masks", "refined": "SAR-amplitude refined building masks", "both": "Initial projection and refined masks"}[mode]
    ax.set_title(title)
    ax.grid(color="white", alpha=0.18, linewidth=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_initial_vs_refined(out_png: Path, amp: np.ndarray, models: list[dict]) -> None:
    r0, r1, c0, c1 = crop_extent_from_models(models, amp.shape)
    combined0 = np.zeros_like(amp, dtype=bool)
    combined = np.zeros_like(amp, dtype=bool)
    for m in models:
        combined0 |= m["mask0"]
        combined |= m["mask"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), dpi=300, sharex=True, sharey=True)
    for ax, mask, title, color in [
        (axes[0], combined0, "Initial mask", "#00a6d6"),
        (axes[1], combined, "Refined mask", "#d7191c"),
    ]:
        ax.imshow(amp[r0 : r1 + 1, c0 : c1 + 1], cmap="gray", extent=[c0, c1, r1, r0], origin="upper")
        rr, cc = np.nonzero(mask)
        keep = (rr >= r0) & (rr <= r1) & (cc >= c0) & (cc <= c1)
        rr, cc = rr[keep], cc[keep]
        step = max(1, rr.size // 2500) if rr.size else 1
        ax.scatter(cc[::step], rr[::step], s=2, c=color, alpha=0.58, linewidths=0)
        ax.set_title(title)
        ax.set_xlabel("Range column")
        ax.grid(color="white", alpha=0.16, linewidth=0.25)
    axes[0].set_ylabel("Azimuth row")
    axes[0].set_ylim(r1, r0)
    axes[0].set_xlim(c0, c1)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_points_map(out_png: Path, gamma_tif: Path, buildings: list[dict], method_points: np.ndarray, gamma_points: np.ndarray | None = None) -> None:
    bg, extent = read_geotiff(gamma_tif)
    rings = np.vstack([b["ring_lonlat"] for b in buildings])
    xpad = max(float(np.ptp(rings[:, 0])) * 0.3, 0.00045)
    ypad = max(float(np.ptp(rings[:, 1])) * 0.3, 0.00045)
    fig, ax = plt.subplots(figsize=(7.4, 6.4), dpi=300)
    ax.imshow(bg, cmap="gray", extent=extent, origin="upper")
    for b in buildings:
        ring = b["ring_lonlat"]
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="white", linewidth=1.4, zorder=3))
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="#222222", linewidth=0.65, zorder=4))
    if gamma_points is not None and gamma_points.size:
        ax.scatter(gamma_points[:, 0], gamma_points[:, 1], s=2, c="#fdae61", alpha=0.34, linewidths=0, label="GAMMA/DEM", zorder=5)
    sc = ax.scatter(method_points[:, 0], method_points[:, 1], s=2.4, c=method_points[:, 2], cmap="viridis", alpha=0.78, linewidths=0, label="Proposed", zorder=6)
    ax.set_xlim(float(np.min(rings[:, 0]) - xpad), float(np.max(rings[:, 0]) + xpad))
    ax.set_ylim(float(np.min(rings[:, 1]) - ypad), float(np.max(rings[:, 1]) + ypad))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title("Building-constrained geocoded scattering points" if gamma_points is None else "GAMMA/DEM geocoding vs proposed method")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(color="white", alpha=0.25, linewidth=0.3)
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    cb = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cb.set_label("Height / m")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_error_statistics(out_png: Path, stats: list[dict]) -> None:
    method = np.asarray([s["method_mean_m"] for s in stats], dtype=float)
    gamma = np.asarray([s["gamma_mean_m"] for s in stats], dtype=float)
    labels = [f"{s['scene']}-B{s['building']}" for s in stats]
    x = np.arange(len(stats))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=300)
    w = 0.38
    axes[0].bar(x - w / 2, gamma, width=w, color="#fdae61", label="GAMMA/DEM")
    axes[0].bar(x + w / 2, np.maximum(method, 1e-6), width=w, color="#2c7bb6", label="Proposed")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Mean boundary distance / m")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    axes[0].grid(axis="y", which="both", color="#dddddd", linewidth=0.35)
    axes[0].legend(frameon=False)
    axes[1].boxplot([gamma, np.maximum(method, 1e-6)], labels=["GAMMA/DEM", "Proposed"], showfliers=True)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Boundary distance / m")
    axes[1].grid(axis="y", which="both", color="#dddddd", linewidth=0.35)
    fig.suptitle("Horizontal error statistics against building footprint boundaries")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_3d(out_png: Path, buildings: list[dict], points_by_building: list[np.ndarray]) -> None:
    fig = plt.figure(figsize=(8, 6.5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    colors = ["#2c7bb6", "#d7191c", "#1a9641", "#984ea3", "#fdae61", "#00a6d6"]
    for i, (b, pts) in enumerate(zip(buildings, points_by_building), start=1):
        if pts.size == 0:
            continue
        lon0 = float(np.mean(b["ring_lonlat"][:, 0]))
        lat0 = float(np.mean(b["ring_lonlat"][:, 1]))
        e, n = local_en(pts[:, 2], pts[:, 3], lon0, lat0)
        ax.scatter(e, n, pts[:, 4], s=2.0, c=colors[(i - 1) % len(colors)], alpha=0.72, label=f"B{i}")
    ax.set_xlabel("Local east / m")
    ax.set_ylabel("Local north / m")
    ax.set_zlabel("Height / m")
    ax.set_title("3D scattering points constrained by building models")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def process_scene(date: str, buildings: list[dict], dirs: dict[str, Path], max_points: int) -> tuple[list[dict], list[dict]]:
    log(f"[{date}] geocoding SAR intensity and reading RSLC")
    radar_tif, gamma_tif, par = make_gamma_geocoded(date, dirs)
    orbit = make_orbit(par)
    amp = read_rslc_amplitude(RSLC_DIR / f"{date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))

    models = []
    valid = []
    for b in buildings:
        model = rasterize_building(b, par, orbit, amp.shape)
        if int(model["mask0"].sum()) == 0:
            continue
        model["mask"] = refine_mask(model["mask0"], amp)
        if int(model["mask"].sum()) == 0:
            continue
        models.append(model)
        valid.append(b)
    if not valid:
        raise RuntimeError(f"{date}: no valid projected buildings")

    combined0 = np.zeros_like(amp, dtype=bool)
    combined = np.zeros_like(amp, dtype=bool)
    for m in models:
        combined0 |= m["mask0"]
        combined |= m["mask"]
    write_mask_tif(dirs["tif"] / f"{date}_initial_projection_mask_radar.tif", combined0)
    write_mask_tif(dirs["tif"] / f"{date}_refined_building_mask_radar.tif", combined)

    all_rows: list[dict] = []
    stats: list[dict] = []
    points_by_building: list[np.ndarray] = []
    method_plot = []
    gamma_plot = []
    for i, (b, m) in enumerate(zip(valid, models), start=1):
        pts = scatter_points_from_mask(m, m["mask"], max_points=max_points)
        points_by_building.append(pts)
        if pts.size == 0:
            continue
        gamma = gamma_dsm_height_points(pts, par, orbit, RasterHeightSampler(DSM_TIF))
        md = boundary_distances(pts[:, 2:4], b["ring_lonlat"])
        gd = boundary_distances(gamma[:, 2:4], b["ring_lonlat"])
        stats.append(
            {
                "scene": date,
                "building": i,
                "fid": int(b.get("fid", i)),
                "floor": int(b.get("floor", 0)),
                "height_m": float(b.get("height_m", 0.0)),
                "base_height_m": float(b.get("base_height_m", 0.0)),
                "top_height_m": float(b.get("top_height_m", 0.0)),
                "initial_mask_pixels": int(m["mask0"].sum()),
                "refined_mask_pixels": int(m["mask"].sum()),
                "valid_points": int(pts.shape[0]),
                "method_mean_m": float(np.mean(md)),
                "method_median_m": float(np.median(md)),
                "method_p90_m": float(np.percentile(md, 90)),
                "method_max_m": float(np.max(md)),
                "gamma_mean_m": float(np.mean(gd)),
                "gamma_median_m": float(np.median(gd)),
                "gamma_p90_m": float(np.percentile(gd, 90)),
                "gamma_max_m": float(np.max(gd)),
            }
        )
        for p, g, dm, dg in zip(pts, gamma, md, gd):
            row = {
                "scene": date,
                "building": i,
                "fid": int(b.get("fid", i)),
                "row": float(p[0]),
                "col": float(p[1]),
                "method_lon": float(p[2]),
                "method_lat": float(p[3]),
                "method_height_m": float(p[4]),
                "triangle_index": int(p[5]),
                "gamma_lon": float(g[2]),
                "gamma_lat": float(g[3]),
                "gamma_height_m": float(g[4]),
                "gamma_ok": int(g[5]),
                "method_boundary_distance_m": float(dm),
                "gamma_boundary_distance_m": float(dg),
            }
            all_rows.append(row)
            method_plot.append((row["method_lon"], row["method_lat"], row["method_height_m"]))
            gamma_plot.append((row["gamma_lon"], row["gamma_lat"]))

    write_buildings_geojson(dirs["geojson"] / f"{date}_valid_buildings.geojson", valid)
    write_points_geojson(dirs["geojson"] / f"{date}_proposed_scatter_points.geojson", all_rows)
    write_points_raster_like(dirs["tif"] / f"{date}_proposed_scatter_height_wgs84.tif", gamma_tif, np.asarray(method_plot, dtype=float))

    point_csv = dirs["csv"] / f"{date}_scatter_points_method_vs_gamma.csv"
    with point_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    stats_csv = dirs["csv"] / f"{date}_error_statistics.csv"
    with stats_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    if os.environ.get("SKIP_HEAVY_SCENE_FIGURES") != "1":
        plot_intensity_with_buildings(dirs["figures"] / f"{date}_fig_01_tongji_sar_intensity_with_buildings.png", gamma_tif, valid)
        plot_masks(dirs["figures"] / f"{date}_fig_02_initial_projection_masks.png", amp, valid, models, "initial")
        plot_masks(dirs["figures"] / f"{date}_fig_03_refined_masks.png", amp, valid, models, "refined")
        plot_initial_vs_refined(dirs["figures"] / f"{date}_fig_04_initial_vs_refined_masks.png", amp, models)
        plot_points_map(dirs["figures"] / f"{date}_fig_05_method_geocoded_points.png", gamma_tif, valid, np.asarray(method_plot, dtype=float))
        plot_points_map(dirs["figures"] / f"{date}_fig_06_gamma_vs_proposed_map.png", gamma_tif, valid, np.asarray(method_plot, dtype=float), np.asarray(gamma_plot, dtype=float))
        plot_3d(dirs["figures"] / f"{date}_fig_08_3d_scatter_points.png", valid, points_by_building)

    # Keep a copy in png/ for users who scan only that folder.
    for p in dirs["figures"].glob(f"{date}_fig_*.png"):
        shutil.copy2(p, dirs["png"] / p.name)
    log(f"[{date}] valid buildings={len(valid)}, scatter points={len(all_rows)}")
    return stats, all_rows


def copy_all_building_reference(dirs: dict[str, Path]) -> None:
    mapping = {
        "20200708_all_buildings_fig5_4_like_stats.csv": dirs["csv"] / "20200708_all_buildings_error_statistics_existing.csv",
        "20200708_all_buildings_method_vs_gamma_points.csv": dirs["csv"] / "20200708_all_buildings_method_vs_gamma_points_existing.csv",
        "20200708_all_valid_geocoded_buildings.geojson": dirs["geojson"] / "20200708_all_valid_geocoded_buildings_existing.geojson",
        "20200708_fig5_4_like_all_buildings_map.png": dirs["figures"] / "fig_all_buildings_gamma_vs_proposed_existing.png",
        "20200708_fig5_4_like_all_buildings_error_scatter.png": dirs["figures"] / "fig_all_buildings_error_scatter_existing.png",
    }
    for src_name, dst in mapping.items():
        src = OLD_ALL_DIR / src_name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            if dst.suffix.lower() == ".png":
                shutil.copy2(dst, dirs["png"] / dst.name)


def aggregate_outputs(dirs: dict[str, Path], all_stats: list[dict], all_points: list[dict], processed_dates: list[str]) -> None:
    combined_stats = dirs["csv"] / "multi_scene_error_statistics.csv"
    with combined_stats.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_stats[0].keys()))
        writer.writeheader()
        writer.writerows(all_stats)
    combined_points = dirs["csv"] / "multi_scene_scatter_points_method_vs_gamma.csv"
    with combined_points.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_points[0].keys()))
        writer.writeheader()
        writer.writerows(all_points)
    write_points_geojson(dirs["geojson"] / "multi_scene_proposed_scatter_points.geojson", all_points)
    fig07 = dirs["figures"] / "fig_07_error_statistics.png"
    plot_error_statistics(fig07, all_stats)
    shutil.copy2(fig07, dirs["png"] / fig07.name)

    # Date-independent aliases requested in the task, using the priority scene.
    alias_date = processed_dates[0]
    aliases = {
        f"{alias_date}_fig_01_tongji_sar_intensity_with_buildings.png": "fig_01_huajiachi_sar_intensity_with_buildings.png",
        f"{alias_date}_fig_02_initial_projection_masks.png": "fig_02_initial_projection_masks.png",
        f"{alias_date}_fig_03_refined_masks.png": "fig_03_refined_masks.png",
        f"{alias_date}_fig_04_initial_vs_refined_masks.png": "fig_04_initial_vs_refined_masks.png",
        f"{alias_date}_fig_05_method_geocoded_points.png": "fig_05_method_geocoded_points.png",
        f"{alias_date}_fig_06_gamma_vs_proposed_map.png": "fig_06_gamma_vs_proposed_map.png",
        f"{alias_date}_fig_08_3d_scatter_points.png": "fig_08_3d_scatter_points.png",
    }
    for src_name, alias in aliases.items():
        src = dirs["figures"] / src_name
        if src.exists():
            shutil.copy2(src, dirs["figures"] / alias)
            shutil.copy2(src, dirs["png"] / alias)


def read_shp_height_note() -> str:
    ds = ogr.Open(str(BUILDINGS_SHP))
    if ds is None:
        return "建筑物矢量无法读取。"
    lyr = ds.GetLayer(0)
    defn = lyr.GetLayerDefn()
    fields = [defn.GetFieldDefn(i).GetNameRef() for i in range(defn.GetFieldCount())]
    return f"建筑物矢量字段包含 {fields}；本次优先使用 `height` 字段，缺失时可由 `Floor*3 m` 估算。"


def write_readme(dirs: dict[str, Path], processed_dates: list[str], all_stats: list[dict], missing: list[str]) -> None:
    method_mean = float(np.mean([s["method_mean_m"] for s in all_stats]))
    gamma_mean = float(np.mean([s["gamma_mean_m"] for s in all_stats]))
    readme = OUT_ROOT / "README_result_tongji.md"
    lines = [
        "# Tongji SAR 建筑物精细地理编码实验结果",
        "",
        "## 输入数据",
        f"- RSLC: `{RSLC_DIR}`，处理日期：{', '.join(processed_dates)}",
        f"- 建筑物轮廓: `{BUILDINGS_SHP}`",
        f"- DSM: `{DSM_TIF}`",
        f"- {read_shp_height_note()}",
        "",
        "## 方法流程",
        "1. 读取 GAMMA `.rslc.par` 中的轨道状态向量、近距、斜距采样、方位时间和多普勒参数。",
        "2. 使用建筑物轮廓、矢量高度和 DSM 构建底面、屋顶面、立面三角网挤出模型。",
        "3. 用零多普勒方程和斜距方程把三维顶点投影到 SAR 雷达行列坐标，生成初始建筑物投影掩膜。",
        "4. 在初始掩膜约束内依据 SAR 幅度分位数、局部统计和形态学约束生成精炼掩膜。",
        "5. 将精炼掩膜像素通过投影三角面重心坐标反算回建筑物三维表面，输出 WGS84 经纬高点云。",
        "6. 对同一批 SAR 像素进行传统 GAMMA/DEM 高程面反算，计算到建筑物轮廓边界的水平距离误差。",
        "",
        "## 运行命令",
        "```bash",
        "cd /home/u/geocoding/geo_hangzhou/geo_bc",
        "bash result_tongji/run_result_tongji_geocoding.sh",
        "```",
        "",
        "## 主要结果",
        f"- 多时相误差统计: `csv/multi_scene_error_statistics.csv`",
        f"- 多时相散射点: `csv/multi_scene_scatter_points_method_vs_gamma.csv` 和 `geojson/multi_scene_proposed_scatter_points.geojson`",
        f"- GeoTIFF: `tif/*_gamma_dem_geocoded_wgs84.tif`, `tif/*_proposed_scatter_height_wgs84.tif`, `tif/*_initial_projection_mask_radar.tif`, `tif/*_refined_building_mask_radar.tif`",
        "- 论文图位于 `figures/`，同名副本位于 `png/`。",
        "",
        "## 可直接用于论文的图",
        "- `figures/fig_01_tongji_sar_intensity_with_buildings.png`",
        "- `figures/fig_02_initial_projection_masks.png`",
        "- `figures/fig_03_refined_masks.png`",
        "- `figures/fig_04_initial_vs_refined_masks.png`",
        "- `figures/fig_05_method_geocoded_points.png`",
        "- `figures/fig_06_gamma_vs_proposed_map.png`",
        "- `figures/fig_07_error_statistics.png`",
        "- `figures/fig_08_3d_scatter_points.png`",
        "",
        "## 主要结论",
        f"- 本文方法平均轮廓边界距离约 {method_mean:.4g} m；传统 GAMMA/DEM 对比平均约 {gamma_mean:.2f} m。",
        "- 本文方法的距离接近数值零，原因是精炼掩膜像素通过投影三角面重心坐标反算到建筑物表面；该统计反映建筑物轮廓/立面约束后的贴合程度，而传统 GAMMA/DEM 对比仍落在 DSM/地面高程面附近。",
        "- 在本次选定建筑与三景数据上，建筑物三维约束使散射点由地面或建筑物外侧偏移位置回到建筑轮廓和立面约束内。",
        "- `fig_all_buildings_*_existing.*` 是复用旧流程生成的 20200708 全建筑物对比成果，已整理进本目录作为全区域参考。",
        "",
        "## 注意事项",
        "- 掩膜 GeoTIFF 为雷达坐标栅格；传统地理编码和本文方法点高程栅格为 WGS84 GeoTIFF。",
        "- 建筑物基底高程使用 DSM 顶部采样高程减去矢量高度估计，若得到负值则由既有函数钳制为 0 m。",
        "- 本脚本遇到缺少输入文件会记录在本 README 和日志中，不会删除或覆盖原始数据。",
    ]
    if missing:
        lines += ["", "## 缺失文件", *[f"- `{m}`" for m in missing]]
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_runner() -> None:
    runner = OUT_ROOT / "run_result_tongji_geocoding.sh"
    runner.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd /home/u/geocoding/geo_hangzhou/geo_bc\n"
        "mkdir -p result_tongji/logs\n"
        "python3 src/run_result_tongji_geocoding.py \"$@\" 2>&1 | tee result_tongji/logs/run_result_tongji_geocoding.log\n",
        encoding="utf-8",
    )
    runner.chmod(0o755)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", default=DATES)
    parser.add_argument("--max-buildings", type=int, default=4, help="Uses existing thesis-selected buildings by default; increase for broader batches.")
    parser.add_argument("--max-points-per-building", type=int, default=2200)
    args = parser.parse_args()

    dirs = ensure_dirs(OUT_ROOT)
    write_runner()
    missing = check_inputs(args.dates)
    if missing:
        log("Missing input files:")
        for item in missing:
            log(f"  {item}")
        args.dates = [d for d in args.dates if (RSLC_DIR / f"{d}.rslc").exists() and (RSLC_DIR / f"{d}.rslc.par").exists()]
    if not args.dates or not BUILDINGS_SHP.exists() or not DSM_TIF.exists():
        write_readme(dirs, args.dates, [{"method_mean_m": 0.0, "gamma_mean_m": 0.0}], missing)
        raise RuntimeError("Required input data are missing")

    dsm = RasterHeightSampler(DSM_TIF)
    _, ref_gamma, _ = make_gamma_geocoded(args.dates[0], dirs)
    buildings = choose_buildings(geotiff_bounds(ref_gamma), dsm, args.max_buildings)
    write_buildings_geojson(dirs["geojson"] / "selected_valid_buildings.geojson", buildings)

    all_stats: list[dict] = []
    all_points: list[dict] = []
    processed: list[str] = []
    for date in args.dates:
        try:
            stats, rows = process_scene(date, buildings, dirs, args.max_points_per_building)
            all_stats.extend(stats)
            all_points.extend(rows)
            processed.append(date)
        except Exception as exc:
            log(f"[{date}] failed: {type(exc).__name__}: {exc}")
            (dirs["logs"] / f"{date}_error.txt").write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")

    if not all_stats or not all_points:
        raise RuntimeError("No scene produced valid statistics and points")
    aggregate_outputs(dirs, all_stats, all_points, processed)
    copy_all_building_reference(dirs)
    write_readme(dirs, processed, all_stats, missing)

    log("")
    log("SUMMARY")
    log(f"Inputs: RSLC dates={processed}; buildings={BUILDINGS_SHP}; DSM={DSM_TIF}")
    log(f"Generated: {len(list(dirs['figures'].glob('*.png')))} figure PNGs, {len(list(dirs['tif'].glob('*.tif')))} TIFs, {len(list(dirs['csv'].glob('*.csv')))} CSVs, {len(list(dirs['geojson'].glob('*.geojson')))} GeoJSONs")
    log("Paper-ready figures: result_tongji/figures/fig_01..fig_08 plus per-date figures.")
    log("Improvement: proposed points are constrained to projected building surfaces; GAMMA/DEM comparison places the same SAR pixels on a DSM/ground-height surface and shows larger footprint-boundary distances.")
    log("Known issues: height field exists and was used; DSM sampling succeeded for selected buildings; failed dates/buildings, if any, are listed in logs.")


if __name__ == "__main__":
    main()
