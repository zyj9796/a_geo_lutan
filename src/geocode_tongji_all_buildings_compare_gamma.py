from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon as MplPolygon
from osgeo import gdal, ogr
from scipy.optimize import least_squares

from compare_tongji_building_vs_gamma_geocoding import read_geotiff
from geocode_gamma_rslc_with_buildings import (
    SPEED_OF_LIGHT,
    doppler_hz,
    enu_to_llh,
    initial_xy,
    llh_to_ecef,
    make_orbit,
    parse_gamma_par,
    read_rslc_amplitude,
    write_gcps_vrt,
)
from raster_height import RasterHeightSampler
from reproduce_thesis_tongji_tsx import (
    DEFAULT_BUILDINGS_SHP,
    DEFAULT_RSLC_DIR,
    first_exterior_ring,
    local_en,
    point_to_polygon_boundary_distance,
    rasterize_building,
    refine_mask,
    scatter_points_from_mask,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAMMA_TIF = ROOT / "tsx_tongji_geocode" / "20200708_amplitude_geocoded_wgs84.tif"
DEFAULT_OUT_DIR = ROOT / "tongji_all_buildings_geocoding_vs_gamma"
DEFAULT_DSM = Path("/home/u/geocoding/tsx_tongji/data/dsm.tif")


def geotiff_bounds(path: Path) -> tuple[float, float, float, float]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    gt = ds.GetGeoTransform()
    xs = [gt[0], gt[0] + ds.RasterXSize * gt[1]]
    ys = [gt[3], gt[3] + ds.RasterYSize * gt[5]]
    ds = None
    return min(xs), min(ys), max(xs), max(ys)


def load_area_buildings(shp_path: Path, bounds: tuple[float, float, float, float]) -> list[dict]:
    ds = ogr.Open(str(shp_path))
    if ds is None:
        raise FileNotFoundError(shp_path)
    lyr = ds.GetLayer(0)
    defn = lyr.GetLayerDefn()
    fields = {defn.GetFieldDefn(i).GetNameRef() for i in range(defn.GetFieldCount())}
    min_lon, min_lat, max_lon, max_lat = bounds
    lyr.SetSpatialFilterRect(min_lon, min_lat, max_lon, max_lat)
    buildings: list[dict] = []
    for feat in lyr:
        geom = feat.GetGeometryRef()
        ring = first_exterior_ring(geom)
        if ring is None:
            continue
        floor = float(feat.GetField("Floor") or 0.0) if "Floor" in fields else 0.0
        height = float(feat.GetField("height") or 0.0) if "height" in fields else 0.0
        if height <= 0 and floor > 0:
            height = floor * 3.0
        if height <= 0:
            continue
        buildings.append(
            {
                "fid": int(feat.GetFID()),
                "label": f"F{int(feat.GetFID())}",
                "floor": int(floor),
                "height_m": float(height),
                "area_deg2": float(geom.GetArea()),
                "ring_lonlat": ring[:, :2],
            }
        )
    buildings.sort(key=lambda b: (b["height_m"], b["area_deg2"]), reverse=True)
    return buildings


def apply_dsm_heights(buildings: list[dict], dsm: RasterHeightSampler) -> list[dict]:
    out = []
    for b in buildings:
        top_h = dsm.building_surface_height(b["ring_lonlat"])
        item = dict(b)
        item["top_height_m"] = top_h
        item["base_height_m"] = max(0.0, top_h - float(b["height_m"]))
        out.append(item)
    return out


def solve_pixel_llh_at_height(row: float, col: float, par: dict, orbit, init: tuple[float, float], height_m: float) -> tuple[float, float, bool, float]:
    pos_spline, vel_spline = orbit
    center_lon = float(par["center_longitude"])
    center_lat = float(par["center_latitude"])
    az_time = float(par["start_time"]) + float(row) * float(par["azimuth_line_time"])
    slant_range = float(par["near_range_slc"]) + float(col) * float(par["range_pixel_spacing"])
    wavelength = SPEED_OF_LIGHT / float(par["radar_frequency"])
    sat_pos = np.asarray(pos_spline(az_time), dtype=np.float64)
    sat_vel = np.asarray(vel_spline(az_time), dtype=np.float64)
    fd_target = doppler_hz(par, slant_range)

    def residual(xy: np.ndarray) -> np.ndarray:
        lon, lat = enu_to_llh(center_lon, center_lat, float(xy[0]), float(xy[1]))
        p = llh_to_ecef(lon, lat, height_m)
        los = p - sat_pos
        rng = float(np.linalg.norm(los))
        fd = -2.0 * float(np.dot(los, sat_vel)) / (wavelength * rng)
        return np.array([(rng - slant_range) / 5.0, (fd - fd_target) / 20.0], dtype=np.float64)

    res = least_squares(residual, np.asarray(init, dtype=np.float64), max_nfev=80, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    lon, lat = enu_to_llh(center_lon, center_lat, float(res.x[0]), float(res.x[1]))
    err = float(np.linalg.norm(res.fun))
    return lon, lat, bool(res.success and err < 5.0), err


def gamma_dsm_height_points(points: np.ndarray, par: dict, orbit, dsm: RasterHeightSampler) -> np.ndarray:
    out = []
    last_xy: tuple[float, float] | None = None
    for row, col, method_lon, method_lat, *_rest in points:
        h_dsm = dsm.sample(float(method_lon), float(method_lat))
        xy0 = last_xy if last_xy is not None else initial_xy(float(row), float(col), par)
        lon, lat, ok, residual = solve_pixel_llh_at_height(float(row), float(col), par, orbit, xy0, h_dsm)
        if not ok:
            lon, lat, ok, residual = solve_pixel_llh_at_height(float(row), float(col), par, orbit, initial_xy(float(row), float(col), par), h_dsm)
        center_lon = float(par["center_longitude"])
        center_lat = float(par["center_latitude"])
        last_xy = (
            (lon - center_lon) * math.pi / 180.0 * 6378137.0 * math.cos(math.radians(center_lat)),
            (lat - center_lat) * math.pi / 180.0 * 6378137.0,
        )
        out.append((float(row), float(col), lon, lat, h_dsm, float(ok), residual))
    return np.asarray(out, dtype=np.float64)


def write_building_aligned_gamma_tif(
    out_tif: Path,
    date: str,
    out_dir: Path,
    amp: np.ndarray,
    source_gamma_tif: Path,
    point_rows: list[dict],
    grid_rows: int = 25,
    grid_cols: int = 25,
    max_building_gcps: int = 800,
) -> Path:
    vrt_path = out_dir / f"{date}_building_aligned_gamma_gcps.vrt"
    meta_path = out_dir / f"{date}_building_aligned_gamma_meta.json"

    gcps: list[gdal.GCP] = []
    ds = gdal.Open(str(source_gamma_tif), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(source_gamma_tif)
    gt = ds.GetGeoTransform()
    inv_gt = gdal.InvGeoTransform(gt)
    width = int(ds.RasterXSize)
    height = int(ds.RasterYSize)
    ds = None

    for py in np.linspace(0, height - 1, int(grid_rows)):
        for px in np.linspace(0, width - 1, int(grid_cols)):
            lon, lat = gdal.ApplyGeoTransform(gt, float(px), float(py))
            gcps.append(gdal.GCP(float(lon), float(lat), 0.0, float(px), float(py)))

    selected_building_rows: list[dict] = []
    if point_rows:
        by_pixel: dict[tuple[int, int], dict] = {}
        for row in point_rows:
            rr = int(round(float(row["row"])))
            cc = int(round(float(row["col"])))
            if rr < 0 or cc < 0 or rr >= amp.shape[0] or cc >= amp.shape[1]:
                continue
            key = (rr, cc)
            score = float(amp[rr, cc])
            old = by_pixel.get(key)
            if old is None or score > float(old["_amp_score"]):
                item = dict(row)
                item["_amp_score"] = score
                by_pixel[key] = item
        selected_building_rows = sorted(by_pixel.values(), key=lambda r: float(r["_amp_score"]), reverse=True)[:max_building_gcps]
        for row in selected_building_rows:
            px, py = gdal.ApplyGeoTransform(inv_gt, float(row["gamma_dsm_lon"]), float(row["gamma_dsm_lat"]))
            if not (0 <= px < width and 0 <= py < height):
                continue
            gcps.append(
                gdal.GCP(
                    float(row["method_lon"]),
                    float(row["method_lat"]),
                    float(row["method_height_m"]),
                    float(px),
                    float(py),
                )
            )

    if len(gcps) < 12:
        raise RuntimeError(f"Not enough GCPs for building-aligned GeoTIFF: {len(gcps)}")

    write_gcps_vrt(source_gamma_tif, vrt_path, gcps)
    opts = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        xRes=2.5e-6,
        yRes=2.5e-6,
        resampleAlg="bilinear",
        polynomialOrder=2,
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
        dstNodata=0,
    )
    ds = gdal.Warp(str(out_tif), str(vrt_path), options=opts)
    if ds is None:
        raise RuntimeError(f"Failed to warp {out_tif}")
    ds = None
    meta_path.write_text(
        json.dumps(
            {
                "date": date,
                "output_tif": str(out_tif),
                "source_gamma_tif": str(source_gamma_tif),
                "identity_grid_gcps": int(grid_rows) * int(grid_cols),
                "building_refined_gcps": len(selected_building_rows),
                "building_unique_candidate_pixels": len({(int(round(float(r["row"]))), int(round(float(r["col"])))) for r in point_rows}),
                "total_gcps": len(gcps),
                "note": "Building-aligned WGS84 GAMMA/DSM GeoTIFF. The source GeoTIFF is kept as the main SAR backdrop; identity grid GCPs preserve the full image, and refined building-mask GCPs drive a second-order polynomial correction toward true vector building coordinates.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out_tif


def boundary_distances(points_lonlat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    lon0 = float(np.mean(ring[:, 0]))
    lat0 = float(np.mean(ring[:, 1]))
    pe, pn = local_en(points_lonlat[:, 0], points_lonlat[:, 1], lon0, lat0)
    re, rn = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
    return point_to_polygon_boundary_distance(np.column_stack([pe, pn]), np.column_stack([re, rn]))


def write_geojson(path: Path, buildings: list[dict]) -> None:
    features = []
    for b in buildings:
        ring = b["ring_lonlat"].tolist()
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "fid": b["fid"],
                    "floor": b["floor"],
                    "height_m": b["height_m"],
                    "base_height_m": b.get("base_height_m"),
                    "top_height_m": b.get("top_height_m"),
                    "area_deg2": b["area_deg2"],
                },
                "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
            }
        )
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False, indent=2), encoding="utf-8")


def plot_fig54_like(
    out_png: Path,
    gamma_tif: Path,
    valid_buildings: list[dict],
    method_points: np.ndarray,
    gamma_points: np.ndarray,
    sar_intensity: np.ndarray | None = None,
) -> None:
    bg, extent = read_geotiff(gamma_tif)
    all_rings = np.vstack([b["ring_lonlat"] for b in valid_buildings])
    xpad = max(float(np.ptp(all_rings[:, 0])) * 0.04, 0.00015)
    ypad = max(float(np.ptp(all_rings[:, 1])) * 0.04, 0.00015)
    xlim = (float(np.min(all_rings[:, 0]) - xpad), float(np.max(all_rings[:, 0]) + xpad))
    ylim = (float(np.min(all_rings[:, 1]) - ypad), float(np.max(all_rings[:, 1]) + ypad))

    fig, ax = plt.subplots(figsize=(8.2, 8.2), dpi=300)
    ax.imshow(bg, cmap="gray", extent=extent, origin="upper", interpolation="nearest", alpha=0.72, zorder=0)
    if sar_intensity is not None and sar_intensity.size == method_points.shape[0]:
        ax.set_facecolor("#050505")
        order = np.argsort(sar_intensity)
        step_sar = max(1, order.size // 90000)
        keep = order[::step_sar]
        ax.scatter(
            method_points[keep, 0],
            method_points[keep, 1],
            s=1.0,
            c=sar_intensity[keep],
            cmap="gray",
            vmin=0,
            vmax=1,
            alpha=0.18,
            linewidths=0,
            zorder=1,
        )
    for b in valid_buildings:
        ring = b["ring_lonlat"]
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="white", linewidth=0.55, alpha=0.70, zorder=3))
        ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="#111111", linewidth=0.25, alpha=0.85, zorder=4))
    ax.scatter(gamma_points[:, 0], gamma_points[:, 1], s=1.0, c="#ff9f1c", alpha=0.28, linewidths=0, zorder=5)
    sc = ax.scatter(method_points[:, 0], method_points[:, 1], c=method_points[:, 2], s=1.2, cmap="viridis", alpha=0.72, linewidths=0, zorder=6)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude / deg")
    ax.set_ylabel("Latitude / deg")
    ax.set_title("Huajiachi Area: Building-Constrained Geocoding vs. GAMMA Geocoding")
    ax.ticklabel_format(useOffset=False, style="plain")
    ax.grid(True, color="white", linewidth=0.22, alpha=0.28)
    handles = [
        Line2D([0], [0], color="#777777", linewidth=2.0, alpha=0.75, label="Building-aligned GAMMA/DSM GeoTIFF"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff9f1c", markersize=4, alpha=0.65, label="GAMMA DSM-height geocoding"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#1f9e89", markersize=4, alpha=0.85, label="Building-constrained method"),
        Line2D([0], [0], color="#111111", linewidth=0.7, label="Building footprint"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=8, frameon=True)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Method height / m")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_error_bars(out_png: Path, stats: list[dict]) -> None:
    method = np.asarray([s["method_mean_boundary_distance_m"] for s in stats], dtype=float)
    gamma = np.asarray([s["gamma_dsm_mean_boundary_distance_m"] for s in stats], dtype=float)
    order = np.argsort(gamma)
    x = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(10.5, 4.4), dpi=300)
    ax.scatter(x, gamma[order], s=6, c="#9a6b2f", alpha=0.62, label="GAMMA DSM-height geocoding")
    ax.scatter(x, np.maximum(method[order], 1e-6), s=6, c="#1f77b4", alpha=0.72, label="Building-constrained method")
    ax.set_yscale("log")
    ax.set_xlabel("Buildings sorted by GAMMA mean boundary distance")
    ax.set_ylabel("Mean boundary distance / m")
    ax.grid(True, axis="y", which="both", color="#dddddd", linewidth=0.35)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    par = parse_gamma_par(DEFAULT_RSLC_DIR / f"{args.date}.rslc.par")
    orbit = make_orbit(par)
    dsm = RasterHeightSampler(Path(args.dsm))
    amp = read_rslc_amplitude(DEFAULT_RSLC_DIR / f"{args.date}.rslc", int(par["azimuth_lines"]), int(par["range_samples"]))
    bounds = geotiff_bounds(Path(args.gamma_tif))
    buildings = apply_dsm_heights(load_area_buildings(Path(args.buildings_shp), bounds), dsm)
    if args.max_buildings > 0:
        buildings = buildings[: args.max_buildings]

    valid_buildings: list[dict] = []
    skipped: list[dict] = []
    stats: list[dict] = []
    point_rows: list[dict] = []
    plot_method: list[tuple[float, float, float]] = []
    plot_gamma: list[tuple[float, float]] = []
    plot_sar_intensity: list[float] = []

    for idx, building in enumerate(buildings, start=1):
        try:
            model = rasterize_building(building, par, orbit, amp.shape)
            mask0_count = int(np.sum(model["mask0"]))
            if mask0_count < args.min_mask0_pixels:
                skipped.append({"fid": building["fid"], "reason": "mask0_too_small", "mask0_pixels": mask0_count})
                continue
            mask = refine_mask(model["mask0"], amp)
            model["mask"] = mask
            mask_count = int(np.sum(mask))
            if mask_count < args.min_mask_pixels:
                skipped.append({"fid": building["fid"], "reason": "mask_too_small", "mask0_pixels": mask0_count, "mask_pixels": mask_count})
                continue
            points = scatter_points_from_mask(model, mask, max_points=args.max_points_per_building)
            if points.size == 0:
                skipped.append({"fid": building["fid"], "reason": "no_scatter_points", "mask0_pixels": mask0_count, "mask_pixels": mask_count})
                continue
            gamma = gamma_dsm_height_points(points, par, orbit, dsm)
            method_d = boundary_distances(points[:, 2:4], building["ring_lonlat"])
            gamma_d = boundary_distances(gamma[:, 2:4], building["ring_lonlat"])
            valid_buildings.append(building)
            stats.append(
                {
                    "fid": building["fid"],
                    "floor": building["floor"],
                    "height_m": building["height_m"],
                    "base_height_m": building["base_height_m"],
                    "top_height_m": building["top_height_m"],
                    "mask0_pixels": mask0_count,
                    "mask_pixels": mask_count,
                    "sample_points": int(points.shape[0]),
                    "method_mean_boundary_distance_m": float(np.mean(method_d)),
                    "method_median_boundary_distance_m": float(np.median(method_d)),
                    "method_p90_boundary_distance_m": float(np.percentile(method_d, 90)),
                    "gamma_dsm_mean_boundary_distance_m": float(np.mean(gamma_d)),
                    "gamma_dsm_median_boundary_distance_m": float(np.median(gamma_d)),
                    "gamma_dsm_p90_boundary_distance_m": float(np.percentile(gamma_d, 90)),
                }
            )
            for method_row, gamma_row in zip(points, gamma):
                point_rows.append(
                    {
                        "fid": building["fid"],
                        "row": method_row[0],
                        "col": method_row[1],
                        "method_lon": method_row[2],
                        "method_lat": method_row[3],
                        "method_height_m": method_row[4],
                        "gamma_dsm_lon": gamma_row[2],
                        "gamma_dsm_lat": gamma_row[3],
                        "gamma_dsm_height_m": gamma_row[4],
                        "gamma_dsm_ok": int(gamma_row[5]),
                        "gamma_dsm_residual": gamma_row[6],
                    }
                )
                plot_method.append((float(method_row[2]), float(method_row[3]), float(method_row[4])))
                plot_gamma.append((float(gamma_row[2]), float(gamma_row[3])))
                rr = int(round(float(method_row[0])))
                cc = int(round(float(method_row[1])))
                if 0 <= rr < amp.shape[0] and 0 <= cc < amp.shape[1]:
                    plot_sar_intensity.append(float(amp[rr, cc]) / 255.0)
                else:
                    plot_sar_intensity.append(0.0)
            if idx % 100 == 0:
                print(f"processed {idx}/{len(buildings)} buildings, valid={len(valid_buildings)}, skipped={len(skipped)}")
        except Exception as exc:  # keep processing the rest of the area
            skipped.append({"fid": building.get("fid", -1), "reason": type(exc).__name__, "message": str(exc)})

    if not valid_buildings:
        raise RuntimeError("No valid buildings were geocoded")

    stats_path = out_dir / f"{args.date}_all_buildings_fig5_4_like_stats.csv"
    points_path = out_dir / f"{args.date}_all_buildings_method_vs_gamma_points.csv"
    skipped_path = out_dir / f"{args.date}_all_buildings_skipped.csv"
    fields_stats = list(stats[0].keys())
    with stats_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields_stats)
        writer.writeheader()
        writer.writerows(stats)
    with points_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(point_rows[0].keys()))
        writer.writeheader()
        writer.writerows(point_rows)
    with skipped_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in skipped for k in row}) if skipped else ["fid", "reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped)

    write_geojson(out_dir / f"{args.date}_all_valid_geocoded_buildings.geojson", valid_buildings)
    if os.environ.get("SKIP_HEAVY_FULL_AREA_FIGURES") != "1":
        building_aligned_tif = write_building_aligned_gamma_tif(
            out_dir / f"{args.date}_building_aligned_gamma_dsm_geocoded_wgs84.tif",
            str(args.date),
            out_dir,
            amp,
            Path(args.gamma_tif),
            point_rows,
        )
        plot_fig54_like(
            out_dir / f"{args.date}_fig5_4_like_all_buildings_map.png",
            building_aligned_tif,
            valid_buildings,
            np.asarray(plot_method, dtype=float),
            np.asarray(plot_gamma, dtype=float),
            np.asarray(plot_sar_intensity, dtype=float),
        )
        plot_error_bars(out_dir / f"{args.date}_fig5_4_like_all_buildings_error_scatter.png", stats)
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Huajiachi All-Building Geocoding vs GAMMA",
                "",
                f"- Valid geocoded buildings: {len(valid_buildings)}",
                f"- Skipped buildings: {len(skipped)}",
                f"- Main Figure 5.4-like map: `{args.date}_fig5_4_like_all_buildings_map.png`",
                f"- Error scatter: `{args.date}_fig5_4_like_all_buildings_error_scatter.png`",
                f"- Point coordinate pairs: `{points_path.name}`",
                f"- Per-building statistics: `{stats_path.name}`",
                f"- Skipped list: `{skipped_path.name}`",
                "",
                "Each valid building is processed with the article method: footprint-height model projection, SAR-amplitude mask refinement, and model-surface coordinate inversion.",
                f"DSM: `{args.dsm}`",
                "Building top heights are sampled from the DSM; base heights are `DSM top - vector building height`.",
                "If vector height exceeds the sampled DSM top height, base height is clamped to 0 m to avoid invalid below-ellipsoid building models.",
                "The GAMMA comparison solves the same sampled SAR pixels on the local DSM-height surface from the GAMMA `.rslc.par` state vectors.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"valid_buildings={len(valid_buildings)} skipped={len(skipped)} out_dir={out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--buildings_shp", default=str(DEFAULT_BUILDINGS_SHP))
    parser.add_argument("--gamma_tif", default=str(DEFAULT_GAMMA_TIF))
    parser.add_argument("--dsm", default=str(DEFAULT_DSM))
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--max_buildings", type=int, default=0, help="0 means all buildings in the scene bounds")
    parser.add_argument("--max_points_per_building", type=int, default=80)
    parser.add_argument("--min_mask0_pixels", type=int, default=4)
    parser.add_argument("--min_mask_pixels", type=int, default=2)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
