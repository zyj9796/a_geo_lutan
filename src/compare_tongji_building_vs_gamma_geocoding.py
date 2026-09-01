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
from osgeo import gdal

from geocode_gamma_rslc_with_buildings import initial_xy, make_orbit, parse_gamma_par, solve_pixel_llh
from reproduce_thesis_tongji_tsx import (
    DEFAULT_RSLC_DIR,
    local_en,
    point_to_polygon_boundary_distance,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPRO_ROOT = ROOT / "thesis_reproduction_tongji_tsx"
DEFAULT_GAMMA_TIF = ROOT / "tsx_tongji_geocode" / "20200708_amplitude_geocoded_wgs84.tif"
DEFAULT_OUT_DIR = ROOT / "tongji_building_vs_gamma_geocoding"


def read_buildings(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    buildings = []
    for i, feat in enumerate(data.get("features", []), start=1):
        props = feat.get("properties", {})
        ring = np.asarray(feat["geometry"]["coordinates"][0], dtype=np.float64)
        if ring.shape[0] > 1 and np.allclose(ring[0], ring[-1]):
            ring = ring[:-1]
        buildings.append(
            {
                "id": int(props.get("building", i)),
                "label": f"B{i}",
                "height_m": float(props.get("height_m", 0.0)),
                "ring_lonlat": ring[:, :2],
            }
        )
    return buildings


def read_method_points(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    float(row["row"]),
                    float(row["col"]),
                    float(row["lon"]),
                    float(row["lat"]),
                    float(row["height_m"]),
                    float(row["triangle_index"]),
                )
            )
    return np.asarray(rows, dtype=np.float64)


def gamma_zero_height_points(points: np.ndarray, par: dict, orbit) -> np.ndarray:
    out = []
    last_xy: tuple[float, float] | None = None
    for row, col, *_rest in points:
        xy0 = last_xy if last_xy is not None else initial_xy(float(row), float(col), par)
        lon, lat, ok, residual = solve_pixel_llh(float(row), float(col), par, orbit, xy0)
        if not ok:
            lon, lat, ok, residual = solve_pixel_llh(float(row), float(col), par, orbit, initial_xy(float(row), float(col), par))
        center_lon = float(par["center_longitude"])
        center_lat = float(par["center_latitude"])
        last_xy = (
            (lon - center_lon) * math.pi / 180.0 * 6378137.0 * math.cos(math.radians(center_lat)),
            (lat - center_lat) * math.pi / 180.0 * 6378137.0,
        )
        out.append((float(row), float(col), lon, lat, 0.0, float(ok), residual))
    return np.asarray(out, dtype=np.float64)


def boundary_distances(points_lonlat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    lon0 = float(np.mean(ring[:, 0]))
    lat0 = float(np.mean(ring[:, 1]))
    pe, pn = local_en(points_lonlat[:, 0], points_lonlat[:, 1], lon0, lat0)
    re, rn = local_en(ring[:, 0], ring[:, 1], lon0, lat0)
    return point_to_polygon_boundary_distance(np.column_stack([pe, pn]), np.column_stack([re, rn]))


def write_points_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "building",
        "row",
        "col",
        "method_lon",
        "method_lat",
        "method_height_m",
        "gamma_lon",
        "gamma_lat",
        "gamma_height_m",
        "gamma_ok",
        "gamma_residual",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_stats_csv(path: Path, stats: list[dict]) -> None:
    fields = [
        "building",
        "method_points",
        "gamma_points",
        "method_mean_boundary_distance_m",
        "method_median_boundary_distance_m",
        "method_p90_boundary_distance_m",
        "gamma_mean_boundary_distance_m",
        "gamma_median_boundary_distance_m",
        "gamma_p90_boundary_distance_m",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(stats)


def read_geotiff(path: Path) -> tuple[np.ndarray, list[float]]:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    max_pixels = int(os.environ.get("MAX_PLOT_GEOTIFF_PIXELS", "4000000"))
    width = int(ds.RasterXSize)
    height = int(ds.RasterYSize)
    if width * height > max_pixels:
        scale = math.sqrt((width * height) / max_pixels)
        buf_xsize = max(1, int(width / scale))
        buf_ysize = max(1, int(height / scale))
        arr = ds.ReadAsArray(buf_xsize=buf_xsize, buf_ysize=buf_ysize)
    else:
        arr = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    extent = [
        gt[0],
        gt[0] + ds.RasterXSize * gt[1],
        gt[3] + ds.RasterYSize * gt[5],
        gt[3],
    ]
    ds = None
    return arr, extent


def plot_comparison(
    out_png: Path,
    gamma_tif: Path,
    buildings: list[dict],
    method_by_building: dict[str, np.ndarray],
    gamma_by_building: dict[str, np.ndarray],
) -> None:
    bg, extent = read_geotiff(gamma_tif)
    rings = [b["ring_lonlat"] for b in buildings]
    all_ring = np.vstack(rings)
    lon_pad = max(float(np.ptp(all_ring[:, 0])) * 0.35, 0.00035)
    lat_pad = max(float(np.ptp(all_ring[:, 1])) * 0.35, 0.00035)
    xlim = (float(np.min(all_ring[:, 0]) - lon_pad), float(np.max(all_ring[:, 0]) + lon_pad))
    ylim = (float(np.min(all_ring[:, 1]) - lat_pad), float(np.max(all_ring[:, 1]) + lat_pad))

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.8), dpi=300, sharex=True, sharey=True)
    titles = ["GAMMA zero-height geocoding", "Footprint-constrained building geocoding"]
    for ax, title in zip(axes, titles):
        ax.imshow(bg, cmap="gray", extent=extent, origin="upper")
        for b in buildings:
            ring = b["ring_lonlat"]
            ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="#f8f8f8", linewidth=1.8))
            ax.add_patch(MplPolygon(ring, closed=True, fill=False, edgecolor="#111111", linewidth=0.75))
            cx, cy = np.mean(ring, axis=0)
            ax.text(cx, cy, b["label"], ha="center", va="center", fontsize=8, color="black", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 1.2})
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Longitude / deg")
        ax.grid(True, color="white", linewidth=0.25, alpha=0.35)
        ax.ticklabel_format(useOffset=False, style="plain")
    axes[0].set_ylabel("Latitude / deg")

    for b in buildings:
        label = b["label"]
        gamma = gamma_by_building[label]
        method = method_by_building[label]
        axes[0].scatter(gamma[:, 2], gamma[:, 3], s=4.0, c="#ff9f1c", alpha=0.55, linewidths=0, zorder=5)
        sc = axes[1].scatter(
            method[:, 2],
            method[:, 3],
            c=method[:, 4],
            s=6.0,
            cmap="viridis",
            alpha=0.95,
            linewidths=0,
            vmin=0,
            vmax=max(105.0, float(np.nanmax(method[:, 4]))),
            zorder=5,
        )

    axes[0].legend(
        handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#ff9f1c", markersize=5, label="GAMMA geocoded scattering pixels")],
        loc="upper right",
        fontsize=8,
        frameon=True,
    )
    cbar = fig.colorbar(sc, ax=axes[1], shrink=0.84, pad=0.018)
    cbar.set_label("Building-surface height / m")
    fig.suptitle("Tongji Area Building Geocoding: Article Method vs. GAMMA Geocoding", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def run(date: str, out_dir: Path, gamma_tif: Path, repro_root: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    buildings = read_buildings(repro_root / "selected_buildings_master.geojson")
    par = parse_gamma_par(DEFAULT_RSLC_DIR / f"{date}.rslc.par")
    orbit = make_orbit(par)

    method_by_building: dict[str, np.ndarray] = {}
    gamma_by_building: dict[str, np.ndarray] = {}
    all_rows: list[dict] = []
    stats = []

    for b in buildings:
        label = b["label"]
        points = read_method_points(repro_root / date / f"building_{b['id']}" / "scatter_points_wgs84.csv")
        gamma_points = gamma_zero_height_points(points, par, orbit)
        method_by_building[label] = points
        gamma_by_building[label] = gamma_points

        method_d = boundary_distances(points[:, 2:4], b["ring_lonlat"])
        gamma_d = boundary_distances(gamma_points[:, 2:4], b["ring_lonlat"])
        stats.append(
            {
                "building": label,
                "method_points": int(points.shape[0]),
                "gamma_points": int(gamma_points.shape[0]),
                "method_mean_boundary_distance_m": float(np.mean(method_d)),
                "method_median_boundary_distance_m": float(np.median(method_d)),
                "method_p90_boundary_distance_m": float(np.percentile(method_d, 90)),
                "gamma_mean_boundary_distance_m": float(np.mean(gamma_d)),
                "gamma_median_boundary_distance_m": float(np.median(gamma_d)),
                "gamma_p90_boundary_distance_m": float(np.percentile(gamma_d, 90)),
            }
        )
        for method_row, gamma_row in zip(points, gamma_points):
            all_rows.append(
                {
                    "building": label,
                    "row": method_row[0],
                    "col": method_row[1],
                    "method_lon": method_row[2],
                    "method_lat": method_row[3],
                    "method_height_m": method_row[4],
                    "gamma_lon": gamma_row[2],
                    "gamma_lat": gamma_row[3],
                    "gamma_height_m": gamma_row[4],
                    "gamma_ok": int(gamma_row[5]),
                    "gamma_residual": gamma_row[6],
                }
            )

    write_points_csv(out_dir / f"{date}_tongji_method_vs_gamma_points.csv", all_rows)
    write_stats_csv(out_dir / f"{date}_tongji_method_vs_gamma_stats.csv", stats)
    plot_comparison(
        out_dir / f"{date}_tongji_method_vs_gamma_comparison.png",
        gamma_tif,
        buildings,
        method_by_building,
        gamma_by_building,
    )
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Tongji Building Geocoding vs GAMMA Geocoding",
                "",
                f"- Comparison figure: `{date}_tongji_method_vs_gamma_comparison.png`",
                f"- Point pairs: `{date}_tongji_method_vs_gamma_points.csv`",
                f"- Boundary-distance statistics: `{date}_tongji_method_vs_gamma_stats.csv`",
                "",
                "The GAMMA side solves the same SAR image pixels on a zero-height WGS84 surface using the GAMMA `.rslc.par` orbit and imaging parameters.",
                "The article-method side uses the building footprint and height model to map those pixels onto building surfaces.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20200708")
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--gamma_tif", default=str(DEFAULT_GAMMA_TIF))
    parser.add_argument("--repro_root", default=str(DEFAULT_REPRO_ROOT))
    args = parser.parse_args()
    run(args.date, Path(args.out_dir), Path(args.gamma_tif), Path(args.repro_root))


if __name__ == "__main__":
    main()
