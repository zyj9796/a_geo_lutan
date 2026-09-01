from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as MplPolygon
from osgeo import gdal, ogr, osr
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares


SPEED_OF_LIGHT = 299792458.0
WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
REPO_ROOT = Path(__file__).resolve().parents[1]
WINDOWS_RSLC_DIR = r"E:\all_data\TSX\Tongji\RE_SLAVES"
WINDOWS_BUILDINGS_VECTOR = (
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


def default_rslc_dir() -> Path:
    return first_existing_path(
        [
            os.environ.get("TSX_RSLC_DIR"),
            REPO_ROOT.parent / "tsx_tongji" / "RE_SLAVES",
            REPO_ROOT / "tsx_tongji" / "RE_SLAVES",
            WINDOWS_RSLC_DIR,
        ]
    )


def default_buildings_vector() -> Path:
    return first_existing_path(
        [
            os.environ.get("TSX_BUILDINGS_VECTOR"),
            os.environ.get("TSX_BUILDINGS_SHP"),
            Path("/home/u/tongji_ps/tongji2/shp/tongji_clip.shp"),
            Path("/home/u/tongji_ps/tongji3/shp/tongji_clip.shp"),
            REPO_ROOT / "tsx_tongji_geocode" / "shanghai_buildings_subset.geojson",
            WINDOWS_BUILDINGS_VECTOR,
        ]
    )


def parse_gamma_par(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    out: dict[str, object] = {"state_vectors": []}
    scalar_keys = {
        "title": str,
        "sensor": str,
        "start_time": float,
        "azimuth_line_time": float,
        "range_samples": int,
        "azimuth_lines": int,
        "center_latitude": float,
        "center_longitude": float,
        "heading": float,
        "range_pixel_spacing": float,
        "azimuth_pixel_spacing": float,
        "near_range_slc": float,
        "center_range_slc": float,
        "radar_frequency": float,
        "incidence_angle": float,
        "time_of_first_state_vector": float,
        "state_vector_interval": float,
    }
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, rest = line.split(":", 1)
        key = key.strip()
        rest = rest.strip()
        if key in scalar_keys:
            if scalar_keys[key] is str:
                out[key] = rest
            else:
                token = rest.split()[0]
                out[key] = scalar_keys[key](float(token))
        elif key == "doppler_polynomial":
            out[key] = [float(x) for x in rest.split()[:4]]
        else:
            m_pos = re.match(r"state_vector_position_(\d+)", key)
            m_vel = re.match(r"state_vector_velocity_(\d+)", key)
            if m_pos or m_vel:
                idx = int((m_pos or m_vel).group(1)) - 1
                values = [float(x) for x in rest.split()[:3]]
                sv = out["state_vectors"]
                while len(sv) <= idx:
                    sv.append({})
                sv[idx]["pos" if m_pos else "vel"] = values
    required = [
        "start_time",
        "azimuth_line_time",
        "range_samples",
        "azimuth_lines",
        "center_latitude",
        "center_longitude",
        "heading",
        "range_pixel_spacing",
        "azimuth_pixel_spacing",
        "near_range_slc",
        "center_range_slc",
        "radar_frequency",
        "time_of_first_state_vector",
        "state_vector_interval",
    ]
    missing = [k for k in required if k not in out]
    if missing:
        raise ValueError(f"Missing required GAMMA par keys in {path}: {missing}")
    return out


def llh_to_ecef(lon_deg: float, lat_deg: float, h_m: float = 0.0) -> np.ndarray:
    lon = math.radians(lon_deg)
    lat = math.radians(lat_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    n = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (n + h_m) * cos_lat * math.cos(lon),
            (n + h_m) * cos_lat * math.sin(lon),
            (n * (1.0 - WGS84_E2) + h_m) * sin_lat,
        ],
        dtype=np.float64,
    )


def enu_to_llh(center_lon: float, center_lat: float, east_m: float, north_m: float) -> tuple[float, float]:
    lat_rad = math.radians(center_lat)
    lon = center_lon + east_m / (WGS84_A * math.cos(lat_rad)) * 180.0 / math.pi
    lat = center_lat + north_m / WGS84_A * 180.0 / math.pi
    return lon, lat


def make_orbit(par: dict):
    t0 = float(par["time_of_first_state_vector"])
    dt = float(par["state_vector_interval"])
    sv = par["state_vectors"]
    times = np.asarray([t0 + i * dt for i in range(len(sv))], dtype=np.float64)
    pos = np.asarray([x["pos"] for x in sv], dtype=np.float64)
    vel = np.asarray([x["vel"] for x in sv], dtype=np.float64)
    pos_spline = CubicSpline(times, pos, axis=0)
    vel_spline = CubicSpline(times, vel, axis=0)
    return pos_spline, vel_spline


def doppler_hz(par: dict, slant_range: float) -> float:
    coeffs = par.get("doppler_polynomial") or [0.0, 0.0, 0.0, 0.0]
    x = float(slant_range) - float(par["center_range_slc"])
    return float(coeffs[0] + coeffs[1] * x + coeffs[2] * x * x + coeffs[3] * x * x * x)


def solve_pixel_llh(row: float, col: float, par: dict, orbit, init_xy: tuple[float, float]) -> tuple[float, float, bool, float]:
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
        p = llh_to_ecef(lon, lat, 0.0)
        los = p - sat_pos
        rng = float(np.linalg.norm(los))
        fd = -2.0 * float(np.dot(los, sat_vel)) / (wavelength * rng)
        return np.array([(rng - slant_range) / 5.0, (fd - fd_target) / 20.0], dtype=np.float64)

    res = least_squares(residual, np.asarray(init_xy, dtype=np.float64), max_nfev=80, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    lon, lat = enu_to_llh(center_lon, center_lat, float(res.x[0]), float(res.x[1]))
    err = float(np.linalg.norm(res.fun))
    return lon, lat, bool(res.success and err < 5.0), err


def initial_xy(row: float, col: float, par: dict) -> tuple[float, float]:
    rows = float(par["azimuth_lines"])
    cols = float(par["range_samples"])
    heading = math.radians(float(par["heading"]))
    inc = math.radians(float(par.get("incidence_angle", 42.0)))
    along = np.array([math.sin(heading), math.cos(heading)], dtype=np.float64)
    right = np.array([math.sin(heading + math.pi / 2.0), math.cos(heading + math.pi / 2.0)], dtype=np.float64)
    d_az = (float(row) - (rows - 1.0) / 2.0) * float(par["azimuth_pixel_spacing"])
    d_gr = (float(col) - (cols - 1.0) / 2.0) * float(par["range_pixel_spacing"]) / max(math.sin(inc), 0.2)
    east, north = along * d_az + right * d_gr
    return float(east), float(north)


def read_rslc_amplitude(path: Path, rows: int, cols: int) -> np.ndarray:
    sample_count = rows * cols * 2
    byte_count = path.stat().st_size
    if byte_count == sample_count * np.dtype(">i2").itemsize:
        raw = np.fromfile(str(path), dtype=">i2")
    elif byte_count == sample_count * np.dtype(">f4").itemsize:
        # GAMMA FCOMPLEX stores interleaved big-endian float32 real/imaginary samples.
        raw = np.fromfile(str(path), dtype=">f4")
    else:
        raise ValueError(
            f"Unexpected RSLC size: got {byte_count} bytes; expected "
            f"{sample_count * 2} bytes (SCOMPLEX) or {sample_count * 4} bytes (FCOMPLEX)"
        )
    z = raw.reshape(rows, cols, 2).astype(np.float32)
    amp = np.hypot(z[:, :, 0], z[:, :, 1])
    p2, p98 = np.percentile(amp[amp > 0], [2, 98]) if np.any(amp > 0) else (0.0, 1.0)
    scaled = np.clip((amp - p2) / max(p98 - p2, 1e-6), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def write_gcps_vrt(src_tif: Path, vrt_path: Path, gcps: list[gdal.GCP]) -> None:
    ds = gdal.Open(str(src_tif), gdal.GA_ReadOnly)
    vrt = gdal.Translate(str(vrt_path), ds, format="VRT")
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    vrt.SetGCPs(gcps, srs.ExportToWkt())
    vrt = None
    ds = None


def export_buildings_subset(shp_path: Path, out_geojson: Path, bounds: tuple[float, float, float, float], margin_deg: float = 0.002) -> int:
    min_lon, min_lat, max_lon, max_lat = bounds
    min_lon -= margin_deg
    min_lat -= margin_deg
    max_lon += margin_deg
    max_lat += margin_deg
    ds = ogr.Open(str(shp_path))
    if ds is None:
        raise ValueError(f"Failed to open shapefile: {shp_path}")
    lyr = ds.GetLayer(0)
    lyr.SetSpatialFilterRect(min_lon, min_lat, max_lon, max_lat)
    driver = ogr.GetDriverByName("GeoJSON")
    if out_geojson.exists():
        driver.DeleteDataSource(str(out_geojson))
    out_ds = driver.CreateDataSource(str(out_geojson))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    out_lyr = out_ds.CreateLayer("buildings", srs, ogr.wkbPolygon)
    in_defn = lyr.GetLayerDefn()
    for i in range(in_defn.GetFieldCount()):
        out_lyr.CreateField(in_defn.GetFieldDefn(i))
    count = 0
    for feat in lyr:
        out_feat = ogr.Feature(out_lyr.GetLayerDefn())
        for i in range(in_defn.GetFieldCount()):
            out_feat.SetField(in_defn.GetFieldDefn(i).GetNameRef(), feat.GetField(i))
        out_feat.SetGeometry(feat.GetGeometryRef().Clone())
        out_lyr.CreateFeature(out_feat)
        count += 1
    out_ds = None
    ds = None
    return count


def plot_overlay(geotiff: Path, buildings_geojson: Path, out_png: Path) -> None:
    ds = gdal.Open(str(geotiff))
    arr = ds.ReadAsArray()
    gt = ds.GetGeoTransform()
    extent = [
        gt[0],
        gt[0] + ds.RasterXSize * gt[1],
        gt[3] + ds.RasterYSize * gt[5],
        gt[3],
    ]
    data = json.loads(buildings_geojson.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9, 7), dpi=180)
    ax.imshow(arr, cmap="gray", extent=extent, origin="upper")
    for feat in data.get("features", []):
        geom = feat.get("geometry") or {}
        rings = []
        if geom.get("type") == "Polygon":
            rings = [geom.get("coordinates", [[]])[0]]
        elif geom.get("type") == "MultiPolygon":
            rings = [poly[0] for poly in geom.get("coordinates", []) if poly]
        for ring in rings:
            pts = np.asarray(ring, dtype=np.float64)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                ax.add_patch(MplPolygon(pts[:, :2], fill=False, edgecolor="#00ff66", linewidth=0.35, alpha=0.75))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{geotiff.stem} geocoded with Shanghai building footprints")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20200708")
    ap.add_argument("--rslc_dir", default=str(default_rslc_dir()))
    ap.add_argument("--buildings_shp", "--buildings_vector", dest="buildings_shp", default=str(default_buildings_vector()))
    ap.add_argument("--output_dir", default="tsx_tongji_geocode")
    ap.add_argument("--gcp_rows", type=int, default=13)
    ap.add_argument("--gcp_cols", type=int, default=17)
    ap.add_argument("--tr_deg", type=float, default=2.5e-6)
    args = ap.parse_args()

    rslc_dir = Path(args.rslc_dir)
    rslc_path = rslc_dir / f"{args.date}.rslc"
    par_path = rslc_dir / f"{args.date}.rslc.par"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    par = parse_gamma_par(par_path)
    rows = int(par["azimuth_lines"])
    cols = int(par["range_samples"])
    amp = read_rslc_amplitude(rslc_path, rows, cols)

    src_tif = out_dir / f"{args.date}_amplitude_radar.tif"
    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create(str(src_tif), cols, rows, 1, gdal.GDT_Byte, options=["COMPRESS=LZW"])
    ds.GetRasterBand(1).WriteArray(amp)
    ds = None

    orbit = make_orbit(par)
    gcps: list[gdal.GCP] = []
    errors = []
    row_vals = np.linspace(0, rows - 1, int(args.gcp_rows))
    col_vals = np.linspace(0, cols - 1, int(args.gcp_cols))
    for r in row_vals:
        last_xy = None
        for c in col_vals:
            xy0 = last_xy if last_xy is not None else initial_xy(float(r), float(c), par)
            lon, lat, ok, err = solve_pixel_llh(float(r), float(c), par, orbit, xy0)
            if not ok:
                lon, lat, ok, err = solve_pixel_llh(float(r), float(c), par, initial_xy(float(r), float(c), par))
            last_xy = initial_xy(float(r), float(c), par)
            gcps.append(gdal.GCP(lon, lat, 0.0, float(c), float(r)))
            errors.append(err)

    vrt_path = out_dir / f"{args.date}_amplitude_gcps.vrt"
    out_tif = out_dir / f"{args.date}_amplitude_geocoded_wgs84.tif"
    write_gcps_vrt(src_tif, vrt_path, gcps)
    warp_opts = gdal.WarpOptions(
        dstSRS="EPSG:4326",
        xRes=float(args.tr_deg),
        yRes=float(args.tr_deg),
        resampleAlg="bilinear",
        tps=True,
        format="GTiff",
        creationOptions=["COMPRESS=LZW", "TILED=YES"],
        dstNodata=0,
    )
    gdal.Warp(str(out_tif), str(vrt_path), options=warp_opts)

    ds = gdal.Open(str(out_tif))
    gt = ds.GetGeoTransform()
    bounds = (
        min(gt[0], gt[0] + ds.RasterXSize * gt[1]),
        min(gt[3], gt[3] + ds.RasterYSize * gt[5]),
        max(gt[0], gt[0] + ds.RasterXSize * gt[1]),
        max(gt[3], gt[3] + ds.RasterYSize * gt[5]),
    )
    ds = None

    buildings_input = Path(args.buildings_shp)
    buildings_geojson = out_dir / "shanghai_buildings_subset.geojson"
    if buildings_input.exists() and buildings_geojson.exists() and buildings_input.resolve() == buildings_geojson.resolve():
        buildings_geojson = out_dir / f"{args.date}_shanghai_buildings_subset.geojson"
    building_count = export_buildings_subset(buildings_input, buildings_geojson, bounds)
    overlay_png = out_dir / f"{args.date}_geocoded_buildings_overlay.png"
    plot_overlay(out_tif, buildings_geojson, overlay_png)

    meta = {
        "date": args.date,
        "rslc": str(rslc_path),
        "par": str(par_path),
        "buildings_shp": str(args.buildings_shp),
        "outputs": {
            "radar_amplitude_tif": str(src_tif),
            "geocoded_wgs84_tif": str(out_tif),
            "buildings_subset_geojson": str(buildings_geojson),
            "overlay_png": str(overlay_png),
        },
        "image_size": {"rows": rows, "cols": cols},
        "geocoded_bounds_wgs84": {
            "min_lon": bounds[0],
            "min_lat": bounds[1],
            "max_lon": bounds[2],
            "max_lat": bounds[3],
        },
        "building_feature_count": building_count,
        "gcp_count": len(gcps),
        "gcp_residual_norm": {
            "median": float(np.median(errors)),
            "max": float(np.max(errors)),
        },
        "note": "Geocoding uses GAMMA RSLC parameters, WGS84 zero-height range-Doppler control points, and GDAL thin-plate-spline warping.",
    }
    meta_path = out_dir / f"{args.date}_geocode_metadata.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Geocoded GeoTIFF: {out_tif}")
    print(f"Buildings subset: {buildings_geojson} ({building_count} features)")
    print(f"Overlay PNG: {overlay_png}")
    print(f"Metadata: {meta_path}")


if __name__ == "__main__":
    main()
