from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.path import Path as MplPath
from osgeo import gdal, osr


class RasterHeightSampler:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.ds = gdal.Open(str(self.path), gdal.GA_ReadOnly)
        if self.ds is None:
            raise FileNotFoundError(path)
        self.band = self.ds.GetRasterBand(1)
        self.gt = self.ds.GetGeoTransform()
        self.inv_gt = gdal.InvGeoTransform(self.gt)
        self.nodata = self.band.GetNoDataValue()
        src = osr.SpatialReference()
        src.ImportFromEPSG(4326)
        src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst = osr.SpatialReference(wkt=self.ds.GetProjection())
        dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        self.to_raster_srs = osr.CoordinateTransformation(src, dst)

    def _pixel_float(self, lon: float, lat: float) -> tuple[float, float]:
        x, y, _ = self.to_raster_srs.TransformPoint(float(lon), float(lat))
        px, py = gdal.ApplyGeoTransform(self.inv_gt, x, y)
        return float(px), float(py)

    def sample(self, lon: float, lat: float) -> float:
        px, py = self._pixel_float(lon, lat)
        x0 = int(np.floor(px))
        y0 = int(np.floor(py))
        x1 = x0 + 1
        y1 = y0 + 1
        if x0 < 0 or y0 < 0 or x1 >= self.ds.RasterXSize or y1 >= self.ds.RasterYSize:
            raise ValueError(f"Point ({lon:.8f}, {lat:.8f}) outside raster {self.path}")
        arr = self.band.ReadAsArray(x0, y0, 2, 2).astype(np.float64)
        if self.nodata is not None:
            arr[arr == float(self.nodata)] = np.nan
        dx = px - x0
        dy = py - y0
        weights = np.array([[((1 - dx) * (1 - dy)), (dx * (1 - dy))], [((1 - dx) * dy), (dx * dy)]], dtype=np.float64)
        ok = np.isfinite(arr)
        if not np.any(ok):
            raise ValueError(f"No valid raster height near ({lon:.8f}, {lat:.8f}) in {self.path}")
        return float(np.sum(arr[ok] * weights[ok]) / np.sum(weights[ok]))

    def sample_many(self, lonlat: np.ndarray) -> np.ndarray:
        return np.asarray([self.sample(float(lon), float(lat)) for lon, lat in lonlat], dtype=np.float64)

    def building_surface_height(self, ring_lonlat: np.ndarray, grid_size: int = 16) -> float:
        ring = np.asarray(ring_lonlat, dtype=np.float64)
        centroid = np.asarray([[float(np.mean(ring[:, 0])), float(np.mean(ring[:, 1]))]], dtype=np.float64)
        min_lon, min_lat = np.min(ring, axis=0)
        max_lon, max_lat = np.max(ring, axis=0)
        lon_grid = np.linspace(min_lon, max_lon, grid_size)
        lat_grid = np.linspace(min_lat, max_lat, grid_size)
        xx, yy = np.meshgrid(lon_grid, lat_grid)
        grid = np.column_stack([xx.ravel(), yy.ravel()])
        inside = MplPath(ring).contains_points(grid)
        samples_xy = np.vstack([centroid, ring, grid[inside]])
        samples = self.sample_many(samples_xy)
        return float(np.nanpercentile(samples, 95))
