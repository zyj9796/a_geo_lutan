from __future__ import annotations

import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


PROJECT_DIR = Path(os.environ.get("SAR_GEOCODE_PROJECT_DIR", Path(__file__).resolve().parents[1])).resolve()
SHARED_CODE_DIR = Path(
    os.environ.get("SAR_GEOCODE_SHARED_CODE_DIR", PROJECT_DIR.parent / "a_geo_huajiachi" / "code")
).resolve()
sys.path.insert(0, str(SHARED_CODE_DIR))

from make_full_area_geobc_discrete_3d import (  # noqa: E402
    CLASS_BOUNDS,
    CLASS_CMAP,
    CLASS_NORM,
    add_discrete_colorbar,
)
from make_huajiachi_full_area_geobc_3d import (  # noqa: E402
    load_full_area_buildings,
    load_full_area_points,
    read_velocity_by_pixel,
)
from make_huajiachi_hotspot_3d_zoom import select_hotspot  # noqa: E402


AREA_LABEL = os.environ.get("SAR_GEOCODE_AREA_LABEL", "华家池")
FIGURE_PREFIX = os.environ.get("SAR_GEOCODE_FIGURE_PREFIX", "lutan")
OUT_PNG = PROJECT_DIR / "results" / "pic_all" / f"fig_10_{FIGURE_PREFIX}_planar_geobc_3d_extent.png"
FRAME_COLOR = "#ffd400"
ZOOM_COLOR = "#00e5ff"


def main() -> None:
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    buildings = load_full_area_buildings()
    rows = load_full_area_points(buildings, read_velocity_by_pixel())
    _, zoom_rows, zoom_extent, _ = select_hotspot(buildings, rows)

    rings = [building["ring_lonlat"] for building in buildings.values()]
    all_rings = np.vstack(rings)
    xmin, ymin = np.min(all_rings, axis=0)
    xmax, ymax = np.max(all_rings, axis=0)
    width = float(xmax - xmin)
    height = float(ymax - ymin)
    pad_x = width * 0.045
    pad_y = height * 0.045

    lon = np.asarray([row["lon"] for row in rows], dtype=float)
    lat = np.asarray([row["lat"] for row in rows], dtype=float)
    velocity = np.asarray([row["velocity"] for row in rows], dtype=float)
    counts = np.histogram(np.clip(velocity, CLASS_BOUNDS[0], CLASS_BOUNDS[-1]), bins=CLASS_BOUNDS)[0]

    fig = plt.figure(figsize=(10.0, 8.4), dpi=300)
    ax = fig.add_axes([0.075, 0.19, 0.85, 0.72])
    ax.set_facecolor("#101416")

    segments = []
    for ring in rings:
        closed = np.vstack([ring, ring[0]])
        segments.extend(np.stack([closed[:-1], closed[1:]], axis=1))
    ax.add_collection(
        LineCollection(segments, colors="#aeb4b8", linewidths=0.20, alpha=0.62, zorder=2)
    )
    scatter = ax.scatter(
        lon,
        lat,
        c=velocity,
        cmap=CLASS_CMAP,
        norm=CLASS_NORM,
        s=1.05,
        alpha=0.92,
        linewidths=0,
        zorder=4,
        rasterized=True,
    )

    extent = (float(xmin), float(ymin), float(xmax), float(ymax))
    ax.add_patch(
        Rectangle(
            (extent[0], extent[1]),
            extent[2] - extent[0],
            extent[3] - extent[1],
            fill=False,
            edgecolor="white",
            linewidth=3.4,
            alpha=0.96,
            zorder=8,
        )
    )
    ax.add_patch(
        Rectangle(
            (extent[0], extent[1]),
            extent[2] - extent[0],
            extent[3] - extent[1],
            fill=False,
            edgecolor=FRAME_COLOR,
            linewidth=1.8,
            linestyle=(0, (6, 3)),
            zorder=9,
        )
    )
    ax.annotate(
        f"全区域三维（图9）\n{len(buildings)} 栋｜{len(rows)} 点",
        xy=(extent[2], extent[3]),
        xytext=(-18, -92),
        textcoords="offset points",
        ha="right",
        va="top",
        fontsize=7.5,
        fontweight="bold",
        color="#17191b",
        bbox={"boxstyle": "round,pad=0.35", "fc": FRAME_COLOR, "ec": "white", "lw": 1.0, "alpha": 0.96},
        arrowprops={"arrowstyle": "-|>", "color": FRAME_COLOR, "lw": 1.6},
        zorder=10,
    )
    ax.add_patch(
        Rectangle(
            (zoom_extent[0], zoom_extent[1]),
            zoom_extent[2] - zoom_extent[0],
            zoom_extent[3] - zoom_extent[1],
            fill=False,
            edgecolor="white",
            linewidth=3.0,
            zorder=10,
        )
    )
    ax.add_patch(
        Rectangle(
            (zoom_extent[0], zoom_extent[1]),
            zoom_extent[2] - zoom_extent[0],
            zoom_extent[3] - zoom_extent[1],
            fill=False,
            edgecolor=ZOOM_COLOR,
            linewidth=1.7,
            zorder=11,
        )
    )
    ax.annotate(
        f"选定三维放大区（图11）\n450 米窗口｜{len(zoom_rows)} 个点",
        xy=(zoom_extent[0], zoom_extent[3]),
        xytext=(15, 14),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=7.5,
        fontweight="bold",
        color="#081416",
        bbox={"boxstyle": "round,pad=0.32", "fc": ZOOM_COLOR, "ec": "white", "lw": 1.0, "alpha": 0.96},
        arrowprops={"arrowstyle": "-|>", "color": ZOOM_COLOR, "lw": 1.5},
        zorder=12,
    )

    ax.set_xlim(float(xmin - pad_x), float(xmax + pad_x))
    ax.set_ylim(float(ymin - pad_y), float(ymax + pad_y))
    mean_lat = float((ymin + ymax) / 2.0)
    ax.set_aspect(1.0 / max(math.cos(math.radians(mean_lat)), 0.1), adjustable="box")
    ax.set_xlabel("经度 / 度", fontsize=8)
    ax.set_ylabel("纬度 / 度", fontsize=8)
    ax.tick_params(labelsize=7, colors="#34383b")
    ax.grid(color="white", linewidth=0.22, alpha=0.16)
    ax.ticklabel_format(useOffset=False, style="plain")

    handles = [
        Line2D([0], [0], color="#aeb4b8", linewidth=0.8, label="三维建筑轮廓"),
        Line2D([0], [0], color=FRAME_COLOR, linewidth=1.8, linestyle="--", label="全区域三维范围（图9）"),
        Line2D([0], [0], color=ZOOM_COLOR, linewidth=1.8, label="选定三维放大区（图11）"),
    ]
    legend = ax.legend(handles=handles, loc="upper left", fontsize=7, frameon=True, framealpha=0.96)
    legend.set_zorder(11)
    fig.text(
        0.025,
        0.965,
        f"{AREA_LABEL}平面 Geo-BC 形变图与三维显示范围",
        fontsize=12,
        fontweight="bold",
        va="top",
        color="#202124",
    )
    fig.text(
        0.025,
        0.928,
        "黄色：全区域三维视图｜青色：选定的 450 米形变热点放大区",
        fontsize=7.5,
        va="top",
        color="#5a6167",
    )
    add_discrete_colorbar(fig, scatter, counts.astype(int).tolist())
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)

    print(f"extent_lonlat={extent}")
    print(f"buildings={len(buildings)}")
    print(f"points={len(rows)}")
    print(f"zoom_extent_lonlat={zoom_extent}")
    print(f"zoom_points={len(zoom_rows)}")
    print(f"output={OUT_PNG}")


if __name__ == "__main__":
    main()
