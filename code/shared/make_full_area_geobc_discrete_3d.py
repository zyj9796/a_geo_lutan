from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from make_publication_3d_geobc import OUT_DIR, TEXT, plot_scene
from search_geobc_3d_view_angles import prepare_data


OUT_PNG = OUT_DIR / "fig_08_geobc_3d_full_area_discrete.png"
POINT_SIZE = 0.7
BUILDING_ALPHA = 0.22
VIEW_ELEV = 38
VIEW_AZIM = 70
VIEW_ROLL = 0

CLASS_BOUNDS = np.asarray([-30, -10, -5, -2, 2, 5, 10, 30], dtype=float)
CLASS_COLORS = [
    "#8b0000",
    "#d73027",
    "#fc8d59",
    "#2ca25f",
    "#41b6c4",
    "#2c7fb8",
    "#084081",
]
CLASS_LABELS = [
    "≤ -10\n严重沉降",
    "-10 至 -5",
    "-5 至 -2",
    "-2 至 2\n稳定",
    "2 至 5",
    "5 至 10",
    "≥ 10\n正形变",
]
CLASS_CMAP = ListedColormap(CLASS_COLORS, name="deformation_classes")
CLASS_NORM = BoundaryNorm(CLASS_BOUNDS, CLASS_CMAP.N, clip=True)


def add_discrete_colorbar(fig, scatter, counts: list[int]) -> None:
    cax = fig.add_axes([0.135, 0.075, 0.73, 0.032])
    cbar = fig.colorbar(
        scatter,
        cax=cax,
        orientation="horizontal",
        boundaries=CLASS_BOUNDS,
        spacing="uniform",
    )
    centers = (CLASS_BOUNDS[:-1] + CLASS_BOUNDS[1:]) / 2.0
    cbar.set_ticks(centers)
    cbar.set_ticklabels(
        [f"{label}\n{count} 个" for label, count in zip(CLASS_LABELS, counts, strict=True)]
    )
    cbar.ax.tick_params(labelsize=6.2, length=0, pad=4)
    cbar.outline.set_linewidth(0.6)
    cbar.set_label(r"形变速率等级（毫米·年$^{-1}$）", fontsize=7.5, labelpad=5)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    buildings, rows, selected_fids = prepare_data()
    velocity = np.asarray([row["velocity"] for row in rows], dtype=float)
    clipped = np.clip(velocity, CLASS_BOUNDS[0], CLASS_BOUNDS[-1])
    counts = np.histogram(clipped, bins=CLASS_BOUNDS)[0].astype(int).tolist()

    fig = plt.figure(figsize=(12.2, 6.2), dpi=300)
    ax = fig.add_axes([0.01, 0.20, 0.98, 0.75], projection="3d")
    scatter = plot_scene(
        ax,
        buildings,
        rows,
        selected_fids,
        detail=False,
        show_guides=False,
        point_size=POINT_SIZE,
        point_cmap=CLASS_CMAP,
        point_norm=CLASS_NORM,
        building_alpha=BUILDING_ALPHA,
    )
    ax.set_box_aspect((2.55, 0.68, 0.42), zoom=2.10)
    ax.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM, roll=VIEW_ROLL)
    fig.text(
        0.025,
        0.965,
        "全区域 Geo-BC 形变等级",
        fontsize=12,
        fontweight="bold",
        color=TEXT,
        va="top",
    )
    fig.text(
        0.025,
        0.915,
        "透明三维建筑背景与缩小的点标记",
        fontsize=7.5,
        color="#555b61",
        va="top",
    )
    fig.text(
        0.975,
        0.165,
        f"{len(buildings)} 栋建筑｜{len(rows)} 个点｜标记面积 {POINT_SIZE:g} 平方磅｜建筑透明度 {BUILDING_ALPHA:.2f}",
        fontsize=6.4,
        color="#666b70",
        ha="right",
    )
    add_discrete_colorbar(fig, scatter, counts)
    fig.savefig(OUT_PNG, dpi=300)
    plt.close(fig)
    print(f"buildings={len(buildings)}")
    print(f"points={len(rows)}")
    print(f"class_counts={counts}")
    print(f"point_size_pt2={POINT_SIZE}")
    print(f"building_alpha={BUILDING_ALPHA}")
    print(f"view=elev_{VIEW_ELEV}_azim_{VIEW_AZIM}_roll_{VIEW_ROLL}")
    print(f"output={OUT_PNG}")


if __name__ == "__main__":
    main()
