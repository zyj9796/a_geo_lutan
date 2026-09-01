from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT_DIR = RESULTS / "all_images"
MANIFEST = OUT_DIR / "MANIFEST.json"


GROUPS: dict[str, tuple[Path, tuple[str, ...]]] = {
    "main": (
        RESULTS / "images" / "main",
        (
            "fig_01_huajiachi_sar_intensity_with_buildings.png",
            "fig_02_initial_projection_masks.png",
            "fig_03_refined_masks.png",
            "fig_04_initial_vs_refined_masks.png",
            "fig_05_method_geocoded_points.png",
            "fig_06_gamma_vs_proposed_map.png",
            "fig_07_error_statistics.png",
            "fig_08_3d_scatter_points.png",
        ),
    ),
    "full_area": (
        RESULTS / "images" / "full_area",
        (
            "20250124_fig5_4_like_all_buildings_error_scatter.png",
            "20250124_fig5_4_like_all_buildings_map.png",
            "20250124_fig_full_area_error_statistics.png",
            "20250124_fig_full_area_gamma_vs_proposed.png",
            "20250124_fig_full_area_planar_method_vs_gamma.png",
            "fig_10_full_area_displacement_vector_field.png",
            "fig_11_full_area_planar_difference_heatmap.png",
            "fig_12_typical_building_zoom_planar_comparison.png",
            "fig_13_ranked_boundary_error_improvement.png",
            "fig_14_height_vs_boundary_error.png",
            "fig_15_boundary_error_cdf.png",
        ),
    ),
    "coordinates": (
        RESULTS / "images" / "full_area_geobc_ps",
        tuple(f"fig_{index:02d}_{suffix}.png" for index, suffix in (
            (1, "full_area_geobc_buildings"),
            (2, "full_area_lutan_coordinates_buildings"),
            (3, "full_area_geobc_vs_lutan_buildings"),
            (4, "full_area_geobc_sar_buildings"),
            (5, "full_area_lutan_coordinates_sar_buildings"),
            (6, "full_area_geobc_vs_lutan_sar_buildings"),
        )),
    ),
    "deformation": (
        RESULTS / "images" / "full_area_geobc_ps" / "defo",
        (
            "fig_07_full_area_geobc_deformation_rate_buildings.png",
            "fig_08_full_area_lutan_deformation_rate_buildings.png",
            "fig_09_full_area_geobc_vs_lutan_deformation_rate_buildings.png",
            "fig_13_full_area_geobc_cumulative_deformation_buildings.png",
            "fig_14_full_area_lutan_cumulative_deformation_buildings.png",
            "fig_15_full_area_geobc_vs_lutan_cumulative_deformation_buildings.png",
        ),
    ),
    "3d": (
        RESULTS / "pic_all",
        (
            "fig_09_lutan_full_area_geobc_3d.png",
            "fig_10_lutan_planar_geobc_3d_extent.png",
            "fig_11_lutan_hotspot_3d_zoom.png",
            "fig_12_lutan_interpolated_building_surfaces_3d.png",
        ),
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if len(header) != 24 or header[:8] != b"\x89PNG\r\n\x1a\n" or header[12:16] != b"IHDR":
        raise ValueError(f"Not a valid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def entries() -> list[tuple[str, Path, Path]]:
    found: list[tuple[str, Path, Path]] = []
    for group, (source_dir, names) in GROUPS.items():
        for name in names:
            source = source_dir / name
            output = OUT_DIR / f"{group}__{name}"
            found.append((group, source, output))
    if len(found) != 35:
        raise AssertionError(f"Expected 35 configured images, got {len(found)}")
    return found


def collect() -> None:
    missing = [source for _group, source, _output in entries() if not source.is_file()]
    if missing:
        formatted = "\n".join(f"- {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(f"Cannot build all_images; missing source figures:\n{formatted}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for group, source, output in entries():
        shutil.copy2(source, output)
        width, height = png_size(output)
        manifest_rows.append(
            {
                "group": group,
                "output": str(output.relative_to(ROOT)),
                "source": str(source.relative_to(ROOT)),
                "sha256": sha256(output),
                "bytes": output.stat().st_size,
                "width_px": width,
                "height_px": height,
            }
        )
    MANIFEST.write_text(
        json.dumps({"image_count": len(manifest_rows), "images": manifest_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"collected_images={len(manifest_rows)}")
    print(f"output_dir={OUT_DIR}")
    print(f"manifest={MANIFEST}")


def check() -> None:
    problems: list[str] = []
    for _group, source, output in entries():
        if not source.is_file():
            problems.append(f"missing source: {source.relative_to(ROOT)}")
            continue
        if not output.is_file():
            problems.append(f"missing output: {output.relative_to(ROOT)}")
            continue
        try:
            png_size(output)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        if sha256(source) != sha256(output):
            problems.append(f"content mismatch: {output.relative_to(ROOT)}")
    if not MANIFEST.is_file():
        problems.append(f"missing manifest: {MANIFEST.relative_to(ROOT)}")
    if problems:
        raise RuntimeError("Image package check failed:\n" + "\n".join(f"- {item}" for item in problems))
    print("image_package_check=ok")
    print(f"checked_images={len(entries())}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and verify the 35 LuTan result figures.")
    parser.add_argument("--check", action="store_true", help="Verify source/output byte identity and PNG headers.")
    args = parser.parse_args()
    if args.check:
        check()
    else:
        collect()


if __name__ == "__main__":
    main()
