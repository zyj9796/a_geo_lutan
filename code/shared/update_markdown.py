from __future__ import annotations

import json
from pathlib import Path

from io_paths import PROJECT_DIR, RESULTS_DIR, SUMMARY_DIR


START = "<!-- AUTO_UPDATE_MD_START -->"
END = "<!-- AUTO_UPDATE_MD_END -->"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _latest_full_area_summary() -> dict | None:
    summaries = sorted(SUMMARY_DIR.glob("*_full_area_summary.json"))
    if not summaries:
        return None
    return _load_json(summaries[-1])


def _image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.png") if item.is_file())


def _auto_block() -> str:
    main = _load_json(SUMMARY_DIR / "main_summary.json")
    full = _latest_full_area_summary()

    lines = [
        START,
        "## 自动更新结果摘要",
        "",
        "本节由 `code/update_markdown.py` 根据 `results/summaries/` 和当前输出目录自动生成。",
        "",
    ]

    if main:
        scenes = ", ".join(str(scene) for scene in main.get("scenes", [])) or "N/A"
        lines.extend(
            [
                "### 小批量主流程",
                "",
                f"- 处理场景: {scenes}",
                f"- 有效建筑-时相组合: {_fmt(main.get('building_scene_records'))}",
                f"- 建筑物约束散射点: {_fmt(main.get('scatter_points'))} 个",
                f"- 建筑物约束方法平均边界距离: {_fmt(main.get('proposed_mean_boundary_distance_m'), 10)} m",
                f"- 建筑物约束方法 90 分位边界距离: {_fmt(main.get('proposed_p90_boundary_distance_m'), 10)} m",
                f"- 传统 GAMMA/DEM 平均边界距离: {_fmt(main.get('gamma_dem_mean_boundary_distance_m'), 4)} m",
                f"- 传统 GAMMA/DEM 90 分位边界距离: {_fmt(main.get('gamma_dem_p90_boundary_distance_m'), 4)} m",
                "",
            ]
        )
    else:
        lines.extend(["### 小批量主流程", "", "- 暂未找到 `results/summaries/main_summary.json`。", ""])

    if full:
        date = str(full.get("date", "N/A"))
        lines.extend(
            [
                "### 全区域流程",
                "",
                f"- 处理日期: {date}",
                f"- 有效建筑物: {_fmt(full.get('valid_buildings'))} 栋",
                f"- 跳过建筑物: {_fmt(full.get('skipped_buildings'))} 栋",
                f"- 建筑物约束散射点: {_fmt(full.get('scatter_points'))} 个",
                f"- 建筑物约束方法平均边界距离: {_fmt(full.get('method_mean_boundary_distance_m'), 6)} m",
                f"- 建筑物约束方法 90 分位边界距离: {_fmt(full.get('method_p90_boundary_distance_m'), 6)} m",
                f"- 传统 GAMMA/DEM 平均边界距离: {_fmt(full.get('gamma_dem_mean_boundary_distance_m'), 3)} m",
                f"- 传统 GAMMA/DEM 90 分位边界距离: {_fmt(full.get('gamma_dem_p90_boundary_distance_m'), 3)} m",
                "",
                "全区域关键输出：",
                "",
                f"- `results/tables/full_area/{date}_all_buildings_fig5_4_like_stats.csv`",
                f"- `results/tables/full_area/{date}_all_buildings_method_vs_gamma_points.csv`",
                f"- `results/tables/full_area/{date}_all_buildings_skipped.csv`",
                f"- `results/geodata/full_area/{date}_all_valid_geocoded_buildings.geojson`",
                f"- `results/geodata/full_area/{date}_all_buildings_proposed_points.geojson`",
                f"- `results/images/full_area/{date}_fig_full_area_gamma_vs_proposed.png`",
                f"- `results/images/full_area/{date}_fig_full_area_error_statistics.png`",
                "",
            ]
        )
    else:
        lines.extend(["### 全区域流程", "", "- 暂未找到全区域摘要 `results/summaries/*_full_area_summary.json`。", ""])

    lines.extend(
        [
            "### 当前图件数量",
            "",
            f"- 主流程图件: {_image_count(RESULTS_DIR / 'images' / 'main')} 张",
            f"- 全区域图件: {_image_count(RESULTS_DIR / 'images' / 'full_area')} 张",
            f"- geo_bc/PS 对比图件: {_image_count(RESULTS_DIR / 'images' / 'full_area_geobc_ps')} 张",
            f"- 形变专题图件: {_image_count(RESULTS_DIR / 'images' / 'full_area_geobc_ps' / 'defo')} 张",
            END,
        ]
    )
    return "\n".join(lines)


def _replace_between_markers(text: str, block: str) -> str | None:
    start = text.find(START)
    end = text.find(END)
    if start == -1 or end == -1 or end < start:
        return None
    end += len(END)
    return text[:start].rstrip() + "\n\n" + block + "\n\n" + text[end:].lstrip()


def _replace_section(text: str, heading: str, next_heading: str, block: str) -> str | None:
    start = text.find(heading)
    if start == -1:
        return None
    end = text.find(next_heading, start + len(heading))
    if end == -1:
        end = len(text)
    return text[:start].rstrip() + "\n\n" + block + "\n\n" + text[end:].lstrip()


def update_file(path: Path) -> bool:
    block = _auto_block()
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    updated = _replace_between_markers(text, block)
    if updated is None:
        updated = _replace_section(text, "## Current Results", "## Images", block)
    if updated is None:
        updated = _replace_section(text, "## 本次结果摘要", "## Huajiachi 全区域地理编码补充实验", block)
    if updated is None:
        updated = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(updated.rstrip() + "\n", encoding="utf-8")
    return updated != text


def main() -> None:
    targets = [
        PROJECT_DIR / "README_HUAJIACHI.md",
        RESULTS_DIR / "README.md",
    ]
    for target in targets:
        changed = update_file(target)
        print(f"{'updated' if changed else 'unchanged'} {target}")


if __name__ == "__main__":
    main()
