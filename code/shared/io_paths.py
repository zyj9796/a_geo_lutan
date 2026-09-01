from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
from matplotlib import font_manager


# 所有项目图件统一使用可显示简体中文的字体。这里放在公共路径模块中，
# 是为了让各个独立绘图脚本无需重复配置，同时保证负号正常显示。
_CJK_FONT_FILE = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
if _CJK_FONT_FILE.exists():
    font_manager.fontManager.addfont(str(_CJK_FONT_FILE))
_CJK_FONT_FAMILY = font_manager.FontProperties(fname=str(_CJK_FONT_FILE)).get_name() if _CJK_FONT_FILE.exists() else "DejaVu Sans"

matplotlib.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": [_CJK_FONT_FAMILY, "DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(os.environ.get("SAR_GEOCODE_PROJECT_DIR", CODE_DIR.parent)).resolve()
REPO_ROOT = Path(os.environ.get("SAR_GEOCODE_REPO_ROOT", PROJECT_DIR.parent)).resolve()

DATA_DIR = Path(os.environ.get("SAR_GEOCODE_DATA_DIR", PROJECT_DIR / "data")).resolve()
RSLC_DIR = Path(os.environ.get("SAR_GEOCODE_RSLC_DIR", DATA_DIR / "RE_SLAVES")).resolve()
BUILDINGS_SHP = Path(
    os.environ.get("SAR_GEOCODE_BUILDINGS_SHP", DATA_DIR / "shp" / "huajiachi_clip.shp")
).resolve()
DSM_SAR_EXTENT_TIF = Path(
    os.environ.get("SAR_GEOCODE_DSM_SAR_EXTENT_TIF", DATA_DIR / "huajiachi_dsm_sar_extent.tif")
).resolve()
PS_POINTS_CSV = Path(os.environ.get("SAR_GEOCODE_PS_POINTS_CSV", DATA_DIR / "ps_points_all.csv")).resolve()


def _discover_dsm() -> Path:
    if DSM_SAR_EXTENT_TIF.exists():
        return DSM_SAR_EXTENT_TIF
    configured = os.environ.get("SAR_GEOCODE_DSM_TIF")
    if configured:
        return Path(configured).resolve()
    for preferred in [DATA_DIR / "huajiachi_dsm.tif", DATA_DIR / "hangzhou_dsm_1m.tif", DATA_DIR / "tongji_dsm.tif"]:
        if preferred.exists():
            return preferred
    candidates = sorted(DATA_DIR.glob("*dsm*.tif")) + sorted(DATA_DIR.glob("*DSM*.tif")) + sorted(DATA_DIR.glob("*.tif"))
    for path in candidates:
        if path.is_file() and path.name != DSM_SAR_EXTENT_TIF.name:
            return path
    return preferred


DSM_TIF = _discover_dsm()

RESULTS_DIR = PROJECT_DIR / "results"
TABLE_ROOT = RESULTS_DIR / "tables"
TABLE_DIR = TABLE_ROOT / "main"
FULL_AREA_TABLE_DIR = TABLE_ROOT / "full_area"
SAME_PIXEL_TABLE_DIR = TABLE_ROOT / "psinsar_same_pixel"
TYPICAL_TABLE_DIR = TABLE_ROOT / "typical_buildings"
LEGACY_TABLE_DIR = TABLE_ROOT / "legacy"

GEOJSON_ROOT = RESULTS_DIR / "geodata"
GEOJSON_DIR = GEOJSON_ROOT / "main"
FULL_AREA_GEOJSON_DIR = GEOJSON_ROOT / "full_area"

RASTER_ROOT = RESULTS_DIR / "rasters"
TIF_DIR = RASTER_ROOT / "main"

LOG_ROOT = RESULTS_DIR / "logs"
LOG_DIR = LOG_ROOT / "main"
FULL_AREA_LOG_DIR = LOG_ROOT / "full_area"

SUMMARY_DIR = RESULTS_DIR / "summaries"
WORK_DIR = RESULTS_DIR / "work"

IMAGE_DIR = RESULTS_DIR / "images"
PIC_ALL_DIR = RESULTS_DIR / "pic_all"
PIC_ALL_KEEP_MANIFEST = RESULTS_DIR / "pic_all_keep.json"
MAIN_IMAGE_DIR = IMAGE_DIR / "main"
FULL_AREA_IMAGE_DIR = IMAGE_DIR / "full_area"
SAME_PIXEL_IMAGE_DIR = IMAGE_DIR / "psinsar_same_pixel"
PPT_IMAGE_DIR = IMAGE_DIR / "ppt_package"
TYPICAL_IMAGE_DIR = IMAGE_DIR / "typical_buildings"
LEGACY_IMAGE_DIR = IMAGE_DIR / "legacy"

FULL_AREA_DIR = FULL_AREA_TABLE_DIR
SAME_PIXEL_DIR = SAME_PIXEL_TABLE_DIR
PPT_DIR = RESULTS_DIR / "psinsar_same_pixel_ppt_package"
PPT_CSV_DIR = PPT_DIR / "csv"
PPT_DOC_DIR = PPT_DIR / "docs"
TYPICAL_DIR = TYPICAL_TABLE_DIR
TRASH_DIR = LEGACY_TABLE_DIR

PPT_ZIP = RESULTS_DIR / "psinsar_same_pixel_ppt_package.zip"


def ensure_core_output_dirs() -> None:
    for path in [
        RESULTS_DIR,
        TABLE_ROOT,
        TABLE_DIR,
        FULL_AREA_TABLE_DIR,
        SAME_PIXEL_TABLE_DIR,
        TYPICAL_TABLE_DIR,
        LEGACY_TABLE_DIR,
        GEOJSON_ROOT,
        GEOJSON_DIR,
        FULL_AREA_GEOJSON_DIR,
        RASTER_ROOT,
        TIF_DIR,
        LOG_ROOT,
        LOG_DIR,
        FULL_AREA_LOG_DIR,
        SUMMARY_DIR,
        WORK_DIR,
        IMAGE_DIR,
        PIC_ALL_DIR,
        MAIN_IMAGE_DIR,
        FULL_AREA_IMAGE_DIR,
        SAME_PIXEL_IMAGE_DIR,
        PPT_IMAGE_DIR,
        TYPICAL_IMAGE_DIR,
        LEGACY_IMAGE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def full_area_file(date: str, suffix: str) -> Path:
    return FULL_AREA_DIR / f"{date}_{suffix}"


def full_area_image(date: str, suffix: str) -> Path:
    return FULL_AREA_IMAGE_DIR / f"{date}_{suffix}"


def typical_image(date: str, suffix: str) -> Path:
    return TYPICAL_IMAGE_DIR / f"{date}_{suffix}"
