# src/utils/paths.py
from pathlib import Path
from typing import Literal

import yaml

# src directory
SRC_DIR: Path = Path(__file__).resolve().parent.parent
# data root
DATA_DIR: Path = SRC_DIR / "data"
# each sub directory
DATASETS_DIR: Path = DATA_DIR / "datasets"
LOGS_DIR: Path = DATA_DIR / "logs"

EGDATA_DIR: Path = DATA_DIR / "egdata"
SDL_LOOP_DATA_DIR: Path = DATA_DIR / "SDLloopdata"

EXPERIMENT_LOG_CSV: Path = DATA_DIR / "experiment_log_new.csv"

# create directories if not exist
for _p in (DATA_DIR, DATASETS_DIR, LOGS_DIR):
    _p.mkdir(parents=True, exist_ok=True)


def get_egdata_path(diagname: str, shot_no: int) -> Path:
    """
    EG データの保存パスを返す。config の `egdata.layout` に従い、
    - flat: src/data/egdata/{diag}@{shot}.dat
    - by_shot: src/data/egdata/{shot}/[{diag}@]{shot}.dat
    """
    config_path = SRC_DIR / "config" / "config.yaml"
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        layout: Literal["flat", "by_shot"] = cfg.get("egdata", {}).get("layout", "flat")
        if layout == "by_shot":
            fmt = cfg.get("egdata", {}).get("by_shot_dirname_format", "{shot_no}")
            shot_dir = EGDATA_DIR / fmt.format(shot_no=int(shot_no))
            shot_dir.mkdir(parents=True, exist_ok=True)
            return shot_dir / f"{diagname}@{int(shot_no)}.dat"
    except Exception:
        pass
    return EGDATA_DIR / f"{diagname}@{int(shot_no)}.dat"


def resolve_egdata_file(filename: str) -> Path:
    """
    与えられたファイル名（例: "tsmap_nel@163402.dat"）に対して、
    現在のレイアウト設定に従った実ファイルパスを返す。
    - まず flat パスを確認
    - 見つからなければ、diag と shot を抽出して by_shot パスを返す
    """
    flat = EGDATA_DIR / filename
    if flat.exists():
        return flat
    try:
        stem = filename
        if stem.endswith('.dat'):
            stem = stem[:-4]
        diag, rest = stem.split('@', 1)
        shot = int(rest)
        by_shot = get_egdata_path(diag, shot)
        return by_shot
    except Exception:
        return flat