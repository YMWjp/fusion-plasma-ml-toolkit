from __future__ import annotations

from pathlib import Path
from typing import Iterable
import requests

from src.utils.paths import EGDATA_DIR


BASE_URL = "http://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi"


def fetch_eg_file(diagname: str, shot_no: int, *, subshot_no: int = 1,
                  save_dir: Path | None = None) -> Path:
    """
    LHD WebAPI から EG データを取得し、`{save_dir}/{diag}@{shot}.dat` で保存。
    既存ファイルがあれば上書き。
    """
    save_dir = save_dir or EGDATA_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    params = {
        "cmd": "getfile",
        "diag": diagname,
        "shotno": int(shot_no),
        "subno": int(subshot_no),
    }
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()

    out_path = save_dir / f"{diagname}@{shot_no}.dat"
    out_path.write_text(resp.text, encoding="utf-8")
    return out_path


def ensure_eg_files(shot_no: int, diagnames: Iterable[str], *, subshot_no: int = 1,
                    save_dir: Path | None = None) -> list[Path]:
    """
    必要なすべての EG ファイルを取得。存在しても最新化のため取得し直す。
    """
    return [fetch_eg_file(diag, shot_no, subshot_no=subshot_no, save_dir=save_dir)
            for diag in diagnames]


