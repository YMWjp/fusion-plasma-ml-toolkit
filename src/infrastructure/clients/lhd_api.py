from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import requests

from src.config.settings import load_config
from src.utils.paths import EGDATA_DIR, SRC_DIR, get_egdata_path

BASE_URL = "http://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi"


def fetch_eg_file(diagname: str, shot_no: int, *, subshot_no: int | None = None,
                  save_dir: Path | None = None) -> Path:
    """
    LHD WebAPI から EG データを取得し、`{save_dir}/{diag}@{shot}.dat` で保存。
    既存ファイルがあれば上書き。
    """
    # 保存先を決定（config の egdata.layout に追従）
    target_path = (save_dir or EGDATA_DIR)
    # save_dir が未指定なら get_egdata_path を使用
    out_path = get_egdata_path(diagname, shot_no) if save_dir is None else (target_path / f"{diagname}@{shot_no}.dat")
    # 既存の場合は取得をスキップ（処理時間短縮）
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # 取得トライ: 指定がなければ config の subshots を順に試す
    cfg = load_config()
    subshot_list = [int(subshot_no)] if subshot_no is not None else cfg.get('egdata', {}).get('subshots', [1, 2, 0])
    last_exc: Exception | None = None
    for sub in subshot_list:
        params = {
            "cmd": "getfile",
            "diag": diagname,
            "shotno": int(shot_no),
            "subno": int(sub),
        }
        for base in (BASE_URL, BASE_URL.replace("http:", "https:")):
            try:
                resp = requests.get(base, params=params, timeout=60)
                resp.raise_for_status()
                # 簡易バリデーション（空や明らかなエラーテキストの場合を弾く）
                text = resp.text or ""
                if not text.strip():
                    raise ValueError("empty response body")
                # 保存
                out_path.write_text(text, encoding="utf-8")
                return out_path
            except Exception as e:  # 次の base / subshot へ
                last_exc = e
                continue
    # すべて失敗
    # ローカルのレガシー egdata からのフォールバック（存在すればコピー）
    legacy_path = (SRC_DIR.parent / "data" / "makedata" / "egdata" / f"{diagname}@{int(shot_no)}.dat")
    if legacy_path.exists():
        out_path.write_text(legacy_path.read_text(encoding="utf-8"), encoding="utf-8")
        return out_path
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unknown error in fetch_eg_file")


def ensure_eg_files(shot_no: int, diagnames: Iterable[str], *, subshot_no: int | None = None,
                    save_dir: Path | None = None) -> list[Path]:
    """
    必要なすべての EG ファイルを取得。既存ファイルがあればスキップ。
    """
    return [fetch_eg_file(diag, shot_no, subshot_no=subshot_no, save_dir=save_dir)
            for diag in diagnames]


