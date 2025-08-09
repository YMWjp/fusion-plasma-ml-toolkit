from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import tempfile
import os
import sys
import importlib
from src.config.settings import get_parameters, get_basic_info_for_header


def _repo_root() -> Path:
    # src/application から2つ上がリポジトリルート
    return Path(__file__).resolve().parents[2]


def _resolve_paths(output_filename: str) -> tuple[Path, Path, Path, Path]:
    """
    Returns:
      makedata_dir, config_json_path, shot_numbers_csv, output_dataset_path
    """
    root = _repo_root()
    makedata_dir = root / "data" / "makedata"
    config_json_path = makedata_dir / "config.json"
    shot_numbers_csv = root / "src" / "config" / "shot_numbers.csv"
    output_dataset_path = root / "src" / "data" / "datasets" / output_filename
    return makedata_dir, config_json_path, shot_numbers_csv, output_dataset_path


def run_plasma_collection(*,
                          output_filename: str,
                          detection_mode: Optional[str] = None,
                          method: Optional[str] = None) -> None:
    """
    既存の data/makedata/plasma_data_collector.py を呼び出し、
    src 配下の入出力ファイルで処理を実行するラッパー。

    - shot 番号は src/config/shot_numbers.csv を使用
    - 出力は src/data/datasets/{output_filename}
    - 設定は data/makedata/config.json を使用（method 指定時は一時的に上書き）
    - 実行時のカレントディレクトリを makedata に切り替えて互換性を確保
    """
    makedata_dir, config_json_path, shot_numbers_csv, output_dataset_path = _resolve_paths(output_filename)

    # 既存スクリプトの相対 import を解決するために sys.path を追加
    sys.path.insert(0, str(makedata_dir))
    # plasma_data_collector は makedata 直下のモジュール名として import する
    legacy_module = importlib.import_module('plasma_data_collector')
    legacy_main = getattr(legacy_module, 'main')

    # 出力先を作成
    output_dataset_path.parent.mkdir(parents=True, exist_ok=True)

    # config.json を一時的に上書き（常に src のヘッダに同期）
    config_path_to_use: Path = config_json_path
    temp_file: tempfile.NamedTemporaryFile | None = None
    try:
        with config_json_path.open('r', encoding='utf-8') as f:
            config = json.load(f)

        # ヘッダを src/config/parameters.yaml に同期
        headers = get_basic_info_for_header() + get_parameters()
        config['header'] = headers

        # 検出モード・メソッドの上書き
        if detection_mode is not None:
            config['detection_mode'] = detection_mode
        if method is not None:
            config.setdefault('automatic_detection', {})['method'] = method

        temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(config, temp_file, indent=2)
        temp_file.flush()
        config_path_to_use = Path(temp_file.name)

        # 既存コードの相対パスを尊重するため CWD を移動
        old_cwd = Path.cwd()
        os.chdir(makedata_dir)
        try:
            legacy_main(
                savename=str(output_dataset_path),
                labelname=str(shot_numbers_csv),
                detection_mode=detection_mode,
                config_file=str(config_path_to_use),
            )
        finally:
            os.chdir(old_cwd)
    finally:
        if temp_file is not None:
            try:
                os.unlink(temp_file.name)
            except OSError:
                pass


