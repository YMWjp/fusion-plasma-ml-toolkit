from __future__ import annotations

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config.settings import (
    get_eg_data_sources,
    get_parameters,
    get_shot_numbers,
    load_config,
)
from src.domain.params.autodiscover import autodiscover
from src.domain.params.context import Context
from src.domain.params.executor import compute, list_required_egs
from src.infrastructure.clients.lhd_api import ensure_eg_files

# moved detailed parsers usage into domain/params modules
from src.utils.paths import DATASETS_DIR, EGDATA_DIR, LOGS_DIR
from src.utils.utils import write_csv_header

cfg = load_config()
# NE_LENGTH = float(cfg['processing']['constants']['ne_length'])
# DT = float(cfg['processing']['sampling']['dt'])

def run_native_pipeline(
        *,
        detection_mode: str | None = None,
        method: str | None = None
        ) -> None:
    
    # 使用するパラメータ一覧
    params = get_parameters()

    # csvファイルのヘッダー
    csv_headers = ["shotNO"] + params
    
    # 作成するcsvのファイル名
    out_name = cfg['files']['output_dataset']
    out_path = DATASETS_DIR / out_name
    
    write_csv_header(out_path, csv_headers, overwrite=True)

    # 使用するショット番号一覧
    shots = get_shot_numbers()

    # 使用するデータソース
    data_sources = get_eg_data_sources()

    autodiscover()

    required_egs = list_required_egs(params)

    pbar = tqdm(shots, desc="Processing shots", unit="shot")
    # エラーログ出力先
    error_log_path = LOGS_DIR / str(cfg['files'].get('error_log', 'errorshot.txt'))
    error_log_path.parent.mkdir(parents=True, exist_ok=True)

    # パラメータの依存関係を解決
    for shot in pbar:
        # just for a progress bar
        pbar.set_postfix_str(f"shot={int(shot)}")
        # 必要ファイルが存在しない/失敗しても個別スキップ
        # try:
        if required_egs:
            ensure_eg_files(int(shot), required_egs)

        ctx = Context(
            shotNO=int(shot),
            data_root=str(EGDATA_DIR) + f"/{shot}",
            data_sources=data_sources,
            cfg=cfg,
        )
        # パラメータの計算
        results = compute(ctx, params, strict=True)
        time = np.asarray(results['time'])
        n = len(time)
        if n == 0:
            tqdm.write(f"[INFO] shot {shot} skipped (no valid window)")
            continue

        rowmap = {"shotNO": np.full(n, int(shot), dtype=int)}
        for p in params:
            rowmap[p] = np.asarray(results[p])

        df = pd.DataFrame(rowmap, columns=csv_headers)
        df.to_csv(out_path, mode='a', header=False, index=False)

        # except Exception as e:
        #     # スキップ理由を表示・記録
        #     tqdm.write(f"[WARN] shot {int(shot)} skipped: {e}")
        #     try:
        #         with error_log_path.open('a', encoding='utf-8') as f:
        #             f.write(f"shot {int(shot)}: {e}\n")
        #     except Exception:
        #         pass
        #     continue
