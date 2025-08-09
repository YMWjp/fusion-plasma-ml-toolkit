from __future__ import annotations

import numpy as np
from src.infrastructure.parsers.egdb import Eg2D


def load_isat_series(shot_no: int, target_channel: str, gdn_channel_index: int | None,
                     target_times: np.ndarray) -> np.ndarray:
    """
    DivIis_tor_sum@{shot}.dat から、指定チャンネルの Isat 波形を補間取得。
    target_channel 例: '7L' なら 'Iis_7L@{gdn}' を読む。
    """
    eg = Eg2D(f"DivIis_tor_sum@{shot_no}.dat")
    # gdn はファイル内 'gdn:' 行にある場合は自動抽出、なければ引数フォールバック
    if gdn_channel_index is None:
        gdn_list = eg.find_gdn_indices()
        if gdn_list is None:
            return np.zeros_like(target_times)
        # 順序はレガシーの並び（2L,2R,4L,4R,6L,6R,7L,7R,8L,8R,9L,9R,10L,10R）
        channel_order = [
            '2L','2R','4L','4R','6L','6R','7L','7R','8L','8R','9L','9R','10L','10R'
        ]
        try:
            idx_in_list = channel_order.index(target_channel)
            gdn_channel_index = gdn_list[idx_in_list]
        except Exception:
            return np.zeros_like(target_times)
    valname = f"Iis_{target_channel}@{gdn_channel_index}"
    try:
        return eg.interpolate_series(valname, target_times)
    except ValueError:
        return np.zeros_like(target_times)


def repair_isat_7l_outliers(isat: np.ndarray) -> np.ndarray:
    """
    既存処理の外れ値補正（単純補間＋前値ホールド）を再現。
    """
    x = isat.copy()
    if x.size < 5:
        return x
    for i in range(2, len(x) - 2):
        if x[i] < 1.0e-4:
            x[i] = (x[i - 2] + x[i + 2]) / 2.0
    prev = x.copy()
    for i in range(1, len(x)):
        if prev[i] < prev[i - 1] / 1.5:
            x[i] = x[i - 1]
    return x

