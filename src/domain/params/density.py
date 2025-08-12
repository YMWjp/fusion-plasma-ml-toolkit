from __future__ import annotations

import numpy as np

from src.config.settings import load_config
from src.infrastructure.parsers.egdb import Eg2D


def build_time_and_density(shot_no: int) -> tuple[np.ndarray, np.ndarray]:
    """
    tsmap_nel@{shot}.dat から Thomson 系の時間軸と nel を構築。
    FIR との係数スケーリングを反映。
    戻り値: (time_list, nel)
    """
    cfg = load_config()
    dt = float(cfg["processing"]["sampling"]["dt"])

    eg = Eg2D(f"tsmap_nel@{shot_no}.dat")
    time_raw = np.array(eg.dimdata[1:], dtype=float) if len(eg.dimdata) > 1 else np.array(eg.dimdata, dtype=float)
    if time_raw.size == 0:
        raise RuntimeError("No Thomson time series")

    th = np.array(eg.interpolate_series("nl_thomson_3669", time_raw))
    fir = np.array(eg.interpolate_series("nl_fir_3669", time_raw))

    win = cfg["processing"]["scaling"]["fir_scale_window"]
    in_window = np.logical_and(time_raw > float(win[0]), time_raw < float(win[1]))
    if np.any(in_window):
        factor = np.nanmean(fir[in_window]) / np.nanmean(th[in_window])
    elif time_raw.size > 3:
        factor = np.nanmean(fir[:3]) / np.nanmean(th[:3])
    else:
        factor = 1.0

    t_min, t_max = float(np.nanmin(time_raw)), float(np.nanmax(time_raw))
    time_list = np.arange(t_min, t_max, dt)
    nel = Eg2D(f"tsmap_nel@{shot_no}.dat").interpolate_series("nl_thomson_3669", time_list) * factor
    return time_list, nel


