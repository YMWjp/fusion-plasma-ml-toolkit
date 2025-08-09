from __future__ import annotations
import numpy as np

def align_sdl_to_times(sdl_matrix: np.ndarray, target_times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    元コードの列想定:
      col0: time, col1: dphi, col2: dtheta(絶対値), col5: dphi_ext
    target_times へ最近点で合わせる（元の最近傍選択を踏襲）。
    """
    time = sdl_matrix[:, 0]
    dphi = sdl_matrix[:, 1]
    dtheta = np.abs(sdl_matrix[:, 2])
    dphi_ext = sdl_matrix[:, 5]

    out_dphi = np.zeros_like(target_times, dtype=float)
    out_dtheta = np.zeros_like(target_times, dtype=float)
    out_dphi_ext = np.zeros_like(target_times, dtype=float)

    for i, t in enumerate(target_times):
        idx = int(np.argmin(np.abs(t - time)))
        out_dphi[i] = dphi[idx]
        out_dphi_ext[i] = dphi_ext[idx]
        out_dtheta[i] = dtheta[idx]
    return out_dphi, out_dphi_ext, out_dtheta