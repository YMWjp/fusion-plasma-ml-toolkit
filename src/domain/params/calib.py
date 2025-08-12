from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from src.infrastructure.parsers.eg3d import TsmapCalib


def load_tsmap_calib_series(shot_no: int, time_list: np.ndarray) -> dict[str, np.ndarray]:
    try:
        tsc = TsmapCalib(f"tsmap_calib@{shot_no}.dat")
        reff100, ne100, dV100 = tsc.ne_from_Te(0.1)
        Te_center, ne_center = tsc.Te_from_reff(0)
        t_tsc = np.asarray(tsc.time)
        f_reff = interp1d(t_tsc, reff100, bounds_error=False, fill_value=0.0)
        f_ne100 = interp1d(t_tsc, ne100, bounds_error=False, fill_value=0.0)
        f_dV100 = interp1d(t_tsc, dV100, bounds_error=False, fill_value=0.0)
        f_Te_center = interp1d(t_tsc, Te_center, bounds_error=False, fill_value=0.0)
        f_ne_center = interp1d(t_tsc, ne_center, bounds_error=False, fill_value=0.0)
        return {
            "reff@100eV": f_reff(time_list),
            "ne@100eV": f_ne100(time_list),
            "dVdreff@100eV": f_dV100(time_list),
            "Te@center": f_Te_center(time_list),
            "ne@center": f_ne_center(time_list),
            "Te@edge": np.zeros_like(time_list),
        }
    except Exception:
        return {
            "reff@100eV": np.zeros_like(time_list),
            "ne@100eV": np.zeros_like(time_list),
            "dVdreff@100eV": np.zeros_like(time_list),
            "Te@center": np.zeros_like(time_list),
            "ne@center": np.zeros_like(time_list),
            "Te@edge": np.zeros_like(time_list),
        }


