from __future__ import annotations

import numpy as np

from src.infrastructure.parsers.egdb import Eg2D


def _nbiabs(through: np.ndarray, nebar: np.ndarray, s: str, bt_scalar: float) -> np.ndarray:
    try:
        s_int = int(s) if s.isdigit() else -1
    except Exception:
        s_int = -1
    if bt_scalar < 0:
        if s_int in (1, 3):
            loss = 0.28127 + 0.091059 * np.exp(-3.5618 * nebar / 10)
        else:
            loss = -0.010049 + 2.0175 * np.exp(-10.904 * nebar / 10)
            loss[loss < 0] = 0
    else:
        if s_int == 2:
            loss = 0.28127 + 0.091059 * np.exp(-3.5618 * nebar / 10)
        else:
            loss = -0.010049 + 2.0175 * np.exp(-10.904 * nebar / 10)
            loss[loss < 0] = 0
    loss[loss > 1] = 1
    return through * (1 - loss)


def load_nbi(shot_no: int, time_list: np.ndarray, nebar: np.ndarray, bt_scalar: float) -> dict[str, np.ndarray]:
    # tangential 1,2,3
    nb_tmp = np.zeros_like(time_list)
    for s in ["1", "2", "3"]:
        eg = Eg2D(f"nb{s}pwr_temporal@{shot_no}.dat")
        unit = eg.valunits[eg.valname2idx(f"Pport-through_nb{s}")]
        series = eg.interpolate_series(f"Pport-through_nb{s}", time_list)
        if unit == "kW":
            series = series / 1000.0
        nb_tmp = np.vstack((nb_tmp, _nbiabs(series, nebar, s, bt_scalar)))
    pnbi_tan = np.sum(np.abs(nb_tmp), axis=0)

    # perpendicular 4a,4b,5a,5b
    nb_tmp = np.zeros_like(time_list)
    for s in ["4a", "4b", "5a", "5b"]:
        eg = Eg2D(f"nb{s}pwr_temporal@{shot_no}.dat")
        unit = eg.valunits[eg.valname2idx(f"Pport-through_nb{s}")]
        series = eg.interpolate_series(f"Pport-through_nb{s}", time_list)
        if unit == "kW":
            series = series / 1000.0
        nb_tmp = np.vstack((nb_tmp, series))
    pnbi_perp = np.sum(np.abs(nb_tmp), axis=0)
    return {"Pnbi-tan": pnbi_tan, "Pnbi-perp": pnbi_perp}


