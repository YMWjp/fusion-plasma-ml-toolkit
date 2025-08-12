from __future__ import annotations

import numpy as np

from src.infrastructure.parsers.egdb import Eg2D


def load_impurities_and_ha(shot_no: int, time_list: np.ndarray, nel: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    eg_imp = Eg2D(f"imp02@{shot_no}.dat")
    out["OVI"] = eg_imp.interpolate_series("OVI", time_list)
    out["CIV"] = eg_imp.interpolate_series("CIV", time_list)
    out["OV"] = eg_imp.interpolate_series("OV", time_list)
    out["CIII"] = eg_imp.interpolate_series("CIII", time_list)
    out["FeXVI"] = eg_imp.interpolate_series("FeXVI", time_list)
    out["HI"] = eg_imp.interpolate_series("HI", time_list)

    # gains
    if 154481 <= int(shot_no) <= 157260:
        out["OV"] = 5.368 * out["OV"]
    if 154539 <= int(shot_no) <= 157260:
        out["OVI"] = 2.622 * out["OVI"]
    if (155146 <= int(shot_no) <= 155207) or (158144 <= int(shot_no) <= 158215):
        out["CIII"] = 2.655 * out["CIII"]
        out["CIV"] = 2.896 * out["CIV"]

    # ha3 は除外: D/(H+D) はゼロ埋め
    out["D/(H+D)"] = np.zeros_like(time_list)

    # HeI（ha1 のみ）
    try:
        eg_ha1 = Eg2D(f"ha1@{shot_no}.dat")
        out["HeI"] = eg_ha1.interpolate_series("HeI(Impmon)", time_list)
    except Exception:
        out["HeI"] = np.zeros_like(time_list)

    return out


