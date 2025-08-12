from __future__ import annotations

import numpy as np

from src.infrastructure.parsers.egdb import Eg2D


def load_geometry(shot_no: int, time_list: np.ndarray) -> dict[str, np.ndarray | float]:
    eg = Eg2D(f"tsmap_nel@{shot_no}.dat")
    out: dict[str, np.ndarray | float] = {}
    out["a99"] = eg.interpolate_series("a99", time_list)
    out["Rax_vmec"] = eg.interpolate_series("Rax_vmec", time_list)
    out["geom_center"] = eg.interpolate_series("geom_center", time_list)

    bt = 0.0
    rax = 0.0
    for line in eg.comments.splitlines():
        if "Bt" in line:
            try:
                bt = float(line.split("Bt")[1].split("=")[-1].strip())
            except Exception:
                pass
        if "Rax" in line:
            try:
                rax = float(line.split("Rax")[1].split("=")[-1].strip())
            except Exception:
                pass
    out["Bt_scalar"] = bt
    out["Rax_scalar"] = rax
    out["B"] = np.full(len(time_list), abs(bt))
    return out


