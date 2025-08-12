from __future__ import annotations

import numpy as np

from src.infrastructure.parsers.egdb import Eg2D


def load_wp_ip(shot_no: int, time_list: np.ndarray) -> dict[str, np.ndarray]:
    eg_wp = Eg2D(f"wp@{shot_no}.dat")
    wpdia = eg_wp.interpolate_series("Wp", time_list) / 1000.0
    beta = eg_wp.interpolate_series("<beta-dia>", time_list)
    eg_ip = Eg2D(f"ip@{shot_no}.dat")
    ip = eg_ip.interpolate_series("Ip", time_list)
    return {"Wp": wpdia, "beta": beta, "Ip": ip}


