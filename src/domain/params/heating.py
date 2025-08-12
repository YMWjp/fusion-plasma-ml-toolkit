from __future__ import annotations

import numpy as np

from src.infrastructure.parsers.egdb import Eg2D


def load_heating_and_radiation(shot_no: int, time_list: np.ndarray) -> dict[str, np.ndarray]:
    eg_ech = Eg2D(f"echpw@{shot_no}.dat")
    pech = eg_ech.interpolate_series("Total ECH", time_list)
    eg_bolo = Eg2D(f"bolo@{shot_no}.dat")
    prad = eg_bolo.interpolate_series("Rad_PW", time_list) / 1000.0
    return {"Pech": pech, "Prad": prad}


