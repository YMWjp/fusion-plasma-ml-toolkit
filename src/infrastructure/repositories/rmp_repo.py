from __future__ import annotations
import numpy as np
from functools import lru_cache
from src.utils.paths import EXPERIMENT_LOG_CSV

SHOT_COL = 2
RMP_COL  = 4

@lru_cache(maxsize=1)
def _load_experiment_log_T() -> np.ndarray:
    return np.loadtxt(EXPERIMENT_LOG_CSV, skiprows=1, delimiter=',').T  # (cols, rows)

def load_rmp_flag_from_csv(shot_no: int) -> int:
    data = _load_experiment_log_T()
    shot_col = data[SHOT_COL]
    rmp_col = data[RMP_COL].copy()
    rmp_col[rmp_col == 0] = 1
    matched = rmp_col[shot_col == shot_no]
    return int(matched[0]) if matched.size else 1