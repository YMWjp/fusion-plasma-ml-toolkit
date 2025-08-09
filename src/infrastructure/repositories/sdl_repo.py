from __future__ import annotations
import numpy as np
from src.utils.paths import SDL_LOOP_DATA_DIR

def load_sdl_file(shot_no: int) -> np.ndarray:
    """Phieff{shot}.dat を読み (skiprows=1, delimiter=','), shape=(N, M)"""
    path = SDL_LOOP_DATA_DIR / f"Phieff{shot_no}.dat"
    return np.loadtxt(path, skiprows=1, delimiter=',')