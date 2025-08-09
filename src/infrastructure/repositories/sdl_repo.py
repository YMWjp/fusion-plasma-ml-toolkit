from __future__ import annotations
import numpy as np
from pathlib import Path
from src.utils.paths import SDL_LOOP_DATA_DIR

def load_sdl_file(shot_no: int) -> np.ndarray:
    """Phieff{shot}.dat を読み (skiprows=1, delimiter=','), shape=(N, M)"""
    primary = SDL_LOOP_DATA_DIR / f"Phieff{shot_no}.dat"
    if primary.exists():
        return np.loadtxt(primary, skiprows=1, delimiter=',')
    # fallback to legacy location
    legacy = Path(__file__).resolve().parents[3] / 'data' / 'makedata' / 'SDLloopdata' / f"Phieff{shot_no}.dat"
    return np.loadtxt(legacy, skiprows=1, delimiter=',')