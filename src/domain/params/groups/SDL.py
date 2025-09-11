from __future__ import annotations

import numpy as np
from scipy import interpolate

from src.utils.paths import SDL_LOOP_DATA_DIR

from .. import param
from ..context import Context


def _load_sdl_data(ctx: Context) -> np.ndarray | None:
    """
    Load SDL data from Phieff file.
    
    File format:
    - Column 0: time(s)
    - Column 1: Phieff(Wb) - dPhi
    - Column 2: Thetaeff(rad) - dTheta  
    - Column 3: Phipl(Wb)
    - Column 4: DelTheta(rad)
    - Column 5: Phiext(Wb) - dPhi_ext
    - Column 6: Thetaext(rad)
    """
    sdl_file_path = SDL_LOOP_DATA_DIR / f"Phieff{ctx.shotNO}.dat"

    if not sdl_file_path.exists():
        raise Warning(f"SDL data file not found: {sdl_file_path}")
    
    try:
        sdl_data = np.loadtxt(sdl_file_path, skiprows=1, delimiter=',')
        return sdl_data
    except Exception as e:
        print(f"Error loading SDL data: {e}")
        raise Warning(f"Error loading SDL data: {e}")


def _interpolate_sdl_value(sdl_data: np.ndarray, time_values: np.ndarray, column_idx: int) -> np.ndarray:
    """Interpolate SDL values for given time array using scipy.interpolate.interp1d."""
    sdl_times = sdl_data[:, 0]
    sdl_values = sdl_data[:, column_idx]
    
    interpolator = interpolate.interp1d(sdl_times, sdl_values, bounds_error=False, fill_value='extrapolate')
    
    return interpolator(time_values)


@param("SDLloop_dPhi", deps=["time"], needs=[], doc="SDLloop dPhi from Phieff data")
def get_SDLloop_dPhi(ctx: Context, deps):
    """Calculate SDLloop dPhi from Phieff data."""
    sdl_data = _load_sdl_data(ctx)
    if sdl_data is None:
        return np.full_like(deps["time"], np.nan)
    
    return _interpolate_sdl_value(sdl_data, deps["time"], 1)


@param("SDLloop_dPhi_ext", deps=["time"], needs=[], doc="SDLloop dPhi_ext from Phieff data")
def get_SDLloop_dPhi_ext(ctx: Context, deps):
    """Calculate SDLloop dPhi_ext from Phieff data."""
    sdl_data = _load_sdl_data(ctx)
    if sdl_data is None:
        return np.full_like(deps["time"], np.nan)
    
    return _interpolate_sdl_value(sdl_data, deps["time"], 5)


@param("SDLloop_dTheta", deps=["time"], needs=[], doc="SDLloop dTheta from Phieff data")
def get_SDLloop_dTheta(ctx: Context, deps):
    """Calculate SDLloop dTheta from Phieff data."""
    sdl_data = _load_sdl_data(ctx)
    if sdl_data is None:
        return np.full_like(deps["time"], np.nan)
    
    dtheta_values = _interpolate_sdl_value(sdl_data, deps["time"], 2)
    return np.abs(dtheta_values)