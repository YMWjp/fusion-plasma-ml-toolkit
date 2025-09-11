from __future__ import annotations

import numpy as np
from scipy import interpolate

from .. import param
from ..context import Context


def _get_nbi_power_for_units(ctx: Context, time_values: np.ndarray, 
                           diagnames: list[str]) -> np.ndarray:
    """Get NBI power for multiple units with unit conversion."""
    nbi_power_list = []
    
    for diag in diagnames:
        data = ctx.load_and_parse_raw_egdb(diag)
        if data is None:
            continue
            
        # Extract time and power values
        data_time = np.array(data["time"], dtype=float)
        
        # Get the NBI unit number from diagnosis name (e.g., "nb1pwr_temporal" -> "1")
        nb_unit = diag.replace("nb", "").replace("pwr_temporal", "")
        power_column = f"Pport-through_nb{nb_unit}"
        
        # Check if the power column exists
        if power_column not in data.columns:
            continue
            
        power_values = np.array(data[power_column], dtype=float)
        
        # Create interpolation function
        interpolator = interpolate.interp1d(data_time, power_values, bounds_error=False, fill_value=0)
        
        # Interpolate to target time
        interpolated_power = interpolator(time_values)
        
        # NBI data is typically already in MW, so no conversion needed
        nbi_power_list.append(interpolated_power)
    
    if not nbi_power_list:
        return np.array([])
    
    return np.vstack(nbi_power_list)


@param("Pnbi-tan", deps=["time"], needs=["nb1pwr_temporal", "nb2pwr_temporal", "nb3pwr_temporal"], 
       doc="NBI tangential power")
def get_Pnbi_tan(ctx: Context, deps):
    """Calculate tangential NBI power from nb1, nb2, nb3 data."""
    time_values = deps["time"]
    tan_diagnames = ["nb1pwr_temporal", "nb2pwr_temporal", "nb3pwr_temporal"]
    
    nbi_power = _get_nbi_power_for_units(ctx, time_values, tan_diagnames)
    
    if nbi_power.size == 0:
        return np.full_like(time_values, np.nan)
    
    nbi_tan_total = np.sum(nbi_power, axis=0)
    
    return nbi_tan_total


@param("Pnbi-perp", deps=["time"], 
       needs=["nb4apwr_temporal", "nb4bpwr_temporal", "nb5apwr_temporal", "nb5bpwr_temporal"], 
       doc="NBI perpendicular power")
def get_Pnbi_perp(ctx: Context, deps):
    """Calculate perpendicular NBI power from nb4a, nb4b, nb5a, nb5b data."""
    time_values = deps["time"]
    perp_diagnames = ["nb4apwr_temporal", "nb4bpwr_temporal", "nb5apwr_temporal", "nb5bpwr_temporal"]
    
    nbi_power = _get_nbi_power_for_units(ctx, time_values, perp_diagnames)
    
    if nbi_power.size == 0:
        return np.full_like(time_values, np.nan)
    
    nbi_perp_total = np.sum(nbi_power, axis=0)
    
    return nbi_perp_total

@param("Pinput", deps=["Pech","Pnbi-tan","Pnbi-perp"], needs=[], doc="総入力パワー")
def get_Pinput(ctx: Context, deps):
    return deps["Pech"] + deps["Pnbi-tan"] + deps["Pnbi-perp"]*0.36

@param("Prad_over_Pinput", deps=["Prad","Pinput"], needs=[], doc="Prad/Pinput ratio")
def get_Prad_over_Pinput(ctx: Context, deps):
    """Calculate Prad/Pinput ratio with proper handling of zero input power."""
    prad = np.asarray(deps["Prad"], dtype=float)
    pinput = np.asarray(deps["Pinput"], dtype=float)

    threshold = 1e-6
    result = np.full_like(prad, np.nan)
    valid_mask = pinput > threshold
    
    # 有効な値のみ計算
    result[valid_mask] = prad[valid_mask] / pinput[valid_mask]
    
    return result