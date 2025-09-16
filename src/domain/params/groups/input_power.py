from __future__ import annotations

import numpy as np
from scipy import interpolate

from .. import param
from ..context import Context


def _get_nbi_power_for_units(ctx: Context, time_values: np.ndarray, 
                           diagnames: list[str]) -> np.ndarray:
    """G et NBI power for multiple units with unit conversion."""
    nbi_power_list = []
    
    for diag in diagnames:
        data = ctx.load_and_parse_raw_egdb(diag)
        if data is None:
            continue
            
        # Extract time and power values
        data_time = np.array(data["time"], dtype=float)
        
        # Get the NBI unit number from diagnosis name (e.g., "nb1pwr" -> "1")
        nb_unit = diag.replace("nb", "").replace("pwr", "")
        power_column = f"Pport-through_nb{nb_unit}"
        
        # Check if the power column exists
        if power_column not in data.columns:
            continue
            
        # Convert to MW
        power_values = np.array(data[power_column]*1e-3, dtype=float)
        
        # Create interpolation function
        interpolator = interpolate.interp1d(data_time, power_values, bounds_error=False, fill_value=0)
        
        # Interpolate to target time
        interpolated_power = interpolator(time_values)
        
        nbi_power_list.append(interpolated_power)
    
    if not nbi_power_list:
        return np.array([])
    
    return np.vstack(nbi_power_list)

@param("Pnbi-tan", deps=["time"], needs=["nb1pwr", "nb2pwr", "nb3pwr"], 
       doc="NBI tangential power")
def get_Pnbi_tan(ctx: Context, deps):
    """Calculate tangential NBI power from nb1, nb2, nb3 data."""
    time_values = deps["time"]
    tan_diagnames = ["nb1pwr", "nb2pwr", "nb3pwr"]
    
    nbi_power = _get_nbi_power_for_units(ctx, time_values, tan_diagnames)
    
    if nbi_power.size == 0:
        return np.full_like(time_values, np.nan)
    
    nbi_tan_total = np.sum(nbi_power, axis=0)
    
    return nbi_tan_total


@param("Pech", deps=["time"], needs=["LHDGAUSS_DEPROF"], doc="ECH power")
def get_Pech(ctx: Context, deps):
    eg_echpw = ctx.load_and_parse_raw_egdb_2D("LHDGAUSS_DEPROF", "reff", "max")
    time_list = np.array(eg_echpw["time"], dtype=float)
    Pech_list = np.array(eg_echpw["Sum_Total_Power"], dtype=float)
    f1_Pech = interpolate.interp1d(time_list, Pech_list, bounds_error=False, fill_value=0)
    return f1_Pech(deps["time"])

@param("Pinput", deps=["Pech","Pnbi-tan"], needs=[], doc="総入力パワー")
def get_Pinput(ctx: Context, deps):
    return deps["Pech"] + deps["Pnbi-tan"]