from __future__ import annotations

import numpy as np
from scipy import interpolate

from .. import param
from ..context import Context


@param("D/(H+D)", deps=["time"], needs=["ha3"], doc="D/(H+D)")
def get_D_H_D(ctx: Context, deps):
    """Calculate D/(H+D) ratio from ha3 data with time interpolation."""
    ha3_data = ctx.load_and_parse_raw_egdb("ha3")
    
    if ha3_data is None:
        return np.full_like(deps["time"], np.nan)
    
    # Extract time and D/(H+D) values
    time_values = np.array(ha3_data["Time"], dtype=float)
    d_h_d_ratio = np.array(ha3_data["D/(H+D)"], dtype=float)
    
    # Create interpolation function
    interpolator = interpolate.interp1d(
        time_values, 
        d_h_d_ratio, 
        bounds_error=False, 
        fill_value=0
    )
    
    return interpolator(deps["time"])