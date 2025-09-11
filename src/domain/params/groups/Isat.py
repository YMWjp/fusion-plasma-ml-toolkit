from __future__ import annotations

import re

import matplotlib.pyplot as plt
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import interpolate

from .. import param
from ..context import Context


def _find_isat_column(data, position: str) -> str | None:
    """Find the Isat column for a given position using pattern matching."""
    pattern1 = re.compile(rf"^Iis_{re.escape(position)}@\d+$")
    for column in data.columns:
        if pattern1.match(column):
            return column
    return None


def _get_isat_data(ctx: Context, deps: dict, position: str) -> np.ndarray:
    """Get Isat data for a specific position with pattern matching."""
    data = ctx.load_and_parse_raw_egdb("DivIis_tor_sum")
    time_list = np.array(data["Time"], dtype=float)

    column_name = _find_isat_column(data, position)
    
    isat_data = np.array(data[column_name], dtype=float)
    interpolator = interpolate.interp1d(time_list, isat_data, bounds_error=False, fill_value=0)
    return interpolator(deps["time"])

def _get_isat_smoothed(ctx: Context, Isat_list: np.ndarray) -> np.ndarray:
    """Get Isat data for a specific position with window smoothing."""
    # 設定から窓サイズを取得（デフォルトは5）
    window = ctx.cfg.get("processing", {}).get("isat_smoothing", {}).get("window", 21)
    window = max(3, window)
    if window % 2 == 0:
        window += 1
    
    half = window // 2
    Isat_list_window = sliding_window_view(np.pad(Isat_list, (half, half), mode="edge"), window)
    return np.quantile(Isat_list_window, 0.5, axis=1)


@param("Isat@4R", deps=["time"], needs=["DivIis_tor_sum"], doc="Isat@4R")
def get_Isat_4R(ctx: Context, deps):
    """Get Isat current at position 4R."""
    Isat_4R_list = _get_isat_data(ctx, deps, "4R")
    Isat_4R_list_smoothed = _get_isat_smoothed(ctx, Isat_4R_list)
    return Isat_4R_list_smoothed


@param("Isat@6L", deps=["time"], needs=["DivIis_tor_sum"], doc="Isat@6L")
def get_Isat_6L(ctx: Context, deps):
    """Get Isat current at position 6L."""
    Isat_6L_list = _get_isat_data(ctx, deps, "6L")
    Isat_6L_list_smoothed = _get_isat_smoothed(ctx, Isat_6L_list)
    return Isat_6L_list_smoothed


@param("Isat@7L", deps=["time"], needs=["DivIis_tor_sum"], doc="Isat@7L")
def get_Isat_7L(ctx: Context, deps):
    """Get Isat current at position 7L."""
    Isat_7L_list = _get_isat_data(ctx, deps, "7L")
    Isat_7L_list_smoothed = _get_isat_smoothed(ctx, Isat_7L_list)
    # plot_isat_comparison(Isat_7L_list, Isat_7L_list_smoothed, deps["time"])
    return Isat_7L_list_smoothed

def plot_isat_comparison(Isat_list: np.ndarray, Isat_list_smoothed: np.ndarray, time_list: np.ndarray):
    """Plot Isat comparison."""
    plt.plot(time_list, Isat_list, label="Raw")
    plt.plot(time_list, Isat_list_smoothed, label="Smoothed")
    plt.xlabel("Time (s)")
    plt.ylabel("Isat (A)")
    plt.title("Isat Comparison")
    plt.legend()
    plt.show()
    return