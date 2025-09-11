from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate, signal

from .. import param
from ..context import Context


@param("Prad",deps=["time"],needs=["bolo"],doc="放射パワー")
def get_Prad(ctx: Context, deps):
    bolo_data = ctx.load_and_parse_raw_egdb("bolo")
    bolo_time = np.array(bolo_data["Time"][1:], dtype=float)
    Rad_PW = np.array(bolo_data["Rad_PW"][1:], dtype=float)
    Rad_PW_smooth = signal.savgol_filter(Rad_PW, 101, 4)

    # get_Prad_comparison(bolo_data)
    
    f1_Rad_PW = interpolate.interp1d(bolo_time, Rad_PW_smooth, bounds_error=False, fill_value=0)
    time_values = deps["time"]
    return f1_Rad_PW(time_values)


def get_Prad_comparison(bolo_data):
    """複数の放射パワーデータを比較表示する関数"""
    bolo_time = np.array(bolo_data["Time"][1:], dtype=float)
    Rad_PW = np.array(bolo_data["Rad_PW"][1:], dtype=float)
    
    Rad_PW_smooth = signal.savgol_filter(Rad_PW, 101, 3)
    Rad_PW_smooth_2 = signal.savgol_filter(Rad_PW, 51, 3)  # より細かい平滑化
    Rad_PW_smooth_3 = signal.savgol_filter(Rad_PW, 201, 3)  # より粗い平滑化
    
    # 同じグラフに複数のデータを表示
    plt.figure(figsize=(12, 8))
    
    # サブプロット1: 時間系列比較
    plt.plot(bolo_time, Rad_PW, label='Raw Rad_PW', alpha=0.6, linewidth=1)
    plt.plot(bolo_time, Rad_PW_smooth_2, label='Smooth (51)', linewidth=2)
    plt.plot(bolo_time, Rad_PW_smooth, label='Smooth (101)', linewidth=2)
    plt.plot(bolo_time, Rad_PW_smooth_3, label='Smooth (201)', linewidth=2)
    plt.xlabel('Time (s)')
    plt.ylabel('Radiated Power (kW)')
    plt.title('Radiated Power: Different Smoothing Parameters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return