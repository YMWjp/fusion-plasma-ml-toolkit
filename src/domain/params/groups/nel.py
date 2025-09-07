from __future__ import annotations

import numpy as np
from scipy import interpolate

from .. import param
from ..context import Context


@param("nel", deps=["time"], needs=["fircall", "tsmap_nel"], doc="Thomson線平均電子密度")
def get_nel(ctx: Context, deps):
    eg_fircall = ctx.load_and_parse_raw_egdb("fircall")
    eg_tsmap_nel = ctx.load_and_parse_raw_egdb("tsmap_nel")
    time_list = deps["time"]

    # fircallの時間軸とデータ
    fir_time = np.array(eg_fircall["Time"], dtype=float)
    thomson_time = np.array(eg_tsmap_nel["Time"][1:], dtype=float)
    # 3849のデータが利用可能かチェック
    if "ne_bar(3849)" in eg_fircall and "nl_thomson_3849" in eg_tsmap_nel:
        fir_nel_bar = np.array(eg_fircall["ne_bar(3849)"], dtype=float)
        thomson_nel = np.array(eg_tsmap_nel["nl_thomson_3849"][1:], dtype=float)
    else:
        # 3849のデータが利用できない場合は3669を使用
        fir_nel_bar = np.array(eg_fircall["ne_bar(3669)"], dtype=float)
        thomson_nel = np.array(eg_tsmap_nel["nl_thomson_3669"][1:], dtype=float)
        print(f"Shot {ctx.shotNO}: Using ne_bar(3669) and nl_thomson_3669 for nel calculation")
    
    # time_listに合わせて補間
    f1_fir = interpolate.interp1d(fir_time, fir_nel_bar, bounds_error=False, fill_value=0)
    f1_thomson = interpolate.interp1d(thomson_time, thomson_nel, bounds_error=False, fill_value=0)
    
    # 補間されたデータ
    fir_nel_interp = f1_fir(time_list)
    thomson_nel_interp = f1_thomson(time_list)
    
    # FIR補正係数の計算（fir_nel_interpの最大値の前後5%を使用）
    fir_max = np.nanmax(fir_nel_interp)
    fir_threshold = fir_max * 0.05  # 最大値の5%
    factor_arg = np.logical_and(fir_nel_interp >= (fir_max - fir_threshold), 
                               fir_nel_interp <= (fir_max + fir_threshold))
    
    factor = np.nanmean(fir_nel_interp[factor_arg]) / np.nanmean(thomson_nel_interp[factor_arg])
    
    # FIR補正を適用
    nel = thomson_nel_interp * factor
    
    return nel

@param("nelgrad", deps=["time","nel"], needs=[], doc="dnL/dt")
def get_nelgrad(ctx: Context, deps):
    t = deps["time"]
    nl = deps["nel"]
    if t.size == 0 or nl.size == 0:
        return np.array([])
    return np.gradient(nl, t)