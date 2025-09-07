from __future__ import annotations

import numpy as np
from scipy import interpolate

from .. import param
from ..context import Context


def _get_impurity_parameter(ctx: Context, deps: dict, param_name: str, 
                          primary_source: str, fallback_source: str | None = None) -> np.ndarray:
    """
    不純物パラメータを取得する共通関数
    
    Args:
        ctx: Context
        deps: 依存関係（timeを含む）
        param_name: パラメータ名（例: "OVI", "CIV"）
        primary_source: 主要データソース（例: "imp02"）
        fallback_source: フォールバックデータソース（例: "imp01"）
        
    Returns:
        補間されたパラメータ値の配列
    """
    # 主要データソースを試す
    eg_primary = ctx.load_and_parse_raw_egdb(primary_source)
    if param_name in eg_primary.columns:
        data_list = np.array(eg_primary[param_name], dtype=float)
        time_list = np.array(eg_primary["Time"], dtype=float)
        f1 = interpolate.interp1d(time_list, data_list, bounds_error=False, fill_value=np.nan)
        result = f1(deps["time"])
        
        # 正規化係数を適用
        norm_factors = ctx.parse_norm_factors(primary_source)
        norm_factor = norm_factors.get(param_name, 1.0)
        return result * norm_factor
    
    # フォールバックデータソースを試す
    if fallback_source:
        eg_fallback = ctx.load_and_parse_raw_egdb(fallback_source)
        if param_name in eg_fallback.columns:
            data_list = np.array(eg_fallback[param_name], dtype=float)
            time_list = np.array(eg_fallback["Time"], dtype=float)
            f1 = interpolate.interp1d(time_list, data_list, bounds_error=False, fill_value=np.nan)
            result = f1(deps["time"])
            
            # 正規化係数を適用
            norm_factors = ctx.parse_norm_factors(fallback_source)
            norm_factor = norm_factors.get(param_name, 1.0)
            return result * norm_factor
    
    # データが存在しない場合はnp.nanで埋める
    return np.full_like(deps["time"], np.nan)


@param("CIII", deps=["time"], needs=["imp01", "imp02"], doc="CIII")
def get_CIII(ctx: Context, deps):
    return _get_impurity_parameter(ctx, deps, "CIII", "imp02", "imp01")

@param("OV", deps=["time"], needs=["imp01", "imp02"], doc="OV")
def get_OV(ctx: Context, deps):
    return _get_impurity_parameter(ctx, deps, "OV", "imp02", "imp01")

@param("OVI", deps=["time"], needs=["imp02"], doc="OVI")
def get_OVI(ctx: Context, deps):
    return _get_impurity_parameter(ctx, deps, "OVI", "imp02")

@param("CIV", deps=["time"], needs=["imp02"], doc="CIV")
def get_CIV(ctx: Context, deps):
    return _get_impurity_parameter(ctx, deps, "CIV", "imp02")

@param("HI", deps=["time"], needs=["imp02"], doc="HI")
def get_HI(ctx: Context, deps):
    return _get_impurity_parameter(ctx, deps, "HI", "imp02")

@param("FeXVI", deps=["time"], needs=["imp02"], doc="FeXVI")
def get_FeXVI(ctx: Context, deps):
    return _get_impurity_parameter(ctx, deps, "FeXVI", "imp02")