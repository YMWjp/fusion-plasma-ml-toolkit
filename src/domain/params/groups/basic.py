from __future__ import annotations

import numpy as np
from scipy import interpolate

from .. import param
from ..context import Context


# 完了
@param("time", deps=[], needs=["tsmap_nel"], doc="Thomson時間軸[s]")
def get_time(ctx: Context, deps):
    cfg = ctx.cfg
    eg = ctx.load_and_parse_raw_egdb("tsmap_nel")
    dt = float(cfg['processing']['sampling']['dt'])
    thomson_time_list = np.array(eg["Time"][1:], dtype=float)
    time_list = np.arange(np.nanmin(thomson_time_list),
                          np.nanmax(thomson_time_list),
                          dt)
    return time_list

@param("B", deps=["time"], needs=["tsmap_nel"], doc="磁場")
def get_B(ctx: Context, deps):
    eg_tsmap_nel_comments = ctx.parse_tsmap_nel_comments("tsmap_nel")
    B = eg_tsmap_nel_comments["Bt"]
    # Bを時間軸の数分だけ複製
    B_list = np.full(len(deps["time"]), B)
    return B_list

# 例：簡単な入力パワー（ダミー実装）
# ここでは Pech, Pnbi-tan, Pnbi-perp を "nel のスカラー変換" として作るだけ
# 実際のロジックに差し替える前提の"取っ掛かり"です。
@param("Pech", deps=["time"], needs=[], doc="ECH power (dummy)")
def get_Pech(ctx: Context, deps):
    t = deps["time"]
    return 0.8 * np.ones_like(t)

@param("Pnbi-tan", deps=["time"], needs=[], doc="NBI tangential (dummy)")
def get_Pnbi_tan(ctx: Context, deps):
    t = deps["time"]
    return 1.2 * np.ones_like(t)

@param("Pnbi-perp", deps=["time"], needs=[], doc="NBI perpendicular (dummy)")
def get_Pnbi_perp(ctx: Context, deps):
    t = deps["time"]
    return 0.5 * np.ones_like(t)

@param("Pinput", deps=["Pech","Pnbi-tan","Pnbi-perp"], needs=[], doc="総入力パワー (dummy)")
def get_Pinput(ctx: Context, deps):
    return deps["Pech"] + deps["Pnbi-tan"] + deps["Pnbi-perp"]

@param("Prad", deps=["time","nel"], needs=[], doc="放射パワー (dummy)")
def get_Prad(ctx: Context, deps):
    # 単に nel に比例させたダミー
    nl = deps["nel"]
    return 0.3 * (nl - nl.min()) / (nl.ptp() + 1e-9)  # 0～0.3 [MW] に正規化

@param("Prad_over_Pinput", deps=["Prad","Pinput"], needs=[], doc="Prad/Pinput (dummy)")
def get_Prad_over_Pinput(ctx: Context, deps):
    prad = deps["Prad"]
    p = deps["Pinput"]
    out = np.zeros_like(prad)
    nz = p != 0
    out[nz] = prad[nz] / p[nz]
    return out

@param("beta", deps=["Pinput"], needs=[], doc="ダミーbeta")
def get_beta(ctx, deps):
    # なんでもOK。まずはスカラー/配列を返せば動きます
    return np.zeros_like(deps["Pinput"])

@param("Wp", deps=["time"], needs=["wp"], doc="Wp (dummy)")
def get_Wp(ctx: Context, deps):
    t = np.asarray(deps["time"])
    return np.zeros_like(t, dtype=float)

@param("Ip", deps=["time"], needs=["ip"], doc="Ip (dummy)")
def get_Ip(ctx: Context, deps):
    t = np.asarray(deps["time"])
    return np.zeros_like(t, dtype=float)

@param("Echpw", deps=["time"], needs=["echpw"], doc="Echpw (dummy)")
def get_Echpw(ctx: Context, deps):
    t = np.asarray(deps["time"])
    return np.zeros_like(t, dtype=float)

@param("type", deps=["time"], needs=[], doc="type")
def get_type(ctx: Context, deps):
    t = np.asarray(deps["time"])
    return np.zeros_like(t, dtype=int)

