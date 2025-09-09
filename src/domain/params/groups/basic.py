from __future__ import annotations

import numpy as np
from scipy import interpolate

from .. import param
from ..context import Context


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

@param("Rax", deps=["time"], needs=["tsmap_nel"], doc="Rax")
def get_Rax(ctx: Context, deps):
    eg_tsmap_nel_comments = ctx.parse_tsmap_nel_comments("tsmap_nel")
    Rax = eg_tsmap_nel_comments["Rax"]
    # Raxを時間軸の数分だけ複製
    Rax_list = np.full(len(deps["time"]), Rax)
    return Rax_list

@param("Bq", deps=["time"], needs=["tsmap_nel"], doc="Bq")
def get_Bq(ctx: Context, deps):
    eg_tsmap_nel_comments = ctx.parse_tsmap_nel_comments("tsmap_nel")
    Bq = eg_tsmap_nel_comments["Bq"]
    # Bqを時間軸の数分だけ複製
    Bq_list = np.full(len(deps["time"]), Bq)
    return Bq_list

@param("Gamma", deps=["time"], needs=["tsmap_nel"], doc="Gamma")
def get_Gamma(ctx: Context, deps):
    eg_tsmap_nel_comments = ctx.parse_tsmap_nel_comments("tsmap_nel")
    Gamma = eg_tsmap_nel_comments["Gamma"]
    # Gammaを時間軸の数分だけ複製
    Gamma_list = np.full(len(deps["time"]), Gamma)
    return Gamma_list

@param("rax_vmec", deps=["time"], needs=["tsmap_nel"], doc="rax_vmec")
def get_rax_vmec(ctx: Context, deps):
    eg_tsmap_nel = ctx.load_and_parse_raw_egdb("tsmap_nel")
    time_list = np.array(eg_tsmap_nel["Time"][1:], dtype=float)
    rax_vmec_list = np.array(eg_tsmap_nel["Rax_vmec"][1:], dtype=float)
    f1_rax = interpolate.interp1d(time_list, rax_vmec_list, bounds_error=False, fill_value=0)
    return f1_rax(deps["time"])

@param("a99", deps=["time"], needs=["tsmap_nel"], doc="a99")
def get_a99(ctx: Context, deps):
    eg_tsmap_nel = ctx.load_and_parse_raw_egdb("tsmap_nel")
    time_list = np.array(eg_tsmap_nel["Time"][1:], dtype=float)
    a99_list = np.array(eg_tsmap_nel["a99"][1:], dtype=float)
    f1_a99 = interpolate.interp1d(time_list, a99_list, bounds_error=False, fill_value=0)
    return f1_a99(deps["time"])

@param("R99", deps=["time"], needs=["tsmap_nel"], doc="R99")
def get_R99(ctx: Context, deps):
    eg_tsmap_nel = ctx.load_and_parse_raw_egdb("tsmap_nel")
    time_list = np.array(eg_tsmap_nel["Time"][1:], dtype=float)
    R99_list = np.array(eg_tsmap_nel["R99"][1:], dtype=float)
    f1_R99 = interpolate.interp1d(time_list, R99_list, bounds_error=False, fill_value=0)
    return f1_R99(deps["time"])

@param("geom_center", deps=["time"], needs=["tsmap_nel"], doc="geom_center")
def get_geom_center(ctx: Context, deps):
    eg_tsmap_nel = ctx.load_and_parse_raw_egdb("tsmap_nel")
    time_list = np.array(eg_tsmap_nel["Time"][1:], dtype=float)
    geom_center_list = np.array(eg_tsmap_nel["geom_center"][1:], dtype=float)
    f1_geom_center = interpolate.interp1d(time_list, geom_center_list, bounds_error=False, fill_value=0)
    return f1_geom_center(deps["time"])

@param("Wp", deps=["time"], needs=["wp"], doc="Wp")
def get_Wp(ctx: Context, deps):
    eg_wp = ctx.load_and_parse_raw_egdb("wp")
    time_list = np.array(eg_wp["Time"], dtype=float)
    # Wpを[kJ]から[MJ]に変換
    Wp_list = np.array(eg_wp["Wp"]/1000, dtype=float)
    f1_Wp = interpolate.interp1d(time_list, Wp_list, bounds_error=False, fill_value=0)
    return f1_Wp(deps["time"])

@param("beta", deps=["time"], needs=["wp"], doc="beta")
def get_beta(ctx: Context, deps):
    eg_wp = ctx.load_and_parse_raw_egdb("wp")
    time_list = np.array(eg_wp["Time"], dtype=float)
    beta_list = np.array(eg_wp["<beta-dia>"], dtype=float)
    f1_beta = interpolate.interp1d(time_list, beta_list, bounds_error=False, fill_value=0)
    return f1_beta(deps["time"])

@param("Ip", deps=["time"], needs=["ip"], doc="Ip")
def get_Ip(ctx: Context, deps):
    eg_ip = ctx.load_and_parse_raw_egdb("ip")
    time_list = np.array(eg_ip["Time"], dtype=float)
    Ip_list = np.array(eg_ip["Ip"], dtype=float)
    f1_Ip = interpolate.interp1d(time_list, Ip_list, bounds_error=False, fill_value=0)
    return f1_Ip(deps["time"])


@param("Pech", deps=["time"], needs=["echpw"], doc="ECH power")
def get_Pech(ctx: Context, deps):
    eg_echpw = ctx.load_and_parse_raw_egdb("echpw")
    time_list = np.array(eg_echpw["Time"], dtype=float)
    Pech_list = np.array(eg_echpw["Total ECH"], dtype=float)
    f1_Pech = interpolate.interp1d(time_list, Pech_list, bounds_error=False, fill_value=0)
    return f1_Pech(deps["time"])

# 例：簡単な入力パワー（ダミー実装）
# ここでは Pech, Pnbi-tan, Pnbi-perp を "nel のスカラー変換" として作るだけ
# 実際のロジックに差し替える前提の"取っ掛かり"です。

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


@param("Echpw", deps=["time"], needs=["echpw"], doc="Echpw (dummy)")
def get_Echpw(ctx: Context, deps):
    t = np.asarray(deps["time"])
    return np.zeros_like(t, dtype=float)

@param("type", deps=["time"], needs=[], doc="type")
def get_type(ctx: Context, deps):
    t = np.asarray(deps["time"])
    return np.zeros_like(t, dtype=int)

