from __future__ import annotations

import numpy as np

from .. import param
from ..context import Context


# 有効窓（例：nL(3669)>0.5 を満たした最初～最後）を切り出す共通ヘルパ
def _active_window(ctx: Context, *, thr=0.5, prefer_key="nL(3669)"):
    df = ctx.load_fircall()

    time = df.iloc[:, 0].to_numpy(dtype=float)
    if prefer_key in df.columns:
        nel = df[prefer_key].to_numpy(dtype=float)
    else:
        if df.shape[1] < 2:
            # 2列目が無いケースは空返し
            return time, np.array([], dtype=float), None, None
        nel = df.iloc[:, 1].to_numpy(dtype=float)
    mask = nel > thr
    idx = np.where(mask)[0]
    if idx.size < 3:
        return time, nel, None, None
    i0, i1 = int(np.nanmin(idx)), int(np.nanmax(idx))
    return time, nel, i0, i1

@param("time", deps=[], needs=["fircall"], doc="有効窓でトリムした時間[s]")
def get_time(ctx: Context, deps):
    t, _, i0, i1 = _active_window(ctx)
    return np.array([]) if i0 is None else t[i0:i1+1]

@param("type", deps=[], needs=["fircall"], doc="有効窓のタイプ")
def get_type(ctx: Context, deps):
    t, _, i0, i1 = _active_window(ctx)
    return np.array([]) if i0 is None else t[i0:i1+1]

@param("nel", deps=["time"], needs=["fircall"], doc="有効窓の nL(3669)")
def get_nel(ctx: Context, deps):
    _, nl, i0, i1 = _active_window(ctx)
    return np.array([]) if i0 is None else nl[i0:i1+1]

@param("B", deps=["time"], needs=["fircall"], doc="有効窓の B")
def get_B(ctx: Context, deps):
    t, _, i0, i1 = _active_window(ctx)
    return np.array([]) if i0 is None else t[i0:i1+1]

@param("nelgrad", deps=["time","nel"], needs=["fircall"], doc="dnL/dt")
def get_nelgrad(ctx: Context, deps):
    t = deps["time"]
    nl = deps["nel"]
    if t.size == 0 or nl.size == 0:
        return np.array([])
    return np.gradient(nl, t)

# 例：簡単な入力パワー（ダミー実装）
# ここでは Pech, Pnbi-tan, Pnbi-perp を "nel のスカラー変換" として作るだけ
# 実際のロジックに差し替える前提の“取っ掛かり”です。
@param("Pech", deps=["time"], needs=[], doc="ECH power (dummy)")
def get_Pech(ctx: Context, deps):
    t = deps["time"]
    return 0.8 * np.ones_like(t)  # 定数 [MW] のダミー

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
    return 0.3 * (nl - nl.min()) / (np.ptp(nl) + 1e-9)  # 0～0.3 [MW] に正規化

@param("Prad_over_Pinput", deps=["Prad","Pinput"], needs=[], doc="Prad/Pinput (dummy)")
def get_Prad_over_Pinput(ctx: Context, deps):
    prad = deps["Prad"]
    p = deps["Pinput"]
    out = np.zeros_like(prad)
    nz = p != 0
    out[nz] = prad[nz] / p[nz]
    return out

@param("beta", deps=["Pinput","B"], needs=[], doc="ダミーbeta")
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

