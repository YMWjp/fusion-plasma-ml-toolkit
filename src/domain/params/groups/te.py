# from __future__ import annotations

# import numpy as np
# from scipy import interpolate

# from .. import param
# from ..context import Context


# def _interp_column(data, time_values, name_candidates) -> np.ndarray | None:
#     """Pick first existing column in name_candidates and interpolate to time_values."""
#     for name in name_candidates:
#         if name in data.columns:
#             src_t = np.asarray(data["Time"], dtype=float)
#             src_v = np.asarray(data[name], dtype=float)
#             f = interpolate.interp1d(src_t, src_v, bounds_error=False, fill_value=0.0)
#             return f(time_values)
#     return None  # returns None or a DataFrame-like


# @param("reff100eV", deps=["time"], needs=["tsmap_calib"], doc="Effective radius at 100eV")
# def get_reff100eV(ctx: Context, deps):
#     t = deps["time"]
#     df = ctx.load_and_parse_raw_egdb("tsmap_calib")
#     if df is None:
#         return np.full_like(t, np.nan, dtype=float)
#     y = _interp_column(df, t, ["reff100eV", "reff@100eV", "reff_100eV"])
#     return y if y is not None else np.full_like(t, np.nan, dtype=float)


# @param("ne100eV", deps=["time"], needs=["tsmap_calib"], doc="Electron density at 100eV")
# def get_ne100eV(ctx: Context, deps):
#     t = deps["time"]
#     df = ctx.load_and_parse_raw_egdb("tsmap_calib")
#     if df is None:
#         return np.full_like(t, np.nan, dtype=float)
#     y = _interp_column(df, t, ["ne100eV", "ne@100eV", "ne_100eV"])
#     return y if y is not None else np.full_like(t, np.nan, dtype=float)


# @param("dV100eV", deps=["time"], needs=["tsmap_calib"], doc="Volume derivative at 100eV")
# def get_dV100eV(ctx: Context, deps):
#     t = deps["time"]
#     df = ctx.load_and_parse_raw_egdb("tsmap_calib")
#     if df is None:
#         return np.full_like(t, np.nan, dtype=float)
#     # dV/dreff at 100 eV
#     y = _interp_column(df, t, ["dVdreff100eV", "dVdreff@100eV", "dV100eV", "dV_100eV"])
#     return y if y is not None else np.full_like(t, np.nan, dtype=float)


# # ---- 中心値（TsmapCalib.Te_from_reff(0) 相当）----

# @param("Te_center", deps=["time"], needs=["tsmap_calib"], doc="Central electron temperature (reff=0)")
# def get_Te_center(ctx: Context, deps):
#     t = deps["time"]
#     df = ctx.load_and_parse_raw_egdb("tsmap_calib")
#     if df is None:
#         return np.full_like(t, np.nan, dtype=float)
#     y = _interp_column(df, t, ["Te_center", "Te@center", "Te_center_keV", "Te0"])
#     return y if y is not None else np.full_like(t, np.nan, dtype=float)


# @param("ne_center", deps=["time"], needs=["tsmap_calib"], doc="Central electron density (reff=0)")
# def get_ne_center(ctx: Context, deps):
#     t = deps["time"]
#     df = ctx.load_and_parse_raw_egdb("tsmap_calib")
#     if df is None:
#         return np.full_like(t, np.nan, dtype=float)
#     y = _interp_column(df, t, ["ne_center", "ne@center", "ne0"])
#     return y if y is not None else np.full_like(t, np.nan, dtype=float)


# # ---- 端値（TsmapCalib.phiEdge() 相当）----
# # 元コードでは Te_edge = (Te_edge_inner + Te_outer)/2

# @param("ne_edge", deps=["time"], needs=["tsmap_calib"], doc="Edge electron density")
# def get_ne_edge(ctx: Context, deps):
#     t = deps["time"]
#     df = ctx.load_and_parse_raw_egdb("tsmap_calib")
#     if df is None:
#         return np.full_like(t, np.nan, dtype=float)
#     y = _interp_column(df, t, ["ne_edge", "ne@edge"])
#     return y if y is not None else np.full_like(t, np.nan, dtype=float)


# @param("Te_edge", deps=["time"], needs=["tsmap_calib"], doc="Edge electron temperature (mean of inner/outer if available)")
# def get_Te_edge(ctx: Context, deps):
#     t = deps["time"]
#     df = ctx.load_and_parse_raw_egdb("tsmap_calib")
#     if df is None:
#         return np.full_like(t, np.nan, dtype=float)

#     # try inner/outer then average (sameロジック as: (Te_edge_inner + Te_outer)/2)
#     te_in = _interp_column(df, t, ["Te_edge_inner", "Te@edge_inner", "Te_edge_in"])
#     te_out = _interp_column(df, t, ["Te_outer", "Te@outer", "Te_edge_outer", "Te_out"])

#     if te_in is not None and te_out is not None:
#         return 0.5 * (te_in + te_out)

#     # fallback: single edge column
#     te_edge = _interp_column(df, t, ["Te_edge", "Te@edge"])
#     if te_edge is not None:
#         return te_edge

#     return np.full_like(t, np.nan, dtype=float)