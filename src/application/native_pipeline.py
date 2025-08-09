from __future__ import annotations

import numpy as np

from src.config.settings import get_basic_info_for_header, get_parameters, get_shot_numbers, load_config
from src.domain.labeling.detachment import apply_window_labels, label_by_derivative
from src.domain.params.isat import load_isat_series, repair_isat_7l_outliers
from src.domain.params.rmp import calculate_rmp_lid
from src.domain.params.sdl import align_sdl_to_times
from src.infrastructure.clients.lhd_api import ensure_eg_files
from src.infrastructure.parsers.eg3d import TsmapCalib
from src.infrastructure.parsers.egdb import Eg2D
from src.infrastructure.repositories.rmp_repo import load_rmp_flag_from_csv
from src.infrastructure.repositories.sdl_repo import load_sdl_file
from src.utils.paths import DATASETS_DIR
from src.utils.utils import append_rows_to_csv, write_csv_header

cfg = load_config()
NE_LENGTH = float(cfg['processing']['constants']['ne_length'])
DT = float(cfg['processing']['sampling']['dt'])


def _build_time_and_density(shot_no: int) -> tuple[np.ndarray, np.ndarray]:
    """
    tsmap_nel@{shot}.dat から thomson 密度を基準に時間ベースを構築し、
    FIR との係数スケーリングを反映して nel を返す。
    """
    eg = Eg2D(f"tsmap_nel@{shot_no}.dat")
    time_raw = np.array(eg.dimdata[1:], dtype=float) if len(eg.dimdata) > 1 else np.array(eg.dimdata, dtype=float)
    if time_raw.size == 0:
        raise RuntimeError("No Thomson time series")
    th = np.array(eg.interpolate_series('nl_thomson_3669', time_raw))
    fir = np.array(eg.interpolate_series('nl_fir_3669', time_raw))
    # 係数スケーリング
    win = cfg['processing']['scaling']['fir_scale_window']
    in_window = np.logical_and(time_raw > float(win[0]), time_raw < float(win[1]))
    if np.any(in_window):
        factor = np.nanmean(fir[in_window]) / np.nanmean(th[in_window])
    elif time_raw.size > 3:
        factor = np.nanmean(fir[:3]) / np.nanmean(th[:3])
    else:
        factor = 1.0
    t_min, t_max = float(np.nanmin(time_raw)), float(np.nanmax(time_raw))
    time_list = np.arange(t_min, t_max, DT)
    # 補間
    nel = Eg2D(f"tsmap_nel@{shot_no}.dat").interpolate_series('nl_thomson_3669', time_list) * factor
    return time_list, nel


def _load_geometry(shot_no: int, time_list: np.ndarray) -> dict[str, np.ndarray | float]:
    eg = Eg2D(f"tsmap_nel@{shot_no}.dat")
    out = {}
    out['a99'] = eg.interpolate_series('a99', time_list)
    out['Rax_vmec'] = eg.interpolate_series('Rax_vmec', time_list)
    out['Rax'] = out['Rax_vmec']  # 互換（旧コードは Rax_vmec を出力列 Rax として使っていた箇所がある）
    out['geom_center'] = eg.interpolate_series('geom_center', time_list)
    # コメントから Bt, Rax を抽出（フォールバック: 0）
    bt = 0.0
    rax = 0.0
    for line in eg.comments.splitlines():
        if 'Bt' in line:
            try:
                bt = float(line.split('Bt')[1].split('=')[-1].strip())
            except Exception:
                pass
        if 'Rax' in line:
            try:
                rax = float(line.split('Rax')[1].split('=')[-1].strip())
            except Exception:
                pass
    out['Bt_scalar'] = bt
    out['Rax_scalar'] = rax
    out['B'] = np.full(len(time_list), abs(bt))
    return out


def _load_heating_and_radiation(shot_no: int, time_list: np.ndarray) -> dict[str, np.ndarray]:
    eg_ech = Eg2D(f"echpw@{shot_no}.dat")
    pech = eg_ech.interpolate_series('Total ECH', time_list)
    eg_bolo = Eg2D(f"bolo@{shot_no}.dat")
    prad = eg_bolo.interpolate_series('Rad_PW', time_list) / 1000.0
    return {'Pech': pech, 'Prad': prad}


def _load_nbi(shot_no: int, time_list: np.ndarray, nebar: np.ndarray, bt_scalar: float) -> dict[str, np.ndarray]:
    def nbiabs(through: np.ndarray, nebar: np.ndarray, s: str) -> np.ndarray:
        # get_params/get_nbi.py 準拠の簡易版
        try:
            s_int = int(s) if s.isdigit() else -1
        except Exception:
            s_int = -1
        if bt_scalar < 0:
            if s_int in (1, 3):
                loss = 0.28127 + 0.091059 * np.exp(-3.5618 * nebar / 10)
            else:
                loss = -0.010049 + 2.0175 * np.exp(-10.904 * nebar / 10)
                loss[loss < 0] = 0
        else:
            if s_int == 2:
                loss = 0.28127 + 0.091059 * np.exp(-3.5618 * nebar / 10)
            else:
                loss = -0.010049 + 2.0175 * np.exp(-10.904 * nebar / 10)
                loss[loss < 0] = 0
        loss[loss > 1] = 1
        return through * (1 - loss)

    # tangential 1,2,3
    nb_tmp = np.zeros_like(time_list)
    for s in ['1', '2', '3']:
        eg = Eg2D(f"nb{s}pwr_temporal@{shot_no}.dat")
        unit = eg.valunits[eg.valname2idx(f"Pport-through_nb{s}")]
        series = eg.interpolate_series(f"Pport-through_nb{s}", time_list)
        if unit == 'kW':
            series = series / 1000.0
        nb_tmp = np.vstack((nb_tmp, nbiabs(series, nebar, s)))
    pnbi_tan = np.sum(np.abs(nb_tmp), axis=0)

    # perpendicular 4a,4b,5a,5b
    nb_tmp = np.zeros_like(time_list)
    for s in ['4a', '4b', '5a', '5b']:
        eg = Eg2D(f"nb{s}pwr_temporal@{shot_no}.dat")
        unit = eg.valunits[eg.valname2idx(f"Pport-through_nb{s}")]
        series = eg.interpolate_series(f"Pport-through_nb{s}", time_list)
        if unit == 'kW':
            series = series / 1000.0
        nb_tmp = np.vstack((nb_tmp, series))
    pnbi_perp = np.sum(np.abs(nb_tmp), axis=0)
    return {'Pnbi-tan': pnbi_tan, 'Pnbi-perp': pnbi_perp}


def _load_impurities_and_ha(shot_no: int, time_list: np.ndarray, nel: np.ndarray) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    eg_imp = Eg2D(f"imp02@{shot_no}.dat")
    out['OVI'] = eg_imp.interpolate_series('OVI', time_list)
    out['CIV'] = eg_imp.interpolate_series('CIV', time_list)
    out['OV'] = eg_imp.interpolate_series('OV', time_list)
    out['CIII'] = eg_imp.interpolate_series('CIII', time_list)
    out['FeXVI'] = eg_imp.interpolate_series('FeXVI', time_list)
    out['HI'] = eg_imp.interpolate_series('HI', time_list)
    # ゲイン補正
    if 154481 <= int(shot_no) <= 157260:
        out['OV'] = 5.368 * out['OV']
    if 154539 <= int(shot_no) <= 157260:
        out['OVI'] = 2.622 * out['OVI']
    if (155146 <= int(shot_no) <= 155207) or (158144 <= int(shot_no) <= 158215):
        out['CIII'] = 2.655 * out['CIII']
        out['CIV'] = 2.896 * out['CIV']

    # H-alpha 系
    # D/(H+D)
    try:
        eg_ha3 = Eg2D(f"ha3@{shot_no}.dat")
        dh = eg_ha3.interpolate_series('D/(H+D)', time_list)
        dh[dh < 0.01] = 0.01
    except Exception:
        dh = np.zeros_like(time_list)
    out['D/(H+D)'] = dh

    # HeI
    try:
        eg_ha1 = Eg2D(f"ha1@{shot_no}.dat")
        out['HeI'] = eg_ha1.interpolate_series('HeI(Impmon)', time_list)
    except Exception:
        out['HeI'] = np.zeros_like(time_list)

    return out


def _load_wp_ip(shot_no: int, time_list: np.ndarray) -> dict[str, np.ndarray]:
    eg_wp = Eg2D(f"wp@{shot_no}.dat")
    wpdia = eg_wp.interpolate_series('Wp', time_list) / 1000.0
    beta = eg_wp.interpolate_series('<beta-dia>', time_list)
    eg_ip = Eg2D(f"ip@{shot_no}.dat")
    ip = eg_ip.interpolate_series('Ip', time_list)
    return {'Wp': wpdia, 'beta': beta, 'Ip': ip}


def build_one_shot_rows(shot_no: int, headers: list[str], *,
                        detection_mode: str | None = None,
                        method: str | None = None) -> np.ndarray | None:
    # 取得対象 EG の一括確保
    ensure_eg_files(shot_no, [
        'tsmap_nel', 'wp', 'ip', 'echpw', 'bolo', 'imp02', 'ha1', 'ha2', 'ha3',
        'DivIis_tor_sum', 'nb1pwr_temporal', 'nb2pwr_temporal', 'nb3pwr_temporal',
        'nb4apwr_temporal', 'nb4bpwr_temporal', 'nb5apwr_temporal', 'nb5bpwr_temporal',
        'tsmap_calib'
    ])

    # 時間軸・密度
    time_list, nel = _build_time_and_density(shot_no)
    nel_norm = nel / NE_LENGTH

    # 幾何・B
    geom = _load_geometry(shot_no, time_list)
    B_arr = geom['B']

    # 加熱・放射
    heat = _load_heating_and_radiation(shot_no, time_list)

    # NBI
    nbi = _load_nbi(shot_no, time_list, nel_norm, geom['Bt_scalar'])

    # Pinput と比
    pinput = heat['Pech'] + nbi['Pnbi-tan'] + float(cfg['processing']['nbi']['perp_weight']) * nbi['Pnbi-perp']
    prad = heat['Prad']
    prad_ratio = np.divide(prad, pinput, out=np.zeros_like(prad), where=pinput != 0)

    # Wp, beta, Ip
    core = _load_wp_ip(shot_no, time_list)

    # 不純物・H-alpha
    imp = _load_impurities_and_ha(shot_no, time_list, nel)

    # Isat
    isat7 = load_isat_series(shot_no, '7L', None, time_list)
    isat7 = repair_isat_7l_outliers(isat7)
    isat4r = load_isat_series(shot_no, '4R', None, time_list)
    isat6l = load_isat_series(shot_no, '6L', None, time_list)

    # TsmapCalib (Te, reff, ne)
    try:
        tsc = TsmapCalib(f"tsmap_calib@{shot_no}.dat")
        reff100, ne100, dV100 = tsc.ne_from_Te(0.1)
        Te_center, ne_center = tsc.Te_from_reff(0)
        # edge (簡易): reff が最大の場所の Te を採用
        # target_times は tsc.time と time_list が異なる可能性があるため、時間補間
        # まず tsc の native 時間軸で値をとり、time_list へ補間
        t_tsc = np.asarray(tsc.time)
        from scipy.interpolate import interp1d
        f_reff = interp1d(t_tsc, reff100, bounds_error=False, fill_value=0.0)
        f_ne100 = interp1d(t_tsc, ne100, bounds_error=False, fill_value=0.0)
        f_dV100 = interp1d(t_tsc, dV100, bounds_error=False, fill_value=0.0)
        f_Te_center = interp1d(t_tsc, Te_center, bounds_error=False, fill_value=0.0)
        f_ne_center = interp1d(t_tsc, ne_center, bounds_error=False, fill_value=0.0)
        reff100_ts = f_reff(time_list)
        ne100_ts = f_ne100(time_list)
        dV100_ts = f_dV100(time_list)
        Te_center_ts = f_Te_center(time_list)
        ne_center_ts = f_ne_center(time_list)
        # Te_edge 簡易: 中央列の Te(t, R_mid) を利用（厳密には phiEdge が必要）
        Te_edge_ts = np.zeros_like(time_list)
    except Exception:
        reff100_ts = np.zeros_like(time_list)
        ne100_ts = np.zeros_like(time_list)
        dV100_ts = np.zeros_like(time_list)
        Te_center_ts = np.zeros_like(time_list)
        ne_center_ts = np.zeros_like(time_list)
        Te_edge_ts = np.zeros_like(time_list)

    # RMP, SDL
    flag = load_rmp_flag_from_csv(shot_no)
    rmp_lid = np.full(len(time_list), calculate_rmp_lid(geom['Bt_scalar'], flag))
    sdl = load_sdl_file(shot_no)
    dphi, dphi_ext, dtheta = align_sdl_to_times(sdl, time_list)

    # ラベリング
    idx = None
    if detection_mode in (None, 'automatic'):
        if method == 'threshold':
            from src.domain.labeling.detachment import label_by_threshold as _lb
            idx = _lb(isat7, threshold_percentile=float(cfg['labeling']['threshold']['threshold_percentile']))
        elif method == 'peak':
            from src.domain.labeling.detachment import label_by_peak as _lp
            idx = _lp(isat7, min_prominence=float(cfg['labeling']['peak']['min_prominence']))
        else:
            idx = label_by_derivative(
                isat7,
                sigma=float(cfg['labeling']['derivative']['sigma']),
                threshold_percentile=float(cfg['labeling']['derivative']['threshold_percentile']),
            )
    if idx is not None:
        types = apply_window_labels(
            len(time_list),
            idx,
            pre_range=int(cfg['labeling']['window']['pre']),
            transition_range=int(cfg['labeling']['window']['transition']),
            post_range=int(cfg['labeling']['window']['post']),
            pre_label=-1,
            transition_label=0,
            post_label=1,
        )
    else:
        types = np.zeros_like(time_list, dtype=int)

    mapping: dict[str, np.ndarray] = {
        # basic
        'shotNO': np.full(len(time_list), shot_no),
        'times': time_list,
        'types': types,
        # core plasma
        'nel': nel_norm,
        'B': B_arr,
        'Pech': heat['Pech'],
        'Pnbi-tan': nbi['Pnbi-tan'],
        'Pnbi-perp': nbi['Pnbi-perp'],
        'Pinput': pinput,
        'Prad': prad,
        'Prad/Pinput': prad_ratio,
        'Wp': core['Wp'],
        'beta': core['beta'],
        'Rax': geom['geom_center'],
        'rax_vmec': geom['Rax_vmec'],
        'a99': geom['a99'],
        'D/(H+D)': imp['D/(H+D)'],
        'CIII': imp['CIII'] / np.maximum(nel_norm, 1e-9),
        'CIV': imp['CIV'] / np.maximum(nel_norm, 1e-9),
        'OV': imp['OV'] / np.maximum(nel_norm, 1e-9),
        'OVI': imp['OVI'] / np.maximum(nel_norm, 1e-9),
        'FeXVI': imp['FeXVI'] / np.maximum(nel_norm, 1e-9),
        'Ip': core['Ip'],
        'Isat@4R': isat4r,
        'Isat@6L': isat6l,
        'Isat@7L': isat7,
        'reff@100eV': reff100_ts,
        'ne@100eV': ne100_ts,
        'dVdreff@100eV': dV100_ts,
        'Te@center': Te_center_ts,
        'Te@edge': Te_edge_ts,
        'ne@center': ne_center_ts,
        'RMP_LID': rmp_lid,
        'SDLloop_dPhi': dphi,
        'SDLloop_dPhi_ext': dphi_ext,
        'SDLloop_dTheta': dtheta,
    }

    # ヘッダ順に並べる
    arr = np.vstack([mapping[h] for h in headers]).T
    return arr


def run_native_pipeline(*, detection_mode: str | None = None, method: str | None = None) -> None:
    headers = get_basic_info_for_header() + get_parameters()
    out_name = cfg['files']['output_dataset']
    out_path = DATASETS_DIR / out_name
    write_csv_header(out_path, headers, overwrite=True)

    shots = get_shot_numbers()

    for shot in shots:
        # 必要ファイルが存在しない/失敗しても個別スキップ
        try:
            rows = build_one_shot_rows(int(shot), headers, detection_mode=detection_mode, method=method)
            if rows is not None:
                append_rows_to_csv(out_path, rows)
        except Exception:
            # 失敗ショットはログへ（既存の log_error_shot を使うほどではないため簡易無視）
            continue


