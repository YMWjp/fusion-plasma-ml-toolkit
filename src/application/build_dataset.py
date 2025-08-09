from __future__ import annotations
import numpy as np
from pathlib import Path
from src.infrastructure.repositories.rmp_repo import load_rmp_flag_from_csv
from src.domain.params.rmp import calculate_rmp_lid
from src.utils.utils import append_rows_to_csv, log_error_shot
from src.infrastructure.repositories.sdl_repo import load_sdl_file
from src.domain.params.sdl import align_sdl_to_times
from src.domain.dataset_builder import build_rows_by_header
from src.infrastructure.clients.lhd_api import ensure_eg_files
from src.infrastructure.parsers.egdb import Eg2D
from src.domain.params.isat import load_isat_series, repair_isat_7l_outliers
from src.domain.labeling.detachment import (
    label_by_derivative,
    apply_window_labels,
)

def process_one_shot(*, shot_no: int, bt: float, time_list: np.ndarray,
                     mapping: dict[str, np.ndarray], header: list[str],
                     out_path: Path) -> bool:
    """
    mapping は pr7_25 の output_dict 相当（最低限 'shotNO','times','types' など先に入れておく）
    True=成功, False=スキップ
    """
    try:
        # 事前に必要な EG ファイルを取得
        ensure_eg_files(shot_no, [
            'wp', 'ip', 'echpw', 'bolo', 'imp02', 'ha1', 'ha2', 'ha3',
            'DivIis_tor_sum'
        ])

        # Wp, Ip, ECH, Prad などを補間取得
        eg_wp = Eg2D(f"wp@{shot_no}.dat")
        mapping['Wp'] = eg_wp.interpolate_series('Wp', time_list) / 1000.0
        mapping['beta'] = eg_wp.interpolate_series('<beta-dia>', time_list)

        eg_ip = Eg2D(f"ip@{shot_no}.dat")
        mapping['Ip'] = eg_ip.interpolate_series('Ip', time_list)

        eg_ech = Eg2D(f"echpw@{shot_no}.dat")
        mapping['Pech'] = eg_ech.interpolate_series('Total ECH', time_list)

        eg_bolo = Eg2D(f"bolo@{shot_no}.dat")
        prad_list = eg_bolo.interpolate_series('Rad_PW', time_list) / 1000.0
        mapping['Prad'] = prad_list

        # Pinput 推定（既存簡易式）
        # nbi は元コードの詳細補正を domain 化予定だが、まずは through 値合算に近い形に留める
        # 簡易: 呼び出し元で 'Pnbi-tan','Pnbi-perp' があればそれを利用
        if 'Pnbi-tan' in mapping and 'Pnbi-perp' in mapping:
            mapping['Pinput'] = mapping['Pech'] + mapping['Pnbi-tan'] + 0.5 * mapping['Pnbi-perp']
        else:
            mapping['Pinput'] = mapping['Pech']

        # Isat@7L の取得と整形（gdn index は暫定 20）
        isat7 = load_isat_series(shot_no, '7L', 20, time_list)
        mapping['Isat@7L'] = repair_isat_7l_outliers(isat7)

        # RMP
        flag = load_rmp_flag_from_csv(shot_no)
        mapping['RMP_LID'] = np.full(len(time_list), calculate_rmp_lid(bt, flag))

        # SDL
        sdl = load_sdl_file(shot_no)
        dphi, dphi_ext, dtheta = align_sdl_to_times(sdl, time_list)
        mapping['SDLloop_dPhi'] = dphi
        mapping['SDLloop_dPhi_ext'] = dphi_ext
        mapping['SDLloop_dTheta'] = dtheta

        # ラベリング（自動）: derivative を既定値で
        idx = label_by_derivative(mapping['Isat@7L'], sigma=2.0, threshold_percentile=90)
        if idx is not None:
            mapping['types'] = apply_window_labels(
                len(time_list), idx,
                pre_range=15, transition_range=5, post_range=15,
                pre_label=-1, transition_label=0, post_label=1,
            )
        else:
            mapping['types'] = np.zeros_like(time_list, dtype=int)

        rows = build_rows_by_header(header, mapping)
        append_rows_to_csv(out_path, rows)
        return True
    except Exception:
        log_error_shot(shot_no)
        return False

def attach_sdl_fields(*, shot_no: int, time_list, mapping: dict[str, np.ndarray]) -> None:
    sdl = load_sdl_file(shot_no)
    dphi, dphi_ext, dtheta = align_sdl_to_times(sdl, time_list)
    mapping['SDLloop_dPhi'] = dphi
    mapping['SDLloop_dPhi_ext'] = dphi_ext
    mapping['SDLloop_dTheta'] = dtheta