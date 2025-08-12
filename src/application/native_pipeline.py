from __future__ import annotations

import numpy as np
from tqdm import tqdm

from src.config.settings import get_basic_info_for_header, get_parameters, get_shot_numbers, load_config
from src.domain.labeling.detachment import apply_window_labels, label_by_derivative
from src.domain.params.calib import load_tsmap_calib_series
from src.domain.params.density import build_time_and_density
from src.domain.params.geometry import load_geometry
from src.domain.params.heating import load_heating_and_radiation
from src.domain.params.impurities import load_impurities_and_ha
from src.domain.params.isat import load_isat_series, repair_isat_7l_outliers
from src.domain.params.nbi import load_nbi
from src.domain.params.rmp import calculate_rmp_lid
from src.domain.params.sdl import align_sdl_to_times
from src.domain.params.wp_ip import load_wp_ip
from src.infrastructure.clients.lhd_api import ensure_eg_files

# moved detailed parsers usage into domain/params modules
from src.infrastructure.repositories.rmp_repo import load_rmp_flag_from_csv
from src.infrastructure.repositories.sdl_repo import load_sdl_file
from src.utils.paths import DATASETS_DIR, LOGS_DIR
from src.utils.utils import append_rows_to_csv, write_csv_header

cfg = load_config()
NE_LENGTH = float(cfg['processing']['constants']['ne_length'])
DT = float(cfg['processing']['sampling']['dt'])




def build_one_shot_rows(shot_no: int, headers: list[str], *,
                        detection_mode: str | None = None,
                        method: str | None = None) -> np.ndarray | None:
    # 取得対象 EG の一括確保
    ensure_eg_files(shot_no, [
        'tsmap_nel', 'wp', 'ip', 'echpw', 'bolo', 'imp02', 'ha1',
        'DivIis_tor_sum', 'nb1pwr_temporal', 'nb2pwr_temporal', 'nb3pwr_temporal',
        'nb4apwr_temporal', 'nb4bpwr_temporal', 'nb5apwr_temporal', 'nb5bpwr_temporal',
        'tsmap_calib'
    ])

    # 時間軸・密度
    time_list, nel = build_time_and_density(shot_no)
    nel_norm = nel / NE_LENGTH

    # 幾何・B
    geom = load_geometry(shot_no, time_list)
    B_arr = geom['B']

    # 加熱・放射
    heat = load_heating_and_radiation(shot_no, time_list)

    # NBI
    nbi = load_nbi(shot_no, time_list, nel_norm, geom['Bt_scalar'])

    # Pinput と比
    pinput = heat['Pech'] + nbi['Pnbi-tan'] + float(cfg['processing']['nbi']['perp_weight']) * nbi['Pnbi-perp']
    prad = heat['Prad']
    prad_ratio = np.divide(prad, pinput, out=np.zeros_like(prad), where=pinput != 0)

    # Wp, beta, Ip
    core = load_wp_ip(shot_no, time_list)

    # 不純物・H-alpha
    imp = load_impurities_and_ha(shot_no, time_list, nel)

    # Isat
    isat7 = load_isat_series(shot_no, '7L', None, time_list)
    isat7 = repair_isat_7l_outliers(isat7)
    isat4r = load_isat_series(shot_no, '4R', None, time_list)
    isat6l = load_isat_series(shot_no, '6L', None, time_list)

    # TsmapCalib (Te, reff, ne)
    calib = load_tsmap_calib_series(shot_no, time_list)
    reff100_ts = calib["reff@100eV"]
    ne100_ts = calib["ne@100eV"]
    dV100_ts = calib["dVdreff@100eV"]
    Te_center_ts = calib["Te@center"]
    ne_center_ts = calib["ne@center"]
    Te_edge_ts = calib["Te@edge"]

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


def _quick_visualize(shot_no: int, headers: list[str], rows: np.ndarray, viz_cfg: dict) -> None:
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = Path(viz_cfg.get('out_dir', './outputs/process/native')).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(viz_cfg.get('dpi', 120))

    # rows: shape (N, M) for one shot; columns correspond to headers
    data = {h: rows[:, i] for i, h in enumerate(headers)}
    t = data['times']

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    ax = axes.ravel()

    ax[0].plot(t, data.get('nel', np.zeros_like(t)), label='nel')
    ax[0].set_ylabel('nel')
    ax[0].legend()

    ax[1].plot(t, data.get('Prad', np.zeros_like(t)), label='Prad')
    ax[1].plot(t, data.get('Pinput', np.zeros_like(t)), label='Pinput')
    ax[1].set_ylabel('Power [MW]')
    ax[1].legend()

    ax[2].plot(t, data.get('Wp', np.zeros_like(t)), label='Wp (MJ)')
    ax[2].set_ylabel('Wp')
    ax[2].legend()

    ax[3].plot(t, data.get('D/(H+D)', np.zeros_like(t)), label='D/(H+D)')
    ax[3].set_ylabel('D/(H+D)')
    ax[3].legend()

    ax[4].plot(t, data.get('Isat@7L', np.zeros_like(t)), label='Isat@7L')
    ax[4].set_ylabel('Isat@7L')
    ax[4].legend()

    ax[5].plot(t, data.get('Prad/Pinput', np.zeros_like(t)), label='Prad/Pinput')
    ax[5].set_ylabel('Prad/Pinput')
    ax[5].legend()

    for a in ax:
        a.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('time [s]')
    axes[-1, 1].set_xlabel('time [s]')
    fig.suptitle(f'Shot {shot_no} quick look')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    fig.savefig(out_dir / f'shot_{shot_no}_quicklook.png', dpi=dpi)
    plt.close(fig)


def run_native_pipeline(*, detection_mode: str | None = None, method: str | None = None) -> None:
    headers = get_basic_info_for_header() + get_parameters()
    out_name = cfg['files']['output_dataset']
    out_path = DATASETS_DIR / out_name
    write_csv_header(out_path, headers, overwrite=True)

    shots = get_shot_numbers()
    pbar = tqdm(shots, desc="Processing shots", unit="shot")
    # エラーログ出力先
    error_log_path = LOGS_DIR / str(cfg['files'].get('error_log', 'errorshot.txt'))
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    for shot in pbar:        
        # just for a progress bar
        pbar.set_postfix_str(f"shot={int(shot)}")
        # 必要ファイルが存在しない/失敗しても個別スキップ
        try:
            rows = build_one_shot_rows(int(shot), headers, detection_mode=detection_mode, method=method)
            if rows is not None:
                append_rows_to_csv(out_path, rows)
                # 可視化（設定が有効な場合）
                viz_cfg = cfg['processing'].get('visualization', {})
                if viz_cfg.get('enabled', False):
                    try:
                        _quick_visualize(int(shot), headers, rows, viz_cfg)
                    except Exception as viz_e:
                        tqdm.write(f"[WARN] visualization failed for shot {int(shot)}: {viz_e}")
        except Exception as e:
            # スキップ理由を表示・記録
            tqdm.write(f"[WARN] shot {int(shot)} skipped: {e}")
            try:
                with error_log_path.open('a', encoding='utf-8') as f:
                    f.write(f"shot {int(shot)}: {e}\n")
            except Exception:
                pass
            continue


