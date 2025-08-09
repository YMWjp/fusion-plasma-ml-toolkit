from __future__ import annotations
import numpy as np
from pathlib import Path
from src.infrastructure.repositories.rmp_repo import load_rmp_flag_from_csv
from src.domain.params.rmp import calculate_rmp_lid
from src.utils.utils import append_rows_to_csv, log_error_shot
from src.infrastructure.repositories.sdl_repo import load_sdl_file
from src.domain.params.sdl import align_sdl_to_times
from src.domain.dataset_builder import build_rows_by_header

def process_one_shot(*, shot_no: int, bt: float, time_list: np.ndarray,
                     mapping: dict[str, np.ndarray], header: list[str],
                     out_path: Path) -> bool:
    """
    mapping は pr7_25 の output_dict 相当（最低限 'shotNO','times','types' など先に入れておく）
    True=成功, False=スキップ
    """
    try:
        # RMP
        flag = load_rmp_flag_from_csv(shot_no)
        mapping['RMP_LID'] = np.full(len(time_list), calculate_rmp_lid(bt, flag))

        # SDL
        sdl = load_sdl_file(shot_no)
        dphi, dphi_ext, dtheta = align_sdl_to_times(sdl, time_list)
        mapping['SDLloop_dPhi'] = dphi
        mapping['SDLloop_dPhi_ext'] = dphi_ext
        mapping['SDLloop_dTheta'] = dtheta

        # ここに今後 Isat / labeling / ほかの項目を順次追加

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