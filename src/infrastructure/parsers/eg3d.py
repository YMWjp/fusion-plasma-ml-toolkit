from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import interpolate

from src.utils.paths import EGDATA_DIR


class Eg3D:
    """
    EG 3D: dims = (time, R) の2次元 + 複数値。
    データは 1D 配列に平坦化されており、(t_idx, r_idx, val_idx) の順で格納されている前提。
    """

    def __init__(self, filename: str | Path):
        self.path = (EGDATA_DIR / filename) if isinstance(filename, str) else Path(filename)
        self.time: list[float] = []
        self.R: list[float] = []
        self.valnames: list[str] = []
        self.valunits: list[str] = []
        self._dim_sizes: tuple[int, int] = (0, 0)
        self._valno: int = 0
        self._data_flat: list[float] = []
        self._read_file()

    def _read_file(self) -> None:
        lines = self.path.read_text(encoding='utf-8').splitlines()
        parsing_data = False
        dimno = 0
        for line in lines:
            if not line:
                continue
            if line.startswith('#'):
                content = line[1:].strip()
                key = content.split('=')[0].strip().upper()
                if key == 'DIMNO':
                    dimno = int(content.split('=')[1])
                if key == 'DIMSIZE':
                    parts = [p.strip() for p in content.split('=')[1].split(',')]
                    self._dim_sizes = (int(parts[0]), int(parts[1]))
                    self.time = [0.0] * self._dim_sizes[0]
                    self.R = [0.0] * self._dim_sizes[1]
                if key == 'DIMNAME':
                    pass
                if key == 'VALNO':
                    self._valno = int(content.split('=')[1])
                if key == 'VALNAME':
                    self.valnames = [v.strip().strip("'") for v in content.split('=')[1].split(',')]
                if key == 'VALUNIT':
                    self.valunits = [v.strip().strip("'") for v in content.split('=')[1].split(',')]
                if key == 'DATA':
                    parsing_data = True
                continue
            if parsing_data:
                cols = [c.strip() for c in line.strip(',').split(',')]
                t = float(cols[0])
                r = float(cols[1])
                # time 軸のインデックスを推定
                if not self.time or t != self.time[-1]:
                    # 先頭または新しい time
                    if len([x for x in self.time if x != 0.0]) < self._dim_sizes[0]:
                        idx_t = len([x for x in self.time if x != 0.0])
                        self.time[idx_t] = t
                # R は逐次更新（最後にユニーク化済みの長さに一致する）
                # 平坦配列へ値を保存
                self._data_flat.extend([float(v) if v else np.nan for v in cols[2:]])
                # R 値は別途収集
                # ここではすべての行の r を蓄積し、最後にユニークな長さで切り出す簡易実装
                self.R.append(r)
        # R 軸の整形
        if len(self.R) >= self._dim_sizes[1]:
            # 最初の time スライスに相当する先頭 N 個を採用
            self.R = self.R[: self._dim_sizes[1]]

    def valname2idx(self, name: str) -> int:
        name_u = name.upper()
        for i, v in enumerate(self.valnames):
            if v.upper() == name_u:
                return i
        raise ValueError(f"value not found: {name}")

    def _reshape(self, val_idx: int) -> np.ndarray:
        """
        戻り shape: (len(time), len(R))
        """
        t_size, r_size = self._dim_sizes
        offset = val_idx
        step = self._valno
        # 平坦配列から val_idx ごとに値を拾い上げ
        vals = self._data_flat[offset::step]
        mat = np.array(vals, dtype=float).reshape(t_size, r_size)
        return mat

    def interpolate_over_time(self, valname: str, R_value: float, target_times: np.ndarray) -> np.ndarray:
        mat = self._reshape(self.valname2idx(valname))  # (T, R)
        # R に沿って最も近い列を選ぶ簡易版（本質は時間方向の補間）
        r_arr = np.asarray(self.R, dtype=float)
        ridx = int(np.argmin(np.abs(r_arr - R_value)))
        series = mat[:, ridx]
        f = interpolate.interp1d(np.asarray(self.time), series, bounds_error=False, fill_value=0.0)
        return f(target_times)

    def extract_time_series(self, valname: str) -> np.ndarray:
        mat = self._reshape(self.valname2idx(valname))
        # 端の R を選ぶことは意味が薄いので中央近傍を採用
        ridx = len(self.R) // 2
        return mat[:, ridx]


class TsmapCalib(Eg3D):
    def ne_from_Te(self, Te_target_keV: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_size, r_size = self._dim_sizes
        reff = self._reshape(self.valname2idx('reff'))
        Te = self._reshape(self.valname2idx('Te_fit'))
        ne = self._reshape(self.valname2idx('ne_fit'))
        dV = self._reshape(self.valname2idx('dVdreff'))
        out_reff = np.zeros(t_size)
        out_ne = np.zeros(t_size)
        out_dV = np.zeros(t_size)
        for i in range(t_size):
            valid = reff[i, :] >= 0
            r = reff[i, valid]
            te = Te[i, valid]
            nne = ne[i, valid]
            dv = dV[i, valid]
            f_reff_from_Te = interpolate.interp1d(te, r, bounds_error=False, fill_value=0.0)
            r_target = f_reff_from_Te(Te_target_keV)
            out_reff[i] = r_target
            f_ne_from_reff = interpolate.interp1d(r, nne, bounds_error=False, fill_value=0.0)
            f_dv_from_reff = interpolate.interp1d(r, dv, bounds_error=False, fill_value=0.0)
            out_ne[i] = f_ne_from_reff(r_target)
            out_dV[i] = f_dv_from_reff(r_target)
        return out_reff, out_ne, out_dV

    def Te_from_reff(self, reff_target: float) -> tuple[np.ndarray, np.ndarray]:
        t_size, r_size = self._dim_sizes
        reff = self._reshape(self.valname2idx('reff'))
        Te = self._reshape(self.valname2idx('Te_fit'))
        ne = self._reshape(self.valname2idx('ne_fit'))
        out_Te = np.zeros(t_size)
        out_ne = np.zeros(t_size)
        for i in range(t_size):
            r = reff[i, :]
            te = Te[i, :]
            nne = ne[i, :]
            f_Te = interpolate.interp1d(r, te, bounds_error=False, fill_value=0.0)
            f_ne = interpolate.interp1d(r, nne, bounds_error=False, fill_value=0.0)
            out_Te[i] = f_Te(reff_target)
            out_ne[i] = f_ne(reff_target)
        return out_Te, out_ne


