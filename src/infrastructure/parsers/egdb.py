from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import interpolate

from src.utils.paths import EGDATA_DIR


class Eg2D:
    def __init__(self, filename: str | Path):
        self.path = (EGDATA_DIR / filename) if isinstance(filename, str) else Path(filename)
        self.dimdata: list[float] = []
        self.data: list[list[float]] = []
        self.valnames: list[str] = []
        self.valunits: list[str] = []
        self.comments: str = ''
        self._read_file()

    def _read_file(self) -> None:
        lines = self.path.read_text(encoding='utf-8').splitlines()
        # dimno = 0
        parsing_data = False
        # raw lines cache for out-of-band markers (e.g., gdn: ...)
        self._raw_lines = lines
        for line in lines:
            if not line:
                continue
            if line.startswith('#'):
                content = line[1:].strip()
                tag = content.split('=')[0].strip().upper()
                # if tag == 'DIMNO':
                #     dimno = int(content.split('=')[1])
                if tag == 'VALNO':
                    valno = int(content.split('=')[1])
                    self.data = [[] for _ in range(valno)]
                if tag == 'VALNAME':
                    rhs = content.split('=')[1]
                    self.valnames = [v.strip().strip("'") for v in rhs.split(',')]
                if tag == 'VALUNIT':
                    rhs = content.split('=')[1]
                    self.valunits = [v.strip().strip("'") for v in rhs.split(',')]
                if tag == 'DATA':
                    parsing_data = True
                if tag not in { 'DIMNO', 'VALNO', 'VALNAME', 'VALUNIT', 'DATA' }:
                    # accumulate original comments
                    self.comments += '#' + content + '\n'
                continue
            if parsing_data:
                cols = [c.strip() for c in line.strip(',').split(',')]
                self.dimdata.append(float(cols[0]))
                for i, v in enumerate(cols[1:]):
                    self.data[i].append(float(v) if v else np.nan)

    def valname2idx(self, name: str) -> int:
        name_u = name.upper()
        for i, v in enumerate(self.valnames):
            if v.upper() == name_u:
                return i
        raise ValueError(f"value not found: {name}")

    def interpolate_series(self, valname: str, target_times: np.ndarray) -> np.ndarray:
        idx = self.valname2idx(valname)
        time = np.asarray(self.dimdata, dtype=float)
        data = np.asarray(self.data[idx], dtype=float)
        f = interpolate.interp1d(time, data, bounds_error=False, fill_value=0.0)
        return f(target_times)

    def find_gdn_indices(self) -> list[int] | None:
        """
        レガシーEGファイル本文に含まれる 'gdn:' 行から、チャンネル毎の gdn インデックス配列を抽出。
        例: 'gdn: 20 20 20 ...' をパース。
        見つからなければ None。
        """
        for line in self._raw_lines:
            if 'gdn:' in line:
                try:
                    rhs = line.split(':', 1)[1]
                    parts = [int(p) for p in rhs.strip().split()]
                    return parts
                except Exception:
                    return None
        return None


class Eg1D:
    def __init__(self, filename: str | Path):
        self.path = (EGDATA_DIR / filename) if isinstance(filename, str) else Path(filename)
        self.dimdata: list[float] = []
        self.valnames: list[str] = []
        self.valunits: list[str] = []
        self.data: list[list[float]] = []
        self._read_file()

    def _read_file(self) -> None:
        lines = self.path.read_text(encoding='utf-8').splitlines()
        parsing_data = False
        for line in lines:
            if not line:
                continue
            if line.startswith('#'):
                kv = line[1:].split('=')
                key = kv[0].strip().upper()
                if key == 'VALNO':
                    valno = int(kv[1])
                    self.data = [[] for _ in range(valno)]
                if key == 'VALNAME':
                    self.valnames = [v.strip().strip("'") for v in kv[1].split(',')]
                if key == 'VALUNIT':
                    self.valunits = [v.strip().strip("'") for v in kv[1].split(',')]
                if key == 'DATA':
                    parsing_data = True
                continue
            if parsing_data:
                cols = [c.strip() for c in line.strip(',').split(',')]
                self.dimdata.append(float(cols[0]))
                for i, v in enumerate(cols[1:]):
                    self.data[i].append(float(v) if v else np.nan)

    def valname2idx(self, name: str) -> int:
        name_u = name.upper()
        for i, v in enumerate(self.valnames):
            if v.upper() == name_u:
                return i
        raise ValueError(f"value not found: {name}")

    def interpolate_series(self, valname: str, target_times: np.ndarray) -> np.ndarray:
        idx = self.valname2idx(valname)
        time = np.asarray(self.dimdata, dtype=float)
        data = np.asarray(self.data[idx], dtype=float)
        f = interpolate.interp1d(time, data, bounds_error=False, fill_value=0.0)
        return f(target_times)


