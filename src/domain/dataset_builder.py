from __future__ import annotations
import numpy as np

def build_rows_by_header(header: list[str], mapping: dict[str, np.ndarray]) -> np.ndarray:
    """
    header の順で 2D 配列を構築。キー不足は KeyError を投げて気付けるように。
    """
    return np.vstack([mapping[name] for name in header]).T