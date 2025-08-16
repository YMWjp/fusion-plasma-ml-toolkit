from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Context:
    shotNO: int
    data_root: str
    data_sources: dict         # config.yml の data_sources セクション
    cfg: dict          
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
        # 必要なら使う。不要なら削ってもOK

    def resolve_path(self, key: str) -> Path:
        """論理キーから実ファイルパスを組み立て"""
        tmpl = self.data_sources[key]
        return Path(str(tmpl).format(root=str(self.data_root), shotNO=int(self.shotNO)))
    
    def load_fircall(self) -> pd.DataFrame:
        if "fircall_df" in self._cache:
            return self._cache["fircall_df"]

        p = self.resolve_path("fircall")  # → {data_root}/fircall@{shotNO}.dat
        text = p.read_text(encoding="utf-8", errors="replace")
        lines = text.splitlines()
        try:
            idx = next(i for i, line in enumerate(lines) if "[data]" in line.lower()) + 1
        except StopIteration:
            raise RuntimeError(f"[data] section not found in {p}")

        df = pd.read_csv(
            io.StringIO("\n".join(lines[idx:])),
            header=None,
            names=["Time",
                   'ne_bar(3849)',
                   'peak',
                   'nL(3309)',
                   'nL(3399)',
                   'nL(3489)',
                   'nL(3579)',
                   'nL(3669)',
                   'nL(3759)',
                   'nL(3849)',
                   'nL(3939)',
                   'nL(4029)',
                   'nL(4119)',
                   'nL(4209)',
                   'nL(4299)',
                   'nL(4389)'],
        )
        self._cache["fircall_df"] = df
        return df