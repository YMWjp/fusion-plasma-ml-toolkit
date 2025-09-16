from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .parsers import EGDBParser


@dataclass
class Context:
    shotNO: int
    data_root: str
    data_sources: dict
    cfg: dict          
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
        # 必要なら使う。不要なら削ってもOK

    def resolve_path(self, key: str) -> Path:
        """論理キーから実ファイルパスを組み立て"""
        tmpl = self.data_sources[key]
        return Path(str(tmpl).format(root=str(self.data_root), shotNO=int(self.shotNO)))
    
    def load_and_parse_raw_egdb(self, key: str) -> pd.DataFrame:
        """
        EGDBテキスト（#ヘッダ + [data] 数値行）を読み込み、DimName→ValNameの順で
        ヘッダを付けたDataFrameを返す。単位は無視。存在チェックは最小限。
        """
        # 簡易キャッシュ
        cache_key = f"egdb:{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        p = self.resolve_path(key)
        if not p.exists():
            return None
        
        text = p.read_text(encoding="utf-8", errors="replace")

        # パーサーモジュールを使用
        df = EGDBParser.parse_raw_egdb(text)
        self._cache[cache_key] = df
        return df

    def load_and_parse_raw_egdb_2D(self, key: str, dim_name: str, dim_value: str) -> pd.DataFrame:
        """
        EGDBテキスト（#ヘッダ + [data] 数値行）を読み込み、DimName→ValNameの順で
        ヘッダを付けたDataFrameを返す。単位は無視。存在チェックは最小限。
        """
        # 簡易キャッシュ
        cache_key = f"egdb:{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        p = self.resolve_path(key)
        if not p.exists():
            return None
        
        text = p.read_text(encoding="utf-8", errors="replace")

        # パーサーモジュールを使用
        df = EGDBParser.parse_raw_egdb_2D(text, dim_name, dim_value)
        self._cache[cache_key] = df
        return df


    def parse_tsmap_nel_comments(self, diag_name: str = "tsmap_nel") -> dict[str, float]:
        """
        EGDBファイルのコメント欄から '= 数字' の形式の値を辞書型で抽出する
        
        Args:
            diag_name: 診断名（デフォルト: "tsmap_nel"）
            
        Returns:
            パラメータ名をキー、数値を値とする辞書
        """
        cache_key = f"comments_{diag_name}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # ファイルパスを取得
        p = self.resolve_path(diag_name)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        
        # ファイルを読み込み
        text = p.read_text(encoding="utf-8", errors="replace")
        
        # パーサーモジュールを使用
        comments = EGDBParser.parse_comments(text)
        
        self._cache[cache_key] = comments
        return comments

    def parse_norm_factors(self, key: str) -> dict[str, float]:
        """
        EGDBファイルのコメント欄から Norm.Factors を辞書型で抽出する
        
        Args:
            key: キー（例: "imp02"）
            
        Returns:
            パラメータ名をキー、正規化係数を値とする辞書
        """
        cache_key = f"norm_factors_{key}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # ファイルパスを取得
        p = self.resolve_path(key)
        
        if not p.exists():
            return {}
        
        # ファイルを読み込み
        text = p.read_text(encoding="utf-8", errors="replace")
        
        # パーサーモジュールを使用
        norm_factors = EGDBParser.parse_norm_factors(text)
        
        self._cache[cache_key] = norm_factors
        return norm_factors
        