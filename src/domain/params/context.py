from __future__ import annotations

import io
import re
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
        text = p.read_text(encoding="utf-8", errors="replace")

        qre = re.compile(r"'([^']*)'")

        dim_names, val_names = [], []
        data_lines = []
        in_data = False

        for raw in text.splitlines():
            line = raw.rstrip("\n")
            s = line.strip()

            if not in_data:
                if "[data]" in s.lower():
                    in_data = True
                    continue
                if not s.startswith("#"):
                    continue
                if "DimName" in s:
                    dim_names = qre.findall(s)  # 例: ["Time","R"]
                elif "ValName" in s:
                    val_names = qre.findall(s)
            else:
                if s and not s.startswith("#"):
                    data_lines.append(line)

        # 列名（Dim → Val）
        columns = dim_names + val_names
        # 数値部をCSVとして読込（カンマ区切り・空白混在・指数表記OK）
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            header=None,
            names=columns if columns else None,
            comment="#",
            skip_blank_lines=True,
            engine="python",
        )
        # 列過剰なら切り詰め（最小限の護身）
        if columns and df.shape[1] > len(columns):
            df = df.iloc[:, :len(columns)]
            df.columns = columns
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
        file_path = f"{diag_name}@{self.shotNO}.dat"
        p = Path(self.data_root) / file_path
        
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        
        # ファイルを読み込み
        text = p.read_text(encoding="utf-8", errors="replace")
        
        # コメント欄の抽出
        comments = {}
        in_comments = False
        
        for line in text.splitlines():
            line = line.strip()
            
            # [Comments]セクションの開始を検出
            if "[Comments]" in line:
                in_comments = True
                continue
            
            # 次のセクション（[data]など）が来たら終了
            if in_comments and line.startswith("[") and line.endswith("]"):
                break
            
            # コメント欄内で '= 数字' の形式を検索
            if in_comments and line.startswith("#"):
                # '#' を除去
                content = line[1:].strip()
                
                # '= 数字' の形式をチェック
                if "=" in content:
                    parts = content.split("=", 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value_str = parts[1].strip()
                        
                        # 数値かどうかをチェック
                        try:
                            # 指数表記（例: 2.02404e+07）も対応
                            value = float(value_str)
                            comments[key] = value
                        except ValueError:
                            # 数値でない場合はスキップ
                            continue
        
        self._cache[cache_key] = comments
        return comments
        