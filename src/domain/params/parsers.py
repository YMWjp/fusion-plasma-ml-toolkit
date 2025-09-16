"""
EGDBファイルのパーサーモジュール

EGDBファイルのコメント欄から各種情報を抽出し、データ部分をDataFrameに変換する機能を提供します。
"""

from __future__ import annotations

import io
import re

import pandas as pd


class EGDBParser:
    """EGDBファイルのパーサークラス"""
    
    @staticmethod
    def parse_raw_egdb(content: str) -> pd.DataFrame:
        """
        EGDBテキスト（#ヘッダ + [data] 数値行）を読み込み、DimName→ValNameの順で
        ヘッダを付けたDataFrameを返す。単位は無視。存在チェックは最小限。
        
        Args:
            content: EGDBファイルの内容（文字列）
            
        Returns:
            パースされたDataFrame
        """
        qre = re.compile(r"'([^']*)'")

        dim_names, val_names = [], []
        data_lines = []
        in_data = False

        for raw in content.splitlines():
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
                    # 行末のカンマとスペースを除去
                    cleaned_line = line.rstrip(", \t")
                    data_lines.append(cleaned_line)

        # 列名（Dim → Val）
        columns = dim_names + val_names
        
        # 重複する列名を修正
        unique_columns = []
        name_counts = {}
        for col in columns:
            if col in name_counts:
                name_counts[col] += 1
                unique_columns.append(f"{col}_{name_counts[col]}")
            else:
                name_counts[col] = 0
                unique_columns.append(col)
        
        # 数値部をCSVとして読込（カンマ区切り・空白混在・指数表記OK）
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            header=None,
            names=unique_columns if unique_columns else None,
            comment="#",
            skip_blank_lines=True,
            engine="python",
        )
        # 列過剰なら切り詰め（最小限の護身）
        if unique_columns and df.shape[1] > len(unique_columns):
            df = df.iloc[:, :len(unique_columns)]
            df.columns = unique_columns

        return df

    @staticmethod
    def parse_raw_egdb_2D(content: str, dim_name: str, dim_value: str) -> pd.DataFrame:
        """
        EGDBテキスト（#ヘッダ + [data] 数値行）を読み込み、DimName→ValNameの順で
        ヘッダを付けたDataFrameを返す。単位は無視。存在チェックは最小限。
        
        Args:
            content: EGDBファイルの内容（文字列）
            dim_name: 次元名
            dim_value: 次元値（"max"の場合は最大値を使用）
        Returns:
            パースされたDataFrame
        """
        qre = re.compile(r"'([^']*)'")

        dim_names, val_names = [], []
        data_lines = []
        in_data = False

        for raw in content.splitlines():
            line = raw.rstrip("\n")
            s = line.strip()

            if not in_data:
                if "[data]" in s.lower():
                    in_data = True
                    continue
                if not s.startswith("#"):
                    continue
                if dim_name in s:
                    dim_names = qre.findall(s)  # 例: ["Time","R"]
                elif "ValName" in s:
                    val_names = qre.findall(s)
            else:
                if s and not s.startswith("#"):
                    # 行末のカンマとスペースを除去
                    cleaned_line = line.rstrip(", \t")
                    data_lines.append(cleaned_line)

        # 列名（Dim → Val）
        columns = dim_names + val_names
        
        # 重複する列名を修正
        unique_columns = []
        name_counts = {}
        for col in columns:
            if col in name_counts:
                name_counts[col] += 1
                unique_columns.append(f"{col}_{name_counts[col]}")
            else:
                name_counts[col] = 0
                unique_columns.append(col)
        
        # 数値部をCSVとして読込（カンマ区切り・空白混在・指数表記OK）
        df = pd.read_csv(
            io.StringIO("\n".join(data_lines)),
            header=None,
            names=unique_columns if unique_columns else None,
            comment="#",
            skip_blank_lines=True,
            engine="python",
        )
        # 列過剰なら切り詰め（最小限の護身）
        if unique_columns and df.shape[1] > len(unique_columns):
            df = df.iloc[:, :len(unique_columns)]
            df.columns = unique_columns

        # dim_valueが"max"の場合は最大値を使用
        if dim_value.lower() == "max" and dim_name in df.columns:
            # dim_name列のユニークな値を取得
            unique_values = df[dim_name].unique()
            # 数値として比較可能な値のみを抽出
            numeric_values = []
            for val in unique_values:
                try:
                    numeric_values.append(float(val))
                except (ValueError, TypeError):
                    continue
            
            if numeric_values:
                # 最大値を見つける
                max_value = max(numeric_values)
                # 最大値に一致する行のみを抽出（数値として比較）
                df = df[df[dim_name].astype(float) == max_value]
        else:
            # 指定されたdim_valueの行のみを抽出
            if dim_name in df.columns:
                # 数値として比較可能かチェック
                try:
                    target_value = float(dim_value)
                    df = df[df[dim_name].astype(float) == target_value]
                except (ValueError, TypeError):
                    # 数値でない場合は文字列として比較（前後の空白を除去）
                    df = df[df[dim_name].astype(str).str.strip() == str(dim_value).strip()]

        return df
    
    @staticmethod
    def parse_comments(content: str) -> dict[str, float]:
        """
        EGDBファイルのコメント欄から数値パラメータを抽出する
        
        Args:
            content: ファイルの内容（文字列）
            
        Returns:
            パラメータ名をキー、数値を値とする辞書
        """
        comments = {}
        in_comments = False
        
        for line in content.splitlines():
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
                content_line = line[1:].strip()
                
                # '= 数字' の形式をチェック
                if "=" in content_line:
                    parts = content_line.split("=", 1)
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
        
        return comments
    
    @staticmethod
    def parse_norm_factors(content: str) -> dict[str, float]:
        """
        EGDBファイルのコメント欄から Norm.Factors を抽出する
        
        Args:
            content: ファイルの内容（文字列）
            
        Returns:
            パラメータ名をキー、正規化係数を値とする辞書
        """
        norm_factors = {}
        in_comments = False
        
        for line in content.splitlines():
            line = line.strip()
            
            # [Comments]セクションの開始を検出
            if "[Comments]" in line:
                in_comments = True
                continue
            
            # 次のセクション（[Data]など）が来たら終了
            if in_comments and line.startswith("[") and line.endswith("]"):
                break
            
            # コメント欄内で Norm.Factors を検索
            if in_comments and line.startswith("#") and "Norm.Factors" in line:
                # '#' を除去
                content_line = line[1:].strip()
                
                # "Norm.Factors = " の部分を除去
                if "Norm.Factors" in content_line:
                    factors_str = content_line.split("Norm.Factors = ", 1)[1]
                    
                    # 各パラメータの係数を抽出
                    # 例: "CIV:  1.000, OVI:  1.000, HI:  1.000"
                    parts = factors_str.split(",")
                    for part in parts:
                        part = part.strip()
                        if ":" in part:
                            param_name, factor_str = part.split(":", 1)
                            param_name = param_name.strip()
                            factor_str = factor_str.strip()
                            
                            try:
                                factor = float(factor_str)
                                norm_factors[param_name] = factor
                            except ValueError:
                                # 数値変換に失敗した場合はスキップ
                                continue
                    break
        
        return norm_factors
