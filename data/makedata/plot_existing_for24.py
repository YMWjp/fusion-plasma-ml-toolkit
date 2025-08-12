#!/usr/bin/env python3
# ruff: noqa: I001
"""
既存EGファイルのみを用い、CalcMPEXP と同等の前処理で data_<shot>_for24.png を生成します。
ダウンロードもデタッチ時刻の手動指定も行いません。（data/makedata 内で実行）

Usage:
  python plot_existing_for24.py --labels labels.csv
  python plot_existing_for24.py --shots 163402,163403 --type quench
"""

from __future__ import annotations

import argparse
import numpy as np
from tqdm import tqdm

from classes.CalcMPEXP import CalcMPEXP
from get_params.get_SDLloop import get_SDLloop


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Render data_<shot>_for24.png from existing EG files')
    p.add_argument('--labels', default='labels.csv', help='CSV with shot numbers (1st column)')
    p.add_argument('--shots', help='Comma-separated shot numbers (override labels.csv)')
    p.add_argument('--type', choices=['quench', 'steady'], default='quench', help='Plot style (affects shading)')
    p.add_argument('--about', type=float, default=4.0, help='About time for time-range logic')
    return p.parse_args()


def process_shot(shot: int, *, typ: str, about: float) -> bool:
    # CalcMPEXP を使うが、ダウンロード(main)を避けて既存ファイルから順に構築
    c = CalcMPEXP(shotNO=shot, type=typ, about=about)
    # FIR（時間軸）→ Thomson → Bolo → MPexp
    if c.get_firc() == -1:
        return False
    if c.get_thomson() == -1:
        return False
    if len(c.nel) == 0:
        return False
    c.get_bolo()
    c.calc_MPEXP()
    # 幾何→時間範囲
    if c.get_geom() == -1:
        return False
    if c.set_time_range() == -1:
        return False
    # 加熱・NBI・コア・不純物・電流など
    c.get_ECH()
    c.get_nbi()
    c.get_wp()
    c.get_imp()
    c.get_Ip()
    # 中性粒子・Isat 等（存在しなければゼロ埋めになる）
    c.get_Pzero()
    c.get_Isat()
    c.get_ha()
    c.get_ha3()
    c.get_ha2()
    c.get_ha1()
    # SDL ループ（plot_labels で参照される）
    try:
        get_SDLloop(c, shot)
    except Exception:
        pass
    # TsmapCalib 由来
    try:
        c.get_te()
    except Exception:
        pass
    # ISS Wp
    try:
        c.ISS_Wp()
    except Exception:
        pass
    # タイプリスト（プロット上 1 段使う）: デタッチ手動はせず、とりあえずゼロで埋める
    c.type_list = np.zeros_like(c.time_list)
    # 出力
    c.plot_labels(save=1)
    return True


def main() -> None:
    args = parse_args()
    if args.shots:
        shots = [int(x) for x in args.shots.split(',') if x.strip()]
    else:
        shots = np.genfromtxt(args.labels, delimiter=',', skip_header=1, usecols=0, dtype=int).tolist()

    for shot in tqdm(shots, desc='Render for24', unit='shot'):
        ok = process_shot(int(shot), typ=args.type, about=args.about)
        if not ok:
            tqdm.write(f'[WARN] skip shot {shot} (missing prerequisites)')


if __name__ == '__main__':
    main()


