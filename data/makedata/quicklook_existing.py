#!/usr/bin/env python3
# ruff: noqa: I001

"""
既存の EG データだけを用いて、各ショットのクイックルック画像を作成します。
ダウンロードやラベリング（デタッチ時刻の指定）は行いません。

Usage (data/makedata 直下で実行):
  python quicklook_existing.py --labels labels.csv --outdir ../..//outputs/process/quicklook
  python quicklook_existing.py --shots 163402,163403
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from classes.eg_read import eg_read
from egdb_class import egdb2d


def _load_time_list(shot: int) -> np.ndarray:
    """時間軸を推定して返す。優先: wp → tsmap_nel。"""
    # 1) wp
    try:
        eg = egdb2d(f"wp@{shot}.dat")
        eg.readFile()
        return np.asarray(eg.dimdata, dtype=float)
    except Exception:
        pass
    # 2) tsmap_nel
    try:
        eg = egdb2d(f"tsmap_nel@{shot}.dat")
        eg.readFile()
        return np.asarray(eg.dimdata, dtype=float)
    except Exception:
        pass
    raise RuntimeError("no available time axis (wp/tsmap_nel)")


def _interp(eg: egdb2d, key: str, times: np.ndarray) -> np.ndarray:
    idx = eg.valname2idx(key)
    src_t = np.asarray(eg.dimdata, dtype=float)
    src_y = np.asarray(eg.data[idx], dtype=float)
    if src_t.size == 0:
        return np.zeros_like(times)
    from scipy.interpolate import interp1d
    f = interp1d(src_t, src_y, bounds_error=False, fill_value=0.0)
    return f(times)


def _plot_quicklook(shot: int, t: np.ndarray, series: dict[str, np.ndarray], outdir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    ax = axes.ravel()

    if 'nel' in series:
        ax[0].plot(t, series['nel'], label='nel')
    ax[0].set_ylabel('nel')
    ax[0].legend()

    if 'Prad' in series:
        ax[1].plot(t, series['Prad'], label='Prad')
    if 'Pech' in series:
        ax[1].plot(t, series['Pech'], label='Pech')
    ax[1].set_ylabel('Power')
    ax[1].legend()

    if 'Wp' in series:
        ax[2].plot(t, series['Wp'], label='Wp (MJ)')
    ax[2].set_ylabel('Wp')
    ax[2].legend()

    if 'Ip' in series:
        ax[3].plot(t, series['Ip'], label='Ip')
    ax[3].set_ylabel('Ip')
    ax[3].legend()

    if 'Isat@7L' in series:
        ax[4].plot(t, series['Isat@7L'], label='Isat@7L')
    ax[4].set_ylabel('Isat@7L')
    ax[4].legend()

    if 'Prad' in series and 'Pech' in series:
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.divide(
                series['Prad'],
                series['Pech'],
                out=np.zeros_like(series['Prad']),
                where=series['Pech'] != 0,
            )
        ax[5].plot(t, ratio, label='Prad/Pech')
    ax[5].set_ylabel('ratio')
    ax[5].legend()

    for a in ax:
        a.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel('time [s]')
    axes[-1, 1].set_xlabel('time [s]')
    fig.suptitle(f'Shot {shot} quick look (existing EG)')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outdir / f'shot_{shot}_quicklook.png', dpi=dpi)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Quicklook using existing EG files (no download, no labeling)')
    p.add_argument('--labels', default='labels.csv', help='CSV with shot numbers (1st column)')
    p.add_argument('--shots', help='Comma-separated shot numbers (override labels.csv)')
    p.add_argument('--outdir', default='../../outputs/process/quicklook', help='Output directory for figures')
    p.add_argument('--dpi', type=int, default=120, help='Figure DPI')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    if args.shots:
        shots = [int(x) for x in args.shots.split(',') if x.strip()]
    else:
        shots = np.genfromtxt(args.labels, delimiter=',', skip_header=1, usecols=0, dtype=int).tolist()

    for shot in tqdm(shots, desc='Quicklook existing', unit='shot'):
        shot = int(shot)
        try:
            t = _load_time_list(shot)

            series: dict[str, np.ndarray] = {}

            # nel (任意)
            try:
                eg = egdb2d(f"tsmap_nel@{shot}.dat")
                eg.readFile()
                series['nel'] = _interp(eg, 'nl_thomson_3669', t)
            except Exception:
                pass

            # Prad / Pech
            try:
                eg = egdb2d(f"bolo@{shot}.dat")
                eg.readFile()
                series['Prad'] = _interp(eg, 'Rad_PW', t) / 1000.0
            except Exception:
                pass
            try:
                eg = egdb2d(f"echpw@{shot}.dat")
                eg.readFile()
                series['Pech'] = _interp(eg, 'Total ECH', t)
            except Exception:
                pass

            # Wp / Ip
            try:
                eg = egdb2d(f"wp@{shot}.dat")
                eg.readFile()
                series['Wp'] = _interp(eg, 'Wp', t) / 1000.0
            except Exception:
                pass
            try:
                eg = egdb2d(f"ip@{shot}.dat")
                eg.readFile()
                series['Ip'] = _interp(eg, 'Ip', t)
            except Exception:
                pass

            # Isat@7L（gdn: 20 などがある場合に eg_read で補間取得）
            try:
                eg = eg_read(f"DivIis_tor_sum@{shot}.dat")
                series['Isat@7L'] = eg.eg_f1('Iis_7L@20', t)
            except Exception:
                pass

            _plot_quicklook(shot, t, series, outdir, args.dpi)
        except Exception:
            continue


if __name__ == '__main__':
    main()


