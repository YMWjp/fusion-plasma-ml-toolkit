# test/easy_visualization.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ── 主要API ───────────────────────────────────────────────────────────
def plot_labels_csv(csv_path: Path | str | None = None, save: int = 0, save_path: Path | None = None):
    """
    CSV を読み込んで 13 個のプロットを描画。
    - csv_path を省略すると、このファイル(test/easy_visualization.py) からの相対で data/test.csv を読む
    - save=0: plt.show() / save!=0: save_path へ保存（省略時は ./datapng/data_<shotNO>_for24.png）
    """
    # パス解決（常にファイル位置基準）
    here = Path(__file__).resolve().parent
    if csv_path is None:
        csv_path = here / "data" / "test.csv"
    else:
        csv_path = Path(csv_path)

    # 読込 & 列名の前後スペース除去
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # 便利関数（列が無い場合は None）
    def col(name):
        return df[name] if name in df.columns else None

    # 時間列（必須）
    t = col("times")
    if t is None:
        raise ValueError("CSV に 'times' 列が見つかりません。列名のスペース等を確認してください。")

    # 値取り出し（存在しない列は None になり、以後のプロットで自動スキップ）
    Wp          = col("Wp")
    nel         = col("nel")
    nel_thomson = col("nel_thomson")  # あれば使用（無ければ描かない）
    Pech        = col("Pech")
    Pnbi_tan    = col("Pnbi-tan")
    Pnbi_perp   = col("Pnbi-perp")
    Prad        = col("Prad")

    OV          = col("OV")
    OVI         = col("OVI")
    CIII        = col("CIII")
    CIV         = col("CIV")
    Ha          = col("Ha")  # 無い場合が多いので任意

    Isat7L      = col("Isat@7L")
    types       = col("types")
    Te_edge     = col("Te@edge")
    reff100     = col("reff@100eV")
    dVdreff100  = col("dVdreff@100eV")
    Te_center   = col("Te@center")
    ne100       = col("ne@100eV")
    ne_center   = col("ne@center")
    ne_edge     = col("ne@edge")  # CSV には無いことが多い

    dphi        = col("SDLloop_dPhi")
    dtheta      = col("SDLloop_dTheta")

    # 図と軸
    fig = plt.figure(figsize=(10, 12))
    axes = [fig.add_subplot(13, 1, i + 1) for i in range(13)]

    def safe_plot(ax, x, y, *args, **kwargs):
        """y が None か全 NaN の時は何も描かない"""
        if y is None:
            return
        if isinstance(y, (pd.Series, np.ndarray)):
            if np.all(pd.isna(y)):
                return
        ax.plot(x, y, *args, **kwargs)

    def legend_from_twins(ax, ax_twin, loc='upper right'):
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax_twin.get_legend_handles_labels()
        if h1 or h2:
            ax.legend(h1 + h2, l1 + l2, loc=loc)

    # 0
    safe_plot(axes[0], t, Wp)
    axes[0].set_ylabel(r'$W_{p}$[MJ]', fontsize=14)

    # 1
    safe_plot(axes[1], t, nel, '.-',)
    safe_plot(axes[1], t, nel_thomson, 'x-')
    axes[1].set_ylabel(r'$\bar{n}_e[10^{19} \mathrm{m}^{-3}]$', fontsize=12)

    # 2 (HEATING)
    if Pnbi_tan is not None or Pnbi_perp is not None:
        nbisum = None
        if Pnbi_tan is not None and Pnbi_perp is not None:
            nbisum = Pnbi_tan + 0.5 * Pnbi_perp
        elif Pnbi_tan is not None:
            nbisum = Pnbi_tan
        elif Pnbi_perp is not None:
            nbisum = 0.5 * Pnbi_perp
        safe_plot(axes[2], t, nbisum, '.-', label='Total NBI')
    safe_plot(axes[2], t, Pech, '.-', label='Total ECH')
    ax2_2 = axes[2].twinx()
    safe_plot(ax2_2, t, Wp, 'g.-', label='Wp')
    axes[2].set_ylabel(r'$Power[\mathrm{MW}]$', fontsize=14)
    ax2_2.set_ylabel(r'$W_p$[MJ]', fontsize=14)
    legend_from_twins(axes[2], ax2_2)

    # 3
    safe_plot(axes[3], t, Prad, '.-')
    axes[3].set_ylabel(r'$P_{rad}$', fontsize=14)

    # 4 (Imp/$\bar{n}_e$ と Ha/$\bar{n}_e$)
    def norm_by_nel(series):
        if series is None or nel is None:
            return None
        with np.errstate(divide='ignore', invalid='ignore'):
            out = series / nel.replace(0, np.nan)
        return out

    safe_plot(axes[4], t, norm_by_nel(OV),  '.-', label='OV/nel')
    safe_plot(axes[4], t, norm_by_nel(CIII),'.-', label='CIII/nel')
    safe_plot(axes[4], t, norm_by_nel(OVI), '.-', label='OVI/nel')
    safe_plot(axes[4], t, norm_by_nel(CIV), '.-', label='CIV/nel')
    axes[4].legend(loc='upper right')
    axes[4].set_xlabel('time[s]', fontsize=14)
    axes[4].set_ylabel(r'Imp/$\bar{n}_e$', fontsize=14)
    axes[4].set_ylim(-0.1, 2)
    ax4_2 = axes[4].twinx()
    safe_plot(ax4_2, t, norm_by_nel(Ha), 'k.-', label='Ha/nel')
    # 元の挙動に合わせて上書き
    axes[4].set_ylabel(r'$H_\alpha/\bar{n}_e$', fontsize=14)

    # 5
    safe_plot(axes[5], t, Isat7L, label='Isat_7L')
    axes[5].set_ylabel(r'$Isat_{7}$[A]', fontsize=13)

    # 6
    safe_plot(axes[6], t, types)
    axes[6].set_ylabel('type', fontsize=14)

    # 7
    safe_plot(axes[7], t, Te_edge)
    axes[7].set_ylabel(r'$Te@edge$', fontsize=14)

    # 8 (reff/dVdreff + Te_center)
    safe_plot(axes[8], t, reff100, '.-', label='reff[m]@100eV')
    if dVdreff100 is not None:
        safe_plot(axes[8], t, dVdreff100 / 100.0, '.-', label=r'dVdreff[$\times 100 \mathrm{m}^2$]@100eV')
    ax8_2 = axes[8].twinx()
    safe_plot(ax8_2, t, Te_center, 'g.-', label='Te[keV]@center')
    axes[8].set_ylabel(r'reff & dVdreff', fontsize=14)
    ax8_2.set_ylabel(r'$Te$[keV]', fontsize=14)
    legend_from_twins(axes[8], ax8_2)

    # 9
    safe_plot(axes[9], t, ne100, '.-', label='ne@100eV')
    safe_plot(axes[9], t, ne_center, '.-', label='ne@center')
    safe_plot(axes[9], t, ne_edge, '.-', label='ne@edge')
    axes[9].set_ylabel(r'$n_e[\times 10^{19} \mathrm{m}^{-3}]$', fontsize=12)
    axes[9].legend(loc='upper right')

    # 10 （線スペクトル）
    safe_plot(axes[10], t, CIII, label='CIII')
    safe_plot(axes[10], t, OV,   label='OV')
    safe_plot(axes[10], t, OVI,  label='OVI')
    axes[10].legend(loc='upper right')
    # 元コードのラベル上書き（意図的）
    axes[10].set_ylabel(r'$H_\alpha$', fontsize=14)

    # 11
    safe_plot(axes[11], t, dphi, label='dphi')
    axes[11].set_ylabel(r'$\Delta\Phi_{eff}$', fontsize=12)

    # 12
    safe_plot(axes[12], t, dtheta, label='dtheta')
    axes[12].set_ylabel(r'$\Delta\Theta_{eff}$', fontsize=12)
    axes[12].legend(loc='upper right')

    # 共通書式
    for i, ax in enumerate(axes):
        if i != len(axes) - 1:
            ax.set_xticklabels('')

    axes[-1].set_xlabel("time[s]", fontsize=14)
    # タイトル（shotNO があれば先頭値を表示）
    shot = df["shotNO"].iloc[0] if "shotNO" in df.columns else "unknown"
    type0 = int(df["types"].iloc[0]) if "types" in df.columns and not pd.isna(df["types"].iloc[0]) else "?"
    axes[0].set_title(f"shotNO={shot}, type={type0}")

    plt.subplots_adjust(hspace=0, bottom=0.05, top=0.95)

    # 保存 or 表示
    if save == 0:
        plt.show()
    else:
        if save_path is None:
            Path("./datapng").mkdir(parents=True, exist_ok=True)
            save_path = Path("./datapng") / f"data_{shot}_for24.png"
        else:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


# ── スクリプト実行時 ─────────────────────────────────────────────────
if __name__ == "__main__":
    # このファイルからの相対で test/data/test.csv を読む
    plot_labels_csv(save=0)