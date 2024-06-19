import datetime

# from getfile_dat import getfile_dat
from getfile_http_2024 import getdata

import numpy as np
from get_params.get_Isat import get_Isat
from get_params.get_nbi import get_nbi
from get_params.get_SDLloop import get_SDLloop
from get_params.get_beta_e import get_beta_e
from get_params.get_soxmos import get_soxmos
from get_params.get_beta0 import get_beta0
from get_params.get_fig import get_fig
from get_params.get_col import get_col
from get_params.get_rmp_lid import get_rmp_lid

# _6 だらだら下がるやつ排除

from make_dataset4 import CalcMPEXP

class DetachData(CalcMPEXP):
    """DetachData クラスの説明
    横山が使っている CalcMPEXPクラスを継承
        # 継承元が持つ変数・メソッドを引き継ぐ
        # 変数・メソッドを追加する事ができる
        # 継承元が持つメソッドと同じ名前のメソッドを定義すると上書き
    ある放電についてデータを取得（igetfile）し，
    CSVファイルに書き込む
    """

    def __init__(
        self,
        shotNO="",
        type="",
        label=0,
        remark="",
        about=4,
        nl_line=1.86,
        savename="dataset_25_7.csv",
        diag_list="diagnames.csv",
    ):
        super().__init__(
            shotNO=shotNO,
            type=type,
            label=label,
            remark=remark,
            about=about,
            nl_line=nl_line,
            savename=savename,
            diag_list=diag_list,
        )

        # Classが保持する値は，__init__で名前だけ定義しておく
        # ほとんどはもとのCalcMPEXPにあるが，必要なものは自分でここで定義する
        self.type_list = []
        self.Isat_6L = []
        self.Isat_7L = []

        self.nel = np.array([])
        self.ech = np.array([])
        self.nbi_tan = np.array([])
        self.nbi_perp = np.array([])
        self.prad = np.array([])

        # 出力用にラベルと変数名を結びつける辞書を用意しておく
        self.output_dict = {}

    def main(self, shotNO):
        getfile = self.getfile_dat()  # データをサーバから取得
        if getfile == -1:
            # 通常
            print("SOME DATA MISSING")
            print(self.missing_list)
            if len(self.missing_list) > 1:
                if self.missing_list == ["fig_h2", "lhdcxs7_nion"]:
                    # pdb.set_trace()
                    pass
                else:
                    return -1
        elif getfile == 2:
            return -1

        self.get_firc()
        # if self.get_thomson(main=False) == -1:
        #     self.nel_thomson = self.nel
        # 密度をthomsonに変更，10ms間隔に固定
        if self.get_thomson() == -1:
            return -1
        if len(self.nel) == 0:
            return -1
        self.get_bolo()

        if self.get_geom() == -1:
            return -1

        # # 使うデータを取得する関数を順に呼び出す
        self.get_ECH()
        get_nbi(self)
        self.get_wp()
        self.get_imp()
        self.get_Ip()
        # self.get_Pzero()
        get_Isat(self)
        self.get_ha()
        self.get_ha3()
        self.get_ha2()
        self.get_ha1()
        self.get_te()
        self.ISS_Wp()
        # get_rmp_lid(self, shotNO)
        # get_SDLloop(self, shotNO)
        # get_beta_e(self)
        # get_col(self, shotNO)
        # get_beta0(self, shotNO)
        # get_fig(self, shotNO)
        # get_soxmos(self, shotNO)

        # ラベル付けをする
        self.def_types(shotNO)

        return 1

    def get_gdn_info(self, file_path):
        with open(file_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if "gdn:" in line:
                    gdn_info = line.split(":")[1].strip()
                    gdn_info = gdn_info.split()
                    gdn_info = [int(info) for info in gdn_info]
                    return gdn_info
        return None
    
    

    def def_types(self, shotNO):
        """各時刻のラベルを self.type_list に格納する"""

        datapath = "./egdata/"

        def get_egdata(shotNO, diagname, valname):
            """データ取得と整形"""
            getdata(shotNO, diagname, subshotNO=1)
            filename = datapath + "{0}@{1:d}.dat".format(diagname, shotNO)
            egfile = egdb2d(filename)
            egfile.readFile()
            time = np.array(egfile.dimdata)
            data = np.array(egfile.data[egfile.valname2idx(valname)])
            return time, data

        try:
            # データ取得
            wp_time, wp_data = get_egdata(shotNO, "wp", "Wp")
            wp_grad = np.gradient(wp_data)
            wp_min_time = wp_time[np.argmin(wp_grad)]
            wp_time_0 = wp_time[wp_data > 50]
            wp_max_time = wp_time_0[0]

            # isat7L データの移動平均と勾配の計算
            isat7L_time = self.time_list
            isat7L_data = self.Isat_7L

            # 動的な移動平均ウィンドウサイズの決定
            window = max(1, len(isat7L_data) // 50)
            w = np.ones(window) / window
            isat7L_data_s = np.convolve(isat7L_data, w, mode="same")
            isat7L_grad = np.gradient(isat7L_data_s)

            # データ範囲を制限
            valid_indices = isat7L_time < wp_min_time - 0.1
            isat7L_grad = isat7L_grad[valid_indices]
            isat7L_time_g = isat7L_time[valid_indices]

            # 動的な閾値の設定
            isat7L_grad_max = max(isat7L_grad)
            grad_threshold = np.percentile(np.abs(isat7L_grad), 95)
            higher_grad_threshold = grad_threshold * 1.5  # 厳しい条件のための閾値を設定
            isat7L_grad_max_time = (
                isat7L_time_g[np.argmax(isat7L_grad)]
                if isat7L_grad_max > grad_threshold
                else wp_min_time - 0.1
            )

            isat7L_grad2 = isat7L_grad[isat7L_time_g > isat7L_grad_max_time]
            isat7L_time2 = isat7L_time_g[isat7L_time_g > isat7L_grad_max_time]

            self.type_list = np.ones_like(self.time_list)

            if len(isat7L_grad2) != 0 and min(isat7L_grad2) < -higher_grad_threshold:
                retouch_time = isat7L_time2[np.argmin(isat7L_grad2)]
                type_list = [
                    0
                    if t < isat7L_grad_max_time - 0.2
                    else 1
                    if t < isat7L_grad_max_time - 0.1
                    else 0
                    if t < isat7L_grad_max_time + 0.1
                    else -1
                    if t < isat7L_grad_max_time + 0.2
                    else 0
                    if t < retouch_time - 0.2
                    else -1
                    if t < retouch_time - 0.1
                    else 0
                    if t < retouch_time + 0.1
                    else 1
                    if t < retouch_time + 0.2
                    else 0
                    for t in self.time_list
                ]

            elif wp_min_time - isat7L_grad_max_time < 0.2:
                type_list = [
                    0
                    if t < isat7L_grad_max_time - 0.2
                    else 1
                    if t < isat7L_grad_max_time - 0.1
                    else 0
                    for t in self.time_list
                ]
            else:
                isat7L_data_s = isat7L_data_s[isat7L_time < wp_min_time - 0.1]
                isat7L_data3 = isat7L_data_s[isat7L_time_g >= isat7L_grad_max_time]
                isat7L_time3 = isat7L_time_g[isat7L_time_g >= isat7L_grad_max_time]
                isat7L_data3 = isat7L_data3[isat7L_time3 < isat7L_grad_max_time + 0.25]
                isat7L_time3 = isat7L_time3[isat7L_time3 < isat7L_grad_max_time + 0.25]
                detach_start_time = isat7L_time3[np.argmin(isat7L_data3)]

                if detach_start_time + 0.05 > isat7L_grad_max_time + 0.2:
                    type_list = [
                        0
                        if t < isat7L_grad_max_time - 0.2
                        else 1
                        if t < isat7L_grad_max_time - 0.1
                        else 0
                        if t < detach_start_time - 0.05
                        else -1
                        if t < detach_start_time + 0.05
                        else 0
                        for t in self.time_list
                    ]
                else:
                    type_list = [
                        0
                        if t < isat7L_grad_max_time - 0.2
                        else 1
                        if t < isat7L_grad_max_time - 0.1
                        else 0
                        if t < isat7L_grad_max_time + 0.1
                        else -1
                        if t < isat7L_grad_max_time + 0.2
                        else 0
                        for t in self.time_list
                    ]

            self.type_list = type_list
            return 1

        except Exception as e:
            print(f"Error occurred: {e}")
            return 0

    def pinput(self, new=False):
        if new:
            return self.ech + self.nbi_tan + self.nbi_perp * 0.5
        else:
            return self.ech + self.nbi_tan + self.nbi_perp * 0.36

    def norm_prad(self):
        pinput = self.pinput()
        if len(pinput) == 0:
            pinput = np.ones_like(self.time_list)
        return self.prad / pinput

    def make_dataset(self, header):  # 修正して使うこと
        # import pdb; pdb.set_trace()
        self.output_dict = {
            "shotNO": np.ones_like(self.time_list) * self.shotNO,
            "times": self.time_list,
            "types": self.type_list,  #'labels':self.label,
            "nel": self.nel / self.ne_length,
            "B": np.ones_like(self.time_list) * np.abs(self.Bt),
            "Pech": self.ech,
            "Pnbi-tan": self.nbi_tan,
            "Pnbi-perp": self.nbi_perp,
            "Pinput": self.pinput(),
            "PinputNEW": self.pinput(new=True),
            "Prad": self.prad,
            "Prad/Pinput": self.norm_prad(),
            "Wp": self.wpdia,
            "beta": self.beta,
            "Rax": self.geom_center,
            "rax_vmec": self.rax_vmec,
            "a99": self.a99,  #'delta_sh':self.sh_shift,
            "D/(H+D)": self.dh,
            "CIII": self.CIII / (self.nel / self.ne_length),
            "CIV": self.CIV / (self.nel / self.ne_length),
            "OV": self.OV / (self.nel / self.ne_length),
            "OVI": self.OVI / (self.nel / self.ne_length),
            "FeXVI": self.FeXVI / (self.nel / self.ne_length),
            "Ip": self.Ip,
            # 'FIG':self.FIG,
            # 'Pcc':self.Pcc,
            "Isat@4R": self.Isat_4R,
            "Isat@6L": self.Isat_6L,
            "Isat@7L": self.Isat_7L,
            "reff@100eV": self.reff100eV,
            "ne@100eV": self.ne100eV,
            "dVdreff@100eV": self.dV100eV,
            "Te@center": self.Te_center,
            "Te@edge": self.Te_edge,
            "ne@center": self.ne_center,
            # 'RMP_LID':self.rmp_lid,
            # 'SDLloop_dPhi':self.SDLloop_dphi,
            # 'SDLloop_dPhi_ext':self.SDLloop_dphi_ext,
            # 'SDLloop_dTheta':self.SDLloop_dtheta,
            # 'beta_e':self.beta_e,
            # 'collision':self.col,
            # 'beta0':self.beta0,
            # 'fig6I':self.fig6I,
            # 'pcc3O':self.pcc3O,
            # 'fig/pcc':self.figpcc,
            # 'ne_soxmos':self.ne_soxmos,
            # 'ar_soxmos':self.ar_soxmos
        }
        # import pdb;pdb.set_trace()
        # return super().make_dataset() #修正するときに消す行
        savelines = np.vstack([self.output_dict[s] for s in header]).T

        with open(self.savename, "a") as f_handle:
            np.savetxt(f_handle, savelines, delimiter=",", fmt="%.5e")

        return

    datapath = "./egdata/"


def main(savename="dataset_25_7.csv", labelname="labels.csv", ion=None):
    """main関数の説明
    1放電ごとに DetachData インスタンスを作り，
    CSVファイルに書き込んでいく
    """
    shotNOs = np.genfromtxt(
        labelname, delimiter=",", skip_header=1, usecols=0, dtype=int
    )
    # 以下，必要な事前ラベルを格納する
    # types = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=1, dtype=str)
    # labels = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=2, dtype=int)
    # remarks = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=3, dtype=str)
    # about = np.genfromtxt(labelname,delimiter=',',skip_header=1, usecols=4,  dtype=float)

    print(shotNOs)
    # データを保存するファイル（CSV）を用意する
    #  labels は，放電のラベル（なければ削除のこと）
    #  types は，データ自体のラベル
    with open(savename, "w") as f_handle:
        header = [
            "shotNO",
            "times",
            "types",
            "nel",
            "B",
            "Pech",
            "Pnbi-tan",
            "Pnbi-perp",
            "Pinput",
            "Prad",
            "Prad/Pinput",
            "Wp",
            "beta",
            "Rax",
            "rax_vmec",
            "a99",  #'delta_sh',
            "D/(H+D)",
            "CIII",
            "CIV",
            "OV",
            "OVI",
            "FeXVI",
            "Ip",
            # 'FIG',
            # 'Pcc',
            "Isat@4R",
            "Isat@6L",
            "Isat@7L",
            "reff@100eV",
            "ne@100eV",
            "dVdreff@100eV",
            "Te@center",
            "Te@edge",
            "ne@center",  #'ne_peak'
            # 'RMP_LID',
            # 現在使用できない
            # 'SDLloop_dPhi','SDLloop_dPhi_ext','SDLloop_dTheta',
            # 'beta_e','collision','beta0'
            # ,'fig6I','pcc3O','fig/pcc'
            # 'ne_soxmos','ar_soxmos'
        ]
        f_handle.write(", ".join(header) + "\n")

    # エラー記録
    with open("errorshot.txt", mode="a") as f:
        datetime.datetime.today().strftime("\n%Y/%m/%d")

    for i, shotNO in enumerate(shotNOs):
        print(shotNO)
        nel_data = DetachData(
            shotNO,
            # types[i], labels[i], remarks[i],about[i], #ここは使うものだけ
            savename=savename,
        )
        nel_data.remove_files()  # 古いegデータがあったら一旦削除
        main_return = nel_data.main(shotNO)
        if main_return == -1:
            nel_data.remove_files()
            continue
        elif main_return == "MPexp error":
            with open("errorshot.txt", mode="a") as f:
                f.write("\n" + str(shotNO))
            nel_data.remove_files()
            continue

        nel_data.make_dataset(header)  # データをCSVへ出力する
        nel_data.plot_labels(save=1)  # 画像として保存する
        nel_data.remove_files()
    return


if __name__ == "__main__":
    main()
