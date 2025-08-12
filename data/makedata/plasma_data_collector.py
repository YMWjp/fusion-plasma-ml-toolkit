import argparse
import datetime
import json
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from classes.CalcMPEXP import CalcMPEXP
from classes.eg_read import eg_read

# from get_params.get_beta_e import get_beta_e
# from get_params.get_soxmos import get_soxmos
# from get_params.get_beta0 import get_beta0
# from get_params.get_fig import get_fig
# from get_params.get_col import get_col
# from get_params.get_rmp_lid import get_rmp_lid
# _6 だらだら下がるやつ排除
from egdb_class import egdb2d
from get_params.get_Isat import get_Isat
from get_params.get_nbi import get_nbi
from get_params.get_SDLloop import get_SDLloop

# from getfile_dat import getfile_dat
from getfile_http_2024 import getdata
from scipy import signal
from scipy.ndimage import gaussian_filter1d


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

    def main(self, shotNO, config=None):
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
        eg = eg_read("DivIis_tor_sum@"+str(self.shotNO)+".dat")
        self.Isat = eg.eg_f1('Iis_7L@20', self.time_list)
        self.get_ha()
        self.get_ha3()
        self.get_ha2()
        self.get_ha1()
        self.get_te()
        self.ISS_Wp()
        # get_rmp_lid(self, shotNO)
        get_SDLloop(self, shotNO)
        # get_beta_e(self)
        # get_col(self, shotNO)
        # get_beta0(self, shotNO)
        # get_fig(self, shotNO)
        # get_soxmos(self, shotNO)

        # ラベル付けをする
        self.def_types(shotNO, config)

        return 1

    def get_gdn_info(self, file_path):
        with open(file_path) as file:
            lines = file.readlines()
            for line in lines:
                if "gdn:" in line:
                    gdn_info = line.split(":")[1].strip()
                    gdn_info = gdn_info.split()
                    gdn_info = [int(info) for info in gdn_info]
                    return gdn_info
        return None
    
    

    def def_types(self, shotNO, config=None):
        """各時刻のラベルを self.type_list に格納する"""
        if config is None:
            with open("config.json") as config_file:
                config = json.load(config_file)
        
        detection_mode = config.get("detection_mode", "manual")
        
        if detection_mode == "manual":
            return self._manual_labeling(shotNO, config)
        elif detection_mode == "automatic":
            return self._automatic_labeling(shotNO, config)
        else:
            raise ValueError(f"Unknown detection mode: {detection_mode}")

    def _get_egdata(self, shotNO, diagname, valname):
        """データ取得と整形"""
        getdata(shotNO, diagname, subshotNO=1)
        filename = f"{diagname}@{shotNO:d}.dat"
        egfile = egdb2d(filename)
        egfile.readFile()
        time = np.array(egfile.dimdata)
        data = np.array(egfile.data[egfile.valname2idx(valname)])
        return time, data

    def _apply_labeling(self, start_index, config):
        """ラベリングを適用する共通関数"""
        labeling_config = config.get("labeling", {})
        pre_range = labeling_config.get("pre_range", 15)
        transition_range = labeling_config.get("transition_range", 5)
        post_range = labeling_config.get("post_range", 15)
        pre_label = labeling_config.get("pre_label", -1)
        transition_label = labeling_config.get("transition_label", 0)
        post_label = labeling_config.get("post_label", 1)
        
        self.type_list = np.zeros_like(self.time_list)
        
        # 境界チェックを追加
        max_index = len(self.time_list) - 1
        pre_start = max(0, start_index - pre_range)
        pre_end = max(0, start_index - transition_range)
        trans_start = max(0, start_index - transition_range)
        trans_end = min(max_index + 1, start_index + transition_range)
        post_start = min(max_index + 1, start_index + transition_range)
        post_end = min(max_index + 1, start_index + post_range)
        
        self.type_list[pre_start:pre_end] = pre_label
        self.type_list[trans_start:trans_end] = transition_label
        self.type_list[post_start:post_end] = post_label

    def _manual_labeling(self, shotNO, config):
        """手動ラベリング（クリックベース）"""
        # データ取得
        wp_time, wp_data = self._get_egdata(shotNO, "wp", "Wp")
        isat7L_time = self.time_list
        isat7L_data = self.Isat

        # グラフを表示してユーザーにクリックさせる
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        
        # Wpのグラフ
        ax1.plot(wp_time, wp_data, label='Wp', color='orange')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Wp')
        ax1.legend()

        # Isat_7Lのグラフ
        ax2.plot(isat7L_time, isat7L_data, label='Isat_7L')
        ax2.set_ylabel('Isat_7L')
        ax2.legend()
        
        # グラフのタイトル
        fig.suptitle(f'Shot Number: {shotNO} (Manual Mode - Click to select detachment point)')

        def onclick(event):
            if event.inaxes:
                click_time = event.xdata
                start_index = (np.abs(isat7L_time - click_time)).argmin()
                self._apply_labeling(start_index, config)
                plt.close()

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()

        return 1

    def _automatic_labeling(self, shotNO, config):
        """自動ラベリング（アルゴリズムベース）"""
        # データ取得
        wp_time, wp_data = self._get_egdata(shotNO, "wp", "Wp")
        isat7L_time = self.time_list
        isat7L_data = self.Isat
        
        # アルゴリズム設定
        auto_config = config.get("automatic_detection", {})
        method = auto_config.get("method", "derivative")
        
        if method == "derivative":
            detachment_index = self._detect_by_derivative(isat7L_data, auto_config)
        elif method == "threshold":
            detachment_index = self._detect_by_threshold(isat7L_data, auto_config)
        elif method == "peak":
            detachment_index = self._detect_by_peak(isat7L_data, auto_config)
        else:
            raise ValueError(f"Unknown detection method: {method}")
        
        if detachment_index is None:
            print(f"Warning: Could not detect detachment point for shot {shotNO}")
            self.type_list = np.zeros_like(self.time_list)
            return -1
        
        # ラベルを適用
        self._apply_labeling(detachment_index, config)
        
        # 結果を可視化（オプション）
        if auto_config.get("show_result", True):
            self._visualize_automatic_detection(shotNO, wp_time, wp_data, 
                                               isat7L_time, isat7L_data, detachment_index)
        
        return 1

    def _detect_by_derivative(self, data, config):
        """微分ベースの検出"""
        sigma = config.get("smoothing_sigma", 2.0)
        threshold_percentile = config.get("threshold_percentile", 90)
        
        # データを平滑化
        smoothed_data = gaussian_filter1d(data, sigma=sigma)
        
        # 微分を計算
        derivative = np.gradient(smoothed_data)
        
        # 負の微分（降下）を検出
        negative_derivative = -derivative
        threshold = np.percentile(negative_derivative, threshold_percentile)
        
        # 閾値を超える点を検出
        candidates = np.where(negative_derivative > threshold)[0]
        
        if len(candidates) == 0:
            return None
        
        # 最初の候補点を選択
        return candidates[0]

    def _detect_by_threshold(self, data, config):
        """閾値ベースの検出"""
        threshold_percentile = config.get("threshold_percentile", 50)
        
        # データの最大値の一定割合を閾値とする
        max_val = np.max(data)
        threshold = max_val * (threshold_percentile / 100.0)
        
        # 閾値を下回る最初の点を検出
        below_threshold = np.where(data < threshold)[0]
        
        if len(below_threshold) == 0:
            return None
        
        return below_threshold[0]

    def _detect_by_peak(self, data, config):
        """ピーク検出ベースの検出"""
        min_prominence = config.get("min_prominence", 0.1)
        
        # ピークを検出（負のデータのピーク = 元データの谷）
        peaks, _ = signal.find_peaks(-data, prominence=min_prominence)
        
        if len(peaks) == 0:
            return None
        
        # 最初のピーク（谷）を選択
        return peaks[0]

    def _visualize_automatic_detection(self, shotNO, wp_time, wp_data, 
                                     isat7L_time, isat7L_data, detachment_index):
        """自動検出結果の可視化"""
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(12, 8))
        
        # Wpのグラフ
        ax1.plot(wp_time, wp_data, label='Wp', color='orange')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Wp')
        ax1.legend()

        # Isat_7Lのグラフと検出点
        ax2.plot(isat7L_time, isat7L_data, label='Isat_7L')
        ax2.axvline(isat7L_time[detachment_index], color='red', linestyle='--', 
                   label=f'Detected detachment at t={isat7L_time[detachment_index]:.3f}s')
        ax2.set_ylabel('Isat_7L')
        ax2.legend()
        
        # ラベルの色分け表示
        for i, label_val in enumerate(self.type_list):
            if label_val == -1:
                ax2.axvspan(isat7L_time[i], isat7L_time[min(i+1, len(isat7L_time)-1)], 
                           alpha=0.3, color='blue', label='Pre-detachment' if i == 0 else "")
            elif label_val == 0:
                ax2.axvspan(isat7L_time[i], isat7L_time[min(i+1, len(isat7L_time)-1)], 
                           alpha=0.3, color='yellow', label='Transition' if i == 0 else "")
            elif label_val == 1:
                ax2.axvspan(isat7L_time[i], isat7L_time[min(i+1, len(isat7L_time)-1)], 
                           alpha=0.3, color='red', label='Post-detachment' if i == 0 else "")
        
        # グラフのタイトル
        fig.suptitle(f'Shot Number: {shotNO} (Automatic Detection)')
        plt.tight_layout()
        plt.show()

        return 1

    def pinput(self, new=False):
        if new:
            return (self.ech + self.nbi_tan + self.nbi_perp * 0.5) 
        else:
            return (self.ech + self.nbi_tan + self.nbi_perp * 0.36)

    def norm_prad(self):
        pinput = self.pinput()
        if len(pinput) == 0:
            pinput = np.ones_like(self.time_list)
        return self.prad / pinput

    def make_dataset(self, header):
        self.output_dict = {}
        if "shotNO" in header:
            self.output_dict["shotNO"] = np.ones_like(self.time_list) * self.shotNO
        if "times" in header:
            self.output_dict["times"] = self.time_list
        if "types" in header:
            self.output_dict["types"] = self.type_list
        if "nel" in header:
            self.output_dict["nel"] = self.nel / self.ne_length
        if "B" in header:
            self.output_dict["B"] = np.ones_like(self.time_list) * np.abs(self.Bt)
        if "Pech" in header:
            self.output_dict["Pech"] = self.ech
        if "Pnbi-tan" in header:
            self.output_dict["Pnbi-tan"] = self.nbi_tan
        if "Pnbi-perp" in header:
            self.output_dict["Pnbi-perp"] = self.nbi_perp
        if "Pinput" in header:
            self.output_dict["Pinput"] = self.pinput()
        if "PinputNEW" in header:
            self.output_dict["PinputNEW"] = self.pinput(new=True)
        if "Prad" in header:
            self.output_dict["Prad"] = self.prad
        if "Prad/Pinput" in header:
            self.output_dict["Prad/Pinput"] = self.norm_prad()
        if "Wp" in header:
            self.output_dict["Wp"] = self.wpdia
        if "beta" in header:
            self.output_dict["beta"] = self.beta
        if "Rax" in header:
            self.output_dict["Rax"] = self.geom_center
        if "rax_vmec" in header:
            self.output_dict["rax_vmec"] = self.rax_vmec
        if "a99" in header:
            self.output_dict["a99"] = self.a99
        if "delta_sh" in header:
            self.output_dict["delta_sh"] = self.sh_shift
        if "D/(H+D)" in header:
            self.output_dict["D/(H+D)"] = self.dh
        if "CIII" in header:
            self.output_dict["CIII"] = self.CIII / (self.nel / self.ne_length)
        if "CIV" in header:
            self.output_dict["CIV"] = self.CIV / (self.nel / self.ne_length)
        if "OV" in header:
            self.output_dict["OV"] = self.OV / (self.nel / self.ne_length)
        if "OVI" in header:
            self.output_dict["OVI"] = self.OVI / (self.nel / self.ne_length)
        if "FeXVI" in header:
            self.output_dict["FeXVI"] = self.FeXVI / (self.nel / self.ne_length)
        if "Ip" in header:
            self.output_dict["Ip"] = self.Ip
        if "FIG" in header:
            self.output_dict["FIG"] = self.FIG
        if "Pcc" in header:
            self.output_dict["Pcc"] = self.Pcc
        if "Isat@4R" in header:
            self.output_dict["Isat@4R"] = self.Isat_4R
        if "Isat@6L" in header:
            self.output_dict["Isat@6L"] = self.Isat_6L
        if "Isat@7L" in header:
            self.output_dict["Isat@7L"] = self.Isat
        if "reff@100eV" in header:
            self.output_dict["reff@100eV"] = self.reff100eV
        if "ne@100eV" in header:
            self.output_dict["ne@100eV"] = self.ne100eV
        if "dVdreff@100eV" in header:
            self.output_dict["dVdreff@100eV"] = self.dV100eV
        if "Te@center" in header:
            self.output_dict["Te@center"] = self.Te_center
        if "Te@edge" in header:
            self.output_dict["Te@edge"] = self.Te_edge
        if "ne@center" in header:
            self.output_dict["ne@center"] = self.ne_center
        if "RMP_LID" in header:
            self.output_dict["RMP_LID"] = self.rmp_lid
        if "SDLloop_dPhi" in header:
            self.output_dict["SDLloop_dPhi"] = self.SDLloop_dphi
        if "SDLloop_dPhi_ext" in header:
            self.output_dict["SDLloop_dPhi_ext"] = self.SDLloop_dphi_ext
        if "SDLloop_dTheta" in header:
            self.output_dict["SDLloop_dTheta"] = self.SDLloop_dtheta
        if "beta_e" in header:
            self.output_dict["beta_e"] = self.beta_e
        if "collision" in header:
            self.output_dict["collision"] = self.col
        if "beta0" in header:
            self.output_dict["beta0"] = self.beta0
        if "fig6I" in header:
            self.output_dict["fig6I"] = self.fig6I
        if "pcc3O" in header:
            self.output_dict["pcc3O"] = self.pcc3O
        if "fig/pcc" in header:
            self.output_dict["fig/pcc"] = self.figpcc
        if "ne_soxmos" in header:
            self.output_dict["ne_soxmos"] = self.ne_soxmos
        if "ar_soxmos" in header:
            self.output_dict["ar_soxmos"] = self.ar_soxmos

        lengths = [len(self.output_dict[s]) for s in header if s in self.output_dict]
        print("Header:", header)
        print("Lengths:", lengths)
        print("Output dict keys:", self.output_dict.keys())
        if len(set(lengths)) != 1:
            print(f"Error: 配列の長さが一致していません: {lengths}")
            return
        savelines = np.vstack([self.output_dict[s] for s in header]).T

        with open(self.savename, "a") as f_handle:
            np.savetxt(f_handle, savelines, delimiter=",", fmt="%.5e")

        return

    datapath = "./egdata/"


def main(savename="dataset_25_7.csv", labelname="labels.csv", ion=None, 
         detection_mode=None, config_file="config.json"):
    """main関数の説明
    1放電ごとに DetachData インスタンスを作り，
    CSVファイルに書き込んでいく
    
    Args:
        savename: 出力CSVファイル名
        labelname: ラベルファイル名
        ion: 未使用（後方互換性のため残存）
        detection_mode: 検出モード ('manual' or 'automatic')
        config_file: 設定ファイルのパス
    """

    with open(config_file) as f:
        config = json.load(f)
    
    # コマンドライン引数で検出モードが指定された場合は上書き
    if detection_mode is not None:
        config["detection_mode"] = detection_mode
        print(f"Detection mode set to: {detection_mode}")

    shotNOs = np.genfromtxt(
        labelname, delimiter=",", skip_header=1, usecols=0, dtype=int
    )
    # 必要な事前ラベルを格納する
    if config["use_types"]:
        types = np.genfromtxt(labelname, delimiter=',', skip_header=1, usecols=1, dtype=str)
    if config["use_remarks"]:
        remarks = np.genfromtxt(labelname, delimiter=',', skip_header=1, usecols=3, dtype=str)
    if config["use_about"]:
        about = np.genfromtxt(labelname, delimiter=',', skip_header=1, usecols=4, dtype=float)

    print(shotNOs)
    with open(savename, "w") as f_handle:
        header = config["header"]
        f_handle.write(", ".join(header) + "\n")

    # エラー記録
    with open("errorshot.txt", mode="a") as f:
        datetime.datetime.today().strftime("\n%Y/%m/%d")

    for i, shotNO in enumerate(shotNOs):
        print(shotNO)
        nel_data = DetachData(
            shotNO,
            type=types[i] if config["use_types"] else "",
            remark=remarks[i] if config["use_remarks"] else "",
            about=about[i] if config["use_about"] else 4,
            savename=savename,
        )
        nel_data.remove_files()  # 古いegデータがあったら一旦削除
        main_return = nel_data.main(shotNO, config)
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


def parse_arguments():
    """コマンドライン引数を解析"""
    parser = argparse.ArgumentParser(description='Plasma Data Collector')
    parser.add_argument('--mode', '-m', 
                       choices=['manual', 'automatic'], 
                       help='Detection mode: manual (click-based) or automatic (algorithm-based)')
    parser.add_argument('--savename', '-s', 
                       default='dataset_25_7.csv',
                       help='Output CSV filename')
    parser.add_argument('--labelname', '-l',
                       default='labels.csv', 
                       help='Label CSV filename')
    parser.add_argument('--config', '-c',
                       default='config.json',
                       help='Configuration file path')
    parser.add_argument('--method',
                       choices=['derivative', 'threshold', 'peak'],
                       help='Automatic detection method (only for automatic mode)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # 設定ファイルを読み込んで、コマンドライン引数で上書き
    detection_mode = args.mode
    
    # アルゴリズムメソッドが指定された場合は自動的にautomaticモードに設定
    if args.method and not detection_mode:
        detection_mode = 'automatic'
        print(f"Detection method '{args.method}' specified, setting mode to 'automatic'")
    
    # methodが指定された場合は設定ファイルを一時的に更新
    if args.method:
        
        # 一時的な設定ファイルを作成
        with open(args.config) as f:
            config = json.load(f)
        
        config['automatic_detection']['method'] = args.method
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config, f, indent=2)
            temp_config_path = f.name
        
        try:
            main(savename=args.savename, 
                 labelname=args.labelname,
                 detection_mode=detection_mode,
                 config_file=temp_config_path)
        finally:
            # 一時ファイルを削除
            import os
            os.unlink(temp_config_path)
    else:
        main(savename=args.savename, 
             labelname=args.labelname,
             detection_mode=detection_mode,
             config_file=args.config)
