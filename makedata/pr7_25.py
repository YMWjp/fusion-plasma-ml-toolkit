
import os
import pdb
import sys
import glob
import datetime
# from getfile_dat import getfile_dat
from getfile_http_2024 import getdata
from egdb_class import *

import numpy as np
# from scipy import interpolate, signal
import matplotlib.pyplot as plt
from scipy import interpolate

#_6 だらだら下がるやつ排除

from make_dataset4 import CalcMPEXP,eg_read
from neubeta import TsmapRead, GiotaRead, IotaGet
from get_p0 import TsmapSmoothRead, P0Get

class DetachData(CalcMPEXP):
    '''DetachData クラスの説明
    横山が使っている CalcMPEXPクラスを継承
        # 継承元が持つ変数・メソッドを引き継ぐ
        # 変数・メソッドを追加する事ができる
        # 継承元が持つメソッドと同じ名前のメソッドを定義すると上書き
    ある放電についてデータを取得（igetfile）し，
    CSVファイルに書き込む
    '''
    def __init__(self, shotNO='', type='', label = 0, remark='',about= 4, nl_line=1.86,savename='dataset_25_7.csv', diag_list='diagnames.csv'):
        # __init__()はインスタンス生成時に必ず実行されるメソッド（＝関数）
        # super().__init__()は，継承元クラスを__init__する
        super().__init__(shotNO=shotNO, type=type, label=label, remark=remark, about=about, nl_line=nl_line, savename=savename, diag_list=diag_list)

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

    def main(self,shotNO):
        getfile = self.getfile_dat() #データをサーバから取得
        if getfile == -1:
            # 通常
            print("SOME DATA MISSING")
            print(self.missing_list)
            if len(self.missing_list)>1:
                if self.missing_list == ['fig_h2','lhdcxs7_nion']:
                    # pdb.set_trace()
                    pass
                else:
                    return -1
            # 検証用データ
            # print("NO DATA")
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
        self.get_nbi()
        self.get_wp()
        self.get_imp()
        self.get_Ip()
        # self.get_Pzero()
        self.get_Isat()
        self.get_ha()
        self.get_ha3()
        self.get_ha2()
        self.get_ha1()
        self.get_te()
        self.ISS_Wp()
        # self.get_rmp_lid(shotNO)
        self.get_SDLloop(shotNO)
        # self.get_beta_e()
        # self.get_col(shotNO)
        # self.get_beta0(shotNO)
        # self.get_fig(shotNO)
        # self.get_soxmos(shotNO)

        # ラベル付けをする
        self.def_types(shotNO)
    

        return 1
    
    def get_gdn_info(self,file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                if 'gdn:' in line:
                    gdn_info = line.split(':')[1].strip()
                    gdn_info = gdn_info.split()
                    gdn_info = [int(info) for info in gdn_info]
                    return gdn_info
        return None

    def get_Isat(self): #上書き
        # return super().get_Isat()
        if os.path.isfile("./DivIis_tor_sum@"+str(self.shotNO)+".dat"):
            eg = eg_read("./DivIis_tor_sum@"+str(self.shotNO)+".dat")
            gdn_info = self.get_gdn_info("./DivIis_tor_sum@"+str(self.shotNO)+".dat")
            print("gdn情報:", gdn_info, self.shotNO)
            # self.Isat_2L = eg.eg_f1('Iis_2L@20', self.time_list)
            # self.Isat_2R = eg.eg_f1('Iis_2R@20', self.time_list)
            # self.Isat_4L = eg.eg_f1('Iis_4L@20', self.time_list)
            # self.Isat_4R = eg.eg_f1('Iis_4R@20', self.time_list)
            # self.Isat_6L = eg.eg_f1('Iis_6L@20', self.time_list)
            # self.Isat_6R = eg.eg_f1('Iis_6R@20', self.time_list)
            # self.Isat_7L = eg.eg_f1('Iis_7L@20', self.time_list)
            # self.Isat_7R = eg.eg_f1('Iis_7R@20', self.time_list)
            # self.Isat_8L = eg.eg_f1('Iis_8L@20', self.time_list)
            # self.Isat_8R = eg.eg_f1('Iis_8R@20', self.time_list)
            # self.Isat_9L = eg.eg_f1('Iis_9L@20', self.time_list)
            # self.Isat_9R = eg.eg_f1('Iis_9R@20', self.time_list)
            # self.Isat_10L = eg.eg_f1('Iis_10L@20', self.time_list)
            # self.Isat_10R = eg.eg_f1('Iis_10R@20', self.time_list)
            for i, nL in enumerate(['2L', '2R', '4L', '4R', '6L', '6R', '7L', '7R', '8L', '8R', '9L', '9R', '10L', '10R']):
                try:
                    print(gdn_info[i],nL,self.shotNO,"get data")
                    setattr(self, 'Isat_' + nL, eg.eg_f1('Iis_' + nL + '@' + str(gdn_info[i]), self.time_list))
                except:
                    pass
            # time_list4R = self.time_list[self.Isat_4R>0]
            # isat_4R = self.Isat_4R[self.Isat_4R>0]
            # isat4R_f = interpolate.interp1d(time_list4R, isat_4R, kind='linear', bounds_error=False,fill_value=0)
            # self.Isat_4R = isat4R_f(self.time_list)
            # # self.Isat_6L = eg.eg_f1('Iis_6L@18', self.time_list)
            # if self.shotNO>=171355&self.shotNO<=171387:
            #     self.Isat_6L = eg.eg_f1('Iis_6L@19', self.time_list)
            # else:
            #     self.Isat_6L = eg.eg_f1('Iis_6L@20', self.time_list)
            # time_list6L = self.time_list[self.Isat_6L>0]
            # isat_6L = self.Isat_6L[self.Isat_6L>0]
            # isat6L_f = interpolate.interp1d(time_list6L, isat_6L, kind='linear', bounds_error=False,fill_value=0)
            # self.Isat_6L = isat6L_f(self.time_list)
            # self.Isat_6R = eg.eg_f1('Iis_6R@6', self.time_list)
            # # self.Isat_7L = eg.eg_f1('Iis_7L@19', self.time_list)
            # self.Isat_7L = eg.eg_f1('Iis_7L@20', self.time_list)
            # self.Isat_7R = eg.eg_f1('Iis_7R@19', self.time_list)
            # self.Isat_8L = eg.eg_f1('Iis_8L@20', self.time_list)
            # self.Isat_8R = eg.eg_f1('Iis_8R@20', self.time_list)
            # self.Isat_9L = eg.eg_f1('Iis_9L@14', self.time_list)
            # self.Isat_9R = eg.eg_f1('Iis_9R@19', self.time_list)
            # self.Isat_10L = eg.eg_f1('Iis_10L@19', self.time_list)
            # self.Isat_10R = eg.eg_f1('Iis_10R@20', self.time_list)
        else:
            self.Isat_2L = np.zeros_like(self.time_list)
            self.Isat_2R = np.zeros_like(self.time_list)
            self.Isat_4L = np.zeros_like(self.time_list)
            self.Isat_4R = np.zeros_like(self.time_list)
            self.Isat_6L = np.zeros_like(self.time_list)
            self.Isat_6R = np.zeros_like(self.time_list)
            self.Isat_7L = np.zeros_like(self.time_list)
            self.Isat_7R = np.zeros_like(self.time_list)
            self.Isat_8L = np.zeros_like(self.time_list)
            self.Isat_8R = np.zeros_like(self.time_list)
            self.Isat_9L = np.zeros_like(self.time_list)
            self.Isat_9R = np.zeros_like(self.time_list)
            self.Isat_10L = np.zeros_like(self.time_list)
            self.Isat_10R = np.zeros_like(self.time_list)
        return 1

    def def_types(self,shotNO):
        '''自分で書いたラベル付けのコードがここに当てはまる
        self.type_listに各時刻のラベルが格納されるように
        '''

        datapath='./egdata/'

        def get_egdata(shotNO, diagname, valname):
            # getfile_dat(shotNO, diagname, datapath=datapath)
            getdata(shotNO, diagname, subshotNO=1)
            filename = datapath + '{0}@{1:d}.dat'.format(diagname,shotNO)
            egfile = egdb2d(filename)
            egfile.readFile()
            time = np.array(egfile.dimdata)
            data = np.array(egfile.data[egfile.valname2idx(valname)])
            return time, data

        wp_time, wp_data = get_egdata(shotNO, 'wp', 'Wp')
        wp_grad = np.gradient(wp_data)
        wp_min_time = wp_time[np.argmin(wp_grad)]
        wp_time_0 = wp_time[wp_data>50]
        wp_max_time = wp_time_0[0]

        print(self.Bt)

        if self.Bt>0:
            isat7L_time, isat7L_data = get_egdata(shotNO, 'DivIis_tor_sum', 'Iis_4R@18')
        else:
            if shotNO <= 171387:
                isat7L_time, isat7L_data = get_egdata(shotNO, 'DivIis_tor_sum', 'Iis_6L@19')
            else:
                isat7L_time, isat7L_data = get_egdata(shotNO, 'DivIis_tor_sum', 'Iis_6L@20')
            #detachshot_extraでは全放電で6L@20を確認


        
        isat7L_time = isat7L_time[isat7L_data>0]
        isat7L_data = isat7L_data[isat7L_data>0]
        isat7L_data_s = [0]*len(isat7L_time)
        window = 20 # 移動平均の範囲
        w = np.ones(window)/window
        isat7L_data_s = np.convolve(isat7L_data, w, mode='same')
        isat7L_grad = np.gradient(isat7L_data_s)
        isat7L_grad = isat7L_grad[isat7L_time<wp_min_time-0.1]
        isat7L_time_g = isat7L_time[isat7L_time<wp_min_time-0.1]
        isat7L_grad_min = min(isat7L_grad)
        if isat7L_grad_min<-0.003:
            isat7L_grad_min_time = isat7L_time_g[np.argmin(isat7L_grad)]
        else:
            isat7L_grad_min_time = wp_min_time-0.1
        isat7L_grad2 = isat7L_grad[isat7L_time_g>isat7L_grad_min_time]
        isat7L_time2 = isat7L_time_g[isat7L_time_g>isat7L_grad_min_time]

        self.type_list = np.ones_like(self.time_list)
        if len(isat7L_grad2) != 0 and max(isat7L_grad2) > 0.03:
            retouch_time = isat7L_time2[np.argmax(isat7L_grad2)]
            if retouch_time-isat7L_grad_min_time > 0.4:
                type_list = [
                    0 if t<isat7L_grad_min_time-0.2
                    else -1 if t<isat7L_grad_min_time-0.1
                    else 0 if t<isat7L_grad_min_time+0.1
                    else 1 if t<isat7L_grad_min_time+0.2
                    else 0 if t<retouch_time-0.2
                    else 1 if t<retouch_time-0.1
                    else 0 if t<retouch_time+0.1
                    else -1 if t<retouch_time+0.2
                    else 0
                    for t in self.time_list
                ]
            elif retouch_time-isat7L_grad_min_time > 0.2:
                type_list = [
                    0 if t<isat7L_grad_min_time-0.2
                    else -1 if t<isat7L_grad_min_time-0.1
                    else 0 if t<isat7L_grad_min_time+0.1
                    else 1 if t<retouch_time-0.1
                    else 0 if t<retouch_time+0.1
                    else -1 if t<retouch_time+0.2
                    else 0
                    for t in self.time_list
                ]
            else:
                type_list = [
                    0 if t<isat7L_grad_min_time-0.2
                    else -1 if t<isat7L_grad_min_time-0.1
                    else 0 if t<retouch_time+0.1
                    else -1 if t<retouch_time+0.2
                    else 0
                    for t in self.time_list
                ]
            # for i in range (len(self.time_list)):
            #     if self.time_list[i]<wp_max_time:
            #         self.type_list[i] = 0
            #     elif self.time_list[i]<isat7L_grad_min_time:
            #         self.type_list[i] = -1
            #     elif self.time_list[i]<retouch_time:
            #         self.type_list[i] = 1
            #     elif self.time_list[i]<wp_min_time-0.1:
            #         self.type_list[i] = -1
            #     else:
            #         self.type_list[i] = 0
        elif wp_min_time-isat7L_grad_min_time<0.2:
            type_list = [
                0 if t<isat7L_grad_min_time-0.2
                else -1 if t<isat7L_grad_min_time-0.1
                else 0
                for t in self.time_list
            ]
        else:
            isat7L_data_s = isat7L_data_s[isat7L_time<wp_min_time-0.1]
            isat7L_data3 = isat7L_data_s[isat7L_time_g>=isat7L_grad_min_time]
            isat7L_time3 = isat7L_time_g[isat7L_time_g>=isat7L_grad_min_time]
            isat7L_data3 = isat7L_data3[isat7L_time3<isat7L_grad_min_time+0.25]
            isat7L_time3 = isat7L_time3[isat7L_time3<isat7L_grad_min_time+0.25]
            isat7L_time3 = isat7L_time3[isat7L_data3>0.08]
            isat7L_data3 = isat7L_data3[isat7L_data3>0.08]
            detach_start_time = isat7L_time3[np.argmin(isat7L_data3)]
            if detach_start_time+0.05>isat7L_grad_min_time+0.2:
                type_list = [
                    0 if t<isat7L_grad_min_time-0.2
                    else -1 if t<isat7L_grad_min_time-0.1
                    else 0 if t<detach_start_time-0.05
                    else 1 if t<detach_start_time+0.05
                    else 0
                    for t in self.time_list
                ]
            else:
                type_list = [
                    0 if t<isat7L_grad_min_time-0.2
                    else -1 if t<isat7L_grad_min_time-0.1
                    else 0 if t<isat7L_grad_min_time+0.1
                    else 1 if t<isat7L_grad_min_time+0.2
                    else 0
                    for t in self.time_list
                ]
            # for i in range (len(self.time_list)):
            #     if self.time_list[i]<wp_max_time:
            #         self.type_list[i] = 0
            #     elif self.time_list[i]<isat7L_grad_min_time:
            #         self.type_list[i] = -1
            #     elif self.time_list[i]<wp_min_time-0.1:
            #         self.type_list[i] = 1
            #     else:
            #         self.type_list[i] = 0
        self.type_list = type_list
        return 1

    def pinput(self,new=False):
        if new:
            return self.ech+self.nbi_tan+self.nbi_perp*0.5
        else:
            return self.ech+self.nbi_tan+self.nbi_perp*0.36

    def norm_prad(self):
        pinput = self.pinput()
        if len(pinput) == 0:
            pinput = np.ones_like(self.time_list)
        return self.prad/pinput

    def get_rmp_lid(self,shotNO):
        '''
        with open ('./RMP_detach_shot_list_extract.csv','r') as f :
            names = f.readline().lstrip(' ').rstrip('\n')
        # print(names)
        
        rmp_data = np.loadtxt(
            './RMP_detach_shot_list_extract.csv', skiprows=1, delimiter=','
        ).T

        header = np.loadtxt(
            './RMP_detach_shot_list_extract.csv', usecols=0, delimiter=',', dtype=str
        ).T
        '''

        with open ('./experiment_log_new.csv','r') as f :
            names = f.readline().lstrip(' ').rstrip('\n')
        # print(names)
        
        rmp_data = np.loadtxt(
            './experiment_log_new.csv', skiprows=1, delimiter=','
            
        ).T

        header = np.loadtxt(
            './experiment_log_new.csv', usecols=0, delimiter=',', dtype=str
        ).T

        rmp_no_list = rmp_data[2]
        rmp_lid_list = rmp_data[4]
        for i in range(len(rmp_lid_list)):
            if rmp_lid_list[i] == 0:
                rmp_lid_list[i] = 1
        rmp_lid_list = rmp_lid_list[rmp_no_list==shotNO]
        print(rmp_lid_list)
        print(rmp_lid_list)
        print(rmp_lid_list)
        self.rmp_lid = [rmp_lid_list[0]]*len(self.time_list)
        self.rmp_lid = [n/abs(self.Bt) for n in self.rmp_lid]
        return 1

    def get_nbi(self):
        nb_tmp = np.zeros_like(self.time_list)
        tan_names = ['1','2','3']
        def nbiabs(through,nebar,s):
            if self.Bt<0:
                if s == 1 or 3:
                    loss_ratio = 0.28127 + 0.091059 * np.exp(-3.5618*nebar/10)
                else:
                    loss_ratio = -0.010049 + 2.0175 * np.exp(-10.904*nebar/10)
                    loss_ratio[loss_ratio<0]=0
                loss_ratio[loss_ratio>1]=1
                # print(through,loss_ratio)
                abs = through*(1-loss_ratio)
                return abs
            else:
                if s == 2:
                    loss_ratio = 0.28127 + 0.091059 * np.exp(-3.5618*nebar/10)
                else:
                    loss_ratio = -0.010049 + 2.0175 * np.exp(-10.904*nebar/10)
                    loss_ratio[loss_ratio<0]=0
                loss_ratio[loss_ratio>1]=1
                # print(through,loss_ratio)
                abs = through*(1-loss_ratio)
                return abs
            

        for s in tan_names:
            # eg = eg_read("./nb"+s+"pwr@"+str(self.shotNO)+".dat")
            # _temporalしかない放電もある．原因不明　そのためtemporalのまま
            eg = eg_read("./nb"+s+"pwr_temporal@"+str(self.shotNO)+".dat")
            #pdb.set_trace()
            unit = eg.eg.valunits()[eg.eg.valname2idx('Pport-through_nb'+s)]
            if unit == 'kW':
                nb_tmp = np.vstack((nb_tmp, nbiabs(eg.eg_f1('Pport-through_nb'+s, self.time_list)/1000,self.nel/self.ne_length,s)))
            elif unit == 'MW':
                nb_tmp = np.vstack((nb_tmp, nbiabs(eg.eg_f1('Pport-through_nb'+s, self.time_list),self.nel/self.ne_length,s)))
        # print(nb_tmp)
        nbi_tan_through = np.sum(np.abs(nb_tmp),axis=0)
        # print(nbi_tan_through)
        self.nbi_tan = nbi_tan_through
        # self.nbi_tan = self.dep_Pnbi_tan(
        #     nbi_tan_through,self.nel_thomson/self.ne_length)
        # # NBI tan　dep計算　0703

        nb_tmp = np.zeros_like(self.time_list)
        perp_names = ['4a','4b','5a','5b']
        for s in perp_names:
            eg = eg_read("./nb"+s+"pwr_temporal@"+str(self.shotNO)+".dat")
            unit = eg.eg.valunits()[eg.eg.valname2idx('Pport-through_nb'+s)]
            if unit == 'kW':
                nb_tmp = np.vstack((nb_tmp, eg.eg_f1('Pport-through_nb'+s, self.time_list)/1000))
            elif unit == 'MW':
                nb_tmp = np.vstack((nb_tmp, eg.eg_f1('Pport-through_nb'+s, self.time_list)))
        self.nbi_perp = np.sum(np.abs(nb_tmp),axis=0)
        # self.prad_grad = np.gradient(self.prad, self.time_list)
        # self.ln_prad_grad = np.gradient(np.log(self.prad), self.time_list)

        return 1

    def get_SDLloop(self,shotNO):
        SDLloop_data = np.loadtxt(
            './SDLloopdata/Phieff'+str(shotNO)+'.dat', skiprows=1, delimiter=','
        )
        self.SDLloop_dphi = np.zeros(len(self.time_list))
        self.SDLloop_dphi_ext = np.zeros(len(self.time_list))
        self.SDLloop_dtheta = np.zeros(len(self.time_list))
        i = 0
        for t in self.time_list:
            ind = np.argmin(abs(t-SDLloop_data[:,0]))
            self.SDLloop_dphi[i] = SDLloop_data[ind,1]
            self.SDLloop_dphi_ext[i] = SDLloop_data[ind,5]
            self.SDLloop_dtheta[i] = np.abs(SDLloop_data[ind,2])
            i = i + 1

    def get_beta_e(self):
        self.beta_e = IotaGet.get_betaiota(self.shotNO,self.time_list)[0]
        return

    def get_col(self,shotNO):
        iota = IotaGet.get_betaiota(shotNO,self.time_list)[2]
        te = IotaGet.get_betaiota(shotNO,self.time_list)[3]
        ne = IotaGet.get_betaiota(shotNO,self.time_list)[4]
        reff = IotaGet.get_betaiota(shotNO,self.time_list)[5]
        def fcol(iota,te,ne,reff,R):
            ln_lambda = 23-np.log((ne*10**13)**0.5/(te*10**3))
            ve = 2.91*(10**7)*ne*ln_lambda/((te*10**3)**1.5)
            et = reff/R
            qR = R/iota
            vth = 4.19*10**5*(te*10**3)**0.5
            vb = ve/((et**1.5)*(vth/qR))
            return vb
        self.col = fcol(iota,te,ne,reff,3.9)
        return 1

    def get_beta0(self,shotNO):
        p0 = P0Get.get_p0(shotNO,self.time_list)
        self.beta0 = p0/self.Bt**2
        return 1

    def get_fig(self,shotNO):
        eg = eg_read("./fig_h2@"+str(self.shotNO)+".dat")
        self.fig6I = eg.eg_f1('FIG(6I_W)', self.time_list)
        self.pcc3O = eg.eg_f1('Pcc(3-O)', self.time_list)
        self.figpcc = self.fig6I/self.pcc3O



    def get_soxmos(self,shotNO):
        if shotNO>=171335 and shotNO<=171387:
            with open('./20211028_Ne_line/s'+str(shotNO)+'_VUV109L_490A.txt','r') as f :
                names = f.readline().lstrip(' ').rstrip('\n') 
            ne_data = np.loadtxt(
                './20211028_Ne_line/s'+str(shotNO)+'_VUV109L_490A.txt'   
            )
            time_data = np.loadtxt(
                './20211028_Ne_line/time_sec.txt'  
            )
            ne_f = interpolate.interp1d(time_data, ne_data, kind='linear', bounds_error=False,fill_value=0)
            ne_list = ne_f(self.time_list)
            ne_list = np.where(ne_list<1,1,ne_list)
            self.ne_soxmos = ne_list
            self.ar_soxmos = np.ones(len(self.time_list))
        elif shotNO>180583 and shotNO<181322:
            with open ('./soxmos/soxmos_data/'+str(shotNO)+'/soxmos_peak@'+str(shotNO)+'_ch1_lambda08667_time.txt','r') as f :
                names = f.readline().lstrip(' ').rstrip('\n') 
            ne_data = np.loadtxt(
                './soxmos/soxmos_data/'+str(shotNO)+'/soxmos_peak@'+str(shotNO)+'_ch1_lambda08667_time.txt', skiprows=1     
            ).T
            # header = np.loadtxt(
            #     './soxmos/soxmos_peak@'+str(shotNO)+'_ch1_lambda08667_time', usecols=0, delimiter=',', dtype=str
            # ).T
            ne_f = interpolate.interp1d(ne_data[0], ne_data[5], kind='linear', bounds_error=False,fill_value=0)
            ne_list = ne_f(self.time_list)
            ne_list = np.where(ne_list<1,1,ne_list)
            self.ne_soxmos = ne_list

            with open ('./soxmos/soxmos_data/'+str(shotNO)+'/soxmos_peak@'+str(shotNO)+'_ch2_lambda22054_time.txt','r') as f :
                names = f.readline().lstrip(' ').rstrip('\n') 
            ar_data = np.loadtxt(
                './soxmos/soxmos_data/'+str(shotNO)+'/soxmos_peak@'+str(shotNO)+'_ch2_lambda22054_time.txt', skiprows=1     
            ).T
            # header = np.loadtxt(
            #     './soxmos/soxmos_peak@'+str(shotNO)+'_ch2_lambda22054_time', usecols=0, delimiter=',', dtype=str
            # ).T
            ar_f = interpolate.interp1d(ar_data[0], ar_data[5], kind='linear', bounds_error=False,fill_value=0)
            ar_list = ar_f(self.time_list)
            ar_list = np.where(ar_list<1,1,ar_list)
            self.ar_soxmos = ar_list
        else:
            self.ne_soxmos = np.ones(len(self.time_list))
            self.ar_soxmos = np.ones(len(self.time_list))
        return 1



    
    def make_dataset(self,header): #修正して使うこと
        # import pdb; pdb.set_trace()
        self.output_dict = {
            'shotNO':np.ones_like(self.time_list) *self.shotNO, 
            'times':self.time_list, 
            'types':self.type_list, #'labels':self.label,
            'nel':self.nel/self.ne_length,
            'B':np.ones_like(self.time_list) * np.abs(self.Bt), 
            'Pech':self.ech,
            'Pnbi-tan': self.nbi_tan,
            'Pnbi-perp':self.nbi_perp,
            'Pinput':self.pinput(),
            'PinputNEW':self.pinput(new=True),
            'Prad':self.prad, 
            'Prad/Pinput':self.norm_prad(), 
            'Wp':self.wpdia,
            'beta':self.beta,
            'Rax':self.geom_center, 
            'rax_vmec':self.rax_vmec, 
            'a99':self.a99, #'delta_sh':self.sh_shift,
            'D/(H+D)':self.dh,
            'CIII':self.CIII/(self.nel/self.ne_length), 
            'CIV':self.CIV/(self.nel/self.ne_length), 
            'OV':self.OV/(self.nel/self.ne_length), 
            'OVI':self.OVI/(self.nel/self.ne_length), 
            'FeXVI':self.FeXVI/(self.nel/self.ne_length),
            'Ip':self.Ip,
            # 'FIG':self.FIG, 
            # 'Pcc':self.Pcc, 
            'Isat@4R':self.Isat_4R, 
            'Isat@6L':self.Isat_6L, 
            'Isat@7L':self.Isat_7L,
            'reff@100eV':self.reff100eV, 
            'ne@100eV':self.ne100eV, 
            'dVdreff@100eV':self.dV100eV,
            'Te@center':self.Te_center,
            'Te@edge':self.Te_edge, 
            'ne@center':self.ne_center,
            # 'RMP_LID':self.rmp_lid,
            'SDLloop_dPhi':self.SDLloop_dphi,
            'SDLloop_dPhi_ext':self.SDLloop_dphi_ext,
            'SDLloop_dTheta':self.SDLloop_dtheta,
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
        savelines = np.vstack(
            [
               self.output_dict[s] for s in header
            ]
        ).T

        with open(self.savename, 'a') as f_handle:
            np.savetxt(f_handle, savelines, delimiter=',',fmt='%.5e')
        
        return
    
    datapath='./egdata/'
        

def main(savename='dataset_25_7.csv',labelname='labels.csv',ion=None):
    '''main関数の説明
    1放電ごとに DetachData インスタンスを作り，
    CSVファイルに書き込んでいく
    '''
    shotNOs = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=0, dtype=int)
    # 以下，必要な事前ラベルを格納する
    # types = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=1, dtype=str)
    # labels = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=2, dtype=int)
    # remarks = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=3, dtype=str)
    # about = np.genfromtxt(labelname,delimiter=',',skip_header=1, usecols=4,  dtype=float)

    print(shotNOs)
    # データを保存するファイル（CSV）を用意する
    #  labels は，放電のラベル（なければ削除のこと）
    #  types は，データ自体のラベル
    with open(savename, 'w') as f_handle:
        header = ['shotNO', 'times', 'types',
                'nel','B', 
                'Pech', 'Pnbi-tan', 'Pnbi-perp', 'Pinput', 
                'Prad', 'Prad/Pinput', 'Wp','beta',
                'Rax', 'rax_vmec', 'a99', #'delta_sh',
                'D/(H+D)',
                'CIII', 'CIV', 'OV', 'OVI', 'FeXVI',
                'Ip',
                # 'FIG', 
                # 'Pcc', 
                'Isat@4R', 'Isat@6L', 'Isat@7L',
                'reff@100eV', 'ne@100eV', 'dVdreff@100eV',
                'Te@center','Te@edge', 'ne@center', #'ne_peak'
                
                # 'RMP_LID',

                # 現在使用できない
                'SDLloop_dPhi','SDLloop_dPhi_ext','SDLloop_dTheta',

                # 'beta_e','collision','beta0'
                # ,'fig6I','pcc3O','fig/pcc'
                # 'ne_soxmos','ar_soxmos'
        ]
        f_handle.write(', '.join(header)+'\n')

    # エラー記録
    with open('errorshot.txt', mode='a') as f:
        datetime.datetime.today().strftime("\n%Y/%m/%d")

    for i,shotNO in enumerate(shotNOs):
        print(shotNO)
        nel_data = DetachData(shotNO,
            #types[i], labels[i], remarks[i],about[i], #ここは使うものだけ
            savename=savename
        )
        nel_data.remove_files() #古いegデータがあったら一旦削除
        main_return = nel_data.main(shotNO)
        if  main_return == -1:
            nel_data.remove_files()
            continue
        elif main_return == 'MPexp error':
            with open('errorshot.txt', mode='a') as f:
                f.write('\n'+str(shotNO))
            nel_data.remove_files()
            continue
        
        nel_data.make_dataset(header) #データをCSVへ出力する
        nel_data.plot_labels(save=1) #画像として保存する
        nel_data.remove_files()
    return

if __name__ == '__main__':
    main()