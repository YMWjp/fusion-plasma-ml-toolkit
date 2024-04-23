"""
他に必要なファイル
- 事前ラベル付けを反映した

やること
- zipを開く代わりにigetfile関数でデータを持ってくる '.dat' 形式
- '.dat'を読んでデータセットにする
- ついでに時間範囲きめて描画する
"""

import os
import sys
import glob
# import zipfile
# import subprocess
# import shutil

import numpy as np
from scipy import interpolate, signal
import matplotlib.pyplot as plt


import datetime

# from ..Kaiseki import egdb_class as egdb
# sys.path.append(os.path.join(os.path.dirname(__file__), '../Kaiseki'))
from egdb_class import *
# sys.path.append(os.path.join(os.path.dirname(__file__), '../PyLHD'))
from igetfile import *

# sys.path.append(os.path.join(os.path.dirname(__file__), './tsmap_calib'))
from read_tsmap_calib import *

class ISS04():
    def __init__(self,R):
        
        self.B_coef = 0.88
        self.LHD_coef = 0.93

        if R == 3.6:
            self.R = 3.68 #3.684
            self.a = 0.63 #0.634
            self.iota = 0.65
        elif R == 3.75:
            self.R = 3.75
            self.a = 0.59
            self.iota = 0.59
        elif R == 3.9:
            self.R = 3.82
            self.a = 0.54
            self.iota = 0.58
        else:
            self.R = np.nan
            self.a = np.nan
            self.iota = np.nan
            print("NO SUCH DATA")

        self.B=0
        self.nel = []
        self.P = []
        self.tau = []
        return
    
    def calc_tau(self,B,nel,P):
        self.B = B
        self.nel = nel
        self.P = P
        # import pdb; pdb.set_trace()

        self.tau = 0.134 * np.power(self.a,2.28) * np.power(self.R,0.64) \
                * np.power(self.P,-0.61) * np.power(self.nel,0.54) \
                * np.power(np.abs(self.B)*self.B_coef,0.84) \
                * np.power(self.iota,0.41)
            
        return self.tau

class eg_read():
    def __init__(self, datname):
        self.eg = egdb2d(datname)
        self.eg.readFile()

    def eg_f1(self, valname, timelist):
        time = np.array(self.eg.dimdata[1:])
        data = np.array(self.eg.data[self.eg.valname2idx(valname)][1:])
        f1 = interpolate.interp1d(time, data, bounds_error=False,fill_value=0)
        return f1(timelist)

class GetFiles(object):
    def __init__(self, shotNO, diag_list='diagnames.csv'):
        self.shotNO = shotNO
        # import pdb; pdb.set_trace()
        print(diag_list)
        self.diag_list = np.genfromtxt(diag_list,dtype='str',usecols=(0),delimiter=',')
        self.missing_list = []

    def getfile_dat(self):
        isfile = 1
        for diag in self.diag_list:
            # import pdb; pdb.set_trace()
            outputname = '{0}@{1}.dat'.format(diag, self.shotNO)
            if os.path.isfile(outputname):
                print(outputname, ": exist")
                continue
            # print(diag)
            # print(self.shotNO.dtype)
            # import pdb; pdb.set_trace()
            print(outputname, ": not exist")
            
        # igetfile.py版
            try:
                if igetfile(diag, self.shotNO, 1, outputname) is None:
                    # import pdb; pdb.set_trace()
                    print('shot:{0} diag:{1} is not exist'.format(self.shotNO, diag))
                    self.missing_list.append(diag)
                    isfile = -1
            except:
                print('Bad Zip File ERROR')
                return 2
        return isfile


    def remove_files(self):
        """
        "*_shotNO_*.zip"
        "*@shotNO.dat"
        を削除したい
        """
        print("*****remove*****")
        zipnames = glob.glob('*_'+str(self.shotNO)+'.zip')
        datnames = glob.glob('*@'+str(self.shotNO)+'.dat')

        print(zipnames)
        print(datnames)

        for name in zipnames:
            os.remove(name)

        for name in datnames:
            os.remove(name)

        return 1

class CalcMPEXP(GetFiles):
    def __init__(self, shotNO='', type='', label = 0, remark='',about= 4, nl_line=1.86,savename='dataset.csv', diag_list='diagnames.csv'):
        super().__init__(shotNO,diag_list=diag_list)
        self.shotNO = shotNO

        self.savename = savename

        self.type_str = type
        self.type = 0
        # import pdb; pdb.set_trace()
        if type == 'quench':
            self.type = 1
        elif type=='steady':
            self.type = -1

        self.label = label
        self.remark = remark
        self.about = about

        self.time_range = []
        self.exp_po = 2.5
        self.exp_ne = 2

        self.time_list = []
        self.time_minus_collapse = []

        self.a99 = []
        self.R99 = []
        self.geom_center = []
        self.rax_vmec = []
        self.sh_shift = []
        self.nel = []
        self.nel_grad = []
        self.ln_nel_grad = []

        self.Bt = 0
        self.Rax = 0
        self.gamma = 0

        self.ne_length = nl_line

        self.prad = []
        self.prad_grad = []
        self.ln_prad_grad = []

        self.MPexp = []
        self.exp_conv = []

        self.OV = []
        self.CIII = []
        self.OVI = []
        self.CIV = []

        self.FeXVI = []
        self.HI = []
        self.HeI = []

        self.ech = []
        self.nbi_tan = []
        self.nbi_perp = []

        self.wpdia = []
        self.beta = []
        self.Ip = []

        self.FIG = []
        self.Pcc = []
        self.Isat = []

        self.reff100eV = []
        self.ne100eV = []
        self.dV100eV = []

        self.Te_center = []
        self.Te_edge_inner = []
        self.Te_edge_outer = []
        self.Te_edge = []
        self.ne_center = []
        self.ne_edge = []

        self.ha = []
        self.dh = []
        self.he_ratio = []
        self.d_ratio = []
        self.h_ratio = []
        self.M_eff = []
        self.Z_eff = []

        self.return_names = ''
        self.return_list = []

        self.Wp_iss = []
        self.Pin_ISS = []

        self.nel_raw = []
        self.nel_thomson = []

        # self.ece4012 = []
        # self.ece4105 = []
        # self.ece4216 = []
        # self.ece4320 = []
        self.ece_ch = [
            (146.5, 73.5),
            (136.5, 68.5),
            (129.5, 64.5),
            (123.5, 61.5),
            (117.5, 58.5)
        ]
        
        self.ece_list = [
            [] for i in range(len(self.ece_ch))
        ]

    def main(self):
        getfile = self.getfile_dat()
    
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
        self.calc_MPEXP()

        if self.get_geom() == -1:
            return -1
        
        if self.set_time_range() == -1:
            return 'MPexp error'
        
        self.get_ECH()
        self.get_nbi()
        self.get_wp()

        self.get_imp()

        # self.get_geom()
        self.get_Ip()
        self.get_Pzero()
        self.get_Isat()

        self.get_ha()
        self.get_ha3()
        self.get_ha2()
        self.get_ha1()
        
        self.get_te()
        # self.get_ece()

        self.ISS_Wp()

        self.output()
        return 1

    def main_ece(self):
        getfile = self.getfile_dat()
    
        if getfile == -1:
            # 通常
            print("SOME DATA MISSING")
            print(self.missing_list)
            if self.missing_list == ['ece_slow']:
                return -1
            elif len(self.missing_list)>1:
                if self.missing_list == ['fig_h2','lhdcxs7_nion']:
                    pdb.set_trace()
                    pass
                else:
                    return -1
            # 検証用データ
            # print("NO DATA")
        elif getfile == 2:
            return -1

        self.get_firc()
        if self.get_thomson(main=False) == -1:
            self.nel_thomson = self.nel
        # 密度をthomsonに変更，10ms間隔に固定
        # if self.get_thomson() == -1:
            # return -1
        if len(self.nel) == 0:
            return -1
        self.get_bolo()
        self.calc_MPEXP()
        
        if self.set_time_range() == -1:
            return 'MPexp error'
        
        if self.get_geom() == -1:
            return -1

        self.get_ECH()
        self.get_nbi()
        self.get_wp()

        self.get_imp()
        self.get_te()
        
        if self.get_ece() == -1:
            return -1

        self.ISS_Wp()

        # if np.abs(self.Bt) != 2.75:
        #     return -1

        self.output()
        return 1
    
    def dep_Pnbi_tan(self,Pnbi,nebar):
            # nebar must be in [e19 m-3]
            # import pdb; pdb.set_trace()
            # ratio_loss = 0.873*np.exp(-4.39*nebar/10)
            # これは3.6mの場合
            ratio_loss = 0.971*np.exp(-6.87*nebar/10) #0507
            ratio_loss[ratio_loss>1] = 1
            return (1-ratio_loss)*Pnbi
    
    def ISS_Wp(self):
        # パラメータはISS04.pdfのTable1,2,3 
        # def dep_Pnbi_tan(Pnbi,nebar):
        #     # nebar must be in [e19 m-3]
        #     # import pdb; pdb.set_trace()
        #     # ratio_loss = 0.873*np.exp(-4.39*nebar/10)
        #     ratio_loss = 0.971*np.exp(-6.87*nebar/10) #0507
        #     ratio_loss[ratio_loss>1] = 1
        #     return (1-ratio_loss)*Pnbi

        # self.Pin_ISS = self.ech+self.dep_Pnbi_tan(self.nbi_tan,self.nel_thomson/self.ne_length)+self.nbi_perp*0.5
        self.Pin_ISS = self.ech+self.nbi_tan+self.nbi_perp*0.5
        # nbi_tanをそもそもdepで計算　0703
        Pin = self.Pin_ISS
        window = np.ones(9)/9# 9にしてるのは160745のため
        Pin_sm = np.convolve(Pin, window, mode='same') 
        # import pdb; pdb.set_trace()
        # Pin_sm = signal.savgol_filter(Pin, 21, 3)
        ISS = ISS04(R=self.Rax)
        # nel_thomsonでISSを計算するように変更
        tau = ISS.calc_tau(B=self.Bt,nel=self.nel_thomson/self.ne_length,P=Pin)
        tau_sm = ISS.calc_tau(B=self.Bt,nel=self.nel_thomson/self.ne_length,P=Pin_sm)
        
        # import pdb; pdb.set_trace()
        if np.all(np.isnan(tau_sm)) or np.all(np.isinf(tau_sm)):
            self.Wp_iss = tau_sm
            return 1
        # self.Wp_iss = tau*Pin
        Wp_iss = tau_sm*Pin_sm #0402 スムージング
        Wp_iss[Pin_sm==0] = 0
        # 0403 nan を補間するように
        f_iss = interpolate.interp1d(self.time_list[~np.isnan(Wp_iss)],Wp_iss[~np.isnan(Wp_iss)],'previous',bounds_error=False, fill_value='extrapolate')
        # self.Wp_iss = f_iss(self.time_list)
        self.Wp_iss = Wp_iss
        self.Wp_iss_sm = tau_sm*Pin_sm
        # self.Wp_iss_sm = tau*signal.savgol_filter(Pin, 101, 3)
        # import pdb; pdb.set_trace()
        return 1


    def get_thomson(self,dt=0.01, main=True):
        eg = egdb2d("./tsmap_nel@"+str(self.shotNO)+".dat")
        eg.readFile()
        # import pdb; pdb.set_trace()
        
        time_list = np.array(eg.dimdata[1:])
        if len(time_list)==0:
            return -1

        thomson = np.array(eg.data[eg.valname2idx('nl_thomson_3669')][1:])
        fir = np.array(eg.data[eg.valname2idx('nl_fir_3669')][1:])

        # plt.close();fig,ax = plt.subplots();ax2 = ax.twinx()
        # ax.plot(time_list,fir);ax2.plot(time_list,thomson);plt.show()
        # import pdb; pdb.set_trace()
        
        f1_thomson = interpolate.interp1d(time_list, thomson,bounds_error=False,fill_value=0)
        factor_arg = np.logical_and(time_list>3.45, time_list<3.75)
        
        if len(time_list[factor_arg]) > 0:
            factor = np.nanmean(fir[factor_arg])/np.nanmean(thomson[factor_arg])
            # self.nel_thomson = nel_thomson
            # self.time_thomson = self.time_list
            # self.nel = nel_thomson   
        elif len(time_list)>3:
            factor = np.nanmean(fir[:3])/np.nanmean(thomson[:3])
            # nel_thomson = f1_thomson(self.time_list) * factor
            # self.nel = nel_thomson   
        else:
            return -1
        
        if main:
            self.time_list = np.arange(np.nanmin(time_list),np.nanmax(time_list),dt)
            self.nel = f1_thomson(self.time_list) * factor
            self.nel_thomson = f1_thomson(self.time_list) * factor
            # import pdb; pdb.set_trace()
            self.nel_grad = np.gradient(self.nel, self.time_list)

            eg = egdb2d("./firc@"+str(self.shotNO)+".dat")
            eg.readFile()
            self.time_firc = np.array(eg.dimdata)
            self.nel_firc = np.array(eg.data[eg.valname2idx('nL(3669)')])
            return 1
        else:
            self.nel_thomson = f1_thomson(self.time_list) * factor
            return 1

    def jump_correct(self, num=5):
        eg = egdb2d("./tsmap_nel@"+str(self.shotNO)+".dat")
        eg.readFile()
        # import pdb; pdb.set_trace()
        time_list = np.array(eg.dimdata[1:])
        thomson = np.array(eg.data[eg.valname2idx('nl_thomson_3669')][1:])
        fir = np.array(eg.data[eg.valname2idx('nl_fir_3669')][1:])
        import pdb; pdb.set_trace()
        factor_arg = np.logical_and(time_list>3.5, time_list<4)
        factor = np.max(fir[factor_arg])/np.max(thomson[factor_arg])
        delta_t = np.convolve(time_list, [0,0,1,0,0], mode='valid')
        delta_firc = np.convolve(fir, [0,0,1,0,0], mode='valid')
        delta = delta_firc - np.convolve(factor*thomson, np.ones(num)/num, mode='valid')

        f_delta = interpolate.interp1d(delta_t, delta, bounds_error=False, fill_value=(delta[0],delta[-1]))

        return f_delta

    def get_geom(self,num=9):
        eg = egdb2d("./tsmap_nel@"+str(self.shotNO)+".dat")
        eg.readFile()

        #print(eg.comments)
        
        time_list = np.array(eg.dimdata[1:])
        a99_list = np.array(eg.data[eg.valname2idx('a99')][1:])
        R99_list = np.array(eg.data[eg.valname2idx('R99')][1:])
        geom_list = np.array(eg.data[eg.valname2idx('geom_center')][1:])
        rax_list = np.array(eg.data[eg.valname2idx('Rax_vmec')][1:])
        
        # import pdb; pdb.set_trace()
        if len(time_list)==0:
            return -1

        f1_a99 = interpolate.interp1d(time_list, a99_list,bounds_error=False,fill_value=0)
        f1_R99 = interpolate.interp1d(time_list, R99_list,bounds_error=False,fill_value=0)
        f1_geom = interpolate.interp1d(time_list, geom_list,bounds_error=False,fill_value=0)
        f1_rax = interpolate.interp1d(time_list, rax_list,bounds_error=False,fill_value=0)
        # import pdb; pdb.set_trace()
        self.a99 = f1_a99(self.time_list)
        self.R99 = f1_R99(self.time_list)
        self.geom_center = f1_geom(self.time_list)
        #geom_center はベストフィットVMEC平衡の幾何中心位置 (m)
        self.rax_vmec = f1_rax(self.time_list)
        # self.nel_thomson = f1_thomson(self.time_list)
        # rax_vmec はベストフィットVMEC平衡の磁気軸位置 (m)
        # pdb.set_trace()
        

        comment = eg.comments
        Bt_start = comment.find('Bt')
        Bt_end = comment.find('\n',Bt_start)
        Bt_str = comment[Bt_start+4: Bt_end]
        Rax_start = comment.find('Rax')
        Rax_end = comment.find('\n',Rax_start)
        Rax_str = comment[Rax_start+5: Rax_end]
        gamma_start = comment.find('Gamma')
        gamma_end = comment.find('\n',gamma_start)
        gamma_str = comment[gamma_start+7: gamma_end]
        # pdb.set_trace()
        self.Bt = float(Bt_str)
        self.Rax = float(Rax_str)
        self.sh_shift = self.rax_vmec - self.Rax
        self.gamma = float(gamma_str)
        # self.sh_shift = (self.rax_vmec - self.Rax)/self.a99

        return 1

    def get_firc(self, jump=True):
        eg = egdb2d("./firc@"+str(self.shotNO)+".dat")
        eg.readFile()
        # f_jump = self.jump_correct()
        # self.time_list = np.array(eg.dimdata[1:])
        # self.nel = np.array(eg.data[eg.valname2idx('nL(3669)')][1:])
        
        # import pdb; pdb.set_trace()
        # 0407 "プラズマがある範囲"の条件式をwhereで書き直し
        # >0.5ではなく，”>0.5になって以降”に変更
        arg_nel = np.where(np.array(eg.data[eg.valname2idx('nL(3669)')]) > 0.5)[0]
        # arg_nel = np.array(eg.data[eg.valname2idx('nL(3669)')]) > 0.5
        # print(arg_nel)
        if len(arg_nel) < 3:
            print("no arg")
            self.nel = np.array([])
            return -1
        self.time_list = np.array(eg.dimdata)[np.nanmin(arg_nel):np.nanmax(arg_nel)+1]
        
        self.nel = np.array(eg.data[eg.valname2idx('nL(3669)')])[np.nanmin(arg_nel):np.nanmax(arg_nel)+1]
        # if jump:
        #     self.nel_raw = np.array(eg.data[eg.valname2idx('nL(3669)')])[np.nanmin(arg_nel):np.nanmax(arg_nel)+1]
        #     f_jump = self.jump_correct()
        #     import pdb; pdb.set_trace()  
        #     self.nel = self.nel_raw - f_jump(self.time_list)
        # else:
        #     self.nel_raw = np.array(eg.data[eg.valname2idx('nL(3669)')])[np.nanmin(arg_nel):np.nanmax(arg_nel)+1]
        #     self.nel = np.array(eg.data[eg.valname2idx('nL(3669)')])[np.nanmin(arg_nel):np.nanmax(arg_nel)+1]

        # self.time_list = np.array(eg.dimdata)[arg_nel]
        # self.nel = np.array(eg.data[eg.valname2idx('nL(3669)')])[arg_nel]
        # if not(np.any(arg_nel)):
        #     return -1
        # import pdb; pdb.set_trace()
        
        self.nel_grad = self.calc_diff(self.time_list, self.nel)

        return 1

    def get_imp(self):
        # if os.path.isfile("./imp01@"+str(self.shotNO)+".dat"):
        #     eg = eg_read("./imp01@"+str(self.shotNO)+".dat")
        #     self.OV = eg.eg_f1('OV', self.time_list)
        #     self.CIII = eg.eg_f1('CIII', self.time_list)
        # else:
        #     self.OV = np.zeros_like(self.time_list)
        #     self.CIII = np.zeros_like(self.time_list)
        eg = eg_read("./imp02@"+str(self.shotNO)+".dat")
        self.OVI = eg.eg_f1('OVI', self.time_list)
        self.CIV = eg.eg_f1('CIV', self.time_list)
        # 0926 imp02 version
        self.OV = eg.eg_f1('OV', self.time_list)
        self.CIII = eg.eg_f1('CIII', self.time_list)
        # 1002 added FeXVI and HI
        self.FeXVI = eg.eg_f1('FeXVI', self.time_list)
        self.HI = eg.eg_f1('HI', self.time_list)

        # ゲイン校正
        # OV: #154481-157260のデータは5.368倍するとそれ以前のデータと比較可能
        # OVI: #154539-157260のデータは2.622倍するとそれ以前のデータと比較可能
        # CIII: #155146~#155207のデータは2.655倍するとそれ前後のデータと比較可能
        # CIV: #155146~#155207のデータは2.896倍するとそれ前後のデータと比較可能
        # Cについては158144-158215もゲインを下げている

        if 154481 <= int(self.shotNO) <= 157260:
            self.OV = 5.368 * self.OV

        if 154539 <= int(self.shotNO) <= 157260:
            self.OVI = 2.622 * self.OVI

        if 155146 <= int(self.shotNO) <= 155207 \
            or 158144 <= int(self.shotNO) <= 158215:
            self.CIII = 2.655 * self.CIII
            self.CIV = 2.896 * self.CIV

        return 1

    def get_ECH(self):
        eg_ech = egdb2d("./echpw@"+str(self.shotNO)+".dat")
        eg_ech.readFile()
        ech_time_list = np.array(eg_ech.dimdata[1:])
        total_list = np.array(eg_ech.data[eg_ech.valname2idx('Total ECH')][1:])
        # import pdb; pdb.set_trace()
        f1_total = interpolate.interp1d(ech_time_list, total_list,bounds_error=False,fill_value=0)
        ech_on_off = (f1_total(self.time_list)>0).astype(float)
        self.ech = f1_total(self.time_list)
        return 1
        # if ~os.path.exists("./LHDGAUSS_DEPROF@"+str(self.shotNO)+".dat"):
        #     self.ech = f1_total(self.time_list)
        #     return 1
        
        # eg_dep = eg3d_read("./LHDGAUSS_DEPROF@"+str(self.shotNO)+".dat")
        # if len(eg_dep.time) < 2:
        #     self.ech = f1_total(self.time_list)
        #     return 1
        # dep_total = eg_dep.data[eg_dep.eg.valname2idx('Sum_Total_Power')].reshape(len(eg_dep.time),len(eg_dep.R))[:,-1]
        # # import pdb; pdb.set_trace()
        # f1_dep = interpolate.interp1d(eg_dep.time, dep_total,bounds_error=False,fill_value=0)
        # self.ech = f1_dep(self.time_list) * ech_on_off
        # return 1       


    def get_nbi(self):
        nb_tmp = np.zeros_like(self.time_list)
        tan_names = ['1','2','3']
        for s in tan_names:
            # eg = eg_read("./nb"+s+"pwr@"+str(self.shotNO)+".dat")
            # _temporalしかない放電もある．原因不明　そのためtemporalのまま
            eg = eg_read("./nb"+s+"pwr_temporal@"+str(self.shotNO)+".dat")
            #pdb.set_trace()
            unit = eg.eg.valunits()[eg.eg.valname2idx('Pport-through_nb'+s)]
            if unit == 'kW':
                nb_tmp = np.vstack((nb_tmp, eg.eg_f1('Pport-through_nb'+s, self.time_list)/1000))
            elif unit == 'MW':
                nb_tmp = np.vstack((nb_tmp, eg.eg_f1('Pport-through_nb'+s, self.time_list)))
        nbi_tan_through = np.sum(np.abs(nb_tmp),axis=0)
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

    def get_bolo(self):
        eg_bolo = egdb2d("./bolo@"+str(self.shotNO)+".dat")
        eg_bolo.readFile()
        bolo_time_list = np.array(eg_bolo.dimdata[1:])
        bolo_prad_list = np.array(eg_bolo.data[eg_bolo.valname2idx('Rad_PW')][1:])/1000 #kW -> MW
        bolo_prad_smooth = signal.savgol_filter(bolo_prad_list, 101, 3)
        bolo_grad_list = np.gradient(bolo_prad_smooth,bolo_time_list)
        bolo_grad_smooth = signal.savgol_filter(bolo_grad_list, 101, 3)
        # f1_prad = interpolate.interp1d(bolo_time_list, bolo_prad_list, kind='quadratic')
        #
        # self.prad = f1_prad(self.time_list)
        # self.prad_grad = np.gradient(self.prad, self.time_list)
        # self.ln_prad_grad = np.gradient(np.log(self.prad), self.time_list)
        
        self.prad = np.array([bolo_prad_list[np.argmin(np.abs(bolo_time_list- t))] for t in self.time_list])
        # import pdb; pdb.set_trace()
        self.prad_grad = np.array([bolo_grad_smooth[np.argmin(np.abs(bolo_time_list- t))] for t in self.time_list])
        # self.prad_grad = np.gradient(self.prad, self.time_list)

        return 1


    def get_Ip(self):
        eg = eg_read("./ip@"+str(self.shotNO)+".dat")
        self.Ip = eg.eg_f1('Ip', self.time_list)

        return 1

    def get_wp(self):
        eg = eg_read("./wp@"+str(self.shotNO)+".dat")
        self.wpdia = eg.eg_f1('Wp', self.time_list)/1000
        #wp は egfileではkJだがMJになおしている
        self.beta = eg.eg_f1('<beta-dia>', self.time_list)

        return 1

    def get_Pzero(self):
        if os.path.isfile("./fig_h2@"+str(self.shotNO)+".dat"):
            eg = eg_read("./fig_h2@"+str(self.shotNO)+".dat")
            self.FIG = eg.eg_f1('FIG(1.5U_W)', self.time_list)
            self.Pcc = eg.eg_f1('Pcc(10-O)', self.time_list)
        else:
            self.FIG = np.zeros_like(self.time_list)
            self.Pcc = np.zeros_like(self.time_list)
        return 1


    def get_Isat(self):
        if os.path.isfile("./DivIis_tor_sum@"+str(self.shotNO)+".dat"):
            eg = eg_read("./DivIis_tor_sum@"+str(self.shotNO)+".dat")
            self.Isat = eg.eg_f1('Iis_7L@20', self.time_list)
        else:
            self.Isat = np.zeros_like(self.time_list)
        return 1

    def get_te(self):
        eg = TsmapCalib('./tsmap_calib@'+str(self.shotNO)+'.dat')
        reff_100, ne_100, dV_100 = eg.ne_from_Te(0.1)
        self.reff100eV = eg.eg_f1(reff_100, self.time_list)
        self.ne100eV = eg.eg_f1(ne_100, self.time_list)
        self.dV100eV = eg.eg_f1(dV_100, self.time_list)
        Te_center, ne_center = eg.Te_from_reff(0)
        self.Te_center = eg.eg_f1(Te_center, self.time_list)
        self.ne_center = eg.eg_f1(ne_center, self.time_list)
        # import pdb; pdb.set_trace()
        # Te_edge, ne_edge = eg.edge()
        # self.ne_edge = eg.eg_f1(ne_edge, self.time_list)
        Te_edge, ne_edge = eg.phiEdge()
        self.ne_edge = eg.eg_f1(ne_edge, self.time_list)

        self.Te_edge_inner = eg.eg_f1(Te_edge, self.time_list)
        self.Te_edge_outer = eg.eg_f1(eg.Te_outer, self.time_list)
        self.Te_edge = (self.Te_edge_inner + self.Te_edge_outer)/2
        
        return eg

    def get_ece(self,ece=-1):
        # import pdb; pdb.set_trace()
        eg = EceSlow("./ece_slow@"+str(self.shotNO)+".dat")
        for i in range(len(self.ece_ch)):
            self.ece_list[i] = eg.get_te(self.ece_ch[i], self.time_list)
        # import pdb; pdb.set_trace()
        if np.all([np.all(l==0) for l in self.ece_list]):
            return -1
        
        # import pdb; pdb.set_trace()
        # eg = eg3d_read("./ece_slow@"+str(self.shotNO)+".dat")
        
        # diag = eg.data[eg.eg.valname2idx('diag_number')]
        # if np.all(diag!=1):
        #     return -1
        # R_all = np.array(eg.R*len(eg.time))
        # R_radH = np.unique(R_all[diag==1])
        # t_radH = np.array(eg.time)
        # Te_radH = eg.data[eg.eg.valname2idx('Te'),diag==1].reshape(len(t_radH),len(R_radH))
        # # import pdb; pdb.set_trace()
        # f1 = interpolate.interp1d(t_radH, Te_radH.T[R_radH==4.012][0], bounds_error=False,fill_value=0)
        # self.ece4012 = f1(self.time_list)
        # f1 = interpolate.interp1d(t_radH, Te_radH.T[R_radH==4.105][0], bounds_error=False,fill_value=0)
        # self.ece4105 = f1(self.time_list)
        # f1 = interpolate.interp1d(t_radH, Te_radH.T[R_radH==4.216][0], bounds_error=False,fill_value=0)
        # self.ece4216 = f1(self.time_list)
        # f1 = interpolate.interp1d(t_radH, Te_radH.T[R_radH==4.320][0], bounds_error=False,fill_value=0)
        # self.ece4320 = f1(self.time_list)

        # plt.plot(self.time_list, self.ece_list[0])
        # plt.plot(self.time_list, self.ece_list[1])
        # plt.plot(self.time_list, self.ece_list[2])
        # plt.plot(self.time_list, self.ece_list[3])
        # plt.plot(self.time_list, self.ece_list[4])
        # plt.show()
        # import pdb; pdb.set_trace()
        return 1


    def get_ha(self):
        eg = eg_read("./ha2@"+str(self.shotNO)+".dat")
        ha = np.zeros_like(self.time_list)
        for i in range(10):
            ha_tmp = eg.eg_f1(str(i+1)+'-O(H)', self.time_list)
            ha = ha + ha_tmp
        self.ha = ha

    def get_ha3(self):
        if os.path.isfile("./ha3@"+str(self.shotNO)+".dat"):
            eg = eg_read("./ha3@"+str(self.shotNO)+".dat")
            dh = eg.eg_f1('D/(H+D)', self.time_list)
            dh[dh<0.01] = 0.01
            self.dh = dh
            # plt.plot( self.time_list, eg.eg_f1('Halpha', self.time_list))
            # plt.plot( self.time_list, eg.eg_f1('Dalpha', self.time_list))
            # plt.show()
        else:
            self.dh = np.zeros_like(self.time_list)
        
        return 1
    
    def get_ha2(self):
        if len(self.dh) == 0:
            self.get_ha3()
        # import pdb; pdb.set_trace()
        eg = eg_read("./ha2@"+str(self.shotNO)+".dat")
        dhhe = eg.eg_f1('(H+D)/(H+D+He)', self.time_list)
        self.he_ratio = 1-dhhe
        self.d_ratio = self.dh * dhhe
        self.h_ratio = (1-self.dh) * dhhe
        self.M_eff = 1*self.h_ratio + 2*self.d_ratio + 4*self.he_ratio
        self.Z_eff = 1*self.h_ratio + 1*self.d_ratio + 2*self.he_ratio
        # import pdb; pdb.set_trace()
        return 1

    
    def get_ha1(self):
        eg = eg_read("./ha1@"+str(self.shotNO)+".dat")
        self.HeI =  eg.eg_f1('HeI(Impmon)', self.time_list)
        return 1

    def get_gas_puf(self):
        eg = eg_read("./gas_puf@"+str(self.shotNO)+".dat")
        # import pdb; pdb.set_trace()
        time = np.array(eg.eg.dimdata)
        data = np.array(eg.eg.data)
        offset = np.mean(data[:,time<1],axis=1)
        data = data.T - offset
        # plt.plot(time,data.T-offset)
        # plt.show()
        return time, data, eg.eg.ValNames

    def calc_diff(self, x, y):
        # print(len(x))
        # print(len(y))
        diff =  [0 if i < 2 or i >= len(y)-2 else (y[i-2]-8*y[i-1]+8*y[i+1]-y[i+2])/(6*(x[i+1]-x[i-1])) for i in range(len(y))]
        diff[0] = (y[1]-y[0])/(x[1]-x[0])
        diff[1] = (y[2]-y[0])/(x[2]-x[0])
        diff[len(y)-2] = (y[len(y)-1]-y[len(y)-3])/(x[len(y)-1]-x[len(y)-3])
        diff[len(y)-1] = (y[len(y)-1]-y[len(y)-2])/(x[len(y)-1]-x[len(y)-2])

        # print(diff)
        return diff

    def calc_MPEXP(self):
        # # 通常
        # self.MPexp = (self.prad_grad/self.prad)/(self.nel_grad/self.nel)
        # nel smooth版
        # nel_smooth = signal.savgol_filter(self.nel, 101, 3)
        prad_smooth = np.convolve(self.prad, np.ones(9)/9, mode='same')
        # nel_smooth = np.convolve(self.nel, np.ones(9)/9, mode='same')
        nel_smooth = np.convolve(self.nel_thomson, np.ones(9)/9, mode='same')
        nel_grad = np.gradient(nel_smooth,self.time_list)
        # nel_grad = np.gradient(self.nel_thomson,self.time_list)
        # nel_grad_smooth = signal.savgol_filter(nel_grad, 101, 3)
        nel_grad_smooth = np.convolve(nel_grad, np.ones(9)/9, mode='same')
        # 2021/01/15 修正
        # nel_grad_smooth = np.convolve(nel_smooth, np.ones(9)/9, mode='same')
        # plt.plot(self.time_list,np.convolve(np.gradient(np.convolve(self.nel_thomson, np.ones(9)/9, mode='same'),self.time_list),np.ones(9)/9, mode='same'))
       
        self.MPexp = self.prad_grad/prad_smooth
        # self.MPexp = (self.prad_grad/prad_smooth)/(nel_grad_smooth/nel_smooth)
        # plt.plot(self.time_list,nel_grad)
        # plt.plot(self.time_list,nel_grad_smooth)
        # plt.plot(self.time_list,nel_smooth)
        # plt.plot(self.time_list,self.prad_grad)
        # plt.plot(self.time_list,self.prad_grad/prad_smooth)
        # plt.plot(self.time_list,self.MPexp)
        # plt.plot(self.time_list,np.convolve(self.MPexp, np.ones(9)/9,mode='same'))
        # plt.ylim(-10,10);plt.show()
        # import pdb; pdb.set_trace()
        
        #self.MPexp = self.ln_prad_grad/self.ln_nel_grad
        return 1
    
    def calc_MPEXP_thomson(self):
        prad_smooth = np.convolve(self.prad, np.ones(9)/9, mode='same')
        # nel_smooth = self.nel_thomson
        nel_smooth = np.convolve(self.nel_thomson, np.ones(9)/9, mode='same')
        nel_grad = np.gradient(nel_smooth,self.time_list)
        nel_grad_smooth = np.convolve(nel_grad, np.ones(9)/9, mode='same')
        
        self.MPexp = (self.prad_grad/prad_smooth)/(nel_grad_smooth/nel_smooth)
        return 1

    def output(self):
        self.return_names = 'Time'
        self.return_list = self.time_list

        self.return_names = self.return_names +',nel'+',nel_grad'
        self.return_list = np.vstack((self.return_list, self.nel, self.nel_grad))

        self.return_names = self.return_names + ',Prad' + ',Prad_grad'
        self.return_list = np.vstack((self.return_list, self.prad, self.prad_grad))

        self.return_names = self.return_names + ',MP-EXP'
        self.return_list = np.vstack((self.return_list, self.MPexp))

        return 1

    def set_time_range_old(self):
        exp_conv = np.convolve(self.MPexp, np.ones(9)/9,mode='same')
        self.exp_conv = exp_conv

        remarks = self.remark
        about = self.about

        if self.type == 1:
            #print(about)
            time_base = np.min(self.time_list[np.logical_and(exp_conv>=4, self.time_list>=about)])
            #print(exp_conv[np.logical_and(self.time_list<=time_base, self.time_list>3.5)])
            #print("time_base: %f" % time_base)

            time_3 = np.min(self.time_list[np.logical_and(exp_conv>=3, np.logical_and(self.time_list<=time_base,self.time_list>=about))])
            # print("time_3: %f" % time_3)
            if sum(np.logical_and(exp_conv<2, np.logical_and(self.time_list<time_3,self.time_list>=about))) > 0:
                time_2 = np.max(self.time_list[np.logical_and(exp_conv<2, np.logical_and(self.time_list<time_3,self.time_list>=about))])
            else:
                time_2 = time_3
            # print("time_2: %f" % time_2)
            if sum(np.logical_and(exp_conv<=1.5, np.logical_and(self.time_list<time_2,self.time_list>=about))) > 0:
                time_15 = np.max(self.time_list[np.logical_and(exp_conv<=1.5, np.logical_and(self.time_list<time_2,self.time_list>=about))])
            else:
                time_15 = time_2
            # print("time_15: %f" % time_15)
            # print([time_base, time_3, time_2, time_15])
            self.time_range = [time_15 - 0.1, time_15, time_2, time_base]
        elif self.type == -1:
            self.time_range = [4,5]

        return 1

    def set_time_range(self):
        # import pdb; pdb.set_trace()
        exp_conv = np.convolve(self.MPexp, np.ones(9)/9,mode='same')
        # exp_conv = self.MPexp
        self.exp_conv = exp_conv

        remarks = self.remark
        about = self.about

        if self.type == 1:
            #print(about)
            # pdb.set_trace()
            positive_start = np.min(self.time_list[np.logical_and(exp_conv>=self.exp_po, self.time_list>=about)])
            # 2021/01/15　変更
            prad_decrease = np.logical_and(
                self.time_list>=positive_start,
                self.prad_grad<0
            )
            time = self.time_list[prad_decrease]
            prad = self.prad[prad_decrease]
            positive_end_tmp = time[np.argmax(prad)]
            # positive_end = np.min(time)
            # import pdb; pdb.set_trace()

            # pdb.set_trace()
            # 2020/01/25 もともとのやつにもどす
            positive_end_tmp = np.min(self.time_list[np.logical_and(exp_conv<self.exp_po, self.time_list>=positive_start)])
            density_jump_arg = np.logical_and(
                np.abs(np.hstack((np.zeros(1),np.diff(self.nel,n=1))))>5, 
                self.time_list>=positive_start
            )
            if np.any(density_jump_arg):
                density_jump = np.min(self.time_list[density_jump_arg])
            else:
                density_jump = 100
            positive_end = min(positive_end_tmp, density_jump)
            positive_range = positive_end - positive_start

            if np.all(np.logical_not(np.logical_and(exp_conv<=self.exp_ne, self.time_list<positive_start))):
                return -1
            negative_end = np.max(self.time_list[np.logical_and(exp_conv<=self.exp_ne, self.time_list<positive_start)])
            negative_start = negative_end - positive_range
        
            self.time_range = [negative_start, negative_end, positive_start, positive_end]
        elif self.type == -1:
            self.time_range = [4,5]
            #self.time_range = [3.7,5]

        return 1

    def graph_2021(self,save=0): #200408 -> 200511
            fig = plt.figure(figsize=(6.5,7))
            axes = [fig.add_subplot(5,1,i+1) for i in range(5)]
            # axes = [fig.add_subplot(5,1,i+1) for i in range(5)]

            plt.rcParams['lines.linewidth'] = 2
            cmap = plt.get_cmap("tab10")
            color_l = '#0079C4'
            color_r = '#E00059'

            color_po = '#FF8DBA'
            color_ne = '#81CFFF'

            axes[0].plot(self.time_list, self.exp_conv , c=color_l)
            axes[0].set_ylim(-0.1,5)
            axes[0].set_ylabel('$\dot{P}_\mathrm{rad}/{P}_\mathrm{rad}$'+'\n'+'(smoothed)',fontsize=14)
            # axes[0].set_ylabel('Density'+'\n'+'exponents'+'\n'+'(smoothed)',fontsize=14)
            # axes[0].hlines(self.exp_po,3,6,linestyle='dotted')
            # axes[0].hlines(self.exp_ne,3,6,linestyle='dotted')

            axes[1].plot(self.time_list, self.nel/self.ne_length, c=color_l)
            # axes[1].plot(self.time_list, self.nel_thomson/self.ne_length, c=color_l)
            axes[1].set_ylabel(r'$\bar{n}_e[10^{19} \mathrm{m}^{-3}]$',fontsize=12)
            ax1_2 = axes[1].twinx()
            ax1_2.plot(self.time_list, self.wpdia, c=color_r)
            # ax1_2.plot(self.time_list, self.Wp_iss, 'k--')
            # ax1_2.plot(self.time_list, self.Wp_iss/2, 'k:')
            # ax1_2.plot(self.time_list, self.Wp_iss*0.3, 'k:')
            ax1_2.set_ylabel(r'$W_\mathrm{p}[\mathrm{MJ}]$',fontsize=14)
            
            Pinput = self.ech+self.nbi_tan+self.nbi_perp*0.5
            axes[2].plot(self.time_list, Pinput, c=color_l)
            axes[2].set_ylabel(r'$P_\mathrm{input}[\mathrm{MW}]$',fontsize=14)
            axes[2].set_ylim(-0.5,4.5)
            ax2_2 = axes[2].twinx()
            ax2_2.plot(self.time_list, self.prad, c=color_r)
            ax2_2.set_ylabel(r'$P_\mathrm{rad}[\mathrm{MW}]$',fontsize=14)
            ax2_2.set_ylim(-0.5,4.5)
            
            axes[3].plot(self.time_list, self.Te_center,c=color_l)
            axes[3].set_ylabel(r'$T_\mathrm{e,center}[\mathrm{keV}]$',fontsize=14)
            axes[3].set_ylim(-0.2,2.2)
            ax3_2 = axes[3].twinx()
            ax3_2.plot(self.time_list,self.Te_edge, label='Te',c=color_r)
            ax3_2.set_ylabel(r'$T_\mathrm{e,edge}[\mathrm{keV}]$',fontsize=14)
            ax3_2.set_ylim(-0.02,0.22)

            axes[4].plot(self.time_list, self.CIII/(self.nel/self.ne_length),label='CIII')
            axes[4].plot(self.time_list, self.CIV/(self.nel/self.ne_length),label='CIV')
            axes[4].plot(self.time_list, self.OV/(self.nel/self.ne_length),label='OV')
            axes[4].plot(self.time_list, self.OVI/(self.nel/self.ne_length),label='OVI')
            axes[4].plot(self.time_list, self.FeXVI/(self.nel/self.ne_length),label='FeXVI')
            axes[4].set_ylabel(r'$\mathrm{Imp02}/\bar{n}_\mathrm{e}$',fontsize=14)
            axes[4].legend(
                bbox_to_anchor=(1.005,1), loc='upper left',fontsize=12,
                borderpad=0.2, handlelength=1, handletextpad=0.4,labelspacing=0.1
                )
            axes[4].set_ylim(-0.01,0.5)
            # axes[4].set_ylim(-0.01,0.2)

            #(a)~(d)
            characters = ['(a)','(b)','(c)','(d)','(e)']

            for i in range(len(axes)):
                ax = axes[i]
                ylim = ax.get_ylim()
                if self.type == 1:
                    # old:
                    # self.time_range = [time_15 - 0.1, time_15, time_2, time_base]
                    # new:
                    # self.time_range = [ne_start, ne_end, po_start, po_end]
                    # ax.vlines(self.time_range[3], ylim[0],ylim[1])
                    #ax.axvspan(time_2, time_3, color = "orangered", alpha = 0.5)
                    ax.axvspan(self.time_range[2], self.time_range[3], color = color_po, alpha = 0.8)
                    ax.axvspan(self.time_range[0], self.time_range[1], color = color_ne, alpha = 0.8)
                    ax.set_ylim(ylim[0],ylim[1])
                    ax.set_xlim(self.time_range[1]-0.3, self.time_range[3]+0.05)
                    ax.annotate(characters[i],(0.05,0.87),xycoords='axes fraction',fontsize=13)
                elif self.type == -1:
                    ax.vlines(5, ylim[0],ylim[1])
                    ax.vlines(3.7, ylim[0],ylim[1])
                    ax.set_ylim(ylim[0],ylim[1])
                    ax.set_xlim(3.4,5.6)

                if i != len(axes)-1:
                    ax.set_xticklabels('')
                # import pdb; pdb.set_trace()
                if i == 0:
                    ax.set_title("#%d" %(self.shotNO))
                    # if self.remark == '':
                    #     ax.set_title("#%d(%s)" %(self.shotNO, self.type_str))
                    # else:
                    #     ax.set_title("#%d(%s): %s" %(self.shotNO, self.type_str, self.remark))

            axes[-1].set_xlabel("Time[s]",fontsize=14)

            plt.subplots_adjust(left=0.175,right=0.8,bottom=0.08,top=0.96,hspace=0)
            #plt.show()


    def graph(self,save=0): #200408 -> 200511
        fig = plt.figure(figsize=(6.5,7))
        axes = [fig.add_subplot(5,1,i+1) for i in range(5)]
        # axes = [fig.add_subplot(5,1,i+1) for i in range(5)]

        plt.rcParams['lines.linewidth'] = 2
        cmap = plt.get_cmap("tab10")
        color_l = '#0079C4'
        color_r = '#E00059'

        color_po = '#FF8DBA'
        color_ne = '#81CFFF'

        axes[0].plot(self.time_list, self.exp_conv , c=color_l)
        axes[0].set_ylim(-0.1,5)
        axes[0].set_ylabel('Density'+'\n'+'exponents'+'\n'+'(smoothed)',fontsize=14)
        # axes[0].hlines(self.exp_po,3,6,linestyle='dotted')
        # axes[0].hlines(self.exp_ne,3,6,linestyle='dotted')

        axes[1].plot(self.time_list, self.nel/self.ne_length, c=color_l)
        # axes[1].plot(self.time_list, self.nel_thomson/self.ne_length, c=color_l)
        axes[1].set_ylabel(r'$\bar{n}_e[10^{19} \mathrm{m}^{-3}]$',fontsize=12)
        ax1_2 = axes[1].twinx()
        ax1_2.plot(self.time_list, self.wpdia, c=color_r)
        # ax1_2.plot(self.time_list, self.Wp_iss, 'k--')
        # ax1_2.plot(self.time_list, self.Wp_iss/2, 'k:')
        # ax1_2.plot(self.time_list, self.Wp_iss*0.3, 'k:')
        ax1_2.set_ylabel(r'$W_\mathrm{p}[\mathrm{MJ}]$',fontsize=14)
        
        Pinput = self.ech+self.nbi_tan+self.nbi_perp*0.5
        axes[2].plot(self.time_list, Pinput, c=color_l)
        axes[2].set_ylabel(r'$P_\mathrm{input}[\mathrm{MW}]$',fontsize=14)
        axes[2].set_ylim(-0.5,4.5)
        ax2_2 = axes[2].twinx()
        ax2_2.plot(self.time_list, self.prad, c=color_r)
        ax2_2.set_ylabel(r'$P_\mathrm{rad}[\mathrm{MW}]$',fontsize=14)
        ax2_2.set_ylim(-0.5,4.5)
        
        axes[3].plot(self.time_list, self.Te_center,c=color_l)
        axes[3].set_ylabel(r'$T_\mathrm{e,center}[\mathrm{keV}]$',fontsize=14)
        axes[3].set_ylim(-0.2,2.2)
        ax3_2 = axes[3].twinx()
        ax3_2.plot(self.time_list,self.Te_edge, label='Te',c=color_r)
        ax3_2.set_ylabel(r'$T_\mathrm{e,edge}[\mathrm{keV}]$',fontsize=14)
        ax3_2.set_ylim(-0.02,0.22)

        axes[4].plot(self.time_list, self.CIII/(self.nel/self.ne_length),label='CIII')
        axes[4].plot(self.time_list, self.CIV/(self.nel/self.ne_length),label='CIV')
        axes[4].plot(self.time_list, self.OV/(self.nel/self.ne_length),label='OV')
        axes[4].plot(self.time_list, self.OVI/(self.nel/self.ne_length),label='OVI')
        axes[4].plot(self.time_list, self.FeXVI/(self.nel/self.ne_length),label='FeXVI')
        axes[4].set_ylabel(r'$\mathrm{Imp02}/\bar{n}_\mathrm{e}$',fontsize=14)
        axes[4].legend(
            bbox_to_anchor=(1.005,1), loc='upper left',fontsize=12,
            borderpad=0.2, handlelength=1, handletextpad=0.4,labelspacing=0.1
            )
        axes[4].set_ylim(-0.01,0.5)
        # axes[4].set_ylim(-0.01,0.2)

        #(a)~(d)
        characters = ['(a)','(b)','(c)','(d)','(e)']

        for i in range(len(axes)):
            ax = axes[i]
            ylim = ax.get_ylim()
            if self.type == 1:
                # old:
                # self.time_range = [time_15 - 0.1, time_15, time_2, time_base]
                # new:
                # self.time_range = [ne_start, ne_end, po_start, po_end]
                # ax.vlines(self.time_range[3], ylim[0],ylim[1])
                #ax.axvspan(time_2, time_3, color = "orangered", alpha = 0.5)
                ax.axvspan(self.time_range[2], self.time_range[3], color = color_po, alpha = 0.8)
                ax.axvspan(self.time_range[0], self.time_range[1], color = color_ne, alpha = 0.8)
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(self.time_range[1]-0.3, self.time_range[3]+0.1)
                ax.annotate(characters[i],(0.05,0.87),xycoords='axes fraction',fontsize=13)
            elif self.type == -1:
                ax.vlines(5, ylim[0],ylim[1])
                ax.vlines(3.7, ylim[0],ylim[1])
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(3.4,5.6)

            if i != len(axes)-1:
                ax.set_xticklabels('')
            # import pdb; pdb.set_trace()
            if i == 0:
                ax.set_title("#%d" %(self.shotNO))
                # if self.remark == '':
                #     ax.set_title("#%d(%s)" %(self.shotNO, self.type_str))
                # else:
                #     ax.set_title("#%d(%s): %s" %(self.shotNO, self.type_str, self.remark))

        axes[-1].set_xlabel("Time[s]",fontsize=14)

        plt.subplots_adjust(left=0.175,right=0.8,bottom=0.08,top=0.96,hspace=0)
        #plt.show()

    def graph_old(self,save=0):
        fig = plt.figure(figsize=(6,8))
        axes = [fig.add_subplot(5,1,i+1) for i in range(5)]

        plt.rcParams['lines.linewidth'] = 2
        cmap = plt.get_cmap("tab10")
        color_l = '#0079C4'
        color_r = '#E00059'

        axes[0].plot(self.time_list, self.exp_conv , c=color_l)
        axes[0].set_ylim(-0.1,5)
        axes[0].set_ylabel('Density'+'\n'+'exponents'+'\n'+'(smoothed)',fontsize=14)
        axes[0].hlines(self.exp_po,3,6,linestyle='dotted')
        axes[0].hlines(self.exp_ne,3,6,linestyle='dotted')

        axes[1].plot(self.time_list, self.nel/self.ne_length, c=color_l)
        axes[1].set_ylabel(r'$\bar{n}_e[\times 10^{19} \mathrm{m}^{-3}]$',fontsize=12)
        ax1_2 = axes[1].twinx()
        ax1_2.plot(self.time_list, self.wpdia, c=color_r)
        # ax1_2.plot(self.time_list, self.Wp_iss, 'k--')
        # ax1_2.plot(self.time_list, self.Wp_iss/2, 'k:')
        # ax1_2.plot(self.time_list, self.Wp_iss*0.3, 'k:')
        ax1_2.set_ylabel(r'$W_p$[MJ]',fontsize=14)

        axes[2].plot(self.time_list, self.prad, c=color_l)
        axes[2].set_ylabel(r'$P_{rad}$[MW]',fontsize=14)
        axes[2].set_ylim(-0.5,5)
        ax2_2 = axes[2].twinx()
        Pinput = self.ech+self.nbi_tan+self.nbi_perp*0.5
        ax2_2.plot(self.time_list, Pinput, c=color_r)
        ax2_2.set_ylabel(r'$P_{input}$[MW]',fontsize=14)
        ax2_2.set_ylim(-0.5,5)

        axes[3].plot(self.time_list, self.Te_center,c=color_l)
        axes[3].set_ylabel(r'$T_e^{center}$[keV]',fontsize=14)
        axes[3].set_ylim(-0.2,2.6)
        ax3_2 = axes[3].twinx()
        ax3_2.plot(self.time_list, self.ne_center,c=color_r)
        ax3_2.set_ylabel(r'$n_e[\times 10^{19} \mathrm{m}^{-3}]$'+'\n'+r'$@100\mathrm{eV}$',fontsize=14)

        axes[4].plot(self.time_list, self.Pcc, c=color_l)
        axes[4].set_ylabel(r'$P_{cc}^{(10-O)}$[Pa]',fontsize=14)
        ax4_2 = axes[4].twinx()
        ax4_2.plot(self.time_list, self.Isat, c=color_r)
        ax4_2.set_ylabel(r'$I_{sat}^{(7L)}$[A]',fontsize=14)

        for i in range(len(axes)):
            ax = axes[i]
            ylim = ax.get_ylim()
            if self.type == 1:
                # old:
                # self.time_range = [time_15 - 0.1, time_15, time_2, time_base]
                # new:
                # self.time_range = [ne_start, ne_end, po_start, po_end]
                ax.vlines(self.time_range[3], ylim[0],ylim[1])
                #ax.axvspan(time_2, time_3, color = "orangered", alpha = 0.5)
                ax.axvspan(self.time_range[2], self.time_range[3], color = "orangered", alpha = 0.5)
                ax.axvspan(self.time_range[0], self.time_range[1], color = "royalblue", alpha = 0.5)
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(self.time_range[1]-0.3, self.time_range[3]+0.1)
            elif self.type == -1:
                ax.vlines(5, ylim[0],ylim[1])
                ax.vlines(3.7, ylim[0],ylim[1])
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(3.4,5.6)

            if i != len(axes)-1:
                ax.set_xticklabels('')
            # import pdb; pdb.set_trace()
            if i == 0:
                if self.remark == '':
                    ax.set_title("#%d(%s)" %(self.shotNO, self.type_str))
                else:
                    ax.set_title("#%d(%s): %s" %(self.shotNO, self.type_str, self.remark))

        axes[-1].set_xlabel("time[s]",fontsize=14)

        plt.subplots_adjust(left=0.15,right=0.85,bottom=0.08,top=0.94,hspace=0)
        #plt.show()


    def graph_ES(self,save=0):
        fig = plt.figure(figsize=(6,8))
        axes = [fig.add_subplot(5,1,i+1) for i in range(5)]

        plt.rcParams['lines.linewidth'] = 2
        cmap = plt.get_cmap("tab10")
        color_l = '#0079C4'
        color_r = '#E00059'

        axes[0].plot(self.time_list, self.exp_conv , c=color_l)
        axes[0].set_ylim(-0.1,5)
        axes[0].set_ylabel('Density'+'\n'+'exponents'+'\n'+'(smoothed)',fontsize=14)
        axes[0].hlines(self.exp_po,3,6,linestyle='dotted')
        axes[0].hlines(self.exp_ne,3,6,linestyle='dotted')

        axes[1].plot(self.time_list, self.nel/self.ne_length, c=color_l)
        axes[1].set_ylabel(r'$\bar{n}_e[\times 10^{19} \mathrm{m}^{-3}]$',fontsize=12)
        ax1_2 = axes[1].twinx()
        ax1_2.plot(self.time_list, self.wpdia, c=color_r)
        
        ax1_2.set_ylabel(r'$W_p$[MJ]',fontsize=14)

        Pinput = self.ech+self.nbi_tan+self.nbi_perp*0.5
        axes[2].plot(self.time_list, self.prad/Pinput, c=color_l)
        axes[2].set_ylabel(r'$P_{rad}/P_{input}$',fontsize=14)
        axes[2].set_ylim(-0.1,0.7)
        ax2_2 = axes[2].twinx()
        ax2_2.plot(self.time_list, self.rax_vmec, c=color_r)
        ax2_2.set_ylabel(r'$R_{axis}^{vmec}$[m]',fontsize=14)
        ax2_2.hlines(3.66,3,6,linestyle='dotted')
        ax2_2.hlines(3.67,3,6,linestyle='dotted')
        ax2_2.set_ylim(3.635,3.72)

        axes[3].plot(self.time_list, self.CIII/(self.nel/self.ne_length),c=color_l)
        axes[3].set_ylabel(r'CIII/$\bar{n}_e$',fontsize=14)
        ax3_2 = axes[3].twinx()
        ax3_2.plot(self.time_list, self.OVI/(self.nel/self.ne_length),c=color_r)
        ax3_2.set_ylabel(r'OVI/$\bar{n}_e$',fontsize=14)
        axes[3].set_ylim(-0.1,1.6)
        ax3_2.set_ylim(-0.1,1.6)

        axes[4].plot(self.time_list, self.Te_center,c=color_l)
        axes[4].set_ylabel(r'$T_e^{center}$[keV]',fontsize=14)
        ax4_2 = axes[4].twinx()
        ax4_2.plot(self.time_list, self.ne_center,c=color_r)
        ax4_2.set_ylabel(r'$n_e[\times 10^{19} \mathrm{m}^{-3}]$'+'\n'+r'$@100\mathrm{eV}$',fontsize=14)

        for i in range(len(axes)):
            ax = axes[i]
            ylim = ax.get_ylim()
            if self.type == 1:
                # old:
                # self.time_range = [time_15 - 0.1, time_15, time_2, time_base]
                # new:
                # self.time_range = [ne_start, ne_end, po_start, po_end]
                ax.vlines(self.time_range[3], ylim[0],ylim[1])
                #ax.axvspan(time_2, time_3, color = "orangered", alpha = 0.5)
                ax.axvspan(self.time_range[2], self.time_range[3], color = "orangered", alpha = 0.5)
                ax.axvspan(self.time_range[0], self.time_range[1], color = "royalblue", alpha = 0.5)
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(self.time_range[1]-0.3, self.time_range[3]+0.1)
            elif self.type == -1:
                ax.vlines(5, ylim[0],ylim[1])
                ax.vlines(3.7, ylim[0],ylim[1])
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(3.4,5.6)

            if i != len(axes)-1:
                ax.set_xticklabels('')
            # import pdb; pdb.set_trace()
            if i == 0:
                if self.remark == '':
                    ax.set_title("#%d(%s)" %(self.shotNO, self.type_str))
                else:
                    ax.set_title("#%d(%s): %s" %(self.shotNO, self.type_str, self.remark))

        axes[-1].set_xlabel("time[s]",fontsize=14)

        plt.subplots_adjust(left=0.15,right=0.85,bottom=0.08,top=0.92,hspace=0)
        #plt.show()

    def graph_ES2(self,save=0):
        fig = plt.figure(figsize=(6,6))
        axes = [fig.add_subplot(4,1,i+1) for i in range(4)]

        plt.rcParams['lines.linewidth'] = 2
        cmap = plt.get_cmap("tab10")
        color_l = '#0079C4'
        color_r = '#E00059'

        axes[0].plot(self.time_list, self.exp_conv , c=color_l)
        axes[0].set_ylim(-0.1,5)
        axes[0].set_ylabel('Density'+'\n'+'exponents'+'\n'+'(smoothed)',fontsize=14)
        axes[0].hlines(self.exp_po,3,6,linestyle='dotted')
        axes[0].hlines(self.exp_ne,3,6,linestyle='dotted')

        ax1_2 = axes[1].twinx()
        ax1_2.plot(self.time_list, self.wpdia, c=color_r, zorder=2)
        axes[1].plot(self.time_list, self.nel/self.ne_length, c=color_l, zorder=1)
        axes[1].set_ylabel(r'$\bar{n}_e[\times 10^{19} \mathrm{m}^{-3}]$',fontsize=12)


        ax1_2.set_ylabel(r'$W_p$[MJ]',fontsize=14)


        axes[2].plot(self.time_list, self.OVI/(self.nel/self.ne_length),c=color_l)
        axes[2].set_ylabel(r'OVI/$\bar{n}_e$',fontsize=14)
        axes[2].hlines(0.2,3,6,linestyle='dotted')
        axes[2].set_ylim(-0.1,0.6)


        axes[3].plot(self.time_list, self.Te_center,c=color_l)
        axes[3].set_ylabel(r'$T_e^{center}$[keV]',fontsize=14)
        axes[3].hlines(1.5,3,6,linestyle='dotted')
        axes[3].set_ylim(-0.1,2.6)

        for i in range(len(axes)):
            ax = axes[i]
            ylim = ax.get_ylim()
            if self.type == 1:
                # old:
                # self.time_range = [time_15 - 0.1, time_15, time_2, time_base]
                # new:
                # self.time_range = [ne_start, ne_end, po_start, po_end]
                ax.vlines(self.time_range[3], ylim[0],ylim[1])
                #ax.axvspan(time_2, time_3, color = "orangered", alpha = 0.5)
                ax.axvspan(self.time_range[2], self.time_range[3], color = "orangered", alpha = 0.5)
                ax.axvspan(self.time_range[0], self.time_range[1], color = "royalblue", alpha = 0.5)
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(self.time_range[1]-0.3, self.time_range[3]+0.1)
            elif self.type == -1:
                ax.vlines(5, ylim[0],ylim[1])
                ax.vlines(3.7, ylim[0],ylim[1])
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(3.4,5.6)

            if i != len(axes)-1:
                ax.set_xticklabels('')
            # import pdb; pdb.set_trace()
            if i == 0:
                if self.remark == '':
                    ax.set_title("#%d(%s)" %(self.shotNO, self.type_str))
                else:
                    ax.set_title("#%d(%s): %s" %(self.shotNO, self.type_str, self.remark))

        axes[-1].set_xlabel("time[s]",fontsize=14)

        plt.subplots_adjust(left=0.15,right=0.85,bottom=0.08,top=0.92,hspace=0)
        #plt.show()

    def graph_single(self,save=0):
        fig = plt.figure(figsize=(5,2))
        ax = fig.add_subplot(1,1,1)

        plt.rcParams['lines.linewidth'] = 2
        cmap = plt.get_cmap("tab10")
        color_l = '#0079C4'
        color_r = '#E00059'

        ax.plot(self.time_list, self.nel/self.ne_length, c=color_l)
        ax.set_ylabel(r'$\bar{n}_e[\times 10^{19} \mathrm{m}^{-3}]$',fontsize=14)
        ax_2 = ax.twinx()
        ax_2.plot(self.time_list, self.wpdia, c=color_r)
        ax_2.set_ylabel(r'$W_p$[MJ]',fontsize=14)

        ylim = ax.get_ylim()
        if self.type == 1:
            # old:
            # self.time_range = [time_15 - 0.1, time_15, time_2, time_base]
            # new:
            # self.time_range = [ne_start, ne_end, po_start, po_end]
            # ax.vlines(self.time_range[3], ylim[0],ylim[1])
            #ax.axvspan(time_2, time_3, color = "orangered", alpha = 0.5)
            # ax.axvspan(self.time_range[2], self.time_range[3], color = "orangered", alpha = 0.5)
            # ax.axvspan(self.time_range[0], self.time_range[1], color = "royalblue", alpha = 0.5)
            ax.set_ylim(ylim[0],ylim[1])
            ax.set_xlim(self.time_range[1]-0.3, self.time_range[3]+0.1)
        elif self.type == -1:
            ax.vlines(5, ylim[0],ylim[1])
            ax.vlines(3.7, ylim[0],ylim[1])
            ax.set_ylim(ylim[0],ylim[1])
            ax.set_xlim(3.4,5.6)

            # import pdb; pdb.set_trace()
            # if i == 0:
            #     if self.remark == '':
            #         ax.set_title("#%d(%s)" %(self.shotNO, self.type_str))
            #     else:
            #         ax.set_title("#%d(%s): %s" %(self.shotNO, self.type_str, self.remark))

        ax.set_xlabel("time[s]",fontsize=14)

        plt.subplots_adjust(left=0.15,right=0.85,bottom=0.24,top=0.92,hspace=0)
        #plt.show()

    def plot_labels(self,save=0):
        remarks = self.remark
        about = self.about
        exp_conv = self.exp_conv
        fig = plt.figure(figsize=(10,12))
        axes = [fig.add_subplot(13,1,i+1) for i in range(13)]

        # axes[0].plot(self.time_list, exp_conv ,'.-')
        # axes[0].set_ylim(-0.1,5)
        # axes[0].set_ylabel('M-P exponents',fontsize=14)
        # axes[0].hlines(self.exp_po,3,6,linestyle='dotted')
        # axes[0].hlines(self.exp_ne,3,6,linestyle='dotted')

        axes[0].plot(self.time_list, self.wpdia)
        # axes[0].set_ylim(-0.1,5)
        axes[0].set_ylabel('$W_{p}$[Mj]',fontsize=14)

        axes[1].plot(self.time_list, self.nel/self.ne_length, '.-')
        axes[1].plot(self.time_list, self.nel_thomson/self.ne_length, 'x-')
        axes[1].set_ylabel(r'$\bar{n}_e[10^{19} \mathrm{m}^{-3}]$',fontsize=12)

        # HEATINGの行
        # self.get_ECH()
        # self.get_nbi()
        # self.get_wp()
        axes[2].plot(self.time_list, self.nbi_tan + 0.5*self.nbi_perp, '.-', label='Total NBI')
        axes[2].plot(self.time_list, self.ech, '.-', label='Total ECH')
        ax2_2 = axes[2].twinx()
        ax2_2.plot(self.time_list, self.wpdia, 'g.-', label='Wp_dia')
        ax2_2.plot(self.time_list, self.Wp_iss, 'k--', label='Wp_ISS')
        ax2_2.plot(self.time_list, self.Wp_iss/3, 'k:', label='30%')
        axes[2].set_ylabel(r'$Power[\mathrm{MW}]$',fontsize=14)
        ax2_2.set_ylabel(r'$W_p$[MJ]',fontsize=14)
        h1, l1 = axes[2].get_legend_handles_labels()
        h2, l2 = ax2_2.get_legend_handles_labels()
        axes[2].legend(h1+h2, l1+l2, loc='upper right')
        #ax2_2.legend()


        axes[3].plot(self.time_list, self.prad, '.-')
        axes[3].set_ylabel(r'$P_{rad}$',fontsize=14)
        # axes[3].plot(self.time_list, self.prad/(self.nel/self.ne_length), '.-')
        # axes[3].set_ylabel(r'$P_{rad}/\bar{n}_e$',fontsize=14)

        # self.get_imp()
        axes[4].plot(self.time_list, self.OV/(self.nel/self.ne_length), '.-', label='OV/nel')
        axes[4].plot(self.time_list, self.CIII/(self.nel/self.ne_length), '.-', label='CIII/nel')
        axes[4].plot(self.time_list, self.OVI/(self.nel/self.ne_length), '.-', label='OVI/nel')
        axes[4].plot(self.time_list, self.CIV/(self.nel/self.ne_length), '.-', label='CIV/nel')
        axes[4].legend(loc='upper right')
        axes[4].set_xlabel('time[s]',fontsize=14)
        axes[4].set_ylabel(r'Imp/$\bar{n}_e$',fontsize=14)
        axes[4].set_ylim(-0.1,2)
        ax4_2 = axes[4].twinx()
        ax4_2.plot(self.time_list, self.ha/(self.nel/self.ne_length), 'k.-', label='Ha/nel')
        axes[4].set_ylabel(r'$H_\alpha/\bar{n}_e$',fontsize=14)

        '''
        axes[5].plot(self.time_list, self.FIG, '.-', label='FIG')
        axes[5].plot(self.time_list, self.Pcc, '.-', label='Pcc')
        ax5_2 = axes[5].twinx()
        ax5_2.plot(self.time_list, self.Isat, 'g.-', label='Isat')
        axes[5].set_ylabel(r'$P_{neutral}$[Pa]',fontsize=14)
        ax5_2.set_ylabel(r'$I_{sat}$[A]',fontsize=14)
        axes[5].set_ylim([-0.001, 0.011])
        h1, l1 = axes[5].get_legend_handles_labels()
        h2, l2 = ax5_2.get_legend_handles_labels()
        axes[5].legend(h1+h2, l1+l2, loc='upper right')
        #ax5_2.legend()
        '''

        self.time_list1 = self.time_list[self.Isat_7L>0]
        self.time_list2 = self.time_list[self.Isat_4R>0]
        self.Isat_7L1 = self.Isat_7L[self.Isat_7L>0]
        self.Isat_4R2 = self.Isat_4R[self.Isat_4R>0]

        axes[5].plot(self.time_list, self.Isat_6L)
        axes[5].plot(self.time_list2, self.Isat_4R2)
        axes[5].set_ylabel(r'$Isat_{7L}$[A]',fontsize=14)

        axes[6].plot(self.time_list, self.type_list)
        axes[6].set_ylabel(r'type',fontsize=14)

        axes[7].plot(self.time_list, self.Te_edge)
        axes[7].set_ylabel(r'$Te@edge$',fontsize=14)

        axes[8].plot(self.time_list, self.reff100eV, '.-', label='reff[m]@100eV')
        axes[8].plot(self.time_list, self.dV100eV/100, '.-', label=r'dVdreff[$\times 100 \mathrm{m}^2$]@100eV')
        ax6_2 = axes[8].twinx()
        ax6_2.plot(self.time_list, self.Te_center, 'g.-', label='Te[keV]@center')
        axes[8].set_ylabel(r'reff & dVdreff',fontsize=14)
        ax6_2.set_ylabel(r'$Te$[keV]',fontsize=14)
        h1, l1 = axes[8].get_legend_handles_labels()
        h2, l2 = ax6_2.get_legend_handles_labels()
        axes[8].legend(h1+h2, l1+l2, loc='upper right')

        axes[9].plot(self.time_list, self.ne100eV, '.-', label='ne@100eV')
        axes[9].plot(self.time_list, self.ne_center, '.-', label='ne@center')
        axes[9].plot(self.time_list, self.ne_edge, '.-', label='ne@edge')
        axes[9].set_ylabel(r'$n_e[\times 10^{19} \mathrm{m}^{-3}]$',fontsize=12)
        axes[9].legend(loc='upper right')

        # axes[10].plot(self.time_list, self.CIII)
        # axes[10].plot(self.time_list, self.CIII, label='CIII')
        # axes[10].plot(self.time_list, self.OV, label='OV')
        # axes[10].plot(self.time_list, self.OVI, label='OVI')
        # axes[10].set_ylabel(r'$\mathrm{CIII}$',fontsize=14)
        # axes[10].set_ylabel(r'$H_\alpha$',fontsize=14)
        # axes[10].legend(loc='upper right')

        axes[10].plot(self.time_list, self.SDLloop_dphi, label='dphi')
        axes[10].set_ylabel(r'$\Delta\Phi_{eff}$',fontsize=12)
        # axes[11].plot(self.time_list, self.SDLloop_dtheta, label='dtheta')
        axes[11].plot(self.time_list, self.CIII, label='dtheta')
        axes[11].set_ylabel(r'$\mathrm{CIII}$',fontsize=12)
        # axes[12].plot(self.time_list, self.ne_soxmos, label='Ne')
        # axes[12].plot(self.time_list, self.ar_soxmos, label='Ar')
        axes[12].set_ylabel(r'$\mathrm{soxmos}$',fontsize=12)
        axes[12].legend(loc='upper right')

        for i in range(len(axes)):
            ax = axes[i]
            ylim = ax.get_ylim()
            if self.type == 1:
                # old:
                # self.time_range = [time_15 - 0.1, time_15, time_2, time_base]
                # new:
                # self.time_range = [ne_start, ne_end, po_start, po_end]
                ax.vlines(self.time_range[3], ylim[0],ylim[1])
                #ax.axvspan(time_2, time_3, color = "orangered", alpha = 0.5)
                ax.axvspan(self.time_range[2], self.time_range[3], color = "orangered", alpha = 0.5)
                ax.axvspan(self.time_range[0], self.time_range[1], color = "royalblue", alpha = 0.5)
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(self.time_range[1]-0.3, self.time_range[3]+0.3)
            elif self.type == -1:
                ax.vlines((4,5), ylim[0],ylim[1])
                ax.set_ylim(ylim[0],ylim[1])
                ax.set_xlim(3.4,5.6)

            if i != len(axes)-1:
                ax.set_xticklabels('')

            if i == 0:
                if remarks == '':
                    ax.set_title("shotNO=%d, type=%s, include=%d" %(self.shotNO, self.type_str, self.label))
                else:
                    ax.set_title("shotNO=%d, type=%s, remarks:%s, include=%d" %(self.shotNO, self.type_str, remarks,  self.label))

        axes[-1].set_xlabel("time[s]",fontsize=14)

        plt.subplots_adjust(hspace=0,bottom=0.05, top=0.95)
        # plt.show()
        if save == 0:
            plt.show()
        else:
            plt.savefig('./datapng/data_'+str(self.shotNO)+'_for24'+'.png')
            # plt.savefig('./EXP_png0703/exp_'+str(self.shotNO)+'.png')
        plt.close()

    def make_dataset(self):
        #pdb.set_trace()

        if self.type == 1:
            #po: positive_class
            args_po = np.logical_and(self.time_list >= self.time_range[2], self.time_list <= self.time_range[3])
            times_po = self.time_list[args_po]
            types_po = np.ones_like(times_po)
            labels_po = np.ones_like(times_po) * self.label
            shotNO_po = np.ones_like(times_po) * self.shotNO
            nel_po = self.nel[args_po]/self.ne_length
            B_po = np.ones_like(times_po) * np.abs(self.Bt)
            Pech_po = self.ech[args_po]
            Ptan_po = self.nbi_tan[args_po]
            Pperp_po = self.nbi_perp[args_po]
            Pinput_po = self.ech[args_po]+self.nbi_tan[args_po]+self.nbi_perp[args_po]*0.5
            PinputNEW_po = self.ech[args_po]+self.nbi_tan[args_po]+self.nbi_perp[args_po]*0.36
            Prad_po = self.prad[args_po]
            Prad_norm_po = Prad_po / Pinput_po
            Wp_po = self.wpdia[args_po]
            beta_po = self.beta[args_po]
            Rax_po = np.ones_like(times_po) * np.abs(self.Rax)
            #center_po = self.geom_center[args_po]
            center_po = self.rax_vmec[args_po]
            a99_po = self.a99[args_po]
            sh_shift_po = self.sh_shift[args_po]
            OV_po = self.OV[args_po]/nel_po
            CIII_po = self.CIII[args_po]/nel_po
            OVI_po = self.OVI[args_po]/nel_po
            CIV_po = self.CIV[args_po]/nel_po
            FeXVI_po = self.FeXVI[args_po]/nel_po
            HI_po = self.HI[args_po]/nel_po
            HeI_po = self.HeI[args_po]/nel_po
            Ha_po = self.ha[args_po]/nel_po
            dh_po = self.dh[args_po]
            he_ratio_po = self.he_ratio[args_po]
            d_ratio_po = self.d_ratio[args_po]
            h_ratio_po = self.h_ratio[args_po]
            M_eff_po = self.M_eff[args_po]
            Z_eff_po = self.Z_eff[args_po]
            Ip_po = self.Ip[args_po]
            FIG_po = self.FIG[args_po]
            Pcc_po = self.Pcc[args_po]
            Isat_po = self.Isat[args_po]

            reff100_po = self.reff100eV[args_po]
            ne100_po = self.ne100eV[args_po]
            dV100_po = self.dV100eV[args_po]
            Te_center_po = self.Te_center[args_po]
            Te_edge_po = self.Te_edge[args_po]
            ne_center_po = self.ne_center[args_po]
            ne_peak_po = nel_po/self.ne_edge[args_po]

            save_po = np.vstack((
                shotNO_po, times_po, types_po, labels_po,
                nel_po, B_po, Pech_po, Ptan_po, Pperp_po, Pinput_po, PinputNEW_po, Prad_po, Prad_norm_po, Wp_po, beta_po,
                Rax_po, center_po, a99_po, sh_shift_po,
                #OV_po, CIII_po, OVI_po, CIV_po, FeXVI_po, HI_po,
                HI_po, HeI_po, Ha_po, dh_po,
                he_ratio_po, d_ratio_po, h_ratio_po, 
                M_eff_po,Z_eff_po,
                CIII_po, CIV_po, OV_po, OVI_po,  FeXVI_po,
                Ip_po,
                FIG_po, Pcc_po, Isat_po,
                reff100_po, ne100_po, dV100_po, Te_center_po, Te_edge_po, ne_center_po, ne_peak_po
            ))

            #ne: negative_class
            args_ne = np.logical_and(self.time_list >= self.time_range[0], self.time_list <= self.time_range[1])
            times_ne = self.time_list[args_ne]
            types_ne = np.ones_like(times_ne) * (-1)
            labels_ne = np.ones_like(times_ne) * self.label
            shotNO_ne = np.ones_like(times_ne) * self.shotNO
            nel_ne = self.nel[args_ne]/self.ne_length
            B_ne = np.ones_like(times_ne) * np.abs(self.Bt)
            Pech_ne = self.ech[args_ne]
            Ptan_ne = self.nbi_tan[args_ne]
            Pperp_ne = self.nbi_perp[args_ne]
            Pinput_ne = self.ech[args_ne]+self.nbi_tan[args_ne]+self.nbi_perp[args_ne]*0.5
            PinputNEW_ne = self.ech[args_ne]+self.nbi_tan[args_ne]+self.nbi_perp[args_ne]*0.36
            Prad_ne = self.prad[args_ne]
            Prad_norm_ne = Prad_ne / Pinput_ne
            Wp_ne = self.wpdia[args_ne]
            beta_ne = self.beta[args_ne]
            Rax_ne = np.ones_like(times_ne) * np.abs(self.Rax)
            # center_ne = self.geom_center[args_ne]
            center_ne = self.rax_vmec[args_ne]
            a99_ne = self.a99[args_ne]
            sh_shift_ne = self.sh_shift[args_ne]
            OV_ne = self.OV[args_ne]/nel_ne
            CIII_ne = self.CIII[args_ne]/nel_ne
            OVI_ne = self.OVI[args_ne]/nel_ne
            CIV_ne = self.CIV[args_ne]/nel_ne
            FeXVI_ne = self.FeXVI[args_ne]/nel_ne
            HI_ne = self.HI[args_ne]/nel_ne
            HeI_ne = self.HeI[args_ne]/nel_ne
            Ha_ne = self.ha[args_ne]/nel_ne
            dh_ne = self.dh[args_ne]
            he_ratio_ne = self.he_ratio[args_ne]
            d_ratio_ne = self.d_ratio[args_ne]
            h_ratio_ne = self.h_ratio[args_ne]
            M_eff_ne = self.M_eff[args_ne]
            Z_eff_ne = self.Z_eff[args_ne]
            Ip_ne = self.Ip[args_ne]
            FIG_ne = self.FIG[args_ne]
            Pcc_ne = self.Pcc[args_ne]
            Isat_ne = self.Isat[args_ne]

            reff100_ne = self.reff100eV[args_ne]
            ne100_ne = self.ne100eV[args_ne]
            dV100_ne = self.dV100eV[args_ne]
            Te_center_ne = self.Te_center[args_ne]
            Te_edge_ne = self.Te_edge[args_ne]
            ne_center_ne = self.ne_center[args_ne]
            ne_peak_ne = nel_ne/self.ne_edge[args_ne]

            save_ne = np.vstack((
                shotNO_ne, times_ne, types_ne, labels_ne,
                nel_ne, B_ne, Pech_ne, Ptan_ne, Pperp_ne, Pinput_ne, PinputNEW_ne, Prad_ne, Prad_norm_ne, Wp_ne, beta_ne,
                Rax_ne, center_ne, a99_ne, sh_shift_ne,
                # OV_ne, CIII_ne, OVI_ne, CIV_ne, FeXVI_ne, HI_ne,
                HI_ne,HeI_ne, Ha_ne, dh_ne,
                he_ratio_ne, d_ratio_ne, h_ratio_ne, 
                M_eff_ne, Z_eff_ne,
                CIII_ne, CIV_ne, OV_ne, OVI_ne,  FeXVI_ne,
                Ip_ne,
                FIG_ne, Pcc_ne, Isat_ne,
                reff100_ne, ne100_ne, dV100_ne, Te_center_ne,Te_edge_ne, ne_center_ne, ne_peak_ne
            ))

            savelines = np.vstack((save_po.T, save_ne.T))

        elif self.type == -1:
            # import pdb; pdb.set_trace()
            args_ne = np.logical_and(self.time_list >= self.time_range[0], self.time_list <= self.time_range[1])
            times_ne = self.time_list[args_ne]
            types_ne = np.ones_like(times_ne) * (-1)
            labels_ne = np.ones_like(times_ne) * self.label
            shotNO_ne = np.ones_like(times_ne) * self.shotNO
            nel_ne = self.nel[args_ne]/self.ne_length
            B_ne = np.ones_like(times_ne) * np.abs(self.Bt)
            Pech_ne = self.ech[args_ne]
            Ptan_ne = self.nbi_tan[args_ne]
            Pperp_ne = self.nbi_perp[args_ne]
            Pinput_ne = self.ech[args_ne]+self.nbi_tan[args_ne]+self.nbi_perp[args_ne]*0.5
            PinputNEW_ne = self.ech[args_ne]+self.nbi_tan[args_ne]+self.nbi_perp[args_ne]*0.36
            Prad_ne = self.prad[args_ne]
            Prad_norm_ne = Prad_ne / Pinput_ne
            Wp_ne = self.wpdia[args_ne]
            beta_ne = self.beta[args_ne]
            Rax_ne = np.ones_like(times_ne) * np.abs(self.Rax)
            # center_ne = self.geom_center[args_ne]
            center_ne = self.rax_vmec[args_ne]
            a99_ne = self.a99[args_ne]
            sh_shift_ne = self.sh_shift[args_ne]
            OV_ne = self.OV[args_ne]/nel_ne
            CIII_ne = self.CIII[args_ne]/nel_ne
            OVI_ne = self.OVI[args_ne]/nel_ne
            CIV_ne = self.CIV[args_ne]/nel_ne
            FeXVI_ne = self.FeXVI[args_ne]/nel_ne
            HI_ne = self.HI[args_ne]/nel_ne
            HeI_ne = self.HI[args_ne]/nel_ne
            Ha_ne = self.ha[args_ne]/nel_ne
            dh_ne = self.dh[args_ne]
            he_ratio_ne = self.he_ratio[args_ne]
            d_ratio_ne = self.d_ratio[args_ne]
            h_ratio_ne = self.h_ratio[args_ne]
            M_eff_ne = self.M_eff[args_ne]
            Z_eff_ne = self.Z_eff[args_ne]
            Ip_ne = self.Ip[args_ne]
            FIG_ne = self.FIG[args_ne]
            Pcc_ne = self.Pcc[args_ne]
            Isat_ne = self.Isat[args_ne]

            reff100_ne = self.reff100eV[args_ne]
            ne100_ne = self.ne100eV[args_ne]
            dV100_ne = self.dV100eV[args_ne]
            Te_center_ne = self.Te_center[args_ne]
            Te_edge_ne = self.Te_edge[args_ne]
            ne_center_ne = self.ne_center[args_ne]
            ne_peak_ne = nel_ne/self.ne_edge[args_ne]

            save_ne = np.vstack((
                shotNO_ne, times_ne, types_ne, labels_ne,
                nel_ne, B_ne, Pech_ne, Ptan_ne, Pperp_ne, Pinput_ne, PinputNEW_ne, Prad_ne, Prad_norm_ne, Wp_ne, beta_ne,
                Rax_ne, center_ne, a99_ne,sh_shift_ne,
                # OV_ne, CIII_ne, OVI_ne, CIV_ne, FeXVI_ne, HI_ne,
                HI_ne,HeI_ne, Ha_ne, dh_ne,
                he_ratio_ne, d_ratio_ne, h_ratio_ne, 
                M_eff_ne, Z_eff_ne,
                CIII_ne, CIV_ne, OV_ne, OVI_ne,  FeXVI_ne,
                Ip_ne,
                FIG_ne, Pcc_ne, Isat_ne,
                reff100_ne, ne100_ne, dV100_ne, Te_center_ne, Te_edge_ne, ne_center_ne, ne_peak_ne
            ))
            savelines = save_ne.T

        # import pdb; pdb.set_trace()
        with open(self.savename, 'a') as f_handle:
            np.savetxt(f_handle, savelines, delimiter=',',fmt='%.5e')
    

def graph(shotNO_str,labelname='labels.csv'):
    shotNOs = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=0, dtype=int)
    types = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=1, dtype=str)
    labels = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=2, dtype=int)
    remarks = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=3, dtype=str)
    abouts = np.genfromtxt(labelname,delimiter=',',skip_header=1, usecols=4,  dtype=float)

    shotNO = int(shotNO_str)
    # import pdb; pdb.set_trace()
    type = types[shotNOs == shotNO][0]
    label = labels[shotNOs == shotNO][0]
    remark = remarks[shotNOs == shotNO][0]
    about = abouts[shotNOs == shotNO][0]
    nel_data = CalcMPEXP(shotNO,type, label, remark,about)
    nel_data.main()
    # import pdb; pdb.set_trace()
    # nel_data.time_accumulate()
    
    nel_data.graph_2021()
    # nel_data.graph()
    # nel_data.graph_single()
    plt.show()
    # nel_data.graph_ES2()
    # plt.show()
    nel_data.remove_files()


def main_labels(savename='dataset.csv',labelname='labels.csv',ion=None):
    print(savename)
    print(labelname)
    shotNOs = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=0, dtype=int)
    types = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=1, dtype=str)
    labels = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=2, dtype=int)
    remarks = np.genfromtxt(labelname,delimiter=',',skip_header=1,usecols=3, dtype=str)
    about = np.genfromtxt(labelname,delimiter=',',skip_header=1, usecols=4,  dtype=float)
    if ion:
        # import pdb; pdb.set_trace()
        ion_list = np.genfromtxt(labelname,delimiter=',',skip_header=1, usecols=5,  dtype=str)
#    print(about)

    with open(savename, 'w') as f_handle:
        header = ['shotNO', 'times', 'types', 'labels',
                'nel','B', 'Pech', 'Pnbi-tan', 'Pnbi-perp', 'Pinput', 'PinputNEW', 'Prad', 'Prad/Pinput', 'Wp','beta',
                'Rax', 'rax_vmec', 'a99', 'delta_sh',
                # 'OV', 'CIII', 'OVI', 'CIV', 'FeXVI', 'HI',
                'HI','HeI','Ha','D/(H+D)',
                'He/(H+D+He)','D/(H+D+He)','H/(H+D+He)',
                'M_eff','Z_eff',
                'CIII', 'CIV', 'OV', 'OVI', 'FeXVI',
                'Ip',
                'FIG', 'Pcc', 'Isat',
                'reff@100eV', 'ne@100eV', 'dVdreff@100eV',
                'Te@center','Te@edge', 'ne@center', 'ne_peak'
        ]
        f_handle.write(', '.join(header)+'\n')
        # shotNO_ne, times_ne, types_ne, labels_ne,
        # nel_ne, B_ne, Pech_ne, Pnbi_ne, Pinput_ne, Prad_ne,
        # Rax_ne, center_ne, a99_ne,
        # OV_ne, CIII_ne, OVI_ne, CIV_ne
    
    with open('errorshot.txt', mode='a') as f:
        datetime.datetime.today().strftime("\n%Y/%m/%d")

    for i in range(len(shotNOs)):
        if ion:
            if ion_list[i] != ion:
                continue
        shotNO = shotNOs[i]
        print(shotNO)
        if shotNO in [147071, 147072, 147073]:
            print("NO DATA")
            continue
        nel_data = CalcMPEXP(shotNO,types[i], labels[i], remarks[i],about[i],savename=savename)
        nel_data.remove_files()
        main_return = nel_data.main()
        if  main_return == -1:
            nel_data.remove_files()
            continue
        elif main_return == 'MPexp error':
            with open('errorshot.txt', mode='a') as f:
                f.write('\n'+str(shotNO))
            nel_data.remove_files()
            continue
        # nel_data.get_geom()
        # import pdb; pdb.set_trace()
        nel_data.make_dataset()
        nel_data.plot_labels(save=1)
        nel_data.remove_files()
    return



if __name__ == '__main__':
    # test()
    # main()
    args = sys.argv
    if len(args) > 1:
        # graph(args[1])        
        graph(args[1],'labels_all_prad.csv')
        # graph(args[1],'labels_all_210113.csv')
        # graph(args[1],'labels_210113.csv')
        # graph(args[1],'labels.csv')
    else:
        # Usage:
        main_labels(dataset_name,label_csv_name)
        # main_labels('dataset_210113.csv','labels_210113.csv')
        # main_labels('dataset_191114.csv','labels_191114.csv')
        # main_labels('dataset.csv','labels.csv')
        # main_labels('dataset_200116_low.csv','labels_200116_low.csv')
        # main_labels('dataset_200116_high.csv','labels_200116_high.csv')
        # main_labels('dataset_combined.csv','labels_combined.csv')

        # main_labels('dataset_2017.csv','labels_2017.csv')

        # main_labels_ece('dataset_all_ece.csv','labels_all.csv')
        # main_labels('dataset_all.csv','labels_all.csv')
        # main_labels('dataset_all_210113.csv','labels_all_210113.csv')
        # main_labels('dataset_all_prad.csv','labels_all_prad.csv')
        
        # main_labels('dataset_H_prad.csv','labels_all_prad.csv',ion='H')
        # main_labels('dataset_D_prad.csv','labels_all_prad.csv',ion='D')
        # main_labels('dataset_He_prad.csv','labels_all_prad.csv',ion='He')