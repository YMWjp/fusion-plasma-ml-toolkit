import os
import sys
import glob
import zipfile
import pdb

import shutil

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

import pdb

import shutil

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '../../Kaiseki'))
from egdb_class import *

class eg3d_read():
    def __init__(self, dataname):
        self.eg = egdb3d(dataname)
        self.eg.readFile()

        self.time = self.eg.dims(0)
        self.R = self.eg.dims(1)

        self.data = np.array(self.eg.data).reshape(-1,self.eg.ValNo).T
        # usage: data[eg.valname2idx('reff')]

    def eg_certain_time(self, time):
        start = self.eg.value2idx(time, 0) * self.eg.DimSizes[1]
        end = start + self.eg.DimSizes[1]
        return start, end

    def eg_f1(self, data, timelist):
        f1 = interpolate.interp1d(self.time, data, bounds_error=False,fill_value=0)
        return f1(timelist)

    def Te_position(self,t,target=10):
        t_range = self.eg_certain_time(t)
        Te_fit = self.data[self.eg.valname2idx('Te_fit')]*1000 #eV
        R = np.array(self.R)*100
        
        Te_t = Te_fit[t_range[0]:t_range[1]][R>400]
        f = interpolate.interp1d(Te_t, R[R>400], bounds_error=False,fill_value=0)
        return f(target)
    
    def EG_2D(self,valname):
        data = self.data[self.eg.valname2idx(valname)]
        # import pdb; pdb.set_trace()
        return data.reshape(len(self.time),len(self.R))


class TsmapCalib(eg3d_read):
    def __init__(self, dataname):
        super().__init__(dataname)
        self.comments = self.eg.comments.split('\n# ')[1:-1]
        self.phiedge = 0
        self.R_separatrix = 0
        self.R_sep_outer = 0

        self.Te_outer = []
        self.ne_outer = []


    def ne_from_Te(self, Te_target):
        reff_list = np.zeros_like(self.time)
        ne_list = np.zeros_like(self.time)
        dVdreff_list = np.zeros_like(self.time)
        for i,t in enumerate(self.time):
            start, end = self.eg_certain_time(t)
            reff_tmp = self.data[self.eg.valname2idx('reff')][start: end]
            Te_tmp = self.data[self.eg.valname2idx('Te_fit')][start: end]
            ne_tmp = self.data[self.eg.valname2idx('ne_fit')][start: end]
            dV_tmp = self.data[self.eg.valname2idx('dVdreff')][start: end]
            reff = reff_tmp[reff_tmp>=0]
            Te = Te_tmp[reff_tmp>=0]
            ne = ne_tmp[reff_tmp>=0]
            dV = dV_tmp[reff_tmp>=0]
            f1_Te = interpolate.interp1d(Te, reff, bounds_error=False,fill_value=0)
            f1_ne = interpolate.interp1d(reff, ne, bounds_error=False,fill_value=0)
            f1_dVdreff = interpolate.interp1d(reff, dV, bounds_error=False,fill_value=0)
            reff_list[i] = f1_Te(Te_target)
            ne_list[i] = f1_ne(reff_list[i])
            dVdreff_list[i] = f1_dVdreff(reff_list[i])
        return reff_list, ne_list, dVdreff_list

    def Te_from_reff(self,reff_target):
        Te_list = np.zeros_like(self.time)
        ne_list = np.zeros_like(self.time)

        for i,t in enumerate(self.time):
            start, end = self.eg_certain_time(t)
            reff = self.data[self.eg.valname2idx('reff')][start: end]
            Te = self.data[self.eg.valname2idx('Te_fit')][start: end]
            ne = self.data[self.eg.valname2idx('ne_fit')][start: end]
            f1_Te = interpolate.interp1d(reff, Te, bounds_error=False,fill_value=0)
            f1_ne = interpolate.interp1d(reff, ne, bounds_error=False,fill_value=0)
            if reff_target < np.max(reff):
                Te_list[i] = f1_Te(reff_target)
                ne_list[i] = f1_ne(reff_target)
            else:
                Te_list[i] = f1_Te(reff_target)
                ne_list[i] = f1_ne(reff_target)
        return Te_list, ne_list

    def edge(self):
        Te_list = np.zeros_like(self.time)
        ne_list = np.zeros_like(self.time)

        for i,t in enumerate(self.time):
            start, end = self.eg_certain_time(t)
            reff = self.data[self.eg.valname2idx('reff')][start: end]
            Te = self.data[self.eg.valname2idx('Te_fit')][start: end]
            ne = self.data[self.eg.valname2idx('ne_fit')][start: end]
            Te_list[i] = Te[np.argmax(reff)]
            ne_list[i] = ne[np.argmax(reff)]
        return Te_list, ne_list
    
    def phiEdge(self):
        phiedge = float([s.replace('phiedge = ','') for s in self.comments if s.startswith('phiedge')][0])
        self.phiedge = phiedge
        start, end = self.eg_certain_time(self.time[0])
        phi_t0 = self.data[self.eg.valname2idx('phi')][start: end]
        phitop = np.max(phi_t0[phi_t0<0])
        arg_phitop = np.argmin(np.abs(phi_t0 - phitop))
        phi_inner = phi_t0[0:arg_phitop]
        R_inner = self.R[0:arg_phitop]
        phi_outer = phi_t0[arg_phitop:]
        R_outer = self.R[arg_phitop:]
        # import pdb; pdb.set_trace()
        f_phi = interpolate.interp1d(phi_inner,R_inner, bounds_error=False,fill_value=0)
        R_separatrix = f_phi(phiedge)
        self.R_separatrix = R_separatrix

        f_phi = interpolate.interp1d(phi_outer,R_outer, bounds_error=False,fill_value=0)
        R_sep_outer = f_phi(phiedge)
        self.R_sep_outer = R_sep_outer
        # print(R_sep_outer)   
        # import pdb; pdb.set_trace()

        Te_list = np.zeros_like(self.time)
        ne_list = np.zeros_like(self.time)
        self.Te_outer = np.zeros_like(self.time)
        self.ne_outer = np.zeros_like(self.time)
        

        for i,t in enumerate(self.time):
            start, end = self.eg_certain_time(t)
            # reff = self.data[self.eg.valname2idx('reff')][start: end]
            Te = self.data[self.eg.valname2idx('Te_fit')][start: end]
            ne = self.data[self.eg.valname2idx('ne_fit')][start: end]
            f1_Te = interpolate.interp1d(self.R, Te, bounds_error=False,fill_value=0)
            f1_ne = interpolate.interp1d(self.R, ne, bounds_error=False,fill_value=0)
            Te_list[i] = f1_Te(R_separatrix)
            ne_list[i] = f1_ne(R_separatrix)
            self.Te_outer[i] = f1_Te(R_sep_outer)
            self.ne_outer[i] = f1_ne(R_sep_outer)
        return Te_list, ne_list

    def test_plot(self):
        fig, ax = plt.subplots()
        reff_100, ne_100, dV_100 = self.ne_from_Te(0.1)
        ax.plot(self.time, reff_100, label='reff[m]@100eV')
        ax.plot(self.time, ne_100, label='ne[e19m-3]@100eV')
        ax.plot(self.time, dV_100/100, label='dVdreff[100m2]@100eV/100')
        Te_center, ne_center = self.Te_from_reff(0)
        ax.plot(self.time, Te_center, label='Te[keV]@center')
        ax.plot(self.time, ne_center, label='ne[e19m-3]@center')
        plt.legend()
        plt.show()

    def profile(self,range):
        Te = np.empty((0,self.eg.getDimSize(1)), int)
        ne = np.empty((0,self.eg.getDimSize(1)), int)
        for i,t in enumerate(self.time):
            if t >= range[0] and t<= range[1]:
                start, end = self.eg_certain_time(t)
                Te_tmp = self.data[self.eg.valname2idx('Te_fit')][start: end]
                ne_tmp = self.data[self.eg.valname2idx('ne_fit')][start: end]
                # import pdb; pdb.set_trace()
                Te = np.append(Te, Te_tmp.reshape(1,-1), axis=0)
                ne = np.append(ne, ne_tmp.reshape(1,-1), axis=0)
        return Te, ne

class EceSlow(eg3d_read):
    def __init__(self, dataname):
        super().__init__(dataname)
        self.Te = self.data[self.eg.valname2idx('Te')]
        self.fece = self.data[self.eg.valname2idx('fece')]
        if 'ecemap' in dataname:
            diag = 1
        else:
            self.diag = self.data[self.eg.valname2idx('diag_number')]
            # radH: diag == 1
            # radL: diag == 2
    
    def get_te(self, ch, time_list):
        if np.any(self.diag == 1): #radH
            # import pdb; pdb.set_trace()
            Te = self.Te[
                np.logical_and(self.diag==1, self.fece==ch[0])
            ]
        elif np.any(self.diag == 2): #radL
            print('!radL!')
            Te = self.Te[
                np.logical_and(self.diag==2, self.fece==ch[1])
            ]
        else:
            Te = np.zeros_like(self.time)
        
        if len(Te) == 0:
            Te = np.zeros_like(self.time)
        f1 = interpolate.interp1d(
            self.time, Te,bounds_error=False,fill_value=0)
        return f1(time_list)

if __name__ == '__main__':
    args = sys.argv
    test = TsmapCalib('tsmap_calib@'+args[1]+'.dat')
    t=4.3
    Te_center, ne_center = test.Te_from_reff(0)
    Te, ne = test.phiEdge()

    fig,ax = plt.subplots(figsize=(4,2.5))
    start, end = test.eg_certain_time(t)
    reff_tmp = test.data[test.eg.valname2idx('reff')][start: end]
    Te_tmp = test.data[test.eg.valname2idx('Te_fit')][start: end]
    ax.plot(test.R,Te_tmp,label='separatrix',c='#E00059')
    # ax.vlines((test.R_separatrix,test.R_sep_outer),-0.1,1)
    # import pdb; pdb.set_trace()
    target_t = np.array(test.time)==t
    ylim=ax.get_ylim()
    ax.vlines((test.R_separatrix,test.R_sep_outer),ylim[0],(Te[target_t],test.Te_outer[target_t]),linestyle='dotted')
    ax.set_ylim(ylim)
    ax.set_title('shotNO='+args[1]+'  t='+str(t)+'[s]')
    ax.set_xlabel('R[m]')
    ax.set_ylabel('Te[keV]')
    plt.subplots_adjust(top=0.9,right=0.97,bottom=0.17,left=0.15)
    plt.show()
    # ax2 = ax.twinx()
    # ax2.plot(test.time,Te_center,label='center',c='orange')
    # ax.legend()
    # ax2.legend()
    # plt.show()

    # Te, ne = test.profile([3,10])
    # plt.contourf(Te.T)
    # plt.show()
