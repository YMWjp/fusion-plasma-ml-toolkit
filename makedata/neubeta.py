import numpy as np
import matplotlib.pyplot as plt
from getfile_dat import getfile_dat
import sys
from egdb_class import *
from scipy import interpolate



# def eg_certain_time(self, time):
#         start = self.eg.value2idx(time, 0) * self.eg.DimSizes[1]
#         end = start + self.eg.DimSizes[1]
#         return start, end

class TsmapRead():
    def __init__(self,shotNO,datapath='./egdata/'):
        self.shotNO = shotNO
        self.diagname = 'tsmap_calib'
        self.datapath = datapath

        self.time = []
        # self.time = list(range(3,8,0.1))
        self.R = []
        self.data = []
        self.egfile = []
        self.get_egdata()

    def get_egdata(self):
        getfile_dat(self.shotNO, self.diagname, self.datapath)
        filename = self.datapath + '{0}@{1:d}.dat'.format(self.diagname,self.shotNO)
        self.egfile = egdb3d(filename)
        self.egfile.readFile()
        # print(egfile)
        self.time = np.array(self.egfile.dimdata[0])
        self.R = np.array(self.egfile.dimdata[1])
        self.data = np.array(self.egfile.data).reshape(-1,self.egfile.ValNo).T
        # print(self.data.shape)
        return

    def get_data(self,valname,R):
        data = self.data[self.egfile.valname2idx(valname)]
        # print(len(data))
        data = data.reshape(len(self.time),len(self.R))
        # print(data)
        data = data.T
        # print(data[np.argmin(np.abs(self.R-R))])
        # return (data[np.argmin(np.abs(self.R-R))])
        return (data[np.argmin(np.abs(self.R-R))]+data[np.argmin(np.abs(self.R-R))+1])/2

class GiotaRead():
    def __init__(self,shotNO,datapath='./egdata/'):
        self.shotNO = shotNO
        self.diagname = 'giota'
        self.datapath = datapath

        self.time = []
        self.reffa99 = []
        self.data = []
        self.egfile = []
        self.get_egdata()

    def get_egdata(self):
        getfile_dat(self.shotNO, self.diagname, self.datapath)
        filename = self.datapath + '{0}@{1:d}.dat'.format(self.diagname,self.shotNO)
        self.egfile = egdb3d(filename)
        self.egfile.readFile()
        # print(egfile)
        self.time = np.array(self.egfile.dimdata[0])
        self.reffa99 = np.array(self.egfile.dimdata[1])
        # print(self.reffa99)
        self.data = np.array(self.egfile.data).reshape(-1,self.egfile.ValNo).T
        # print(self.data.shape)
        return

    def get_data(self,valname,reffa99):
        data = self.data[self.egfile.valname2idx(valname)]
        # print(len(data))
        data = data.reshape(len(self.time),len(self.reffa99))
        data = data.T
        return data[np.argmin(np.abs(self.reffa99-reffa99))]

class IotaGet:
    def get_betaiota(shotNO1,timelist,plot_R1=4.6):
        # shotNO1 = 152708
        # plot_R1 = 6.5
        TR1 = TsmapRead(shotNO1)
        GR1 = GiotaRead(shotNO1)
        beta = TR1.get_data('beta_e',plot_R1)
        # print(beta)
        reffa99 = TR1.get_data('reff/a99',plot_R1)
        te = TR1.get_data('Te_fit',plot_R1)
        ne = TR1.get_data('ne_fit',plot_R1)
        reff = TR1.get_data('reff',plot_R1)
        # print(reffa99)
        beta_f = interpolate.interp1d(TR1.time, beta, kind='linear', bounds_error=False,fill_value=0)
        beta_list = beta_f(timelist)
        reffa99_f = interpolate.interp1d(TR1.time, reffa99, bounds_error=False,fill_value=0)
        reffa99_list = reffa99_f(timelist)
        te_time = TR1.time
        # te_time = te_time[te > 0.01]
        # te = te[te > 0.01]
        # te_time = te_time[te < 7]
        # te = te[te < 7]
        te_f = interpolate.interp1d(te_time, te, bounds_error=False,fill_value=0)
        te_list = te_f(timelist)
        ne_f = interpolate.interp1d(TR1.time, ne, bounds_error=False,fill_value=0)
        ne_list = ne_f(timelist)
        reff_f = interpolate.interp1d(TR1.time, reff, bounds_error=False,fill_value=0)
        reff_list = reff_f(timelist)
        # plt.plot(TR1.time, reffa99)
        # plt.plot(timelist, reffa99_list)
        # print(reffa99_list)
        iota_list = np.zeros(len(timelist))
        j = 0
        for i in timelist:
            data = GR1.data[GR1.egfile.valname2idx('iota')]
            data = data.reshape(len(GR1.time),len(GR1.reffa99))
            # print(data)
            iota_list[j] = data[np.argmin(np.abs(GR1.time-i))][np.argmin(np.abs(GR1.reffa99-reffa99_list[j]))]
            j = j + 1

        # beta = beta[TR1.R>3.3]
        # iota_list = iota_list[TR1.R>3.3]
        # R = TR1.R
        # R = R[R>3.3]
        # beta = beta[R<4.5]
        # iota_list = iota_list[R<4.5]

        # print(len(iota_list))
        # print(len(reffa99_list))
        # print(iota_list)

        # plt.scatter(TR1.R,beta)
        # plt.scatter(GR1.reffa99,iota)
        # plt.scatter(TR1.R,iota_list)
        return np.array([beta_list, reffa99_list, iota_list, te_list, ne_list, reff_list])


if __name__=='__main__':
    # shotNO1 = 152708
    # shotNO1 = 148582
    shotNO1 = 174925
    TR1 = TsmapRead(shotNO1)
    time_list = np.arange(3,8,0.01)
    # print(time_list)

    iota = IotaGet.get_betaiota(shotNO1,time_list)[2]
    te = IotaGet.get_betaiota(shotNO1,time_list)[3]
    ne = IotaGet.get_betaiota(shotNO1,time_list)[4]
    reff = IotaGet.get_betaiota(shotNO1,time_list)[5]
    def fcol(iota,te,ne,reff,R):
        ln_lambda = 23-np.log((ne*10**13)**0.5/(te*10**3))#=10
        ve = 2.91*(10**7)*ne*ln_lambda/((te*10**3)**1.5)#=e+6
        et = reff/R#=0.1
        qR = R/iota#=4
        vth = 4.19*10**5*(te*10**3)**0.5
        vb = ve/((et**1.5)*(vth/qR))
        # print(vb)
        return vb

    fcol(iota,te,ne,reff,3.9)

    # print(get_betaiota(152708,6.5))
    # plt.scatter(IotaGet.get_betaiota(152708,time_list)[3],IotaGet.get_betaiota(152708,time_list)[4],color='blue')
    # plt.scatter(IotaGet.get_betaiota(152645,timelist)[0],IotaGet.get_betaiota(152645,timelist)[1],color='red')
    # print(IotaGet.get_betaiota(152708,time_list)[1])
    # print(IotaGet.get_betaiota(152708,time_list)[2])
    # print(IotaGet.get_betaiota(152678,time_list)[3])
    # print(IotaGet.get_betaiota(152708,time_list)[4])
    # print(IotaGet.get_betaiota(152708,time_list)[5])
    '''
    a = IotaGet.get_betaiota(152708,time_list)[5]
    a = a[time_list>=5.0]
    time_list = time_list[time_list>=5.0]
    a = a[time_list<=5.1]
    print(a)
    '''
    # print(TR1.get_data('Te',4.6))
    # print(TR1.get_data('beta_e',4.6))
    # plt.plot(time_list,te)
    # plt.plot(time_list,IotaGet.get_betaiota(152708,time_list)[0],color='red')
    # plt.plot(TR1.time,TR1.get_data('beta_e',4.6))
    # plt.plot(time_list,IotaGet.get_betaiota(174922,time_list)[3],color='red')
    plt.plot(TR1.time,TR1.get_data('Te',4.6))
    plt.plot(TR1.time,TR1.get_data('Te_fit',4.6))
    plt.ylabel('Te[keV]')
    plt.xlabel('Time[s]')
    # plt.xlim(3,5)
    # plt.ylim(-0.5,1)
    plt.ylim(0,0.5)
    plt.savefig('Te_plot_174923.png')
    plt.show()
