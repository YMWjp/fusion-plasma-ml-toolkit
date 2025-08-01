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

class TsmapSmoothRead():
    def __init__(self,shotNO,datapath='./egdata/'):
        self.shotNO = shotNO
        self.diagname = 'tsmap_smooth'
        self.datapath = datapath

        self.time = []
        # self.time = list(range(3,8,0.1))
        self.reff = []
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
        self.reff = np.array(self.egfile.dimdata[1])
        self.data = np.array(self.egfile.data).reshape(-1,self.egfile.ValNo).T
        # print(self.data.shape)
        return

    def get_data(self,valname,reff):
        data = self.data[self.egfile.valname2idx(valname)]
        # print(len(data))
        data = data.reshape(len(self.time),len(self.reff))
        # print(data)
        data = data.T
        # print(data[np.argmin(np.abs(self.R-R))])
        # return (data[np.argmin(np.abs(self.R-R))])
        return data[np.argmin(np.abs(self.reff-reff))]


class P0Get:
    def get_p0(shotNO,timelist,plot_reff=0):
        # shotNO1 = 152708
        # plot_R1 = 6.5
        TSR1 = TsmapSmoothRead(shotNO)
        te = TSR1.get_data('Te_fit',plot_reff)
        ne = TSR1.get_data('ne_fit',plot_reff)
        te_f = interpolate.interp1d(TSR1.time, te, bounds_error=False,fill_value=0)
        te_list = te_f(timelist)
        ne_f = interpolate.interp1d(TSR1.time, ne, bounds_error=False,fill_value=0)
        ne_list = ne_f(timelist)
        p0 = te_list*ne_list
        # print(p0)
        return p0


if __name__=='__main__':
    # shotNO1 = 152708
    # shotNO1 = 148582
    shotNO = 174925
    TSR1 = TsmapSmoothRead(shotNO)
    # PG = P0Get
    time_list = np.arange(3,8,0.01)
    # print(time_list)

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
    # plt.plot(TSR1.time,TSR1.get_data('Te',4.6))
    plt.plot(TSR1.time,TSR1.get_data('ne_fit',0)*TSR1.get_data('Te_fit',0))
    plt.plot(time_list,P0Get.get_p0(174925,time_list))
    P0Get.get_p0(174925,time_list)
    plt.ylabel('Te[keV]')
    plt.xlabel('Time[s]')
    # plt.xlim(3,5)
    # plt.ylim(-0.5,1)
    # plt.ylim(0,0.5)
    # plt.savefig('Te_plot_174923.png')
    plt.show()
