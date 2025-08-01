import numpy as np


def get_SDLloop(self, shotNO):
    SDLloop_data = np.loadtxt(
        "../makedata/SDLloopdata/Phieff" + str(shotNO) + ".dat", skiprows=1, delimiter=","
    )
    self.SDLloop_dphi = np.zeros(len(self.time_list))
    self.SDLloop_dphi_ext = np.zeros(len(self.time_list))
    self.SDLloop_dtheta = np.zeros(len(self.time_list))
    i = 0
    for t in self.time_list:
        ind = np.argmin(abs(t - SDLloop_data[:, 0]))
        self.SDLloop_dphi[i] = SDLloop_data[ind, 1]
        self.SDLloop_dphi_ext[i] = SDLloop_data[ind, 5]
        self.SDLloop_dtheta[i] = np.abs(SDLloop_data[ind, 2])
        i = i + 1
