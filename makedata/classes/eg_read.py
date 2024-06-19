import numpy as np
from egdb_class import egdb2d
from scipy import interpolate


class eg_read:
    def __init__(self, datname):
        self.eg = egdb2d(datname)
        self.eg.readFile()

    def eg_f1(self, valname, timelist):
        time = np.array(self.eg.dimdata[1:])
        data = np.array(self.eg.data[self.eg.valname2idx(valname)][1:])
        f1 = interpolate.interp1d(time, data, bounds_error=False, fill_value=0)
        return f1(timelist)
