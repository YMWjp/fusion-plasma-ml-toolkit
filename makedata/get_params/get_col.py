import numpy as np
from neubeta import IotaGet


def get_col(self, shotNO):
    iota = IotaGet.get_betaiota(shotNO, self.time_list)[2]
    te = IotaGet.get_betaiota(shotNO, self.time_list)[3]
    ne = IotaGet.get_betaiota(shotNO, self.time_list)[4]
    reff = IotaGet.get_betaiota(shotNO, self.time_list)[5]

    def fcol(iota, te, ne, reff, R):
        ln_lambda = 23 - np.log((ne * 10**13) ** 0.5 / (te * 10**3))
        ve = 2.91 * (10**7) * ne * ln_lambda / ((te * 10**3) ** 1.5)
        et = reff / R
        qR = R / iota
        vth = 4.19 * 10**5 * (te * 10**3) ** 0.5
        vb = ve / ((et**1.5) * (vth / qR))
        return vb

    self.col = fcol(iota, te, ne, reff, 3.9)
    return 1
