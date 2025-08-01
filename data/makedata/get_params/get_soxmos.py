import numpy as np
from scipy import interpolate

def get_soxmos(self, shotNO):
        if shotNO >= 171335 and shotNO <= 171387:
            with open(
                "../makedata/20211028_Ne_line/s" + str(shotNO) + "_VUV109L_490A.txt", "r"
            ) as f:
                names = f.readline().lstrip(" ").rstrip("\n")
            ne_data = np.loadtxt(
                "../makedata/20211028_Ne_line/s" + str(shotNO) + "_VUV109L_490A.txt"
            )
            time_data = np.loadtxt("../makedata/20211028_Ne_line/time_sec.txt")
            ne_f = interpolate.interp1d(
                time_data, ne_data, kind="linear", bounds_error=False, fill_value=0
            )
            ne_list = ne_f(self.time_list)
            ne_list = np.where(ne_list < 1, 1, ne_list)
            self.ne_soxmos = ne_list
            self.ar_soxmos = np.ones(len(self.time_list))
        elif shotNO > 180583 and shotNO < 181322:
            with open(
                "../makedata/soxmos/soxmos_data/"
                + str(shotNO)
                + "/soxmos_peak@"
                + str(shotNO)
                + "_ch1_lambda08667_time.txt",
                "r",
            ) as f:
                names = f.readline().lstrip(" ").rstrip("\n")
            ne_data = np.loadtxt(
                "../makedata/soxmos/soxmos_data/"
                + str(shotNO)
                + "/soxmos_peak@"
                + str(shotNO)
                + "_ch1_lambda08667_time.txt",
                skiprows=1,
            ).T
            # header = np.loadtxt(
            #     './soxmos/soxmos_peak@'+str(shotNO)+'_ch1_lambda08667_time', usecols=0, delimiter=',', dtype=str
            # ).T
            ne_f = interpolate.interp1d(
                ne_data[0], ne_data[5], kind="linear", bounds_error=False, fill_value=0
            )
            ne_list = ne_f(self.time_list)
            ne_list = np.where(ne_list < 1, 1, ne_list)
            self.ne_soxmos = ne_list

            with open(
                "../makedata/soxmos/soxmos_data/"
                + str(shotNO)
                + "/soxmos_peak@"
                + str(shotNO)
                + "_ch2_lambda22054_time.txt",
                "r",
            ) as f:
                names = f.readline().lstrip(" ").rstrip("\n")
            ar_data = np.loadtxt(
                "../makedata/soxmos/soxmos_data/"
                + str(shotNO)
                + "/soxmos_peak@"
                + str(shotNO)
                + "_ch2_lambda22054_time.txt",
                skiprows=1,
            ).T
            # header = np.loadtxt(
            #     './soxmos/soxmos_peak@'+str(shotNO)+'_ch2_lambda22054_time', usecols=0, delimiter=',', dtype=str
            # ).T
            ar_f = interpolate.interp1d(
                ar_data[0], ar_data[5], kind="linear", bounds_error=False, fill_value=0
            )
            ar_list = ar_f(self.time_list)
            ar_list = np.where(ar_list < 1, 1, ar_list)
            self.ar_soxmos = ar_list
        else:
            self.ne_soxmos = np.ones(len(self.time_list))
            self.ar_soxmos = np.ones(len(self.time_list))
        return 1