import numpy as np


def get_rmp_lid(self, shotNO):
    with open("../makedata/experiment_log_new.csv", "r") as f:
        f.readline().lstrip(" ").rstrip("\n")

    rmp_data = np.loadtxt(
        "../makedata/experiment_log_new.csv", skiprows=1, delimiter=","
    ).T

    rmp_no_list = rmp_data[2]
    rmp_lid_list = rmp_data[4]
    for i in range(len(rmp_lid_list)):
        if rmp_lid_list[i] == 0:
            rmp_lid_list[i] = 1
    rmp_lid_list = rmp_lid_list[rmp_no_list == shotNO]
    self.rmp_lid = [rmp_lid_list[0]] * len(self.time_list)
    self.rmp_lid = [n / abs(self.Bt) for n in self.rmp_lid]
    return 1
