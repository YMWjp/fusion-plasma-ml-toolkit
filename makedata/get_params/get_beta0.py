from get_p0 import P0Get

def get_beta0(self, shotNO):
    p0 = P0Get.get_p0(shotNO, self.time_list)
    self.beta0 = p0 / self.Bt**2
    return 1