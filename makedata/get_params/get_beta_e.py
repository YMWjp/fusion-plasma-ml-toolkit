from neubeta import IotaGet

def get_beta_e(self):
        self.beta_e = IotaGet.get_betaiota(self.shotNO, self.time_list)[0]
        return