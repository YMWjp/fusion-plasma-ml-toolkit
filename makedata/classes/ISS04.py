import numpy as np


class ISS04():
    def __init__(self,R):
        
        self.B_coef = 0.88
        self.LHD_coef = 0.93

        if R == 3.6:
            self.R = 3.68 #3.684
            self.a = 0.63 #0.634
            self.iota = 0.65
        elif R == 3.75:
            self.R = 3.75
            self.a = 0.59
            self.iota = 0.59
        elif R == 3.9:
            self.R = 3.82
            self.a = 0.54
            self.iota = 0.58
        else:
            self.R = np.nan
            self.a = np.nan
            self.iota = np.nan
            print("NO SUCH DATA")

        self.B=0
        self.nel = []
        self.P = []
        self.tau = []
        return
    
    def calc_tau(self,B,nel,P):
        self.B = B
        self.nel = nel
        self.P = P
        # import pdb; pdb.set_trace()

        self.tau = 0.134 * np.power(self.a,2.28) * np.power(self.R,0.64) \
                * np.power(self.P,-0.61) * np.power(self.nel,0.54) \
                * np.power(np.abs(self.B)*self.B_coef,0.84) \
                * np.power(self.iota,0.41)
            
        return self.tau