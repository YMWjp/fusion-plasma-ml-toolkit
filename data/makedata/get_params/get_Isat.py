import os
import numpy as np
from classes.eg_read import eg_read


def get_Isat(self):  # 上書き
    if os.path.isfile("../egdata/DivIis_tor_sum@" + str(self.shotNO) + ".dat"):
        eg = eg_read("DivIis_tor_sum@" + str(self.shotNO) + ".dat")
        gdn_info = self.get_gdn_info("./DivIis_tor_sum@" + str(self.shotNO) + ".dat")
        for i, nL in enumerate(
            [
                "2L",
                "2R",
                "4L",
                "4R",
                "6L",   
                "6R",
                "7L",
                "7R",
                "8L",
                "8R",
                "9L",
                "9R",
                "10L",
                "10R",
            ]
        ):
            try:
                setattr(
                    self,
                    "Isat_" + nL,
                    eg.eg_f1("Iis_" + nL + "@" + str(gdn_info[i]), self.time_list),
                )
            except ValueError:
                pass
        # Isat_7Lの外れ値をなくし整形する
        # まず、0.0001以下の場合はひとつ飛ばした前後の値の平均を代入する
        for i in range(2, len(self.Isat_7L) - 2):
            if self.Isat_7L[i] < 0.0001:
                self.Isat_7L[i] = (self.Isat_7L[i - 2] + self.Isat_7L[i + 2]) / 2

        # 次に、一個前の値の2/3以下の場合、ひとつ前の値を使用する
        Isat_7L_copy = self.Isat_7L.copy()
        for i in range(1, len(self.Isat_7L)):
            if Isat_7L_copy[i] < Isat_7L_copy[i - 1] / 1.5:
                self.Isat_7L[i] = Isat_7L_copy[i - 1]
    else:
        for nL in [
            "2L",
            "2R",
            "4L",
            "4R",
            "6L",
            "6R",
            "7L",
            "7R",
            "8L",
            "8R",
            "9L",
            "9R",
            "10L",
            "10R",
        ]:
            setattr(self, f"Isat_{nL}", np.zeros_like(self.time_list))
    return 1
