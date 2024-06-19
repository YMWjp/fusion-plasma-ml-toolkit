import numpy as np
from make_dataset4 import eg_read

def get_nbi(self):
        nb_tmp = np.zeros_like(self.time_list)
        tan_names = ["1", "2", "3"]

        def nbiabs(through, nebar, s):
            if self.Bt < 0:
                if s == 1 or 3:
                    loss_ratio = 0.28127 + 0.091059 * np.exp(-3.5618 * nebar / 10)
                else:
                    loss_ratio = -0.010049 + 2.0175 * np.exp(-10.904 * nebar / 10)
                    loss_ratio[loss_ratio < 0] = 0
                loss_ratio[loss_ratio > 1] = 1
                # print(through,loss_ratio)
                abs = through * (1 - loss_ratio)
                return abs
            else:
                if s == 2:
                    loss_ratio = 0.28127 + 0.091059 * np.exp(-3.5618 * nebar / 10)
                else:
                    loss_ratio = -0.010049 + 2.0175 * np.exp(-10.904 * nebar / 10)
                    loss_ratio[loss_ratio < 0] = 0
                loss_ratio[loss_ratio > 1] = 1
                # print(through,loss_ratio)
                abs = through * (1 - loss_ratio)
                return abs

        for s in tan_names:
            # eg = eg_read("./nb"+s+"pwr@"+str(self.shotNO)+".dat")
            # _temporalしかない放電もある．原因不明　そのためtemporalのまま
            eg = eg_read("./nb" + s + "pwr_temporal@" + str(self.shotNO) + ".dat")
            # pdb.set_trace()
            unit = eg.eg.valunits()[eg.eg.valname2idx("Pport-through_nb" + s)]
            if unit == "kW":
                nb_tmp = np.vstack(
                    (
                        nb_tmp,
                        nbiabs(
                            eg.eg_f1("Pport-through_nb" + s, self.time_list) / 1000,
                            self.nel / self.ne_length,
                            s,
                        ),
                    )
                )
            elif unit == "MW":
                nb_tmp = np.vstack(
                    (
                        nb_tmp,
                        nbiabs(
                            eg.eg_f1("Pport-through_nb" + s, self.time_list),
                            self.nel / self.ne_length,
                            s,
                        ),
                    )
                )
        # print(nb_tmp)
        nbi_tan_through = np.sum(np.abs(nb_tmp), axis=0)
        # print(nbi_tan_through)
        self.nbi_tan = nbi_tan_through
        # self.nbi_tan = self.dep_Pnbi_tan(
        #     nbi_tan_through,self.nel_thomson/self.ne_length)
        # # NBI tan　dep計算　0703

        nb_tmp = np.zeros_like(self.time_list)
        perp_names = ["4a", "4b", "5a", "5b"]
        for s in perp_names:
            eg = eg_read("./nb" + s + "pwr_temporal@" + str(self.shotNO) + ".dat")
            unit = eg.eg.valunits()[eg.eg.valname2idx("Pport-through_nb" + s)]
            if unit == "kW":
                nb_tmp = np.vstack(
                    (nb_tmp, eg.eg_f1("Pport-through_nb" + s, self.time_list) / 1000)
                )
            elif unit == "MW":
                nb_tmp = np.vstack(
                    (nb_tmp, eg.eg_f1("Pport-through_nb" + s, self.time_list))
                )
        self.nbi_perp = np.sum(np.abs(nb_tmp), axis=0)
        # self.prad_grad = np.gradient(self.prad, self.time_list)
        # self.ln_prad_grad = np.gradient(np.log(self.prad), self.time_list)

        return 1