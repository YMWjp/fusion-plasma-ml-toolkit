from classes.eg_read import eg_read


def get_fig(self, shotNO):
    eg = eg_read("./fig_h2@" + str(self.shotNO) + ".dat")
    self.fig6I = eg.eg_f1("FIG(6I_W)", self.time_list)
    self.pcc3O = eg.eg_f1("Pcc(3-O)", self.time_list)
    self.figpcc = self.fig6I / self.pcc3O
