1.  →
    cd datamake
    python pr7_25.py

2.  →
    python pr8.py [date] [K(1~14)]

3.  →
    python F1score.py

変更点
numpy のバージョンアップに対応

np.int → int に変更
if self.weight_list == []: → if len(self.weight_list) == 0:に変更
