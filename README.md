1.  →
    cd makedata
    python pr7_25.py

2.  →
    python pr8.py [date] [K(1~14)]

3.  →
    python F1score.py -d [date]

4.  →
    python pr9_5.py

変更点
numpy のバージョンアップに対応

np.int → int に変更
if self.weight_list == []: → if len(self.weight_list) == 0:に変更

def_type 関数を修正してラベルずけを行う

get_Isat を修正する
