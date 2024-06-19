import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib

# フォントサイズを設定
plt.rcParams.update({'font.size': 20})

# CSVファイルからデータを読み込む
data = pd.read_csv('dataset_25_7.csv')

# shotNOが115083のデータをフィルタリング
filtered_data = data[(data['shotNO'] == 115083) & (data[' times'] >= 3.96) & (data[' times'] <= 8.28)]

# timeが4.86の時と7.4の時のnelの値を取得
nel_4_86 = filtered_data[filtered_data[' times'] == 4.86][' nel'].values[0]
nel_7_4 = filtered_data[filtered_data[' times'] == 7.4][' nel'].values[0]

# プロットの設定
fig, ax1 = plt.subplots(figsize=(8, 5.5))

# x軸の範囲を設定（バッファを追加）
nel_min = filtered_data[' nel'].min()
nel_max = filtered_data[' nel'].max()
buffer = (nel_max - nel_min) * 0.05  # 5%のバッファ
ax1.set_xlim(nel_min - buffer, nel_max + buffer)

# 背景色を追加（バッファに合わせて調整）
ax1.axvspan(nel_min - buffer, nel_4_86, color='lightpink', alpha=0.5)
ax1.axvspan(nel_4_86, nel_7_4, color='lavender', alpha=0.5)
ax1.axvspan(nel_7_4, nel_max + buffer, color='lightblue', alpha=0.5)

# 左の縦軸: Isat@7L
ax1.plot(filtered_data[' nel'], filtered_data[' Isat@7L'], label=r'$I_{sat}$', color='blue')
ax1.set_xlabel(r'密度$\ [10^{19}\ \mathrm{m}^{-3}]$')
ax1.set_ylabel('ダイバータへのイオン粒子束[A]', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True)

# 右の縦軸: Prad
ax2 = ax1.twinx()
ax2.plot(filtered_data[' nel'], filtered_data[' Prad'], label=r'$P_{rad}$', color='orange')
ax2.set_ylabel("放射パワー[MW]", color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

# 凡例を追加
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# グラフのタイトルを設定
fig.suptitle('ShotNO 115083: Isat and Prad vs Nel')

# グラフを表示
plt.show()