import pandas as pd
import matplotlib.pyplot as plt

# フォントサイズを設定
plt.rcParams.update({'font.size': 14})

# CSVファイルからデータを読み込む
data = pd.read_csv('dataset_25_7.csv')

# shotNOが115083のデータをフィルタリング
filtered_data = data[data['shotNO'] == 115083]

# プロットの設定
fig, axs = plt.subplots(3, 1, figsize=(8, 5.5), sharex=True)

# グラフの間隔を狭くする
plt.subplots_adjust(hspace=0.1)

# 一番上のグラフ: Wpとnel
ax1 = axs[0]
ax2 = ax1.twinx()
ax1.plot(filtered_data[' times'], filtered_data[' nel'], label='Nel')
ax2.plot(filtered_data[' times'], filtered_data[' Wp'] * 1000, 'orange', label='Wp')
ax1.set_ylabel(r'$N_{e}\ [10^{19}\ \mathrm{m}^{-3}]$')
ax2.set_ylabel('$W_{p}$[kJ]')

ax1.grid(True)

# 中央のグラフ: PinputとPrad
axs[1].plot(filtered_data[' times'], filtered_data[' Pinput'], 'orange', label='Pinput')
axs[1].plot(filtered_data[' times'], filtered_data[' Prad'], label='Prad')
axs[1].set_ylabel('$P_{rad},P_{in}$[MW]')
axs[1].grid(True)

# 一番下のグラフ: Isat@7L
axs[2].plot(filtered_data[' times'], filtered_data[' Isat@7L'], label='Isat@7L')
axs[2].set_xlabel('Time[s]')
axs[2].set_ylabel('$I_{sat}$[A]')
axs[2].grid(True)
# 横軸の始まりを3.0に設定
for ax in axs:
    ax.set_xlim(left=3.0, right=8.7)

# timeが4.86から7.4の部分を淡い青色で塗りつぶす
for ax in axs:
    ax.axvspan(4.86, 7.4, color='lightblue', alpha=0.4)

# グラフのタイトルを設定
fig.suptitle('ShotNO 115083: Various Parameters vs Time')

# timeが4.86, 7.4の時のnelの値をプリント
nel_486 = filtered_data.loc[filtered_data[' times'] == 4.86, ' nel'].values
nel_740 = filtered_data.loc[filtered_data[' times'] == 7.4, ' nel'].values

print(f"nel at time 4.86: {nel_486}")
print(f"nel at time 7.4: {nel_740}")

# グラフを表示
plt.show()