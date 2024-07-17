import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from scipy.signal import savgol_filter
import numpy as np

# フォントサイズを設定
plt.rcParams.update({'font.size': 20})

# CSVファイルからデータを読み込む
data = pd.read_csv('dataset_25_7.csv')

# shotNOが115083のデータをフィルタリング
filtered_data = data[(data['shotNO'] == 115083) & (data[' times'] >= 3.96) & (data[' times'] <= 8.28)]

def detect_and_interpolate_outliers(series, threshold=3):
    median = series.median()
    diff = np.abs(series - median)
    mad = diff.median()
    modified_z_score = 0.6745 * diff / mad
    outliers = modified_z_score > threshold
    series[outliers] = np.nan
    return series.interpolate()

# Isat@7Lの値を外れ値補間後にスムージング
filtered_data[' Isat@7L_cleaned'] = detect_and_interpolate_outliers(filtered_data[' Isat@7L'])
filtered_data[' Isat@7L_smoothed'] = savgol_filter(filtered_data[' Isat@7L_cleaned'], window_length=11, polyorder=2)

# timeが4.86の時と7.4の時のnelの値を取得
nel_4_86 = filtered_data[filtered_data[' times'] == 4.86][' nel'].values[0]
nel_7_4 = filtered_data[filtered_data[' times'] == 7.4][' nel'].values[0]

# Pinputのプロット
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(filtered_data[' nel'], filtered_data[' Pinput'], label='Pinput', color='green')
ax1.set_xlabel(r'密度$\ [10^{19}\ \mathrm{m}^{-3}]$')
ax1.set_ylabel('Pinput [MW]')
ax1.grid(True)
ax1.legend(loc='upper left')
fig1.suptitle('ShotNO 115083: Pinput vs Nel')

# Bのプロット
fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.plot(filtered_data[' nel'], filtered_data[' B'], label='B', color='red')
ax2.set_xlabel(r'密度$\ [10^{19}\ \mathrm{m}^{-3}]$')
ax2.set_ylabel('B [T]')
ax2.grid(True)
ax2.legend(loc='upper left')
fig2.suptitle('ShotNO 115083: B vs Nel')

# Prad/Pinputのプロット
fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.plot(filtered_data[' nel'], filtered_data[' Prad'] / filtered_data[' Pinput'], label='Prad/Pinput', color='purple')
ax3.set_xlabel(r'密度$\ [10^{19}\ \mathrm{m}^{-3}]$')
ax3.set_ylabel('Prad/Pinput')
ax3.grid(True)
ax3.legend(loc='upper left')
fig3.suptitle('ShotNO 115083: Prad/Pinput vs Nel')

# 横軸Prad, 縦軸Isatのプロット
fig4, ax4 = plt.subplots(figsize=(8, 5))
ax4.plot(filtered_data[' Prad'], filtered_data[' Isat@7L_smoothed'], label='Isat vs Prad', color='blue')
ax4.set_xlabel('Prad [MW]')
ax4.set_ylabel('Isat [A]')
ax4.grid(True)
ax4.legend(loc='upper left')
fig4.suptitle('ShotNO 115083: Isat vs Prad')

# 横軸Prad, 縦軸Pinputのプロット
fig5, ax5 = plt.subplots(figsize=(8, 5))
ax5.plot(filtered_data[' Prad'], filtered_data[' Pinput'], label='Pinput vs Prad', color='green')
ax5.set_xlabel('Prad [MW]')
ax5.set_ylabel('Pinput [MW]')
ax5.grid(True)
ax5.legend(loc='upper left')
fig5.suptitle('ShotNO 115083: Pinput vs Prad')

# 横軸Isat, 縦軸Pinputのプロット
fig6, ax6 = plt.subplots(figsize=(8, 5))
ax6.plot(filtered_data[' Isat@7L_smoothed'], filtered_data[' Pinput'], label='Pinput vs Isat', color='red')
ax6.set_xlabel('Isat [A]')
ax6.set_ylabel('Pinput [MW]')
ax6.grid(True)
ax6.legend(loc='upper left')
fig6.suptitle('ShotNO 115083: Pinput vs Isat')

# グラフを表示
plt.show()