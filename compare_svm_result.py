from common import names_dict
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace

# Global constants
DATE = '20240923'
RESULTS_DIR = f'./results/{DATE}/'
TARGET_NUMBER = 11
EPSILON = 1e-10
SVM_RESULT_INDEX = 224

# New global variables for axis limits
X_AXIS_LIMITS = (-1, 1e1)  # 横軸の範囲を指定
Y_AXIS_LIMITS = (0, 0.0011)  # 縦軸の範囲を指定

# File paths definition
FILE_PATHS = {
    'dataset': f'{RESULTS_DIR}dataset.csv',
    'label': f'{RESULTS_DIR}label.csv',
    'result': f'{RESULTS_DIR}result4.tsv',
    'parameter': f'{RESULTS_DIR}parameter.csv',
    'dataset_minmax': f'{RESULTS_DIR}dataset_minmax.csv'
}

def load_data():
    try:
        data = np.loadtxt(FILE_PATHS['dataset'], delimiter=',')
        label = np.loadtxt(FILE_PATHS['label'])
        svm_result_data = np.loadtxt(FILE_PATHS['result'], dtype=str)
        use_parameter_list = np.loadtxt(FILE_PATHS['parameter'], delimiter=',', dtype=str)
        minmax_data = np.loadtxt(FILE_PATHS['dataset_minmax'], skiprows=1, delimiter=',')
    except IOError as e:
        print(f"File reading error: {e}")
        return None, None, None, None, None
    return data, label, minmax_data, use_parameter_list, svm_result_data

def process_data(data, label, minmax_data):
    n0, n1, n2 = 2, 6, 7
    data0 = np.e**(data[:, n0] * (minmax_data[0, n0] - minmax_data[1, n0]) + minmax_data[1, n0])
    data1 = np.e**(data[:, n1] * (minmax_data[0, n1] - minmax_data[1, n1]) + minmax_data[1, n1])
    data2 = np.e**(data[:, n2] * (minmax_data[0, n2] - minmax_data[1, n2]) + minmax_data[1, n2])
    
    return {
        'data0': {'all': data0, 'pos': data0[label==1], 'neg': data0[label==-1]},
        'data1': {'all': data1, 'pos': data1[label==1], 'neg': data1[label==-1]},
        'data2': {'all': data2, 'pos': data2[label==1], 'neg': data2[label==-1]}
    }

def calculate_weights(shotdata_f1max, use_parameter_list, minmax_data):
    weight_before = [float(s) for s in shotdata_f1max[1:len(use_parameter_list)+1]]
    weight_after = weight_before / (minmax_data[0] - minmax_data[1] + EPSILON)
    bias_after = float(shotdata_f1max[len(use_parameter_list)+1])
    for i in range(len(use_parameter_list)):
        bias_after -= weight_after[i] * minmax_data[1, i]
    return weight_after, bias_after

def func_func(datain_shot, weight_after, bias_after, use_parameter_list, minmax_data):
    f = 1
    datain_shot_row = np.zeros(len(use_parameter_list))
    for i in range(len(use_parameter_list)):
        datain_shot_row[i] = np.e**(datain_shot[i] * (minmax_data[0, i] - minmax_data[1, i]) + minmax_data[1, i])
        f *= datain_shot_row[i]**weight_after[i]
    f *= np.e ** bias_after
    return f

def func_func1(n, datain_shot, weight_after, bias_after, use_parameter_list, minmax_data):
    f = 1
    datain_shot2 = np.delete(datain_shot, n)
    min_data = np.delete(minmax_data[1], n)
    max_data = np.delete(minmax_data[0], n)
    weight_after2 = np.delete(weight_after, n) * -1 / weight_after[n]
    datain_shot_row = np.zeros(len(use_parameter_list) - 1)
    for i in range(len(use_parameter_list) - 1):
        datain_shot_row[i] = np.e**(datain_shot2[i] * (max_data[i] - min_data[i]) + min_data[i])
        f *= datain_shot_row[i]**weight_after2[i]
    f *= np.e ** (bias_after * -1 / weight_after[n])
    return f

def create_xlabel():
    # 指定された式をそのまま返す
    return r'$(e^{-10.0})(B^{1.61})((Prad/Pin)^{-2.63})(Pin^{-0.72})$'

def plot_scatter(func_func1_list1, target_parameter_list1, func_func1_list2, target_parameter_list2, y2lim_space, xlabel, ylabel):
    plt.rcParams["font.size"] = 25
    plt.rcParams['lines.linewidth'] = 2
    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=0.2, right=0.90, bottom=0.125, top=0.94, hspace=0.3)

    axes4 = plt.subplot()
    axes4.plot(func_func1_list1, target_parameter_list1, '.', color='blue', label='detach', markersize=20)
    axes4.plot(func_func1_list2, target_parameter_list2, '.', color='red', label='attach', markersize=20)
    axes4.plot(y2lim_space, y2lim_space, color='black')
    axes4.legend(labelcolor='linecolor', markerscale=1)
    axes4.set_xscale("log")
    axes4.set_xlabel(xlabel, fontsize=18)
    axes4.set_ylabel(ylabel)

    # Set axis limits using global variables
    axes4.set_xlim(X_AXIS_LIMITS)
    axes4.set_ylim(Y_AXIS_LIMITS)

    plt.savefig(f'./hist&sccaterpng/sccater_rmp_{DATE}target{TARGET_NUMBER}.png')
    plt.show()

def run_analysis():
    data, label, minmax_data, use_parameter_list, svm_result_data = load_data()
    if data is None:
        return

    # 必要な列のインデックスを取得
    B_index = 1  # Bのインデックス
    Prad_Pin_index = 4  # Prad/Pinのインデックス
    Pin_index = 2  # Pinのインデックス

    # データの事前チェック
    if np.any(data[:, Prad_Pin_index] == 0):
        print("Warning: Prad/Pin column contains zero values.")
    if np.any(data[:, Pin_index] == 0):
        print("Warning: Pin column contains zero values.")
    if np.any(np.isnan(data)):
        print("Warning: Data contains NaN values.")

    # 横軸データの計算
    x_axis_data = (np.e**-10.0) * (data[:, B_index]**1.61) * ((data[:, Prad_Pin_index])**-2.63) * (data[:, Pin_index]**-0.72)

    # デバッグ情報の出力
    print("x_axis_data:", x_axis_data)

    # nanやinfのチェック
    if np.any(np.isnan(x_axis_data)):
        print("Warning: x_axis_data contains NaN values.")
    if np.any(np.isinf(x_axis_data)):
        print("Warning: x_axis_data contains Inf values.")
    if np.any(x_axis_data == 0):
        print("Warning: x_axis_data contains zero values.")

    x_axis_data1 = x_axis_data[label == 1]
    x_axis_data2 = x_axis_data[label == -1]

    target_parameter_list = data[:, TARGET_NUMBER]
    target_parameter_row_list = np.e**(target_parameter_list * (minmax_data[0, TARGET_NUMBER] - minmax_data[1, TARGET_NUMBER]) + minmax_data[1, TARGET_NUMBER])
    target_parameter_list1 = target_parameter_row_list[label == 1]
    target_parameter_list2 = target_parameter_row_list[label == -1]

    # 座標データの整形
    coordinates1 = list(zip(x_axis_data1, target_parameter_list1))
    coordinates2 = list(zip(x_axis_data2, target_parameter_list2))

    # 座標データをテキストファイルに保存
    with open('coordinates.txt', 'w') as f:
        f.write("Detach Points (Label 1):\n")
        for coord in coordinates1:
            f.write(f"x: {coord[0]}, y: {coord[1]}\n")

        f.write("\nAttach Points (Label -1):\n")
        for coord in coordinates2:
            f.write(f"x: {coord[0]}, y: {coord[1]}\n")

    y2lim = [0, 0.0011]
    y2lim_space = linspace(y2lim[0], y2lim[1], 10000)

    with open(FILE_PATHS['parameter'], 'r') as f:
        parameter_list = [line.strip() for line in f]
    ylabel = names_dict.get(parameter_list[TARGET_NUMBER], parameter_list[TARGET_NUMBER])

    xlabel = create_xlabel()

    # plot_scatter 関数に新しい横軸データを渡す
    plot_scatter(x_axis_data1, target_parameter_list1, x_axis_data2, target_parameter_list2, y2lim_space, xlabel, ylabel)

def main():
    # Entry point of the script
    run_analysis()

if __name__ == "__main__":
    main()