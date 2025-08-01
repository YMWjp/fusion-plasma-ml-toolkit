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

def create_xlabel(shotdata_f1max, bias_after, weight_after, use_parameter_list):
    func_parameter_list_int = [int(s) for s in shotdata_f1max[0].split(',')]
    if TARGET_NUMBER in func_parameter_list_int:
        func_parameter_list_int.remove(TARGET_NUMBER)
    
    xlabel = '${e}^{%s}$' % str(round(bias_after * -1 / weight_after[TARGET_NUMBER], 2))
    for i in func_parameter_list_int:
        if i == 2:
            xlabel += r'$P_{in}^{%s}$' % (round(weight_after[i] * -1 / weight_after[TARGET_NUMBER], 2))
        elif i == 11:
            xlabel += '$\Delta\Phi_{eff}^{%s}$' % (round(weight_after[i] * -1 / weight_after[TARGET_NUMBER], 2))
        elif i == 12:
            xlabel += '$\Delta\Theta_{eff}^{%s}$' % (round(weight_after[i] * -1 / weight_after[TARGET_NUMBER], 2))
        elif use_parameter_list[i] == 'Prad/Pinput':
            xlabel += '$P_\mathrm{rad}/P_\mathrm{in}^{%s}$' % (round(weight_after[i] * -1 / weight_after[TARGET_NUMBER], 2))
        else:
            xlabel += '$\mathrm{%s}^{%s}$' % (use_parameter_list[i], round(weight_after[i] * -1 / weight_after[TARGET_NUMBER], 2))
    
    return xlabel

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

    plt.savefig(f'./hist&sccaterpng/sccater_rmp_{DATE}target{TARGET_NUMBER}.png')
    plt.show()

def run_analysis():
    data, label, minmax_data, use_parameter_list, svm_result_data = load_data()
    if data is None:
        return

    shotdata_f1max = svm_result_data[SVM_RESULT_INDEX]
    weight_after, bias_after = calculate_weights(shotdata_f1max, use_parameter_list, minmax_data)

    func_func1_list = np.array([func_func1(TARGET_NUMBER, d, weight_after, bias_after, use_parameter_list, minmax_data) for d in data])
    func_func1_list1 = func_func1_list[label == 1]
    func_func1_list2 = func_func1_list[label == -1]

    target_parameter_list = data[:, TARGET_NUMBER]
    target_parameter_row_list = np.e**(target_parameter_list * (minmax_data[0, TARGET_NUMBER] - minmax_data[1, TARGET_NUMBER]) + minmax_data[1, TARGET_NUMBER])
    target_parameter_list1 = target_parameter_row_list[label == 1]
    target_parameter_list2 = target_parameter_row_list[label == -1]

    y2lim = [0, 0.0011]
    y2lim_space = linspace(y2lim[0], y2lim[1], 10000)

    with open(FILE_PATHS['parameter'], 'r') as f:
        parameter_list = [line.strip() for line in f]
    ylabel = names_dict.get(parameter_list[TARGET_NUMBER], parameter_list[TARGET_NUMBER])

    xlabel = create_xlabel(shotdata_f1max, bias_after, weight_after, use_parameter_list)

    plot_scatter(func_func1_list1, target_parameter_list1, func_func1_list2, target_parameter_list2, y2lim_space, xlabel, ylabel)

def main():
    # Entry point of the script
    run_analysis()

if __name__ == "__main__":
    main()