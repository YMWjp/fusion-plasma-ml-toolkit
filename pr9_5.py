#1パラメータ分離させた領域図_rap影響評価に使用
from common import names_dict
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace

date = '20240707'
datapath = './results/'+date+'/dataset.csv'
datapath2 = './results/'+date+'/label.csv'
datapath3 = './results/'+date+'/result6.tsv'
datapath4 = './results/'+date+'/parameter.csv'
datapath5 = './results/'+date+'/dataset_minmax.csv'

data = np.loadtxt(
    datapath, delimiter=','
    )
data_row = np.loadtxt(
    datapath, dtype=str, delimiter=',', comments='&'
    )
label = np.loadtxt(
    datapath2
    )
header = data_row[0]
header[0] = header[0].strip('# ')
with open (datapath3,'r') as f :
    names = f.readline().lstrip(' ').rstrip('\n')
svm_result_data = np.loadtxt(
    datapath3,dtype = str
    )
svm_result_data_t = np.loadtxt(
    datapath3,dtype = str, skiprows=1
    ).T
with open (datapath4,'r') as f :
    names = f.readline().lstrip(' ').rstrip('\n')
use_parameter_list = np.loadtxt(
    datapath4,delimiter=',',dtype = str
    )
with open (datapath5,'r') as f :
    names = f.readline().lstrip(' ').rstrip('\n')
minmax_data = np.loadtxt(
    datapath5,skiprows=1,delimiter=','
    )


n0 = 2
n1 = 6
n2 = 7
data0 = data[:,n0]
data0 = np.e**(data0*(minmax_data[0,n0]-minmax_data[1,n0])+minmax_data[1,n0])
data01 = data0[label==1]
data02 = data0[label==-1]
data1 = data[:,n1]
data1 = np.e**(data1*(minmax_data[0,n1]-minmax_data[1,n1])+minmax_data[1,n1])
data11 = data1[label==1]
data12 = data1[label==-1]
data2 = data[:,n2]
data2 = np.e**(data2*(minmax_data[0,n2]-minmax_data[1,n2])+minmax_data[1,n2])
data21 = data2[label==1]
data22 = data2[label==-1]



n3 = 2
data3 = data[:,n3]
data3 = np.e**(data3*(minmax_data[0,n3]-minmax_data[1,n3])+minmax_data[1,n3])
data31 = data3[label==1]
data32 = data3[label==-1]



plt.rcParams["font.size"] =25 #フォントサイズを一括で変更
plt.rcParams['lines.linewidth'] = 2 #線の太さを一括で変更
fig = plt.figure(figsize=(10,8)) #figureを準備
plt.subplots_adjust(
    left=0.2,right=0.90,bottom=0.125,top=0.94,hspace=0.3
    )

# axes = [0]*4
# axes1 = plt.subplot2grid((3, 3), (0, 0))
# axes2 = plt.subplot2grid((3, 3), (0, 1))
# axes3 = plt.subplot2grid((3, 3), (0, 2))
axes4 = plt.subplot()
# axes4 = plt.subplot2grid((3, 3), (1, 0),colspan=3,rowspan=2)
# axes = [fig.add_subplot2grid((2,2),) for i in range(3)] #axのリストを準備
# axes[0] = fig.add_subplot2grid((2,3),(0,0))
# axes[1] = fig.add_subplot2grid((2,3),(0,1))
# plt = fig.add_subplot2grid((2,3),(0,2))
# axes[3] = fig.add_subplot2grid((2,3),(1,0),colspan=2)

'''

axes1.hist(data01,bins=50,color='blue',alpha=0.5)
axes1.hist(data02,bins=50,color='red',alpha=0.5)
axes1.set_ylim(0,65)
axes1.set_ylabel('number')
# plt.xlabel('$P$'+'rad/'+'$P$'+'input')
axes1.set_xlabel('$P\mathrm{rad}/P\mathrm{input}$')

axes2.hist(data11,bins=50,color='blue',alpha=0.5)
axes2.hist(data12,bins=50,color='red',alpha=0.5)
axes2.set_ylim(0,65)
# plt.ylabel('number')
axes2.set_xlabel('%s' % header[n1] )

axes3.hist(data21,bins=50,color='blue',alpha=0.5)
axes3.hist(data22,bins=50,color='red',alpha=0.5)
axes3.set_ylim(0,65)
# plt.ylabel('number')
axes3.set_xlabel('%s' % header[n2] )
# plt.savefig('hist_%s%s%s_%s.png' % (header[n0],header[n1],header[n2],date))

'''

results_header = svm_result_data[0]
svm_result_data_onlyfloat = svm_result_data_t[1:]

np.where(results_header == 'F1score')[0][0]

#カエルのここ
# 使いたいものの列番号から1を引いた値
shotdata_f1max = svm_result_data[200]
target_number = 3

weight_before = [float(s) for s in shotdata_f1max[1:len(use_parameter_list)+1]]
# weight_afterの計算でゼロ除算を避けるためのチェックを追加
epsilon = 1e-10  # 小さな値を追加してゼロ除算を避ける
weight_after = weight_before / (minmax_data[0] - minmax_data[1] + epsilon)
bias_after = float(shotdata_f1max[len(use_parameter_list)+1])
for i in range(len(use_parameter_list)):
    bias_after = bias_after - weight_after[i]*minmax_data[1,i]

def func_func(datain_shot=[]):
    #まずwとbの補正
    f = 1
    datain_shot_row  = np.zeros(len(use_parameter_list))
    for i in range(len(use_parameter_list)):
        datain_shot_row[i] = np.e**(datain_shot[i]*(minmax_data[0,i]-minmax_data[1,i])+minmax_data[1,i])
        f = f * datain_shot_row[i]**weight_after[i]
    f = f * np.e ** bias_after
    return f

func_func_list = np.zeros(len(data))

for i in range(len(data)):
    func_func_list[i] = func_func(datain_shot=data[i])


func_func_list1 = func_func_list[label==1]
func_func_list2 = func_func_list[label==-1]

x0lim_log = [np.log10(np.min(func_func_list))-(np.log10(np.max(func_func_list))+np.log10(np.min(func_func_list)))/10,np.log10(np.max(func_func_list))+(np.log10(np.max(func_func_list))+np.log10(np.min(func_func_list)))/10]



#以下、一要素と他要素の散布図
def func_func1(n,datain_shot=[]):
    f = 1
    datain_shot2 = np.delete(datain_shot,n)
    min_data = minmax_data[1]
    max_data = minmax_data[0]
    min_data = np.delete(min_data,n)
    max_data = np.delete(max_data,n)
    weight_after2 = np.delete(weight_after,n)
    weight_after2 = weight_after2*-1/weight_after[n]
    datain_shot_row  = np.zeros(len(use_parameter_list)-1)
    for i in range(len(use_parameter_list)-1):
        datain_shot_row[i] = np.e**(datain_shot2[i]*(max_data[i]-min_data[i])+min_data[i])
        f = f * datain_shot_row[i]**weight_after2[i]
    f = f * np.e ** (bias_after*-1/weight_after[n])
    return f

func_func1_list = np.zeros(len(data))
for i in range(len(data)):
    func_func1_list[i] = func_func1(target_number,datain_shot=data[i])
func_func1_list1 = func_func1_list[label==1]
func_func1_list2 = func_func1_list[label==-1]
target_parameter_list = data[:,target_number]
target_parameter_row_list = np.e**(target_parameter_list*(minmax_data[0,target_number]-minmax_data[1,target_number])+minmax_data[1,target_number])
target_parameter_list1 = target_parameter_row_list[label==1]
target_parameter_list2 = target_parameter_row_list[label==-1]



# target_parameter_list1 = target_parameter_list1/data31
# target_parameter_list2 = target_parameter_list2/data32


# y2lim = [0.0001,0.001]
y2lim = [0.3,3]
# y2lim = [0.001,7]
y2lim_space = linspace(y2lim[0],y2lim[1],10000)
x_forfig2 = y2lim_space
# print(len(func_func1_list))
# print(len(target_parameter_list))

axes4.plot(func_func1_list1,target_parameter_list1,'.',color='blue',label='detach', markersize=20)
axes4.plot(func_func1_list2,target_parameter_list2,'.',color='red',label='attach', markersize=20)
axes4.plot(x_forfig2,y2lim_space,color='black')
axes4.legend(labelcolor='linecolor',markerscale=1)
# plt.ylim(np.min(target_parameter_row_list),np.max(target_parameter_row_list))
axes4.set_xscale("log")


with open(datapath4, 'r') as f:
    parameter_list = [line.strip() for line in f]
axes4.set_ylabel(names_dict.get(parameter_list[target_number], parameter_list[target_number]))

# Parse func_parameter_list
func_parameter_list = shotdata_f1max[0].split(',')
func_parameter_list_int = [int(s) for s in func_parameter_list]

print(func_parameter_list_int)

# Remove target_number from the list
func_parameter_list_int.remove(target_number)
# for i in func_parameter_list_int:
#     param_name = names_dict.get(parameter_list[i], parameter_list[i])
#     exponent = round(weight_after[i]*-1/weight_after[target_number], 3)
#     xlavel += f'{param_name}^{exponent}'

print(func_parameter_list_int)
xlavel = '${e}^{%s}$' % str(round(bias_after*-1/weight_after[target_number], 3))
for i in func_parameter_list_int:
    print(use_parameter_list[i], i)
    if i == 2:
        xlavel = xlavel + r'$P_{in}^{%s}$' % (round(weight_after[i]*-1/weight_after[target_number],3))
    elif i == 10:
        xlavel = xlavel + '$\Delta\Phi_{eff}^{%s}$' % (round(weight_after[i]*-1/weight_after[target_number],3))
    elif i == 11:
        xlavel = xlavel + '$\Delta\Theta_{eff}^{%s}$' % (round(weight_after[i]*-1/weight_after[target_number],3))
    else:
        xlavel = xlavel + '$\mathrm{%s}^{%s}$' % (use_parameter_list[i],round(weight_after[i]*-1/weight_after[target_number],3))

axes4.set_xlabel(xlavel, fontsize=18)
for i in range(len(target_parameter_list1)):
    if target_parameter_list1[i] >= 2:
        print(f"Index: {i}, Value: {target_parameter_list1[i]}, Func Value: {func_func1_list1[i]}")


plt.savefig('./hist&sccaterpng/sccater_rmp_'+str(date)+'target'+str(target_number)+'.png')
plt.show()