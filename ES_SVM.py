import numpy as np
import matplotlib.pyplot as plt

import itertools
import os.path
import pdb
import sys

from joblib import Parallel, delayed
from time import time

from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, cross_val_predict

class ExhaustiveSearch():
    """docstring for ExhaustiveSearch."""

    def __init__(self, dataname='dataset.csv'):
        self.dataname = dataname

        f = open(dataname, 'r')
        self.header = f.readline().rstrip('\n').split(', ')
        print(self.header)
        # makedata
        self.parameters = []
        self.shotNOs = []
        self.times = []
        #self.include = [] #datasetのlabels
        self.labels = [] #datasetのtypes
        self.data = []

        # preparation
        self.skf = []
        self.date = ''
        self.output_dir = ''

    def makeData(self, noUseLabel=0,use_parameters=[]):
        # use_parameters = ['nel','B','Pinput','center','a99']
        if not use_parameters:
            use_parameters = self.header[4:len(self.header)]

        usecols_list = [self.header.index(x) for x in use_parameters]
        self.parameters = use_parameters

        # import pdb; pdb.set_trace()
        #self.include = np.loadtxt(self.dataname, delimiter=',', skiprows=1, usecols=(self.header.index('labels'))) #放電のラベル

        labels_raw = np.loadtxt(self.dataname,delimiter=',', skiprows=1, usecols=(self.header.index('types'))) #データ自体のラベル
        #self.labels = labels_raw[self.include != 0]
        self.labels = labels_raw[labels_raw != noUseLabel]

        times_raw = np.loadtxt(self.dataname,delimiter=',', skiprows=1, usecols=(self.header.index('times')))
        #self.times = times_raw[self.include != 0]
        self.time = times_raw[labels_raw != noUseLabel]

        data_raw = np.loadtxt(self.dataname, delimiter=',', skiprows=1, usecols=tuple(usecols_list))
        #self.data = data_raw[self.include != 0]
        self.data = data_raw[labels_raw != noUseLabel]
        # import pdb; pdb.set_trace()

        shotNOs_raw = np.loadtxt(self.dataname, delimiter=',', skiprows=1, usecols=(self.header.index('shotNO')))
        #self.shotNOs = shotNOs_raw[self.include != 0]
        self.shotNOs = shotNOs_raw[labels_raw != noUseLabel]
        #labels_log = labels[~(data == 0).any(axis=1)]
        #ata_log = np.log(data[~(data == 0).any(axis=1)])
        return 1

    def preparation(self,date='MMDD',n_split=10,scale='',log=False, seed=10):
        # d_ex = self.data[(self.data <= 0).any(axis=1)]
        # parameters = np.array(self.parameters)
        # for l in d_ex:
        #     print(parameters[l<=0])
        # pdb.set_trace()
        self.date = date
        self.output_dir = './results/'+str(date)+'/'

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        """
        #極端なデータの除外
        filtered = np.logical_not(#np.logical_and(
            self.data[:,self.parameters.index('Prad/Pinput')] > 10
        # )
        )
        # import pdb; pdb.set_trace()
        self.data = self.data[filtered]
        self.labels = self.labels[filtered]
        """
        
        if log:
            # import pdb; pdb.set_trace()
            self.labels = self.labels[(self.data > 0).all(axis=1)]
            self.data = np.log(self.data[(self.data > 0).all(axis=1)])
        else:
            # print('no log')
            self.labels = self.labels[~np.isinf(self.data).any(axis=1)]
            self.data = self.data[~np.isinf(self.data).any(axis=1)]

        # k-fold CVの分割作成
        self.skf = StratifiedKFold(n_splits=n_split,shuffle=True,random_state=seed)
        # for train, test in skf.split(self.data,self.labels) と使う

        # 正規化

        if scale == 'MinMax':
            # import pdb; pdb.set_trace()
            scaler = preprocessing.MinMaxScaler()
            
            scaler.fit(self.data)
            self.data = scaler.transform(self.data)
            

            np.savetxt(self.output_dir+'dataset_minmax.csv',
                np.vstack((scaler.data_max_,scaler.data_min_)),
                header=','.join(self.parameters),
                delimiter=','
                )
        
        # elif scale == 'bolo':
        #     import pdb; pdb.set_trace()

        np.savetxt(self.output_dir+'dataset.csv',
            self.data,
            header=','.join(self.parameters),
            delimiter=','
            )
        np.savetxt(self.output_dir+'label.csv',
            self.labels,
            header='labels',
            delimiter=','
            )

        np.savetxt(self.output_dir+'parameter.csv',
            self.parameters,
            fmt = '%s',
            delimiter=','
            )


    def CV_SVM(self,param_names=[]):
        if not param_names:
            param_names = self.parameters
        params = np.array([self.parameters.index(x) for x in param_names])
        TP = np.zeros(self.skf.get_n_splits())  #Recall
        FP = np.zeros(self.skf.get_n_splits())
        ACC = np.zeros(self.skf.get_n_splits()) #accuracy
        PRC = np.zeros(self.skf.get_n_splits()) #precision
        F1 = np.zeros(self.skf.get_n_splits()) #F1score
        #weight_list = np.array([[] for i in range(self.skf.get_n_splits())])
        #weight_list = np.zeros(self.skf.get_n_splits())
        weight_list = np.zeros((0,len(params)))
        bias_list = np.zeros(self.skf.get_n_splits())


        i = 0
        for train_index, test_index in self.skf.split(self.data,self.labels):
            # dataとlabelの分割
            train_data = self.data[train_index,:][:,params]
            train_label = self.labels[train_index]
            test_data = self.data[test_index,:][:,params]
            test_label = self.labels[test_index]
            # モデル
            if len(train_data.shape) == 1:
                #print('reshape')
                train_data = train_data.reshape(-1,1)
                test_data = test_data.reshape(-1,1)
            model = svm.SVC(kernel='linear')
            model.fit(train_data, train_label)
            predict_label = model.predict(test_data)
            # 評価値
            # CM = confusion_matrix(test_label, predict_label)
            # TP[i] = CM[0,0]/(CM[0,0]+CM[1,0])
            # FP[i] = CM[0,1]/(CM[0,1]+CM[1,1])
            TruePositive = np.count_nonzero(np.logical_and(predict_label==1, test_label==1))
            FalsePositive = np.count_nonzero(np.logical_and(predict_label==1, test_label==-1))
            TrueNegative = np.count_nonzero(np.logical_and(predict_label==-1, test_label==-1))
            FalseNegative = np.count_nonzero(np.logical_and(predict_label==-1, test_label==1))

            # print(TruePositive)
            # print(FalsePositive)
            # print(TrueNegative)
            # print(FalseNegative)

            # import pdb; pdb.set_trace()

            TP[i] = TruePositive/(TruePositive + FalseNegative)
            FP[i] = FalsePositive/(TrueNegative + FalsePositive)
            ACC[i] = (TruePositive + TrueNegative)/(TruePositive + FalseNegative + TrueNegative + FalsePositive)
            try:
                PRC[i] = TruePositive / (TruePositive + FalsePositive)
                F1[i] = 2*TP[i]*PRC[i] / (TP[i]+PRC[i])
            except ZeroDivisionError:
                #Positiveと判別されたものがない場合
                PRC[i] = 0
                F1[i] = 0
            # print(np.count_nonzero(test_label==1))
            # print(np.count_nonzero(test_label==-1))


            weight_list = np.vstack((weight_list,model.coef_))
            bias_list[i] = model.intercept_

            # print(i)
            i = i + 1

        # print(TP)
        # print(FP)
        return params, TP, FP, ACC, PRC, F1, weight_list, bias_list

    def K_main_func(self,comb_str,output_name):
        comb, TP_CV, FP_CV, ACC_CV, PRC_CV, F1_CV, weight_CV, bias_CV = self.CV_SVM(param_names=comb_str)
        print(comb)
        weight_list = [weight_CV.mean(0)[comb_str.index(x)] if x in comb_str else 0 for x in self.parameters]
        write_line = np.hstack((
        ','.join(comb.astype(str)), weight_list, bias_CV.mean(),
        TP_CV.mean(),FP_CV.mean(),ACC_CV.mean(),PRC_CV.mean(), F1_CV.mean(),
        TP_CV.std(),FP_CV.std(),ACC_CV.std(),PRC_CV.std(), F1_CV.std()
        ))
        with open(output_name, 'a') as f_handle:
            np.savetxt(f_handle, write_line.reshape(1,-1), fmt='%s', delimiter='\t',newline='\n')
        return 1

    def K_main(self, K, output_dir='./', multiFlag=False):
        output_name = output_dir + 'result{0}.tsv'.format(K)
        with open(output_name, 'w') as f_handle:
            header_weight = ['weight'+str(i) for i in range(len(self.parameters))]
            f_handle.write('Combination\t' + '\t'.join(header_weight) + '\t' + '\t'.join(['bias', 'TruePositive', 'FalsePositive', 'Accuracy', 'Precision', 'F1score', 'TruePositive_std', 'FalsePositive_std', 'Accuracy_std', 'Precision_std', 'F1score_std']) + '\n')

        if multiFlag:
            Parallel(n_jobs=-1)([delayed(self.K_main_func)(comb_str, output_name) for comb_str in list(itertools.combinations(self.parameters, K))])
        else:
            for comb_str in list(itertools.combinations(self.parameters, K)):
                self.K_main_func(comb_str,output_name)

        print(str(K)+' finish')
        return K



    def ES_main(self,start=1, end=-1, multiFlag=False):
        output_dir = self.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if end == -1:
            end = len(self.parameters)
        for K in range(start, end+1):
            print('K={}'.format(K))
            self.K_main(K,output_dir,multiFlag)
                #self.wrapper_K_main([K, output_dir])

def main(date,seed=0):
    start = time()
    ES = ExhaustiveSearch(dataname='dataset_all_prad.csv')
    ES.makeData(use_parameters=['nel', 'B', 'Pech', 'Pnbi-tan', 'Pnbi-perp', 'Pinput']) 
    ES.preparation(date=date, log=True, scale='MinMax',seed=seed)
    ES.ES_main(multiFlag=True)
    print('処理時間', (time() - start)/60, "分")

if __name__ == '__main__':
    args = sys.argv #date
    if len(args)>2:
        main(args[1],int(args[2]))
    else:
        main(args[1])
