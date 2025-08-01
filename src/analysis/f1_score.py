"""Overview:
    Process results of ES-SVM for radiation-collapse study.
    Compare combinations using F1score.

Usage:
    F1score.py [-h | --help]
    F1score.py [-d|--dos] (<date>) [-k <k>]
    F1score.py [--ryoiki] (<date>) [-k <k>]
    F1score.py [--bolo] (<date>) [-k <k>]
    F1score.py [--seed] (<date>) [<ion>]
    F1score.py [--layer] (<date>) (-k <k>) [<target>]
    F1score.py [--double] (<date>) (-k <k>) [<target1>] [<target2>]
    F1score.py [--triple] (<date>) (-k <k>) [<target1>] [<target2>] [<target3>]


options:
    -h --help    : show this screen
    -d --dos     : draw DoS and weight diagram
    --ryoiki     : For ryoiki-project
    --seed       : multiple seeds
    --layer      : draw multiple DoS
    <date>       : date as MMDD
    -k <k>       : target K (optional)
    <target>     : target parameters (optional)
"""

from docopt import docopt
import os
import sys
import glob
import zipfile

import re

import pdb

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcs
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import mpl_toolkits.axes_grid1

from src.utils import common

class ResultsClass():
    def __init__(self, date, K_list, params_order=[],bolo_ch=[]):
        self.date = date #str, MMDD
        self.K_list = K_list
        self.mode = ''

        # self.bins = np.arange(0,1.01,0.01)
        self.bins = np.arange(0,1.02,0.02)

        self.data = []
        self.labels = []

        self.parameters = []
        self.min_list = []
        self.min_std_list = []
        self.TP_list = []
        self.FP_list = []
        self.ACC_list = []
        self.PRC_list = []
        self.F_list = []
        self.bias_list = []
        self.weight_list = []

        self.params_order_str = params_order
        self.params_order = []
        # import pdb; pdb.set_trace()

        self.namesDict = common.names_dict
        if len(bolo_ch)>0:
            d_bolo = dict(zip(bolo_ch,bolo_ch))
            self.namesDict.update(d_bolo)

    def load_dataset(self):
        f = open('./results/'+self.date+'/dataset.csv','r')
        self.parameters = f.readline().lstrip('# ').rstrip('\n').split(',')
        # print(self.parameters)

        if len(self.params_order_str) > 0:
            self.params_order = np.array([self.parameters.index(s) for s in self.params_order_str])
        else:
            self.params_order = np.arange(0,len(self.parameters),1)
        # print(self.params_order)

        self.data = np.loadtxt('./results/'+self.date+'/dataset.csv', skiprows=1, delimiter=',')
        self.labels = np.loadtxt('./results/'+self.date+'/label.csv', skiprows=1, delimiter=',')
        return 1

    def load_result(self, result_name, index, name):
        if name in index:
            return np.loadtxt(result_name, delimiter='\t', skiprows=1,  usecols=index.index(name))
        else:
            return np.array([])

    def hist1D(self, ax, data, ax_label,color='darkcyan'):
        # import pdb; pdb.set_trace()
        H = ax.hist(data, bins=self.bins,log=True, ec='k', color='darkcyan',label='All combinations')
        # H = ax.hist(data, bins=np.arange(0,1.02,0.02),log=True, ec='k', color='darkcyan',label='All combinations')
        ax.set_title("(a) Density of States",fontsize=18)
        ax.set_xlabel(ax_label,fontsize=18)
        ax.set_ylabel("Frequency",fontsize=18)
        ax.set_xticks(np.arange(0,1.01,0.1))
        ax.tick_params(axis='both', labelsize=16)
        ylim = ax.get_ylim()
        ax.set_ylim(0.9,ylim[1])
        #ax.set_yticks(np.arange(0,101,10))
        #ax.set_aspect('equal')
        # plt.show()
        return 1

    def indicator(self, ax2, grids):
        grids = grids[:,self.params_order]

        x = np.arange(1,len(grids)+2,1)
        y = np.arange(1,len(grids[0])+2,1)
        X,Y = np.meshgrid(x,y)
        cm_m = np.nanmax(np.abs(grids))
        np.place(grids,grids==0,np.nan)
        m =  np.ma.masked_where(np.isnan(grids),grids)
        C = ax2.pcolor(X,Y,m.T,cmap='jet',vmax=cm_m, vmin=-1*cm_m)
        cbar = plt.colorbar(C, ax=ax2)
        cbar.set_label('Weight in decision function',fontsize=16)
        ax2.set_xticks(np.arange(1,len(grids)+1,1))
        ax2.set_yticks(np.arange(0,len(self.parameters)+2,1)-0.5)
        ax2.set_ylim(1,len(self.parameters)+1)
        # import pdb; pdb.set_trace()
        # y_parameters = ['','']+[self.parameters[i] for i in self.params_order]
        y_parameters = ['','']+[self.namesDict[self.parameters[i]] for i in self.params_order]
        ax2.set_yticklabels(y_parameters,fontsize=14)
        # import pdb; pdb.set_trace()
        ax2.invert_yaxis()
        ax2.grid(axis='x')
        ax2.tick_params(axis='x',direction='in')
        ax2.set_xlabel('Combinations',fontsize=18)
        ax2.set_title("(b) Weight Diagram",fontsize=18)
        ax_grid = ax2.twinx()
        ax_grid.set_yticks(np.arange(0,len(self.parameters)+3,1))
        ax_grid.set_ylim(1,len(self.parameters)+1)
        ax_grid.grid()
        ax_grid.set_xticklabels([])
        ax_grid.set_yticklabels([])
        ax_grid.tick_params(direction='in')
        return 1

    def dos_fig(self, K, save=False, istitle=True, sort=[],grid_num=50):
        fig = plt.figure(figsize=(13,7))
        ax = plt.subplot2grid((1,10), (0,0), colspan=6)
        ax2 = plt.subplot2grid((1,10), (0,7),colspan=4)

        result_name = './results/'+self.date+'/result'+str(K)+'.tsv'
        f = open(result_name,'r')
        index = f.readline().rstrip('\n').split('\t')
        weight_index = [index.index('weight'+str(i)) for i in range(len(self.parameters))]

        weights = np.loadtxt(result_name, delimiter='\t', skiprows=1, usecols=weight_index)
        bias = np.loadtxt(result_name, delimiter='\t', skiprows=1, usecols=index.index('bias'))

        TP = self.load_result(result_name,index,'TruePositive')
        TP_std = self.load_result(result_name,index,'TruePositive_std')
        FP = self.load_result(result_name,index,'FalsePositive')
        FP_std = self.load_result(result_name,index,'FalsePositive_std')
        ACC = self.load_result(result_name,index,'Accuracy')
        ACC_std = self.load_result(result_name,index,'Accuracy_std')
        PRC = self.load_result(result_name,index,'Precision')
        PRC_std = self.load_result(result_name,index,'Precision_std')
        F1 = self.load_result(result_name,index,'F1score')
        F1_std = self.load_result(result_name,index,'F1score_std')
        if len(F1) == 0:
            F1 = 2 * TP * PRC / (TP+PRC)
        F1[np.isnan(F1)] = 0

        self.hist1D(ax, F1, 'F1 score',color='#005B93')
        order = np.argsort(F1)[::-1]
        sorted_weight = weights[order,:]
        sorted_bias = bias[order]
        sorted_TP = TP[order]
        sorted_FP = FP[order]
        sorted_PRC = PRC[order]
        sorted_F1 = F1[order]
        max_F1 = np.max(F1)
        if len(F1_std) == 0:
            max_F1std = 0
        else:
            max_F1std = F1_std[np.argmax(F1)]

        grids = sorted_weight[0:min(grid_num, len(F1))]
        self.indicator(ax2,grids)
        self.min_list.append(np.max(F1))
        self.min_std_list.append(np.max(F1))

        # import pdb; pdb.set_trace()
        np.savetxt('output.csv',np.vstack((grids.T, np.vstack((sorted_bias[0:min(grid_num, len(F1))], sorted_F1[0:min(grid_num, len(F1))])))),delimiter=',')

        # print("self.weight_list")
        # print("self.weight_list")
        # print(self.weight_list)
        # print("self.weight_list")
        # print("self.weight_list")

        if len(self.weight_list) == 0:
            self.weight_list = sorted_weight[0,:]
            self.TP_list = np.array([sorted_TP[0]])
            self.FP_list = np.array([sorted_FP[0]])
            self.PRC_list = np.array([sorted_PRC[0]])
            self.F_list = np.array(max_F1)
            self.bias_list = np.array([sorted_bias[0]])
        else:
            self.weight_list = np.vstack((self.weight_list,sorted_weight[0,:]))
            self.TP_list = np.append(self.TP_list,sorted_TP[0])
            self.FP_list = np.append(self.FP_list,sorted_FP[0])
            self.PRC_list = np.append(self.PRC_list,sorted_PRC[0])
            self.F_list = np.append(self.F_list,max_F1)
            self.bias_list = np.append(self.bias_list,sorted_bias[0])
        # print(np.min(distance))
        # print(self.min_list)

        # if istitle:
        #     fig.suptitle('Result of ES-{0}-SVM'.format(K),fontsize=18)

        if save == True:
            return fig
        else:
            plt.show()
            plt.close()

            return -1

    def result_last(self):
        result_name = './results/'+self.date+'/result'+str(len(self.parameters))+'.tsv'
        if not os.path.exists(result_name):
            return -1
        else:
            f = open(result_name,'r')
            index = f.readline().rstrip('\n').split('\t')
            weight_index = [index.index('weight'+str(i)) for i in range(len(self.parameters))]
            self.K_list.append(len(self.parameters))

            weights = np.loadtxt(result_name, delimiter='\t', skiprows=1, usecols=weight_index)
            bias = np.loadtxt(result_name, delimiter='\t', skiprows=1, usecols=index.index('bias'))

            TP = self.load_result(result_name,index,'TruePositive')
            TP_std = self.load_result(result_name,index,'TruePositive_std')
            FP = self.load_result(result_name,index,'FalsePositive')
            FP_std = self.load_result(result_name,index,'FalsePositive_std')
            ACC = self.load_result(result_name,index,'Accuracy')
            ACC_std = self.load_result(result_name,index,'Accuracy_std')
            PRC = self.load_result(result_name,index,'Precision')
            PRC_std = self.load_result(result_name,index,'Precision_std')
            F1 = self.load_result(result_name,index,'F1score')
            F1_std = self.load_result(result_name,index,'F1score_std')
            if F1 ==[]:
                F1 = 2 * TP * PRC / (TP+PRC)

            self.min_list.append(F1)
            self.min_std_list.append(F1_std)
            # print(self.min_list)

            self.weight_list = np.vstack((self.weight_list,weights))
            self.TP_list = np.append(self.TP_list,TP)
            self.FP_list = np.append(self.FP_list,FP)
            self.F_list = np.append(self.F_list,F1)
            if 'Precision' in index:
                self.PRC_list = np.append(self.PRC_list,PRC)
            else:
                self.PRC_list = np.append(self.PRC_list,0)
            self.bias_list = np.append(self.bias_list,bias)
            return 1


    def summary_fig(self,seed=False,ion=None):
        # self.weight_list = self.weight_list.T
        # print(self.weight_list)
        self.min_list = np.array(self.min_list)
        self.min_std_list = np.array(self.min_std_list)

        fig = plt.figure(figsize=(13,7))
        ax = plt.subplot2grid((1,10), (0,0), colspan=6)
        ax2 = plt.subplot2grid((1,10), (0,7),colspan=4)
        if seed:
            from seed import seed_shade
            nums, upper, bottom, mean = seed_shade(self.date,ion=ion)
            # import pdb; pdb.set_trace()
            ax.fill_between(nums,upper,bottom,facecolor='lightblue')
        line, = ax.plot(np.arange(len(self.min_list))+1, self.min_list, 'ko-')
        
        # F1スコアの値をラベルとして追加
        for i, txt in enumerate(self.min_list):
            ax.annotate(f'{txt:.3f}', (i+1, self.min_list[i]), 
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', va='bottom', fontsize=12, color='red')
        # ax.errorbar(np.arange(len(self.min_list))+1, self.min_list, fmt='ko-',yerr=self.min_std_list/np.sqrt(10))
        # ax.plot(np.arange(len(self.min_list))+1, self.min_list, 'ko-')
        ylim = ax.get_ylim()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.tick_params(labelsize=14)
        ax.set_xlim([0.3,len(self.min_list)+0.3])
        ax.grid()
        ax.set_xlabel("K number",fontsize=16)
        ax.set_ylabel("F1 score",fontsize=16)

        ax.set_title("(a) Maximum F1 scores to K",fontsize=16)

        self.indicator(ax2, self.weight_list)
        # fig.suptitle('Summary of ES-K-SVM',fontsize=18)
        return fig

    def summary_ryoiki(self,seed=False,ion=None):
        # self.weight_list = self.weight_list.T
        # print(self.weight_list)
        self.min_list = np.array(self.min_list)
        self.min_std_list = np.array(self.min_std_list)
        # import pdb; pdb.set_trace()
        plt.rcParams['font.size'] = 16
        fig = plt.figure(figsize=(8,7))
        ax = fig.add_subplot()
        ax.plot(self.K_list[:-1], self.min_list[:-1], 'ko-')
        ylim = ax.get_ylim()
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.tick_params(labelsize=14)
        ax.hlines(self.min_list[-1],0,self.K_list[-1],linestyle='dashed',linewidths=3,colors='k')
        ax.text(1.1,self.min_list[-1]+0.001,'K={:.0f}'.format(self.K_list[-1]),fontsize=20)
        ax.set_xlim([0.3,len(self.min_list)-0.7])
        ax.grid()
        ax.set_xlabel("K number",fontsize=18)
        ax.set_ylabel("F1 score",fontsize=18)

        # plt.show()
        ax.set_title("Maximum F1 scores to K",fontsize=18)
        return fig

    def dos_main(self, layer=False, target=[], target2=[],target3=[],  seed=False,ion=None,ryoiki=False):
        self.load_dataset()
        if not os.path.exists('./process/'+self.date):
            os.makedirs('./process/'+self.date)
        if len(self.K_list) == 0:
            for i in range(len(self.parameters)-1):
                # print(i+1)
                if not os.path.exists('./results/'+self.date+'/result'+str(i+1)+'.tsv'):
                    break
                self.K_list.append(i+1)
                fig = self.dos_fig(i+1,save=True,grid_num=20)
                if fig == -1:
                    return -1
                fig.savefig('./process/'+self.date+'/dos_{0}.png'.format(i+1))
                plt.close()
                # print(self.weight_list.shape)
                # if i ==2:
                #     break
            # import pdb; pdb.set_trace()
            self.result_last()
            # print(self.TP_list)
            # import pdb; pdb.set_trace()
            if ryoiki:
                fig = self.summary_ryoiki(seed,ion=ion)
                # plt.show()
                fig.savefig('./process/'+self.date+'/summary.png')
                plt.close()
            else:
                fig = self.summary_fig(seed,ion=ion)
                # plt.show()
                fig.savefig('./process/'+self.date+'/summary.png')
                plt.close()
        else:
            for i in self.K_list:
                # print(i)
                fig = self.dos_fig(i,save=True)
                if layer:
                    ax = fig.axes[0]
                    if len(target3)>0:
                        self.hist1D_target(i, ax, target,color='#FFC6DD',alpha=0.8)
                        target_str = [str(j) for j in target]
                        self.hist1D_target(i, ax, target2,color='#FF5398')
                        target2_str = [str(j) for j in target2]
                        self.hist1D_target(i, ax, target3,color='#E00059')
                        target3_str = [str(j) for j in target3]
                        ax.legend(fontsize=18)
                        fig.savefig('./process/'+self.date+'/dos_{0}_'.format(i) + '-'.join(target_str)+'_'+ '-'.join(target2_str)+'_'+ '-'.join(target3_str)+'.png',transparent=True)
                    elif len(target2)>0:
                        self.hist1D_target(i, ax, target,color='#FFC6DD',alpha=0.8)
                        target_str = [str(j) for j in target]
                        self.hist1D_target(i, ax, target2,color='#E00059')
                        target2_str = [str(j) for j in target2]
                        ax.legend(fontsize=18)
                        fig.savefig('./process/'+self.date+'/dos_{0}_'.format(i) + '-'.join(target_str)+'_'+ '-'.join(target2_str)+'.png',transparent=True)
                    else:
                        self.hist1D_target(i, ax, target,color='#E00059')
                        target_str = [str(j) for j in target]
                        ax.legend(fontsize=18)
                        fig.savefig('./process/'+self.date+'/dos_{0}_'.format(i) + '-'.join(target_str)+'.png',transparent=True)
                # plt.show()
                else:
                    fig.savefig('./process/'+self.date+'/dos_{0}.png'.format(i))
                plt.close()
        return 1

    def hist1D_target(self, K, ax, target,save=False,color='magenta',alpha=1):
        result_name = './results/'+self.date+'/result'+str(K)+'.tsv'
        f = open(result_name,'r')
        index = f.readline().rstrip('\n').split('\t')
        weight_index = [index.index('weight'+str(i)) for i in range(len(self.parameters))]

        weights = np.loadtxt(result_name, delimiter='\t', skiprows=1, usecols=weight_index)
        bias = np.loadtxt(result_name, delimiter='\t', skiprows=1, usecols=index.index('bias'))
        F1 = self.load_result(result_name,index,'F1score')
        # for p in ax.patches:
        #     p.set_alpha(0.6)
        weights_target = weights[:,target]
        F1_target = F1[np.all(weights_target !=0, 1)]
        label_target = ','.join([self.namesDict[self.parameters[i]] for i in target])
        H = ax.hist(F1_target, bins=self.bins,log=True, ec='k', color=color,label=label_target,alpha=alpha)
        return

    def output(self):
        # import pdb; pdb.set_trace()
        output_array = np.vstack((np.array(self.K_list, dtype=int), self.TP_list, self.FP_list,self.PRC_list, self.F_list, self.weight_list.T, self.bias_list))
        output_array[np.isnan(output_array)] = 0

        output_index = np.array(['K','TruePositive','FalsePositive','Precision','F-measure']
            +['weight'+str(i) for i in range(len(self.parameters))]
            +['bias']
            )
        output = np.hstack((output_index.reshape(-1,1), output_array))
        np.savetxt('./process/'+self.date+'/summary.csv', output, delimiter=',', fmt='%s')
        return 1



def main_dos(date, K_list=[], layer=False, target=[],target2=[],target3=[], params_order=[],bolo_ch=[],ryoiki=False):
    results = ResultsClass(date, K_list,params_order=params_order,bolo_ch=bolo_ch)
    results.dos_main(layer, target,target2,target3,ryoiki=ryoiki)
    results.output()

def main_seed(date, K_list=[], layer=False, target=[], params_order=[],ion=None):
    results = ResultsClass(date, K_list,params_order=params_order)
    results.dos_main(layer, target,seed=True,ion=ion)
    results.output()

def main_nbi(date, filename='nbi.txt'):
    nbi = NbiConsider(date, filename)
    nbi.weight_hist()
    nbi.P_compare()

if __name__ == '__main__':
    args = docopt(__doc__)
    # print( args )
    params_order = []
    # params_order = ['nel', 'Prad/Pinput', 'beta', 'rax_vmec', 'a99', 'CIII', 'CIV', 'OV', 'OVI', 'FeXVI', 'FIG', 'Pcc', 'Isat', 'reff@100eV', 'ne@100eV', 'Te@center', 'ne@center','RMP_LID']
    if args['--dos']:
        if args['-k']:
            main_dos(args['<date>'], [int(x) for x in args['-k'].split(',')], params_order=params_order)
        else:
            main_dos(args['<date>'],[],params_order=params_order)
    elif args['--ryoiki']:
        if args['-k']:
            main_dos(args['<date>'], [int(x) for x in args['-k'].split(',')], params_order=params_order,ryoiki=True)
        else:
            main_dos(args['<date>'],[],params_order=params_order,ryoiki=True)
    elif args['--layer']:
        main_dos(args['<date>'], [int(x) for x in args['-k'].split(',')], layer=True, target=[int(x) for x in args['<target>'].split(',')], params_order=params_order)
    elif args['--double']:
        main_dos(args['<date>'], [int(x) for x in args['-k'].split(',')], layer=True, target=[int(x) for x in args['<target1>'].split(',')], target2=[int(x) for x in args['<target2>'].split(',')],
        target3=[], params_order=params_order)
    elif args['--triple']:
        main_dos(args['<date>'], [int(x) for x in args['-k'].split(',')], layer=True, target=[int(x) for x in args['<target1>'].split(',')], target2=[int(x) for x in args['<target2>'].split(',')], target3=[int(x) for x in args['<target3>'].split(',')], params_order=params_order)
    elif args['--seed']: #乱数シードを変化させたときのやつ
        main_seed(args['<date>'],[],params_order=params_order,ion=args['<ion>'])
    elif args['--bolo']: #ボロメータのやつ
        # ch_list = ['ch'+str(i+1) for i in range(20)]
        ch_list = ['ch'+str(i+3) for i in range(20-6)] # ch3-16
        main_dos(args['<date>'],[],params_order=params_order,bolo_ch=ch_list)