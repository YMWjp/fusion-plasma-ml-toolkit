"""
ES_SVMを呼び出して，ESを実行する

実行する前に書き換える点：
    dataset_nameを指定
    parametersを指定

Usage: 
    python3.X ES.py date K
        date : 実行日（保存するファイル名になる）
        K    : 実行するK

必要なパッケージ（動かない場合に確認）：
    conda install joblib
    conda install itertools

"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from time import time

from ES_SVM import ExhaustiveSearch

dataset_name = './makedata/dataset_25_7.csv'
parameters = [
    'nel',
    'B', 
    # 'Pech', 'Pnbi-tan', 'Pnbi-perp', 
    'Pinput', 
    'Prad', 
    'Prad/Pinput', 
    # 'Wp',
    'beta',
    # 'Rax', 
    # 'rax_vmec', #'a99', #'delta_sh',
    # 'D/(H+D)',
    'CIII', 
    'CIV', 'OV', 'OVI', 'FeXVI',
    # 'Ip',
    # 'FIG', 
    # 'Pcc', 
    # 'Isat@6L', 
    # 'Isat@7L',
    # 'reff@100eV', 'ne@100eV', 'dVdreff@100eV',
    #'Te@center',
    # 'Te@edge',
    #, 'ne@center'#, 'ne_peak'
    # 'RMP_LID',
    'SDLloop_dPhi',
    'SDLloop_dTheta'
    # ,'beta_e','collision','beta0'
    # ,'fig6I', 'pcc3O', 'fig/pcc'
    # , 'ne_soxmos', 'ar_soxmos'
    ]

def main_K(date,K,seed=0):
    start = time()
    ES = ExhaustiveSearch(dataname=dataset_name)
    ES.makeData(use_parameters=parameters) 
    ES.preparation(date=date, log=True, scale='MinMax',seed=seed)
    ES.ES_main(start=K, end=K, multiFlag=True)
    print('処理時間', (time() - start)/60, "分")

def main(date,seed=0):
    start = time()
    ES = ExhaustiveSearch(dataname=dataset_name)
    ES.makeData(use_parameters=parameters) 
    ES.preparation(date=date, log=False, scale=None,seed=seed)
    ES.ES_main(multiFlag=False)
    print('処理時間', (time() - start)/60, "分")

if __name__ == '__main__':
    args = sys.argv #date
    date = args[1]
    K = int(args[2])
    main_K(date, K)
    
    # 全Kについて実行する場合
    # main(date)