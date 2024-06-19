import numpy as np
import matplotlib.pyplot as plt
from make_dataset4 import eg_read
import requests
import os

def plot_isat(shotNO):
    filename = f"./DivLis_tor_sum@{shotNO}.dat"
    eg_data = eg_read(filename)
    time_list = np.array(eg_data.time_list)
    Isat_data = np.array(eg_data.Isat_7L)  # 例としてIsat_7Lを使用

    plt.figure()
    plt.plot(time_list, Isat_data, label='Isat_7L')
    plt.xlabel('Time [s]')
    plt.ylabel('Isat [A]')
    plt.title(f'Shot {shotNO} Isat_7L')
    plt.legend()
    plt.grid(True)
    plt.show()


def main():
    shotNOs = np.genfromtxt('labels.csv', delimiter=',', skip_header=1, usecols=0, dtype=int)
    temp_folder = 'temp_data'
    # 一時フォルダが存在しない場合、作成する
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    
    for shotNO in shotNOs:
        # 一時ファイル名を指定（フォルダを含む）
        temp_filename = os.path.join(temp_folder, f"temp_DivIis_tor_sum@{shotNO}.dat")
        ftpGetFromHttp(shotNO, "DivIis_tor_sum", subshotNO=1, savename=temp_filename)
        plot_isat(shotNO)
        # 一時ファイルを削除
        os.remove(temp_filename)
    
    # 一時フォルダを削除（フォルダが空であることを確認）
    os.rmdir(temp_folder)

def ftpGetFromHttp(shotNO, diagname, subshotNO=1, savename=''):
    # HTTPサーバーからデータを取得するURLを構築
    url = 'http://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi?cmd=getfile&diag={0}&shotno={1}&subno={2}'.format(diagname, shotNO, subshotNO)
    response = requests.get(url)

    # HTTPリクエストが成功した場合
    if response.status_code == 200:
        # savenameが空の場合、デフォルトのファイル名を生成
        if savename == '':
            savename = '{0}@{1}.dat'.format(diagname, shotNO)
        # テキストデータを指定されたファイル名で保存
        with open(savename, 'w') as f:
            f.write(response.text)
    else:
        # HTTPリクエストが失敗した場合、エラーを表示
        print(response.status_code)
        print(f'error in HTTP request: {diagname} {shotNO} {subshotNO}')
    
    # HTTPステータスコードを返す
    return response.status_code

if __name__ == '__main__':
    main()