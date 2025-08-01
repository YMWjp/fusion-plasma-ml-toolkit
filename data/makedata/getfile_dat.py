import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../PyLHD'))
from igetfile import *

# def getfile_dat(shotNO, diag, datapath=''):
#     isfile = 1
#     # import pdb; pdb.set_trace()
#     outputname = datapath+'{0}@{1}.dat'.format(diag, shotNO)
#     if os.path.isfile(outputname):
#         print(outputname, ": exist")
#         return 1
#     # print(diag)
#     # print(shotNO.dtype)
#     # import pdb; pdb.set_trace()
#     print(outputname, ": not exist")
        
#     # igetfile.py版
#     try:
#         print('igetfile start')
#         if igetfile(diag, shotNO, 1, outputname) is None:
#             print('shot:{0} diag:{1} is not exist'.format(shotNO, diag))
#             isfile = -1
#     except:
#         print('Bad Zip File ERROR')
#         return 2
#     return isfile

def getfile_dat(shotNO, diag, datapath=''):
    # デバッグ用のコメントアウト（必要に応じて有効化してください）
    # import pdb; pdb.set_trace()

    subshotNO = 1  # サブショット番号の設定
    # HTTPリクエスト用のURLを生成
    url = 'http://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi?cmd=getfile&diag={}&shotno={}&subno={}'.format(diag, shotNO, subshotNO)

    # HTTPリクエストを送信
    response = requests.get(url)

    # データパスの設定（デフォルトの場合はファイル名を生成）
    if datapath == '':
        datapath = '{0}@{1}.dat'.format(diag, shotNO)

    # レスポンスの処理
    if response.status_code == 200:
        # ファイルが存在するかチェック
        if os.path.isfile(datapath):
            print(datapath, ": exist")
            return 1
        # ファイルが存在しない場合は新しく作成
        with open(datapath, 'w') as f:
            f.write(response.text)
        print(datapath, ": created")
    else:
        # ステータスコードとエラーメッセージを出力
        print('Error in HTTP request:', response.status_code)
        print('Failed request URL:', url)
    
    return response.status_code

# if __name__ == "__main__":
#     args = sys.argv
#     getdata(int(args[1]),args[2])

if __name__=='__main__':
    args = sys.argv
    getfile_dat(int(args[1]), args[2])

    