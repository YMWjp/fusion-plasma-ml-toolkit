import requests
import sys

def getdata(shotNO,diagname,subshotNO=1,savename=''):
    # import pdb; pdb.set_trace()
    response = requests.get(
        'http://exp.lhd.nifs.ac.jp/opendata/LHD/webapi.fcgi?cmd=getfile&diag={0}&shotno={1}&subno={2}'.format(diagname,shotNO,subshotNO)
        )

    if response.status_code == 200:
        if savename == '':
            savename = '{0}@{1}.dat'.format(diagname,shotNO)
        with open(savename,'w') as f:
            f.write(response.text)
    else:
        print(response.status_code)
        print('error in HTTP request')
    
    return response.status_code

if __name__ == "__main__":
    args = sys.argv
    getdata(int(args[1]),args[2])