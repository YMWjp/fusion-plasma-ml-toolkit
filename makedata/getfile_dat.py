import os
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), '../PyLHD'))
from igetfile import *

def getfile_dat(shotNO, diag, datapath=''):
    isfile = 1
    # import pdb; pdb.set_trace()
    outputname = datapath+'{0}@{1}.dat'.format(diag, shotNO)
    if os.path.isfile(outputname):
        print(outputname, ": exist")
        return 1
    # print(diag)
    # print(shotNO.dtype)
    # import pdb; pdb.set_trace()
    print(outputname, ": not exist")
        
    # igetfile.py版
    try:
        print('igetfile start')
        if igetfile(diag, shotNO, 1, outputname) is None:
            print('igetfile')
            print('shot:{0} diag:{1} is not exist'.format(shotNO, diag))
            isfile = -1
    except:
        print('Bad Zip File ERROR')
        return 2
    return isfile

if __name__=='__main__':
    args = sys.argv
    getfile_dat(int(args[1]), args[2])