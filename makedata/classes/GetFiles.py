import glob
import os

import numpy as np
from igetfile import igetfile


class GetFiles(object):
    def __init__(self, shotNO, diag_list='diagnames.csv'):
        self.shotNO = shotNO
        # import pdb; pdb.set_trace()
        print(diag_list)
        self.diag_list = np.genfromtxt(diag_list,dtype='str',usecols=(0),delimiter=',')
        self.missing_list = []

    def getfile_dat(self):
        isfile = 1
        for diag in self.diag_list:
            # import pdb; pdb.set_trace()
            outputname = '{0}@{1}.dat'.format(diag, self.shotNO)
            if os.path.isfile(outputname):
                print(outputname, ": exist")
                continue
            # print(diag)
            # print(self.shotNO.dtype)
            # import pdb; pdb.set_trace()
            print(outputname, ": not exist")
            
        # igetfile.py版
            try:
                if igetfile(diag, self.shotNO, 1, outputname) is None:
                    # import pdb; pdb.set_trace()
                    print('shot:{0} diag:{1} is not exist'.format(self.shotNO, diag))
                    self.missing_list.append(diag)
                    isfile = -1
            except ValueError:
                print('Bad Zip File ERROR')
                return 2
        return isfile


    def remove_files(self):
        """
        "*_shotNO_*.zip"
        "*@shotNO.dat"
        を削除したい
        """
        print("*****remove*****")
        zipnames = glob.glob('*_'+str(self.shotNO)+'.zip')
        datnames = glob.glob('*@'+str(self.shotNO)+'.dat')

        print(zipnames)
        print(datnames)

        for name in zipnames:
            os.remove(name)

        for name in datnames:
            os.remove(name)

        return 1