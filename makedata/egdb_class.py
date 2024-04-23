import re
import math
import os
import numpy

class egdb():
    """
    egdb() is pure python module for parsing EGDB formatted file.
    egdb()はEGDB形式のファイルを分解して読み込むためのモジュールです。

    if you want to open 'temp.dat', make instance of the file with edgb(),
    and then call readFile() method.
    'temp.dat'というファイルを開きたい場合は、egdb()を使ってインスタンスを作成し、
    readFile()メソッドを実行します。
    ex)
        egfile = egdb('temp.dat')
        egfile.readFile()

    """
    def __init__(self, fname=None, delim=',', tdic={}):
        """
        fname    full path of the file you want to open.
                 開きたいファイルのフルパスを設定します。
        delim    delimiter of the data.
                 データの区切り文字を指定します。
        tdic     the dectionary for translation from non-numeric value
                 to the value in the dictionary.
                 非数値の文字を、辞書を使って変換します。
                 ex) tdic = {'+Inf': '99999.9', '-Inf': '-99999.9'}
        """
        self.fileName = fname
        self.delim = delim
        self.parameters = ''
        self.comments = ''
        self.Name = ''
        self.ShotNo = ''
        self.Date = ''
        self.DimNo=0
        self.DimSize=[]
        self.DimName=[]
        self.DimUnit=[]
        self.ValNo=0
        self.ValName=[]
        self.ValUnit=[]
        self.parametersDict = {}
        self.parametersFlg = False
        self.commentsFlg = False
        self.dataFlg = False
        self.idx = []
        self.invidx = []
        self.idxn = [] #各Dimのインデックスカウンタ
        self.data = []
        self.dataSize = 0
        #self.dimbackup = [] # 前の行のDimを保存しておく
        self.transdic = tdic
        self.reobjTitle = re.compile(r'\s*\[(.+)\]\s*')
        self.reobjValue = re.compile(r'(.+?)\s*=\s*(.+)')
        self.linecount = 0

    def parseLine(self, line):
        """
        parsing a line of headder and getting the size of the data.
        ヘッダを解析して、メモリサイズを得て、適切なサイズのリストを作成する
        """
        #reobjTitle = re.compile(r'\s*\[(.+)\]\s*')
        m = self.reobjTitle.match(line)
        if m:
            idx = m.groups()[0].upper()
            if 'PARAMETERS' == idx:
                self.parametersFlg = True
                self.commentsFlg = False
                self.dataFlg = False
            if 'COMMENTS' == idx:
                self.parametersFlg = False
                self.commentsFlg = True
                self.dataFlg = False
            if 'DATA' == idx:
                self.parametersFlg = False
                self.commentsFlg = False
                self.dataFlg = True
            return

        #reobjValue = re.compile(r'(.+)\s*=\s*(.+)')
        m = self.reobjValue.match(line)
        if m:
            #print m.groups()
            nm = m.groups()[0].upper()
            nm = nm.strip()
            vl = m.groups()[1]
            if 'NAME' == nm:
                clm = vl.strip()
                clm = clm.strip('\'')
                self.Name = clm
            if 'SHOTNO' == nm:
                clm = vl.strip()
                self.ShotNo = clm
            if 'DATE' == nm:
                clm = vl.strip()
                clm = clm.strip('\'')
                self.Date = clm

            if 'DIMNO' == nm:
                self.DimNo = int(vl)
                for i in range(self.DimNo):
                    self.idx.append({})
                    self.invidx.append({})
                    self.idxn.append(0)
                    #self.dimbackup.append('')
            if 'DIMSIZE' == nm:
                for i in range(self.DimNo):
                    clm = vl.split(',')
                    if len(clm) != self.DimNo:
                        clm = re.split(' +?', vl)
                    self.DimSize.append(int(clm[i]))
            if 'DIMNAME' == nm:
                for i in range(self.DimNo):
                    clm = vl.split(',')
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.DimName.append(clmd)
            if 'DIMUNIT' == nm:
                for i in range(self.DimNo):
                    clm = vl.split(',')
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.DimUnit.append(clmd)
            if 'VALNO' == nm:
                self.ValNo = int(vl)
                for i in range(self.ValNo):
                    self.data.append([])
                    self.dataSize = 1
                    for j in range(self.DimNo):
                        self.dataSize = self.dataSize * self.DimSize[j]
                    for j in range(self.dataSize):
                        self.data[i].append(0.0)
            if 'VALNAME' == nm:
                for i in range(self.ValNo):
                    clm = vl.split(',')
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.ValName.append(clmd)
            if 'VALUNIT' == nm:
                for i in range(self.ValNo):
                    clm = vl.split(',')
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.ValUnit.append(clmd)
            self.parametersDict[nm] = vl

    def makeIndex(self, idxlist):
        """
        calculate the position in the list data stored from the list of indexs.
        各変数に対するインデックス番号のリストからメモリ位置を示すインデックスを得る
        ex) get value at [5, 10, 1]
            pos = makeIndex([5, 10])
            v = egfile.data[1][pos]

        """
        pos = 0
        for i in range(self.DimNo):
            weight = 1
            for j in range(i):
                weight = weight * self.DimSize[j]
            pos = pos + weight * idxlist[i]
        return pos


    def parseData(self, line):
        """
        parsing a line of data and set the value at the proper postion in the list.
        1行分のデータを変数の適切なメモリ位置に登録していく
        """
        clm = line.split(self.delim)
        idxl = []
        #dimbackup = []
        linecount = self.linecount

        for i in range(self.DimNo):
            if i < self.DimNo-1:
                blocksize = self.DimSize[i-1]
            else:
                blocksize = 1

            idxl.append(self.idxn[i])
            label = "%s_%d" % (clm[i], self.idxn[i])
            self.idx[i][label] = self.idxn[i] #辞書に登録
            self.invidx[i][self.idxn[i]] = clm[i] #逆引き辞書
            if 0 == ((linecount+1) % blocksize): #つぎのブロックに移動したら
                self.idxn[i] = self.idxn[i]+1
                if self.idxn[i] >= self.DimSize[i]: #カウンターがオーバーしたら
                    self.idxn[i] = 0

        #self.dimbackup = dimbackup

        pos = self.makeIndex(idxl)

        for i in range(self.ValNo):
            vstr = clm[i + self.DimNo].strip(' ')
            #vstr = vstr.strip('\n')
            #if vstr[-1] == '\n':
            #    vstr = vstr[:-1]
            #if self.transdic.has_key(vstr):
            #    vstr = self.transdic[vstr]
            vstr = self.transdic.get(vstr, vstr)
            try:
                d = float(vstr)
            except(ValueError):
                logger.error("Error(%d): Could not convert string to float '%s'" % (linecount+1, vstr))
                d = 0.0
            self.data[i][pos] = d

        self.linecount = self.linecount + 1

    def readFile(self):
        """
        reading file.
        ファイルの読み込みを実行する
        """
        if self.fileName[-3:].upper() == 'ZIP':
            # open zip file
            zf = zipfile.ZipFile(self.fileName, 'r')
            targetname = zf.namelist()[0]
            buff = zf.read(targetname)
            zf.close()
            lines = buff.split('\n')
        else:
            fp = open(self.fileName, 'r')
            lines = fp.readlines()
            fp.close()
        # lines = self.fileName.readlines()

        for i, l in enumerate(lines):
            #l = l.decode()
            l = l.strip('\n')
            l = l.strip('\r')
            l = l.strip(',')
            if len(l) == 0: continue
            if l[0] == '#':
                self.parseLine(l[1:])
                if self.parametersFlg:
                    self.parameters = self.parameters+l+'\n'
                if self.commentsFlg:
                    self.comments = self.comments+l+'\n'
            else:
                if self.DimNo != 1:
                    #logger.error("Not Single Dimension Data")
                    return False
                clm = l.split(self.delim)
                self.dimdata.append(float(clm[0]))
                for j, v in enumerate(clm[1:]):
                    try:
                        vm = float(v)
                    except ValueError:
                        #print v
                        vm = numpy.nan
                    self.data[j].append(float(vm))
        return True

    def getName(self):
        """
        return the name of diagnostic
        """
        return self.Name

    def getShotNo(self):
        """
        return the shot number written in the file.
        """
        return self.ShotNo

    def getDate(self):
        """
        return the DATE written in the file.
        """
        return self.Date

    def getDimNo(self):
        """
        return the number of dimensions.
        """
        return self.DimNo

    def getDimSize(self, dim):
        """
        return the size of dimensions.
        """
        return self.DimSize[dim]

    def getValNo(self):
        """
        return the number of variables (Values).
        """
        return self.ValNo

    def getParameters(self):
        """
        return the block of the parameters.
        パラメータブロックを得る
        """
        return self.parameters

    def getComments(self):
        """
        return the block of the comments.
        コメントを得る
        """
        return self.comments

    def getParameterByName(self, name):
        """
        return the parameter in the comment block indicated by 'name ='.
        """
        nameu = name.upper()
        return self.parametersDict.get(nameu)
        #if self.parametersDict.has_key(nameu):
        #    return self.parametersDict[nameu]
        #return None

    def value2idx(self, dim, v):
        """
        get an index value of the dimension at the value near the argument value.
        ある次元dimの値vに近い場所のインデックス値を得る

        ex) idx = value2idx(0, 3.4)

        """
        keylist = self.idx[dim].keys()
        klist = [k.split('_')[0] for k in keylist]
        ilist = [int(k.split('_')[1]) for k in keylist]
        idx = []
        dlist = []
        min = math.sqrt((v - float(klist[0]))**2)
        dlist.append(min)
        for i in range(1, len(klist)):
            d = math.sqrt((v - float(klist[i]))**2)
            dlist.append(d)
            if d < min:
                min = d
        i = 0
        for d in dlist:
            if d == min:
                idx.append(ilist[i])
            i = i + 1
        return idx

    def idx2value(self, dim, idx):
        """
        get the value at the idx of the dimmension.
        ある次元dimの実際の値をインデックス値から得る

        ex) time = idx2value(self, 0, 10)
        """
        v = self.invidx[dim][idx]
        return float(v)

    def dims(self, dim):
        """
        get the list of value in the dimension.
        次元dimの値のリストをインデックス順に得る
        """
        lst = []
        for i in range(len(self.invidx[dim])):
            lst.append(float(self.invidx[dim][i]))
        return lst

    def DimNames(self):
        """
        get the list of the dimenstion names.
        次元の名前のリストを得る
        """
        return self.DimName

    def DimUnits(self):
        """
        get the list of the dimenstion names.
        次元の単位のリストを得る
        """
        return self.DimUnit

    def ValNames(self):
        """
        get the list of the variable (Value) names.
        変数の名前のリストを得る
        """
        return self.ValName

    def ValUnits(self):
        """
        get the list of the units of variable (Value).
        変数の単位のリストを得る
        """
        return self.ValUnit

    def dimname2idx(self, name):
        """
        get the index of the dimension by the name.
        次元名から、インデックスを得る
        """
        for i in range(len(self.DimName)):
            dn = self.DimName[i].upper()
            name = name.upper()
            if dn == name:
                return i
        return -1

    def valname2idx(self, name):
        """
        get the index of the variable (Value) by the name.
        変数(値)名から、インデックスを得る
        """
        for i in range(len(self.ValName)):
            vn = self.ValName[i].upper()
            name = name.upper()
            if vn == name:
                return i
        return -1

    def idx2dimname(self, idx):
        """
        get the dimension name by the index.
        インデックスから次元名を得る
        """
        return self.DimName[idx]

    def idx2dimunit(self, idx):
        """
        get the unit of the dimension by the index.
        インデックスから次元単位を得る
        """
        return self.DimUnit[idx]

    def idx2dimtitle(self, idx):
        """
        get the title of the dimension by the index.
        the title is formatted as 'name(unit)'.
        インデックスから次元タイトルを得る
        """
        return self.idx2dimname(idx)+' ('+self.idx2dimunit(idx)+')'

    def idx2valname(self, idx):
        """
        get the variable (Value) name by the index.
        インデックスから変数(値)名を得る
        """
        return self.ValName[idx]

    def idx2valunit(self, idx):
        """
        get the unit of the variable (Value) by the index.
        インデックスから変数(値)の単位を得る
        """
        return self.ValUnit[idx]

    def idx2valtitle(self, idx):
        """
        get the title of the variable (Value) by the index.
        the title is formatted as 'name(unit)'.
        インデックスから変数(値)タイトルを得る
        """
        return self.idx2valname(idx)+' ('+self.idx2valunit(idx)+')'

    def getData(self, idxlist):
        """
        get the data at the indexs in the idxlist.
        idxlistで指定されたデータを取得する
        ex) getting the data of variable (Value) 1 at (dim1, dim2) = (5, 10).
            v = egfile.getData([5, 10, 1])
        """
        if self.DimNo+1 != len(idxlist):
            return None
        pos = self.makeIndex(idxlist[:-1])
        val = idxlist[-1]
        return self.data[val][pos]



class egdb2d():
    """
    egdb() is pure python module for parsing EGDB formatted file.
    egdb()はEGDB形式のファイルを分解して読み込むためのモジュールです。

    if you want to open 'temp.dat', make instance of the file with edgb(),
    and then call readFile() method.
    'temp.dat'というファイルを開きたい場合は、egdb()を使ってインスタンスを作成し、
    readFile()メソッドを実行します。
    ex)
        egfile = egdb('temp.dat')
        egfile.readFile()

    """
    def __init__(self, fname=None, delim=',', tdic={}):
        """
        fname    full path of the file you want to open.
                 開きたいファイルのフルパスを設定します。
        delim    delimiter of the data.
                 データの区切り文字を指定します。
        tdic     the dectionary for translation from non-numeric value
                 to the value in the dictionary.
                 非数値の文字を、辞書を使って変換します。
                 ex) tdic = {'+Inf': '99999.9', '-Inf': '-99999.9'}
        """
        self.fileName = fname
        self.delim = delim
        self.parameters = ''
        self.comments = ''
        self.Name = ''
        self.ShotNo = ''
        self.Date = ''
        self.DimNo=0
        self.DimSize=0
        self.DimName=""
        self.DimUnit=""
        self.ValNo=0
        self.ValNames=[]
        self.ValUnits=[]
        self.parametersDict = {}
        self.parametersFlg = False
        self.commentsFlg = False
        self.dataFlg = False

        self.dimdata = []
        self.data = []

        self.reobjTitle = re.compile(r'\s*\[(.+)\]\s*')
        self.reobjValue = re.compile(r'(.+?)\s*=\s*(.+)')

    def parseLine(self, line):
        """
        parsing a line of headder and getting the size of the data.
        ヘッダを解析して、メモリサイズを得て、適切なサイズのリストを作成する
        """
        #reobjTitle = re.compile(r'\s*\[(.+)\]\s*')
        m = self.reobjTitle.match(line)
        if m:
            idx = m.groups()[0].upper()
            if 'PARAMETERS' == idx:
                self.parametersFlg = True
                self.commentsFlg = False
                self.dataFlg = False
            if 'COMMENTS' == idx:
                self.parametersFlg = False
                self.commentsFlg = True
                self.dataFlg = False
            if 'DATA' == idx:
                self.parametersFlg = False
                self.commentsFlg = False
                self.dataFlg = True
            return

        #reobjValue = re.compile(r'(.+)\s*=\s*(.+)')
        m = self.reobjValue.match(line)
        if m:
            #print m.groups()
            nm = m.groups()[0].upper()
            nm = nm.strip()
            vl = m.groups()[1]
            if 'NAME' == nm:
                clm = vl.strip()
                clm = clm.strip('\'')
                self.Name = clm
            if 'SHOTNO' == nm:
                clm = vl.strip()
                self.ShotNo = clm
            if 'DATE' == nm:
                clm = vl.strip()
                clm = clm.strip('\'')
                self.Date = clm

            if 'DIMNO' == nm:
                self.DimNo = int(vl)

            if 'DIMSIZE' == nm:
                for i in range(self.DimNo):
                    clm = vl.split(',')
                    #print(clm)
                    self.DimSize = int(clm[0])

            if 'DIMNAME' == nm:
                clm = vl.split(',')
                clmd = clm[0].strip()
                clmd = clmd.strip('\'')
                clmd = clmd.strip()
                self.DimName = clmd

            if 'DIMUNIT' == nm:
                clm = vl.split(',')
                clmd = clm[0].strip()
                clmd = clmd.strip('\'')
                clmd = clmd.strip()
                self.DimUnit = clmd
            if 'VALNO' == nm:
                self.ValNo = int(vl)
                for i in range(self.ValNo):
                    self.data.append([])
            if 'VALNAME' == nm:
                for i in range(self.ValNo):
                    clm = vl.split(',')
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.ValNames.append(clmd)
            if 'VALUNIT' == nm:
                for i in range(self.ValNo):
                    clm = vl.split(',')
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.ValUnits.append(clmd)
            self.parametersDict[nm] = vl

    def readFile(self):
        """
        reading file.
        ファイルの読み込みを実行する
        """
        if self.fileName[-3:].upper() == 'ZIP':
            # open zip file
            zf = zipfile.ZipFile(self.fileName, 'r')
            targetname = zf.namelist()[0]
            buff = zf.read(targetname)
            zf.close()
            lines = buff.split('\n')
        else:
            fp = open(self.fileName, 'r')
            lines = fp.readlines()
            fp.close()
        # lines = self.fileName.readlines()

        for i, l in enumerate(lines):
            #l = l.decode()
            l = l.strip('\n')
            l = l.strip('\r')
            l = l.strip(',')
            if len(l) == 0: continue
            if l[0] == '#':
                self.parseLine(l[1:])
                if self.parametersFlg:
                    self.parameters = self.parameters+l+'\n'
                if self.commentsFlg:
                    self.comments = self.comments+l+'\n'
            else:
                if self.DimNo != 1:
                    #logger.error("Not Single Dimension Data")
                    return False
                l = l.rstrip(', ')
                # print(l)
                clm = l.split(self.delim)
                # print(clm)
                self.dimdata.append(float(clm[0]))
                for j, v in enumerate(clm[1:]):
                    try:
                        vm = float(v)
                    except ValueError:
                        #print v
                        vm = numpy.nan
                    self.data[j].append(float(vm))
        return True

    def getName(self):
        """
        return the name of diagnostic
        """
        return self.Name

    def getShotNo(self):
        """
        return the shot number written in the file.
        """
        return self.ShotNo

    def getDate(self):
        """
        return the DATE written in the file.
        """
        return self.Date

    def getDimNo(self):
        """
        return the number of dimensions.
        """
        return self.DimNo

    def getDimSize(self, dim):
        """
        return the size of dimensions.
        """
        return self.DimSize[dim]

    def getValNo(self):
        """
        return the number of variables (Values).
        """
        return self.ValNo

    def getParameters(self):
        """
        return the block of the parameters.
        パラメータブロックを得る
        """
        return self.parameters

    def getComments(self):
        """
        return the block of the comments.
        コメントを得る
        """
        return self.comments

    def getParameterByName(self, name):
        """
        return the parameter in the comment block indicated by 'name ='.
        """
        nameu = name.upper()
        return self.parametersDict.get(nameu)

    def value2idx(self, v):
        """
        get an index value of the dimension at the value near the argument value.
        あるdimの値vに近い場所のインデックス値を得る

        ex) idx = value2idx(3.4)

        """
        dlist = []
        for i, dimvalue in enumerate(self.dimdata):
            d = abs(v - dimvalue)
            dlist.append([d, i])
        dlist.sort()
        idx = dlist[0][1]
        return idx

    def idx2value(self, idx):
        """
        get the value at the idx of the dimmension.
        ある次元dimの実際の値をインデックス値から得る

        ex) time = idx2value(self, 0, 10)
        """
        v = self.dimdata[idx]
        return v

    def dims(self, dim):
        """
        get the list of value in the dimension.
        次元dimの値のリストをインデックス順に得る
        """
        return self.dimdata

    def dimname(self):
        """
        get the list of the dimenstion name.
        次元の名前を得る
        """
        return self.DimName

    def dimunit(self):
        """
        get the list of the dimenstion names.
        次元の単位を得る
        """
        return self.DimUnit

    def valnames(self):
        """
        get the list of the variable (Value) names.
        変数の名前のリストを得る
        """
        return self.ValNames

    def valunits(self):
        """
        get the list of the units of variable (Value).
        変数の単位のリストを得る
        """
        return self.ValUnits

    def valname2idx(self, name):
        """
        get the index of the variable (Value) by the name.
        変数(値)名から、インデックスを得る
        """
        vnlist = [vn.upper() for vn in self.ValNames]
        if name.upper() in vnlist:
            return vnlist.index(name.upper())
        else:
            return -1

    def idx2valname(self, idx):
        """
        get the variable (Value) name by the index.
        インデックスから変数(値)名を得る
        """
        return self.ValNames[idx]

    def idx2valunit(self, idx):
        """
        get the unit of the variable (Value) by the index.
        インデックスから変数(値)の単位を得る
        """
        return self.ValUnits[idx]


class egdb3d():
    """
    egdb() is pure python module for parsing EGDB formatted file.
    egdb()はEGDB形式のファイルを分解して読み込むためのモジュールです。

    if you want to open 'temp.dat', make instance of the file with edgb(),
    and then call readFile() method.
    'temp.dat'というファイルを開きたい場合は、egdb()を使ってインスタンスを作成し、
    readFile()メソッドを実行します。
    ex)
        egfile = egdb('temp.dat')
        egfile.readFile()

    """
    def __init__(self, fname=None, delim=',', tdic={}):
        """
        fname    full path of the file you want to open.
                 開きたいファイルのフルパスを設定します。
        delim    delimiter of the data.
                 データの区切り文字を指定します。
        tdic     the dectionary for translation from non-numeric value
                 to the value in the dictionary.
                 非数値の文字を、辞書を使って変換します。
                 ex) tdic = {'+Inf': '99999.9', '-Inf': '-99999.9'}
        """
        self.fileName = fname
        self.delim = delim
        self.parameters = ''
        self.comments = ''
        self.Name = ''
        self.ShotNo = ''
        self.Date = ''
        self.DimNo=2
        self.DimSizes=[0, 0]
        self.DimNames=["", ""]
        self.DimUnits=["", ""]
        self.ValNo=0
        self.ValNames=[]
        self.ValUnits=[]
        self.parametersDict = {}
        self.parametersFlg = False
        self.commentsFlg = False
        self.dataFlg = False
        self.tdic = tdic
        self.dimdata = [[], []]
        self.data = []

        self.reobjTitle = re.compile(r'\s*\[(.+)\]\s*')
        self.reobjValue = re.compile(r'(.+?)\s*=\s*(.+)')

    def parseLine(self, line):
        """
        parsing a line of headder and getting the size of the data.
        ヘッダを解析して、メモリサイズを得て、適切なサイズのリストを作成する
        """
        #reobjTitle = re.compile(r'\s*\[(.+)\]\s*')
        m = self.reobjTitle.match(line)
        if m:
            idx = m.groups()[0].upper()
            if 'PARAMETERS' == idx:
                self.parametersFlg = True
                self.commentsFlg = False
                self.dataFlg = False
            if 'COMMENTS' == idx:
                self.parametersFlg = False
                self.commentsFlg = True
                self.dataFlg = False
            if 'DATA' == idx:
                self.parametersFlg = False
                self.commentsFlg = False
                self.dataFlg = True
            return

        #reobjValue = re.compile(r'(.+)\s*=\s*(.+)')
        m = self.reobjValue.match(line)
        if m:
            #print m.groups()
            nm = m.groups()[0].upper()
            nm = nm.strip()
            vl = m.groups()[1]
            if 'NAME' == nm:
                clm = vl.strip()
                clm = clm.strip('\'')
                self.Name = clm
            if 'SHOTNO' == nm:
                clm = vl.strip()
                self.ShotNo = clm
            if 'DATE' == nm:
                clm = vl.strip()
                clm = clm.strip('\'')
                self.Date = clm

            if 'DIMNO' == nm:
                self.DimNo = int(vl)

            if 'DIMSIZE' == nm:
                for i in range(self.DimNo):
                    clm = vl.split(',')
                    self.DimSizes[0] = int(clm[0])
                    self.DimSizes[1] = int(clm[1])
                self.dimdata[0] = [0.0 for i in range(self.DimSizes[0])]
                self.dimdata[1] = [0.0 for i in range(self.DimSizes[1])]

            if 'DIMNAME' == nm:
                clm = vl.split(',')
                for i in range(self.DimNo):
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.DimNames[i] = clmd

            if 'DIMUNIT' == nm:
                clm = vl.split(',')
                for i in range(self.DimNo):
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.DimUnits[i] = clmd

            if 'VALNO' == nm:
                self.ValNo = int(vl)
                for i in range(self.ValNo):
                    self.data.append([])
                self.data = [0.0 for i in
                        range(self.DimSizes[0] * self.DimSizes[1] * self.ValNo)]
            if 'VALNAME' == nm:
                for i in range(self.ValNo):
                    clm = vl.split(',')
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.ValNames.append(clmd)
            if 'VALUNIT' == nm:
                for i in range(self.ValNo):
                    clm = vl.split(',')
                    clmd = clm[i].strip()
                    clmd = clmd.strip('\'')
                    clmd = clmd.strip()
                    self.ValUnits.append(clmd)
            self.parametersDict[nm] = vl

    def readFile(self):
        """
        reading file.
        ファイルの読み込みを実行する
        """
        if self.fileName[-3:].upper() == 'ZIP':
            # open zip file
            zf = zipfile.ZipFile(self.fileName, 'r')
            targetname = zf.namelist()[0]
            buff = zf.read(targetname)
            zf.close()
            lines = buff.split('\n')
        else:
            fp = open(self.fileName, 'r')
            lines = fp.readlines()
            fp.close()


        c0 = 0
        c1 = 0
        flg = True # Check First for Dim(0)
        for i, l in enumerate(lines):
            l = l.strip('\n')
            l = l.strip(',')
            if len(l) == 0: continue
            if l[0] == '#':
                self.parseLine(l[1:])
                if self.parametersFlg:
                    self.parameters = self.parameters+l+'\n'
                if self.commentsFlg:
                    self.comments = self.comments+l+'\n'
            else:
                if self.DimNo != 2:
                    logger.error("Not Two Dimension Data")
                    return False
                clm = l.split(self.delim)

                if flg:
                    v = float(clm[0])
                    self.dimdata[0][c0] = v
                    c1 = 0
                    c0 = c0 + 1
                    flg = False

                if c1 >= self.DimSizes[1]:
                    v = float(clm[0])
                    self.dimdata[0][c0] = v
                    c1 = 0
                    c0 = c0 + 1

                self.dimdata[1][c1] = float(clm[1])
                c1 = c1 + 1

                for j, v in enumerate(clm[2:]):
                    idx = j + (c1-1)*self.ValNo + (c0-1)*self.DimSizes[1]*self.ValNo
                    #print c0, c1, self.DimSizes[0], self.DimSizes[1], self.ValNo, idx
                    try:
                        self.data[idx] = float(v)
                    except ValueError:
                        v = self.tdic.get(v, None)
                        if v:
                            self.data[idx] = float(v)
                        else:
                            self.data[idx] = 0.0


        return True

    def position(self, idx_dim0, idx_dim1, idx_val):
        """
        return index of (idx_dim0, idx_dim1, idx_val)
        例
        idx_dim0: 時刻tの範囲 eg.value2idx(t, 0<-tのindex)
        idx_dim1: 上の範囲で，ある位置を表すindex
        idx_val: valueのindex
        """
        idx = idx_val + idx_dim1*self.ValNo + idx_dim0*self.DimSizes[1]*self.ValNo
        return idx

    def getName(self):
        """
        return the name of diagnostic
        """
        return self.Name

    def getShotNo(self):
        """
        return the shot number written in the file.
        """
        return self.ShotNo

    def getDate(self):
        """
        return the DATE written in the file.
        """
        return self.Date

    def getDimNo(self):
        """
        return the number of dimensions.
        """
        return self.DimNo

    def getDimSize(self, dim):
        """
        return the size of dimensions.
        """
        return self.DimSizes[dim]

    def getValNo(self):
        """
        return the number of variables (Values).
        """
        return self.ValNo

    def getParameters(self):
        """
        return the block of the parameters.
        パラメータブロックを得る
        """
        return self.parameters

    def getComments(self):
        """
        return the block of the comments.
        コメントを得る
        """
        return self.comments

    def getParameterByName(self, name):
        """
        return the parameter in the comment block indicated by 'name ='.
        """
        nameu = name.upper()
        return self.parametersDict.get(nameu)

    def value2idx(self, v, dim):
        """
        get an index value of the dimension at the value near the argument value.
        あるdimの値vに近い場所のインデックス値を得る

        ex) idx = value2idx(3.4)

        """
        dlist = []
        for i, dimvalue in enumerate(self.dimdata[dim]):
            d = abs(v - dimvalue)
            dlist.append([d, i])
        dlist.sort()
        idx = dlist[0][1]
        return idx

    def idx2value(self, idx, dim):
        """
        get the value at the idx of the dimmension.
        ある次元dimの実際の値をインデックス値から得る

        ex) time = idx2value(self, 0, 10)
        """
        v = self.dimdata[dim][idx]
        return v

    def dims(self, dim):
        """
        get the list of value in the dimension.
        次元dimの値のリストをインデックス順に得る
        """
        return self.dimdata[dim]

    def dimname(self, dim):
        """
        get the list of the dimenstion name.
        次元の名前を得る
        """
        return self.DimNames[dim]

    def dimunit(self, dim):
        """
        get the list of the dimenstion names.
        次元の単位を得る
        """
        return self.DimUnits[dim]

    def valnames(self):
        """
        get the list of the variable (Value) names.
        変数の名前のリストを得る
        """
        return self.ValNames

    def valunits(self):
        """
        get the list of the units of variable (Value).
        変数の単位のリストを得る
        """
        return self.ValUnits

    def valunit(self, vidx):
        """
        get the list of the unit of variable with validx(Value).
        vidx 番目の変数の単位を得る
        """
        return self.ValUnits[vidx]

    def valname2idx(self, name):
        """
        get the index of the variable (Value) by the name.
        変数(値)名から、インデックスを得る
        """
        vnlist = [vn.upper() for vn in self.ValNames]
        if name.upper() in vnlist:
            return vnlist.index(name.upper())
        else:
            return -1

    def idx2valname(self, idx):
        """
        get the variable (Value) name by the index.
        インデックスから変数(値)名を得る
        """
        return self.ValNames[idx]

    def idx2valunit(self, idx):
        """
        get the unit of the variable (Value) by the index.
        インデックスから変数(値)の単位を得る
        """
        return self.ValUnits[idx]

    def dimname2idx(self, name):
        """
        get the index of the dimension by the name.
        次元名から、インデックスを得る
        """
        for i in range(len(self.DimNames)):
            dn = self.DimNames[i].upper()
            name = name.upper()
            if dn == name:
                return i
        return -1
