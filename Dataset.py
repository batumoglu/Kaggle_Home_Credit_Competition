#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:23:21 2018
@author: U2R
"""

#region Imported modules
import pandas as pd
from io import StringIO
import Gather_Data as gd
#endregion

#region Internal module variables
_blockStartKey_ = "<!--"
_blockEndKey_ = "-->\n"

_trainSection_ = "".join([_blockStartKey_, "TRAIN\n", "[DATAFRAME]", _blockEndKey_])
_testSection_ = "".join([_blockStartKey_, "TEST\n", "[DATAFRAME]", _blockEndKey_])
_labelSection_ = "".join([_blockStartKey_, "LABEL\n", "[DATAFRAME]", _blockEndKey_])
#endregion

#region Public implementations 
def Save(data, filename=None):
    # data shall be of type string or DataFrame;
    #   string    : corresponding function name in Gather_Data module
    #   DataFrame : Pandas dataframe
    if isinstance(data,str):
        gatherDataFunc = getattr(gd, data)
        train, test, y = gatherDataFunc()
        filename = "".join([data,".data"])
    elif isinstance(data, tuple):
        train, test, y = data
    else:
        raise ValueError("Invalid type provided for param 'data'")

    with open(filename,"w") as file:
        _FlushSection_(file, train, _trainSection_)
        _FlushSection_(file, test, _testSection_)
        _FlushSection_(file, y, _labelSection_)

def SaveDf(data):
    gatherDataFunc = getattr(gd, data)
    train, test, y = gatherDataFunc()
    train.to_csv("".join([data,".train"]))
    test.to_csv("".join([data,".test"]))
    y.to_csv("".join([data,".label"]))

def ReadDf(path, dataset):
    train = pd.read_csv("".join([path, dataset, ".train"]))
    test = pd.read_csv("".join([path, dataset, ".test"]))
    label = pd.read_csv("".join([path, dataset, ".label"]))
    return (train, test, label)

def Read(filename):
    line_num = 0
    with open(filename,"r") as file:
        data = {}
        section = None
        sectionData = ""
        for line in file:
            line_num += 1
            print("Processing line " + str(line_num))
            if line[:4] == _blockStartKey_:
                section = line[4:len(line)-1]
            elif _blockEndKey_ in line and section is not None:
                data[section] = sectionData
                section = None
                sectionData = ""
            elif len(line) > 0:
                sectionData += line

    train = _StringToDataframe_(data["TRAIN"])
    test = _StringToDataframe_(data["TEST"])
    label = _StringToDataframe_(data["LABEL"])
    return (train, test, label)
#endregion

#region Internal implementations
def _FlushSection_(stream, data, section):
    dataStr = StringIO() 
    data.to_csv(dataStr)
    stream.write(section.replace("[DATAFRAME]", dataStr.getvalue()))

def _StringToDataframe_(data):
    strIO = StringIO(data)
    return pd.read_csv(strIO,sep=",")
#endregion

#region Public fields
# Gather_Data datasets
AllData_v2              = (0,0,0)
AllData_v3              = (0,0,0)
ApplicationBuroBalance  = (0,0,0)
ApplicationBuro         = (0,0,0)
ApplicationOnly         = (0,0,0)
ApplicationBuroAndPrev  = (0,0,0)
AllData                 = (0,0,0)

def LoadDatasets():
    # AllData_v2              = Read("..\\input\\AllData_v2.data")
    AllData_v3              = ReadDf("..\\input\\", "AllData_v3")
    print("AllData_v3 LOADED!!!!!!!!!!!!!!")
    train, test, y = AllData_v3
    print(train.shape)
    print(test.shape)
    print(y.shape)
    # ApplicationBuroBalance  = Read("..\\input\\ApplicationBuroBalance.data")
    # ApplicationBuro         = Read("..\\input\\ApplicationBuro.data")
    # ApplicationOnly         = Read("..\\input\\ApplicationOnly.data")
    # ApplicationBuroAndPrev  = Read("..\\input\\ApplicationBuroAndPrev.data")
    # AllData                 = Read("..\\input\\AllData.data")
#endregion

# EXAMPLE CODE
# -------------
# df_train = pd.DataFrame({"A":[10,20], "B":[30,40]})
# df_test = pd.DataFrame({"A":[60,70], "B":[80,90]})
# df_label = pd.DataFrame({"Y":[1,0]})

# Save((df_train,df_test,df_label),"test1.data")

# df_train, df_test, df_label = Read("test1.data")
# print(df_train)
# print(df_test)
# print(df_label)

if __name__ == "__main__":
    LoadDatasets()