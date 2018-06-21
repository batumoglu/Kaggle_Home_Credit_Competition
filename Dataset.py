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

def Read(filename):
    with open(filename,"r") as file:
        data = {}
        section = None
        sectionData = ""
        for line in file:
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
AllData_v2              = Read("..\\input\\AllData_v2.data")
AllData_v3              = Read("..\\input\\AllData_v3.data")
ApplicationBuroBalance  = Read("..\\input\\ApplicationBuroBalance.data")
ApplicationBuro         = Read("..\\input\\ApplicationBuro.data")
ApplicationOnly         = Read("..\\input\\ApplicationOnly.data")
ApplicationBuroAndPrev  = Read("..\\input\\ApplicationBuroAndPrev.data")
AllData                 = Read("..\\input\\AllData.data")
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