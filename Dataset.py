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
_trainfilename_ = "../input/[filename].train"
_testfilename_ = "../input/[filename].test"
_labelfilename_ = "../input/[filename].label"
#endregion

#region Public implementations 
def Save(dataset_name):
    ''' Stores specified dataset into disk
    Args:
        dataset_name     (str) : name of dataset function in Gather_Data module

    Returns:
        No return value(s)
    '''
    if isinstance(dataset_name,str):
        print("Searching " + dataset_name + " dataset...")
        gatherDataFunc = getattr(gd, dataset_name)
        print("Loading " + dataset_name + " dataset...")
        train, test, y = gatherDataFunc()
    else:
        raise ValueError("No matching dataset function found in Gather_Data")

    print("Writing train dataset to disk...")
    train.to_csv(_trainfilename_.replace("[filename]", dataset_name), index=False)
    print("Writing test dataset to disk...")
    test.to_csv(_testfilename_.replace("[filename]", dataset_name), index=False)
    print("Writing labels to disk...")
    y.to_csv(_labelfilename_.replace("[filename]", dataset_name), index=False, header=True)
    print("Dataset files have been successfully created...")


def Load(dataset_name):
    print("Loading " + dataset_name + " dataset files...")
    try:
        filenames = _GetFileNames_(dataset_name)
        train = pd.read_csv(filenames[0])
        test = pd.read_csv(filenames[1])
        label = pd.read_csv(filenames[2])
        print(dataset_name + " dataset files have been successfully loaded...")
    except:
        train, test, label = None, None, None
        print("ERROR: An error occured while loading dataset from file.")
    return (train, test, label["TARGET"])

#endregion

#region Internal implementations

def _GetFileNames_(dataset_name):
    return (_trainfilename_.replace("[filename]", dataset_name),
    _testfilename_.replace("[filename]", dataset_name),
    _labelfilename_.replace("[filename]", dataset_name))

#endregion

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        func = sys.argv[1]
        dataset_name = sys.argv[2]
        if func == "save":
            Save(dataset_name)
        elif func == "load":
            Load(dataset_name)


