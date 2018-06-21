import pandas as pd
from io import StringIO
import Gather_Data as gd

blockStartKey = "<!--"
blockEndKey = "-->\n"

trainSection = "".join([blockStartKey, "TRAIN\n", "[DATAFRAME]", blockEndKey])
testSection = "".join([blockStartKey, "TEST\n", "[DATAFRAME]", blockEndKey])
labelSection = "".join([blockStartKey, "LABEL\n", "[DATAFRAME]", blockEndKey])

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
        _FlushSection_(file, train, trainSection)
        _FlushSection_(file, test, testSection)
        _FlushSection_(file, y, labelSection)

def Read(filename):
    with open(filename,"r") as file:
        data = {}
        section = None
        sectionData = ""
        for line in file:
            if line[:4] == blockStartKey:
                section = line[4:len(line)-1]
            elif blockEndKey in line and section is not None:
                data[section] = sectionData
                section = None
                sectionData = ""
            elif len(line) > 0:
                sectionData += line

    train = _StringToDataframe_(data["TRAIN"])
    test = _StringToDataframe_(data["TEST"])
    label = _StringToDataframe_(data["LABEL"])
    return (train, test, label)

@property
def AllData_v2():
    return Read("..\\input\\AllData_v2.data")

@property
def AllData_v3():
    return Read("..\\input\\AllData_v3.data")

@property
def ApplicationBuroBalance():
    return Read("..\\input\\ApplicationBuroBalance.data")

@property
def ApplicationBuro():
    return Read("..\\input\\ApplicationBuro.data")

@property
def ApplicationOnly():
    return Read("..\\input\\ApplicationOnly.data")

@property
def ApplicationBuroAndPrev():
    return Read("..\\input\\ApplicationBuroAndPrev.data")

@property
def AllData():
    return Read("..\\input\\AllData.data")

def _FlushSection_(stream, data, section):
    dataStr = StringIO() 
    data.to_csv(dataStr)
    stream.write(section.replace("[DATAFRAME]", dataStr.getvalue()))

def _StringToDataframe_(data):
    strIO = StringIO(data)
    return pd.read_csv(strIO,sep=",")



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