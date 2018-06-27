import Dataset
import pandas as pd

# Save datasets from Gather_Data module to a file on disk
# created file's name shall be the dataset name with .data extension
# E.g. AllData_v2.data
Dataset.Save("AllData")
Dataset.Save("AllData_v2")
Dataset.Save("AllData_v3")
Dataset.Save("ApplicationBuroBalance")

# Save specified dataset to a file on disk
# Filename argument must be specified
# E.g. AllData_v2.data
df_train = pd.DataFrame({"A":[10,20], "B":[30,40]})
df_test = pd.DataFrame({"A":[60,70], "B":[80,90]})
df_label = pd.DataFrame({"Y":[1,0]})
dataset = (df_train,df_test,df_label)
Dataset.Save(dataset, "DummyDataset.data")

# Read Gather_Data dataset from a file on disk
alldata_v2 = Dataset.AllData_v2
alldata_v3 = Dataset.AllData_v3
applicationBuroBalance = Dataset.ApplicationBuroBalance

# Read specified dataset from a file on disk
df_train, df_test, df_label = Dataset.Read("DummyDataset.data")