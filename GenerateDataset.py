import Dataset

# Generate AllData_v3
# Dataset.SaveDf("AllData_v3")

train, test, y = Dataset.AllData_v3
print(train.shape)
print(test.shape)
print(y.shape)
