import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from collections import Counter
import Dataset

X, test, y = Dataset.Load("AllData_v4")

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=47, stratify=y)

print("Replacing Inf values")
X = X_train.replace([np.inf, -np.inf], np.nan)
print("Replacing NaN values")
X = X.fillna(0)

y = pd.DataFrame(y_train)

columns_X = {}
columns_y = {}

for idx, col in enumerate(X.columns):
    columns_X[idx] = col

num_class_0 = len(y[y["TARGET"]==0])
num_class_1 = len(y[y["TARGET"]==1])
num_total = len(y)

current_ratio = num_class_1 / num_total
ratio = current_ratio * 3

num_sample_to_add = (ratio*num_total-num_class_1)/(1-ratio)

print("current ratio: " + str(current_ratio))
print("desired ratio: " + str(ratio))
print("to add: " + str((int)(num_sample_to_add)))

print("Shape of dataset before sampling: " + str(y.shape))

sm = SMOTE(random_state=47, ratio={1:(int)(num_sample_to_add)})
X_res, y_res = sm.fit_sample(X, y)

print('Resampled dataset shape %s' % Counter(y_res))

X = pd.DataFrame(X_res)
y = pd.DataFrame(y_res)

num_class_0 = len(y[y[0]==0])
num_class_1 = len(y[y[0]==1])
num_total = len(y)

current_ratio = num_class_1 / num_total

print("ratio after add: " + str(current_ratio))
print("total num after add " + str(len(y)))

X = X.rename(columns=columns_X)
y = y.rename(columns={0:"TARGET"})


X.to_csv("../input/AllData_v4_os.train")
y.to_csv("../input/AllData_v4_os.label")
X_valid.to_csv("../input/AllData_v4_os_valid.train")
y_valid.to_csv("../input/AllData_v4_os_valid.label")