# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:38:38 2018

@author: Ozkan
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import Dataset
import xgboost as xgb
import graphviz

# Gather Data
train_X, test_X, train_Y = Dataset.Load('AllData_v3')

# Convert data to DMatrix
dtrain = xgb.DMatrix(train_X, train_Y)
dtest = xgb.DMatrix(test_X)

# Define parameters
params = {'eta'             :0.3,
          'gamma'           :0,
          'max_depth'       :6,
          'min_child_weight':1,
          'subsample'       :1,
          'colsample_bytree':1,
          'colsample_bylevel':1,
          'lambda'          :1,
          'alpha'           :0,
          'scale_pos_weight':92/8,
          'objective'       :'binary:logistic',
          'eval_metric'     :'auc'}

# Calculate cv
cv_results = xgb.cv(params=params,
                    dtrain=dtrain,
                    num_boost_round=1000,
                    nfold=5,
                    maximize=True,
                    early_stopping_rounds=10,
                    verbose_eval=1)

# Get best number of iterations
num_boost_rounds = len(cv_results)
print(num_boost_rounds)

# Generate model by best iteration
model   = xgb.train(params=params,
                    dtrain=dtrain,
                    num_boost_round=num_boost_rounds,
                    maximize=True,
                    verbose_eval=10)

# Prediction
sub_preds = model.predict(dtest)
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_XGB_CV_v1.csv', index=False)
"""
num_boost_round=42, train-auc:0.865, valid-auc:0.772, test-auc: 0.771
"""

"""
# Plot importance
xgb.plot_importance(model,
                    max_num_features=20)

# Plot tree
xgb.plot_tree(model)
"""