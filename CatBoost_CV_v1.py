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
import catboost as cat

# Gather Data
train_X, test_X, train_Y = Dataset.Load('AllData_v3')

# Convert data to DMatrix
cat_train = cat.Pool(train_X, train_Y)
cat_test = cat.Pool(test_X)

# Define parameters
params = {'loss_function'   :'Logloss',
          'eval_metric'     :'AUC',
          'random_seed'     :1453,
          'l2_leaf_reg'     :3,
          'od_type'         :'Iter',
          'depth'           :6,
          'scale_pos_weight':92/8,
          'od_wait'         :50}

# Calculate cv
cv_results = cat.cv(pool        =cat_train,
                    params      =params,
                    iterations  =2000,
                    fold_count  =5,
                    logging_level='Verbose')

# Get best number of iterations
num_boost_rounds = len(cv_results)
print(num_boost_rounds)

# Generate model by best iteration
model   = cat.train(params=params,
                    pool=cat_train,
                    num_boost_round=num_boost_rounds,
                    logging_level='Verbose')

#Prediction
sub_preds = model.predict(cat_test)
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_CatBoost_CV_v1.csv', index=False)