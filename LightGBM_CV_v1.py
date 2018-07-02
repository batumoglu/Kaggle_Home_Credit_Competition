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
import lightgbm as lgb
import graphviz

# Gather Data
train_X, test_X, train_Y = Dataset.Load('ApplicationOnly')

# Convert data to DMatrix
lgb_train = lgb.Dataset(train_X, train_Y)
lgb_test = lgb.Dataset(test_X)

# Define parameters
params = {'task'                    :'train',
          'objective'               :'binary',
          'learning_rate'           :0.1,
          'num_leaves'              :31,
          'max_depth'               :-1,
          'min_data_in_leaf'        :20,
          'min_sum_hessian_in_leaf' :0.001,
          'lambda_l1'               :0,
          'lambda_l2'               :0,
          'scale_pos_weight'        :92/8,
          'metric'                  :'auc'}

# Calculate cv
cv_results = lgb.cv(params=params,
                    train_set=lgb_train,
                    num_boost_round=1000,
                    nfold=5,
                    early_stopping_rounds=50,
                    verbose_eval=1)

# Get best number of iterations
num_boost_rounds = len(cv_results['auc-mean'])
print(num_boost_rounds)

# Generate model by best iteration
model   = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=num_boost_rounds,
                    verbose_eval=1)

# Plot importance
lgb.plot_importance(model,
                    max_num_features=20)

# Plot tree
lgb.plot_tree(model)