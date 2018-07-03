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
#import graphviz

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


# Search over certain Grid
Grid = []
for num_leaves in range(10,51,10):
    params['num_leaves'] = num_leaves
    cv_results = lgb.cv(params=params,
                        train_set=lgb_train,
                        num_boost_round=1000,
                        nfold=5,
                        early_stopping_rounds=10,
                        verbose_eval=10)
    best = cv_results['auc-mean'][-1]
    num_boost_rounds = len(cv_results['auc-mean'])
    stat = list(params.values())
    stat.append(num_boost_rounds)
    stat.append(best)    
    Grid.append(stat)
     
# Get best results
best_result = 0
best_param = 0
for CV in Grid:
    if(CV[13]>best_result):
        best_result = CV[13]
        best_param = CV[3]
params['num_leaves'] = best_param

# Convert Grid list to dataframe
cols = list(params.keys())
cols.append('num_boost_rounds')
cols.append('CV_Result')
df = pd.DataFrame(data=Grid, columns=cols)
        

# Generate model by best iteration
model   = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=num_boost_rounds,
                    verbose_eval=1)

sub_preds = model.predict(test_X)
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_LightGBM_CV_v1.csv', index=False)

"""
num_boost_round=138, valid-auc:0.784, test-auc: 0.782
"""

"""
# Plot importance
lgb.plot_importance(model,
                    max_num_features=20)

# Plot tree
lgb.plot_tree(model)

"""