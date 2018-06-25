#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 02:12:25 2018

@author: ozkan
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc

""" Custom functions for data gathering """
import Gather_Data

""" Models """
import lightgbm as lgb  # Will use GPU

""" Gather data: 
    - ApplicationOnly
    - ApplicationBuroAndPrev
    - AllData
    - ApplicationBuro
    - ApplicationBuroBalance
    - AllData_v2      
    - AllData_v3  """
train_X, test_X, train_Y = Gather_Data.AllData_v3(reduce_mem=False)

oof_preds = np.zeros(train_X.shape[0])
sub_preds = np.zeros(test_X.shape[0])

params = {'task'        :'train',
          'objective'   :'binary',
          'device'      :'gpu',
          'max_bin'     :255,
          'gpu_platform_id' :0,
          'gpu_device_id'   :0,
          'gpu_use_dp'      :True,
          'max_depth'       :5,
          'num_leaves'      :31,
          'min_data_in_leaf':20,
          'bagging_fraction':1,
          'feature_fraction':1,
          'lambda_l1'       :0,
          'lambda_l2'       :0,
          'boosting'        :'gbdt',
          'learning_rate'   :0.1,
          'sparse_threshold':1}

lgb_train = lgb.Dataset(train_X, train_Y)

#print('AUC : %.3f' % roc_auc_score(train_Y, oof_preds))

gridsearch_params = [
    (max_depth, min_data_in_leaf)
    for max_depth in range(3,7)
    for min_data_in_leaf in range(15,26,5)
]


#gridsearch_params = [max_depth for max_depth in range(2,12)]

# Define initial best params and MAE
max_auc = float(0)
best_params = None
for max_depth, min_data_in_leaf in gridsearch_params:
    print("CV with max_depth={}, min_data_in_leaf={}".format(
                             max_depth,
                             min_data_in_leaf))

    # Update our parameters
    params['max_depth'] = max_depth
    params['num_leaves'] = 2**max_depth
    params['min_data_in_leaf'] = min_data_in_leaf

    # Run CV
    cv_results = lgb.cv(params,
                        lgb_train, 
                        seed=1453, 
                        nfold=5,
                        num_boost_round = 500,
                        early_stopping_rounds = 50,
                        metrics='auc')

    # Update best MAE
    mean_auc = max(cv_results['auc-mean'])
    boost_rounds = np.argmax(cv_results['auc-mean'])
    print("\AUC {0[0]:.5f} for {0[1]} rounds".format([mean_auc, boost_rounds]))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = (max_depth, min_data_in_leaf)
        
print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))

""""""

gridsearch_params = [
    (bagging_fraction, bagging_freq)
    for bagging_fraction in np.arange(0.5,1.01,0.1)
    for bagging_freq in range(10,60,10)
]

max_auc = float(0)
best_params = None
for bagging_fraction, bagging_freq in gridsearch_params:
    print("CV with bagging_fraction={}, bagging_freq={}".format(
                             bagging_fraction,
                             bagging_freq))

    # Update our parameters
    params['max_depth'] = 4
    params['num_leaves'] = 16
    params['min_data_in_leaf'] = 25
    
    params['bagging_fraction'] = bagging_fraction
    params['bagging_freq'] = bagging_freq

    # Run CV
    cv_results = lgb.cv(params,
                        lgb_train, 
                        seed=1453, 
                        nfold=5,
                        num_boost_round = 500,
                        early_stopping_rounds = 50,
                        metrics='auc')

    # Update best MAE
    mean_auc = max(cv_results['auc-mean'])
    boost_rounds = np.argmax(cv_results['auc-mean'])
    print("\AUC {0[0]:.5f} for {0[1]} rounds".format([mean_auc, boost_rounds]))
    if mean_auc > max_auc:
        max_auc = mean_auc
        best_params = (bagging_fraction, bagging_freq)
        
print("Best params: {}, {}, AUC: {}".format(best_params[0], best_params[1], max_auc))

"""
Best params: 4, 25, AUC: 0.78782  # 300 Boost
Best params: 4, 25, AUC: 0.78902  # 500 Boost
"""


lgb_train = lgb.Dataset(train_X, train_Y)
cv_results = lgb.cv(params, lgb_train, seed=1453, nfold=5, metrics='auc')

sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_Installments_LightGBM_v1.csv', index=False)