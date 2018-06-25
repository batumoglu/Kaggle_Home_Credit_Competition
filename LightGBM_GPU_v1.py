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

params['max_depth']         = 4
params['min_data_in_leaf']  = 25
params['seed']              = 1453
params['metric']            ='auc'

folds = KFold(n_splits=5, shuffle=True, random_state=1453)

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_X)):
    lgb_train = lgb.Dataset(train_X.iloc[trn_idx], train_Y.iloc[trn_idx])
    lgb_eval = lgb.Dataset(train_X.iloc[val_idx], train_Y.iloc[val_idx])
        
    gbm = lgb.train(params,
                    lgb_train, 
                    num_boost_round         = 500,
                    valid_sets              = lgb_eval,
                    early_stopping_rounds   = 50)
    oof_preds[val_idx] = gbm.predict(train_X.iloc[val_idx])
    sub_preds += gbm.predict(test_X) / folds.n_splits
    gc.collect()

print('AUC : %.3f' % roc_auc_score(train_Y, oof_preds))

sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_Installments_LightGBM_v1.csv', index=False)