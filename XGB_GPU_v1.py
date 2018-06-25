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
import xgboost as xgb  # Will use GPU

""" Gather data: 
    - ApplicationOnly
    - ApplicationBuroAndPrev
    - AllData
    - ApplicationBuro
    - ApplicationBuroBalance
    - AllData_v2      
    - AllData_v3  """
train_X, test_X, train_Y = Gather_Data.ApplicationOnly(reduce_mem=False)

oof_preds = np.zeros(train_X.shape[0])
sub_preds = np.zeros(test_X.shape[0])

params = {'gpu_id'  :0,
          'max_bin' :63,
          'tree_method'     :'gpu_hist',
          'eval_metric'     :'auc'}

folds = KFold(n_splits=5, shuffle=True, random_state=1453)
dtest = xgb.DMatrix(test_X)

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_X)):    
    dtrain = xgb.DMatrix(train_X.iloc[trn_idx], label=train_Y.iloc[trn_idx])
    dvalid = xgb.DMatrix(train_X.iloc[val_idx], label=train_Y.iloc[val_idx])
    #evallist = [(dvalid, 'valid'), (dtrain, 'train')]
    gbm = xgb.train(params, dtrain, evals=[(dvalid, 'valid')], num_boost_round=100, early_stopping_rounds=100, maximize=True)
    oof_preds[val_idx] = gbm.predict(dvalid)
    sub_preds += gbm.predict(dtest) / folds.n_splits
    gc.collect()

print('AUC : %.3f' % roc_auc_score(train_Y, oof_preds))

sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_Installments_LightGBM_v1.csv', index=False)