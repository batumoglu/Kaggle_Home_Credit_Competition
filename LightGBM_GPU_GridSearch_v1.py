#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 02:12:25 2018

@author: ozkan
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import gc
from sklearn.model_selection import GridSearchCV

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
          'max_bin'     :63,
          'num_leaves'  :255,
          'num_threads'  :1,
          'metric'      :'auc',
          'device'      :'gpu',
          'gpu_platform_id' :0,
          'gpu_device_id'   :0,
          'seed'            :1453}

param_test1 = {'max_depth':range(3,10,1)}

train_set = lgb.Dataset(data=train_X, label=train_Y)

estimator = lgb.LGBMClassifier(objective='binary',
                               metric='auc',
                               device = 'gpu',
                               gpu_platform_id = 0,
                               gpu_device_id = 0,
                               seed = 1453)

gsearch1 = GridSearchCV(estimator,
                        param_grid = param_test1,
                        scoring='roc_auc',
                        n_jobs=1,
                        iid=False, 
                        cv=5)
gsearch1.fit(train_X,train_Y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_

print('AUC : %.3f' % roc_auc_score(train_Y, oof_preds))

sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_Installments_LightGBM_v1.csv', index=False)