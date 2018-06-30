#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 02:12:25 2018

@author: ozkan
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
import time

""" Custom functions for data gathering """
import Gather_Data

""" Models """
import lightgbm as lgb  # Will use GPU

# Data related transactions
train_X, test_X, train_Y = Gather_Data.ApplicationOnly(reduce_mem=False)

# Generate an estimator
start = time.time()
estimator = lgb.LGBMClassifier(objective='binary',
                               metric='auc',
                               seed = 1453)

param_test = {
        'max_depth':range(3,10,1),
        'min_child_weight':range(1,6,1)
        }

gsearch = GridSearchCV(estimator,
                        param_grid  = param_test,
                        scoring     = 'roc_auc',
                        n_jobs      = 4,
                        iid         = False, 
                        cv          = 5)
gsearch.fit(train_X,train_Y)
end = time.time()
print('Train time: %s seconds' %(str(end-start)))
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

# Generate an estimator
start = time.time()
estimator = lgb.LGBMClassifier(objective='binary',
                               metric='auc',
                               seed = 1453)

param_test = {
        'max_depth':range(3,10,1),
        'min_child_weight':range(1,6,1)
        }

gsearch = GridSearchCV(estimator,
                        param_grid  = param_test,
                        scoring     = 'roc_auc',
                        n_jobs      = 4,
                        iid         = False, 
                        cv          = 5)
gsearch.fit(train_X,train_Y)
end = time.time()
print('Train time: %s seconds' %(str(end-start)))
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

"""

"""
# Phase 2
start = time.time()
estimator = lgb.LGBMClassifier(objective='binary',
                               max_depth = gsearch.best_params_['max_depth'],
                               min_child_weight = gsearch.best_params_['min_child_weight'],
                               metric='auc',
                               seed = 1453)

param_test = {
        'gamma':[i/10.0 for i in range(0,5)]
        }

gsearch = GridSearchCV(estimator,
                        param_grid  = param_test,
                        scoring     = 'roc_auc',
                        n_jobs      = 4,
                        iid         = False, 
                        cv          = 5)
gsearch.fit(train_X,train_Y)
end = time.time()
print('Train time: %s seconds' %(str(end-start)))
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_

"""

"""