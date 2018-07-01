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
import Dataset

""" Models """
from xgboost import XGBClassifier

start = time.time()
# Data related transactions
train_X, test_X, train_Y = Dataset.Load('ApplicationOnly')

# Step 1
estimator = XGBClassifier(random_state      = 1453,
                          eval_metric  = 'auc')

param_test = {
        'max_depth':range(3,4,1),
        'min_child_weight':range(20,25,5)
        }

gsearch = GridSearchCV(estimator,
                        param_grid  = param_test,
                        scoring     = 'roc_auc',
                        n_jobs      = 4,
                        iid         = False, 
                        cv          = 5)
gsearch.fit(train_X,train_Y)
gsearch.grid_scores_
print(gsearch.best_params_)
print(gsearch.best_score_)

# Step 2
estimator = xgb.XGBClassifier(objective     ='binary',
                              metric        ='auc',
                              seed          = 1453,
                              max_depth     = gsearch.best_params_['max_depth'],
                              min_child_weight = gsearch.best_params_['min_child_weight'])

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
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
print(gsearch.best_params_)
print(gsearch.best_score_)

# Step 3
estimator = xgb.XGBClassifier(objective     ='binary',
                              metric        ='auc',
                              seed          = 1453,
                              max_depth     = gsearch.best_params_['max_depth'],
                              min_child_weight = gsearch.best_params_['min_child_weight'],
                              gamma         = gsearch.best_params_['gamma'])

param_test = {
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch = GridSearchCV(estimator,
                       param_grid  = param_test,
                       scoring     = 'roc_auc',
                       n_jobs      = 4,
                       iid         = False, 
                       cv          = 5)
gsearch.fit(train_X,train_Y)
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
print(gsearch.best_params_)
print(gsearch.best_score_)

# Step 4
estimator = xgb.XGBClassifier(objective     ='binary',
                              metric        ='auc',
                              seed          = 1453,
                              max_depth     = gsearch.best_params_['max_depth'],
                              min_child_weight = gsearch.best_params_['min_child_weight'],
                              gamma         = gsearch.best_params_['gamma'],
                              subsample     = gsearch.best_params_['subsample'],
                              colsample_bytree = gsearch.best_params_['colsample_bytree'])

param_test = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

gsearch = GridSearchCV(estimator,
                       param_grid  = param_test,
                       scoring     = 'roc_auc',
                       n_jobs      = 4,
                       iid         = False, 
                       cv          = 5)
gsearch.fit(train_X,train_Y)
gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_
print(gsearch.best_params_)
print(gsearch.best_score_)
print(gsearch)



end = time.time()
print('Train time: %s seconds' %(str(end-start)))
