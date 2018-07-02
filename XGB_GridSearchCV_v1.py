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
from xgboost.sklearn import XGBClassifier

start = time.time()
# Data related transactions
<<<<<<< HEAD
train_X, test_X, train_Y = Dataset.Load('AllData_v3')

train_X, test_X, train_Y = Gather_Data.ApplicationOnly(reduce_mem=False)

# Step 1
estimator = XGBClassifier(random_state   = 1453,
                          eval_metric    = 'auc')
=======
train_X, test_X, train_Y = Dataset.Load('ApplicationOnly')

# Step 1
estimator = XGBClassifier(base_score=0.5)
>>>>>>> 3ba5dc5579ed36aedf3b23e6585178a170b43651

param_test = {
        'max_depth':range(3,9,1),
        'min_child_weight':range(20,56,5)
        }

gsearch1 = GridSearchCV(estimator,
                        param_grid  = param_test,
                        scoring     = 'roc_auc',
                        n_jobs      = 4,
                        iid         = False, 
                        cv          = 5)
gsearch1.fit(train_X,train_Y)
gsearch1.grid_scores_
print(gsearch1.best_params_)
print(gsearch1.best_score_)

# Step 2
estimator = XGBClassifier(metric        ='auc',
                          seed          = 1453,
                          max_depth     = gsearch1.best_params_['max_depth'],
                          min_child_weight = gsearch1.best_params_['min_child_weight'])

param_test = {
        'gamma':[i/10.0 for i in range(0,5)]
        }

gsearch2 = GridSearchCV(estimator,
                       param_grid  = param_test,
                       scoring     = 'roc_auc',
                       n_jobs      = 4,
                       iid         = False, 
                       cv          = 5)
gsearch2.fit(train_X,train_Y)
print(gsearch2.best_params_)
print(gsearch2.best_score_)

# Step 3
estimator = XGBClassifier(metric        ='auc',
                          seed          = 1453,
                          max_depth     = gsearch1.best_params_['max_depth'],
                          min_child_weight = gsearch1.best_params_['min_child_weight'],
                          gamma         = gsearch2.best_params_['gamma'])

param_test = {
        'subsample':[i/10.0 for i in range(6,10)],
        'colsample_bytree':[i/10.0 for i in range(6,10)]
}

gsearch3 = GridSearchCV(estimator,
                       param_grid  = param_test,
                       scoring     = 'roc_auc',
                       n_jobs      = 4,
                       iid         = False, 
                       cv          = 5)
gsearch3.fit(train_X,train_Y)
print(gsearch3.best_params_)
print(gsearch3.best_score_)

# Step 4
estimator = XGBClassifier(metric        ='auc',
                          seed          = 1453,
                          max_depth     = gsearch1.best_params_['max_depth'],
                          min_child_weight = gsearch1.best_params_['min_child_weight'],
                          gamma         = gsearch2.best_params_['gamma'],
                          subsample     = gsearch3.best_params_['subsample'],
                          colsample_bytree = gsearch3.best_params_['colsample_bytree'])

param_test = {'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100]}

gsearch4 = GridSearchCV(estimator,
                       param_grid  = param_test,
                       scoring     = 'roc_auc',
                       n_jobs      = 4,
                       iid         = False, 
                       cv          = 5)
gsearch4.fit(train_X,train_Y)
print(gsearch4.best_params_)
print(gsearch4.best_score_)
print(gsearch4)

end = time.time()
print('Train time: %s seconds' %(str(end-start)))
