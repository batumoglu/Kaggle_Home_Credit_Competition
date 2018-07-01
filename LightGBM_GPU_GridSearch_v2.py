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
import gc

""" Custom functions for data gathering """
import Dataset

""" Models """
from lightgbm import LGBMClassifier

train_X, test_X, train_Y = Dataset.Load('AllData_v3')

# Step 1
estimator = LGBMClassifier(num_leaves=63)

param_test = {
        'max_depth':range(3,9,1),
        'min_child_samples':range(10,41,10)
        }

gsearch1 = GridSearchCV(estimator,
                        param_grid  = param_test,
                        scoring     = 'roc_auc',
                        n_jobs      = 4,
                        iid         = False, 
                        cv          = 5)
gsearch1.fit(train_X,train_Y)
print(gsearch1.best_params_)
print(gsearch1.best_score_)

# Step 2
estimator = LGBMClassifier(num_leaves=63,
                           max_depth=gsearch1.best_params_['max_depth'],
                           min_child_samples = gsearch1.best_params_['min_child_samples'])

param_test = {
        'reg_alpha':[i/10.0 for i in range(0,6)],
        'reg_lambda':[i/10.0 for i in range(0,6)]
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
estimator = LGBMClassifier(num_leaves   =63,
                           max_depth    =gsearch1.best_params_['max_depth'],
                           min_child_samples = gsearch1.best_params_['min_child_samples'],
                           reg_alpha    = gsearch2.best_params_['reg_alpha'],
                           reg_lambda   = gsearch2.best_params_['reg_lambda']
                           )

param_test = {
        'colsample_bytree':[i/10.0 for i in range(5,11)],
        'subsample':[i/10.0 for i in range(5,11)]
        }

gsearch = GridSearchCV(estimator,
                       param_grid  = param_test,
                       scoring     = 'roc_auc',
                       n_jobs      = 4,
                       iid         = False, 
                       cv          = 5)
gsearch.fit(train_X,train_Y)
print(gsearch.best_params_)
print(gsearch.best_score_)