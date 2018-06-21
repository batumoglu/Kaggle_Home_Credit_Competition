# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 15:30:57 2018

@author: Ozkan.Batumoglu
"""

""" Standard Libraries """
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn import cross_validation, metrics
from sklearn.model_selection import GridSearchCV

""" Custom functions for data gathering """
import Gather_Data

""" Models """
from xgboost import XGBClassifier
import xgboost as xgb

""" Gather data: 
    - ApplicationOnly
    - ApplicationBuroAndPrev
    - AllData
    - ApplicationBuro
    - ApplicationBuroBalance
    - AllData_v2      
    - AllData_v3  """
train_X, test_X, train_Y = Gather_Data.AllData_v3(reduce_mem=False)
 
xgb1 = XGBClassifier()

param_test1 = {
        'max_depth':[3,5,7,9]
}

gsearch1 = GridSearchCV(estimator = XGBClassifier(), param_grid = param_test1, 
                        scoring='roc_auc', iid=False, cv=5, n_jobs=-1)
gsearch1.fit(train_X,train_Y)
gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_
