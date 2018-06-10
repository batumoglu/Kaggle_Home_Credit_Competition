#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 18:40:02 2018

@author: ozkan
"""
""" Standard Libraries """
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

""" Custom functions for data gathering """
import Gather_Data

""" Models """
from xgboost import XGBClassifier

""" Gather data: ApplicationOnly, ApplicationBuroAndPrev, AllData """
train_X, test_X, train_Y = Gather_Data.ApplicationOnly(reduce_mem=False)

clf = XGBClassifier(tree_method='exact',
                    objective = 'binary:logistic', 
                    booster = "gbtree",
                    eval_metric = 'auc')
clf.fit(train_X, train_Y)
train_preds = clf.predict(train_X)
test_preds = clf.predict_proba(test_X)[:,1]

print('AUC : %.6f' % roc_auc_score(train_Y, train_preds))