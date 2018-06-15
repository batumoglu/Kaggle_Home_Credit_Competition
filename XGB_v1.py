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
from sklearn.model_selection import KFold
import gc

""" Custom functions for data gathering """
import Gather_Data

""" Models """
from xgboost import XGBClassifier

""" Gather data: ApplicationOnly, ApplicationBuroAndPrev, AllData, ApplicationBuro, ApplicationBuroBalance"""
train_X, test_X, train_Y = Gather_Data.ApplicationBuroBalance(reduce_mem=False)

oof_preds = np.zeros(train_X.shape[0])
sub_preds = np.zeros(test_X.shape[0])

folds = KFold(n_splits=5, shuffle=True, random_state=1453)

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_X)):
    trn_X, trn_y = train_X.iloc[trn_idx], train_Y.iloc[trn_idx]
    val_X, val_y = train_X.iloc[val_idx], train_Y.iloc[val_idx]
    
    clf = XGBClassifier()
    clf.fit(trn_X, trn_y, eval_metric='auc')
    oof_preds[val_idx] = clf.predict_proba(val_X)[:,1]
    sub_preds += clf.predict_proba(test_X)[:,1] / folds.n_splits
    
    del clf, trn_X, trn_y, val_X, val_y
    gc.collect()

print('AUC : %.3f' % roc_auc_score(train_Y, oof_preds))

sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('ApplicationBuroBalance_XGB_v1.csv', index=False)

"""
ApplicationOnly
random_state    =1453
train AUC       =0.751
test AUC        =0.738
XGB Parameters: Null

ApplicationBuroAndPrev
random_state    =1453
train AUC       =0.762
test AUC        =0.750
XGB Parameters: Null

AllData
random_state    =1453
train AUC       =0.767
test AUC        =0.756
XGB Parameters: Null

ApplicationBuro
random_state    =1453
train AUC       =0.754
test AUC        =0.744
XGB Parameters: Null

ApplicationBuroBalance
random_state    =1453
train AUC       =0.755
test AUC        =0.744
XGB Parameters: Null

AllData_v2
random_state    =1453
train AUC       =0.767
test AUC        =0.775
XGB Parameters: Null

"""
