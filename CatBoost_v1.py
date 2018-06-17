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
from catboost import CatBoostClassifier

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

folds = KFold(n_splits=5, shuffle=True, random_state=1453)

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_X)):
    trn_X, trn_y = train_X.iloc[trn_idx], train_Y.iloc[trn_idx]
    val_X, val_y = train_X.iloc[val_idx], train_Y.iloc[val_idx]
    
    clf = CatBoostClassifier(eval_metric='AUC')
    clf.fit(trn_X, trn_y)
    oof_preds[val_idx] = clf.predict_proba(val_X)[:,1]
    sub_preds += clf.predict_proba(test_X)[:,1] / folds.n_splits
    
    del clf, trn_X, trn_y, val_X, val_y
    gc.collect()

print('AUC : %.3f' % roc_auc_score(train_Y, oof_preds))

sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_Buro_CatBoost_v1.csv', index=False)

"""
Application Only
random_state    =1453
train AUC       =0.759
test AUC        =0.744
CatBoost Parameters: Null

Application Buro
random_state    =1453
train AUC       =0.765
test AUC        =0.751
CatBoost Parameters: Null

Application Buro Balance
random_state    =1453
train AUC       =0.765
test AUC        =0.753
CatBoost Parameters: Null

ApplicationBuroAndPrev
random_state    =1453
train AUC       =0.772
test AUC        =0.760
CatBoost Parameters: Null

AllData
random_state    =1453
train AUC       =0.778
test AUC        =0.769
CatBoost Parameters: Null

AllData_v2
random_state    =1453
train AUC       =0.778
test AUC        =0.775
CatBoost Parameters: Null

AllData_v3_prev_2
random_state    =1453
train AUC       =0.779
test AUC        =0.775
CatBoost Parameters: Null

AllData_v3_CC_2
random_state    =1453
train AUC       =0.780
test AUC        =0.776
CatBoost Parameters: Null

AllData_v3_Buro_2
random_state    =1453
train AUC       =0.781
test AUC        =0.775
CatBoost Parameters: Null

"""