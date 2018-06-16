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
from lightgbm import LGBMClassifier

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
    
    clf = LGBMClassifier()
    clf.fit(trn_X, trn_y, eval_metric='auc')
    oof_preds[val_idx] = clf.predict_proba(val_X)[:,1]
    sub_preds += clf.predict_proba(test_X)[:,1] / folds.n_splits
    
    del clf, trn_X, trn_y, val_X, val_y
    gc.collect()

print('AUC : %.3f' % roc_auc_score(train_Y, oof_preds))

sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_CC_LightGBM_v1.csv', index=False)

"""
ApplicationOnly
random_state    =1453
train AUC       =0.756
test AUC        =0.745
LigGBM Parameters: Null

ApplicationBuro
random_state    =1453
train AUC       =0.763
test AUC        =0.754
LigGBM Parameters: Null

ApplicationBuroBalance
random_state    =1453
train AUC       =0.762
test AUC        =0.755
LigGBM Parameters: Null

ApplicationBuroAndPrev
random_state    =1453
train AUC       =0.770
test AUC        =0.762
LigGBM Parameters: Null

AllData
random_state    =1453
train AUC       =0.777
test AUC        =0.771
LigGBM Parameters: Null

AllData_v2
random_state    =1453
train AUC       =0.777
test AUC        =0.777
LigGBM Parameters: Null

AllData_v2_Prev_v2
random_state    =1453
train AUC       =0.778
test AUC        =0.778
LigGBM Parameters: Null

AllData_v2_CC_v2
random_state    =1453
train AUC       =0.779
test AUC        =0.779
LigGBM Parameters: Null

"""