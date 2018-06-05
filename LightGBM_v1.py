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

""" Gather data: ApplicationOnly, ApplicationBuroAndPrev, AllData """
train_X, test_X, train_Y = Gather_Data.ApplicationOnly(reduce_mem=False)

oof_preds = np.zeros(train_X.shape[0])
sub_preds = np.zeros(test_X.shape[0])

folds = KFold(n_splits=5, shuffle=True, random_state=1453)

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_X)):
    trn_X, trn_y = train_X.iloc[trn_idx], train_Y.iloc[trn_idx]
    val_X, val_y = train_X.iloc[val_idx], train_Y.iloc[val_idx]
    
    clf = LGBMClassifier()
    clf.fit(trn_X, trn_y)
    oof_preds[val_idx] = clf.predict_proba(val_X)[:,1]
    sub_preds += clf.predict_proba(test_X)[:,1] / folds.n_splits
    
    del clf, trn_X, trn_y, val_X, val_y
    gc.collect()

print('AUC : %.6f' % roc_auc_score(train_Y, oof_preds))
sub_preds.to_csv('ApplicationOnly_LightGBM_v1.csv')

"""
random_state    =1453
train AUC       =0.756493
test AUC        =
LigGBM Parameters:
Null
"""