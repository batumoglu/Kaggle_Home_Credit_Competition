# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 12:19:01 2018

@author: Ozkan
"""

import pandas as pd
import copy
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import numpy as np
import gc

# Reading model train predictions
train1 = pd.read_csv('GridSearch/AllData_LGBM_TrainPreds.csv')
train2 = pd.read_csv('GridSearch/AllData_v2_LGBM_TrainPreds.csv')
train3 = pd.read_csv('GridSearch/AllData_v3_LGBM_TrainPreds.csv')
train4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_TrainPreds.csv')
train5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_TrainPreds.csv')

trainMerged = copy.deepcopy(train1)
trainMerged['train1'] = train1['preds']
trainMerged['train2'] = train2['preds']
trainMerged['train3'] = train3['preds']
trainMerged['train4'] = train4['preds']
trainMerged['train5'] = train5['preds']
trainMerged.drop(['preds','SK_ID_CURR'], axis=1, inplace=True)

# Prepare data for model
data = pd.read_csv('../input/application_train.csv')
train_Y = data.pop('TARGET')
lgb_train = lgb.Dataset(trainMerged, train_Y)

# Reading model predictions
sub1 = pd.read_csv('GridSearch/AllData_LGBM_Preds.csv')
sub2 = pd.read_csv('GridSearch/AllData_v2_LGBM_Preds.csv')
sub3 = pd.read_csv('GridSearch/AllData_v3_LGBM_Preds.csv')
sub4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_Preds.csv')
sub5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_Preds.csv')

subMerged = copy.deepcopy(sub1)
subMerged['train1'] = sub1['TARGET']
subMerged['train2'] = sub2['TARGET']
subMerged['train3'] = sub3['TARGET']
subMerged['train4'] = sub4['TARGET']
subMerged['train5'] = sub5['TARGET']
subMerged.drop(['TARGET','SK_ID_CURR'], axis=1, inplace=True)

# Train the model
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(subMerged.shape[0])

folds = KFold(n_splits=5, shuffle=True, random_state=1453)

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(trainMerged)):
    trn_X, trn_y = trainMerged.iloc[trn_idx], train_Y.iloc[trn_idx]
    val_X, val_y = trainMerged.iloc[val_idx], train_Y.iloc[val_idx]
    
    clf = lgb.LGBMClassifier()
    clf.fit(trn_X, trn_y, eval_metric='auc')
    oof_preds[val_idx] = clf.predict_proba(val_X)[:,1]
    sub_preds += clf.predict_proba(subMerged)[:,1] / folds.n_splits
    
    del clf, trn_X, trn_y, val_X, val_y
    gc.collect()

print('AUC : %.3f' % roc_auc_score(train_Y, oof_preds))

# Generate Submission
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('Ensemble_5models_SklearnClassifier.csv', index=False)
"""
Test AUC: 0.786
"""