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

""" ModelRunner functions """
from Tasks import Task

""" Models """
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

# Define CatBoost_v1 model to run it on model runner framework 
class CatBoost_v1(Task):
    def Run(self):
        # Set unique name for model
        self.SetId("CatBoost_v1")

        # Set description to see in browser
        self.SetDescription("This model is trained with CatBoost classifier " +
        " with 5-Fold CV. Predictions are evaluated based on AUC metric.")

        # Datasets
        x_train = self.Data.X_Train
        x_test = self.Data.X_Test
        y_train = self.Data.Y_Train

        # Model
        oof_preds = np.zeros(x_train.shape[0])
        sub_preds = np.zeros(x_test.shape[0])

        folds = KFold(n_splits=5, shuffle=True, random_state=1453)

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train)):
            trn_X, trn_y = x_train.iloc[trn_idx], y_train.iloc[trn_idx]
            val_X, val_y = x_train.iloc[val_idx], y_train.iloc[val_idx]
            
            clf = CatBoostClassifier(eval_metric='AUC')
            clf.fit(trn_X, trn_y)
            oof_preds[val_idx] = clf.predict_proba(val_X)[:,1]
            sub_preds += clf.predict_proba(x_test)[:,1] / folds.n_splits
            
            del clf, trn_X, trn_y, val_X, val_y
            gc.collect()

        # Calculate and submit score
        roc_auc = roc_auc_score(y_train, oof_preds)
        self.SubmitScore("AUC",roc_auc)

        # Prepare submission results
        sub = pd.read_csv('../input/sample_submission.csv')
        sub['TARGET'] = sub_preds
        sub.to_csv('AllData_v3_Installments_CatBoost_v1.csv', index=False)


# Define LightGBM_v1 model to run it on model runner framework 
class LightGBM_v1(Task):
    def Run(self):
        self.SetId("LightGBM_v1")

        # Datasets
        x_train = self.Data.X_Train
        x_test = self.Data.X_Test
        y_train = self.Data.Y_Train

        # Model
        oof_preds = np.zeros(x_train.shape[0])
        sub_preds = np.zeros(x_test.shape[0])

        folds = KFold(n_splits=5, shuffle=True, random_state=1453)

        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train)):
            trn_X, trn_y = x_train.iloc[trn_idx], y_train.iloc[trn_idx]
            val_X, val_y = x_train.iloc[val_idx], y_train.iloc[val_idx]
            
            clf = LGBMClassifier()
            clf.fit(trn_X, trn_y, eval_metric='auc')
            oof_preds[val_idx] = clf.predict_proba(val_X)[:,1]
            sub_preds += clf.predict_proba(x_test)[:,1] / folds.n_splits
            
            del clf, trn_X, trn_y, val_X, val_y
            gc.collect()

        # Calculate and submit score
        roc_auc = roc_auc_score(y_train, oof_preds)
        self.SubmitScore("AUC",roc_auc)

        # Prepare submission results
        sub = pd.read_csv('../input/sample_submission.csv')
        sub['TARGET'] = sub_preds
        sub.to_csv('AllData_v3_Installments_LightGBM_v1.csv', index=False)
