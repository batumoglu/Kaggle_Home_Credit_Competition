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
from lightgbm import LGBMClassifier

""" Gather data """
train_X, test_X, train_Y = Gather_Data.ApplicationOnly()

clf = LGBMClassifier()
clf.fit(train_X, train_Y, eval_metric='auc')
train_preds = clf.predict(train_X)
test_preds = clf.predict_proba(test_X)[:,1]

print('AUC : %.6f' % roc_auc_score(train_Y, train_preds))