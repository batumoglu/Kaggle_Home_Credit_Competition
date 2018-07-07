# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 11:58:07 2018

@author: Ozkan
"""

import pandas as pd
import copy

# Reading model predictions
sub1 = pd.read_csv('GridSearch/AllData_LGBM_Preds.csv')
sub2 = pd.read_csv('GridSearch/AllData_v2_LGBM_Preds.csv')
sub3 = pd.read_csv('GridSearch/AllData_v3_LGBM_Preds.csv')
sub4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_Preds.csv')
sub5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_Preds.csv')

"""
# Score of the models
result1 = 0.78344
result2 = 0.78423
result3 = 0.78987
result4 = 0.77715
result5 = 0.76761
"""

# Calculating correlation of models
sub = pd.read_csv('../input/sample_submission.csv')
merged = copy.deepcopy(sub)
merged['sub1'] = sub1['TARGET']
merged['sub2'] = sub2['TARGET']
merged['sub3'] = sub3['TARGET']
merged['sub4'] = sub4['TARGET']
merged['sub5'] = sub5['TARGET']
merged.drop('TARGET', axis=1, inplace=True)
merged.set_index('SK_ID_CURR', inplace=True)
merged.corr()

# Getting average of 5 models
sub['TARGET'] = (sub1['TARGET']*0.2)+(sub2['TARGET']*0.2)+(sub3['TARGET']*0.2)+(sub4['TARGET']*0.2)+(sub5['TARGET']*0.2)
sub.to_csv('../Merge5Model.csv', index=False)
""" Scored 0.784 """

# Getting average of 3 top models
sub['TARGET'] = (sub1['TARGET']/3)+(sub2['TARGET']/3)+(sub3['TARGET']/3)
sub.to_csv('../Merge3Model.csv', index=False)
""" Scored 0.789 """