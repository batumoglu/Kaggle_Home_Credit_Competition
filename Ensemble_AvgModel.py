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
sub6 = pd.read_csv('GridSearch/AppOnly_LGBM_Preds.csv')
sub7 = pd.read_csv('GridSearch/AppBuroBal_LGBM_Preds.csv')
sub8 = pd.read_csv('GridSearch/AllData_v4_AddFeat_LGBM_Preds.csv')
sub9 = pd.read_csv('GridSearch/AllData_v3_XGB_Preds.csv')
sub10 = pd.read_csv('GridSearch/AllData_XGB_Preds.csv')

"""
# Score of the models
result1 = 0.78344
result2 = 0.78423
result3 = 0.79030
result4 = 0.77715
result5 = 0.76761
result6 = 0.76008
result7 = 0.76780
result8 = 0.79058
result9 = 0.78612
result10 = 0.78026
"""

# Calculating correlation of models
sub = pd.read_csv('../input/sample_submission.csv')
merged = copy.deepcopy(sub)
merged['sub1'] = sub1['TARGET']
merged['sub2'] = sub2['TARGET']
merged['sub3'] = sub3['TARGET']
merged['sub4'] = sub4['TARGET']
merged['sub5'] = sub5['TARGET']
merged['sub6'] = sub6['TARGET']
merged['sub7'] = sub7['TARGET']
merged['sub8'] = sub8['TARGET']
merged['sub9'] = sub9['TARGET']
merged['sub10'] = sub10['TARGET']


merged.drop('TARGET', axis=1, inplace=True)
merged.set_index('SK_ID_CURR', inplace=True)
merged.corr()

# Getting average of 7 models
merged['Avg'] = merged.sum(axis=1)/6
sub['TARGET'] = sub['SK_ID_CURR'].map(merged['Avg'])
sub.to_csv('../Merge7Model.csv', index=False)

"""
First 3 models average: TestAUC: 0.789
First 5 models average: TestAUC: 0.784
First 7 models average: TestAUC: 0.780
Best 6 models average:  TestAUC: 0.792
"""