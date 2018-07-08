# -*- coding: utf-8 -*-
"""
Created on Sat Jul  7 12:19:01 2018

@author: Ozkan
"""

import pandas as pd
import copy
import lightgbm as lgb

# Reading model train predictions
train1 = pd.read_csv('GridSearch/AllData_LGBM_TrainPreds.csv')
train2 = pd.read_csv('GridSearch/AllData_v2_LGBM_TrainPreds.csv')
train3 = pd.read_csv('GridSearch/AllData_v3_LGBM_TrainPreds.csv')
train4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_TrainPreds.csv')
train5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_TrainPreds.csv')
train6 = pd.read_csv('GridSearch/AppOnly_LGBM_TrainPreds.csv')
train7 = pd.read_csv('GridSearch/AppBuroBal_LGBM_TrainPreds.csv')

trainMerged = copy.deepcopy(train1)
trainMerged['train1'] = train1['preds']
trainMerged['train2'] = train2['preds']
trainMerged['train3'] = train3['preds']
trainMerged['train4'] = train4['preds']
trainMerged['train5'] = train5['preds']
trainMerged['train6'] = train6['preds']
trainMerged['train7'] = train7['preds']
trainMerged.drop(['preds','SK_ID_CURR'], axis=1, inplace=True)

# Prepare data for model
data = pd.read_csv('../input/application_train.csv')
train_Y = data.pop('TARGET')
lgb_train = lgb.Dataset(trainMerged, train_Y)

# Define parameters
"""
params = {'task'                    :'train',
          'objective'               :'binary',
          'learning_rate'           :0.1,
          'num_leaves'              :31,
          'max_depth'               :5,
          'min_data_in_leaf'        :30,
          'min_sum_hessian_in_leaf' :0.001,
          'lambda_l1'               :0,
          'lambda_l2'               :0,
          'scale_pos_weight'        :1,
          'metric'                  :'auc'}
"""

params = {  'task'      :'train',
            'objective' :'binary',
            'metric'    :'auc'}

# Calculate cv
cv_results = lgb.cv(params=params,
                    train_set=lgb_train,
                    num_boost_round=1000,
                    nfold=5,
                    early_stopping_rounds=50,
                    verbose_eval=1)

# Get best number of iterations
num_boost_rounds = len(cv_results['auc-mean'])
print(num_boost_rounds)

# Generate model by best iteration
model   = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=int(num_boost_rounds/0.8),
                    verbose_eval=1)

# Reading model predictions
sub1 = pd.read_csv('GridSearch/AllData_LGBM_Preds.csv')
sub2 = pd.read_csv('GridSearch/AllData_v2_LGBM_Preds.csv')
sub3 = pd.read_csv('GridSearch/AllData_v3_LGBM_Preds.csv')
sub4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_Preds.csv')
sub5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_Preds.csv')
sub6 = pd.read_csv('GridSearch/AppOnly_LGBM_Preds.csv')
sub7 = pd.read_csv('GridSearch/AppBuroBal_LGBM_Preds.csv')

subMerged = copy.deepcopy(sub1)
subMerged['train1'] = sub1['TARGET']
subMerged['train2'] = sub2['TARGET']
subMerged['train3'] = sub3['TARGET']
subMerged['train4'] = sub4['TARGET']
subMerged['train5'] = sub5['TARGET']
subMerged['train6'] = sub6['TARGET']
subMerged['train7'] = sub7['TARGET']
subMerged.drop(['TARGET','SK_ID_CURR'], axis=1, inplace=True)

# Generate Submission
sub_preds = model.predict(subMerged)
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('Ensemble_7model_LGBM.csv', index=False)

""" 
5 models, Valid Score: 0.833, Test Score: 0.786 
7 models, Valid Score: 0.846, Test Score: 0.791
"""