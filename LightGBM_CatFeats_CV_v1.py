# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 20:38:38 2018

@author: Ozkan
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import Dataset
import lightgbm as lgb
import graphviz
import Dataset

# Gather Data
train_X, test_X, train_Y = Dataset.Load('AllData_v3')

cat_cols = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE',
         'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','FLAG_MOBIL','FLAG_EMP_PHONE','FLAG_WORK_PHONE',
         'FLAG_CONT_MOBILE','FLAG_PHONE','FLAG_EMAIL','REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY',
         'WEEKDAY_APPR_PROCESS_START','REG_REGION_NOT_LIVE_REGION','REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION',
         'REG_CITY_NOT_LIVE_CITY','REG_CITY_NOT_WORK_CITY','LIVE_CITY_NOT_WORK_CITY','DEF_60_CNT_SOCIAL_CIRCLE','FLAG_DOCUMENT_2',
         'FLAG_DOCUMENT_3','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7','FLAG_DOCUMENT_8',
         'FLAG_DOCUMENT_9','FLAG_DOCUMENT_10','FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14',
         'FLAG_DOCUMENT_15','FLAG_DOCUMENT_16','FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19','FLAG_DOCUMENT_20',
         'FLAG_DOCUMENT_21','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK']
        
# Convert data to DMatrix
lgb_train = lgb.Dataset(train_X, train_Y, categorical_feature=cat_cols, free_raw_data=False)

# Define parameters
params = {'task'                    :'train',
          'objective'               :'binary',
          'metric'                  :'auc'}

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
                    num_boost_round=int(num_boost_rounds)/0.8,
                    verbose_eval=1)

sub_preds = model.predict(test_X)
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_LGBM_CatFets.csv', index=False)

"""
num_boost_round=138, valid-auc:0.784, test-auc: 0.782
"""

"""
# Plot importance
lgb.plot_importance(model,
                    max_num_features=20)

# Plot tree
lgb.plot_tree(model)

"""