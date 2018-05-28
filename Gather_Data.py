#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:51:12 2018

@author: ozkan
"""

import pandas as pd
import numpy as np

def ApplicationOnly():

    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    
    categorical_feats = [
        f for f in data.columns if data[f].dtype == 'object'
    ]
    
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])
    
    y = data['TARGET']
    data.drop(['SK_ID_CURR','TARGET'], axis=1, inplace=True)
    test.drop(['SK_ID_CURR'], axis=1, inplace=True)
    
    return(data, test, y)
    
def ApplicationBuroAndPrev():

    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    prev = pd.read_csv('../input/previous_application.csv')
    buro = pd.read_csv('../input/bureau.csv')
    
    categorical_feats = [
        f for f in data.columns if data[f].dtype == 'object'
    ]
    
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])
        
    prev_cat_features = [f_ for f_ in prev.columns if prev[f_].dtype == 'object']
    for f_ in prev_cat_features:
        prev = pd.concat([prev, pd.get_dummies(prev[f_], prefix=f_)], axis=1)
        
    cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    prev['SK_ID_PREV'] = prev['SK_ID_CURR'].map(cnt_prev['SK_ID_PREV'])
    
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    avg_prev.columns = ['prev_app_' + f_ for f_ in avg_prev.columns]
        
    buro_cat_features = [f_ for f_ in buro.columns if buro[f_].dtype == 'object']
    for f_ in buro_cat_features:
        buro = pd.concat([buro, pd.get_dummies(buro[f_], prefix=f_)], axis=1)
    
    avg_buro = buro.groupby('SK_ID_CURR').mean()
    avg_buro['buro_count'] = buro[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
    
    avg_buro.columns = ['bureau_' + f_ for f_ in avg_buro.columns]
    
    data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    
    test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    
    y = data['TARGET']
    data.drop(['SK_ID_CURR','TARGET'], axis=1, inplace=True)
    test.drop(['SK_ID_CURR'], axis=1, inplace=True)
    
    return(data, test, y)