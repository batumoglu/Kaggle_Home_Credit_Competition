#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:51:12 2018

@author: ozkan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def ApplicationOnly():

    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    
    categorical_feats = [f for f in data.columns if data[f].dtype == 'object']    
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
    
    categorical_feats = [f for f in data.columns if data[f].dtype == 'object']    
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
    
def AllData():
    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    prev = pd.read_csv('../input/previous_application.csv')
    buro = pd.read_csv('../input/bureau.csv')
    buro_balance = pd.read_csv('../input/bureau_balance.csv')
    credit_card  = pd.read_csv('../input/credit_card_balance.csv')
    POS_CASH  = pd.read_csv('../input/POS_CASH_balance.csv')
    payments = pd.read_csv('../input/installments_payments.csv')
    
    categorical_feats = [f for f in data.columns if data[f].dtype == 'object']    
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])
    
    y = data['TARGET']
    del data['TARGET']
    
    #Pre-processing buro_balance
    print('Pre-processing buro_balance...')
    buro_grouped_size = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].size()
    buro_grouped_max = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].max()
    buro_grouped_min = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].min()
    
    buro_counts = buro_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False)
    buro_counts_unstacked = buro_counts.unstack('STATUS')
    buro_counts_unstacked.columns = ['STATUS_0', 'STATUS_1','STATUS_2','STATUS_3','STATUS_4','STATUS_5','STATUS_C','STATUS_X',]
    buro_counts_unstacked['MONTHS_COUNT'] = buro_grouped_size
    buro_counts_unstacked['MONTHS_MIN'] = buro_grouped_min
    buro_counts_unstacked['MONTHS_MAX'] = buro_grouped_max
    
    buro = buro.join(buro_counts_unstacked, how='left', on='SK_ID_BUREAU')
    
    #Pre-processing previous_application
    print('Pre-processing previous_application...')
    #One-hot encoding of categorical features in previous application data set
    prev_cat_features = [pcol for pcol in prev.columns if prev[pcol].dtype == 'object']
    prev = pd.get_dummies(prev, columns=prev_cat_features)
    avg_prev = prev.groupby('SK_ID_CURR').mean()
    cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
    avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
    del avg_prev['SK_ID_PREV']
    
    #Pre-processing buro
    print('Pre-processing buro...')
    #One-hot encoding of categorical features in buro data set
    buro_cat_features = [bcol for bcol in buro.columns if buro[bcol].dtype == 'object']
    buro = pd.get_dummies(buro, columns=buro_cat_features)
    avg_buro = buro.groupby('SK_ID_CURR').mean()
    avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
    del avg_buro['SK_ID_BUREAU']
    
    #Pre-processing POS_CASH
    print('Pre-processing POS_CASH...')
    le = LabelEncoder()
    POS_CASH['NAME_CONTRACT_STATUS'] = le.fit_transform(POS_CASH['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = POS_CASH[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
    POS_CASH['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    POS_CASH['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
    POS_CASH.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)
    
    #Pre-processing credit_card
    print('Pre-processing credit_card...')
    credit_card['NAME_CONTRACT_STATUS'] = le.fit_transform(credit_card['NAME_CONTRACT_STATUS'].astype(str))
    nunique_status = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').nunique()
    nunique_status2 = credit_card[['SK_ID_CURR', 'NAME_CONTRACT_STATUS']].groupby('SK_ID_CURR').max()
    credit_card['NUNIQUE_STATUS'] = nunique_status['NAME_CONTRACT_STATUS']
    credit_card['NUNIQUE_STATUS2'] = nunique_status2['NAME_CONTRACT_STATUS']
    credit_card.drop(['SK_ID_PREV', 'NAME_CONTRACT_STATUS'], axis=1, inplace=True)
    
    #Pre-processing payments
    print('Pre-processing payments...')
    avg_payments = payments.groupby('SK_ID_CURR').mean()
    avg_payments2 = payments.groupby('SK_ID_CURR').max()
    avg_payments3 = payments.groupby('SK_ID_CURR').min()
    del avg_payments['SK_ID_PREV']
    
    #Join data bases
    print('Joining databases...')
    data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(POS_CASH.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(credit_card.groupby('SK_ID_CURR').mean().reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_payments.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_payments2.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=avg_payments3.reset_index(), how='left', on='SK_ID_CURR')
    
    return(data, test, y)

    