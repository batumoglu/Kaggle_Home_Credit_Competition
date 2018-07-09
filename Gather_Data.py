#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 28 19:51:12 2018

@author: ozkan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy import stats
import GatherTables

def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df

def checkTrainTestConsistency(train, test):
    
    return (train,test)

def AllData_v2(reduce_mem=True):    
    app_data, len_train = GatherTables.getAppData()
    app_data = GatherTables.generateAppFeatures(app_data)
    
    merged_df = GatherTables.handlePrev(app_data)
    merged_df = GatherTables.handleCreditCard(merged_df)
    merged_df = GatherTables.handleBuro(merged_df)
    merged_df = GatherTables.handleBuroBalance(merged_df)
    merged_df = GatherTables.handlePosCash(merged_df)
    merged_df = GatherTables.handleInstallments(merged_df)
    
    categorical_feats = [f for f in merged_df.columns if merged_df[f].dtype == 'object']    
    for f_ in categorical_feats:
        merged_df[f_], indexer = pd.factorize(merged_df[f_])
                                   
    merged_df.drop('SK_ID_CURR', axis=1, inplace=True)
    
    data = merged_df[:len_train]
    test = merged_df[len_train:]
    y = data.pop('TARGET')
    test.drop(['TARGET'], axis=1, inplace=True)    
    return(data, test, y)  
    
def AllData_v3(reduce_mem=True):    
    app_data, len_train = GatherTables.getAppData()
    app_data = GatherTables.generateAppFeatures(app_data)
    
    merged_df = GatherTables.handlePrev_v2(app_data)
    merged_df = GatherTables.handleCreditCard_v2(merged_df)
    merged_df = GatherTables.handleBuro_v2(merged_df)
    merged_df = GatherTables.handleBuroBalance_v2(merged_df)
    merged_df = GatherTables.handlePosCash_v2(merged_df)
    merged_df = GatherTables.handleInstallments_v2(merged_df)
    
    categorical_feats = [f for f in merged_df.columns if merged_df[f].dtype == 'object']    
    for f_ in categorical_feats:
        merged_df[f_], indexer = pd.factorize(merged_df[f_])
                                   
    merged_df.drop('SK_ID_CURR', axis=1, inplace=True)
    
    data = merged_df[:len_train]
    test = merged_df[len_train:]
    y = data.pop('TARGET')
    test.drop(['TARGET'], axis=1, inplace=True)    
    return(data, test, y)  

def AllData_v4(reduce_mem=True):    
    app_data, len_train = GatherTables.getAppData()
    app_data = GatherTables.generateAppFeatures_v4(app_data)
    
    merged_df = GatherTables.handlePrev_v2(app_data)
    merged_df = GatherTables.handleCreditCard_v2(merged_df)
    merged_df = GatherTables.handleBuro_v2(merged_df)
    merged_df = GatherTables.handleBuroBalance_v2(merged_df)
    merged_df = GatherTables.handlePosCash_v2(merged_df)
    merged_df = GatherTables.handleInstallments_v2(merged_df)
    
    merged_df = one_hot_encoder(merged_df)
                                   
    merged_df.drop('SK_ID_CURR', axis=1, inplace=True)
    
    data = merged_df[:len_train]
    test = merged_df[len_train:]
    y = data.pop('TARGET')
    test.drop(['TARGET'], axis=1, inplace=True)    
    return(data, test, y)  
    
    

def ApplicationBuroBalance(reduce_mem=True):

    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    buro = pd.read_csv('../input/bureau.csv')
    buro_balance = pd.read_csv('../input/bureau_balance.csv')
    
    # Handle Buro Balance
    buro_balance.loc[buro_balance['STATUS']=='C', 'STATUS'] = '0'
    buro_balance.loc[buro_balance['STATUS']=='X', 'STATUS'] = '0'
    buro_balance['STATUS'] = buro_balance['STATUS'].astype('int64')
    
    buro_balance_group = buro_balance.groupby('SK_ID_BUREAU').agg({'STATUS':['max','mean'], 'MONTHS_BALANCE':'max'})
    buro_balance_group.columns = [' '.join(col).strip() for col in buro_balance_group.columns.values]
    
    idx = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].transform(max) == buro_balance['MONTHS_BALANCE']
    Buro_Balance_Last = buro_balance[idx][['SK_ID_BUREAU','STATUS']]
    Buro_Balance_Last.rename(columns={'STATUS': 'Buro_Balance_Last_Value'}, inplace=True)
    
    Buro_Balance_Last['Buro_Balance_Max'] = Buro_Balance_Last['SK_ID_BUREAU'].map(buro_balance_group['STATUS max'])
    Buro_Balance_Last['Buro_Balance_Mean'] = Buro_Balance_Last['SK_ID_BUREAU'].map(buro_balance_group['STATUS mean'])
    Buro_Balance_Last['Buro_Balance_Last_Month'] = Buro_Balance_Last['SK_ID_BUREAU'].map(buro_balance_group['MONTHS_BALANCE max'])
    
    # Handle Buro Data
    def nonUnique(x):
        return x.nunique()
    def modeValue(x):
        return stats.mode(x)[0][0]
    def totalBadCredit(x):
        badCredit = 0
        for value in x:
            if(value==2 or value==3):
                badCredit+=1
        return badCredit
    def creditOverdue(x):
        overdue=0
        for value in x:
            if(value>0):
                overdue+=1
        return overdue

    categorical_feats = [f for f in buro.columns if buro[f].dtype == 'object']    
    for f_ in categorical_feats:
        buro[f_], indexer = pd.factorize(buro[f_])
    
    categorical_feats = [f for f in data.columns if data[f].dtype == 'object']    
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])
    
    # Aggregate Values on All Credits
    buro_group = buro.groupby('SK_ID_CURR').agg({'SK_ID_BUREAU':'count', 
                             'AMT_CREDIT_SUM':'sum', 
                             'AMT_CREDIT_SUM_DEBT':'sum',
                             'CREDIT_CURRENCY': [nonUnique, modeValue],
                             'CREDIT_TYPE': [nonUnique, modeValue],
                             'CNT_CREDIT_PROLONG': 'sum',
                             'CREDIT_ACTIVE': totalBadCredit,
                             'CREDIT_DAY_OVERDUE': creditOverdue
                             })
    buro_group.columns = [' '.join(col).strip() for col in buro_group.columns.values]
    
    # Aggregate Values on Active Credits
    buro_active = buro.loc[buro['CREDIT_ACTIVE']==1]
    buro_group_active = buro_active.groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM': ['sum', 'count'],
                                           'AMT_CREDIT_SUM_DEBT': 'sum',
                                           'AMT_CREDIT_SUM_LIMIT': 'sum'
                                           })
    buro_group_active.columns = [' '.join(col).strip() for col in buro_group_active.columns.values]
    
    # Getting last credit for each user
    idx = buro.groupby('SK_ID_CURR')['SK_ID_BUREAU'].transform(max) == buro['SK_ID_BUREAU']
    Buro_Last = buro[idx][['SK_ID_CURR','CREDIT_TYPE','DAYS_CREDIT_UPDATE','DAYS_CREDIT',
                    'DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT', 'SK_ID_BUREAU']]
    
    Buro_Last['Credit_Count'] = Buro_Last['SK_ID_CURR'].map(buro_group['SK_ID_BUREAU count'])
    Buro_Last['Total_Credit_Amount'] = Buro_Last['SK_ID_CURR'].map(buro_group['AMT_CREDIT_SUM sum'])
    Buro_Last['Total_Debt_Amount'] = Buro_Last['SK_ID_CURR'].map(buro_group['AMT_CREDIT_SUM_DEBT sum'])
    Buro_Last['NumberOfCreditCurrency'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_CURRENCY nonUnique'])
    Buro_Last['MostCommonCreditCurrency'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_CURRENCY modeValue'])
    Buro_Last['NumberOfCreditType'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_TYPE nonUnique'])
    Buro_Last['MostCommonCreditType'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_TYPE modeValue'])
    Buro_Last['NumberOfCreditProlong'] = Buro_Last['SK_ID_CURR'].map(buro_group['CNT_CREDIT_PROLONG sum'])
    Buro_Last['NumberOfBadCredit'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_ACTIVE totalBadCredit'])
    Buro_Last['NumberOfDelayedCredit'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_DAY_OVERDUE creditOverdue'])
    
    Buro_Last['Active_Credit_Amount'] = Buro_Last['SK_ID_CURR'].map(buro_group_active['AMT_CREDIT_SUM sum'])
    Buro_Last['Active_Credit_Count'] = Buro_Last['SK_ID_CURR'].map(buro_group_active['AMT_CREDIT_SUM count'])
    Buro_Last['Active_Debt_Amount'] = Buro_Last['SK_ID_CURR'].map(buro_group_active['AMT_CREDIT_SUM_DEBT sum'])
    Buro_Last['Active_Credit_Card_Limit'] = Buro_Last['SK_ID_CURR'].map(buro_group_active['AMT_CREDIT_SUM_LIMIT sum'])
    Buro_Last['BalanceOnCreditBuro'] = Buro_Last['Active_Debt_Amount'] / Buro_Last['Active_Credit_Amount']
    
    # Merge buro with Buro Balance
    buro_merged = pd.merge(buro, Buro_Balance_Last, how='left', on='SK_ID_BUREAU')
    buro_merged = buro_merged[['SK_ID_CURR','SK_ID_BUREAU','Buro_Balance_Last_Value','Buro_Balance_Max',
                 'Buro_Balance_Mean','Buro_Balance_Last_Month']]
    buro_merged_group = buro_merged.groupby('SK_ID_CURR').agg(np.mean)    
    buro_merged_group.reset_index(inplace=True)
    buro_merged_group.drop('SK_ID_BUREAU', axis=1, inplace=True)
    
    # Add Tables to main Data
    data = data.merge(right=Buro_Last.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=Buro_Last.reset_index(), how='left', on='SK_ID_CURR')
    
    data = data.merge(right=buro_merged_group.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=buro_merged_group.reset_index(), how='left', on='SK_ID_CURR')    
    
    y = data['TARGET']
    data.drop(['SK_ID_CURR','TARGET'], axis=1, inplace=True)
    test.drop(['SK_ID_CURR'], axis=1, inplace=True)
    
    if(reduce_mem==True):
        data = reduce_mem_usage(data)
        test = reduce_mem_usage(test)
    
    return(data, test, y)

def ApplicationBuro(reduce_mem=True):

    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    buro = pd.read_csv('../input/bureau.csv')
    
    def nonUnique(x):
        return x.nunique()
    def modeValue(x):
        return stats.mode(x)[0][0]
    def totalBadCredit(x):
        badCredit = 0
        for value in x:
            if(value==2 or value==3):
                badCredit+=1
        return badCredit
    def creditOverdue(x):
        overdue=0
        for value in x:
            if(value>0):
                overdue+=1
        return overdue

    categorical_feats = [f for f in buro.columns if buro[f].dtype == 'object']    
    for f_ in categorical_feats:
        buro[f_], indexer = pd.factorize(buro[f_])
    
    categorical_feats = [f for f in data.columns if data[f].dtype == 'object']    
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])
    
    # Aggregate Values on All Credits
    buro_group = buro.groupby('SK_ID_CURR').agg({'SK_ID_BUREAU':'count', 
                             'AMT_CREDIT_SUM':'sum', 
                             'AMT_CREDIT_SUM_DEBT':'sum',
                             'CREDIT_CURRENCY': [nonUnique, modeValue],
                             'CREDIT_TYPE': [nonUnique, modeValue],
                             'CNT_CREDIT_PROLONG': 'sum',
                             'CREDIT_ACTIVE': totalBadCredit,
                             'CREDIT_DAY_OVERDUE': creditOverdue
                             })
    buro_group.columns = [' '.join(col).strip() for col in buro_group.columns.values]
    
    # Aggregate Values on Active Credits
    buro_active = buro.loc[buro['CREDIT_ACTIVE']==1]
    buro_group_active = buro_active.groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM': ['sum', 'count'],
                                           'AMT_CREDIT_SUM_DEBT': 'sum',
                                           'AMT_CREDIT_SUM_LIMIT': 'sum'
                                           })
    buro_group_active.columns = [' '.join(col).strip() for col in buro_group_active.columns.values]
    
    # Getting last credit for each user
    idx = buro.groupby('SK_ID_CURR')['SK_ID_BUREAU'].transform(max) == buro['SK_ID_BUREAU']
    Buro_Last = buro[idx][['SK_ID_CURR','CREDIT_TYPE','DAYS_CREDIT_UPDATE','DAYS_CREDIT',
                    'DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT']]
    
    Buro_Last['Credit_Count'] = Buro_Last['SK_ID_CURR'].map(buro_group['SK_ID_BUREAU count'])
    Buro_Last['Total_Credit_Amount'] = Buro_Last['SK_ID_CURR'].map(buro_group['AMT_CREDIT_SUM sum'])
    Buro_Last['Total_Debt_Amount'] = Buro_Last['SK_ID_CURR'].map(buro_group['AMT_CREDIT_SUM_DEBT sum'])
    Buro_Last['NumberOfCreditCurrency'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_CURRENCY nonUnique'])
    Buro_Last['MostCommonCreditCurrency'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_CURRENCY modeValue'])
    Buro_Last['NumberOfCreditType'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_TYPE nonUnique'])
    Buro_Last['MostCommonCreditType'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_TYPE modeValue'])
    Buro_Last['NumberOfCreditProlong'] = Buro_Last['SK_ID_CURR'].map(buro_group['CNT_CREDIT_PROLONG sum'])
    Buro_Last['NumberOfBadCredit'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_ACTIVE totalBadCredit'])
    Buro_Last['NumberOfDelayedCredit'] = Buro_Last['SK_ID_CURR'].map(buro_group['CREDIT_DAY_OVERDUE creditOverdue'])
    
    Buro_Last['Active_Credit_Amount'] = Buro_Last['SK_ID_CURR'].map(buro_group_active['AMT_CREDIT_SUM sum'])
    Buro_Last['Active_Credit_Count'] = Buro_Last['SK_ID_CURR'].map(buro_group_active['AMT_CREDIT_SUM count'])
    Buro_Last['Active_Debt_Amount'] = Buro_Last['SK_ID_CURR'].map(buro_group_active['AMT_CREDIT_SUM_DEBT sum'])
    Buro_Last['Active_Credit_Card_Limit'] = Buro_Last['SK_ID_CURR'].map(buro_group_active['AMT_CREDIT_SUM_LIMIT sum'])
    Buro_Last['BalanceOnCreditBuro'] = Buro_Last['Active_Debt_Amount'] / Buro_Last['Active_Credit_Amount']
    
    data = data.merge(right=Buro_Last.reset_index(), how='left', on='SK_ID_CURR')
    test = test.merge(right=Buro_Last.reset_index(), how='left', on='SK_ID_CURR')
    
    y = data['TARGET']
    data.drop(['SK_ID_CURR','TARGET'], axis=1, inplace=True)
    test.drop(['SK_ID_CURR'], axis=1, inplace=True)
    
    if(reduce_mem==True):
        data = reduce_mem_usage(data)
        test = reduce_mem_usage(test)
    
    return(data, test, y)
    
def ApplicationOnly(reduce_mem=True):

    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    
    
    categorical_feats = [f for f in data.columns if data[f].dtype == 'object']    
    for f_ in categorical_feats:
        data[f_], indexer = pd.factorize(data[f_])
        test[f_] = indexer.get_indexer(test[f_])
    
    y = data['TARGET']
    data.drop(['SK_ID_CURR','TARGET'], axis=1, inplace=True)
    test.drop(['SK_ID_CURR'], axis=1, inplace=True)
    
    if(reduce_mem==True):
        data = reduce_mem_usage(data)
        test = reduce_mem_usage(test)
    
    return(data, test, y)
    
def ApplicationBuroAndPrev(reduce_mem=True):

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
    
    if(reduce_mem==True):
        data = reduce_mem_usage(data)
        test = reduce_mem_usage(test)
    
    return(data, test, y)
    
def AllData(reduce_mem=True):
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
    
    if(reduce_mem==True):
        data = reduce_mem_usage(data)
        test = reduce_mem_usage(test)
    
    return(data, test, y)
    
def reduce_mem_usage(df):
    for col in df.columns:
        col_type = df[col].dtype
    
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    if col_type != object:
        c_min = df[col].min()
        c_max = df[col].max()
        if str(col_type)[:3] == 'int':
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df[col] = df[col].astype(np.int8)
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df[col] = df[col].astype(np.int16)
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df[col] = df[col].astype(np.int32)
            elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                df[col] = df[col].astype(np.int64)  
        else:
            if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                df[col] = df[col].astype(np.float16)
            elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
            else:
                df[col] = df[col].astype(np.float64)
    else:
        df[col] = df[col].astype('category')
        
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

    