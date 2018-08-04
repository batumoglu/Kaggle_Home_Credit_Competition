#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 15 01:48:49 2018

@author: ozkan
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from scipy import stats
from contextlib import contextmanager
import time
import gc

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

def getAppData():
    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    
    len_train = len(data)
    app_data = pd.concat([data, test])
            
    print('Combined train & test input shape before any merging  = {}'.format(app_data.shape))
    return app_data, len_train

def generateAppFeatures(app_data):
    app_data['LOAN_INCOME_RATIO'] = app_data['AMT_CREDIT'] / app_data['AMT_INCOME_TOTAL']
    app_data['ANNUITY_INCOME_RATIO'] = app_data['AMT_ANNUITY'] / app_data['AMT_INCOME_TOTAL']
    app_data['ANNUITY LENGTH'] = app_data['AMT_CREDIT'] / app_data['AMT_ANNUITY']
    app_data['WORKING_LIFE_RATIO'] = app_data['DAYS_EMPLOYED'] / app_data['DAYS_BIRTH']
    app_data['INCOME_PER_FAM'] = app_data['AMT_INCOME_TOTAL'] / app_data['CNT_FAM_MEMBERS']
    app_data['CHILDREN_RATIO'] = app_data['CNT_CHILDREN'] / app_data['CNT_FAM_MEMBERS']
    #appdata_['INCOME_CREDIT_PCT'] = app_data['AMT_INCOME_TOTAL'] / app_data['AMT_CREDIT']
    print('Shape after extra features = {}'.format(app_data.shape))
    return app_data

def generateAppFeatures_v4(app_data):
    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    print("Train samples: {}, test samples: {}".format(len(data), len(test)))
    df = pd.concat([data, test])

    df.loc[df['CODE_GENDER']=='XNA','CODE_GENDER'] = 'F'
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = (df['EXT_SOURCE_1']+1) * (df['EXT_SOURCE_2']+1) * (df['EXT_SOURCE_3']+1)
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    return df

def handlePrev(app_data):

    prev = pd.read_csv('../input/previous_application.csv')
    prev_group = prev.groupby('SK_ID_CURR').agg({'SK_ID_CURR': 'count',
                            'AMT_CREDIT': ['sum', 'mean', 'max', 'min']})
    prev_group.columns = [' '.join(col).strip() for col in prev_group.columns.values]
    
    for column in prev_group.columns:
        prev_group = prev_group.rename(columns={column:'PREV_'+column})
        
    merged_app_data = app_data.merge(prev_group, left_on='SK_ID_CURR', right_index=True, how='left')
    
    categorical_feats = [f for f in prev.columns if prev[f].dtype == 'object']    
    for f_ in categorical_feats:
        prev[f_], indexer = pd.factorize(prev[f_])
                               
    prev_apps_cat_mode = prev.groupby('SK_ID_CURR').agg({categorical_feats[0]:modeValue,
                                     categorical_feats[1]:modeValue,
                                     categorical_feats[2]:modeValue,
                                     categorical_feats[3]:modeValue,
                                     categorical_feats[4]:modeValue,
                                     categorical_feats[5]:modeValue,
                                     categorical_feats[6]:modeValue,
                                     categorical_feats[7]:modeValue,
                                     categorical_feats[8]:modeValue,
                                     categorical_feats[9]:modeValue,
                                     categorical_feats[10]:modeValue,
                                     categorical_feats[11]:modeValue,
                                     categorical_feats[12]:modeValue,
                                     categorical_feats[13]:modeValue,
                                     categorical_feats[14]:modeValue,
                                     categorical_feats[15]:modeValue})
                             
    merged_app_data = merged_app_data.merge(prev_apps_cat_mode, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_PRVMODE'])
    print('Shape after merging with PREV = {}'.format(merged_app_data.shape))
    return merged_app_data

def handlePrev_v2(app_data):
    prev = pd.read_csv('../input/previous_application.csv')
    prev, cat_cols = one_hot_encoder(prev)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left')

    return prev_agg   

def handlePrev_v4(app_data):
    prev = pd.read_csv('../input/previous_application.csv')
    prev, cat_cols = one_hot_encoder(prev)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY'               : [ 'max', 'mean', 'min', 'sum', 'std'],
        'AMT_APPLICATION'           : [ 'max', 'mean', 'min', 'sum', 'std'],
        'AMT_CREDIT'                : [ 'max', 'mean', 'min', 'sum', 'std'],
        'APP_CREDIT_PERC'           : [ 'max', 'mean', 'min', 'sum', 'std'],
        'AMT_DOWN_PAYMENT'          : [ 'max', 'mean', 'min', 'sum', 'std'],
        'AMT_GOODS_PRICE'           : [ 'max', 'mean', 'min', 'sum', 'std'],
        'HOUR_APPR_PROCESS_START'   : [ 'max', 'mean', 'min', 'sum', 'std'],
        'RATE_DOWN_PAYMENT'         : [ 'max', 'mean', 'min', 'sum', 'std'],
        'DAYS_DECISION'             : [ 'max', 'mean', 'min', 'sum', 'std'],
        'CNT_PAYMENT'               : [ 'max', 'mean', 'min', 'sum', 'std'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean', 'sum']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left')

    return prev_agg   

def handleCreditCard(app_data):
    credit_card  = pd.read_csv('../input/credit_card_balance.csv')
    # Value Counts
    app_data = app_data.merge(pd.DataFrame(credit_card['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_CRED_CARD'])    
    # Last Values
    most_recent_index = credit_card.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = credit_card.columns[credit_card.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    app_data = app_data.merge(credit_card.loc[most_recent_index, cat_feats], left_on='SK_ID_CURR', 
                                right_on='SK_ID_CURR', how='left', suffixes=['', '_CCLAST'])    

    print('Shape after merging with credit card data = {}'.format(app_data.shape))
    return app_data

def handleCreditCard_v2(app_data):
    credit_card  = pd.read_csv('../input/credit_card_balance.csv')
    idColumns = ['SK_ID_CURR', 'SK_ID_PREV']
    cat_feats = [f for f in credit_card.columns if credit_card[f].dtype == 'object'] 
    for f_ in cat_feats:
        credit_card[f_], indexer = pd.factorize(credit_card[f_])
    cat_feats = cat_feats + ['MONTHS_BALANCE']
    nonNum_feats = idColumns + cat_feats    
    num_feats = [f for f in credit_card.columns if f not in nonNum_feats]
    
    # Numeric Features
    trans =  ['sum', 'mean', 'max', 'min']
    aggs = {}
    for feat in num_feats:
        aggs[feat]=trans
    aggs['SK_ID_CURR']='count'    
    
    cc_numeric_group = credit_card.groupby('SK_ID_CURR').agg(aggs)
    cc_numeric_group.columns = [' '.join(col).strip() for col in cc_numeric_group.columns.values]
    
    for column in cc_numeric_group.columns:
        cc_numeric_group = cc_numeric_group.rename(columns={column:'CC_'+column})
        
    app_data = app_data.merge(cc_numeric_group, left_on='SK_ID_CURR', right_index=True, 
                               how='left', suffixes=['','_CC'])    
    
    # Categorical Features
    trans = modeValue
    aggs = {}
    for feat in cat_feats:
        aggs[feat]=trans
                                 
    cc_cat_group = credit_card.groupby('SK_ID_CURR').agg(aggs)

    for column in cc_cat_group.columns:
        cc_cat_group = cc_cat_group.rename(columns={column:'CC_'+column})
                             
    app_data = app_data.merge(cc_cat_group, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_CCMODE'])
    
    # Last Features
    most_recent_index = credit_card.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    app_data = app_data.merge(credit_card.loc[most_recent_index], on='SK_ID_CURR',
                              how='left', suffixes=['','_CCLAST'])  

    print('Shape after merging with credit card data = {}'.format(app_data.shape))
    return app_data

def handleCreditCard_v4(app_data):
    credit_card  = pd.read_csv('../input/credit_card_balance.csv')
    idColumns = ['SK_ID_CURR', 'SK_ID_PREV']
    cat_feats = [f for f in credit_card.columns if credit_card[f].dtype == 'object'] 
    for f_ in cat_feats:
        credit_card[f_], indexer = pd.factorize(credit_card[f_])
    cat_feats = cat_feats + ['MONTHS_BALANCE']
    nonNum_feats = idColumns + cat_feats    
    num_feats = [f for f in credit_card.columns if f not in nonNum_feats]
    
    # Numeric Features
    trans =  ['sum', 'mean', 'max', 'min']
    aggs = {}
    for feat in num_feats:
        aggs[feat]=trans
    aggs['SK_ID_CURR']='count'    
    
    cc_numeric_group = credit_card.groupby('SK_ID_CURR').agg(aggs)
    cc_numeric_group.columns = [' '.join(col).strip() for col in cc_numeric_group.columns.values]
    
    for column in cc_numeric_group.columns:
        cc_numeric_group = cc_numeric_group.rename(columns={column:'CC_'+column})
        
    app_data = app_data.merge(cc_numeric_group, left_on='SK_ID_CURR', right_index=True, 
                               how='left', suffixes=['','_CC'])    
    
    # Categorical Features
    trans = modeValue
    aggs = {}
    for feat in cat_feats:
        aggs[feat]=trans
                                 
    cc_cat_group = credit_card.groupby('SK_ID_CURR').agg(aggs)

    for column in cc_cat_group.columns:
        cc_cat_group = cc_cat_group.rename(columns={column:'CC_'+column})
                             
    app_data = app_data.merge(cc_cat_group, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_CCMODE'])
    
    # Last Features
    most_recent_index = credit_card.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    app_data = app_data.merge(credit_card.loc[most_recent_index], on='SK_ID_CURR',
                              how='left', suffixes=['','_CCLAST'])  

    print('Shape after merging with credit card data = {}'.format(app_data.shape))
    return app_data

def handleBuro(app_data):
    buro = pd.read_csv('../input/bureau.csv')
    # Value Counts
    app_data = app_data.merge(pd.DataFrame(buro['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_BUREAU'])

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
    
    for column in buro_group.columns:
        buro_group = buro_group.rename(columns={column:'BURO_ALL_'+column})
    
    app_data = app_data.merge(buro_group, left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_BURO'])
    
    # Aggregate Values on Active Credits
    buro_active = buro.loc[buro['CREDIT_ACTIVE']==1]
    buro_group_active = buro_active.groupby('SK_ID_CURR').agg({'AMT_CREDIT_SUM': ['sum', 'count'],
                                           'AMT_CREDIT_SUM_DEBT': 'sum',
                                           'AMT_CREDIT_SUM_LIMIT': 'sum'
                                           })
    buro_group_active.columns = [' '.join(col).strip() for col in buro_group_active.columns.values]
    
    for column in buro_group_active.columns:
        buro_group_active = buro_group_active.rename(columns={column:'BURO_ACT_'+column})
    
    app_data = app_data.merge(buro_group_active, left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_BURO_ACT'])
    
    # Buro_Last['LastBalanceOnCreditBuro'] = Buro_Last['Active_Debt_Amount'] / Buro_Last['Active_Credit_Amount']
    
    # Getting last credit for each user
    idx = buro.groupby('SK_ID_CURR')['SK_ID_BUREAU'].transform(max) == buro['SK_ID_BUREAU']
    Buro_Last = buro[idx][['SK_ID_CURR','CREDIT_TYPE','DAYS_CREDIT_UPDATE','DAYS_CREDIT',
                    'DAYS_CREDIT_ENDDATE','DAYS_ENDDATE_FACT', 'SK_ID_BUREAU']]
    
    app_data = app_data.merge(Buro_Last, left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_BURO_LAST'])
       
    print('Shape after merging with credit bureau data = {}'.format(app_data.shape))
    return app_data

def handleBuro_v2(app_data):
    bureau = pd.read_csv('../input/bureau.csv')
    bb = pd.read_csv('../input/bureau_balance.csv')
    bb, bb_cat = one_hot_encoder(bb)
    bureau, bureau_cat = one_hot_encoder(bureau)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left')

    return bureau_agg

def handleBuroBalance(app_data):
    buro_balance = pd.read_csv('../input/bureau_balance.csv')
    
    # Historical Buro Balance
    buro_balance.loc[buro_balance['STATUS']=='C', 'STATUS'] = '0'
    buro_balance.loc[buro_balance['STATUS']=='X', 'STATUS'] = '0'
    buro_balance['STATUS'] = buro_balance['STATUS'].astype('int64')
    
    buro_balance_group = buro_balance.groupby('SK_ID_BUREAU').agg({'STATUS':['max','mean'], 'MONTHS_BALANCE':'max'})
    buro_balance_group.columns = [' '.join(col).strip() for col in buro_balance_group.columns.values]
    
    app_data = app_data.merge(buro_balance_group, left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_BALANCE_HIST'])
    # Last Buro Balance
    idx = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].transform(max) == buro_balance['MONTHS_BALANCE']
    Buro_Balance_Last = buro_balance[idx][['SK_ID_BUREAU','STATUS']]
    
    app_data = app_data.merge(Buro_Balance_Last, left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_BALANCE_HIST'])
    
    print('Shape after merging with Bureau Balance Data = {}'.format(app_data.shape))  
    return app_data

def handleBuroBalance_v2(app_data):
    buro_balance = pd.read_csv('../input/bureau_balance.csv')
    buro = pd.read_csv('../input/bureau.csv')
    buro = buro[['SK_ID_CURR','SK_ID_BUREAU']]
    
    # Add Historical Buro Balance
    buro_balance.loc[buro_balance['STATUS']=='C', 'STATUS'] = '0'
    buro_balance.loc[buro_balance['STATUS']=='X', 'STATUS'] = '0'
    buro_balance['STATUS'] = buro_balance['STATUS'].astype('int64')
    
    buro_balance_group = buro_balance.groupby('SK_ID_BUREAU').agg({'STATUS':['max','mean','min','sum'],
                                             'MONTHS_BALANCE':['count']})
    buro_balance_group.columns = [' '.join(col).strip() for col in buro_balance_group.columns.values]
    
    buro = buro.merge(buro_balance_group, left_on='SK_ID_BUREAU', 
                                right_index=True, how='left', suffixes=['', '_BALANCE_HIST'])
    
    # Add Last Buro Balance
    most_recent_index = buro_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].idxmax()
    Buro_Balance_Last = buro_balance.loc[most_recent_index]    
    buro = buro.merge(Buro_Balance_Last, on='SK_ID_BUREAU', how='left', suffixes=['', '_BALANCE_LAST'])
    
    # All historical data for each credit is now one line
    # Buro Balance summary merged with all credits in Buro
    trans =  ['sum', 'mean', 'max', 'min']
    aggs = {}
    aggregateColumns = ['STATUS max','STATUS mean','STATUS min','STATUS sum','MONTHS_BALANCE count']
    for col in aggregateColumns:
        aggs[col]=trans    
    
    BuroBal_AllHist_group = buro.groupby('SK_ID_CURR').agg(aggs)
    BuroBal_AllHist_group.columns = [' '.join(col).strip() for col in BuroBal_AllHist_group.columns.values]
            
    app_data = app_data.merge(BuroBal_AllHist_group, left_on='SK_ID_CURR', right_index=True, 
                               how='left', suffixes=['','_BBHist'])    
    # Buro Balance summary merged with active credits in Buro
    # Posponed for now

    # Buro Balance summary merged with last credit in Buro
    most_recent_index = buro.groupby('SK_ID_CURR')['SK_ID_BUREAU'].idxmax()
    buroLast = buro.loc[most_recent_index]   
    buroLastBeforeMerge = buroLast.drop('SK_ID_BUREAU', axis=1)    
    
    app_data = app_data.merge(buroLastBeforeMerge, on='SK_ID_CURR',how='left', suffixes=['','_BuroBalLAST'])      
    
    print('Shape after merging with Bureau Balance Data = {}'.format(app_data.shape))  
    return app_data

def handlePosCash(app_data):
    POS_CASH  = pd.read_csv('../input/POS_CASH_balance.csv')
    
    # Weighted by recency
    wm = lambda x: np.average(x, weights=-1/POS_CASH.loc[x.index, 'MONTHS_BALANCE'])
    f = {'CNT_INSTALMENT': wm, 'CNT_INSTALMENT_FUTURE': wm, 'SK_DPD': wm, 'SK_DPD_DEF':wm}
    cash_avg = POS_CASH.groupby('SK_ID_CURR')['CNT_INSTALMENT','CNT_INSTALMENT_FUTURE',
                                                 'SK_DPD', 'SK_DPD_DEF'].agg(f)
    
    app_data = app_data.merge(cash_avg, left_on='SK_ID_CURR', right_index=True,
                                how='left', suffixes=['', '_CashAVG'])
    
    # Historical Data
    PosCashGroup = POS_CASH.groupby('SK_ID_CURR')['CNT_INSTALMENT','CNT_INSTALMENT_FUTURE','SK_DPD',
                                   'SK_DPD_DEF'].agg({
                                   'CNT_INSTALMENT':['mean', 'max', 'min'],
                                   'CNT_INSTALMENT_FUTURE':['mean', 'max', 'min'],
                                   'SK_DPD':['mean', 'max', 'min'],
                                   'SK_DPD_DEF':['mean', 'max', 'min']})     
    PosCashGroup.columns = [' '.join(col).strip() for col in PosCashGroup.columns.values]    
    for column in PosCashGroup.columns:
        PosCashGroup = PosCashGroup.rename(columns={column:'PosCash_'+column})
    
    app_data = app_data.merge(PosCashGroup, left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_POSCASH'])
        
    # Last Values
    most_recent_index = POS_CASH.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    cat_feats = POS_CASH.columns[POS_CASH.dtypes == 'object'].tolist()  + ['SK_ID_CURR']
    app_data = app_data.merge(POS_CASH.loc[most_recent_index, cat_feats], on='SK_ID_CURR',
                              how='left', suffixes=['', '_PosCashLast'])
    print('Shape after merging with pos cash data = {}'.format(app_data.shape))
    return app_data

def handlePosCash_v2(app_data):
    POS_CASH  = pd.read_csv('../input/POS_CASH_balance.csv')
    
    idColumns = ['SK_ID_CURR', 'SK_ID_PREV']
    cat_feats = [f for f in POS_CASH.columns if POS_CASH[f].dtype == 'object'] 
    for f_ in cat_feats:
        POS_CASH[f_], indexer = pd.factorize(POS_CASH[f_])
    num_feats = ['SK_DPD','SK_DPD_DEF','CNT_INSTALMENT','CNT_INSTALMENT_FUTURE']
    
    # Numeric Features
    trans =  ['sum', 'mean', 'max', 'min']
    aggs = {}
    for feat in num_feats:
        aggs[feat]=trans
    aggs['SK_ID_CURR']='count' 
    aggs['NAME_CONTRACT_STATUS']=modeValue
    
    PosCash_Group = POS_CASH.groupby('SK_ID_CURR').agg(aggs)
    PosCash_Group.columns = [' '.join(col).strip() for col in PosCash_Group.columns.values]
    
    for column in PosCash_Group.columns:
        PosCash_Group = PosCash_Group.rename(columns={column:'PosCash_'+column})
        
    app_data = app_data.merge(PosCash_Group, left_on='SK_ID_CURR', right_index=True, 
                               how='left', suffixes=['','_PosCashAvg'])    
    
    # Last Features
    most_recent_index = POS_CASH.groupby('SK_ID_CURR')['MONTHS_BALANCE'].idxmax()
    PosCashBeforeMerge = POS_CASH.loc[most_recent_index]
    PosCashBeforeMerge = PosCashBeforeMerge.drop('SK_ID_PREV', axis=1) 
    app_data = app_data.merge(POS_CASH.loc[most_recent_index], on='SK_ID_CURR',
                              how='left', suffixes=['','_PosCashLast'])      

    print('Shape after merging with pos cash data = {}'.format(app_data.shape))
    return app_data

def handleInstallments(app_data):
    installments = pd.read_csv('../input/installments_payments.csv')
    # Value Counts
    app_data = app_data.merge(pd.DataFrame(installments['SK_ID_CURR'].value_counts()), left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_CNT_INSTALL'])
    # Historical Data
    installmentsGroup = installments.groupby('SK_ID_CURR').agg({'NUM_INSTALMENT_VERSION':['mean', 'max', 'min'],
                                            'NUM_INSTALMENT_NUMBER':['mean', 'max', 'min'],
                                            'DAYS_INSTALMENT':['mean', 'max', 'min'],
                                            'DAYS_ENTRY_PAYMENT':['mean', 'max', 'min'],
                                            'AMT_INSTALMENT':['mean', 'max', 'min'],
                                            'AMT_PAYMENT':['mean', 'max', 'min']})
    installmentsGroup.columns = [' '.join(col).strip() for col in installmentsGroup.columns.values]

    app_data = app_data.merge(installmentsGroup, left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_INST'])

    print('Shape after merging with installments data = {}'.format(app_data.shape))    
    return app_data

def handleInstallments_v2(app_data):
    installments = pd.read_csv('../input/installments_payments.csv')
    
    # Fill NaN values --> Delay on payment max 150
    installments['DAYS_ENTRY_PAYMENT'].fillna(installments['DAYS_INSTALMENT']+150,inplace=True)
    installments['AMT_PAYMENT'].fillna(0, inplace=True)
    
    # Generate Features
    installments['DaysDelayOnPayment'] = installments['DAYS_ENTRY_PAYMENT']-installments['DAYS_INSTALMENT']
    installments['MissingOnPayment'] = installments['AMT_INSTALMENT']-installments['AMT_PAYMENT']
        
    # Numeric Features
    trans =  ['sum', 'mean', 'max', 'min']
    num_feats = ['DaysDelayOnPayment','MissingOnPayment','DAYS_INSTALMENT','DAYS_ENTRY_PAYMENT','AMT_INSTALMENT','AMT_PAYMENT']
    aggs = {}
    for feat in num_feats:
        aggs[feat]=trans
    aggs['SK_ID_CURR']='count' # number of installments
    aggs['NUM_INSTALMENT_VERSION']=modeValue  # most common installment version
    # Historical Data
    installmentsGroup = installments.groupby('SK_ID_CURR').agg(aggs)
    installmentsGroup.columns = [' '.join(col).strip() for col in installmentsGroup.columns.values]

    app_data = app_data.merge(installmentsGroup, left_on='SK_ID_CURR', 
                                right_index=True, how='left', suffixes=['', '_INST'])

    print('Shape after merging with installments data = {}'.format(app_data.shape))    
    return app_data

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na=True)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# Preprocess application_train.csv and application_test.csv
def application_train_test():
    # Read data and merge
    data = pd.read_csv('../input/application_train.csv')
    test = pd.read_csv('../input/application_test.csv')
    print("Train samples: {}, test samples: {}".format(len(data), len(test)))
    df = pd.concat([data, test])
    df.loc[df['CODE_GENDER']=='XNA','CODE_GENDER'] = 'F'
    
    docs = [_f for _f in df.columns if 'FLAG_DOC' in _f]
    live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    
    # NaN values for DAYS_EMPLOYED: 365.243 -> nan
    df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

    inc_by_org = df[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']

    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_DOC_IND_KURT'] = df[docs].kurtosis(axis=1)
    df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_INC_BY_ORG'] = df['ORGANIZATION_TYPE'].map(inc_by_org)
    df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
    df['NEW_SOURCES_PROD'] = (df['EXT_SOURCE_1']+1) * (df['EXT_SOURCE_2']+1) * (df['EXT_SOURCE_3']+1)
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_PHONE_TO_BIRTH_RATIO_EMPLOYER'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # Categorical features with Binary encode (0 or 1; two categories)
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        df[bin_feature], uniques = pd.factorize(df[bin_feature])
    # Categorical features with One-Hot encode
    df, cat_cols = one_hot_encoder(df)
    dropcolum=['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_5','FLAG_DOCUMENT_6','FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_8','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10', 
    'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13',
    'FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_16',
    'FLAG_DOCUMENT_17','FLAG_DOCUMENT_18','FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20','FLAG_DOCUMENT_21']
    df= df.drop(dropcolum,axis=1)
    return df

# Preprocess bureau.csv and bureau_balance.csv
def bureau_and_balance():
    bureau = pd.read_csv('../input/bureau.csv')
    bb = pd.read_csv('../input/bureau_balance.csv')
    bb, bb_cat = one_hot_encoder(bb)
    bureau, bureau_cat = one_hot_encoder(bureau)
    
    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    
    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': [ 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': [ 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': [ 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']
    
    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left')

    return bureau_agg

# Preprocess previous_applications.csv
def previous_applications():
    prev = pd.read_csv('../input/previous_application.csv')
    prev, cat_cols = one_hot_encoder(prev)
    # Days 365.243 values -> nan
    prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
    prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
    prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
    prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
    # Add feature: value ask / value received percentage
    prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
    # Previous applications numeric features
    num_aggregations = {
        'AMT_ANNUITY': [ 'max', 'mean'],
        'AMT_APPLICATION': [ 'max','mean'],
        'AMT_CREDIT': [ 'max', 'mean'],
        'APP_CREDIT_PERC': [ 'max', 'mean'],
        'AMT_DOWN_PAYMENT': [ 'max', 'mean'],
        'AMT_GOODS_PRICE': [ 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': [ 'max', 'mean'],
        'RATE_DOWN_PAYMENT': [ 'max', 'mean'],
        'DAYS_DECISION': [ 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
    }
    # Previous applications categorical features
    cat_aggregations = {}
    for cat in cat_cols:
        cat_aggregations[cat] = ['mean']
    
    prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
    # Previous Applications: Approved Applications - only numerical features
    approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
    approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
    approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
    prev_agg = prev_agg.join(approved_agg, how='left')
    # Previous Applications: Refused Applications - only numerical features
    refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
    refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
    refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
    prev_agg = prev_agg.join(refused_agg, how='left')

    return prev_agg

# Preprocess POS_CASH_balance.csv
def pos_cash():
    pos = pd.read_csv('../input/POS_CASH_balance.csv')
    pos, cat_cols = one_hot_encoder(pos)
    # Features
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    
    pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
    pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
    # Count pos cash accounts
    pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()

    return pos_agg
    
# Preprocess installments_payments.csv
def installments_payments():
    ins = pd.read_csv('../input/installments_payments.csv')
    ins, cat_cols = one_hot_encoder(ins)
    # Percentage and difference paid in each installment (amount paid and installment value)
    ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
    ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
    # Days past due and days before due (no negative values)
    ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
    ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
    ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
    ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
    # Features: Perform aggregations
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DPD': ['max', 'mean', 'sum','min','std' ],
        'DBD': ['max', 'mean', 'sum','min','std'],
        'PAYMENT_PERC': [ 'max','mean',  'var','min','std'],
        'PAYMENT_DIFF': [ 'max','mean', 'var','min','std'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum','min','std'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum','std'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum','std']
    }
    for cat in cat_cols:
        aggregations[cat] = ['mean']
    ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
    ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
    # Count installments accounts
    ins_agg['INSTAL_COUNT'] = ins.groupby('SK_ID_CURR').size()
    del ins
    gc.collect()
    return ins_agg

# Preprocess credit_card_balance.csv
def credit_card_balance():
    cc = pd.read_csv('../input/credit_card_balance.csv')
    cc, cat_cols = one_hot_encoder(cc)
    # General aggregations
    cc.drop(['SK_ID_PREV'], axis= 1, inplace = True)
    cc_agg = cc.groupby('SK_ID_CURR').agg([ 'max', 'mean', 'sum', 'var'])
    cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
    # Count credit card lines
    cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()
    del cc
    gc.collect()
    return cc_agg