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

def handlePrev(app_data):

    prev = pd.read_csv('../input/previous_application.csv')
    prev_group = prev.groupby('SK_ID_CURR').agg({'SK_ID_CURR': 'count',
                            'AMT_CREDIT': ['sum', 'mean', 'max', 'min']})
    prev_group.columns = [' '.join(col).strip() for col in prev_group.columns.values]
    
    for column in prev_group.columns:
        prev_group = prev_group.rename(columns={column:'PREV_'+column})
        
    merged_df = app_data.merge(prev_group, left_on='SK_ID_CURR', right_index=True, how='left')
    
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
                             
    merged_df = merged_df.merge(prev_apps_cat_mode, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_PRVMODE'])
    print('Shape after merging with PREV = {}'.format(merged_df.shape))
    return merged_df

def handlePrev_v2(app_data):

    prev = pd.read_csv('../input/previous_application.csv')
    idColumns = ['SK_ID_CURR', 'SK_ID_PREV']
    cat_feats = [f for f in prev.columns if prev[f].dtype == 'object'] 
    for f_ in cat_feats:
        prev[f_], indexer = pd.factorize(prev[f_])
    cat_feats = cat_feats + ['HOUR_APPR_PROCESS_START','DAYS_DECISION','NFLAG_LAST_APPL_IN_DAY']
    nonNum_feats = idColumns + cat_feats    
    num_feats = [f for f in prev.columns if f not in nonNum_feats]
    
    # Numeric Features
    trans =  ['sum', 'mean', 'max', 'min']
    aggs = {}
    for feat in num_feats:
        aggs[feat]=trans
    aggs['SK_ID_CURR']='count'    
    
    prev_numeric_group = prev.groupby('SK_ID_CURR').agg(aggs)
    prev_numeric_group.columns = [' '.join(col).strip() for col in prev_numeric_group.columns.values]
    
    for column in prev_numeric_group.columns:
        prev_numeric_group = prev_numeric_group.rename(columns={column:'PREV_'+column})
        
    app_data = app_data.merge(prev_numeric_group, left_on='SK_ID_CURR', right_index=True, 
                               how='left', suffixes=['','_PRV'])    

    # Categorical Features
    trans = modeValue
    aggs = {}
    for feat in cat_feats:
        aggs[feat]=trans
                                 
    prev_cat_group = prev.groupby('SK_ID_CURR').agg(aggs)

    for column in prev_cat_group.columns:
        prev_cat_group = prev_cat_group.rename(columns={column:'PREV_'+column})
                             
    app_data = app_data.merge(prev_cat_group, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_PRVMODE'])
    
    # Last Features
    most_recent_index = prev.groupby('SK_ID_CURR')['DAYS_DECISION'].idxmax()
    app_data = app_data.merge(prev.loc[most_recent_index], on='SK_ID_CURR', how='left', suffixes=['','_PRVLAST'])   
    
    print('Shape after merging with PREV = {}'.format(app_data.shape))
    return app_data

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
    buro = pd.read_csv('../input/bureau.csv')
    idColumns = ['SK_ID_CURR', 'SK_ID_BUREAU']
    cat_feats = [f for f in buro.columns if buro[f].dtype == 'object'] 
    for f_ in cat_feats:
        buro[f_], indexer = pd.factorize(buro[f_])
    cat_feats = cat_feats + ['DAYS_CREDIT']
    nonNum_feats = idColumns + cat_feats    
    num_feats = [f for f in buro.columns if f not in nonNum_feats]
    
    # Numeric Features
    trans =  ['sum', 'mean', 'max', 'min']
    aggs = {}
    for feat in num_feats:
        aggs[feat]=trans
    aggs['SK_ID_CURR']='count'    
    
    buro_numeric_group = buro.groupby('SK_ID_CURR').agg(aggs)
    buro_numeric_group.columns = [' '.join(col).strip() for col in buro_numeric_group.columns.values]
    
    for column in buro_numeric_group.columns:
        buro_numeric_group = buro_numeric_group.rename(columns={column:'Buro_'+column})
        
    app_data = app_data.merge(buro_numeric_group, left_on='SK_ID_CURR', right_index=True, 
                               how='left', suffixes=['','_BURO'])    
    
    # Active Credits
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
    
    # Categorical Features
    cat_feats = ['CREDIT_CURRENCY','CREDIT_TYPE']
    trans = modeValue
    aggs = {}
    for feat in cat_feats:
        aggs[feat]=trans
                                 
    buro_cat_group = buro.groupby('SK_ID_CURR').agg(aggs)

    for column in buro_cat_group.columns:
        buro_cat_group = buro_cat_group.rename(columns={column:'Buro_'+column})
                             
    app_data = app_data.merge(buro_cat_group, left_on='SK_ID_CURR', right_index=True,
                            how='left', suffixes=['', '_BuroMODE'])
    
    # Last Features
    most_recent_index = buro.groupby('SK_ID_CURR')['SK_ID_BUREAU'].idxmax()
    app_data = app_data.merge(buro.loc[most_recent_index], on='SK_ID_CURR',
                              how='left', suffixes=['','_BuroLAST'])  
       
    print('Shape after merging with credit bureau data = {}'.format(app_data.shape))
    return app_data

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