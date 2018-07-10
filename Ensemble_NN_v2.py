import copy
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc


# Reading LGBM model train predictions
print("Loading model train predictions")
train_lgb_1 = pd.read_csv('GridSearch/AllData_LGBM_TrainPreds.csv')
train_lgb_2 = pd.read_csv('GridSearch/AllData_v2_LGBM_TrainPreds.csv')
train_lgb_3 = pd.read_csv('GridSearch/AllData_v3_LGBM_TrainPreds.csv')
train_lgb_4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_TrainPreds.csv')
train_lgb_5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_TrainPreds.csv')
train_lgb_6 = pd.read_csv('GridSearch/AppOnly_LGBM_TrainPreds.csv')
train_lgb_7 = pd.read_csv('GridSearch/AppBuroBal_LGBM_TrainPreds.csv')

print("Merging model train predictions")
trainMerged_lgb = copy.deepcopy(train_lgb_1)
trainMerged_lgb['train1'] = train_lgb_1['preds']
trainMerged_lgb['train2'] = train_lgb_2['preds']
trainMerged_lgb['train3'] = train_lgb_3['preds']
trainMerged_lgb['train4'] = train_lgb_4['preds']
trainMerged_lgb['train5'] = train_lgb_5['preds']
trainMerged_lgb['train6'] = train_lgb_6['preds']
trainMerged_lgb['train7'] = train_lgb_7['preds']
trainMerged_lgb.drop(['preds','SK_ID_CURR'], axis=1, inplace=True)

# Reading model predictions
print("Loading model test predictions")
sub_lgb_1 = pd.read_csv('GridSearch/AllData_LGBM_Preds.csv')
sub_lgb_2 = pd.read_csv('GridSearch/AllData_v2_LGBM_Preds.csv')
sub_lgb_3 = pd.read_csv('GridSearch/AllData_v3_LGBM_Preds.csv')
sub_lgb_4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_Preds.csv')
sub_lgb_5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_Preds.csv')
sub_lgb_6 = pd.read_csv('GridSearch/AppOnly_LGBM_Preds.csv')
sub_lgb_7 = pd.read_csv('GridSearch/AppBuroBal_LGBM_Preds.csv')

print("Merging model test predictions")
subMerged_lgb = copy.deepcopy(sub_lgb_1)
subMerged_lgb['train1'] = sub_lgb_1['TARGET']
subMerged_lgb['train2'] = sub_lgb_2['TARGET']
subMerged_lgb['train3'] = sub_lgb_3['TARGET']
subMerged_lgb['train4'] = sub_lgb_4['TARGET']
subMerged_lgb['train5'] = sub_lgb_5['TARGET']
subMerged_lgb['train6'] = sub_lgb_6['TARGET']
subMerged_lgb['train7'] = sub_lgb_7['TARGET']
subMerged_lgb.drop(['TARGET','SK_ID_CURR'], axis=1, inplace=True)


# Reading XGB model train predictions
print("Loading model train predictions")
# train_xgb_1 = pd.read_csv('GridSearch/AllData_LGBM_TrainPreds.csv')
# train_xgb_2 = pd.read_csv('GridSearch/AllData_v2_LGBM_TrainPreds.csv')
train_xgb_3 = pd.read_csv('GridSearch/AllData_v3_XGB_TrainPreds.csv')
# train_xgb_4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_TrainPreds.csv')
# train_xgb_5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_TrainPreds.csv')
# train_xgb_6 = pd.read_csv('GridSearch/AppOnly_LGBM_TrainPreds.csv')
# train_xgb_7 = pd.read_csv('GridSearch/AppBuroBal_LGBM_TrainPreds.csv')

print("Merging model train predictions")
trainMerged_xgb = copy.deepcopy(train_xgb_3)
# trainMerged_xgb['train1'] = train_xgb_1['preds']
# trainMerged_xgb['train2'] = train_xgb_2['preds']
trainMerged_xgb['train3'] = train_xgb_3['preds']
# trainMerged_xgb['train4'] = train_xgb_4['preds']
# trainMerged_xgb['train5'] = train_xgb_5['preds']
# trainMerged_xgb['train6'] = train_xgb_6['preds']
# trainMerged_xgb['train7'] = train_xgb_7['preds']
trainMerged_xgb.drop(['preds','SK_ID_CURR'], axis=1, inplace=True)


# Reading model predictions
print("Loading model test predictions")
# sub_xgb_1 = pd.read_csv('GridSearch/AllData_LGBM_Preds.csv')
# sub_xgb_2 = pd.read_csv('GridSearch/AllData_v2_LGBM_Preds.csv')
sub_xgb_3 = pd.read_csv('GridSearch/AllData_v3_XGB_Preds.csv')
# sub_xgb_4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_Preds.csv')
# sub_xgb_5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_Preds.csv')
# sub_xgb_6 = pd.read_csv('GridSearch/AppOnly_LGBM_Preds.csv')
# sub_xgb_7 = pd.read_csv('GridSearch/AppBuroBal_LGBM_Preds.csv')

print("Merging model test predictions")
subMerged_xgb = copy.deepcopy(sub_xgb_3)
# subMerged_xgb['train1'] = sub_xgb_1['TARGET']
# subMerged_xgb['train2'] = sub_xgb_2['TARGET']
subMerged_xgb['train3'] = sub_xgb_3['TARGET']
# subMerged_xgb['train4'] = sub_xgb_4['TARGET']
# subMerged_xgb['train5'] = sub_xgb_5['TARGET']
# subMerged_xgb['train6'] = sub_xgb_6['TARGET']
# subMerged_xgb['train7'] = sub_xgb_7['TARGET']
subMerged_xgb.drop(['TARGET','SK_ID_CURR'], axis=1, inplace=True)

# Prepare data for model
print("Loading application train dataset")
data = pd.read_csv('../input/application_train.csv')
train_Y = data.pop('TARGET')

# Train the model
oof_preds_lgb = np.zeros(data.shape[0])
sub_preds_lgb = pd.DataFrame()

folds = KFold(n_splits=5, shuffle=True, random_state=1453)

fold_weights_lgb = np.zeros(folds.n_splits)

print("Starting LGBM 5-Folds")
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(trainMerged_lgb)):
    trn_X, trn_y = trainMerged_lgb.iloc[trn_idx], train_Y.iloc[trn_idx]
    val_X, val_y = trainMerged_lgb.iloc[val_idx], train_Y.iloc[val_idx]

    clf = MLPClassifier(
            activation='relu', 
            alpha=1e-05, 
            batch_size='auto',
            beta_1=0.9, 
            beta_2=0.999, 
            early_stopping=True,
            epsilon=1e-08, 
            hidden_layer_sizes=(100,10), 
            learning_rate='adaptive',
            learning_rate_init=0.001, 
            max_iter=200, 
            momentum=0.9,
            nesterovs_momentum=True, 
            power_t=0.5, 
            random_state=1, 
            shuffle=False,
            solver='adam', 
            tol=0.0001, 
            validation_fraction=0.2, 
            verbose=True,
            warm_start=False)

    clf.fit(trn_X, trn_y)
    proba = clf.predict_proba(val_X)[:,1]
    oof_preds_lgb[val_idx] = proba
    fold_auc = roc_auc_score(val_y, proba)
    print("AUC-LGBM (Fold-" + str(n_fold) + "): %.3f" % fold_auc)
    # sub_preds += clf.predict_proba(subMerged)[:,1] / folds.n_splits
    sub_preds_lgb[str(n_fold)] = clf.predict_proba(subMerged_lgb)[:,1]
    fold_weights_lgb[n_fold] = fold_auc
    
    del clf, trn_X, trn_y, val_X, val_y
    gc.collect()
    print("Fold-" + str(n_fold) + " completed.")

print('AUC-LGBM : %.3f' % roc_auc_score(train_Y, oof_preds_lgb))

fold_weights_lgb = fold_weights_lgb / fold_weights_lgb.sum()

sub_preds_lgb = sub_preds_lgb * fold_weights_lgb



# Train the model
oof_preds_xgb = np.zeros(data.shape[0])
sub_preds_xgb = pd.DataFrame()

fold_weights_xgb = np.zeros(folds.n_splits)

print("Starting LGBM 5-Folds")
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(trainMerged_xgb)):
    trn_X, trn_y = trainMerged_xgb.iloc[trn_idx], train_Y.iloc[trn_idx]
    val_X, val_y = trainMerged_xgb.iloc[val_idx], train_Y.iloc[val_idx]

    clf = MLPClassifier(
            activation='relu', 
            alpha=1e-05, 
            batch_size='auto',
            beta_1=0.9, 
            beta_2=0.999, 
            early_stopping=True,
            epsilon=1e-08, 
            hidden_layer_sizes=(100,10), 
            learning_rate='adaptive',
            learning_rate_init=0.001, 
            max_iter=200, 
            momentum=0.9,
            nesterovs_momentum=True, 
            power_t=0.5, 
            random_state=1, 
            shuffle=False,
            solver='adam', 
            tol=0.0001, 
            validation_fraction=0.2, 
            verbose=True,
            warm_start=False)

    clf.fit(trn_X, trn_y)
    proba = clf.predict_proba(val_X)[:,1]
    oof_preds_xgb[val_idx] = proba
    fold_auc = roc_auc_score(val_y, proba)
    print("AUC-XGB (Fold-" + str(n_fold) + "): %.3f" % fold_auc)
    # sub_preds += clf.predict_proba(subMerged)[:,1] / folds.n_splits
    sub_preds_xgb[str(n_fold)] = clf.predict_proba(subMerged_xgb)[:,1]
    fold_weights_xgb[n_fold] = fold_auc
    
    del clf, trn_X, trn_y, val_X, val_y
    gc.collect()
    print("Fold-" + str(n_fold) + " completed.")

print('AUC-XGB : %.3f' % roc_auc_score(train_Y, oof_preds_xgb))

fold_weights_xgb = fold_weights_xgb / fold_weights_xgb.sum()

sub_preds_xgb = sub_preds_xgb * fold_weights_xgb

# from hyperopt import hp, tpe, fmin, STATUS_OK

def objective(x):
    train_preds = np.zeros(len(oof_preds_lgb))
    for idx in range(len(train_preds)):
        train_preds[idx] = x*oof_preds_lgb[idx] + (1-x)*oof_preds_xgb[idx]
    return 1 - roc_auc_score(train_Y, train_preds)

# best = fmin(objective, space=hp.uniform("x",0,1) ,algo=tpe.suggest, max_evals=2000)
# print("Best k constant: " + str(best))

'''
best = {'x': 0.9999976130817497}
'''

best_auc = 1 - objective(0.9999976130817497)
print("Best overall AUC: " + str(best_auc))

sub_preds = pd.DataFrame()
sub_preds["LGBM"] = sub_preds_lgb.sum(axis=1) * 0.9999976130817497
sub_preds["XGB"] = sub_preds_xgb.sum(axis=1) * (1 - 0.9999976130817497)

# Generate Submission
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds.sum(axis=1)
sub.to_csv('Ensemble_8models_MLPClassifier_BayesianOpt.csv', index=False)
print("Ensembling has been completed and submission file has been generated.")
