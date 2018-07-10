import copy
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import gc


# Reading model train predictions
print("Loading model train predictions")
train1 = pd.read_csv('GridSearch/AllData_LGBM_TrainPreds.csv')
train2 = pd.read_csv('GridSearch/AllData_v2_LGBM_TrainPreds.csv')
train3 = pd.read_csv('GridSearch/AllData_v3_LGBM_TrainPreds.csv')
train4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_TrainPreds.csv')
train5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_TrainPreds.csv')
train6 = pd.read_csv('GridSearch/AppOnly_LGBM_TrainPreds.csv')
train7 = pd.read_csv('GridSearch/AppBuroBal_LGBM_TrainPreds.csv')

print("Merging model train predictions")
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
print("Loading application train dataset")
data = pd.read_csv('../input/application_train.csv')
train_Y = data.pop('TARGET')


# Reading model predictions
print("Loading model test predictions")
sub1 = pd.read_csv('GridSearch/AllData_LGBM_Preds.csv')
sub2 = pd.read_csv('GridSearch/AllData_v2_LGBM_Preds.csv')
sub3 = pd.read_csv('GridSearch/AllData_v3_LGBM_Preds.csv')
sub4 = pd.read_csv('GridSearch/AppBuroPrev_LGBM_Preds.csv')
sub5 = pd.read_csv('GridSearch/ApplicationBuro_LGBM_Preds.csv')
sub6 = pd.read_csv('GridSearch/AppOnly_LGBM_Preds.csv')
sub7 = pd.read_csv('GridSearch/AppBuroBal_LGBM_Preds.csv')

print("Merging model test predictions")
subMerged = copy.deepcopy(sub1)
subMerged['train1'] = sub1['TARGET']
subMerged['train2'] = sub2['TARGET']
subMerged['train3'] = sub3['TARGET']
subMerged['train4'] = sub4['TARGET']
subMerged['train5'] = sub5['TARGET']
subMerged['train6'] = sub6['TARGET']
subMerged['train7'] = sub7['TARGET']
subMerged.drop(['TARGET','SK_ID_CURR'], axis=1, inplace=True)


# Train the model
oof_preds = np.zeros(data.shape[0])
sub_preds = pd.DataFrame()

folds = KFold(n_splits=10, shuffle=True, random_state=1453)

fold_weights = np.zeros(folds.n_splits)

print("Starting 5-Folds")
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(trainMerged)):
    trn_X, trn_y = trainMerged.iloc[trn_idx], train_Y.iloc[trn_idx]
    val_X, val_y = trainMerged.iloc[val_idx], train_Y.iloc[val_idx]

    clf = MLPClassifier(
            activation='relu', 
            alpha=1e-05, 
            batch_size='auto',
            beta_1=0.9, 
            beta_2=0.999, 
            early_stopping=True,
            epsilon=1e-08, 
            hidden_layer_sizes=(1200,), 
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
    oof_preds[val_idx] = proba
    fold_auc = roc_auc_score(val_y, proba)
    print("AUC (Fold-" + str(n_fold) + "): %.3f" % fold_auc)
    # sub_preds += clf.predict_proba(subMerged)[:,1] / folds.n_splits
    sub_preds[str(n_fold)] = clf.predict_proba(subMerged)[:,1]
    fold_weights[n_fold] = fold_auc
    
    del clf, trn_X, trn_y, val_X, val_y
    gc.collect()
    print("Fold-" + str(n_fold) + " completed.")

print('AUC : %.3f' % roc_auc_score(train_Y, oof_preds))

fold_weights = fold_weights / fold_weights.sum()

sub_preds = sub_preds * fold_weights

# Generate Submission
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds.sum(axis=1)
sub.to_csv('Ensemble_7models_MLPClassifier.csv', index=False)
print("Ensembling has been completed and submission file has been generated.")


# folds = KFold(n_splits=5, shuffle=True, random_state=1453)

# for n_fold, (trn_idx, val_idx) in enumerate(folds.split(trainMerged)):
#     trn_X, trn_y = trainMerged.iloc[trn_idx], train_Y.iloc[trn_idx]
#     val_X, val_y = trainMerged.iloc[val_idx], train_Y.iloc[val_idx]
    
#     clf = MLPClassifier(
#             activation='relu', 
#             alpha=1e-05, 
#             batch_size='auto',
#             beta_1=0.9, 
#             beta_2=0.999, 
#             early_stopping=True,
#             epsilon=1e-08, 
#             hidden_layer_sizes=(5, 2), 
#             learning_rate='constant',
#             learning_rate_init=0.001, 
#             max_iter=200, momentum=0.9,
#             nesterovs_momentum=True, 
#             power_t=0.5, 
#             random_state=1, 
#             shuffle=True,
#             solver='lbfgs', 
#             tol=0.0001, 
#             validation_fraction=0.1, 
#             verbose=False,
#             warm_start=False)

#     clf.fit(trn_X, trn_y)
#     oof_preds[val_idx] = clf.predict_proba(val_X)[:,1]
#     sub_preds += clf.predict_proba(subMerged)[:,1] / folds.n_splits
    
#     del clf, trn_X, trn_y, val_X, val_y
#     gc.collect()