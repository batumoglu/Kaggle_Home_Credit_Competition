import Dataset
from Estimators import XGB
from Utils import Profiler
import pandas as pd
from IPython.display import display
import xgboost

profile = Profiler()
profile.Start()

# Gather Data
train_X, test_X, train_Y = Dataset.Load('AllData_v3')

# Convert data to DMatrix
dtrain = xgb.DMatrix(train_X, train_Y)
dtest = xgb.DMatrix(test_X)

# Define estimator parameters
params = {'eta'                 :0.3,
          'gamma'               :0,
          'max_depth'           :6,
          'min_child_weight'    :1,
          'subsample'           :1,
          'colsample_bytree'    :1,
          'colsample_bylevel'   :1,
          'lambda'              :1,
          'alpha'               :0,
          'scale_pos_weight'    :92/8,
          'objective'           :'binary:logistic',
          'eval_metric'         :'auc'
}

# Parameters that are to be supplied to cross-validation
cv_params = {
    "dtrain"                : dtrain,
    "num_boost_round"       : 1000,
    "nfold"                 : 5,
    "maximize"              : True,
    "early_stopping_rounds" : 10,
    "verbose_eval"          : 1
}

# Step 1
param_grid = {"num_leaves"    : range(10,101,10)}
xgb = XGB(params)
gs_results, params = xgb.gridsearch(param_grid, cv_params)
gs_summary = gs_results

# Step 2
param_grid = {"max_depth"    : range(3,10,1)}
xgb = XGB(params)
gs_results, params = xgb.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 3
param_grid = {"min_data_in_leaf"    : range(10,81,10)}
xgb = XGB(params)
gs_results, params = xgb.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 4
param_grid = {"lambda_l1"    : [i/10.0 for i in range(0,8)]}
xgb = XGB(params)
gs_results, params = xgb.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 5
param_grid = {"lambda_l2"    : [i/10.0 for i in range(0,8)]}
xgb = XGB(params)
gs_results, params = xgb.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 6
param_grid = {"scale_pos_weight"    : [92/8, 1]}
xgb = XGB(params)
gs_results, params = xgb.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

print('All Iterations')
display(gs_summary)
print('Best parameters: ')
best_cv = gs_results.loc[gs_results['result'].idxmax()]
display(best_cv)

profile.End()
print('Time elapsed: %s mins' %str(profile.ElapsedMinutes))


# Save CV process
gs_summary.to_csv('../AllData_v3_XGB_GS.csv')

# Generate model by best iteration
model   = xgboost.train(params=params,
                        dtrain=dtrain,
                        num_boost_round=best_cv[1],
                        maximize=True,
                        verbose_eval=10)

# Save model for possible coded ensemble
model.save_model('../AllData_v3_XGB_Model', num_iteration=best_cv[1])

# Generate train prediction for future ensemble
train_preds = model.predict(train_X)
data = pd.read_csv('../input/application_train.csv')
data['preds'] = train_preds
data = data[['SK_ID_CURR', 'preds']]
data.to_csv('../AllData_v3_XGB_TrainPreds.csv', index=False)

# Generate sub prediction for Kaggle
sub_preds = model.predict(test_X)
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('../AllData_v3_XGB_Preds.csv', index=False)