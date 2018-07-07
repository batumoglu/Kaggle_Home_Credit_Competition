import Dataset as dataset
from lightgbm import Dataset
from Estimators import LGBM
from Utils import Profiler
import pandas as pd
from IPython.display import display
import lightgbm as lgb

profile = Profiler()
profile.Start()

# Gather Data
train_X, test_X, train_Y = dataset.Load('AllData_v3')

# Convert data to DMatrix
lgb_train = Dataset(train_X, train_Y)
lgb_test = Dataset(test_X)

# Define estimator parameters
params = {'task'                    :'train',
          'objective'               :'binary',
          'learning_rate'           :0.1,
          'num_leaves'              :31,
          'max_depth'               :8,
          'min_data_in_leaf'        :20,
          'min_sum_hessian_in_leaf' :0.001,
          'lambda_l1'               :0,
          'lambda_l2'               :0,
          'scale_pos_weight'        :1,
          'metric'                  :'auc',
          'verbose'                 :-1}

# Parameters that are to be supplied to cross-validation
cv_params = {
    "train_set"             : lgb_train,
    "num_boost_round"       : 10000,
    "nfold"                 : 5,
    "early_stopping_rounds" : 50,
    "verbose_eval"          : 10
}

# Step 1
param_grid = {"num_leaves"    : range(10,101,10)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = gs_results

# Step 2
param_grid = {"max_depth"    : range(3,10,1)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 3
param_grid = {"min_data_in_leaf"    : range(10,81,10)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 4
param_grid = {"lambda_l1"    : [i/10.0 for i in range(0,8)]}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 5
param_grid = {"lambda_l2"    : [i/10.0 for i in range(0,8)]}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 6
param_grid = {"scale_pos_weight"    : [92/8, 1]}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 7
param_grid = {"learning_rate"    : [0.01,0.02, 0.03,0.05,0.08,0.1]}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

print('All Iterations')
display(gs_summary)
print('Best parameters: ')
best_cv = gs_results.loc[gs_results['result'].idxmax()]
display(best_cv)

profile.End()
print('Time elapsed: %s mins' %str(profile.ElapsedMinutes))


# Save CV process
gs_summary.to_csv('../AllData_v3_LGBM_GS.csv')

# Generate model by best iteration
model   = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=int(best_cv[1]/0.8),
                    verbose_eval=1)

# Save model for possible coded ensemble
model.save_model('../AllData_v3_LGBM_Model', num_iteration=best_cv[1])

# Generate train prediction for future ensemble
train_preds = model.predict(train_X)
data = pd.read_csv('../input/application_train.csv')
data['preds'] = train_preds
data = data[['SK_ID_CURR', 'preds']]
data.to_csv('../AllData_v3_LGBM_TrainPreds.csv', index=False)

# Generate sub prediction for Kaggle
sub_preds = model.predict(test_X)
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('../AllData_v3_LGBM_Preds.csv', index=False)