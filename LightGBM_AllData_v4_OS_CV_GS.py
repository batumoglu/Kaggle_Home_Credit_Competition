import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from Utils import Profiler
from IPython.display import display
from Estimators import LGBM

profile = Profiler()
profile.Start()

print("Loading resampled train data")
train_X = pd.read_csv("../input/AllData_v4_os.train")
train_X.pop("Unnamed: 0")

print("Loading resampled train labels")
train_y = pd.read_csv("../input/AllData_v4_os.label")
train_y = train_y.pop("TARGET")

print("Loading resampled validation data")
valid_X = pd.read_csv("../input/AllData_v4_os_valid.train")
valid_X.pop("Unnamed: 0")

print("Loading resampled validation labels")
valid_y = pd.read_csv("../input/AllData_v4_os_valid.label")
valid_y = valid_y.pop("TARGET")

print("Loading application test data")
test_X = pd.read_csv("../input/AllData_v4.test")

print("train_y shape: " + str(train_y.shape))
print("train_X shape: " + str(train_X.shape))
print("valid_y shape: " + str(valid_y.shape))
print("valid_X shape: " + str(valid_X.shape))
print("test_X shape: " + str(test_X.shape))

lgb_train = lgb.Dataset(train_X, train_y)
lgb_test = lgb.Dataset(valid_X)

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
print("Performing Gridsearch Step-1")
param_grid = {"num_leaves"    : range(10,101,10)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = gs_results

# Step 2
print("Performing Gridsearch Step-2")
param_grid = {"max_depth"    : range(3,10,1)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 3
print("Performing Gridsearch Step-3")
param_grid = {"min_data_in_leaf"    : range(10,81,10)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 4
print("Performing Gridsearch Step-4")
param_grid = {"lambda_l1"    : [i/10.0 for i in range(0,8)]}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 5
print("Performing Gridsearch Step-5")
param_grid = {"lambda_l2"    : [i/10.0 for i in range(0,8)]}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 6
print("Performing Gridsearch Step-6")
param_grid = {"scale_pos_weight"    : [92/8, 1]}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 7
print("Performing Gridsearch Step-7")
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
gs_summary.to_csv('../AllData_v4_OS_LGBM_GS.csv')

# Generate model by best iteration
print("Model training started...")
model   = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=int(best_cv[1]/0.8),
                    verbose_eval=1)
print("Model training completed...")

# Save model for possible coded ensemble
model.save_model('GridSearch/AllData_v4_OS_LGBM_Model', num_iteration=best_cv[1])

print("Predicting validation set...")
valid_preds = model.predict(valid_X)
print("Validation set prediction completed...")

print("Predicting test set...")
test_preds = model.predict(test_X)
print("Test set prediction completed...")

auc = roc_auc_score(valid_y, valid_preds)
print("Validation AUC: " + str(auc))

valid_preds = pd.DataFrame(valid_preds)
valid_preds.to_csv("GridSearch/AllData_v4_OS_GS_LGBM_ValidPreds.csv", index=False)

sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = test_preds
sub.to_csv('GridSearch/AllData_v4_OS_GS_LGBM_Preds.csv', index=False)


