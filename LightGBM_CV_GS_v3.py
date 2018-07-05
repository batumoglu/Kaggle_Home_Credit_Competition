import Dataset as dataset
from lightgbm import Dataset
from Estimators import LGBM
from Utils import Profiler
import pandas as pd
from IPython.display import display

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
          'scale_pos_weight'        :92/8,
          'metric'                  :'auc',
          'verbose'                 :-1}

# Parameters that are to be supplied to cross-validation
cv_params = {
    "train_set"             : lgb_train,
    "num_boost_round"       : 100,
    "nfold"                 : 3,
    "early_stopping_rounds" : 20,
    "verbose_eval"          : 10
}

param_grid = {"num_leaves"    : range(20,101,20)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = gs_results

param_grid = {"max_depth"    : range(3,8,1)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

display(gs_summary)
print('Best parameters: ')
display(gs_results.loc[gs_results['result'].idxmax()])

profile.End()

gs_summary.to_csv('../AllData_v3_LGBM_GS.csv')


