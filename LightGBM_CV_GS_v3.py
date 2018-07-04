import Dataset as dataset
from lightgbm import Dataset
from Estimators import LGBM


# Gather Data
train_X, test_X, train_Y = dataset.Load('ApplicationOnly')

# Convert data to DMatrix
lgb_train = Dataset(train_X, train_Y)
lgb_test = Dataset(test_X)

# Define estimator parameters
params = {'task'                    :'train',
          'objective'               :'binary',
          'learning_rate'           :0.1,
          'num_leaves'              :31,
          'max_depth'               :-1,
          'min_data_in_leaf'        :20,
          'min_sum_hessian_in_leaf' :0.001,
          'lambda_l1'               :0,
          'lambda_l2'               :0,
          'scale_pos_weight'        :92/8,
          'metric'                  :'auc'}

# Parameters that are to be supplied to cross-validation
cv_params = {
    "train_set"             : lgb_train,
    "num_boost_round"       : 1000,
    "nfold"                 : 5,
    "early_stopping_rounds" : 20,
    "verbose_eval"          : 10
}

# 5 grids shall be explored with different values of both max_depth and num_leaves
param_grid = {
    "max_depth"     : range(3,8,1),
    "num_leaves"    : range(10,51,10)
}

# Create LightGBM wrapper instance with estimator parameters
lgbm = LGBM(params)

# Search grid space with CV, cv_params specifies cross-validation parameters
gs_results = lgbm.gridsearch(param_grid, cv_params)

# best score of explored grids returned with corresponding parameter set
# in this case results shall include 9 scores
print(gs_results["all"]) # all results
print(gs_results["best"]) # best result

import json
with open("../gs_results.txt","w") as gs_log:
    gs_log.write(json.dumps(gs_results))
