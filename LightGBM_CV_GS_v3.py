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

param_grid = {"num_leaves"    : range(20,41,20)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = gs_results

param_grid = {"max_depth"    : range(3,5,1)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

print('All Iterations')
display(gs_summary)
print('Best parameters: ')
best_cv = gs_results.loc[gs_results['result'].idxmax()]
display(best_cv)

profile.End()

gs_summary.to_csv('../AllData_v3_LGBM_GS.csv')

# Generate model by best iteration
model   = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=best_cv[1],
                    verbose_eval=1)

model.savel_model('AllData_v3_LGBM_Model', num_iteration=best_cv[1])

train_preds = model.predict(train_X)
data = pd.read_csv('../input/application_train.csv')
data = data['SK_ID_CURR']
data['preds'] = train_preds
data.to_csv('AllData_v3_LGBM_TrainPreds.csv', index=False)

sub_preds = model.predict(test_X)
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('AllData_v3_LGBM_Score.csv', index=False)