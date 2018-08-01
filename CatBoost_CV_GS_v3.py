import Dataset
from Estimators import CATBOOST
from Utils import Profiler
import pandas as pd
from IPython.display import display
import catboost as catb

profile = Profiler()
profile.Start()

# Gather Data
train_X, test_X, train_Y = Dataset.Load('AllData_v2')

# Convert data to DMatrix
cat_train = catb.Pool(train_X, train_Y)
cat_test = catb.Pool(test_X)

# Define estimator parameters
params = {'loss_function'   :'Logloss',
          'eval_metric'     :'AUC',
          'random_seed'     :1453,
          'l2_leaf_reg'     :3,
          'od_type'         :'Iter',
          'depth'           :6,
          'scale_pos_weight':1,
          'od_wait'         :50
}

# Parameters that are to be supplied to cross-validation
cv_params = {
    "pool"          : cat_train,
    "iterations"    : 10000,
    "fold_count"    : 5,
    "logging_level" : "Silent"
}

# Step 1
param_grid = {"depth"    :  range(3,9,1)}
cat = CATBOOST(params)
gs_results, params = cat.gridsearch(param_grid, cv_params)
gs_summary = gs_results

# Step 2
param_grid = {"l2_leaf_reg"    : [0.01,0.03,0.1,0.3]}
cat = CATBOOST(params)
gs_results, params = cat.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 3
param_grid = {"border_count"    : [32,64,127,255]}
cat = CATBOOST(params)
gs_results, params = cat.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 4
param_grid = {"ctr_border_count"    : [32,64,127,255]}
cat = CATBOOST(params)
gs_results, params = cat.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 5
param_grid = {"learning_rate"    : [0.001,0.003,0.01,0.03,0.1,0.3]}
cat = CATBOOST(params)
gs_results, params = cat.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

# Step 6
param_grid = {"scale_pos_weight"    : [92/8, 1]}
cat = CATBOOST(params)
gs_results, params = cat.gridsearch(param_grid, cv_params)
gs_summary = pd.concat([gs_summary, gs_results], ignore_index=True)

print('All Iterations')
display(gs_summary)
print('Best parameters: ')
best_cv = gs_results.loc[gs_results['result'].idxmax()]
display(best_cv)

profile.End()
print('Time elapsed: %s mins' %str(profile.ElapsedMinutes))


# Save CV process
gs_summary.to_csv('../AllData_v2_CATBOOST_GS.csv')

# Generate model by best iteration
model   = catb.train(params=params,
                     pool=cat_train,
                     num_boost_round=best_cv[1],
                     logging_level="Verbose")

# Save model for possible coded ensemble
model.save_model('../AllData_v2_CATBOOST_Model')

# Generate train prediction for future ensemble
train_preds = model.predict(train_X, prediction_type="Probability")
data = pd.read_csv('../input/application_train.csv')

data['preds'] = train_preds[:,1]
data = data[['SK_ID_CURR', 'preds']]
data.to_csv('../AllData_v2_CATBOOST_TrainPreds.csv', index=False)

# Generate sub prediction for Kaggle
sub_preds = model.predict(test_X, prediction_type="Probability")
sub = pd.read_csv('../input/sample_submission.csv')

sub['TARGET'] = sub_preds[:,1]
sub.to_csv('../AllData_v2₺_CATBOOST_Preds.csv', index=False)