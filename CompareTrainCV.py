import pandas as pd

import lightgbm
import xgboost
import catboost

""" LightGBM Model """

params = {'task'            :'train',
          'objective'       :'binary',
          'device'          :'gpu',
          'max_bin'         :255,
          'gpu_platform_id' :0,
          'gpu_device_id'   :0,
          'gpu_use_dp'      :True,
          'sparse_threshold':1}

# Tunable Parameters
# Tree Design Parameters
# Type of tree
params['boosting']              = 'gbdt'    # ['gbdt', 'rf', 'dart', 'goss']

# Properties of tree
params['max_depth']             = 3         # np.arange(3,7,1)
params['num_leaves']            = 31        # np.arange(10,41,10)
params['min_data_in_leaf']      = 20        # np.arange(10,41,10)

# Punishment on unnecessary features
params['lambda_l1']             = 0.1       # np.arange(0.1,0.6,0.1)
params['lambda_l2']             = 0.1       # np.arange(0.1,0.6,0.1)

# Iterations and learning rate
params['learning_rate']         = 0.1       # [0.01,0.03,0.1,0.3]
params['num_boost_round']       = 5000      # Fixed as early stopping will decide when to stop
params['early_stopping_rounds'] = 50        # 30, 50, 100

# Data Design Parameters
params['bagging_fraction']      = []        # [0.5,0.6,0.7,0.8,0.9,1]
params['bagging_freq']          = []        # np.arange(10,51,10)
params['feature_fraction']      = []        # [0.5,0.6,0.7,0.8,0.9,1]

lightgbm.cv(params, train_set, num_boost_round=100, 
fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto', early_stopping_rounds=None, callbacks=None, verbose_eval=None,
folds=None, nfold=5, stratified=True, shuffle=True, metrics=None, fpreproc=None,  show_stdv=True, seed=0)

lightgbm.train(params , train_set , num_boost_round=100,
fobj=None, feval=None, init_model=None, feature_name='auto', categorical_feature='auto', early_stopping_rounds=None, callbacks=None, verbose_eval=True,
valid_sets=None, valid_names=None, evals_result=None, learning_rates=None, keep_training_booster=False)

"""  Xgboost model  """

params = {  'tree_method'       :'gpu_hist',
            'objective'         :'gpu:binary:logistic',
            'eval_metric'       :'auc',
            'gpu_id'            :0}



# Tunable Parameters
# Tree Design Parameters
# Type of tree
params['booster']              = 'gbtree'    # ['gbtree', 'gblinear', 'dart']  ok

# Properties of tree
params['max_depth']             = 3         # np.arange(3,8,1)
params['min_child_weight']      = 5         # np.arange(1,11,1)

# Punishment on unnecessary features
params['alpha']                 = 0.1       # np.arange(0.1,0.6,0.1)        
params['lambda']                = 0.1       # np.arange(0.1,0.6,0.1)

# Iterations and learning rate
params['eta']                   = 0.1       # [0.01,0.03,0.1,0.3]
params['num_boost_round']       = 5000      # Fixed as early stopping will decide when to stop
params['early_stopping_rounds'] = 50        # 30, 50, 100

# Data Design Parameters
params['subsample']             = []        # [0.5,0.6,0.7,0.8,0.9,1]
params['colsample_bytree']      = []        # np.arange(10,51,10)
params['colsample_bylevel']     = []        # [0.5,0.6,0.7,0.8,0.9,1]

xgboost.cv(params, dtrain, num_boost_round=10, obj=None, feval=None, maximize=False, early_stopping_rounds=None, verbose_eval=None, callbacks=None,
nfold=3, stratified=False, folds=None, metrics=(), fpreproc=None, as_pandas=True, show_stdv=True, seed=0,  shuffle=True)

xgboost.train(params, dtrain, num_boost_round=10, obj=None, feval=None, maximize=False, early_stopping_rounds=None, verbose_eval=True, callbacks=None,
evals=(), evals_result=None,  xgb_model=None, learning_rates=None)¶


""" CatBoost Model """

params = {  'task_type'         :'GPU',
            'devices'           :0,
            'eval_metric'       :'auc'}



# Tunable Parameters
# Tree Design Parameters
# Type of tree
params['booster']              = 'gbtree'    # ['gbtree', 'gblinear', 'dart']  ok

# Properties of tree
params['max_depth']             = 3         # np.arange(3,8,1)
params['min_child_weight']      = 5         # np.arange(1,11,1)

# Punishment on unnecessary features
params['alpha']                 = 0.1       # np.arange(0.1,0.6,0.1)        
params['lambda']                = 0.1       # np.arange(0.1,0.6,0.1)

# Iterations and learning rate
params['eta']                   = 0.1       # [0.01,0.03,0.1,0.3]
params['num_boost_round']       = 5000      # Fixed as early stopping will decide when to stop
params['early_stopping_rounds'] = 50        # 30, 50, 100

# Data Design Parameters
params['subsample']             = []        # [0.5,0.6,0.7,0.8,0.9,1]
params['colsample_bytree']      = []        # np.arange(10,51,10)
params['colsample_bylevel']     = []        # [0.5,0.6,0.7,0.8,0.9,1]

catboost.cv(params, pool=None,  iterations=None, num_boost_round=None, 
fold_count=3, nfold=None, inverted=False, partition_random_seed=0, seed=None, shuffle=True, logging_level=None, stratified=False, as_pandas=True)

catboost.train(params, pool=None, iterations=None, num_boost_round=None, 
evals=None, verbose=None, logging_level=None, eval_set=None, plot=None)
