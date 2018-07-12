import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import numpy as np

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

print("Replacing Inf values in valid_X")
valid_X = valid_X.replace([np.inf, -np.inf], np.nan)
print("Replacing NaN values in valid_X")
valid_X = valid_X.fillna(0)

print("Replacing Inf values in test_X")
test_X = test_X.replace([np.inf, -np.inf], np.nan)
print("Replacing NaN values in test_X")
test_X = test_X.fillna(0)

print("train_y shape: " + str(train_y.shape))
print("train_X shape: " + str(train_X.shape))
print("valid_y shape: " + str(valid_y.shape))
print("valid_X shape: " + str(valid_X.shape))
print("test_X shape: " + str(test_X.shape))

lgb_train = lgb.Dataset(train_X, train_y)
lgb_test = lgb.Dataset(valid_X)

params = {'task'                    :'train',
          'objective'               :'binary',
          'learning_rate'           :0.01,
          'num_leaves'              :10,
          'max_depth'               :3,
          'min_data_in_leaf'        :80,
          'min_sum_hessian_in_leaf' :0.001,
          'lambda_l1'               :0.2,
          'lambda_l2'               :0,
          'scale_pos_weight'        :1,
          'metric'                  :'auc',
          'verbose'                 :-1}

print("Model training started...")
model   = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=10000,
                    verbose_eval=True)
print("Model training completed...")

model.save_model('GridSearch/AllData_v4_OS_LGBM_v2_Model')

print("Predicting validation set...")
valid_preds = model.predict(valid_X)
print("Validation set prediction completed...")

print("Predicting test set...")
test_preds = model.predict(test_X)
print("Test set prediction completed...")

auc = roc_auc_score(valid_y, valid_preds)
print("Validation AUC: " + str(auc))

valid_preds = pd.DataFrame(valid_preds)
valid_preds.to_csv("GridSearch/AllData_v4_OS_LGBM_ValidPreds_v2.csv", index=False)

sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = test_preds
sub.to_csv('GridSearch/AllData_v4_OS_LGBM_Preds_v2.csv', index=False)


