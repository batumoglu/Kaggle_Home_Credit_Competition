import os
from lightgbm import Dataset
from Estimators import LGBM
from Utils import Profiler
import pandas as pd
from IPython.display import display
import lightgbm as lgb

profile = Profiler()
profile.Start()

def get_preds(pred_type):
    directory = "C:/Users/U2R/Desktop/ensemble all/Preds/{}".format(pred_type)
    preds = []
    for filename in os.listdir(directory):
        try:
            df = pd.read_csv("{}/{}".format(directory,filename))
        except:
            print("Error reading file {}".format(filename))
            return
        df.set_index("SK_ID_CURR", inplace=True)
        if pred_type == "Test":
            col = "TARGET"
        else:
            col = "preds"
        df.rename(columns={col:"preds_{}".format(filename[:2])}, inplace=True)
        preds.append(df)
    all_preds = pd.concat(preds, axis=1, join_axes=[preds[0].index])
    return all_preds

X_train = get_preds("Train")
y_train = pd.read_csv("../input/application_train.csv")
y_train.set_index("SK_ID_CURR", inplace=True)
y_train = y_train["TARGET"]

X_test = get_preds("Test").drop(["preds_07"], axis=1)

print(X_test.head(10))

# corr = preds.corr()

# corr.to_csv("ModelCorr.csv")

# # for col in corr.columns:
# #     print(corr[col][corr[col] < 0.9].index)

# selected_columns = ["preds_01", "preds_03", "preds_07", "preds_10"]

# df = preds[selected_columns]

# print(df.head(10))
# df["TARGET"] = df.mean(axis=1)
# df[["TARGET"]].to_csv("Ensemble_LowCorr.csv")

# Convert data to DMatrix
lgb_train = Dataset(X_train, y_train)
lgb_test = Dataset(X_test)

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
param_grid = {"num_leaves"    : range(10,51,10)}
lgbm = LGBM(params)
gs_results, params = lgbm.gridsearch(param_grid, cv_params)
gs_summary = gs_results


print('All Iterations')
display(gs_summary)
print('Best parameters: ')
best_cv = gs_results.loc[gs_results['result'].idxmax()]
display(best_cv)

profile.End()
print('Time elapsed: %s mins' %str(profile.ElapsedMinutes))

# Save CV process
gs_summary.to_csv('../AllModel_Preds_LGBM_GS.csv')

# Generate model by best iteration
model   = lgb.train(params=params,
                    train_set=lgb_train,
                    num_boost_round=int(best_cv[1]/0.8),
                    verbose_eval=1)

# Save model for possible coded ensemble
model.save_model('../AllModel_Preds_LGBM_Model', num_iteration=best_cv[1])

# Generate train prediction for future ensemble
train_preds = model.predict(X_train)
data = pd.read_csv('../input/application_train.csv')
data['preds'] = train_preds
data = data[['SK_ID_CURR', 'preds']]
data.to_csv('../AllModel_Preds_LGBM_TrainPreds.csv', index=False)

# Generate sub prediction for Kaggle
sub_preds = model.predict(X_test)
sub = pd.read_csv('../input/sample_submission.csv')
sub['TARGET'] = sub_preds
sub.to_csv('../AllModel_Preds_LGBM_Preds.csv', index=False)

