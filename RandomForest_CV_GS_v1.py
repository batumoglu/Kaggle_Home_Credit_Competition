import Dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import Imputer
from Utils import Profiler

profile = Profiler()
profile.Start()


# Gather Data
train_X, test_X, train_Y = Dataset.Load('AllData_v3')

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
train_X = imp.fit_transform(train_X)

clf = RandomForestClassifier(n_estimators=1000,
                             oob_score=True,
                             random_state=50, 
                             max_features="auto", 
                             min_samples_leaf=50)


param_grid = {
    "min_samples_leaf" : range(10,51,10)
}

grid_search = GridSearchCV(clf,
                         param_grid = param_grid,
                         scoring="roc_auc",
                         iid=False,
                         cv=5)

grid_search.fit(train_X, train_Y)

profile.End()

print("RF best score: " + grid_search.best_score_)
print("RF best score: " + grid_search.best_params_)

import json
with open("../rf_gs_results.txt","w") as gs_log:
    gs_log.write("RandomForest_CV_GS_v1 ran for " + str(profile.ElapsedMinutes) + " minutes\n\n")
    gs_log.write("Best score: " + json.dumps(grid_search.best_score_) + "\n")
    gs_log.write("Best params: " + json.dumps(grid_search.best_params_) + "\n")
    gs_log.write(json.dumps(grid_search.cv_results_))