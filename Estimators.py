import copy
import lightgbm as lgb
from joblib import Parallel, delayed

class LGBM(object):
    def __init__(self, params):
        self._params_ = params

    def gridsearch(self, param_grid, cv_params):
        cv_params_grid = self._create_cv_params_(param_grid, cv_params)
        gs_cv_results = Parallel(n_jobs=-1)(delayed(self._cv_)(param_set) for param_set in cv_params_grid)
        return gs_cv_results

    def _cv_(self, cv_params):
        cv_results = lgb.cv(**cv_params)
        del cv_params["train_set"]
        return (cv_results[self._params_["metric"] + "-mean"][-1], cv_params)

    def _build_cv_params_(self, cv_params, param_set):
        params = copy.deepcopy(cv_params)
        params.update(params=param_set)
        return params

    def _create_cv_params_(self, param_grid, cv_params):
        param_sets = None
        for param_name, param_values in iter(param_grid.items()):
            if param_sets is None:
                param_sets = [copy.deepcopy(self._params_) for i in range(len(param_values))]
            for idx, param_set in enumerate(param_sets):
                param_set[param_name] = param_values[idx]
        return [self._build_cv_params_(cv_params, param_set) for param_set in param_sets]


