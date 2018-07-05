import copy
import itertools
import gc
import lightgbm as lgb
from joblib import Parallel, delayed
import pandas as pd

class LGBM(object):
    def __init__(self, params):
        self._params_ = params

    def gridsearch(self, param_grid, cv_params):
        cv_params_grid = self._create_cv_params_(param_grid, cv_params)
        # gs_cv_results = Parallel(n_jobs=-1)(delayed(self._cv_)(param_set) for param_set in cv_params_grid)
        gs_cv_results = []
        for param_set in cv_params_grid:
            cv_result = self._cv_(param_set)
            gs_cv_results.append(cv_result)
        gs_cv_results = sorted(gs_cv_results, key = lambda x : x[0])
        best_params_ = gs_cv_results[-1][2:]
        best_params = dict(zip(list(self._params_.keys()), best_params_))
        cols = ['result', 'iterations']
        cols.extend(list(self._params_.keys()))
        gs_cv_results = pd.DataFrame(data=gs_cv_results, columns=cols)
        return gs_cv_results, best_params

    def _cv_(self, cv_params):
        cv_results = lgb.cv(**cv_params)
        cv_best_result = cv_results[self._params_["metric"] + "-mean"][-1]
        iterations = len(cv_results['auc-mean'])
        del cv_params["train_set"]
        del cv_results
        results = [cv_best_result, iterations]
        results.extend(list(cv_params['params'].values()))
        gc.collect()
        return results

    def _build_cv_params_(self, cv_params, param_set):
        params = copy.deepcopy(cv_params)
        params.update(params=param_set)
        return params

    def _create_cv_params_(self, param_grid, cv_params):
        param_sets = None
        self._create_grid_space_(param_grid)
        for param_name, param_values in iter(param_grid.items()):
            if param_sets is None:
                param_sets = [copy.deepcopy(self._params_) for i in range(len(param_values))]
            for idx, param_set in enumerate(param_sets):
                param_set[param_name] = param_values[idx]
        return [self._build_cv_params_(cv_params, param_set) for param_set in param_sets]

    def _create_grid_space_(self, param_grid):
        value_space =  list(itertools.product(*param_grid.values()))
        value_matrix = [list(i) for i in zip(*value_space)]
        key_idx = -1
        for param_name in param_grid:
            key_idx += 1
            param_grid[param_name] = value_matrix[key_idx]
