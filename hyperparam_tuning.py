from sklearn.experimental import enable_halving_search_cv  # required
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import paths
from copy import deepcopy
from data import load_data, build_data_prep_pipe, build_cv_pipe, stratified_split
from config import get_config, update_estimator_in_config, inherit_from_config, FROM_CONFIG
import pandas as pd
from typing import Dict, List, Tuple
import xgboost as xgb
import lightgbm as lgb
import catboost as catb
import numpy as np
import cv_result_manager
from itertools import product
from pipe_cross_val import set_pipecv
import os
os.environ['PYTHONWARNINGS'] = 'ignore'

config_grid = {
    'random_state': [1337]
}

balance_grid = {
    'RandomUnderSampler': {
        'random_state': [FROM_CONFIG],
    },
    'SMOTENC': {
        'random_state': [FROM_CONFIG],
        'categorical_features': ['auto'],
        'k_neighbors': [3, 5]
    },
    'NearMiss': {
        'version': [1, 2, 3],
        'n_neighbors': [3],
    },
    'InstanceHardnessThreshold': {
        'estimator': [None, RandomForestClassifier(n_estimators=10, random_state=1)],
        'cv': [3, 5],
        'random_state': [FROM_CONFIG]
    }
}

model_grids = {
    'XGB': {
        'estimator': xgb.XGBClassifier,
        'base_grid': {
            'max_depth': np.arange(3, 5),
            'learning_rate': np.linspace(0.01, 0.1, 5)
        },
        'fine_grid': {
            'min_child_weight': [1, 3, 5, 7, 10],
            'gamma': [0, .1, .2, .3, .4],
            'subsample': [.6, .8, 1.],
            'colsample_bytree': [.6, .8, 1.],
            'reg_alpha': [1e-5, 1e-3, 1e-2, 1, 1e2],
            'reg_lambda': [1e-5, 1e-3, 1e-2, 1, 1e2],
        }
    },
    'LGB': {
        'estimator': lgb.LGBMClassifier,
        'base_grid': {
            'max_depth': np.arange(3, 5),
            'learning_rate': np.linspace(0.01, 0.1, 5)
        },
        'fine_grid': {
            'min_child_weight': [1, 3, 5, 7, 10],
            'subsample': [.6, .8, 1.],
        }
    },
    'CATB': {
        'estimator': catb.CatBoostClassifier,
        'base_grid': {
            'max_depth': np.arange(3, 5),
            'learning_rate': np.linspace(0.01, 0.1, 5)
        },
        'fine_grid': {
            'bagging_temperature': [0, .5, 1, 5],
        }
    }
}


def get_best_iteration(estimator):
    for attr in ['best_iteration', 'best_iteration_', '_best_iteration']:
        if hasattr(estimator, attr):
            return getattr(estimator, attr)
    assert AttributeError("Estimator doesnt have best iteration attribute")


def yield_from_grid(grid_dict: Dict, default_dict: Dict = None):
    default_dict = deepcopy(default_dict) if default_dict else {}
    keys = grid_dict.keys()
    if default_dict:
        assert set(keys).issubset(default_dict.keys())
    for vals in product(*grid_dict.values()):
        result = default_dict
        result.update(dict(zip(keys, vals)))
        yield result


def search_model_and_config():
    for config in yield_from_grid(grid_dict=config_grid, default_dict=get_config()):

        for balance_method, balance_params_grid in balance_grid.items():
            for balance_params in yield_from_grid(balance_params_grid):
                config['balance'] = {
                    'method': balance_method,
                    'params': inherit_from_config(balance_params, config)
                }
                search_model(config)


def search_model(config: Dict):

    cv_args = {'cv': config['cv.n_folds'], 'scoring': config['cv.scores'],
               'verbose': 0, 'refit': config['cv.main_score'],
               'return_train_score': True, 'n_jobs': -1}

    seed = config['random_state']

    raw_data = load_data(config)
    prep_pipe = build_data_prep_pipe(config)
    Xy = prep_pipe.fit_transform(raw_data)
    cv_pipe = build_cv_pipe(config, Xy)

    Xy_train, Xy_test = stratified_split(Xy, test_size=config['data.test_size'], random_state=seed)
    Xy_base_train, Xy_base_val = stratified_split(Xy_train, test_size=config['cv.base.val_size'], random_state=seed)
    Xy_base_val = cv_pipe.fit(Xy_base_train).transform(Xy_base_val)

    for model_name in model_grids:

        estimator_class = model_grids[model_name]['estimator']
        base_grid = model_grids[model_name]['base_grid']
        fine_grid = model_grids[model_name]['fine_grid']

        DEV = False
        if DEV:
            # smaller grid for dev/debug..
            base_grid = _reduce_grid(base_grid)
            fine_grid = _reduce_grid(fine_grid)

        # -----
        # base:

        print(model_name, "- Searching base params...")

        cv = GridSearchCV(estimator_class(
            early_stopping_rounds=config['cv.base.early_stopping_rounds'],
            random_state=seed), param_grid=base_grid, **cv_args)
        set_pipecv(cv, cv_pipe)

        cv.fit(*Xy_base_train, eval_set=[Xy_base_val])

        params = cv.best_params_
        params['n_estimators'] = get_best_iteration(cv.best_estimator_)

        print(model_name + ":")
        print(" Best base params:", params)
        print(f" Score ({cv.refit}): {cv.best_score_:2.3f}")

        # -----
        # fine tune:

        print(model_name, "- Fine tuning...")

        cv = GridSearchCV(
            estimator_class(**params, random_state=seed),
            param_grid=fine_grid, **cv_args)
        set_pipecv(cv, cv_pipe)

        cv.fit(*Xy_train)

        params = {**cv.best_params_, **params}

        fitted_config = update_estimator_in_config(config, model_name, params)

        print(model_name + ":")
        print(" Best fine-tuned params:", cv.best_params_)
        print(f" Score ({cv.refit}): {cv.best_score_:2.3f}")

        cv_result_manager.process_result(fitted_config, cv)


def _reduce_grid(grid: Dict) -> Dict:
    return {k: [v[0], v[-1]] if len(v) > 1 else v for k, v in grid.items()}


if __name__ == "__main__":
    search_model_and_config()
