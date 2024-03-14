from sklearn.experimental import enable_halving_search_cv  # required
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, StratifiedKFold
import paths
from data import load_data, build_pipeline, DataSplitter
from config import get_config, get_config_id
import pandas as pd
from typing import Dict, List, Tuple
import xgboost as xgb
import lightgbm as lgb
import catboost as catb
import numpy as np
import cv_result_manager


grids = {
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


def _get_best_iteration(clf):
    for attr in ['best_iteration', 'best_iteration_', '_best_iteration']:
        if hasattr(clf, attr):
            return getattr(clf, attr)
    assert AttributeError()


def tune(config: Dict):

    cv_args = {'cv': config['cv.n_folds'], 'scoring': config['cv.scores'],
               'verbose': 0, 'refit': config['cv.main_score'], 'return_train_score': True, 'n_jobs': -1}

    random_state = config['random_state']

    pipe = build_pipeline(config, verbose=2)
    raw_data = load_data(config)
    Xy = pipe.fit_transform(raw_data)

    data_splitter = DataSplitter(random_state=random_state, shuffle=True, stratify=True)
    Xy_train, Xy_test = data_splitter.split(Xy, test_size=config['data.test_size'])
    Xy_base_train, Xy_base_val = data_splitter.split(Xy_train, test_size=config['cv.base.val_size'])

    for model_name in grids:

        estimator_class = grids[model_name]['estimator']
        base_grid = grids[model_name]['base_grid']
        fine_grid = grids[model_name]['fine_grid']

        DEV = False
        if DEV:
            # smaller grid for dev/debug..
            base_grid = _reduce_grid(base_grid)
            fine_grid = _reduce_grid(fine_grid)

        model = estimator_class(early_stopping_rounds=config['cv.base.early_stopping_rounds'],
                                random_state=random_state)

        # find best base params:
        print(model_name, "- Searching for best base params...")
        cv = GridSearchCV(model, param_grid=base_grid, **cv_args)
        cv.fit(*Xy_base_train, eval_set=[Xy_base_val])
        params = cv.best_params_
        params['n_estimators'] = _get_best_iteration(cv.best_estimator_)
        print(model_name + ":")
        print(" Best base params:", params)
        print(" Score:", cv.best_score_)

        # fine tune:
        print(model_name, "- Fine tuning...")
        model = estimator_class(**params, random_state=random_state)
        cv = GridSearchCV(model, param_grid=fine_grid, **cv_args)
        cv.fit(*Xy_train)

        print(model_name + ":")
        print(" Best fine-tuned params:", params)
        print(" Score:", cv.best_score_)

        results_df = cv_result_manager.make_dataframe(config, model_name, random_state, cv.cv_results_)
        cv_result_manager.display(results_df)
        cv_result_manager.save(results_df)


def _reduce_grid(grid: Dict) -> Dict:
    return {k: [v[0], v[-1]] if len(v) > 1 else v for k, v in grid.items()}


if __name__ == "__main__":
    tune(config=get_config())
