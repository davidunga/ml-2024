from sklearn.experimental import enable_halving_search_cv  # required
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV, StratifiedKFold
import paths
from data import load_data, build_pipeline, DataSplitter
from config import get_config, get_config_id
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.model_selection import train_test_split
import numpy as np


def _reduce_grid(grid: Dict) -> Dict:
    return {k: [v[0], v[-1]] if len(v) > 1 else v for k, v in grid.items()}


def tune_xgboost(config: Dict):

    base_grid = {
        'max_depth': np.arange(3, 5),
        'learning_rate': np.linspace(0.01, 0.1, 5)
    }

    finetune_grid = {
        'min_child_weight': [1, 3, 5, 7, 10],
        'gamma': [0, .1, .2, .3, .4],
        'subsample': [.6, .8, 1.],
        'colsample_bytree': [.6, .8, 1.],
        'reg_alpha': [1e-5, 1e-3, 1e-2, 1, 1e2],
        'reg_lambda': [1e-5, 1e-3, 1e-2, 1, 1e2],
    }

    cv_args = {'cv': config['cv.n_folds'], 'scoring': config['cv.scores'],
               'verbose': 0, 'refit': config['cv.main_score'], 'return_train_score': True, 'n_jobs': -1}

    random_state = config['random_state']

    pipe = build_pipeline(config, verbose=2)
    raw_data = load_data(config)
    Xy = pipe.fit_transform(raw_data)

    data_splitter = DataSplitter(random_state=random_state, shuffle=True, stratify=True)
    Xy_train, Xy_test = data_splitter.split(Xy, test_size=config['data.test_size'])
    Xy_base_train, Xy_base_val = data_splitter.split(Xy_train, test_size=config['cv.base.val_size'])

    # find best base params:
    print("Searching for best base params...")

    model = xgb.XGBClassifier(early_stopping_rounds=config['cv.base.early_stopping_rounds'], random_state=random_state)
    cv = GridSearchCV(model, param_grid=base_grid, **cv_args)
    cv.fit(*Xy_base_train, eval_set=[Xy_base_val])
    params = cv.best_params_
    params['n_estimators'] = cv.best_estimator_.best_iteration

    print("Best base params:", params)
    print("Score:", cv.best_score_)

    # fine tune:
    print("Fine tuning...")
    model = xgb.XGBClassifier(**params, random_state=random_state)
    cv = GridSearchCV(model, param_grid=finetune_grid, **cv_args)
    cv.fit(*Xy_train)

    print("Best fine-tuned params:", params)
    print("Score:", cv.best_score_)


if __name__ == "__main__":
    tune_xgboost(config=get_config())
