import numpy as np
import pandas as pd
from pathlib import Path
import paths
from typing import Dict, Tuple, List
from config import get_config_name
import json
from sklearn.model_selection._search import BaseSearchCV
import pickle
from copy import deepcopy
from glob import glob


def process_result(config: Dict, cv: BaseSearchCV):
    results = make_result_dict(config, cv)
    display(results)
    save(results)
    refresh_bests_file()


def make_result_dict(config: Dict, cv: BaseSearchCV) -> Dict:
    estimator_name = cv.estimator.__class__.__name__
    df = pd.DataFrame({"estimator_name": estimator_name, **cv.cv_results_, "config": json.dumps(config)})
    res = {
        'df': df,
        'config': config,
        'estimator_name': estimator_name,
        'best_estimator': cv.best_estimator_,
        'best_params': cv.best_params_,
        'best_score': cv.best_score_
    }
    return res


def display(result_dict: Dict):
    results_df = result_dict['df']
    score_cols = [col for col in results_df.columns if col.startswith('mean_test_')]
    report_cols = [col for col in results_df.columns if col.startswith(('mean_test', 'mean_train', 'params'))]
    for col in score_cols:
        i = results_df[col].argmax()
        print(f"Results for max {col}:")
        print(results_df.loc[i, report_cols].to_string())
    print("Best score:", result_dict['best_score'])


def save(result_dict: Dict):
    config_name = get_config_name(result_dict['config'])
    fname = f"{result_dict['estimator_name']} {config_name}"

    results_csv = paths.CV_RESULTS_PATH / (fname + ".csv")
    results_csv.parent.mkdir(exist_ok=True, parents=True)
    result_dict['df'].to_csv(str(results_csv))

    best_model_pkl = paths.BEST_MODELS_PATH / (fname + ".pkl")
    best_model_pkl.parent.mkdir(exist_ok=True, parents=True)
    with best_model_pkl.open('wb') as f:
        pickle.dump({k: v for k, v in result_dict.items() if k != 'df'}, f)


def refresh_bests_file():
    best_txt = paths.BEST_MODELS_PATH / "best.txt"
    files = glob(str(paths.BEST_MODELS_PATH / "*.pkl"))
    lines = []
    scores = []
    for file in files:
        with open(file, 'rb') as f:
            res = pickle.load(f)
        score = res['best_score']
        lines.append(f"{Path(file).stem} - score={score:2.4f}, params={res['best_params']}")
        scores.append(score)
    lines = [lines[i] for i in np.argsort(scores)[::-1]]
    with best_txt.open('w') as f:
        f.writelines("\n".join(lines))
