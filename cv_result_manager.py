import pandas as pd
from pathlib import Path
import paths
from typing import Dict, Tuple, List
from config import get_config_name
import json
from sklearn.model_selection._search import BaseSearchCV
import pickle


def make_result_dict(config: Dict, cv: BaseSearchCV) -> Dict:
    df = pd.DataFrame({"model_name": config['estimator.name'],
                       **cv.cv_results_, "config": json.dumps(config)})
    res = {
        'df': df,
        'config': config,
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

    results_csv = paths.CV_RESULTS_PATH / (config_name + ".csv")
    results_csv.parent.mkdir(exist_ok=True, parents=True)
    result_dict['df'].to_csv(str(results_csv))

    best_model_pkl = paths.BEST_MODELS_PATH / (config_name + ".pkl")
    best_model_pkl.parent.mkdir(exist_ok=True, parents=True)
    with best_model_pkl.open('wb') as f:
        pickle.dump(result_dict['best_estimator'], f)
