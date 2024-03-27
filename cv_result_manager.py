import pandas as pd
from pathlib import Path
import paths
from typing import Dict, Tuple, List
from config import get_config_name
import json


def display(results_df: pd.DataFrame):
    score_cols = [col for col in results_df.columns if col.startswith('mean_test_')]
    report_cols = [col for col in results_df.columns if col.startswith(('mean_test', 'mean_train', 'params'))]
    for col in score_cols:
        i = results_df[col].argmax()
        print(f"Results for max {col}:")
        print(results_df.loc[i, report_cols].to_string())


def save(results_df: pd.DataFrame):
    config = results_df.attrs['config']
    results_csv = get_csv_path(config)
    results_csv.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(str(results_csv))


def make_dataframe(config: Dict, cv_results: Dict) -> pd.DataFrame:
    results_df = pd.DataFrame({"model_name": config['estimator.name'], **cv_results, "config": json.dumps(config)})
    results_df.attrs['config'] = config
    return results_df


def get_csv_path(config) -> Path:
    return paths.CV_RESULTS_PATH / (get_config_name(config) + ".csv")

