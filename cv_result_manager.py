import pandas as pd
import paths
from typing import Dict, Tuple, List
from config import get_config_id


def display(results_df: pd.DataFrame):
    score_cols = [col for col in results_df.columns if col.startswith('mean_test_')]
    report_cols = [col for col in results_df.columns if col.startswith(('mean_test', 'mean_train', 'params'))]
    for col in score_cols:
        i = results_df[col].argmax()
        print(f"Results for max {col}:")
        print(results_df.loc[i, report_cols].to_string())


def save(results_df: pd.DataFrame):
    config = results_df.attrs['config']
    model_name = results_df.loc[0, 'model_name']
    seed = results_df.loc[0, 'random_state']
    results_csv = paths.CV_RESULTS_PATH / (make_name(config, model_name, seed) + ".csv")
    results_csv.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(str(results_csv))


def make_dataframe(config: Dict, model_name: str, random_state: int, cv_results: Dict) -> pd.DataFrame:
    results_df = pd.DataFrame({"model_name": model_name, "random_state": random_state, **cv_results})
    results_df.attrs['config'] = config
    return results_df


def make_name(config: Dict, model_name: str, seed: int):
    return f"cfg{get_config_id(config)} {model_name} seed{seed}"

