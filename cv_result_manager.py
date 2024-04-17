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


def process_result(config: Dict, cv: BaseSearchCV, save: bool = True):
    results = make_result_dict(config, cv)
    display(results)
    if save:
        save_result(results)
        refresh_bests_file()


def make_result_dict(config: Dict, cv: BaseSearchCV) -> Dict:
    name = get_config_name(config)
    df = pd.DataFrame({'name': name, **cv.cv_results_, 'config': json.dumps(config)})
    res = {'name': name,
           'df': df,
           'best': {'estimator': cv.best_estimator_, 'params': cv.best_params_, 'score': cv.best_score_}}
    return res


def display(result_dict: Dict):
    df, score_names, main_score = make_report_df(result_dict['df'])
    print("Results for : " + result_dict['name'])
    for score_name in score_names:
        print(f" best by {score_name}:")
        i = df[score_name].argmax()
        print("   ", df.iloc[i]['text'])
        print("   ", df.iloc[i]['params'])


def save_result(result_dict: Dict):
    paths.CV_RESULTS_PATH.mkdir(exist_ok=True, parents=True)

    results_csv = paths.CV_RESULTS_PATH / (result_dict['name'] + '.csv')
    result_dict['df'].to_csv(str(results_csv))

    best_pkl = paths.CV_RESULTS_PATH / (result_dict['name'] + '.best.pkl')
    with best_pkl.open('wb') as f:
        pickle.dump({k: v for k, v in result_dict.items() if k != 'df'}, f)


def load_results(csvs: str | List[str] = None) -> pd.DataFrame:

    if not csvs:
        csvs = glob(str(paths.CV_RESULTS_PATH / "*.csv"))
    elif isinstance(csvs, str):
        csvs = [csvs]

    def _safe_load(csv):
        df_ = pd.read_csv(csv)
        name, config_str = df_.iloc[0][['name', 'config']]
        config = json.loads(config_str)
        assert name == get_config_name(config)
        assert np.all(df_['name'] == name)
        assert np.all(df_['config'] == config_str)
        return df_

    df = pd.concat((_safe_load(csv) for csv in csvs), ignore_index=True)
    df.reset_index(inplace=True)

    return df


def make_report_df(df: pd.DataFrame):

    df = df.copy()
    config = json.loads(df.iloc[0]['config'])
    main_score = config['cv.main_score']

    score_names = [col.replace('mean_test_', '') for col in df.columns if col.startswith('mean_test_')]
    score_names = [main_score] + [s for s in score_names if s != main_score]
    df.rename(columns={f'mean_test_{s}': s for s in score_names}, inplace=True)

    for score_name in score_names:
        df[f'{score_name}.rank'] = df[score_name].rank(method='dense', ascending=False).astype(int)

    keep_cols = ['name'] + [col for col in df.columns if col.split('.')[0] in score_names] + ['config', 'params']
    df = df[keep_cols]

    text = []
    for i in range(len(df)):
        row = df.iloc[i]
        txt = [f"{score_name}:{row[score_name]:2.3f} [{row[f'{score_name}.rank']:4d}]" for score_name in score_names]
        text.append(", ".join(txt))
    df.loc[:, ['text']] = text

    return df, score_names, main_score


def refresh_bests_file():

    top_k = 5
    best_txt = paths.CV_RESULTS_PATH / "best.txt"
    df = load_results()
    df, score_names, main_score = make_report_df(df)

    text = []
    for i in range(len(df)):
        text.append(f"{df.iloc[i]['name']:30s} -- " + df.iloc[i]['text'] + " " + df.iloc[i]['params'])
    df.loc[:, ['text']] = text

    lines = []
    for score_name in score_names:
        lines += ["Top ranked by " + score_name + ":"]
        lines += df.sort_values(by=score_name, ascending=False).iloc[:top_k]['text'].to_list()

    lines.append(f"Full results, ranked by {main_score}:")
    lines += df.sort_values(by=main_score, ascending=False)['text'].to_list()

    with best_txt.open('w') as f:
        f.writelines("\n".join(lines))


#refresh_bests_file()
