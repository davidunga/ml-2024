import json
import numpy as np
from typing import  Dict

import paths
from data import load_data, build_data_prep_pipe, build_cv_pipe, stratified_split
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, roc_curve
import matplotlib.pyplot as plt
import pickle
import pandas as pd


def calc_scores(estimator: BaseEstimator, X: np.ndarray, y_true: np.ndarray):
    y_score = estimator.predict_proba(X)[:, 1]
    y_pred = estimator.predict(X)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    scores = {
        'auc': roc_auc_score(y_true=y_true, y_score=y_score),
        'acc': balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
        'f1': f1_score(y_true=y_true, y_pred=y_pred),
        'fpr': fpr,
        'tpr': tpr,
    }
    return scores


def plot_roc_curve(scores: Dict):
    plt.plot([0, 1], [0, 1], 'k--')
    for name, scores in scores.items():
        s = ', '.join([f'{k}={v:2.4f}' for k, v in scores.items() if k not in ('fpr', 'tpr')])
        plt.plot(scores['fpr'], scores['tpr'], '-', label=f"{name.upper()} {s}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.gca().set_aspect('equal')
    plt.legend(loc="lower right")


def evaluate_from_data(estimator, X_train, y_train, X_test, y_test, draw: bool = True):
    scores = {
        'train': calc_scores(estimator, X_train, y_train),
        'test': calc_scores(estimator, X_test, y_test),
    }
    if draw:
        plt.figure(figsize=(6, 6))
        plt.tight_layout()
        plot_roc_curve(scores)
        plt.title('ROC & Scores')
        plt.show()
    return scores


def evaluate_from_config(estimator: BaseEstimator, config: Dict = None, draw: bool = True):
    raw_data = load_data(config)
    prep_pipe = build_data_prep_pipe(config)
    Xy = prep_pipe.fit_transform(raw_data)
    cv_pipe = build_cv_pipe(config, Xy)
    Xy_train, Xy_test = stratified_split(Xy, test_size=config['data.test_size'], seed=config['random_state'])
    Xy_train = cv_pipe.fit_transform(Xy_train)
    Xy_test = cv_pipe.transform(Xy_test)
    return evaluate_from_data(estimator, *Xy_train, *Xy_test, draw = draw)


if __name__ == "__main__":
    def _get_pkl(s):
        if s.endswith('.pkl'):
            return s
        return str(paths.CV_RESULTS_PATH / (s + '.best.pkl'))

    best_pkl = _get_pkl('XGB OptimSearchCV NoBalance TUNED auc s7 87ac')
    df = pd.read_csv(best_pkl.replace('best.pkl', 'csv'))
    config = json.loads(df.iloc[0]['config'])
    items = pickle.load(open(best_pkl, 'rb'))
    estimator = items['best']['estimator']

    evaluate_from_config(estimator, config)

