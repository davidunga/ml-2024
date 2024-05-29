import json
import os
import numpy as np
from typing import Dict
import seaborn as sns
import paths
from data import load_data, build_data_prep_pipe, build_cv_pipe, stratified_split
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, roc_curve
import matplotlib.pyplot as plt
import pickle
import pandas as pd

SPLIT_COLORS = {'test': 'crimson', 'train': 'DodgerBlue'}
SPLITS = ['train', 'test']
SCORE_NAMES = {'auc': 'AUC', 'acc': 'Balanced Accuracy', 'f1': 'F1 Score'}


def calc_scores(estimator: BaseEstimator, X: np.ndarray, y_true: np.ndarray):
    y_score = estimator.predict_proba(X)[:, 1]
    y_pred = estimator.predict(X)
    fpr, tpr, ths = roc_curve(y_true, y_score)
    scores = {
        'auc': roc_auc_score(y_true=y_true, y_score=y_score),
        'acc': balanced_accuracy_score(y_true=y_true, y_pred=y_pred),
        'f1': f1_score(y_true=y_true, y_pred=y_pred),
        'fpr': fpr,
        'tpr': tpr,
        'ths': ths
    }
    return scores


def plot_roc_curve(scores: Dict):
    plt.plot([0, 1], [0, 1], 'k--')
    for name, scores in scores.items():
        s = ', '.join([f'{k}={v:2.4f}' for k, v in scores.items() if k not in ('fpr', 'tpr', 'ths')])
        plt.plot(scores['fpr'], scores['tpr'], '-', label=f"{name.upper()} {s}")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.gca().set_aspect('equal')
    plt.legend(loc="lower right")


def calc_importance_scores(estimator, reverse_feature_names):
    importance_scores = {}
    for type_ in ['weight', 'gain']:
        importance_scores[type_] = {feature_name: [] for feature_name in reverse_feature_names}
        for fnum, value in estimator.get_booster().get_score(importance_type=type_).items():
            feature_name = reverse_feature_names[int(fnum[1:])]
            importance_scores[type_][feature_name].append(value)
    return importance_scores


def evaluate_from_data(estimator, X_train, y_train, X_test, y_test, refit: bool = False, draw: bool = True,
                       reverse_feature_names = None):
    if refit:
        estimator.fit(X_train, y_train)
    scores = {
        'train': calc_scores(estimator, X_train, y_train),
        'test': calc_scores(estimator, X_test, y_test),
        '_estimator': estimator,
        'importance_scores': calc_importance_scores(estimator, reverse_feature_names)
    }
    if draw:
        plt.figure(figsize=(6, 6))
        plt.tight_layout()
        plot_roc_curve(scores)
        plt.title('ROC & Scores')
        plt.show()
    return scores


def load_Xys(config):
    raw_data = load_data(config)
    prep_pipe = build_data_prep_pipe(config)
    Xy = prep_pipe.fit_transform(raw_data)
    cv_pipe = build_cv_pipe(config, Xy)
    Xy_train, Xy_test = stratified_split(Xy, test_size=config['data.test_size'], seed=config['random_state'])
    Xy_train = cv_pipe.fit_transform(Xy_train)
    Xy_test = cv_pipe.transform(Xy_test)
    if 'features' in config:
        feature_names = cv_pipe[-2].reverse_feature_names
        feature_ixs = [ix for ix, feature_name in enumerate(feature_names) if feature_name in config['features']]
        Xy_test = Xy_test[0][:, feature_ixs], Xy_test[1]
        Xy_train = Xy_train[0][:, feature_ixs], Xy_train[1]
    return Xy_train, Xy_test, cv_pipe[-2].reverse_feature_names


def evaluate_from_config(estimator: BaseEstimator, config: Dict = None, refit: bool = False, draw: bool = True):
    Xy_train, Xy_test, reverse_feature_names = load_Xys(config)
    return evaluate_from_data(estimator, *Xy_train, *Xy_test, refit=refit, draw=draw,
                              reverse_feature_names=reverse_feature_names)


def plot_multi_roc(scores_per_seed):

    plt.figure(figsize=(10, 8))
    plt.plot([0, 1], [0, 1], 'k:')
    common_fpr = np.linspace(0, 1, 100)

    for split in SPLITS:
        all_fpr = [metrics[split]['fpr'] for metrics in scores_per_seed]
        all_tpr = [metrics[split]['tpr'] for metrics in scores_per_seed]
        aucs = [metrics[split]['auc'] for metrics in scores_per_seed]

        tprs = np.array([np.interp(common_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)])
        tpr_mu = np.mean(tprs, axis=0)
        tpr_sd = np.std(tprs, axis=0)

        plt.plot(common_fpr, tpr_mu, label=f'{split.capitalize()} AUC: {np.mean(aucs):2.3f}±{np.std(aucs):2.3f}',
                 color=SPLIT_COLORS[split])
        plt.fill_between(common_fpr, tpr_mu - tpr_sd, tpr_mu + tpr_sd, color=SPLIT_COLORS[split], alpha=0.2)

    plt.grid()
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC - Mean & SD over {len(scores_per_seed)} seeds')
    plt.gca().set_aspect('equal')
    plt.legend(loc="lower right")


def plot_feature_importance(scores_per_seed, types_=('weight',)):
    for importance_type in types_:
        importances_ = [s['importance_scores'][importance_type] for s in scores_per_seed]
        data = []
        for feature in importances_[0]:
            for importance_ in importances_:
                data.append([feature, np.max(importance_[feature])])
        data = pd.DataFrame(data, columns=['Feature', 'Importance'])
        sorted_features = data.groupby('Feature')['Importance'].mean().sort_values().index[::-1]
        plt.figure(figsize=(12, 8))
        sns.barplot(data=data, x='Importance', y='Feature', estimator=np.mean, color='dodgerBlue',
                    err_kws={'linewidth': 1}, errorbar=('sd'), order=sorted_features)
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        type_str = f' [{importance_type}]' if len(types_) > 1 else ''
        plt.title(f'Feature Importance{type_str} - Mean & SD over {len(scores_per_seed)} seeds')
        plt.gca().yaxis.set_tick_params(labelsize=10)
        plt.tight_layout(pad=3)


def plot_score_distributions(scores_per_seed):
    score_names = [k for k in scores_per_seed[0]['train'] if k not in ('tpr', 'fpr', 'ths')]
    _, axs = plt.subplots(nrows=2, ncols=len(score_names))
    for i, split in enumerate(SPLITS):
        color = SPLIT_COLORS[split]
        for j, score in enumerate(score_names):
            v = [s[split][score] for s in scores_per_seed]
            data = pd.DataFrame({score: v, 'split': split})
            sns.histplot(data=data, x=score, ax=axs[i, j], bins=8, color=color, edgecolor=color)
            textstr = f'{split.capitalize()} {SCORE_NAMES[score]}\nMean: {np.mean(v):.3f}, SD: {np.std(v):.3f}'
            props = dict(boxstyle='round', facecolor='w', alpha=0.25)
            axs[i, j].text(0.05, 0.95, textstr, transform=axs[i, j].transAxes, fontsize=10,
                           verticalalignment='top', bbox=props)
            axs[i, j].set_xlabel(None)
            axs[i, j].set_ylabel(None)
    for i, split in enumerate(SPLITS):
        axs[i, 0].set_ylabel(split.capitalize())
    for j, score in enumerate(score_names):
        axs[0, j].set_title(SCORE_NAMES[score])
    plt.suptitle(f'Score distributions over {len(scores_per_seed)} seeds')


def get_scores_per_seed(best_pkl: str, n_seeds: int, estimator_modifiers: Dict, config_modifiers: Dict, force: bool):
    from hashlib import md5
    best_pkl = str(best_pkl)
    modifiers_hash = md5(json.dumps({'n_seeds': n_seeds,
                                     **estimator_modifiers, **config_modifiers}).encode()).hexdigest()[:5]
    scores_pkl = best_pkl.replace('best.pkl', f'SCORES-{modifiers_hash}.pkl')
    if force or not os.path.isfile(scores_pkl):
        scores_per_seed = []
        for seed in range(n_seeds):
            df = pd.read_csv(best_pkl.replace('best.pkl', 'csv'))
            config = json.loads(df.iloc[0]['config'])
            assert set(config_modifiers).issubset(list(config) + ['features'])
            for k, v in config_modifiers.items():
                config[k] = v
            config['random_state'] = seed
            estimator = pickle.load(open(best_pkl, 'rb'))['best']['estimator']
            for attr, val in estimator_modifiers.items():
                setattr(estimator, attr, val)
            scores_per_seed.append(evaluate_from_config(estimator, config, refit=True, draw=False))
        with open(scores_pkl, 'wb') as f:
            pickle.dump(scores_per_seed, f)
    with open(scores_pkl, 'rb') as f:
        scores_per_seed = pickle.load(f)
    return scores_per_seed


if __name__ == "__main__":
    best_pkl = paths.CV_RESULTS_PATH / 'XGB OptimSearchCV NoBalance TUNED auc s7 87ac.best.pkl'
    scores_per_seed = get_scores_per_seed(best_pkl=best_pkl, n_seeds=50, force=False,
                                          estimator_modifiers={'scale_pos_weight': 10}, config_modifiers={})
    plot_multi_roc(scores_per_seed)
    plot_feature_importance(scores_per_seed)
    plot_score_distributions(scores_per_seed)
    plt.show()
