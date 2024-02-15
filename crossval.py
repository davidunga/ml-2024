from sklearn.experimental import enable_halving_search_cv  # required
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
import paths
from data import load_data, make_data_init_pipe, make_feature_prep_pipe
from config import get_config, get_config_id
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


tree_grid_base = {'n_estimators': [50, 100], 'max_depth': [3, 4, 6]}

GRID = [
    # list of (model_type, params_dict)
    (RandomForestClassifier, tree_grid_base),
    (ExtraTreesClassifier, tree_grid_base),
    (LogisticRegression, [
        {'penalty': ['l1'], 'solver': ['liblinear']},
        {'penalty': ['l2'], 'solver': ['newton-cg']}])
]

SMALL_GRID = [
    # (for debugging)
    (RandomForestClassifier, {'n_estimators': [5], 'max_depth': [5, 10]}),
    (LogisticRegression, [
        {'penalty': ['l1'], 'solver': ['liblinear']},
        {'penalty': ['l2'], 'solver': ['newton-cg']}])
]


def make_cv_name(config: dict, model_name: str, seed: int):
    return f"cfg{get_config_id(config)} {model_name} seed{seed}"


def report_cv_results(results_df: pd.DataFrame):
    score_cols = [col for col in results_df.columns if col.startswith('mean_test_')]
    report_cols = [col for col in results_df.columns if col.startswith(('mean_test', 'mean_train', 'params'))]
    for col in score_cols:
        i = results_df[col].argmax()
        print(f"Results for max {col}:")
        print(results_df.loc[i, report_cols].to_string())


def save_cv_results(results_df: pd.DataFrame):
    model_name = results_df.loc[0, 'model_name']
    seed = results_df.loc[0, 'seed']
    results_csv = paths.CV_RESULTS_PATH / (make_cv_name(config, model_name, seed) + ".csv")
    results_csv.parent.mkdir(exist_ok=True, parents=True)
    results_df.to_csv(str(results_csv))


def run_crossval(config: dict, grid: list):

    # we use two pipelines:
    # data_init_pipe is used for data initialization; it removes some of the rows, processes the target
    #   variable, and separates the features from the target.
    # feature_prep_pipe is a 'regular' preprocessing pipe: it operates only on X (features)
    #   and is not allowed to remove rows or operate on the target variable

    data_init_pipe = make_data_init_pipe(config)
    feature_prep_pipe = make_feature_prep_pipe(config)

    raw_data = load_data(config)
    raw_features, target = data_init_pipe.fit_transform(raw_data)
    features = feature_prep_pipe.fit_transform(raw_features)

    cv_args = {'cv': config['cv.n_folds'], 'scoring': config['cv.scores'],
               'verbose': 0, 'refit': False, 'return_train_score': True}

    for model_type, params in grid:
        model_name = str(model_type).split('.')[-1][:-2]
        print(f"Grid searching {model_name}")
        for seed in range(config['cv.n_seeds']):
            model = model_type(random_state=seed)
            grid_search = GridSearchCV(model, params, **cv_args)
            grid_search.fit(features, target)
            results_df = pd.DataFrame({"model_name": model_name, "seed": seed, **grid_search.cv_results_})
            report_cv_results(results_df)
            save_cv_results(results_df)


if __name__ == "__main__":
    config = get_config()
    run_crossval(config=config, grid=SMALL_GRID)
