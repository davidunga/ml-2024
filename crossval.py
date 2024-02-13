from sklearn.experimental import enable_halving_search_cv  # required
from sklearn.model_selection import GridSearchCV, HalvingGridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from data import load_data, make_data_init_pipe, make_feature_prep_pipe
from config import get_config

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

    for model_type, params in grid:
        model_name = str(model_type).split('.')[-1][:-2]
        print(f"Grid searching {model_name}")
        for seed in range(config['cv.n_seeds']):
            model = model_type(random_state=seed)
            grid_search = GridSearchCV(model, params, cv=config['cv.n_folds'], scoring=config['cv.score'], verbose=0)
            grid_search.fit(features, target)
            print(f" Seed {seed} best score {grid_search.best_score_:2.2f}, params: {grid_search.best_params_}")



if __name__ == "__main__":
    config = get_config()
    run_crossval(config=config, grid=SMALL_GRID)
