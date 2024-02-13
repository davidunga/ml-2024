from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from data import load_data, make_data_init_pipe, make_feature_prep_pipe
from config import get_config

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression


tree_grid_base = {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8], 'max_depth': [3, 4, 6]}

GRID = [
    # list of (model_type, params_dict)
    (RandomForestClassifier, tree_grid_base),
    (ExtraTreesClassifier, tree_grid_base),
    (LogisticRegression, {'penalty': ['l1', 'l2', 'elasticnet'], 'class_weight': ['balanced', None]})
]


def run_crossval(config: dict, grid: list):

    # we use two pipelines:
    # data_init_pipe is used for data initialization; it removes some of the rows, processes the target
    #   variable, and separates the features from the target.
    # feature_prep_pipe is a 'regular' preprocessing pipe: it operates only on X (features)
    #   and is not allowed to remove rows or operate on the target variable

    df = load_data(config)
    data_init_pipe = make_data_init_pipe(config)
    feature_prep_pipe = make_feature_prep_pipe(config)

    features_df, target_df = data_init_pipe.fit_transform(df)

    for model_type, params in grid:
        for seed in range(config['cv.n_seeds']):
            model = model_type(random_state=seed)
            grid_search = GridSearchCV(model, params, cv=config['cv.n_folds'], scoring=config['cv.score'], verbose=2)
            pipe = Pipeline(steps=[("data_prep", feature_prep_pipe), ("grid_search", grid_search)])
            pipe.fit(features_df, target_df)


if __name__ == "__main__":
    config = get_config()
    run_crossval(config=config, grid=GRID)
