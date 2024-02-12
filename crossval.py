from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from data import load_data, make_data_transformer
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

    featuers_df, target_df = load_data(config)
    data_transformer = make_data_transformer(config, featuers_df)

    for model_type, params in grid:
        for seed in range(config['cv.n_seeds']):
            model = model_type(random_state=seed)
            grid_search = GridSearchCV(model, params, cv=config['cv.n_folds'], scoring=config['cv.score'], verbose=2)
            pipe = Pipeline(steps=[("data_transformer", data_transformer), ("grid_search", grid_search)])
            pipe.fit(featuers_df, target_df)


if __name__ == "__main__":
    config = get_config()
    run_crossval(config=config, grid=GRID)
