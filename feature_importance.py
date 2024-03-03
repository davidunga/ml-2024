import pandas as pd
import numpy as np
from data import load_data, make_data_init_pipe, make_feature_prep_pipe
from config import get_config, get_config_id
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns


def get_feature_importance(model,
                           reverse_feature_names: list[str],
                           reduce: str = 'sum',
                           normalize: bool = True) -> pd.DataFrame:
    try:
        raw_importance = model.feature_importances_
    except AttributeError:
        raw_importance = model.coef_
    importance = np.abs(raw_importance.squeeze())
    importance_df = pd.DataFrame({'feature': reverse_feature_names, 'importance': importance})
    if reduce == 'sum':
        importance_df = importance_df.groupby('feature').sum()
    else:
        assert reduce is None
    if normalize:
        importance_df /= importance_df.max()
    return importance_df


def draw_feature_importance(importance_df: pd.DataFrame):
    sns.barplot(data=importance_df, y='importance', x='feature')
    plt.tight_layout()
    plt.tick_params(axis='x', labelrotation=75, labelsize=8)


def analyze_feature_importance(config: dict, model = None, normalize: bool = True):

    if model is None:
        model = LogisticRegression(penalty='l2')

    data_init_pipe = make_data_init_pipe(config)
    feature_prep_pipe = make_feature_prep_pipe(config)

    raw_data = load_data(config)
    raw_features, target = data_init_pipe.fit_transform(raw_data)
    features = feature_prep_pipe.fit_transform(raw_features)

    model.fit(features, target)

    importance_df = get_feature_importance(
        model, feature_prep_pipe[-1].reverse_feature_names, normalize=normalize)

    draw_feature_importance(importance_df)
    plt.title("Feature Importance by " + str(model))
    plt.show()


if __name__ == "__main__":
    config = get_config()
    analyze_feature_importance(config)
