import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from data_transformers import (
    ColumnTypeSetter, FeatureRemoverByBias, FeatureRemoverByName,
    RowRemoverByFeatureValue, CategoryReducer, TargetSeparator, ICDConverter,
    OneHotConverter, CategoryGroupOthers, Balancer, Standardizer,
    RowRemoverByDuplicates, AddFeatureAverageAge, AddFeatureByNormalizing,
    AddFeatureBySumming, AddFeatureEncounter, AddFeatureByCounting
)


def build_data_prep_pipe(config: Dict) -> Pipeline:

    steps = []

    # ----
    # Add/remove FEATURES:

    steps.append(AddFeatureByNormalizing(config['data.add_features.by_normalize']))
    steps.append(AddFeatureBySumming(config['data.add_features.by_sum']))

    if config['data.add_features.construct']['average_age']:
        steps.append(AddFeatureAverageAge(age_group_col='age'))
    if config['data.add_features.construct']['encounter']:
        steps.append(AddFeatureEncounter())

    features_to_remove = config['data.exclude_features.by_name']

    for kws in config['data.add_features.by_count']:
        steps.append(AddFeatureByCounting(mapping=kws['mapping'],
                                          values_to_count=kws['values_to_count'],
                                          invert=kws['invert']))
        if kws['drop_originals']:
            features_to_remove += [v for vs in kws['mapping'].values() for v in vs]

    # ----
    # Reduce/group CATEGORIES:

    for col, lookup in config['data.categories.reduce'].items():
        steps.append(CategoryReducer(feature=col, lookup=lookup))

    steps.append(CategoryGroupOthers(config['data.categories.group_others']))

    steps.append(ICDConverter(features=config['diagnosis_cols']))

    # ----
    # Remove ROWS:

    if config['data.exclude_rows.pregnancy_diabetes']:
        steps.append(RowRemoverByFeatureValue(feature=config['diagnosis_cols'],
                                              exclude_vals=[ICDConverter.PREGNANCY_DIABETES_ICD]))

    for col, exclude_vals in config['data.exclude_rows.where'].items():
        steps.append(RowRemoverByFeatureValue(feature=col, exclude_vals=exclude_vals))

    steps.append(RowRemoverByDuplicates(config['data.exclude_rows.duplicate']))

    # ----
    # Finalize:

    steps.append(FeatureRemoverByName(features_to_remove=features_to_remove))
    steps.append(ColumnTypeSetter())
    steps.append(TargetSeparator(target_col=config['target_col']))

    pipe = Pipeline(steps=[(step.name, step) for step in steps])
    print("Data prep pipe:")
    describe_pipe(pipe)

    return pipe


def build_cv_pipe(config: Dict, Xy):
    onehot_converter = OneHotConverter().fit(*Xy)
    steps = []
    steps.append(Standardizer(**config['data.standardize']))
    balancer = Balancer(method=config['balance.method'], params=config['balance.params'])
    if balancer.is_categorical:
        steps += [balancer, onehot_converter]
    else:
        steps += [onehot_converter, balancer]

    pipe = Pipeline(steps=[(step.name, step) for step in steps])
    print("CV pipe:")
    describe_pipe(pipe)

    return pipe


def describe_pipe(pipe: Pipeline):
    max_row_len = 80
    lines = [""]
    for i, step in enumerate(pipe):
        lines[-1] += f"{' ->' if i else ''} ({i}) {step.name}"
        if len(lines[-1]) >= max_row_len:
            lines.append("")
    print("\n".join(lines))



def verbolize_pipeline(pipe):
    from functools import wraps
    """
    Modifies an sklearn pipeline to print a message to the terminal whenever
    fit, transform, or fit_transform methods are called. The message includes
    the step's name, the input size, and the function that was called.
    """

    # Function to create a wrapper around the methods
    def create_wrapper(original_func):
        @wraps(original_func)
        def wrapper(*args, **kwargs):
            def _typ(a) -> str:
                if isinstance(a, list):
                    return "[" + ", ".join([_typ(aa) for aa in a]) + "]"
                if isinstance(a, tuple):
                    return "(" + ",".join([_typ(aa) for aa in a]) + ")"
                elif a is None:
                    return "None"
                else:
                    return str(type(a)).split(".")[-1].split("'")[0]
            print(original_func.__self__.__class__, original_func.__name__, _typ(list(args)))

            return original_func(*args, **kwargs)

        return wrapper

    for name, estimator in pipe.steps:
        for method_name in ['fit', 'transform', 'fit_transform']:
            if hasattr(estimator, method_name):
                original_method = getattr(estimator, method_name)
                setattr(estimator, method_name, create_wrapper(original_method))

    return pipe


def load_data(config: Dict) -> pd.DataFrame:
    data_file = str(DATA_PATH / "diabetic_data.csv")
    df = pd.read_csv(data_file, na_values="?", dtype={'payer_code': str})
    return df


def stratified_split(Xy, test_size, random_state: int):
    X_train, X_test, y_train, y_test = train_test_split(
        *Xy, test_size=test_size, shuffle=True, random_state=random_state, stratify=Xy[1])
    return (X_train, y_train), (X_test, y_test)
