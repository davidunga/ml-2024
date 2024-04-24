import pandas as pd
import numpy as np
import paths
from config import get_section, get_hash, get_base_config
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from data_transformers import (
    RowRemoverByFeatureValue, CategoryReducer, TargetSeparator, ICDConverter,
    OneHotConverter, CategoryGroupOthers, Balancer, Standardizer,
    RowRemoverByDuplicates, AddFeatureAverageAge, AddFeatureByNormalizing,
    AddFeatureBySumming, AddFeatureEncounter, AddFeatureByCounting, PropertySetter, DataTransformer
)


def build_data_prep_pipe(config: Dict) -> Pipeline:
    """ Builds data preparation pipe, to be applied to the full dataset. """

    # make sure pipe is fully defined by data section of config:
    data_config = get_section(config, 'data')

    prop_setter = PropertySetter(default='cat').register(
        num=data_config['numeric_cols'],
        drop=data_config['exclude_features.by_name'])

    steps = []

    def add_step(step: DataTransformer):
        step.prop_setter = prop_setter
        steps.append(step)

    # ----
    # Remove ROWS:

    if data_config['exclude_rows.pregnancy_diabetes']:
        add_step(RowRemoverByFeatureValue(feature=data_config['diagnosis_cols'],
                                          exclude_vals=[ICDConverter.PREGNANCY_DIABETES_ICD]))

    for col, exclude_vals in data_config['exclude_rows.where'].items():
        add_step(RowRemoverByFeatureValue(feature=col, exclude_vals=exclude_vals))

    add_step(RowRemoverByDuplicates(data_config['exclude_rows.duplicate']))

    # ----
    # Add FEATURES:

    add_step(AddFeatureByNormalizing(data_config['add_features.by_normalize']))
    add_step(AddFeatureBySumming(data_config['add_features.by_sum']))
    for kws in data_config['add_features.by_count']:
        add_step(AddFeatureByCounting(**kws))

    add_step(AddFeatureBySumming(data_config['add_features.by_nnz_sum'], method='nnz'))

    if data_config['add_features.construct']['average_age']:
        add_step(AddFeatureAverageAge(age_group_col='age'))
    if data_config['add_features.construct']['encounter']:
        add_step(AddFeatureEncounter())

    # ----
    # Regroup CATEGORIES:

    for col, lookup in data_config['categories.reduce'].items():
        add_step(CategoryReducer(feature=col, lookup=lookup))

    add_step(CategoryGroupOthers(data_config['categories.group_others']))

    add_step(ICDConverter(features=data_config['diagnosis_cols']))

    # ----
    # Finalize:

    steps.append(prop_setter)
    steps.append(TargetSeparator(target_col=data_config['target_col'],
                                 sanity_mode=data_config['sanity_mode']))

    pipe = Pipeline(steps=[(step.name, step) for step in steps])
    print("Data prep pipe:")
    describe_pipe(pipe)

    return pipe


def build_cv_pipe(config: Dict, full_Xy) -> Pipeline:
    """ Builds pipe that goes into the cross-validator, after prep pipe was applied  """

    onehot_converter = OneHotConverter().fit(*full_Xy)  # fit over full dataset
    onehot_converter.frozen = True  # prevent re-fitting during train

    standardizer = Standardizer(**config['data.standardize'])
    balancer = Balancer(method=config['balance.method'], params=config['balance.params'])

    if balancer.is_dataframe_in:
        steps = [standardizer, balancer, onehot_converter]
    else:
        steps = [standardizer, onehot_converter, balancer]

    pipe = Pipeline(steps=[(step.name, step) for step in steps])
    print("CV pipe:")
    describe_pipe(pipe)

    return pipe


def make_prepped_csv(config: Dict):
    """
    Creates two csv files:
        ... NonStandardized.csv = Data after passing through prep pipe.
        ... Standardized.csv = Data after passing through prep pipe & standardizer.
    """

    folder = paths.DATA_PATH / "prepped"
    folder.mkdir(parents=True, exist_ok=True)
    print("Making prepped data csvs")

    csv_name = f"prepped_data {get_hash(get_section(config, 'data'))[:6]}"

    raw_data = load_data(config)
    prep_pipe = build_data_prep_pipe(config)
    Xy = prep_pipe.fit_transform(raw_data)

    df = pd.concat(Xy, axis=1)
    df.to_csv(folder / (csv_name + " NonStandardized.csv"))

    standardizer = Standardizer(**config['data.standardize'])
    Xy = standardizer.fit_transform(Xy)

    df = pd.concat(Xy, axis=1)
    df.to_csv(folder / (csv_name + " Standardized.csv"))
    print("Done. Saved to: " + str(folder))


def describe_pipe(pipe: Pipeline):
    """ print pipeline steps """
    max_row_len = 80
    lines = [""]
    for i, step in enumerate(pipe):
        lines[-1] += f"{' ->' if i else ''} ({i}) {step.name}"
        if len(lines[-1]) >= max_row_len:
            lines.append("")
    print("\n".join(lines))


def load_data(config: Dict) -> pd.DataFrame:
    data_file = str(DATA_PATH / "diabetic_data.csv")
    df = pd.read_csv(data_file, na_values="?", dtype={'payer_code': str})
    return df


def stratified_split(Xy, test_size, seed: int):
    X_train, X_test, y_train, y_test = train_test_split(
        *Xy, test_size=test_size, shuffle=True, random_state=seed, stratify=Xy[1])
    return (X_train, y_train), (X_test, y_test)


if __name__ == "__main__":
    make_prepped_csv(get_base_config())
