import pandas as pd
from typing import List, Tuple, Dict
from sklearn.model_selection import train_test_split
import numpy as np
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from data_transformers import \
    ColumnTypeSetter, FeatureRemoverByBias, FeatureRemoverByName, \
    RowRemoverByFeatureValue, CategoryReducer, XySplitter, ICDConverter, \
    OneHotConverter, CategoryGroupOthers, Balancer, Standardizer, RowRemoverByDuplicates, \
    AddFeatureAverageAge, AddFeatureByNormalizing, AddFeatureBySumming, AddFeatureEncounter, AddFeatureByCounting


TARGET_COL = 'readmitted'
NUMERIC_COLS = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses', 'weight']
DIAGNOSIS_COLS = ['diag_1', 'diag_2', 'diag_3']


def build_pipeline(config: Dict, verbose: int = 1) -> Pipeline:

    steps = []

    # ----
    # Add/remove FEATURES:

    for normalize_by, (to_normalize, suffix) in config['data.add_features.by_normalize'].items():
        steps.append(AddFeatureByNormalizing(to_normalize=to_normalize, normalize_by=normalize_by, suffix=suffix))

    steps.append(AddFeatureBySumming(config['data.add_features.by_sum']))

    if config['data.add_features.construct']['average_age']:
        steps.append(AddFeatureAverageAge(age_group_col='age'))
    if config['data.add_features.construct']['encounter']:
        steps.append(AddFeatureEncounter())

    features_to_remove = config['data.exclude_features.by_name']

    for kws in config['data.add_features.by_count']:
        for (new_feature, features) in kws['mapping'].items():
            steps.append(AddFeatureByCounting(features=features, new_feature=new_feature,
                                              values_to_count=kws['values_to_count'], invert=kws['invert']))
            if kws['drop_originals']:
                features_to_remove += features

    # ----
    # Reduce/group CATEGORIES:

    for col, lookup in config['data.categories.reduce'].items():
        steps.append(CategoryReducer(feature=col, lookup=lookup))

    steps.append(CategoryGroupOthers(config['data.categories.group_others']))

    steps.append(ICDConverter(features=DIAGNOSIS_COLS))

    # ----
    # Remove ROWS:

    if config['data.exclude_rows.pregnancy_diabetes']:
        steps.append(RowRemoverByFeatureValue(feature=DIAGNOSIS_COLS,
                                              exclude_vals=[ICDConverter.PREGNANCY_DIABETES_ICD]))

    for col, exclude_vals in config['data.exclude_rows.where'].items():
        steps.append(RowRemoverByFeatureValue(feature=col, exclude_vals=exclude_vals))

    steps.append(RowRemoverByDuplicates(config['data.exclude_rows.duplicate']))

    # ----
    # Finalize:

    steps.append(FeatureRemoverByName(features_to_remove=features_to_remove))

    # set column types to be either categorical or numeric
    steps.append(ColumnTypeSetter(exclude=NUMERIC_COLS))

    # split to X, y
    steps.append(XySplitter(target_col=TARGET_COL))

    # standardize numeric features
    #steps.append(Standardizer(**config['data.standardize']))

    # balance & convert to one-hot
    balancer = Balancer(method=config['balance.method'], params=config['balance.params'])
    if balancer.is_categorical:
        steps += [balancer, OneHotConverter()]
    else:
        steps += [OneHotConverter(), balancer]

    if verbose:
        print("Balancing method:", balancer.method)
        if verbose > 1:
            print("Pipeline:", " -> ".join([step.name for step in steps]))

    return Pipeline(steps=[(step.name, step) for step in steps])


def load_data(config: Dict) -> pd.DataFrame:
    data_file = str(DATA_PATH / "diabetic_data.csv")
    df = pd.read_csv(data_file, na_values="?", dtype={'payer_code': str})
    return df


class DataSplitter:

    def __init__(self, random_state: int, shuffle: bool = True, stratify: bool = True):
        self.random_state = random_state
        self.shuffle = shuffle
        self.stratify = stratify

    def split(self, Xy, test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            *Xy, test_size=test_size, shuffle=self.shuffle, random_state=self.random_state,
            stratify=Xy[1] if self.stratify else None)
        return (X_train, y_train), (X_test, y_test)
