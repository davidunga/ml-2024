import pandas as pd
import numpy as np
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from data_transformers import \
    ColumnTypeSetter, FeatureRemoverByBias, FeatureRemoverByName, \
    RowRemoverByFeatureValue, CategoryReducer, XySplitter, ICDConverter, \
    OneHotConverter, SetRaresToOther, Balancer, Standardizer, RowRemoverByDuplicates, \
    AddFeatureAverageAge, AddFeatureByNormalizing, AddFeatureBySumming


TARGET_COL = 'readmitted'
NUMERIC_COLS = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses', 'weight']
DIAGNOSIS_COLS = ['diag_1', 'diag_2', 'diag_3']


def build_pipeline(config: dict, verbose: int = 1) -> Pipeline:
    """ make pipeline which operates on columns, and is applied only to features """

    steps = []

    steps.append(AddFeatureAverageAge(age_group_col='age'))

    # for normalize_by, (to_normalize, suffix) in config['data.add_by_normalize'].items():
    #     steps.append(AddFeatureByNormalizing(to_normalize=to_normalize, normalize_by=normalize_by, suffix=suffix))

    steps.append(AddFeatureBySumming(config['data.add_by_sum']))

    for col, exclude_vals in config['data.exclude_rows_where'].items():
        steps.append(RowRemoverByFeatureValue(feature=col, exclude_vals=exclude_vals))

    if config['data.exclude_pregnancy_diabetes']:
        steps.append(RowRemoverByFeatureValue(feature=DIAGNOSIS_COLS,
                                              exclude_vals=[ICDConverter.PREGNANCY_DIABETES_ICD]))

    steps.append(RowRemoverByDuplicates(config['data.exclude_rows_by_duplicates']))

    # remove columns by name
    steps.append(FeatureRemoverByName(features_to_remove=config['data.exclude_cols']))

    # remove high bias (=near uniform) columns
    steps.append(FeatureRemoverByBias(thresh=config['data.bias_thresh']))

    # merge categories according to manually defined mapping
    for col, lookup in config['data.recategorize'].items():
        steps.append(CategoryReducer(feature=col, lookup=lookup))

    # process diagnosis category levels
    steps.append(ICDConverter(features=DIAGNOSIS_COLS))

    # merge rare category levels to a single level ('other')
    steps.append(SetRaresToOther(thresh=config['data.rare_to_other_thresh'],
                                 features=config['data.rare_to_other_features']))

    # set column types to be either categorical or numeric
    steps.append(ColumnTypeSetter(exclude=NUMERIC_COLS))

    # standardize numeric features
    steps.append(Standardizer(**config['data.standardize']))

    # split to X, y
    steps.append(XySplitter(target_col=TARGET_COL))

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


def load_data(config: dict) -> pd.DataFrame:
    data_file = str(DATA_PATH / "diabetic_data.csv")
    df = pd.read_csv(data_file, na_values="?", dtype={'payer_code': str})
    return df
