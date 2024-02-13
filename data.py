import pandas as pd
import numpy as np
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from data_transformers import ColumnTypeSetter, FeatureRemoverByBias, FeatureRemoverByName, \
    RowRemoverByFeatureValue, CategoryReducer, XySplitter, ICDConverter, OneHotConverter

NUMERIC_COLS = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses', 'weight']


def make_data_init_pipe(config: dict) -> Pipeline:
    """ make pipeline that operates on both features and target variables, and performs the X-y separation """
    steps = []
    for col, exclude_vals in config['data.exclude_rows_where'].items():
        steps.append(RowRemoverByFeatureValue(feature=col, exclude_vals=exclude_vals))
    steps.append(CategoryReducer(feature='readmitted', lookup=config['data.readmitted_groups']))
    steps.append(XySplitter('readmitted'))
    return Pipeline(steps=[(step.name, step) for step in steps])


def make_feature_prep_pipe(config: dict) -> Pipeline:
    """ make pipeline which operates on columns, and is applied only to features """

    steps = []

    # remove columns by name
    steps.append(FeatureRemoverByName(features_to_remove=config['data.exclude_cols']))

    # remove high bias (=near uniform) columns
    steps.append(FeatureRemoverByBias(thresh=config['data.bias_thresh']))

    # reduce category by merging labels
    for col, lookup in config['data.recategorize'].items():
        steps.append(CategoryReducer(feature=col, lookup=lookup))

    steps.append(ICDConverter(features=['diag_1', 'diag_2', 'diag_3']))

    # set column types to be either categorical or numeric
    steps.append(ColumnTypeSetter(type_='category', exclude=NUMERIC_COLS))

    steps.append(OneHotConverter())

    steps = [(step.name, step) for step in steps]
    return Pipeline(steps=steps)


def load_data(config: dict) -> pd.DataFrame:
    data_file = str(DATA_PATH / "diabetic_data.csv")
    df = pd.read_csv(data_file, na_values="?", dtype={'payer_code': str})
    return df
