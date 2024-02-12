import pandas as pd
import numpy as np
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERIC_COLS = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
                'num_medications', 'number_outpatient', 'number_emergency',
                'number_inpatient', 'number_diagnoses', 'weight']


def value_count_bias(df: pd.DataFrame) -> dict[str, float]:
    """ the normalized count of the most common value in each column """
    return {col: max(df[col].value_counts()) / len(df) for col in df.columns}


def clean_data(config: dict, df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unwanted rows & columns, and convert to categorical/numeric
    """

    df.drop(columns=config['data.exclude_cols'], inplace=True)

    drop_mask = np.zeros(len(df), bool)
    for col, exclude_vals in config['data.exclude_rows_where'].items():
        drop_mask = drop_mask | df[col].isin(exclude_vals)
    df[~drop_mask].reset_index(drop=True)

    high_bias_cols = [col for col, bias in value_count_bias(df).items() if bias > config['data.bias_thresh']]
    df.drop(columns=high_bias_cols, inplace=True)

    for col, lookup in config['data.recategorize'].items():
        new_labels = np.array(['Missing' if is_na else 'Other' for is_na in df[col].isna()], str)
        for new_label, current_labels in lookup.items():
            new_labels[df[col].isin(current_labels)] = new_label
        df[col] = new_labels

    for col in df.columns:
        if col not in NUMERIC_COLS:
            df[col] = df[col].astype('category')
        else:
            assert df[col].dtype.kind == 'i'

    return df


def load_data(config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_file = str(DATA_PATH / "diabetic_data.csv")
    df = pd.read_csv(data_file, na_values="?", dtype={'payer_code': str})
    df = clean_data(config, df)
    features_df = df.drop("readmitted", axis=1)
    target_df = df["readmitted"]
    return features_df, target_df


def make_data_transformer(config: dict, df: pd.DataFrame) -> ColumnTransformer:

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns

    numeric_pipe = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_pipe = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    transformer = ColumnTransformer(transformers=[
        ('numeric', numeric_pipe, numeric_cols),
        ('categorical', categorical_pipe, categorical_cols)
    ])

    return transformer

