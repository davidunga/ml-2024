
from catboost_transformers_dekel import DropDuplicatesTransformer
from typing import Union, Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
import optuna
from catboost_transformers_dekel import DropDuplicatesTransformer,  DropColWithLowVarious, GroupCategoriesUsinValues, SkewFixTransformer, ScalerTransformer, OutlierTransformer
import pandas as pd
def instantiate_DropDuplicatesTransformer(trial: optuna.Trial, drop_suggest_categorical=[True, False]) -> DropDuplicatesTransformer:
    """
    Instantiate a DropDuplicatesTransformer object based on Optuna trial parameters.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object used for suggesting parameters.
    drop_suggest_categorical : list, default=[True, False]
        List of values to suggest for the 'drop_duplicates' parameter.

    Returns:
    --------
    DropDuplicatesTransformer
        An instance of DropDuplicatesTransformer with specified parameters.
        """

    
    drop_duplicates = trial.suggest_categorical('drop_duplicates', drop_suggest_categorical)
    if drop_duplicates:
        keep = trial.suggest_categorical('keep', ['first'])
        columns_to_check_duplicates = ['patient_nbr']
    else:
        keep = 'first'
        columns_to_check_duplicates = None
    return DropDuplicatesTransformer(columns_to_check_duplicates=columns_to_check_duplicates, keep=keep)




def instantiate_TransformerLookup(trial: optuna.Trial, lookup: Union[Dict[str, List], List], transformer, must_True_coloumns=['readmitted'], suggest_categorical=[True, False], **kwargs):
    """
    Instantiate a transformer based on Optuna trial parameters and a lookup dictionary or list.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object used for suggesting parameters.
    lookup : Union[Dict[str, List], List]
        A dictionary or list containing categories or values to drop.
    transformer : class
        The transformer class to instantiate, e.g., DropRowsByValue.
    must_True_coloumns : list, default=['readmitted']
        Columns that must always be dropped if present in the dataset.
    suggest_categorical : list, default=[True, False]
        List of values to suggest for whether to drop each category or value.
    **kwargs : Additional keyword arguments to pass to the transformer constructor.

    Returns:
    --------
    transformer
        An instance of the specified transformer class from catboost_transformers_dekel
    """
    if type(lookup) == dict:
        categories_to_drop = []
        values_to_drop = []
        for key, values in lookup.items():
            if (trial.suggest_categorical(f'{str(transformer)}_{str(key)}', suggest_categorical) or (key in must_True_coloumns)):
                categories_to_drop.append(key)
                values_to_drop.append(values)
        new_lookup = {category: values_to_drop[index] for index, category in enumerate(categories_to_drop)}
    elif type(lookup) == list:
        new_lookup = []
        for value in lookup:
            if trial.suggest_categorical(f'{str(transformer)}_{str(value)}', suggest_categorical):
                new_lookup.append(value)
    # Create the transformer instance
    return transformer(lookup=new_lookup, **kwargs)



def instantiate_DropColWithLowVarious(trial: optuna.Trial, low_value=75, high_value=100) -> DropColWithLowVarious:
    """
    Instantiate a DropColWithLowVarious transformer based on Optuna trial parameters.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object used for suggesting parameters.
    low_value : int, optional
        The lower bound for the various_threshold parameter suggested by Optuna. Defaults to 75.
    high_value : int, optional
        The upper bound for the various_threshold parameter suggested by Optuna. Defaults to 100.

    Returns:
    --------
    DropColWithLowVarious
    """
    various_threshold = trial.suggest_int('various_threshold', low_value, high_value)
    return DropColWithLowVarious(various_threshold=various_threshold)



def instantiate_GroupCategoriesUsinValues(trial: optuna.Trial, lookup) -> GroupCategoriesUsinValues:
    """
    Instantiate a GroupCategoriesUsinValues transformer based on Optuna trial parameters.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object used for suggesting parameters.
    lookup : dict, optional
        Dictionary containing the lookup for group categories. Defaults to `group_features_lookup`.

    Returns:
    --------
    GroupCategoriesUsinValues
        An instance of GroupCategoriesUsinValues with specified parameters.

    """
    mode = trial.suggest_categorical('mode', ['sum', 'mean', 'median'])
    return GroupCategoriesUsinValues(lookup=lookup, mode=mode)



def instantiate_SkewFixTransformer(trial: optuna.Trial, numerical_columns, skew_suggest_categorical=[True, False], skew_threshold: "number >= 0" = 3) -> SkewFixTransformer:
    """
    Instantiate a SkewFixTransformer based on Optuna trial parameters.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object used for suggesting parameters.
    numerical_columns : list
        List of numerical columns to consider for skew correction.
    skew_suggest_categorical : list, default=[True, False]
        List of values to suggest for whether to include each numerical column for skew correction.
    skew_threshold : float, default=1
        The threshold value for skewness. If the skewness of a numerical column exceeds this threshold, a transformation will be applied.

    Returns:
    --------
    SkewFixTransformer
        An instance of SkewFixTransformer with specified parameters.

    Notes:
    ------
    If the skewness of your data is between -1 and 1, it's considered moderately skewed, and you may not need to apply any transformation.
    If the skewness is less than -1 or greater than 1, it's considered highly skewed, and applying a transformation may help improve the performance of your model.
"""
    new_lookup = []
    for value in numerical_columns:
        if trial.suggest_categorical(f'skew_{str(value)}', skew_suggest_categorical):
            new_lookup.append(value)

    skew_threshold = trial.suggest_float('skew_threshold', skew_threshold * -1, skew_threshold)
    method_type = trial.suggest_categorical('skew_method', ['PowerTransformer', 'QuantileTransformer', 'LogTransformer', 'BoxCoxTransformer', 'NoTransform'])
    return SkewFixTransformer(skew_threshold=skew_threshold, method=method_type, numerical_columns=new_lookup)


class LogTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)



def instantiate_ScalerTransformer(trial: optuna.Trial, numerical_columns, scalar_suggest_categorical=[True, False], method_choices = ['RobustScaler', 'StandardScaler', 'NoTransform']) -> ScalerTransformer:
    """
    Instantiate a ScalerTransformer based on Optuna trial parameters.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object used for suggesting parameters.
    numerical_columns : list
        List of numerical columns to scale.
    scalar_suggest_categorical : list, default=[True, False]
        List of values to suggest for whether to include each numerical column for scaling.

    Returns:
    --------
    ScalerTransformer
        An instance of ScalerTransformer with specified parameters.
    """
    new_lookup = []
    for value in numerical_columns:
        if trial.suggest_categorical(f'scaler_{str(value)}', scalar_suggest_categorical):
            new_lookup.append(value)

    
    method_type = trial.suggest_categorical('scalar_method', method_choices)
        
    return ScalerTransformer(method=method_type, numerical_columns=new_lookup)






def instantiate_OutlierTransformer(trial: optuna.Trial, low_range=[0, 0.5], high_range=[0.5, 1]) -> OutlierTransformer:
    """
    Instantiate an OutlierTransformer based on Optuna trial parameters.

    Parameters:
    -----------
    trial : optuna.Trial
        Optuna trial object used for suggesting parameters.
    low_range : list, default=[0, 0.5]
        The range for suggesting the lower quantile value. Defaults to [0, 0.5].
    high_range : list, default=[0.5, 1]
        The range for suggesting the upper quantile value. Defaults to [0.5, 1].

    Returns:
    --------
    OutlierTransformer
        An instance of OutlierTransformer with specified parameters.
    """
    transformer = trial.suggest_categorical('transformer', ['RankGaussTransformer', 'Winsorizer', 'NoTransform'])
    lower_quantile = trial.suggest_float('lower_quantile', low_range[0], low_range[1])
    upper_quantile = trial.suggest_float('upper_quantile', high_range[0], high_range[1])
    return OutlierTransformer(transformer=transformer, lower_quantile=lower_quantile, upper_quantile=upper_quantile)

class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values in a DataFrame using scikit-learn's imputer.

    Parameters:
    -----------
    imputer : object
        The imputer object from scikit-learn used for imputation.

    Attributes:
    -----------
    imputer : object
        The fitted imputer object.

    Methods:
    --------
    fit(X, y=None):
        Fit the imputer to the data.

    transform(X):
        Impute missing values in the input DataFrame and return a new DataFrame with imputed values.

    Returns:
    --------
    DataFrame
        A DataFrame with missing values imputed.
    """

    def __init__(self, imputer):
        self.imputer = imputer

    def fit(self, X, y=None):
        self.imputer.fit(X, y)
        return self

    def transform(self, X):

        X_transformed = self.imputer.transform(X)
        return pd.DataFrame(X_transformed, columns=X.columns)
