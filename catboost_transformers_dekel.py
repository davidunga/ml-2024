# Transformer module for catboost hyperparameter - Dekel
# this modoul include all the transformers for preprosessing the readmitted df
from copy import deepcopy
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer, QuantileTransformer
from sklearn.base import BaseEstimator, TransformerMixin 
from scipy.stats import skew
from scipy.stats import rankdata

class ChangeValueToValue(BaseEstimator, TransformerMixin):
    """
    Transformer that changes occurrences of a specified value in a DataFrame to another specified value.

    Parameters:
    -----------
    value_from : str or int
        The value in the dataset to be replaced.
    value_to : str or int
        The value to replace `value_from` with.
    **kwargs : Additional keyword arguments, passed to parent classes.

    Methods:
    --------
    fit(X, y=None):
        This method does nothing and is present for compatibility with scikit-learn's Transformer API.
    
    transform(X):
        Replaces all occurrences of `value_from` in the dataset `X` with `value_to`.

    Example:
    --------
    >>> import pandas as pd
    >>> from sklearn.pipeline import Pipeline
    >>> df = pd.DataFrame({'A': [1, 2, 1, 3], 'B': ['a', 'b', 'a', 'c']})
    >>> transformer = ChangeValueToValue(value_from=1, value_to=99)
    >>> transformer.transform(df)
         A  B
    0   99  a
    1    2  b
    2   99  a
    3    3  c
    """
    def __init__(self, value_from: str | int , value_to: str | int, **kwargs) -> None:
        self.value_from = value_from
        self.value_to = value_to

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[X_copy == self.value_from] = self.value_to
        return X_copy



class CategoryReducer(BaseEstimator, TransformerMixin):
    """
    Transformer that reduces the number of unique values in categorical features based on a lookup dictionary.

    Parameters:
    -----------
    lookup : dict of str: dict of str: list
        A nested dictionary where the keys are feature names and the values are dictionaries. 
        In these inner dictionaries, keys are the target categories to convert to, and values are lists of values to be replaced by the corresponding key.
    **kwargs : Additional keyword arguments, passed to parent classes.

    Methods:
    --------
    fit(X, y=None):
        This method does nothing and is present for compatibility with scikit-learn's Transformer API.
    
    transform(X):
        Transforms the DataFrame `X` by reducing the categories in specified features according to the lookup dictionary.
        If a category is not found in the lookup dictionary, it is replaced with 'other' unless 'other' is specified in the lookup.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': ['apple', 'banana', 'cherry', 'date'], 'B': ['cat', 'dog', 'elephant', 'fish']})
    >>> lookup = {
    ...     'A': {'fruit': ['apple', 'banana', 'cherry']},
    ...     'B': {'pet': ['cat', 'dog']}
    ... }
    >>> reducer = CategoryReducer(lookup=lookup)
    >>> reducer.transform(df)
           A      B
    0  fruit    pet
    1  fruit    pet
    2  fruit  other
    3  other  other
    """

    def __init__(self, lookup: dict[str, dict[str, list]], **kwargs):
        self.lookup = lookup

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for feature, feature_lookup in self.lookup.items():
            for change_to, change_from_list in feature_lookup.items():
                X_copy.loc[X_copy[feature].isin(change_from_list), feature] = change_to
            
            if 'other' not in feature_lookup.keys():
                X_copy.loc[~X_copy[feature].isin(feature_lookup.keys()), feature] = 'other'
        return X_copy




class DropRowsByValue(BaseEstimator, TransformerMixin):
    """
    Transformer that drops rows from a dataset based on specified values in certain features.

    Parameters:
    -----------
    lookup : dict of str: list
        A dictionary where the keys are feature names and the values are lists of values to drop from the DataFrame.

    Methods:
    --------
    fit(X, y=None):
        This method does nothing and is present for compatibility with scikit-learn's Transformer API.
    
    transform(X):
        Drops rows from the dataset `X` where the specified features contain the values to drop.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'a', 'b']})
    >>> lookup = {'A': [1, 3], 'B': ['b']}
    >>> dropper = DropRowsByValue(lookup=lookup)
    >>> dropper.transform(df)
       A  B
    2  3  a
    """

    def __init__(self, lookup: dict[str, list], **kwargs):
        self.lookup = lookup
        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for feature, value_to_drop in self.lookup.items():
            X_copy = X_copy[~X_copy[feature].isin(value_to_drop)]
        return X_copy




class DropColByValue(BaseEstimator, TransformerMixin):
    """
    Transformer that drops specified columns from a DataFrame.

    Parameters:
    -----------
    lookup : list
        A list of column names to be dropped from the DataFrame.

    Methods:
    --------
    fit(X, y=None):
        This method does nothing and is present for compatibility with scikit-learn's Transformer API.
    
    transform(X):
        Drops the specified columns from the DataFrame `X`. If a column in `lookup` does not exist in the DataFrame, it is ignored.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a', 'b', 'a', 'b'], 'C': [5, 6, 7, 8]})
    >>> lookup = ['A', 'C']
    >>> dropper = DropColByValue(lookup=lookup)
    >>> dropper.transform(df)
       B
    0  a
    1  b
    2  a
    3  b
    """

    def __init__(self, lookup: list, **kwargs):
        self.lookup = lookup
        

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.drop(columns=self.lookup, errors='ignore')
        return X_copy



class DropColWithLowVarious(BaseEstimator, TransformerMixin):
    """
    Transformer that drops columns from a DataFrame where the most common value's ratio exceeds a specified threshold.

    Parameters:
    -----------
    various_threshold : int, default=100
        The threshold percentage above which columns are dropped. If the most common value in a column appears in at least this percentage of the rows, the column is dropped.

    Methods:
    --------
    unique_values_and_ratio(X):
        Computes the number of unique values and the ratio of the most common value for each column in the DataFrame `X`.

    fit(X, y=None):
        Identifies the columns to drop based on the threshold.

    transform(X):
        Drops the identified columns from the DataFrame `X`.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 1, 1, 1], 'B': [2, 2, 2, 3], 'C': [3, 4, 5, 6]})
    >>> dropper = DropColWithLowVarious(various_threshold=75)
    >>> dropper.fit(df)
    >>> dropper.transform(df)
       B  C
    0  2  3
    1  2  4
    2  2  5
    3  3  6
    """

    def __init__(self, various_threshold: int = 100, **kwargs):
        self.various_threshold = various_threshold
        

    def unique_values_and_ratio(self, X):
        unique_counts = X.nunique()
        most_common_ratios = ((X == X.mode().values).sum() / len(X)) * 100
        self.common_ratio_df = pd.DataFrame({"Unique Values": unique_counts, "Most Common Ratio": most_common_ratios})

    def fit(self, X, y=None):
        self.unique_values_and_ratio(X)
        self.columns_to_drop = self.common_ratio_df[self.common_ratio_df['Most Common Ratio'] >= self.various_threshold].index
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy = X_copy.drop(columns=self.columns_to_drop, errors='ignore')
        return X_copy



class GroupCategoriesUsinValues(BaseEstimator, TransformerMixin):
    """
    Transformer that groups specified columns in a DataFrame into new features by applying a specified aggregation function.

    Parameters:
    -----------
    lookup : dict of str: list
        A dictionary where the keys are new feature names and the values are lists of old feature names to be aggregated.
    mode : str, default='sum'
        The aggregation function to apply. It can be any numpy function like 'sum', 'mean', 'max', etc.

    Methods:
    --------
    convert_to_float(df: pd.DataFrame) -> pd.DataFrame:
        Attempts to convert all columns in the DataFrame `df` to float. Columns that cannot be converted are left unchanged.

    fit(X, y=None):
        Prepares the lookup dictionary by verifying and converting specified columns to numeric types.

    transform(X):
        Applies the specified aggregation function to group the columns as defined in the lookup dictionary and creates new features.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': ['1', '2', '3', '4'],
    ...     'B': ['5', '6', '7', '8'],
    ...     'C': ['9', '10', '11', '12'],
    ...     'D': ['13', '14', '15', '16']
    ... })
    >>> lookup = {
    ...     'sum_AB': ['A', 'B'],
    ...     'sum_CD': ['C', 'D']
    ... }
    >>> grouper = GroupCategoriesUsinValues(lookup=lookup, mode='sum')
    >>> grouper.fit(df)
    >>> grouper.transform(df)
       A  B   C   D  sum_AB  sum_CD
    0  1  5   9  13     6.0    22.0
    1  2  6  10  14     8.0    24.0
    2  3  7  11  15    10.0    26.0
    3  4  8  12  16    12.0    28.0
    """

    def convert_to_float(self, df: pd.DataFrame) -> pd.DataFrame:
        for col in df.keys():
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                # If conversion to float raises an error, skip this column
                pass
        return df

    def __init__(self, lookup: dict[str, list], mode: str = 'sum', **kwargs):
        self.lookup = deepcopy(lookup)
        self.mode_str = mode
        self.mode = getattr(np, mode, np.sum)
        

    def fit(self, X, y=None):
        for new_feature, old_features in self.lookup.items():
            X = self.convert_to_float(X)
            try:
                self.lookup[new_feature] = X[old_features].select_dtypes(include=np.number).columns
            except KeyError:
                pass 
        return self

    def transform(self, X):
        X_copy = X.copy()
        for new_feature, old_features in self.lookup.items():
            try:
                # Try calculating mode for all old_features
                X_copy[new_feature] = self.mode(X_copy[old_features], axis=1).astype(float)
            except KeyError as e:
                # Remove missing keys and recalculate mode
                missing_key = e.args[0]
                print(f'missing key: {missing_key}')
                old_features = [feature for feature in old_features if feature != missing_key]
                X_copy[new_feature] = self.mode(X_copy[old_features], axis=1).astype(float)
                
            # Drop all old_features (including any removed missing keys) this is on hold for now
            # X_copy = X_copy.drop(columns=old_features, errors='ignore')
        
        return X_copy




class ScalerTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that scales numerical columns in a DataFrame using specified scaling methods.

    Parameters:
    -----------
    method : str, default='RobustScaler'
        The scaling method to apply. Options are 'RobustScaler', 'StandardScaler', or 'NoTransform'.
    numerical_columns : list, optional
        A list of numerical column names to be scaled. If None, all numerical columns will be considered.
    **kwargs : Additional keyword arguments, passed to the chosen scaler.

    Methods:
    --------
    fit(X, y=None):
        Fits the scaler to the specified numerical columns in the DataFrame `X`.

    transform(X):
        Transforms the specified numerical columns in the DataFrame `X` using the fitted scaler.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4],
    ...     'B': [10, 20, 30, 40],
    ...     'C': ['a', 'b', 'c', 'd']
    ... })
    >>> scaler = ScalerTransformer(method='StandardScaler', numerical_columns=['A', 'B'])
    >>> scaler.fit(df)
    >>> scaler.transform(df)
              A         B  C
    0 -1.341641 -1.341641  a
    1 -0.447214 -0.447214  b
    2  0.447214  0.447214  c
    3  1.341641  1.341641  d
    """

    def __init__(self, method='RobustScaler', numerical_columns=None, **kwargs):
        self.method = method
        self.numerical_columns = numerical_columns
        self.kwargs = kwargs
        self.scaler = None

    def fit(self, X, y=None):
        if self.method == 'RobustScaler':
            self.scaler = RobustScaler(**self.kwargs)
        elif self.method == 'StandardScaler':
            self.scaler = StandardScaler(**self.kwargs)
        elif self.method == 'NoTransform':
            self.scaler = NoTransform()
        else:
            raise ValueError("Unsupported method. Choose one of: 'RobustScaler', 'StandardScaler', or 'NoTransform'.")
        
        if self.numerical_columns is None:
            self.numerical_columns = X.select_dtypes(include='number').columns.tolist()

        try:
            self.scaler.fit(X[self.numerical_columns])
        except KeyError as e:
            missing_column = str(e).strip('\'')
            try:
                self.numerical_columns.remove(missing_column)
            except ValueError:
                pass  # Column is already removed
            return self.fit(X)
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy[self.numerical_columns] = self.scaler.transform(X_copy[self.numerical_columns])
        return X_copy



class LogTransformer(BaseEstimator, TransformerMixin):
    """A transformer that applies log transformation."""
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.log1p(X)

class SkewFixTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that fixes skewness in numerical columns of a DataFrame using specified transformation methods.

    Parameters:
    -----------
    skew_threshold : float, default=0.5
        The threshold above which a column is considered skewed and requires transformation.
    method : str, default='PowerTransformer'
        The transformation method to apply. Options are 'PowerTransformer', 'QuantileTransformer', 'LogTransformer', 'BoxCoxTransformer', or 'NoTransform'.
    numerical_columns : list, optional
        A list of numerical column names to be transformed. If None, all numerical columns will be considered.
    **kwargs : Additional keyword arguments, passed to the chosen transformer.

    Methods:
    --------
    fit(X, y=None):
        Identifies the skewed columns and fits the specified transformer to those columns.

    transform(X):
        Transforms the skewed columns in the DataFrame `X` using the fitted transformer.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 100],
    ...     'B': [10, 20, 30, 40, 50],
    ...     'C': [5, 5, 5, 5, 5]
    ... })
    >>> transformer = SkewFixTransformer(method='LogTransformer', numerical_columns=['A', 'B'])
    >>> transformer.fit(df)
    >>> transformer.transform(df)
              A         B  C
    0  0.693147  2.397895  5
    1  1.098612  3.044522  5
    2  1.386294  3.433987  5
    3  1.609438  3.713572  5
    4  4.615121  3.931826  5
    """

    def __init__(self, skew_threshold=0.5, method='PowerTransformer', numerical_columns=None, **kwargs):
        self.skew_threshold = skew_threshold
        self.method = method
        self.numerical_columns = numerical_columns
        self.kwargs = kwargs
        self.skewed_columns = []
        self.transformer = None
        self.box_cox_addition = 0

    # def fit(self, X, y=None):
    #     if self.numerical_columns is None:
    #         self.numerical_columns = X.select_dtypes(include='number').columns.tolist()

    #     self.skewed_columns = []
    #     for col in self.numerical_columns:
    #         col_skew = skew(X[col].dropna())  # Calculate skewness
    #         if np.abs(col_skew) > self.skew_threshold:
    #             self.skewed_columns.append(col)
        
    #     if self.method == 'PowerTransformer':
    #         self.transformer = PowerTransformer(**self.kwargs)
    #     elif self.method == 'QuantileTransformer':
    #         self.transformer = QuantileTransformer(output_distribution='normal', **self.kwargs)
    #     elif self.method == 'LogTransformer':
    #         self.transformer = LogTransformer()
    #     elif self.method == 'BoxCoxTransformer':
    #         self.box_cox_addition = X[self.numerical_columns].min()
    #         self.transformer = PowerTransformer(method='box-cox')
    #     elif self.method == 'NoTransform':
    #         self.transformer = NoTransform()
    #     else:
    #         raise ValueError("Unsupported method. Choose one of: 'PowerTransformer', 'QuantileTransformer', 'LogTransformer', 'BoxCoxTransformer', or 'NoTransform'.")
        
    #     return self

    def fit(self, X, y=None):
        if self.numerical_columns is None:
            self.numerical_columns = X.select_dtypes(include='number').columns.tolist()
    
        self.skewed_columns = []
        cols_to_remove = []
        for col in self.numerical_columns:
            try:
                numeric_col = X[col].dropna().astype(float)  # Convert to float
                if numeric_col.size == 0:  # Skip if column contains no valid numeric values
                    cols_to_remove.append(col)
                    continue
                col_skew = skew(numeric_col)  # Calculate skewness
                if np.abs(col_skew) > self.skew_threshold:
                    self.skewed_columns.append(col)
            except KeyError:
                cols_to_remove.append(col)
    
        # Remove missing columns from self.numerical_columns
        self.numerical_columns = [col for col in self.numerical_columns if col not in cols_to_remove]
        
        # Initialize transformer based on selected method
        if self.method == 'PowerTransformer':
            self.transformer = PowerTransformer(**self.kwargs)
        elif self.method == 'QuantileTransformer':
            self.transformer = QuantileTransformer(output_distribution='normal', **self.kwargs)
        elif self.method == 'LogTransformer':
            self.transformer = LogTransformer()
        elif self.method == 'BoxCoxTransformer':
            self.box_cox_addition = X[self.numerical_columns].min()
            self.transformer = PowerTransformer(method='box-cox')
        elif self.method == 'NoTransform':
            self.transformer = NoTransform()
        else:
            raise ValueError("Unsupported method. Choose one of: 'PowerTransformer', 'QuantileTransformer', 'LogTransformer', 'BoxCoxTransformer', or 'NoTransform'.")
        
        return self


    def transform(self, X):
        X_copy = X.copy()
        for col in self.skewed_columns:
            try:
                X_copy[col] = X_copy[col] - self.box_cox_addition[col]
            except TypeError:
                pass  # Skip if there's an issue with subtraction
            try:
                X_copy[col] = self.transformer.fit_transform(X_copy[col].values.reshape(-1, 1))
            except ValueError:
                X_copy[col] += 1  # Adjust to avoid log(0) or similar issues
                X_copy[col] = self.transformer.fit_transform(X_copy[col].values.reshape(-1, 1))
        return X_copy


class RankGaussTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies RankGauss normalization to numerical columns.

    Methods:
    --------
    fit(X, y=None):
        Returns self without any changes, as RankGauss does not require fitting.

    transform(X):
        Applies RankGauss normalization by ranking the data, scaling it to [-0.5, 0.5], and then applying the arcsin function.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> transformer = RankGaussTransformer()
    >>> transformer.fit(df)
    >>> transformer.transform(df)
           A         B
    0 -0.523599 -0.523599
    1  0.000000  0.000000
    2  0.523599  0.523599
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_rank = np.apply_along_axis(rankdata, 0, X)
        X_rank /= X_rank.max()
        X_rank -= 0.5
        return np.arcsin(X_rank)

class Winsorizer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies Winsorization to numerical columns to limit extreme values.

    Parameters:
    -----------
    lower_quantile : float, default=0.05
        The lower quantile for Winsorization.
    upper_quantile : float, default=0.95
        The upper quantile for Winsorization.

    Methods:
    --------
    fit(X, y=None):
        Calculates the lower and upper bounds for Winsorization based on the specified quantiles.

    transform(X):
        Applies Winsorization to the data, capping values outside the calculated bounds.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 100]})
    >>> transformer = Winsorizer(lower_quantile=0.05, upper_quantile=0.95)
    >>> transformer.fit(df)
    >>> transformer.transform(df)
         A
    0   1
    1   2
    2   3
    3   4
    4  94
    """
    def __init__(self, lower_quantile=0.05, upper_quantile=0.95):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X, y=None):
        self.lower_bound = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_bound = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for i in range(X_copy.shape[1]):
            X_copy.iloc[:, i] = np.where(X_copy.iloc[:, i] < self.lower_bound[i], self.lower_bound[i], X_copy.iloc[:, i])
            X_copy.iloc[:, i] = np.where(X_copy.iloc[:, i] > self.upper_bound[i], self.upper_bound[i], X_copy.iloc[:, i])
        return X_copy

class OutlierTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that handles outliers in numerical columns using specified transformation methods.

    Parameters:
    -----------
    transformer : str, default='RankGaussTransformer'
        The transformation method to apply. Options are 'RankGaussTransformer', 'Winsorizer', or 'NoTransform'.
    lower_quantile : float, default=0.05
        The lower quantile for Winsorization if 'Winsorizer' is chosen.
    upper_quantile : float, default=0.95
        The upper quantile for Winsorization if 'Winsorizer' is chosen.

    Methods:
    --------
    fit(X, y=None):
        Identifies numerical columns and fits the specified transformer.

    transform(X):
        Transforms the numerical columns using the fitted transformer.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4, 100], 'B': [10, 20, 30, 40, 50]})
    >>> transformer = OutlierTransformer(transformer='Winsorizer')
    >>> transformer.fit(df)
    >>> transformer.transform(df)
         A   B
    0   1  10
    1   2  20
    2   3  30
    3   4  40
    4  94  50
    """
    def __init__(self, transformer='RankGaussTransformer', lower_quantile=0.05, upper_quantile=0.95):
        self.transformer_name = transformer
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.transformer = None

    def fit(self, X, y=None):
        self.numerical_columns = X.select_dtypes(include=np.number).columns
        
        if self.transformer_name == 'RankGaussTransformer':
            self.transformer = RankGaussTransformer()
        elif self.transformer_name == 'Winsorizer':
            self.transformer = Winsorizer(lower_quantile=self.lower_quantile, upper_quantile=self.upper_quantile)
        elif self.transformer_name == 'NoTransform':
            self.transformer = NoTransform()
        else:
            raise ValueError("Unsupported transformer. Choose either 'RankGaussTransformer' or 'Winsorizer'.")
        
        self.transformer.fit(X[self.numerical_columns])
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed[self.numerical_columns] = self.transformer.transform(X[self.numerical_columns])
        return X_transformed


class ObjectToCategoryTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that converts object and specified columns to category dtype in a DataFrame.
    before changing to category, the data is transfomed to str

    Methods:
    --------
    fit(X, y=None):
        Returns self without any changes, as no fitting is required.

    transform(X):
        Converts columns of dtype 'object' or specified patterns to 'category'.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['1', '2', '3'], 'ID': ['1', '2', '3']})
    >>> transformer = ObjectToCategoryTransformer()
    >>> transformer.fit(df)
    >>> transformer.transform(df)
    A  B ID
    0  a  1  1
    1  b  2  2
    2  c  3  3
    """
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_transformed = X.copy()
        for col in X_transformed.columns:
            if X_transformed[col].dtype in ['category' ,'object'] or '_ID' in col.upper() or 'DIAG_' in col.upper():
                X_transformed[col] = X_transformed[col].astype('str')
                X_transformed[col] = X_transformed[col].astype('category')
        return X_transformed

class DropDuplicatesTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that drops duplicate rows in a DataFrame based on specified columns.

    Parameters:
    -----------
    columns_to_check_duplicates : list, optional
        A list of column names to check for duplicates. If None, all columns are checked.
    keep : {'first', 'last', False}, default='first'
        Determines which duplicates (if any) to keep. 'first' retains the first occurrence, 'last' retains the last occurrence, and False drops all duplicates.

    Methods:
    --------
    fit(X, y=None):
        Returns self without any changes, as no fitting is required.

    transform(X):
        Drops duplicate rows based on the specified columns.

    Example:
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'A': [1, 2, 2, 4], 'B': [5, 6, 6, 8]})
    >>> transformer = DropDuplicatesTransformer(columns_to_check_duplicates=['A', 'B'])
    >>> transformer.fit(df)
    >>> transformer.transform(df)
       A  B
    0  1  5
    1  2  6
    3  4  8
    """
    def __init__(self, columns_to_check_duplicates=None, keep='first'):
        self.columns_to_check_duplicates = columns_to_check_duplicates
        self.keep = keep
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        if self.columns_to_check_duplicates is None:
            return X
        X_transformed = X.copy()
        X_transformed = X_transformed.drop_duplicates(subset=self.columns_to_check_duplicates, keep=self.keep)
        return X_transformed



class NoTransform(BaseEstimator, TransformerMixin):
    """A dummy transformer that performs notthing."""
    def __init__(self):
        pass
        
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X