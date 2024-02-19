import pandas as pd
import numpy as np
from paths import DATA_PATH
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureRemoverByName(BaseEstimator, TransformerMixin):
    '''Drops features based on feature list'''
    def __init__(self, features_to_remove: list):
        self.name = "FeatureRemoverByName"
        self.features_to_remove = features_to_remove

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.features_to_remove, inplace=False)

class RowRemoverByFeatureValue(BaseEstimator, TransformerMixin):
    '''Drops rows based on values list in a feature'''
    def __init__(self, feature: str, exclude_vals: list):
        self.name = f"RowRemoverByFeatureValue {feature}"
        self.feature = feature
        self.exclude_vals = exclude_vals
        self.rows_to_drop = []

    def fit(self, X, y=None):
        self.rows_to_drop = X.loc[X[self.feature].isin(self.exclude_vals)].index
        return self

    def transform(self, X):
        return X.drop(index=self.rows_to_drop).reset_index(drop=True)


class GroupFeatureValuesByDict(BaseEstimator, TransformerMixin):
    '''group multiple values inside a feature using dict: {new_value: [list_of_old_values]}
        all values that are not defined in the group_dict will return as other '''
    def __init__(self, feature: str, group_dict: dict):
        self.name = f"GroupFeatureValuesByDict {feature}"
        self.frature = feature
        self.group_dict = group_dict

    def fit(self, X):
        return self

    def transform(self, X):
        for key in self.group_dict.keys():
            X.loc[X[self.frature].isin(self.group_dict[key]), self.frature] = key
        if not 'other' in self.group_dict:
            X.loc[~X[self.frature].isin(self.group_dict.keys()), self.frature] = 'other'
        X[self.frature].replace(np.nan, 'missing')
        return X




class FeatureRemoverByBias(BaseEstimator, TransformerMixin):

    def __init__(self, thresh: float = .95):
        self.name = "FeatureRemoverByBias"
        self.thresh = thresh
        self.bias_scores = {}

    def fit(self, X, y=None):
        self.bias_scores = {col: max(X[col].value_counts(normalize=True))
                            for col in X.columns}
        return self

    def transform(self, X):
        return X[[col for col, bias in self.bias_scores.items() if bias < self.thresh]]



class CategoryReducer(BaseEstimator, TransformerMixin):

    def __init__(self, feature: str, lookup: 'dict[str, list]'):
        self.name = f"CategoryReducer {feature}"
        self.feature = feature
        self.lookup = lookup
        self.new_labels = None

    def fit(self, X, y=None):
        self.new_labels = np.array(['Missing' if is_na else 'Other' for is_na in X[self.feature].isna()], str)
        for new_label, current_labels in self.lookup.items():
            self.new_labels[X[self.feature].isin(current_labels)] = new_label
        return self

    def transform(self, X):
        assert len(self.new_labels) == len(X)
        X[self.feature] = self.new_labels
        return X


class ColumnTypeSetter(BaseEstimator, TransformerMixin):
    def __init__(self, type_: str, exclude: list = None):

        self.name = "ColumnTypeSetter"
        self.type_ = type_
        self.exclude = exclude if exclude else []

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in set(X.columns).difference(self.exclude):
            X[col] = X[col].astype(self.type_)
        return X


class XySplitter(BaseEstimator, TransformerMixin):
    def __init__(self, target_col: str, encode_target: bool = True, sanity_mode: str = 'none'):
        assert sanity_mode in ('none', 'must_fail', 'must_succeed')
        self.name = "XySplitter"
        self.target_col = target_col
        self.encode_target = encode_target
        self.sanity_mode = sanity_mode
        self._label_encoder = LabelEncoder()

    def fit(self, X, y=None):
        if self.encode_target:
            self._label_encoder.fit(X[self.target_col])
        return self

    def transform(self, X):
        y = X[self.target_col]

        if self.sanity_mode == 'must_succeed':
            print("! sanity_mode=", self.sanity_mode, "-> Features contain target labels.")
        else:
            X = X.drop(self.target_col, axis=1)

        if self.sanity_mode == 'must_fail':
            y = np.random.default_rng(0).permutation(y)
            print("! sanity_mode=", self.sanity_mode, "-> Shuffled target labels.")

        if self.encode_target:
            y = self._label_encoder.transform(y)
        return X, y


class ICDConverter(BaseEstimator, TransformerMixin):

    _ICD_GROUPS = {
        1: range(0, 140), 2: range(140, 240), 3: range(240, 280), 4: range(280, 290), 5: range(290, 320),
        6: range(320, 390), 7: range(390, 460), 8: range(460, 520), 9: range(520, 580), 10: range(580, 630),
        11: range(630, 680), 12: range(680, 710), 13: range(710, 740), 14: range(740, 760),
        15: range(760, 780), 16: range(780, 800), 17: range(800, 1000), 18: ['E'], 19: ['V'], 0: ['nan']
    }

    def __init__(self, features: list):
        self.name = "ICDConverter"
        self.lookup = {value: key for key, values_list in self._ICD_GROUPS.items() for value in values_list}
        self.features = features

    def _convert_value(self, value):
        if not isinstance(value, str):
            assert np.isnan(value)
            normalized_value = 'nan'
        elif value[0] in ('E', 'V'):
            normalized_value = value[0]
        else:
            normalized_value = int(value.split('.')[0])
        return self.lookup[normalized_value]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for feature in self.features:
            X[feature] = [self._convert_value(value) for value in X[feature]]
        return X


class OneHotConverter(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.name = "OneHotConverter"
        self._transformer = None
        pass

    def fit(self, X, y=None):
        if isinstance(X, tuple):
            X = X[0]
        features = list(X.select_dtypes('category').columns)
        self._transformer = ColumnTransformer(transformers=[
            (self.name, Pipeline(steps=[(self.name, OneHotEncoder(handle_unknown="error"))]), features)])
        self._transformer.fit(X)
        return self

    def transform(self, X):
        return self._transformer.transform(X)

class ReplaceValueToNan(BaseEstimator, TransformerMixin):

    def __init__(self, value='?'):
        self.name = 'ValueToNan'
        self.value = value

    def fit(self, X):
        return X == self.value

    def transform(self, X):
        return X.replace(self.value, np.nan)