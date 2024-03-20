import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, PowerTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import imblearn as imbl
from scipy.stats import shapiro
from typing import List, Dict, Tuple
from copy import deepcopy


def categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(include=['category']).columns


def numerical_cols(df: pd.DataFrame):
    return df.select_dtypes(include=['number']).columns


class DataTransformer(BaseEstimator, TransformerMixin):

    def fit(self, *args):
        return self

    @property
    def name(self):
        name = self.__class__.__name__
        for attr in ['feature', 'method', 'new_feature']:
            if hasattr(self, attr):
                name += f"[{getattr(self, attr)}]"
        return name


class Standardizer(DataTransformer):
    """ Transform numeric features to be approximately normal and remove outliers """

    TRANSFORMS = {'none': lambda x: x, 'sqrt': np.sqrt}

    def __init__(self, outlier_p: float = .05, offset: float = 0):
        self.outlier_p = outlier_p
        self.offset = offset
        self.transforms = {}

    def fit(self, X: pd.DataFrame):
        if isinstance(X, tuple):
            X = X[0]
        inlier_range = [100 * self.outlier_p, 100 * (1 - self.outlier_p)]
        for feature in numerical_cols(X):
            x = X[feature].to_numpy()
            best_normality_score = 0
            for tform, func in self.TRANSFORMS.items():
                try:
                    x_transformed = func(x)
                except:
                    continue
                normality_score = shapiro(x_transformed).statistic
                if normality_score > best_normality_score:
                    best_normality_score = normality_score
                    center = np.median(x_transformed)
                    x_transformed -= center
                    scale = np.median(np.abs(x_transformed))
                    x_transformed /= scale
                    clip_lims = np.percentile(x_transformed, inlier_range)
                    self.transforms[feature] = (tform, center, scale, clip_lims)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature, (tform, center, scale, clip_lims) in self.transforms.items():
            func = self.TRANSFORMS[tform]
            x = func(X[feature].to_numpy())
            x = (x - center) / scale + self.offset
            X[feature] = np.clip(x, *clip_lims)
        return X


class Balancer(DataTransformer):
    """ wrapper for data balancing classes """

    METHODS = {
        # method_name : (sampler_type, sampler_params)
        'RandomUnderSampler': (imbl.under_sampling.RandomUnderSampler, {'random_state': 1}),
        'NearMiss1': (imbl.under_sampling.NearMiss, {'version': 1}),
        'InstanceHardnessThreshold': (imbl.under_sampling.InstanceHardnessThreshold,
                                      {'random_state': 1, 'cv': 5, 'estimator': LogisticRegression()}),
        'SMOTENC': (imbl.over_sampling.SMOTENC, {'categorical_features': 'auto'})
    }

    # SDV

    CATEGORICAL_METHODS = ('SMOTENC',)

    def __init__(self, method: str, params: Dict = None):
        """
        method: method name, must me key of METHODS
        params: override params specified in METHODS. keys that do not appear in METHODS params are ignored.
        """
        self.method = method
        sampler_type, sampler_params = Balancer.METHODS[method]
        if params:
            sampler_params = {k: params.get(k, v) for k, v in sampler_params.items()}
        self._sampler = sampler_type(**sampler_params)

    @property
    def is_categorical(self) -> bool:
        """ does transformer expect categorical (dataframe) input? """
        return self.method in Balancer.CATEGORICAL_METHODS

    def fit_transform(self, Xy, *args):
        return self._sampler.fit_resample(Xy[0], Xy[1])

    def fit(self, *args): raise AssertionError("Only fit_transform should be called")
    def transform(self, *args): raise AssertionError("Only fit_transform should be called")


class ReplaceValueToNan(DataTransformer):

    def __init__(self, value='?'):
        self.value = value

    def transform(self, X):
        return X.replace(self.value, np.nan)


class CategoryGroupOthers(DataTransformer):

    def __init__(self, nonother: Dict):
        self.nonother = nonother

    def transform(self, X):
        for col in self.nonother:
            X.loc[~X[col].isin(self.nonother[col]), col] = 'Other'
        return X


class FeatureRemoverByName(DataTransformer):
    """ Drop features based on feature list """

    def __init__(self, features_to_remove: List):
        self.features_to_remove = features_to_remove

    def transform(self, X):
        return X.drop(columns=self.features_to_remove, inplace=False)


class RowRemoverByFeatureValue(DataTransformer):
    """ Drop rows based on values list in a feature """

    def __init__(self, feature: str | List[str], exclude_vals: List):
        self.feature = ','.join(feature) if isinstance(feature, list) else feature
        self.exclude_vals = exclude_vals
        self.rows_to_drop = []

    def fit(self, X, y=None):
        cols = self.feature.split(',')
        self.rows_to_drop = X.loc[X[cols].isin(self.exclude_vals).any(axis=1)].index
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop(index=self.rows_to_drop).reset_index(drop=True)


class RowRemoverByDuplicates(DataTransformer):
    """ Drop duplicate rows """

    def __init__(self, feature: str | List[str]):
        self.feature = ','.join(feature) if isinstance(feature, list) else feature

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.drop_duplicates(self.feature.split(',')).reset_index(drop=True)


class AddFeatureAverageAge(DataTransformer):
    """ Convert age group category to numeric """

    def __init__(self, age_group_col: str):
        self.age_group_col = age_group_col

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        new_colname = self.age_group_col + '_avg'
        assert new_colname not in X
        X[new_colname] = [(int(bounds[0]) + int(bounds[1])) // 2
                          for bounds in (s[1:-1].split('-') for s in X[self.age_group_col])]
        return X


class FeatureRemoverByBias(DataTransformer):

    def __init__(self, thresh: float = .95):
        self.thresh = thresh
        self.bias_scores = {}
        self.features_to_keep = []

    def fit(self, X, y=None):
        self.bias_scores = {col: max(X[col].value_counts(normalize=True))
                            for col in X.columns}
        self.features_to_keep = [col for col, bias in self.bias_scores.items() if bias < self.thresh]
        return self

    def transform(self, X):
        return X[self.features_to_keep]


class CategoryReducer(DataTransformer):
    """ Reduce category values according to lookup
        NA values are converted to 'missing'
        Values that are not in lookup are converted to 'Other'
    """

    def __init__(self, feature: str, lookup: Dict[str, List]):
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


class CategoryReducer_with_other(DataTransformer):
    """ Reduce category values according to lookup
        NA values are converted to 'missing'
        Values that are not in lookup are converted to 'Other'
        if "other" is in the lookup, keys that are not defined in the lookup will not change to "Other"
    """
    def __init__(self, feature: str, lookup: Dict[str, List]):
       
        self.feature = feature
        self.lookup = lookup
        self.new_labels = None
        self.keys = [str.lower(key) for key in lookup.keys()]

    def fit(self, X):
        self.new_labels = deepcopy(X[self.feature])
        
        for key in self.lookup.keys():
            self.new_labels[self.new_labels.isin(self.lookup[key])] = key
        if not ('other' in self.keys):
            self.new_labels[~self.new_labels.isin(self.lookup.keys())] = 'Other'
        self.new_labels.replace(np.nan, 'Missing')
        return self

    def transform(self, X):
        assert len(self.new_labels) == len(X)
        X[self.feature] = self.new_labels
        return X





class ColumnTypeSetter(DataTransformer):

    def __init__(self, type_: str = 'category', exclude: List[str] = None):
        self.type_ = type_
        self.exclude = exclude if exclude else []

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in set(X.columns).difference(self.exclude):
            X[col] = X[col].astype(self.type_)
        return X


class XySplitter(DataTransformer):
    """ Split dataframe to X, y dataframes """

    def __init__(self, target_col: str, sanity_mode: str = 'none'):
        assert sanity_mode in ('none', 'must_fail', 'must_succeed')
        self.target_col = target_col
        self.sanity_mode = sanity_mode

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        y = X[self.target_col]

        if self.sanity_mode == 'must_succeed':
            print("! sanity_mode=", self.sanity_mode, "-> Features contain target labels.")
        else:
            X = X.drop(self.target_col, axis=1)

        if self.sanity_mode == 'must_fail':
            y = np.random.default_rng(0).permutation(y)
            print("! sanity_mode=", self.sanity_mode, "-> Shuffled target labels.")

        return X, y


class ICDConverter(DataTransformer):

    ICD_GROUPS = {
        1: range(0, 140), 2: range(140, 240), 3: range(240, 280), 4: range(280, 290), 5: range(290, 320),
        6: range(320, 390), 7: range(390, 460), 8: range(460, 520), 9: range(520, 580), 10: range(580, 630),
        11: range(630, 680), 12: range(680, 710), 13: range(710, 740), 14: range(740, 760),
        15: range(760, 780), 16: range(780, 800), 17: range(800, 1000), 18: ['E'], 19: ['V'], 0: ['nan']
    }
    PREGNANCY_DIABETES_ICD = 648.8

    def __init__(self, features: List):
        self.lookup = {value: key for key, values_list in self.ICD_GROUPS.items() for value in values_list}
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

    def transform(self, X):
        for feature in self.features:
            X[feature] = [self._convert_value(value) for value in X[feature]]
        return X


class OneHotConverter(DataTransformer):

    def __init__(self):
        self._feature_transformer = None
        self._label_transformer = LabelEncoder()
        self.reverse_feature_names = []
        pass

    def fit(self, Xy: Tuple[pd.DataFrame, pd.DataFrame]):

        sep = "."

        def _combinator(feature, category):
            return f"{feature}{sep}{str(category)}"

        onehot_encoder = OneHotEncoder(handle_unknown="error", feature_name_combiner=_combinator)
        self._feature_transformer = ColumnTransformer(
            remainder='passthrough',
            transformers=[(self.name, Pipeline(steps=[(self.name, onehot_encoder)]), categorical_cols(Xy[0]))]
        )
        self._feature_transformer.fit(Xy[0])
        self._label_transformer.fit(Xy[1])
        self.reverse_feature_names = [s.split('__')[-1].split(sep)[0]
                                      for s in self._feature_transformer.get_feature_names_out()]

        return self

    def transform(self, Xy: Tuple[pd.DataFrame, pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray]:
        X = self._feature_transformer.transform(Xy[0])
        y = self._label_transformer.transform(Xy[1])
        return X, y


class AddFeatureByNormalizing(DataTransformer):

    def __init__(self, to_normalize: List[str], normalize_by: str, suffix: str = "norm"):
        self.to_normalize = to_normalize
        self.normalize_by = normalize_by
        self.suffix = suffix

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature in self.to_normalize:
            new_feature = f"{feature}_{self.suffix}"
            X[new_feature] = X[feature].to_numpy() / (1e-6 + X[self.normalize_by].to_numpy())
        return X


class AddFeatureBySumming(DataTransformer):

    def __init__(self, features_to_sum: Dict[str, List[str]]):
        self.features_to_sum = features_to_sum

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for new_feature, features in self.features_to_sum.items():
            X[new_feature] = X[features].sum(axis=1)
        return X


class AddFeatureByCounting(DataTransformer):
    """ Create feature by counting value(s) occurrences in a subset of features """

    def __init__(self, features: List[str], values_to_count: List, new_feature: str, invert: bool):
        """
            features: list of features to count in
            values_to_count: list of values to count
            new_feature: name of new feature to create
            invert: if True, count how many times value does not appear
        """
        self.features = features
        self.values_to_count = values_to_count
        self.new_feature = new_feature
        self.invert = invert

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        counts = X[self.features].isin(self.values_to_count).sum(axis=1)
        X[self.new_feature] = counts if not self.invert else len(self.features) - counts
        return X


class AddFeatureEncounter(DataTransformer):

    def transform(self, X):
        X['encounter'] = 'None'
        X.loc[X['A1Cresult'].isin(('>7', '>8')), 'encounter'] = '7_No'
        X.loc[X['A1Cresult'].isin(('>7', '>8')) & (X['change'] == 'Ch'), 'encounter'] = '7_Ch'
        X.loc[X['A1Cresult'] == 'Norm', 'encounter'] = 'Norm'
        return X
