import pandas as pd
import numpy as np
import object_builder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import imblearn as imbl
from typing import List, Dict, Tuple
from collections import defaultdict


def unpack_args(X, y):
    if isinstance(X, tuple):
        assert y is None and len(X) == 2
        return X
    return X, y


class DataTransformer(BaseEstimator, TransformerMixin):

    prop_setter = None
    frozen: bool = False  # if true, fit() does nothing
    
    def _fit(self, *args):
        return self

    def fit(self, *args):
        if self.frozen:
            return self
        else:
            return self._fit(*args)

    @property
    def name(self):
        name = self.__class__.__name__
        for attr in ['feature', 'method', 'new_feature']:
            if hasattr(self, attr):
                name += f"[{getattr(self, attr)}]"
        return name


class PropertySetter(DataTransformer):
    """
    Keeps track of column data types and column removals, and
        transforms the data accordingly
    """

    _PROPS = {'num': 'float', 'cat': 'category', 'drop': 'drop'}

    def __init__(self, default: str = None, verbose: int = 1):
        self.props = {}
        self.default = default
        self.verbose = verbose

    def register(self, **kwargs):
        for prop, cols in kwargs.items():
            assert prop in self._PROPS
            for col in cols:
                if col not in self.props or prop == 'drop':
                    self.props[col] = prop
                else:
                    assert self.props[col] in (prop, 'drop')
        return self

    def transform(self, X):

        report = defaultdict(list)

        # validate
        non_numeric = list(X.select_dtypes('object').columns)
        assert all(self.props.get(col, None) != 'num' for col in non_numeric)

        # set missing columns to default
        if self.default:
            missing_cols = [col for col in X.columns if col not in self.props]
            self.register(**{self.default: missing_cols})

        # drop:
        cols_to_drop = [col for col in X.columns if self.props.get(col, None) == 'drop']
        X.drop(columns=cols_to_drop, inplace=True)

        report['Dropped'] = cols_to_drop

        # set type:
        for col in X.columns:
            dtype = self._PROPS[self.props[col]]
            X[col] = X[col].astype(dtype)
            report[f'Converted to {dtype}'].append(col)

        if self.verbose:
            print(self.name + ":")
            for action, cols in report.items():
                print(f"  {action} ({len(cols)}):", cols)

        return X


class Standardizer(DataTransformer):
    """ Transform numeric features to be approximately normal and remove outliers """

    TRANSFORMS = {'none': lambda x: x.astype(float), 'sqrt': np.sqrt}

    def __init__(self, default_transform: str, feature_transforms: Dict[str, str],
                 outlier_p: float = .01, offset: float = 0):
        self.outlier_p = outlier_p
        self.offset = offset
        self.default_transform = default_transform
        self.feature_transforms = feature_transforms
        self.standardize_params = {}

    def _fit(self, Xy, y=None, **kwargs):
        X, y = unpack_args(Xy, y)

        def _get_clip_lims(x):
            if not self.outlier_p:
                return None
            else:
                return np.percentile(x, [100 * self.outlier_p, 100 * (1 - self.outlier_p)])

        for feature in X.select_dtypes('number').columns:
            tform = self.feature_transforms.get(feature, self.default_transform)
            func = self.TRANSFORMS[tform]
            x = X[feature].to_numpy(dtype=float)
            x = func(x)
            center = np.mean(x)
            scale = np.std(x)
            x = self._apply_transform(x, tform, center, scale)
            self.standardize_params[feature] = (tform, center, scale, _get_clip_lims(x))
        return self

    def _apply_transform(self, x, tform, center, scale, clip_lims=None):
        func = self.TRANSFORMS[tform]
        x = func(x)
        x = (x - center) / scale
        x += self.offset
        if clip_lims is not None:
            x = np.clip(x, *clip_lims)
        return x

    def transform(self, Xy, y=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        X = Xy[0].copy()
        for feature, (tform, center, scale, clip_lims) in self.standardize_params.items():
            X[feature] = self._apply_transform(X[feature].to_numpy(), tform, center, scale, clip_lims)
        return X, Xy[1]


class Balancer(DataTransformer):
    """ wrapper for data balancing classes
        provides a transformer-like api and manages sampling from train/test splits
    """

    _required_params = ['random_state']  # require even if sampler has a default value
    _dataframe_input = ['SMOTENC']  # samplers that allow or require dataframe input

    # @TODO add SDV

    def __init__(self, method: str, params: Dict):
        self.method = method
        self.params = params
        if self.method != 'none':
            self._get_new_sampler()  # verify we're good to go

    def _get_new_sampler(self):
        sampler_class = object_builder.get_class(self.method, [imbl.under_sampling, imbl.over_sampling, imbl.combine])
        params = {k: object_builder.get_instance(**v) if k == 'estimator' and v is not None else v
                  for k, v in self.params.items()}
        sampler = sampler_class(**params)
        for param in self._required_params:
            if hasattr(sampler, param) and param not in self.params:
                raise ValueError(f"Missing required parameter: {param}")

        return sampler

    @property
    def is_dataframe_in(self) -> bool:
        return self.method in self._dataframe_input

    def fit_transform(self, Xy, y=None, **kwargs):
        # called on training set -- resample and forget
        if self.method == 'none':
            return unpack_args(Xy, y)
        else:
            return self._get_new_sampler().fit_resample(*unpack_args(Xy, y), **kwargs)

    def transform(self, Xy, **kwargs):
        # called on validation set -- do nothing
        return Xy


class ReplaceValueToNan(DataTransformer):

    def __init__(self, value='?'):
        self.value = value

    def transform(self, X):
        return X.replace(self.value, np.nan)


class CategoryGroupOthers(DataTransformer):
    """ Set values that do not appear in 'nonother', to 'Other' """

    def __init__(self, nonother: Dict):
        self.nonother = nonother

    def transform(self, X):
        for col in self.nonother:
            X.loc[~X[col].isin(self.nonother[col]), col] = 'Other'
        self.prop_setter.register(cat=list(self.nonother.keys()))
        return X


class RowRemoverByFeatureValue(DataTransformer):
    """ Drop rows based on values list in a feature """

    def __init__(self, feature: str | List[str], exclude_vals: List):
        self.feature = ','.join(feature) if isinstance(feature, list) else feature
        self.exclude_vals = exclude_vals
        self.rows_to_drop = []

    def _fit(self, X, y=None):
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
        self.prop_setter.register(num=[new_colname], cat=[self.age_group_col])
        return X


class CategoryReducer(DataTransformer):
    """ Reduce category values according to lookup
        NA values are converted to 'missing'
        Values that are not in lookup are converted to 'Other'
    """

    def __init__(self, feature: str, lookup: Dict[str, List]):
        self.feature = feature
        self.lookup = lookup
        self.new_labels = None

    def _fit(self, X, y=None):
        self.new_labels = np.array(['Missing' if is_na else 'Other' for is_na in X[self.feature].isna()], str)
        for new_label, current_labels in self.lookup.items():
            self.new_labels[X[self.feature].isin(current_labels)] = new_label
        return self

    def transform(self, X):
        assert len(self.new_labels) == len(X)
        X[self.feature] = self.new_labels
        self.prop_setter.register(cat=[self.feature])
        return X


class TargetSeparator(DataTransformer):
    """ Split dataframe to X, y dataframes,
        Possibly manipulate the target variable for sanity tests
    """

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
    """ Group ICD category levels """

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
        self.prop_setter.register(cat=self.features)
        return X


class OneHotConverter(DataTransformer):
    """
    Onehot encode features and target, keep track of original feature names
    """

    def __init__(self):
        self._feature_transformer = None
        self._label_transformer = LabelEncoder()
        self.reverse_feature_names = []
        pass

    def _fit(self, Xy, y=None):

        X, y = unpack_args(Xy, y)
        name_sep = "."

        def _combinator(feature, category):
            return f"{feature}{name_sep}{str(category)}"

        onehot_encoder = OneHotEncoder(
            handle_unknown="error", feature_name_combiner=_combinator, sparse_output=False)
        cat_cols = X.select_dtypes('category').columns
        self._feature_transformer = ColumnTransformer(
            remainder='passthrough',
            transformers=[(self.name, Pipeline(steps=[(self.name, onehot_encoder)]), cat_cols)]
        )
        self._feature_transformer.fit(X)
        self._label_transformer.fit(y)
        self.reverse_feature_names = [s.split('__')[-1].split(name_sep)[0]
                                      for s in self._feature_transformer.get_feature_names_out()]

        return self

    def transform(self, Xy, y=None) -> Tuple[np.ndarray, np.ndarray]:
        X, y = unpack_args(Xy, y)
        X = self._feature_transformer.transform(X)
        y = self._label_transformer.transform(y)
        return X, y


class AddFeatureByNormalizing(DataTransformer):

    def __init__(self, mapping: Dict[str, Tuple[List[str], str]]):
        """
        mapping: dict of the form-
            feature_to_normalize_by -> (list_of_features_to_normalize, suffix_for_new_feature_name)

        each feature in [list_of_features_to_normalize] will be normalized by [feature_to_normalize_by] and
        added as a feature called <feature_name>_[suffix]
        """
        self.mapping = mapping

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for by, (features, suffix) in self.mapping.items():
            factor = X[by].to_numpy(dtype=float)
            mask = np.abs(factor) > np.finfo(float).eps
            for feature in features:
                new_feature = f"{feature}_{suffix}"
                X[new_feature] = np.divide(X[feature].to_numpy(), factor, out=np.zeros_like(factor), where=mask)
                self.prop_setter.register(num=[feature, new_feature, by])
        return X


class AddFeatureBySumming(DataTransformer):
    """ Create feature by summing other features """

    def __init__(self, features_to_sum: Dict[str, List[str]]):
        self.features_to_sum = features_to_sum

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for new_feature, features in self.features_to_sum.items():
            X[new_feature] = X[features].sum(axis=1)
            self.prop_setter.register(num=[new_feature] + features)
        return X


class AddFeatureByCounting(DataTransformer):
    """ Create feature by counting value(s) occurrences in a other features """

    def __init__(self, mapping: Dict, values_to_count: List, invert: bool, drop_originals: bool):
        """
            mapping: dict of form {new_feature: features_to_count_in}
            values_to_count: list of values to count
            invert: if True, count how many times value does not appear
            drop_originals: drop the counted features?
        """
        self.mapping = mapping
        self.values_to_count = values_to_count
        self.invert = invert
        self.drop_originals = drop_originals

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for (new_feature, features) in self.mapping.items():
            counts = X[features].isin(self.values_to_count).sum(axis=1)
            X[new_feature] = counts if not self.invert else len(features) - counts
            if self.drop_originals:
                self.prop_setter.register(drop=features)
        self.prop_setter.register(num=list(self.mapping.keys()))
        return X


class AddFeatureEncounter(DataTransformer):
    """ Construct 'encounter' from 'A1Cresult', and drop 'A1Cresult' """

    def transform(self, X):
        X['encounter'] = 'None'
        X.loc[X['A1Cresult'].isin(('>7', '>8')), 'encounter'] = '7_No'
        X.loc[X['A1Cresult'].isin(('>7', '>8')) & (X['change'] == 'Ch'), 'encounter'] = '7_Ch'
        X.loc[X['A1Cresult'] == 'Norm', 'encounter'] = 'Norm'
        self.prop_setter.register(cat=['encounter'], drop=['A1Cresult'])
        return X
