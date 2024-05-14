from catboost._catboost import AllMetricsParams as catboost_AllMetricsParams
from typing import Dict

# --------------
_have_early_stopping = ['XGBClassifier', 'LGBMClassifier', 'CatBoostClassifier']
_early_stopping_params = ['early_stopping_rounds', 'eval_metric']
_best_iter_variations = ['best_iteration', 'best_iteration_', '_best_iteration']
# --------------


class EstimatorParamsManager:
    """ Provides a uniform interface with estimator parameters across packages """

    def __init__(self, estimator_name):
        self.estimator_name = estimator_name

    def supports_early_stopping(self) -> bool:
        return self.estimator_name in _have_early_stopping

    def adjust_to_estimator(self, params: Dict) -> Dict:
        if not self.supports_early_stopping() or params.get('early_stopping_rounds', None) == -1:
            params = _drop_early_stopping_params(params)
        if 'early_stopping_rounds' in params and 'n_estimators' in params:
            del params['n_estimators']
        if self.estimator_name == 'CatBoostClassifier':
            params = _convert_to_catboost_format(params)
        return params

    @staticmethod
    def get_best_iteration(estimator) -> int:
        return _get_best_iteration(estimator)


# --------------


def _convert_to_catboost_format(params: Dict) -> Dict:
    mapping = {k.lower(): k for k in catboost_AllMetricsParams()}
    return {k: mapping.get(v.lower(), v) for k, v in params.items() if 'metric' in k}


def _drop_early_stopping_params(params: Dict) -> Dict:
    return {k: v for k, v in params.items() if k not in _early_stopping_params}


def _get_best_iteration(estimator) -> int:
    for attr in _best_iter_variations:
        if hasattr(estimator, attr):
            return getattr(estimator, attr)
    assert AttributeError("Estimator doesnt have best iteration attribute")
