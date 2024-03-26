from sklearn.model_selection import BaseCrossValidator, StratifiedKFold
from sklearn.model_selection._search import BaseSearchCV
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from typing import Union


def set_pipecv(searchcv: BaseSearchCV, pipe: Pipeline, cv: Union[BaseCrossValidator, int] = None):
    """
    Replace the cv-splitter of a cross-val search object, with a pipe cross validator.
    """
    if cv is None:
        cv = searchcv.cv
    pipecv = PipeCrossValidator(pipe=pipe, cv=cv)
    _attach_pipecv_to_searchcv(pipecv, searchcv)


class PipeCrossValidator:
    """
    A cross validator that applies a pipe to the train/test splits,
        and returns the transformed datas as-well as the split indexes.
    This allows to manipulate the data within each cv iteration, e.g., to
        upsample the training set.
    """

    def __init__(self, pipe: Pipeline, cv: BaseCrossValidator | int = 5):
        self.cv = cv if isinstance(cv, BaseCrossValidator) else self._get_default_cv(n_splits=cv)
        self.pipe = pipe

    @staticmethod
    def _get_default_cv(n_splits):
        return StratifiedKFold(n_splits=n_splits, shuffle=True)

    def get_n_splits(self, *args, **kwargs):
        return self.cv.get_n_splits(*args, **kwargs)

    def split(self, X, y=None, groups=None):

        def _get_by_index(ixs):
            if hasattr(X, 'iloc'):
                return X.iloc[ixs], y.iloc[ixs]
            else:
                return X[ixs], y[ixs]

        for train_ixs, test_ixs in self.cv.split(X, y, groups):

            X_train, y_train = self.pipe.fit_transform(_get_by_index(train_ixs))
            X_test, y_test = self.pipe.transform(_get_by_index(test_ixs))

            XX = np.r_[X_train, X_test]
            yy = np.r_[y_train, y_test]

            train = np.arange(len(X_train))
            test = np.arange(len(X_train), len(XX))

            yield train, test, XX, yy


def _attach_pipecv_to_searchcv(pipecv: PipeCrossValidator, searchcv: BaseSearchCV):
    searchcv.cv = pipecv
    searchcv.fit = fit_with_pipecv.__get__(searchcv, searchcv.__class__)

# ------------------------------------------

import time
from collections import defaultdict
from itertools import product
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, _fit_context, is_classifier
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _check_method_params, check_is_fitted, indexable
from sklearn.model_selection._split import check_cv
from sklearn.model_selection._validation import (
    _aggregate_score_dicts,
    _fit_and_score,
    _insert_error_scores,
    _normalize_score_results,
    _warn_or_raise_about_fit_failures,
)


@_fit_context(
    # *SearchCV.estimator is not validated yet
    prefer_skip_nested_validation=False
)
def fit_with_pipecv(self: BaseSearchCV, X, y=None, **params):
    """

    Mod of BaseSearchCV.fit which uses PipeCrossValidator instead of regular CV-Splitter

    ---------------------------------------------------

    Run fit with all sets of parameters.

    Parameters
    ----------

    X : array-like of shape (n_samples, n_features)
        Training vector, where `n_samples` is the number of samples and
        `n_features` is the number of features.

    y : array-like of shape (n_samples, n_output) \
        or (n_samples,), default=None
        Target relative to X for classification or regression;
        None for unsupervised learning.

    **params : dict of str -> object
        Parameters passed to the ``fit`` method of the estimator, the scorer,
        and the CV splitter.

        If a fit parameter is an array-like whose length is equal to
        `num_samples` then it will be split across CV groups along with `X`
        and `y`. For example, the :term:`sample_weight` parameter is split
        because `len(sample_weights) = len(X)`.

    Returns
    -------
    self : object
        Instance of fitted estimator.
    """
    estimator = self.estimator
    # Here we keep a dict of scorers as is, and only convert to a
    # _MultimetricScorer at a later stage. Issue:
    # https://github.com/scikit-learn/scikit-learn/issues/27001
    scorers, refit_metric = self._get_scorers(convert_multimetric=False)

    X, y = indexable(X, y)
    params = _check_method_params(X, params=params)

    routed_params = self._get_routed_params_for_fit(params)

    cv_orig = check_cv(self.cv, y, classifier=is_classifier(estimator))
    n_splits = cv_orig.get_n_splits(X, y, **routed_params.splitter.split)

    base_estimator = clone(self.estimator)

    parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

    fit_and_score_kwargs = dict(
        scorer=scorers,
        fit_params=routed_params.estimator.fit,
        score_params=routed_params.scorer.score,
        return_train_score=self.return_train_score,
        return_n_test_samples=True,
        return_times=True,
        return_parameters=False,
        error_score=self.error_score,
        verbose=self.verbose,
    )
    results = {}
    with parallel:
        all_candidate_params = []
        all_out = []
        all_more_results = defaultdict(list)

        def evaluate_candidates(candidate_params, cv=None, more_results=None):
            cv = cv or cv_orig
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)

            if self.verbose > 0:
                print(
                    "Fitting {0} folds for each of {1} candidates,"
                    " totalling {2} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits
                    )
                )

            out = parallel(
                delayed(_fit_and_score)(
                    clone(base_estimator),
                    XX,
                    yy,
                    train=train,
                    test=test,
                    parameters=parameters,
                    split_progress=(split_idx, n_splits),
                    candidate_progress=(cand_idx, n_candidates),
                    **fit_and_score_kwargs,
                )
                for (cand_idx, parameters), (split_idx, (train, test, XX, yy)) in product(
                    enumerate(candidate_params),
                    enumerate(cv.split(X, y, **routed_params.splitter.split)),
                )
            )

            if len(out) < 1:
                raise ValueError(
                    "No fits were performed. "
                    "Was the CV iterator empty? "
                    "Were there no candidates?"
                )
            elif len(out) != n_candidates * n_splits:
                raise ValueError(
                    "cv.split and cv.get_n_splits returned "
                    "inconsistent results. Expected {} "
                    "splits, got {}".format(n_splits, len(out) // n_candidates)
                )

            _warn_or_raise_about_fit_failures(out, self.error_score)

            # For callable self.scoring, the return type is only know after
            # calling. If the return type is a dictionary, the error scores
            # can now be inserted with the correct key. The type checking
            # of out will be done in `_insert_error_scores`.
            if callable(self.scoring):
                _insert_error_scores(out, self.error_score)

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

            if more_results is not None:
                for key, value in more_results.items():
                    all_more_results[key].extend(value)

            nonlocal results
            results = self._format_results(
                all_candidate_params, n_splits, all_out, all_more_results
            )

            return results

        self._run_search(evaluate_candidates)

        # multimetric is determined here because in the case of a callable
        # self.scoring the return type is only known after calling
        first_test_score = all_out[0]["test_scores"]
        self.multimetric_ = isinstance(first_test_score, dict)

        # check refit_metric now for a callabe scorer that is multimetric
        if callable(self.scoring) and self.multimetric_:
            self._check_refit_for_multimetric(first_test_score)
            refit_metric = self.refit

    # For multi-metric evaluation, store the best_index_, best_params_ and
    # best_score_ iff refit is one of the scorer names
    # In single metric evaluation, refit_metric is "score"
    if self.refit or not self.multimetric_:
        self.best_index_ = self._select_best_index(
            self.refit, refit_metric, results
        )
        if not callable(self.refit):
            # With a non-custom callable, we can select the best score
            # based on the best index
            self.best_score_ = results[f"mean_test_{refit_metric}"][
                self.best_index_
            ]
        self.best_params_ = results["params"][self.best_index_]

    if self.refit:
        # here we clone the estimator as well as the parameters, since
        # sometimes the parameters themselves might be estimators, e.g.
        # when we search over different estimators in a pipeline.
        # ref: https://github.com/scikit-learn/scikit-learn/pull/26786
        self.best_estimator_ = clone(base_estimator).set_params(
            **clone(self.best_params_, safe=False)
        )

        X, y = self.cv.pipe.fit_transform((X, y))
        refit_start_time = time.time()
        if y is not None:
            self.best_estimator_.fit(X, y, **routed_params.estimator.fit)
        else:
            self.best_estimator_.fit(X, **routed_params.estimator.fit)
        refit_end_time = time.time()
        self.refit_time_ = refit_end_time - refit_start_time

        if hasattr(self.best_estimator_, "feature_names_in_"):
            self.feature_names_in_ = self.best_estimator_.feature_names_in_

    # Store the only scorer not as a dict for single metric evaluation
    self.scorer_ = scorers

    self.cv_results_ = results
    self.n_splits_ = n_splits

    return self
