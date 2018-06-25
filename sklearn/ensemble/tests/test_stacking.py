"""
Testing for the stacking ensemble module (sklearn.ensemble.stacking).
"""

# Author: Caio Oliveira
# License BSD 3 clause


from copy import deepcopy
import numpy as np
from scipy.sparse import csr_matrix

from sklearn.utils.mocking import CheckingClassifier
from sklearn.utils.testing import (assert_equal, assert_array_equal,
                                   assert_false)
from sklearn.utils.testing import SkipTest
from sklearn.ensemble import (StackableTransformer, StackingPipeline,
                              StackingLayer, make_stack_layer)
from sklearn.linear_model import (RidgeClassifier, LinearRegression)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, LinearSVR, SVC
from sklearn import datasets
from sklearn.base import clone
from sklearn.model_selection import (ParameterGrid, StratifiedKFold)

iris = datasets.load_iris()
X, y = iris.data[:, 1:3], iris.target

RANDOM_SEED = 8939

META_ESTIMATOR_PARAMS = {'cv': [2, StratifiedKFold()],
                         'n_jobs': [1, 2]}
META_ESTIMATOR_FIT_PARAMS = [{}, {"sample_weight": np.ones(y.shape)}]


def _check_estimator(estimator, **fit_params):
    # checks that we can fit_transform to the data
    Xt = estimator.fit_transform(X, y, **fit_params)

    # checks that we get a column vector
    assert_equal(Xt.ndim, 2)

    # checks that `fit` is available
    estimator.fit(X, y, **fit_params)

    # checks that we can transform the data after it's fitted
    Xt2 = estimator.transform(X)

    # checks that transformed data is always a column vector
    assert_equal(Xt.ndim, 2)

    # checks that transform is equal to fit_transform
    assert_array_equal(Xt, Xt2)

    # checks for determinism: every `transform` should yield the same result
    for i in range(10):
        assert_array_equal(Xt2, estimator.transform(X))


def test_regression():
    # tests regression with various parameter settings

    meta_params = {'method': ['auto', 'predict']}
    meta_params.update(META_ESTIMATOR_PARAMS)

    regressors = [LinearRegression(), LinearSVR(random_state=RANDOM_SEED)]

    for reg in regressors:
        for params in ParameterGrid(meta_params):
            blended_reg = StackableTransformer(reg, **params)
            for fit_params in META_ESTIMATOR_FIT_PARAMS:
                _check_estimator(blended_reg, **fit_params)


def test_transformer_from_classification():
    # tests classification with various parameter settings

    testcases = [{'clf': RandomForestClassifier(random_state=RANDOM_SEED),
                  'extra_params': {'method': ['auto', 'predict',
                                              'predict_proba']}},
                 {'clf': LinearSVC(random_state=RANDOM_SEED),
                  'extra_params': {'method': ['auto', 'predict',
                                              'decision_function']}},
                 {'clf': RidgeClassifier(random_state=RANDOM_SEED),
                  'extra_params': {'method': ['auto', 'predict']}}]

    for testcase in testcases:
        clf = testcase['clf']

        meta_params = deepcopy(testcase['extra_params'])
        meta_params.update(META_ESTIMATOR_PARAMS)

        for params in ParameterGrid(meta_params):
            blended_clf = StackableTransformer(clf, **params)
            for fit_params in META_ESTIMATOR_FIT_PARAMS:
                _check_estimator(blended_clf, **fit_params)


def test_multi_output_classification():
    raise SkipTest("Test is broken while #8773 is not fixed")
    clf_base = RandomForestClassifier(random_state=RANDOM_SEED)
    clf = StackableTransformer(clf_base, method='predict_proba')
    X, y = datasets.make_multilabel_classification()
    clf.fit_transform(X[:-10], y[:-10])


def _check_restack(X, Xorig):
    # checks that original data is appended to the rest of the features
    assert_array_equal(Xorig, X[:, -Xorig.shape[1]:])


def _check_layer(l, restack):
    l_ = clone(l)

    # check that we can fit_transform the data
    Xt = l_.fit_transform(X, y)
    if restack:
        _check_restack(Xt, X)

    # check that we can transform the data
    Xt = l_.transform(X)
    if restack:
        _check_restack(Xt, X)

    # check that `fit` is accessible
    l_ = clone(l)
    l_.fit(X, y)

    # check that we can blend the data
    Xt = l_.blend(X, y)
    if restack:
        _check_restack(Xt, X)

    # check that `fit_blend` is accessible
    l_ = clone(l)
    Xt = l_.fit_blend(X, y)
    if restack:
        _check_restack(Xt, X)

    # check that `fit_blend` fits the layer
    l_ = clone(l)
    Xt0 = l_.fit_blend(X, y)

    Xt = l_.blend(X, y)
    if restack:
        _check_restack(Xt, X)

    # check results match
    assert_array_equal(Xt0, Xt)


STACK_LAYER_PARAMS = {'n_jobs': [1, 2]}


def test_layer_regression():
    base_regs = [
        ('lr', StackableTransformer(
            LinearRegression())),
        ('svr', StackableTransformer(
            LinearSVR(random_state=RANDOM_SEED)))]

    for params in ParameterGrid(STACK_LAYER_PARAMS):
        # assert constructor
        reg_layer = StackingLayer(base_regs, **params)
        _check_layer(reg_layer, False)


def test_layer_classification():
    base_clfs = [
        ('rf1', StackableTransformer(RandomForestClassifier(
            random_state=RANDOM_SEED, criterion='gini'))),
        ('rf2', StackableTransformer(RandomForestClassifier(
            random_state=RANDOM_SEED, criterion='entropy')))]

    for params in ParameterGrid(STACK_LAYER_PARAMS):
        # assert constructor
        clf_layer = StackingLayer(base_clfs, **params)
        _check_layer(clf_layer, False)


STACK_LAYER_FULL_PARAMS = {'cv': [3, StratifiedKFold()],
                           'restack': [False, True],
                           'method': ['auto', 'predict', 'predict_proba'],
                           'n_jobs': [1, 2],
                           'n_cv_jobs': [1, 2]}


def test_layer_helper_constructor():
    base_estimators = [LinearRegression(), LinearRegression()]
    for params in ParameterGrid(STACK_LAYER_FULL_PARAMS):
        if params['n_jobs'] != 1 and params['n_cv_jobs'] != 1:
            continue  # nested parallelism is not supported

        if params['method'] is 'predict_proba':
            continue

        reg_layer = make_stack_layer(*base_estimators, **params)
        _check_layer(reg_layer, params["restack"])


# def test_method_selection():
#     clf = SVC()
#     X = np.asarray([[1, 2], [1, 2], [1, 2], [1, 2]])
#     y = np.asarray([1, 0, 1, 0])
#     clf_T = StackableTransformer(clf, cv=2, method='auto')
#
#     # asserts that fit results are taken into consideration when choosing
#     # method name
#     clf_T.set_params(estimator__probability=False)
#     assert(not hasattr(clf_T.estimator, 'predict_proba'))
#     Xt1 = clf_T.fit_transform(X, y)
#     assert_equal(clf_T._method_name(), "decision_function")
#
#     clf_T.set_params(estimator__probability=True)
#     Xt2 = clf_T.fit_transform(X, y)
#     assert_equal(clf_T._method_name(), "predict_proba")
#
#     # asserts that cross_val_predict is called with different methods for each
#     # case
#     assert_false(np.allclose(Xt1, Xt2))
#
#
# def test_pipeline_consistency():
#     datasets = [{'X': np.asarray([[1, 2], [1, 2], [1, 2], [1, 2]]),
#                  'y': np.asarray([1, 0, 1, 0])},
#                 {'X': csr_matrix(([1, 2, 1, 2, 1, 2, 1, 2],
#                                   ([0, 0, 1, 1, 2, 2, 3, 3],
#                                    [0, 1, 0, 1, 0, 1, 0, 1])),
#                                  shape=(4, 2)),
#                  'y': np.asarray([1, 0, 1, 0])}]
#     try:
#         from pandas import DataFrame, Series
#         datasets.append({'X': DataFrame({'col1': [1, 1, 1, 1],
#                                          'col2': [2, 2, 2, 2]}),
#                          'y': Series([1, 0, 1, 0])})
#     except ImportError:
#         pass
#
#     # checks that estimator receives input of the same type as it was passed to
#     # StackableTransformer
#     for data in datasets:
#         X_class = type(data['X'])
#         clf = CheckingClassifier(check_X=lambda X: isinstance(X, X_class))
#         clf_T = StackableTransformer(clf, cv=2)
#         clf_T.fit_transform(data['X'], data['y'])
