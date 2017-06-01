"""Stacking API"""

# Author: Caio Oliveira <caioaao@gmail.com>
# License: BSD 3 clause

from ..base import (BaseEstimator, TransformerMixin, MetaEstimatorMixin)
from ..model_selection import cross_val_predict
from ..pipeline import (make_union, make_pipeline)
from ..preprocessing import FunctionTransformer


class BlendedEstimator(BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    """Transformer to turn estimators into blended estimators

    This is used for stacking models. Blending an estimator prevents data leaks
    between the model layers. Blending will happen only when calling
    `fit_transform`, as it's the only stage where this makes sense.

    Parameters
    ----------
    base_estimator: the estimator to be blended.

    cv: cv to be used, optional, default: 3
        Will be passed to `cross_val_predict` during `fit_transform`.

    method: string, optional, default: 'auto'
        Invokes the passed method name of the passed estimator. If the method
        is `auto`, will try to invoke `predict_proba` or `predict` in that
        order.

    n_jobs: int, optional, default: 1
        Number of jobs to be passed to `cross_val_predict` during
        `fit_transform`.
    """
    def __init__(self, base_estimator, cv=3, method='auto', n_jobs=1):
        self.base_estimator = base_estimator
        self.cv = cv
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, *args, **kwargs):
        self.base_estimator = self.base_estimator.fit(*args, **kwargs)
        return self

    def _method_name(self):
        if self.method == 'auto':
            if getattr(self.base_estimator, 'predict_proba', None):
                method = 'predict_proba'
            else:
                method = 'predict'
        else:
            method = self.method

        return method

    def transform(self, *args, **kwargs):
        t = getattr(self.base_estimator, self._method_name())
        preds = t(*args, **kwargs)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds

    def fit_transform(self, X, y, **fit_params):
        preds = cross_val_predict(self.base_estimator, X, y, cv=self.cv,
                                  method=self._method_name(),
                                  n_jobs=self.n_jobs, fit_params=fit_params)

        self.base_estimator.fit(X, y, **fit_params)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds


def _identity_transformer():
    """Contructs a transformer that does nothing"""
    return FunctionTransformer(lambda x: x)

def make_stack_layer(*base_estimators, **kwargs):
    """Construct a single layer for a stacked model.

    This is a wrapper around pipelines to provide a more convenient API for
    stacking models.

    Parameters
    ----------
    *base_estimators: list of base estimators.

    restacking: optional, bool, default: False
        When true, the transformer will return the input.

    **kwargs: Keyword arguments to be passed to `BlendedEstimator`.

    Returns
    -------
    f : FeatureUnion with every base estimator wrapped in a
         `BlendedEstimator`.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import make_stack_layer
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.svm import SVC
    >>> t = make_stack_layer(KNeighborsClassifier(n_neighbors=3), SVC())
    >>> X = np.array([[1, 3], [.12, 1], [.5, -2], [1, -1], [-2, .1], [7, -84]])
    >>> y = np.array([1, 0, 0, 1, 0, 1])
    >>> t.fit_transform(X, y) # doctest: +NORMALIZE_WHITESPACE
    array([[ 0.66666667,  0.33333333,  0.        ],
           [ 0.66666667,  0.33333333,  1.        ],
           [ 0.66666667,  0.33333333,  1.        ],
           [ 0.66666667,  0.33333333,  0.        ],
           [ 0.66666667,  0.33333333,  0.        ],
           [ 0.66666667,  0.33333333,  0.        ]])
    """
    restacking = kwargs.pop('restacking', False)
    estimators = [BlendedEstimator(estimator, **kwargs)
                  for estimator in base_estimators]
    if restacking:
        estimators.append(_identity_transformer())
    return make_union(*estimators)


def stack_estimators(estimators_matrix, meta_estimator, **kwargs):
    """Construct a stacked estimator

    This is a wrapper around pipelines to provide a more convenient API for
    stacking models. Estimators in `estimators_matrix` are wrapped in
    `BlendedEstimator`.

    Parameters
    ----------
    estimators_matrix: 2D matrix with base estimators. Each row will be
        turned into a layer in the stack.

    meta_estimator: Estimator that will stay on top of the stack.

    **kwargs: Keyword arguments to be passed to `make_stacked_layer`.

    Returns
    -------
    p: Pipeline

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import stack_estimators
    >>> from sklearn.neighbors import KNeighborsClassifier
    >>> from sklearn.svm import SVC
    >>> from sklearn.linear_model import LogisticRegression
    >>> eclf = stack_estimators([[KNeighborsClassifier(n_neighbors=2), SVC()],
    ...                          [KNeighborsClassifier(n_neighbors=3), SVC()]],
    ...                         LogisticRegression())
    >>> X = np.array([[1, 3], [.12, 1], [.5, -2], [1, -1], [-2, .1], [7, -84]])
    >>> y = np.array([1, 0, 0, 1, 0, 1])
    >>> eclf.fit(X, y).predict(X)
    array([0, 1, 1, 0, 1, 0])
    """
    estimators = [make_stack_layer(*row, **kwargs)
                  for row in estimators_matrix]
    estimators.append(meta_estimator)

    return make_pipeline(*estimators)
