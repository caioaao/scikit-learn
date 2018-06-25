import numpy as np

from ...pipeline import Pipeline, FeatureUnion, _apply_weight
from ...externals.joblib import Parallel, delayed
from .transformer import StackableTransformer


def _blend_one(transformer, X, y, weight, **fit_params):
    res = transformer.blend(X, y, **fit_params)
    return _apply_weight(res, weight)


def _fit_blend_one(transformer, X, y, weight, **fit_params):
    Xt = _blend_one(transformer, X, y, weight, **fit_params)
    return Xt, transformer.fit(X, y, **fit_params)


class StackingPipeline(Pipeline):
    def __init__(self, steps, memory=None):
        super(StackingPipeline, self).__init__(steps, memory)

    @property
    def _fit_transform_one(self):
        return _fit_blend_one


class StackingLayer(FeatureUnion):
    def __init__(self, transformer_list, n_jobs=1, transformer_weights=None,
                 restack=False, cv=3, method='auto', n_cv_jobs=1):
        super(StackingLayer, self).__init__(
            transformer_list, n_jobs=n_jobs,
            transformer_weights=transformer_weights)
        self.restack = restack
        self.cv = cv
        self.method = method
        self.n_cv_jobs = n_cv_jobs

    def _validate_one_transformer(self, t):
        if not hasattr(t, "blend"):
            raise TypeError("All transformers should implement 'blend'."
                            " '%s' (type %s) doesn't" %
                            (t, type(t)))
        return super(StackingLayer, self)._validate_one_transformer(t)

    @property
    def _blend_one(self):
        return _blend_one

    @property
    def _fit_blend_one(self):
        return _fit_blend_one

    def blend(self, X, y, weight=None, **fit_params):
        self._validate_transformers()
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(self._blend_one)(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter())

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        return self._stack_results(Xs)

    def fit_blend(self, X, y, weight=None, **fit_params):
        self._validate_transformers()

        result = Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_blend_one)(trans, X, y, weight, **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))
        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        return self._stack_results(Xs)
