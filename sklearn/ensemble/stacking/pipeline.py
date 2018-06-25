import numpy as np

from ...pipeline import Pipeline, FeatureUnion, _apply_weight, _name_estimators
from ...externals.joblib import Parallel, delayed
from ...preprocessing import FunctionTransformer
from .transformer import StackableTransformer


def _blend_one(transformer, X, y, weight, **fit_params):
    res = transformer.blend(X, y, **fit_params)
    return _apply_weight(res, weight)


def _fit_blend_one(transformer, X, y, weight, **fit_params):
    Xt = _blend_one(transformer, X, y, weight, **fit_params)
    return Xt, transformer.fit(X, y, **fit_params)


class StackingLayer(FeatureUnion):
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


class StackingPipeline(Pipeline):
    def __init__(self, steps, memory=None):
        super(StackingPipeline, self).__init__(steps, memory)

    @property
    def _fit_transform_one(self):
        return _fit_blend_one

    def _validate_steps(self):
        super(StackingPipeline, self)._validate_steps()
        names, estimators = zip(*self.steps)

        # validate estimators
        transformers = estimators[:-1]

        for t in transformers:
            if t is None:
                continue
            if hasattr(t, "blend"):
                raise TypeError("All intermediate steps should be "
                                "transformers and implement blend."
                                " '%s' (type %s) doesn't" % (t, type(t)))


def _identity(x):
    return x


def _identity_transformer():
    """Contructs a transformer that returns its input unchanged"""
    return FunctionTransformer(_identity, accept_sparse=True)


def _wrap_estimators(estimators, cv=3, method='auto', n_cv_jobs=1):
    return [(name, StackableTransformer(
        est, cv=cv, method=method, n_jobs=n_cv_jobs))
            for name, est in estimators]


def make_stack_layer(*transformers, n_jobs=1, cv=3, method='auto',
                     n_cv_jobs=1, restack=False):

    named_transformers = _name_estimators(transformers)

    transformer_list = _wrap_estimators(
        named_transformers, cv=cv, method=method, n_cv_jobs=n_cv_jobs)
    if restack:
        transformer_list.append(
            ('identity-transformer', StackableTransformer(
                _identity_transformer(), cv=None, method='transform',
                n_jobs=1)))

    return StackingLayer(transformer_list, n_jobs=n_jobs)
