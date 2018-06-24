from ...pipeline import Pipeline, FeatureUnion, _apply_weight
from .transformer import StackableTransformer


def _fit_blend_one(transformer, X, y, weight, **fit_params):
    transformer = transformer.fit(X, y, **fit_params)
    res = transformer.blend(X)
    # if we have a weight for this transformer, multiply output
    return _apply_weight(res, weight), transformer


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
            transformer_list, n_jobs, transformer_weights)
        self.restack = restack
        self.cv = cv
        self.method = method
        self.n_cv_jobs = n_cv_jobs

    @property
    def _fit_transform_one(self):
        return _fit_blend_one

    def _wrap_one_transformer(self, transformer):
        return StackableTransformer(
            transformer, cv=self.cv, method=self.method, n_jobs=self.n_cv_jobs)

    def _iter(self):
        original_iter = super(StackingLayer, self)._iter()
        return ((name, self._wrap_one_transformer(trans), weight)
                for name, trans, weight in original_iter
                if trans is not None)
