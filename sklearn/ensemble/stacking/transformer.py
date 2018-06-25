"""Stacked ensembles"""

# Author: Caio Oliveira <caioaao@gmail.com>
# License: BSD 3 clause

from ...base import (BaseEstimator, TransformerMixin, MetaEstimatorMixin, clone)
from ...model_selection import cross_val_predict


class StackableTransformer(BaseEstimator, MetaEstimatorMixin,
                           TransformerMixin):
    """Transformer to turn estimators into meta-estimators for model stacking

    In stacked generalization, meta estimators are combined in layers to
    improve the final result. To prevent data leaks between layers, a procedure
    similar to cross validation is adopted, where the model is trained in one
    part of the set and predicts the other part. In ``StackableTransformer``,
    it happens during ``blend``, as the result of this procedure is what should
    be used by the next layers. Read more in the :ref:`User Guide
    <stacking_transformer>`.

    Parameters
    ----------
    estimator : predictor
        The estimator to be blended.

    cv : int, cross-validation generator or an iterable, optional (default=3)
        Determines the cross-validation splitting strategy to be used for
        generating features to train the next layer on the stacked ensemble or,
        more specifically, during ``blend``.

        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass,
        :class:`sklearn.model_selection.StratifiedKFold` is used. In all
        other cases, :class:`sklearn.model_selection.KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    method : string, optional (default='auto')
        This method will be called on the estimator to produce the output of
        transform. If the method is ``auto``, will try to invoke, for each
        estimator, ``predict_proba``, ``decision_function`` or ``predict``
        in that order.

    n_jobs : int, optional (default=1)
        Number of jobs to be passed to ``cross_val_predict`` during
        ``blend``.

    """
    def __init__(self, estimator, cv=3, method='auto', n_jobs=1):
        self.estimator = estimator
        self.cv = cv
        self.method = method
        self.n_jobs = n_jobs

    def fit(self, X, y=None, **fit_params):
        """Fit the estimator.

        This should only be used in special situations. Read more in the
        :ref:`User Guide <stacking_transformer>`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        **fit_params : parameters to be passed to the base estimator.

        Returns
        -------
        self : object

        """
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X, y, **fit_params)
        return self

    @property
    def _estimator_function_name(self):
        if self.method == 'auto':
            if getattr(self.estimator_, 'predict_proba', None):
                method = 'predict_proba'
            elif getattr(self.estimator_, 'decision_function', None):
                method = 'decision_function'
            else:
                method = 'predict'
        else:
            method = self.method

        return method

    @property
    def _estimator_function(self):
        return getattr(self.estimator_, self._estimator_function_name)

    def transform(self, *args, **kwargs):
        """Transform dataset.

        Note that, unlike ``fit_transform()``, this won't return the cross
        validation predictions. Read more in the
        :ref:`User Guide <stacking_transformer>`.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Input data to be transformed. Use ``dtype=np.float32`` for maximum
            efficiency. Sparse matrices are also supported, use sparse
            ``csr_matrix`` for maximum efficiency.

        Returns
        -------
        X_transformed : sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.

        """
        preds = self._estimator_function(*args, **kwargs)

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds

    def fit_transform(self, X, y=None, **fit_params):
        """Fit estimator and transform dataset.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Input data used to build forests. Use ``dtype=np.float32`` for
            maximum efficiency.

        y : array-like, shape = [n_samples]
            Target values.

        **fit_params : parameters to be passed to the base estimator.

        Returns
        -------
        X_transformed : sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.
        """
        return self.fit(X, y, **fit_params).transform(X)

    def blend(self, X, y, **fit_params):
        """Transform dataset using cross validation.

        Read more in the :ref:`User Guide <stacking_transformer>`.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Input data used to build forests. Use ``dtype=np.float32`` for
            maximum efficiency.

        y : array-like, shape = [n_samples]
            Target values.

        **fit_params : parameters to be passed to the base estimator.

        Returns
        -------
        X_transformed : sparse matrix, shape=(n_samples, n_out)
            Transformed dataset.

        """
        self.estimator_ = clone(self.estimator)
        preds = cross_val_predict(self.estimator_, X, y, cv=self.cv,
                                  method=self._estimator_function_name,
                                  n_jobs=self.n_jobs, fit_params=fit_params)
        self.estimator_ = None

        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return preds
