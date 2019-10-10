# -*- coding: utf-8 -*-
# pylint:disable=no-else-return
from __future__ import absolute_import
import numpy as np
from sklearn.ensemble.voting import _parallel_fit_estimator
from scipy.stats import zscore
from sklearn.calibration import CalibratedClassifierCV
from ml_insights import SplineCalibratedClassifierCV


class VotingClassifier:
    """Custom version of VotingClassifier that uses prefit estimators and that has support for
        - Probability calibration
        - Multiclass as I like it (prediction gives the label and not the index)
    https://gist.github.com/tomquisel/a421235422fdf6b51ec2ccc5e3dee1b4"""

    def __init__(self, estimators, voting="hard", weights=None):
        self._estimators = [e[1] for e in estimators]
        self.estimators = self._estimators
        self.named_estimators = dict(estimators)
        self.voting = voting
        self.weights = weights
        self.calibration = None
        self.calibrated = False
        self.refitted = False
        self.classes = None

    def _fit(self, X, y, sample_weight=None):
        """Important for randomization tests, refits each estimator"""
        self._estimators = [
            _parallel_fit_estimator(e, X, y, sample_weight) for e in self.estimators
        ]
        self.calibrated = False
        self.refitted = True
        self.classes = np.unique(y)

    def _calibrate_base_estimators(self, method, X, y):
        self.calibration = method
        self._check_is_fitted()
        self._estimators = [
            self._calibrate_model(model, method, X, y) for model in self._estimators
        ]
        self.calibrated = True

    def _calibrate_model(
        self, model, method: str, X_valid: np.array, y_valid: np.array
    ):
        if method == "isotonic":
            calibrated = CalibratedClassifierCV(model, cv="prefit", method="isotonic")
            calibrated.fit(X_valid, y_valid)
        elif method == "sigmoid":
            calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
            calibrated.fit(X_valid, y_valid)
        elif method == "none":
            calibrated = model
        elif method == "spline":
            calibrated = SplineCalibratedClassifierCV(model, cv="prefit")
            calibrated.fit(X_valid, y_valid)
        else:
            calibrated = CalibratedClassifierCV(model, cv="prefit", method="sigmoid")
            calibrated.fit(X_valid, y_valid)

        return calibrated

    def predict(self, X):
        """ Predict class labels for X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        maj : array-like, shape = [n_samples]
            Predicted class labels.
        """

        self._check_is_fitted()
        if self.voting == "soft":
            m = np.argmax(self.predict_proba(X), axis=1)
            if self.classes is not None:
                maj = self.classes[m]
            else:
                maj = m
        else:  # 'hard' voting
            predictions = self._predict(X)
            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions.astype("int"),
            )
        return maj

    def _voting_agreement(self, X):
        """
        Returns the average z-score
        """
        predictions = self._predict(X)
        return np.mean(zscore(predictions, axis=-1), axis=-1)

    def _collect_probas(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict_proba(X) for clf in self._estimators])

    def _check_is_fitted(self):
        for estimator in self._estimators:
            if not hasattr(estimator, "classes_"):
                raise ValueError("Classifier not fitted")

    def _predict_proba(self, X):
        """Predict class probabilities for X in 'soft' voting """
        self._check_is_fitted()
        if self.voting == "hard":
            raise AttributeError(
                "predict_proba is not available when" " voting=%r" % self.voting
            )
        avg = np.average(self._collect_probas(X), axis=0, weights=self.weights)
        return avg

    @property
    def predict_proba(self):
        """Compute probabilities of possible outcomes for samples in X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        ----------
        avg : array-like, shape = [n_samples, n_classes]
            Weighted average probability for each class per sample.
        """
        return self._predict_proba

    def transform(self, X):
        """Return class labels or probabilities for X for each estimator.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.
        Returns
        -------
        If `voting='soft'`:
          array-like = [n_classifiers, n_samples, n_classes]
            Class probabilities calculated by each classifier.
        If `voting='hard'`:
          array-like = [n_samples, n_classifiers]
            Class labels predicted by each classifier.
        """
        self._check_is_fitted()
        if self.voting == "soft":
            return self._collect_probas(X)
        else:
            return self._predict(X)

    def _predict(self, X):
        """Collect results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self._estimators]).T
