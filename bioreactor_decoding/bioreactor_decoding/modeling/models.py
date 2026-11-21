"""Estimators and the challenge metric.

All estimators follow the scikit-learn API, so ``sklearn.base.clone`` gives an unfitted copy
per CV fold. Anything that learns from data (including any preprocessing passed in as a
``Pipeline``) is therefore refit on the training fold only.

Targets are modelled as log fold change relative to t = 0, z = log(y_t) - log(y_0):
  * a stateless, per-row normalisation (needs only that batch's x(0), so it cannot leak),
  * it removes between-batch scale differences, and
  * predictions exp(z_hat) * y_0 are positive by construction.
Squared error on log values approximates relative error, which suits MAPE.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

from bioreactor_decoding.dataset import TARGETS


def mape(y_true, y_pred) -> float:
    """Standard MAPE in percent: mean(|y - y_hat| / |y|) * 100."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100)


def challenge_score(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict[str, float]:
    """Per-target MAPE plus the final score (unweighted mean of the 4 MAPEs)."""
    scores = {c: mape(y_true[c], y_pred[c]) for c in TARGETS}
    scores["score"] = float(np.mean([scores[c] for c in TARGETS]))
    return scores


def _log_fold_change(X: pd.DataFrame, Y: pd.DataFrame, target: str) -> np.ndarray:
    return np.log(Y[target].to_numpy()) - X[f"log0_{target}"].to_numpy()


class PerTargetRegressor(BaseEstimator, RegressorMixin):
    """One clone of ``estimator`` per target, fitted directly on the raw target values.

    No target transform and no normalisation: the plain "straight-up" setup.
    """

    def __init__(self, estimator=None, targets=tuple(TARGETS)):
        self.estimator = estimator
        self.targets = targets

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.models_ = {c: clone(self.estimator).fit(X, Y[c]) for c in self.targets}
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({c: m.predict(X) for c, m in self.models_.items()}, index=X.index)


class FoldChangeRegressor(BaseEstimator, RegressorMixin):
    """One clone of ``estimator`` per target, fitted on log fold change vs. t = 0."""

    def __init__(self, estimator=None, targets=tuple(TARGETS)):
        self.estimator = estimator
        self.targets = targets

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.models_ = {
            c: clone(self.estimator).fit(X, _log_fold_change(X, Y, c)) for c in self.targets
        }
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            {c: np.exp(m.predict(X) + X[f"log0_{c}"].to_numpy()) for c, m in self.models_.items()},
            index=X.index,
        )


def sampling_day(X: pd.DataFrame) -> np.ndarray:
    """Sampling day index: elapsed time rounded to the nearest whole day (samples are ~daily)."""
    return np.rint(X["t"].to_numpy()).astype(int)


class PerDayNullModel(BaseEstimator, RegressorMixin):
    """Per-day null model: for each sampling day, the mean of each target across training batches.

    A step function of time. Ignores x(0) and feeds, so it is the shared "process recipe" curve.
    """

    def __init__(self, targets=tuple(TARGETS)):
        self.targets = targets

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.means_ = Y[list(self.targets)].groupby(sampling_day(X)).mean()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        day = np.clip(sampling_day(X), self.means_.index.min(), self.means_.index.max())
        return pd.DataFrame(
            self.means_.reindex(day).to_numpy(), columns=list(self.targets), index=X.index
        )


class ConstantNullModel(BaseEstimator, RegressorMixin):
    """Constant null model: one constant per target, the mean over all times and training batches.

    Ignores time, x(0) and feeds entirely, so it is the floor any model must clear.
    """

    def __init__(self, targets=tuple(TARGETS)):
        self.targets = targets

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.means_ = Y[list(self.targets)].mean()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(
            np.tile(self.means_.to_numpy(), (len(X), 1)),
            columns=list(self.targets),
            index=X.index,
        )


class MedianFoldChange(BaseEstimator, RegressorMixin):
    """Baseline: per sample index k, the median log fold change in the training fold, times x(0)."""

    def __init__(self, targets=tuple(TARGETS)):
        self.targets = targets

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        z = pd.DataFrame({c: _log_fold_change(X, Y, c) for c in self.targets}, index=X.index)
        self.medians_ = z.groupby(X["k"].to_numpy()).median()
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        k = X["k"].clip(self.medians_.index.min(), self.medians_.index.max()).to_numpy()
        z = self.medians_.reindex(k).to_numpy()
        log0 = X[[f"log0_{c}" for c in self.targets]].to_numpy()
        return pd.DataFrame(np.exp(z + log0), columns=list(self.targets), index=X.index)


class AverageEnsemble(BaseEstimator, RegressorMixin):
    """Average of the members' predictions, each weighted equally.

    Every member is cloned and fitted on the same training data, so the ensemble is itself a
    plain estimator: CV, holdout_check and submissions treat it like any other model.
    """

    def __init__(self, estimators=()):
        self.estimators = estimators

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        self.estimators_ = [clone(e).fit(X, Y) for e in self.estimators]
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        preds = [e.predict(X) for e in self.estimators_]
        return sum(preds[1:], preds[0]) / len(preds)
