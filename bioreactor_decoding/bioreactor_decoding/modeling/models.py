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
