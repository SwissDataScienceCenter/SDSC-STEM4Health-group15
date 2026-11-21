"""ML correction on top of a base prediction (hybrid model).

    y_hat = base(t) * exp( g(features) )        per state (vcd, tcd, substrate, byproduct)

- base: the mechanistic model, or the median training curve.
- g:    learned on the training batches; target = log(y_obs) - log(y_base), i.e. the relative
        error of the base (consistent with MAPE). g = 0 means "trust the base".
- features (all known for a test batch before the run): time, day-0 values, cumulative feed
  volumes up to t (feeds strictly before the sample), reactor volume at t, base prediction.

Methods
- "ridge_day": one ridge regression per state and per day (alpha by inner CV). Linear, so the
  coefficients are interpretable ("+1 L feed B by day t -> +x % VCD").
- "gbr":       gradient-boosted shallow trees per state across all days (non-linear).
Corrections are clipped to +-MAX_CORR in log space (~ +-40 %) to avoid wild extrapolation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FEEDS, STATES

EPS = 1e-3
MAX_CORR = np.log(1.4)


# ---------------------------------------------------------------- base predictions
def median_curve_preds(df, train_ids, arr):
    """Median training curve per (rounded) day, at the sample times of arr."""
    m = df[df.batch_id.isin(train_ids) & df.vcd.notna()].copy()
    m["day"] = m.elapsed_time.round().astype(int)
    med = m.groupby("day")[STATES].median()
    return [med.reindex(np.round(t).astype(int)).values for t in arr.obs_t]


# ---------------------------------------------------------------- features
def feature_table(df, arr, base_preds):
    """One row per (batch, sample time): features, base prediction, observation (NaN for test)."""
    rows = []
    for j, b in enumerate(arr.ids):
        g = df[df.batch_id == b]
        d0 = arr.x0[j]
        vol = g[g.volume.notna()]
        for k, t in enumerate(arr.obs_t[j]):
            before = g[g.elapsed_time < t - 1e-9]
            r = {"batch_id": b, "t": t, "day": int(round(t))}
            r.update({f"d0_{s}": d0[i] for i, s in enumerate(STATES)})
            r.update({f"cum_{c}": before[c].sum() for c in FEEDS})
            r["cum_feed_total"] = sum(r[f"cum_{c}"] for c in FEEDS)
            last = before[before.elapsed_time >= t - 1.0]          # feeds in the last day
            r["lastday_feed_b"] = last["feed_b"].sum()
            r["lastday_feed_substrate"] = last["feed_substrate"].sum()
            r["volume_t"] = np.interp(t, vol.elapsed_time.values, vol.volume.values)
            r.update({f"base_{s}": base_preds[j][k, i] for i, s in enumerate(STATES)})
            r.update({f"obs_{s}": arr.obs_y[j][k, i] for i, s in enumerate(STATES)})
            rows.append(r)
    return pd.DataFrame(rows)


def feature_columns(state, method):
    cols = [f"d0_{s}" for s in STATES] + [f"cum_{c}" for c in FEEDS] + \
           ["cum_feed_total", "lastday_feed_b", "lastday_feed_substrate", "volume_t"] + \
           [f"logbase_{s}" for s in STATES]
    if method == "gbr":
        cols = ["t"] + cols
    return cols


def _prep(F):
    F = F.copy()
    for s in STATES:
        F[f"logbase_{s}"] = np.log(np.maximum(F[f"base_{s}"], 0) + EPS)
    return F


# ---------------------------------------------------------------- corrector
class ResidualCorrector:
    def __init__(self, method="ridge_day", alphas=np.logspace(-1, 4, 26), gbr_kw=None):
        self.method = method
        self.alphas = alphas
        self.gbr_kw = dict(n_estimators=300, learning_rate=0.03, max_depth=2,
                           min_samples_leaf=10, subsample=0.8, random_state=0)
        self.gbr_kw.update(gbr_kw or {})
        self.models = {}

    @staticmethod
    def _target(F, s):
        return np.log(F[f"obs_{s}"] + EPS) - np.log(np.maximum(F[f"base_{s}"], 0) + EPS)

    def fit(self, F):
        F = _prep(F)
        for s in STATES:
            cols = feature_columns(s, self.method)
            y = self._target(F, s)
            if self.method == "ridge_day":
                for d, Fd in F.groupby("day"):
                    m = make_pipeline(StandardScaler(), RidgeCV(alphas=self.alphas))
                    m.fit(Fd[cols].values, y.loc[Fd.index].values)
                    self.models[(s, d)] = m
            elif self.method == "gbr":
                m = GradientBoostingRegressor(**self.gbr_kw)
                m.fit(F[cols].values, y.values)
                self.models[s] = m
            else:
                raise ValueError(self.method)
        return self

    def correction(self, F):
        """Log-correction per row and state, DataFrame with columns STATES."""
        F = _prep(F)
        out = pd.DataFrame(0.0, index=F.index, columns=STATES)
        for s in STATES:
            cols = feature_columns(s, self.method)
            if self.method == "ridge_day":
                for d, Fd in F.groupby("day"):
                    m = self.models.get((s, d))
                    if m is None:   # day not seen in training: use nearest day
                        days = np.array([k[1] for k in self.models if k[0] == s])
                        m = self.models[(s, int(days[np.argmin(np.abs(days - d))]))]
                    out.loc[Fd.index, s] = m.predict(Fd[cols].values)
            else:
                out[s] = self.models[s].predict(F[cols].values)
        return out.clip(-MAX_CORR, MAX_CORR)

    def predict(self, F):
        """Corrected predictions, DataFrame with columns STATES (same row order as F)."""
        c = self.correction(F)
        base = F[[f"base_{s}" for s in STATES]].values
        return pd.DataFrame(np.maximum(base, 0) * np.exp(c.values), index=F.index, columns=STATES)

    def ridge_coefficients(self, state):
        """Standardised ridge coefficients per day (rows) for one state -> interpretation."""
        assert self.method == "ridge_day"
        cols = feature_columns(state, self.method)
        rows = {d: m[-1].coef_ for (s, d), m in self.models.items() if s == state}
        return pd.DataFrame(rows, index=cols).T.sort_index()


def to_list(F, P, arr):
    """DataFrame predictions (rows of a feature_table) -> list per batch, aligned with arr."""
    return [P.loc[F.batch_id == b, STATES].values for b in arr.ids]
