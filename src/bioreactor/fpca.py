"""Whole-curve prediction with functional PCA + ridge ("FPCA-ridge").

Instead of one model per day, each batch's curve (per state) is described by a few
principal-component scores, and only those scores are predicted from batch-level inputs:

    z_b(k)   = log y_b(k) - log base_b(k)                 k = 1..13 (sample index, ~ day)
    z_b      ~ mean(k) + sum_j  score_bj * PC_j(k)         PCA over the training batches
    score_bj ~ ridge(batch inputs)                          one RidgeCV per state (multi-output)
    y_hat    = base * exp(mean + sum_j score_hat_j * PC_j)

- base: median training curve (then the PCs describe how batches differ from the typical
  curve, e.g. "peak higher" / "peak later") or the mechanistic model (PCs describe its
  systematic errors).
- Sharing one curve model across all days smooths out day-to-day noise and needs far fewer
  parameters than 13 per-day models (n_components x n_inputs instead of 13 x n_inputs).
- joint=True: one PCA over the four states concatenated (each state standardised), so
  states that move together (VCD/TCD, byproduct/feed C ...) share components.
Curves are aligned by sample index k (sampling times jitter by ~1 h only).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .data import FEEDS, STATES

EPS = 1e-3
N_T = 13                       # samples after day 0
PHASES = [(0, 4), (4, 8), (8, 14)]   # day ranges for phase-wise feed totals


# ---------------------------------------------------------------- batch-level inputs
def batch_inputs(df, ids):
    """One row per batch: everything known before / during the run for a test batch."""
    rows = []
    for b in ids:
        g = df[df.batch_id == b]
        x0 = g[g.elapsed_time == g.elapsed_time.min()].iloc[0]
        r = {"batch_id": b}
        for s in STATES:
            r[f"log0_{s}"] = np.log(x0[s])
        r["viab0"] = x0["vcd"] / x0["tcd"]
        r["v0"] = x0["volume"]
        for c in FEEDS:
            ev = g[g[c].notna()]
            r[f"tot_{c}"] = ev[c].sum()
            for lo, hi in PHASES:
                r[f"{c}_d{lo}_{hi}"] = ev.loc[(ev.elapsed_time >= lo) & (ev.elapsed_time < hi), c].sum()
        fc = g[g.feed_c.notna()]
        r["n_feed_c"] = len(fc)
        for lo, hi in PHASES:
            r[f"n_feed_c_d{lo}_{hi}"] = int(((fc.elapsed_time >= lo) & (fc.elapsed_time < hi)).sum())
        fs = g[g.feed_substrate.notna()]
        r["n_feed_substrate"] = len(fs)
        r["t_first_feed_substrate"] = fs.elapsed_time.min() if len(fs) else 13.5
        rows.append(r)
    X = pd.DataFrame(rows).set_index("batch_id")
    # drop inputs that are constant (e.g. phase sums that are always 0)
    return X.loc[:, X.std() > 0]


# ---------------------------------------------------------------- curves
def log_ratio_curves(arr, base_preds):
    """dict state -> (n_batches, 13) array of log(obs / base), ordered by sample time."""
    Z = {s: np.zeros((len(arr.ids), N_T)) for s in STATES}
    for j, (t, y, yb) in enumerate(zip(arr.obs_t, arr.obs_y, base_preds)):
        if len(t) != N_T:
            raise ValueError(f"{arr.ids[j]}: {len(t)} samples, expected {N_T}")
        o = np.argsort(t)
        for i, s in enumerate(STATES):
            Z[s][j] = np.log(y[o, i] + EPS) - np.log(np.maximum(yb[o, i], 0) + EPS)
    return Z


def to_pred_list(arr, base_preds, Zhat):
    """Back-transform: list per batch of (13, 4) predictions in arr's sample order."""
    out = []
    for j, (t, yb) in enumerate(zip(arr.obs_t, base_preds)):
        o = np.argsort(t)
        P = np.empty((len(t), len(STATES)))
        for i, s in enumerate(STATES):
            P[o, i] = np.maximum(yb[o, i], 0) * np.exp(Zhat[s][j])
        out.append(P)
    return out


# ---------------------------------------------------------------- model
class FPCARidge:
    """PCA of the (log-ratio) curves + ridge from batch inputs to the PC scores.

    n_components: number of PCs per state (or for the joint PCA)
    joint:        one PCA over all four states (each standardised), else one per state
    """

    def __init__(self, n_components=2, joint=False, alphas=np.logspace(-2, 4, 31)):
        self.n_components, self.joint, self.alphas = n_components, joint, alphas

    # -- PCA helpers (plain SVD, centred)
    @staticmethod
    def _pca(M, k):
        mean = M.mean(axis=0)
        U, sv, Vt = np.linalg.svd(M - mean, full_matrices=False)
        k = min(k, Vt.shape[0])
        expl = sv ** 2 / np.sum(sv ** 2)
        return mean, Vt[:k], expl[:k]

    def _stack(self, Z):
        return np.hstack([(Z[s] - self.mu_[s]) / self.sd_[s] for s in STATES])

    def _unstack(self, M):
        return {s: M[:, i * N_T:(i + 1) * N_T] * self.sd_[s] + self.mu_[s] for i, s in enumerate(STATES)}

    def fit(self, X, Z):
        """X: batch inputs (DataFrame, rows = batches in the same order as Z); Z: dict of curves."""
        self.inputs_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X.values)
        Xs = self.scaler_.transform(X.values)
        self.pca_, self.ridge_ = {}, {}
        groups = {"joint": None} if self.joint else {s: None for s in STATES}
        if self.joint:
            self.mu_ = {s: Z[s].mean() for s in STATES}
            self.sd_ = {s: Z[s].std() + 1e-12 for s in STATES}
        for name in groups:
            M = self._stack(Z) if self.joint else Z[name]
            mean, comps, expl = self._pca(M, self.n_components)
            scores = (M - mean) @ comps.T
            self.pca_[name] = dict(mean=mean, comps=comps, expl=expl, scores=scores)
            self.ridge_[name] = RidgeCV(alphas=self.alphas).fit(Xs, scores)
        return self

    def predict(self, X):
        """dict state -> (n_batches, 13) predicted log-ratio curves."""
        Xs = self.scaler_.transform(X[self.inputs_].values)
        rec = {}
        for name, pc in self.pca_.items():
            S = self.ridge_[name].predict(Xs).reshape(len(Xs), -1)
            rec[name] = pc["mean"] + S @ pc["comps"]
        return self._unstack(rec["joint"]) if self.joint else rec

    # -- interpretation
    def explained_variance(self):
        return pd.DataFrame({n: pd.Series(pc["expl"], index=[f"PC{i+1}" for i in range(len(pc["expl"]))])
                             for n, pc in self.pca_.items()}).round(3)

    def score_coefficients(self, name):
        """Standardised ridge coefficients: rows = PCs, columns = batch inputs."""
        coef = np.atleast_2d(self.ridge_[name].coef_)
        return pd.DataFrame(coef, index=[f"PC{i+1}" for i in range(coef.shape[0])], columns=self.inputs_)
