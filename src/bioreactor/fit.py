"""Fit global parameters by (robust) least squares across batches.

Residual types
- "log"    : log(pred + eps) - log(obs + eps). Penalises relative error, so it optimises roughly
             what MAPE measures (small values, e.g. early cell counts, count as much as large ones).
- "scaled" : (pred - obs) / scale, scale = per-state std of the measurements (previous default).

Loss (scipy.optimize.least_squares)
- "linear" : ordinary least squares.
- "soft_l1": robust; residuals larger than ~f_scale are down-weighted (quadratic -> linear),
             so single odd points (e.g. last-day byproduct in Batch29/34) pull the fit less.
             With log residuals, f_scale = 0.1 means "beyond ~10 % error, treat as outlier-ish".
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .model import P0, LB, UB, predict_at_obs

EPS = 1e-3


def residuals(theta, arr, scale, residual="log"):
    p = np.exp(theta)
    preds = predict_at_obs(p, arr)
    if residual == "log":
        res = [(np.log(np.maximum(yh, 0) + EPS) - np.log(y + EPS)).ravel()
               for yh, y in zip(preds, arr.obs_y)]
    elif residual == "scaled":
        res = [((yh - y) / scale).ravel() for yh, y in zip(preds, arr.obs_y)]
    else:
        raise ValueError(f"unknown residual type: {residual}")
    return np.nan_to_num(np.concatenate(res), nan=1e3, posinf=1e3, neginf=-1e3)


def fit(arr, scale, p_init=P0, max_nfev=300, verbose=0,
        residual="log", loss="soft_l1", f_scale=0.1):
    lo = np.log(np.maximum(LB, 1e-8)); hi = np.log(UB)
    th0 = np.clip(np.log(np.maximum(p_init, 1e-8)), lo + 1e-6, hi - 1e-6)
    sol = least_squares(residuals, th0, bounds=(lo, hi), args=(arr, scale, residual),
                        loss=loss, f_scale=f_scale, x_scale=1.0, diff_step=1e-3,
                        max_nfev=max_nfev, verbose=verbose)
    return np.exp(sol.x), sol
