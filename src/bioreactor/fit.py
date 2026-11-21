"""Fit global parameters by weighted least squares across batches."""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .model import P0, LB, UB, predict_at_obs


def residuals(theta, arr, scale):
    p = np.exp(theta)
    preds = predict_at_obs(p, arr)
    res = [((yh - y) / scale).ravel() for yh, y in zip(preds, arr.obs_y)]
    r = np.concatenate(res)
    return np.nan_to_num(r, nan=1e3)


def fit(arr, scale, p_init=P0, max_nfev=300, verbose=0):
    lo = np.log(np.maximum(LB, 1e-8)); hi = np.log(UB)
    th0 = np.clip(np.log(np.maximum(p_init, 1e-8)), lo + 1e-6, hi - 1e-6)
    sol = least_squares(residuals, th0, bounds=(lo, hi), args=(arr, scale),
                        x_scale=1.0, diff_step=1e-3, max_nfev=max_nfev, verbose=verbose)
    return np.exp(sol.x), sol
