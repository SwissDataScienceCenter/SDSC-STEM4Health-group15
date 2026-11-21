"""Fit global parameters by (robust) least squares across batches.

Residual types
- "log"    : log(pred + eps) - log(obs + eps). Penalises relative error (~ what MAPE measures).
- "scaled" : (pred - obs) / scale, scale = per-state std of the measurements.
Loss (scipy.optimize.least_squares): "linear" or "soft_l1" (robust, f_scale = transition point).

Strategy (fit_staged)
1. Starting values estimated from the data (initial_from_data) instead of generic guesses.
2. Staged fit: each block of parameters is fitted on the state(s) it mainly drives,
   with the others fixed:  cells -> substrate -> byproduct.
3. Final joint fit of all parameters on all states, run until convergence (or max_nfev),
   optionally from several perturbed starting points (multi-start); best cost wins.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .data import STATES
from . import model as M
from .model import NAMES, predict_at_obs   # NAMES is updated in place by model.set_options

EPS = 1e-3
STAGES = [  # (name, free parameters, states used in the residual); inactive ones are skipped
    ("cells",     ["mu_max", "KN", "Yxn", "Nb", "kd0", "kd1", "KdN", "klys", "t_lag"], ["vcd", "tcd"]),
    ("substrate", ["Yxs", "ms", "Sb", "Ss", "Ks"],                                     ["substrate"]),
    ("byproduct", ["Ypx", "qPc", "KPc", "KI", "KPP", "K_mu"],                         ["byproduct"]),
]


# ---------------------------------------------------------------- starting values from data
def initial_from_data(df, ids):
    """Rough parameter estimates from the median behaviour of the training batches.
    Returns a full parameter vector (same order as model.NAMES)."""
    s = df[df.batch_id.isin(ids) & df.vcd.notna()].copy()
    s["day"] = s.elapsed_time.round().astype(int)
    w = s.pivot_table(index="batch_id", columns="day", values=STATES)
    v, S, P = w["vcd"], w["substrate"], w["byproduct"]
    p = dict(zip(NAMES, M.P0))
    # max specific growth: steepest early log-slope of VCD (days 0-2, no limitation yet)
    slope = max((np.log(v[1]) - np.log(v[0])).median(), (np.log(v[2]) - np.log(v[1])).median())
    p["mu_max"] = slope / 0.85            # Monod / nutrient factors < 1 at the start
    # biomass yield on substrate: day 0->1 has no feed yet
    p["Yxs"] = ((v[1] - v[0]) / (S[0] - S[1])).median()
    # byproduct per new cell: day 0->1
    p["Ypx"] = ((P[1] - P[0]) / (v[1] - v[0])).median()
    # byproduct consumption per cell, days 7-10 (P falling), corrected for P/(KPc+P) ~ 0.75
    p["qPc"] = ((P[7] - P[10]) / 3 / v[[7, 8, 9, 10]].mean(axis=1)).median() / 0.75
    p.update(
        Ks=0.1, KI=10.0, KPc=0.5,
        # latent nutrient: runs out around day 7 -> Yxn ~ 10, Nb ~ 7 (see notebook text)
        Yxn=10.0, Nb=7.0, KN=0.05,
        # death: dead cells grow ~0.3 /day at VCD ~27 late; the VCD decline is mostly dilution
        kd0=0.005, kd1=0.01, KdN=0.1, klys=0.01, ms=0.01,
        # substrate concentration of feed B / substrate feed [g/L], order-of-magnitude balance
        Sb=50.0, Ss=300.0,
    )
    # optional terms (only used if active): ~0.5 d lag, plateau of the byproduct at ~2.5 g/L,
    # consumption switching on once growth falls below ~0.2 /d
    p.update(t_lag=0.5, KPP=2.0, K_mu=0.2)
    if "t_lag" in NAMES:   # early slope includes the lag -> true max. growth rate is higher
        p["mu_max"] = p["mu_max"] / 0.8
    return np.clip(np.array([p[n] for n in NAMES]), M.LB * 1.01, M.UB * 0.99)


# ---------------------------------------------------------------- residuals / single fit
def residuals_full(p, arr, scale, residual="log", states=None):
    preds = predict_at_obs(p, arr)
    cols = [STATES.index(s) for s in (states or STATES)]
    if residual == "log":
        res = [(np.log(np.maximum(yh[:, cols], 0) + EPS) - np.log(y[:, cols] + EPS)).ravel()
               for yh, y in zip(preds, arr.obs_y)]
    elif residual == "scaled":
        sc = np.asarray(scale)[cols]
        res = [((yh[:, cols] - y[:, cols]) / sc).ravel() for yh, y in zip(preds, arr.obs_y)]
    else:
        raise ValueError(f"unknown residual type: {residual}")
    return np.nan_to_num(np.concatenate(res), nan=10.0, posinf=10.0, neginf=-10.0)


def fit(arr, scale, p_init=None, max_nfev=300, verbose=0, residual="log", loss="soft_l1",
        f_scale=0.1, free=None, states=None):
    """Fit the parameters in `free` (default: all) on `states` (default: all); others fixed."""
    p_init = M.P0 if p_init is None else np.asarray(p_init, float)
    if len(p_init) != len(NAMES):
        raise ValueError(f"p_init has {len(p_init)} entries, active model ({M.model_tag()}) needs {len(NAMES)}")
    free = [n for n in (free or NAMES) if n in NAMES]          # skip inactive optional parameters
    free_idx = np.array([NAMES.index(n) for n in free])
    lo = np.log(M.LB[free_idx]); hi = np.log(M.UB[free_idx])
    th0 = np.clip(np.log(p_init[free_idx]), lo + 1e-6, hi - 1e-6)

    def full(th):
        p = p_init.copy(); p[free_idx] = np.exp(th); return p

    def fun(th):
        return residuals_full(full(th), arr, scale, residual, states)

    sol = least_squares(fun, th0, bounds=(lo, hi), loss=loss, f_scale=f_scale,
                        x_scale=1.0, diff_step=1e-3, max_nfev=max_nfev, verbose=verbose)
    return full(sol.x), sol


# ---------------------------------------------------------------- staged + multi-start
def fit_staged(arr, scale, p_init=None, df=None, ids=None, max_nfev_stage=30, max_nfev_final=100,
               n_starts=1, spread=0.3, seed=0, verbose=True, **kw):
    """1) data-based start, 2) block-wise stages, 3) joint fit (multi-start). Returns p, log."""
    t0 = time.time(); log = []
    if p_init is None:
        p_init = initial_from_data(df, ids) if df is not None else M.P0.copy()
    p = np.asarray(p_init, float).copy()

    def cost(pp):
        r = residuals_full(pp, arr, scale, kw.get("residual", "log"), None)
        return 0.5 * np.sum(r ** 2)

    log.append(("start", cost(p), time.time() - t0))
    if verbose: print(f"{'start':10s} cost {log[-1][1]:9.3f}")
    for name, free, states in STAGES:
        p, sol = fit(arr, scale, p, max_nfev=max_nfev_stage, free=free, states=states, **kw)
        log.append((name, cost(p), time.time() - t0))
        if verbose: print(f"{name:10s} cost {log[-1][1]:9.3f}  ({log[-1][2]:.0f} s)")

    rng = np.random.default_rng(seed)
    best_p, best_c = None, np.inf
    for k in range(n_starts):
        start = p if k == 0 else np.clip(p * np.exp(rng.normal(0, spread, len(p))), M.LB * 1.01, M.UB * 0.99)
        pk, sol = fit(arr, scale, start, max_nfev=max_nfev_final, **kw)
        c = cost(pk)
        log.append((f"joint#{k}", c, time.time() - t0))
        if verbose: print(f"joint #{k:<4d} cost {c:9.3f}  ({log[-1][2]:.0f} s)  {sol.message}")
        if c < best_c: best_p, best_c = pk, c
    return best_p, pd.DataFrame(log, columns=["step", "cost", "time_s"])
