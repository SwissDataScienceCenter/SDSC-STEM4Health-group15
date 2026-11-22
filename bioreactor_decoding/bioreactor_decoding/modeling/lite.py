"""A light stage 1 for hybrid v3: a mechanistic model without neural nets, fitted in stages.

Ported from branch 06_hybrid_tr_based (src/bioreactor/model.py and fit.py, the "v1" model with
all optional terms off). It replaces modeling.hybrid as stage 1 of HybridV3 in `hybrid_lite`:
stage 2 (soft sensors, two branches, blend) does most of the work, so a cheaper, cruder stage 1
scores about the same.

States per batch: Xv viable cells, Xd dead cells (TCD = Xv + Xd), S substrate, P byproduct and
N, a latent limiting nutrient (unmeasured, N(0) = 1).

  mu  = mu_max * S/(Ks+S) * N/(KN+N) * KI/(KI+P)       growth
  kd  = kd0 + kd1 * KdN/(KdN+N)                        death rises as N runs out
  dXv = (mu - kd) Xv            dXd = kd Xv - klys Xd
  dS  = -(mu/Yxs + ms) Xv       dP  = (Ypx mu - qPc P/(KPc+P)) Xv       dN = -(mu/Yxn) Xv
Feeds are boluses: everything is diluted by V/(V+F), then S += (Ss F_substrate + Sb F_b)/V_new
and N += Nb F_b / V_new. Feeds C and D only add volume. Forward Euler on a 0.005-day grid, all
batches at once; a feed is applied at the grid step of its time, after a sample at that step.

Fitting (log residuals, soft-L1 loss): starting values estimated from the training observations,
then block-wise fits (cells -> substrate -> byproduct), then a joint fit of all parameters.

Leakage: arrays hold only a batch's inputs (t = 0 state, feeds, volumes); training observations
come from the caller (Y), and starting values are estimated from those observations only.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

from loguru import logger
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from bioreactor_decoding.dataset import FEEDS, TARGETS
from bioreactor_decoding.modeling.mechanistic import run_parallel

PARAMS = {  # name: (initial guess, lower, upper)
    "mu_max": (1.0, 0.1, 3.0),
    "Ks": (0.2, 1e-3, 5.0),
    "KI": (3.0, 0.1, 50.0),
    "KN": (0.1, 1e-3, 2.0),
    "Yxn": (20.0, 0.5, 500.0),
    "Nb": (5.0, 1e-3, 100.0),
    "kd0": (0.01, 1e-4, 0.5),
    "kd1": (0.1, 1e-4, 2.0),
    "KdN": (0.1, 1e-3, 5.0),
    "klys": (0.02, 1e-4, 1.0),
    "Yxs": (10.0, 0.05, 200.0),
    "ms": (0.01, 1e-5, 0.5),
    "Sb": (10.0, 1e-3, 200.0),
    "Ss": (100.0, 1e-3, 2000.0),
    "Ypx": (0.3, 1e-3, 5.0),
    "qPc": (0.02, 1e-5, 1.0),
    "KPc": (0.5, 1e-3, 10.0),
}
NAMES = list(PARAMS)
P0 = np.array([v[0] for v in PARAMS.values()])
LB = np.array([v[1] for v in PARAMS.values()])
UB = np.array([v[2] for v in PARAMS.values()])
STAGES = [  # (name, free parameters, targets in the residual)
    ("cells", ["mu_max", "KN", "Yxn", "Nb", "kd0", "kd1", "KdN", "klys"], ["vcd", "tcd"]),
    ("substrate", ["Yxs", "ms", "Sb", "Ss", "Ks"], ["substrate"]),
    ("byproduct", ["Ypx", "qPc", "KPc", "KI"], ["byproduct"]),
]
EPS = 1e-3


# --------------------------------------------------------------------------
# Inputs on the Euler grid
# --------------------------------------------------------------------------


@dataclass
class LiteArrays:
    ids: list
    t: np.ndarray  # (K,) grid [d]
    feeds: np.ndarray  # (K, B, 4) feed volume added at step k
    x0: np.ndarray  # (B, 4) t = 0 vcd, tcd, substrate, byproduct
    v0: np.ndarray  # (B,)
    obs: pd.DataFrame  # training observations: batch_idx, k, t, TARGETS (may be empty)

    def subset(self, idx) -> LiteArrays:
        idx = np.asarray(idx)
        remap = {old: new for new, old in enumerate(idx)}
        obs = self.obs[self.obs["batch_idx"].isin(idx)].copy()
        obs["batch_idx"] = obs["batch_idx"].map(remap)
        return LiteArrays([self.ids[i] for i in idx], self.t, self.feeds[:, idx], self.x0[idx],
                          self.v0[idx], obs.reset_index(drop=True))  # fmt: skip


def build_arrays(samples, feeds, batch_ids, obs=None, dt=0.005, t_end=13.2) -> LiteArrays:
    """Arrays from our tables; only the t = 0 sample of each batch is read from `samples`.

    obs: optional training observations with batch_id, elapsed_time and TARGETS (from Y).
    """
    t = np.arange(0.0, t_end + dt, dt)
    K, B = len(t), len(batch_ids)
    F = np.zeros((K, B, 4))
    x0, v0 = np.zeros((B, 4)), np.zeros(B)
    s_by, f_by = dict(tuple(samples.groupby("batch_id"))), dict(tuple(feeds.groupby("batch_id")))
    for j, b in enumerate(batch_ids):
        first = s_by[b].sort_values("elapsed_time").iloc[0]
        if not np.isclose(first["elapsed_time"], 0.0):
            raise ValueError(f"Batch {b} has no t = 0 sample")
        x0[j] = first[TARGETS].to_numpy(float)
        v0[j] = first["volume"]
        f = f_by.get(b)
        if f is not None:
            for c, name in enumerate(FEEDS):
                ev = f[f["feed"] == name]
                # a feed comes after the sample at the same grid step -> ceil
                k = np.minimum(np.ceil(ev["elapsed_time"].to_numpy() / dt).astype(int), K - 1)
                np.add.at(F[:, j, c], k, ev["amount"].to_numpy())
    if obs is None:
        o = pd.DataFrame(columns=["batch_idx", "k", "t", *TARGETS])
    else:
        pos = {b: i for i, b in enumerate(batch_ids)}
        tt = obs["elapsed_time"].to_numpy(float)
        o = pd.DataFrame({"batch_idx": obs["batch_id"].map(pos).to_numpy(),
                          "k": np.rint(tt / dt).astype(int), "t": tt,
                          **{c: obs[c].to_numpy(float) for c in TARGETS}})  # fmt: skip
    return LiteArrays(list(batch_ids), t, F, x0, v0, o)


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------


def simulate(p, arr: LiteArrays) -> np.ndarray:
    """Trajectories on arr.t for all batches: (K, B, 4) = vcd, tcd, substrate, byproduct."""
    q = dict(zip(NAMES, p))
    t, F = arr.t, arr.feeds
    dt = t[1] - t[0]
    Xv = arr.x0[:, 0].copy()
    Xd = np.maximum(arr.x0[:, 1] - arr.x0[:, 0], 0.0)
    S, P = arr.x0[:, 2].copy(), arr.x0[:, 3].copy()
    N, V = np.ones_like(Xv), arr.v0.copy()
    out = np.empty((len(t), len(Xv), 4))
    fed = F.sum(axis=2) > 0
    for k in range(len(t)):
        out[k, :, 0], out[k, :, 1], out[k, :, 2], out[k, :, 3] = Xv, Xv + Xd, S, P
        if fed[k].any():
            f = F[k]
            Vn = V + f.sum(axis=1)
            r = V / Vn
            Xv, Xd, P = Xv * r, Xd * r, P * r
            S = S * r + (q["Ss"] * f[:, 0] + q["Sb"] * f[:, 1]) / Vn
            N = N * r + q["Nb"] * f[:, 1] / Vn
            V = Vn
        Sp, Pp, Np = np.maximum(S, 0), np.maximum(P, 0), np.maximum(N, 0)
        mu = q["mu_max"] * Sp / (q["Ks"] + Sp) * Np / (q["KN"] + Np) * q["KI"] / (q["KI"] + Pp)
        kd = q["kd0"] + q["kd1"] * q["KdN"] / (q["KdN"] + Np)
        dXv, dXd = (mu - kd) * Xv, kd * Xv - q["klys"] * Xd
        dS = -(mu / q["Yxs"] + q["ms"]) * Xv
        dP = (q["Ypx"] * mu - q["qPc"] * Pp / (q["KPc"] + Pp)) * Xv
        dN = -(mu / q["Yxn"]) * Xv
        Xv, Xd = Xv + dt * dXv, Xd + dt * dXd
        S, P, N = (
            np.maximum(S + dt * dS, 0),
            np.maximum(P + dt * dP, 0),
            np.maximum(N + dt * dN, 0),
        )
    return out


def observe(traj, batch_idx, k) -> np.ndarray:
    return traj[np.asarray(k, int), np.asarray(batch_idx, int), :]


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def initial_from_data(arr: LiteArrays) -> np.ndarray:
    """Rough starting values from the median behaviour of the training observations."""
    o = arr.obs.assign(day=np.rint(arr.obs["t"].to_numpy()).astype(int))
    w = o.pivot_table(index="batch_idx", columns="day", values=TARGETS)
    x0 = pd.DataFrame(arr.x0, columns=TARGETS)  # index = batch_idx, like w
    v, S, P = w["vcd"].copy(), w["substrate"].copy(), w["byproduct"].copy()
    v[0], S[0], P[0] = x0["vcd"], x0["substrate"], x0["byproduct"]
    p = dict(zip(NAMES, P0))
    slope = max((np.log(v[1]) - np.log(v[0])).median(), (np.log(v[2]) - np.log(v[1])).median())
    p["mu_max"] = slope / 0.85  # Monod / nutrient factors < 1 at the start
    p["Yxs"] = ((v[1] - v[0]) / (S[0] - S[1])).median()  # day 0 -> 1 has no feed yet
    p["Ypx"] = ((P[1] - P[0]) / (v[1] - v[0])).median()
    p["qPc"] = ((P[7] - P[10]) / 3 / v[[7, 8, 9, 10]].mean(axis=1)).median() / 0.75
    p.update(Ks=0.1, KI=10.0, KPc=0.5, Yxn=10.0, Nb=7.0, KN=0.05, kd0=0.005, kd1=0.01,
             KdN=0.1, klys=0.01, ms=0.01, Sb=50.0, Ss=300.0)  # fmt: skip
    return np.clip(np.array([p[n] for n in NAMES]), LB * 1.01, UB * 0.99)


def _residuals(p, arr, targets):
    cols = [TARGETS.index(c) for c in targets]
    pred = observe(simulate(p, arr), arr.obs["batch_idx"], arr.obs["k"])[:, cols]
    y = arr.obs[targets].to_numpy(float)
    r = (np.log(np.maximum(pred, 0) + EPS) - np.log(y + EPS)).ravel()
    return np.nan_to_num(r, nan=10.0, posinf=10.0, neginf=-10.0)


def _fit_block(arr, p, free, targets, max_nfev, f_scale):
    idx = np.array([NAMES.index(n) for n in free])
    lo, hi = np.log(LB[idx]), np.log(UB[idx])
    th0 = np.clip(np.log(p[idx]), lo + 1e-6, hi - 1e-6)

    def full(th):
        q = p.copy()
        q[idx] = np.exp(th)
        return q

    sol = least_squares(lambda th: _residuals(full(th), arr, targets), th0, bounds=(lo, hi),
                        loss="soft_l1", f_scale=f_scale, x_scale=1.0, diff_step=1e-3,
                        max_nfev=max_nfev)  # fmt: skip
    return full(sol.x)


@dataclass
class LiteODEConfig:
    """Settings of the light stage 1 (defaults = branch 06_hybrid_tr_based)."""

    max_nfev_stage: int = 20
    max_nfev_final: int = 60
    f_scale: float = 0.1


@dataclass
class LiteODE:
    """A fitted light stage 1."""

    config: LiteODEConfig
    params: np.ndarray

    def predict(self, arr: LiteArrays, batch_idx, times) -> np.ndarray:
        k = np.rint(np.asarray(times, float) / (arr.t[1] - arr.t[0])).astype(int)
        return observe(simulate(self.params, arr), batch_idx, k)


def _fit_one(args) -> np.ndarray:
    arr, config = args
    p = initial_from_data(arr)
    for _, free, targets in STAGES:
        p = _fit_block(arr, p, free, targets, config.max_nfev_stage, config.f_scale)
    return _fit_block(arr, p, NAMES, TARGETS, config.max_nfev_final, config.f_scale)


def fit_lite_odes(jobs: dict, config: LiteODEConfig, n_jobs: int = 1) -> dict:
    """Fit one LiteODE per LiteArrays in `jobs` ({name: arrays}), in parallel."""
    t0 = time.time()
    names = list(jobs)
    params = run_parallel(_fit_one, [(jobs[n], config) for n in names], n_jobs)
    logger.info(f"stage 1 (lite): {len(jobs)} fits in {time.time() - t0:.0f}s")
    return {n: LiteODE(config, p) for n, p in zip(names, params)}
