"""Sartorius-style mechanistic fed-batch model (ported from branch hybrid_v3_tr, src/sartorius_model.py).

States y = [Xv, Xd, Xl, gamma, S, L]: viable, dead and lysed cells, cumulative viable-cell
integral ("biomaterial"), glucose (substrate) and lactate (byproduct). Feeds are exact bolus
jumps (dilution plus glucose from feeds substrate, B and D); measured volumes reset V.
Observed outputs: VCD = Xv, TCD = Xv + Xd, substrate = S, byproduct = L.

Integration is RK4 on a fixed grid of DT days, compiled with numba. Parameters are fitted by
multi-start least squares on relative error (soft-L1 loss), which approximates MAPE.

Leakage: a BatchSet is built from a batch's inputs only (t = 0 state, feeds, volumes). Training
observations come from the targets passed in by the caller, never from the raw file.
"""

from __future__ import annotations

from dataclasses import dataclass

from numba import njit
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from bioreactor_decoding.dataset import FEEDS, TARGETS

DT = 0.01  # integration step [d]; events are placed on this grid (<= 7 min error)
SOFTPLUS_SHARPNESS = 0.005  # [g/(1e9 cells d)] smoothing width of max(0, x)

# name, initial guess, lower bound, upper bound, unit, meaning
PARAMS = [
    ("mu_max", 1.0, 0.2, 4.0, "1/d", "maximum effective growth rate"),
    ("K_S_mu", 0.3, 1e-3, 5.0, "g/L", "substrate scale of tanh^2 growth limitation"),
    ("K_gamma_mu", 100.0, 5.0, 3000.0, "1e6 cells*d/mL", "biomaterial level that halves growth"),
    ("k_d", 3e-3, 1e-6, 0.3, "1/d", "basal death rate"),
    ("k_Td", 5e-5, 1e-8, 1e-2, "mL/(1e6 cells*d^2)", "death increase per unit biomaterial"),
    ("k_l", 1e-2, 1e-6, 0.5, "1/d", "basal lysis rate of viable cells"),
    ("k_Tl", 1e-2, 1e-7, 1.0, "mL/(1e6 cells*d)", "lysis increase per lysed cell (toxicity)"),
    ("Y_G", 0.5, 1e-2, 10.0, "g/1e9 cells", "glucose consumed per unit growth"),
    ("K_I", 3.0, 0.1, 100.0, "g/L", "lactate inhibition of glucose uptake"),
    ("m_S", 0.02, 1e-5, 1.0, "g/(1e9 cells*d)", "maintenance glucose uptake"),
    ("K_S_up", 0.2, 1e-3, 5.0, "g/L", "substrate scale of tanh^2 uptake limitation"),
    ("v_ox", 0.05, 1e-4, 3.0, "g/(1e9 cells*d)", "oxidative capacity (overflow threshold)"),
    ("K_L", 0.5, 1e-2, 20.0, "g/L", "lactate scale of tanh^2 consumption limitation"),
    ("Y_P", 1.0, 1e-2, 10.0, "g/g", "lactate produced per overflow glucose"),
    ("Y_C", 1.0, 1e-2, 20.0, "g/g", "lactate consumed per unit spare oxidative capacity"),
    ("c_S", 300.0, 10.0, 1500.0, "g/L", "glucose in substrate feed"),
    ("c_B", 40.0, 0.1, 300.0, "g/L", "glucose in feed B"),
    ("c_D", 5.0, 1e-3, 1000.0, "g/L", "glucose in feed D"),
]
PARAM_NAMES = [p[0] for p in PARAMS]
P0 = np.array([p[1] for p in PARAMS])
P_LO = np.array([p[2] for p in PARAMS])
P_HI = np.array([p[3] for p in PARAMS])


# --------------------------------------------------------------------------
# Kinetics and simulator
# --------------------------------------------------------------------------


def _softplus(x):
    z = x / SOFTPLUS_SHARPNESS
    return SOFTPLUS_SHARPNESS * (np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))))


softplus_nb = njit(_softplus)


@njit(cache=True)
def _rates_scalar(Xv, Xd, Xl, gamma, S, L, p):
    S = max(S, 0.0)
    L = max(L, 0.0)
    gamma = max(gamma, 0.0)
    Xl = max(Xl, 0.0)
    mu_eff = p[0] * np.tanh(S / p[1]) ** 2 / ((gamma / p[2]) ** 3 + 1.0)
    mu_d = p[3] + p[4] * gamma
    mu_l = p[5] + p[6] * Xl
    v_S = (p[7] * mu_eff / ((L / p[8]) ** 3 + 1.0) + p[9]) * np.tanh(S / p[10]) ** 2
    lac_prod = softplus_nb(v_S - p[11])
    lac_cons = softplus_nb(p[11] - v_S) * np.tanh(L / p[12]) ** 2
    v_L = p[13] * lac_prod - p[14] * lac_cons
    return mu_eff, mu_d, mu_l, v_S, v_L


@njit(cache=True)
def _rhs(y, p, dy):
    Xv = max(y[0], 0.0)
    mu_eff, mu_d, mu_l, v_S, v_L = _rates_scalar(Xv, y[1], y[2], y[3], y[4], y[5], p)
    dy[0] = (mu_eff - mu_d - mu_l) * Xv
    dy[1] = mu_d * Xv
    dy[2] = mu_l * Xv
    dy[3] = Xv
    dy[4] = -v_S * Xv
    dy[5] = v_L * Xv


@njit(cache=True)
def _simulate(y0, v0, feeds, vsnap, p):
    """RK4 on a fixed grid with exact bolus jumps. Returns (nb, n_steps, 6).

    out[b, k] is the state at t = k*DT *before* any event logged at that step,
    so a sample taken just before a feed sees the pre-feed state.
    """
    nb, ns, _ = feeds.shape
    out = np.empty((nb, ns, 6))
    c_S, c_B, c_D = p[15], p[16], p[17]
    h = DT
    y = np.empty(6)
    tmp = np.empty(6)
    k1 = np.empty(6)
    k2 = np.empty(6)
    k3 = np.empty(6)
    k4 = np.empty(6)
    for b in range(nb):
        for j in range(6):
            y[j] = y0[b, j]
        V = v0[b]
        for k in range(ns):
            for j in range(6):
                out[b, k, j] = y[j]
            if not np.isnan(vsnap[b, k]):
                V = vsnap[b, k]
            f_s = feeds[b, k, 0]
            f_b = feeds[b, k, 1]
            f_c = feeds[b, k, 2]
            f_d = feeds[b, k, 3]
            total = f_s + f_b + f_c + f_d
            if total > 0.0:
                v_new = V + total
                ratio = V / v_new
                for j in range(6):
                    y[j] *= ratio
                y[4] += (c_S * f_s + c_B * f_b + c_D * f_d) / v_new
                V = v_new
            if k == ns - 1:
                break
            _rhs(y, p, k1)
            for j in range(6):
                tmp[j] = y[j] + 0.5 * h * k1[j]
            _rhs(tmp, p, k2)
            for j in range(6):
                tmp[j] = y[j] + 0.5 * h * k2[j]
            _rhs(tmp, p, k3)
            for j in range(6):
                tmp[j] = y[j] + h * k3[j]
            _rhs(tmp, p, k4)
            for j in range(6):
                y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
                y[j] = max(y[j], 0.0)
    return out


# --------------------------------------------------------------------------
# Batch inputs on the integration grid
# --------------------------------------------------------------------------


@dataclass
class BatchSet:
    """Inputs of a set of batches on the DT grid, plus (optionally) training observations.

    obs: one row per training observation with batch_idx, step and the 4 targets; empty when
    the BatchSet is only used for prediction.
    """

    ids: list
    y0: np.ndarray  # (nb, 6)
    v0: np.ndarray  # (nb,)
    feeds: np.ndarray  # (nb, n_steps, 4) feed volumes logged at each grid step
    vsnap: np.ndarray  # (nb, n_steps) measured volume, NaN where none
    obs: pd.DataFrame

    def subset(self, idx) -> BatchSet:
        idx = np.asarray(idx)
        remap = {old: new for new, old in enumerate(idx)}
        obs = self.obs[self.obs["batch_idx"].isin(idx)].copy()
        obs["batch_idx"] = obs["batch_idx"].map(remap)
        return BatchSet(
            [self.ids[i] for i in idx],
            self.y0[idx],
            self.v0[idx],
            self.feeds[idx],
            self.vsnap[idx],
            obs.reset_index(drop=True),
        )


def build_batches(
    samples: pd.DataFrame,
    feeds: pd.DataFrame,
    batch_ids: list,
    obs: pd.DataFrame | None = None,
) -> BatchSet:
    """BatchSet from our tables (see dataset.load_tables).

    Only the inputs are read from `samples`: the t = 0 measurement, sampling times and volumes.
    obs: optional training observations with batch_id, elapsed_time and TARGETS (from Y).
    """
    t_max = max(samples["elapsed_time"].max(), feeds["elapsed_time"].max())
    n_steps = int(np.ceil(t_max / DT)) + 2
    nb = len(batch_ids)
    y0 = np.zeros((nb, 6))
    v0 = np.zeros(nb)
    grid_feeds = np.zeros((nb, n_steps, 4))
    vsnap = np.full((nb, n_steps), np.nan)
    s_by = dict(tuple(samples.groupby("batch_id")))
    f_by = dict(tuple(feeds.groupby("batch_id")))
    for i, b in enumerate(batch_ids):
        s = s_by[b].sort_values("elapsed_time")
        first = s.iloc[0]
        if not np.isclose(first["elapsed_time"], 0.0):
            raise ValueError(f"Batch {b} has no t = 0 sample")
        y0[i] = [
            first["vcd"], max(first["tcd"] - first["vcd"], 0.0), 0.0, 0.0,
            first["substrate"], first["byproduct"],
        ]  # fmt: skip
        v0[i] = first["volume"]
        steps = np.rint(s["elapsed_time"].to_numpy() / DT).astype(int)
        vsnap[i, steps] = s["volume"].to_numpy()
        f = f_by.get(b)
        if f is not None:
            for j, name in enumerate(FEEDS):
                ev = f[f["feed"] == name]
                np.add.at(
                    grid_feeds[i, :, j],
                    np.rint(ev["elapsed_time"].to_numpy() / DT).astype(int),
                    ev["amount"].to_numpy(),
                )
    if obs is None:
        obs_df = pd.DataFrame(columns=["batch_idx", "step", *TARGETS])
    else:
        pos = {b: i for i, b in enumerate(batch_ids)}
        obs_df = pd.DataFrame(
            {
                "batch_idx": obs["batch_id"].map(pos).to_numpy(),
                "step": np.rint(obs["elapsed_time"].to_numpy() / DT).astype(int),
                **{c: obs[c].to_numpy(float) for c in TARGETS},
            }
        )
    return BatchSet(list(batch_ids), y0, v0, grid_feeds, vsnap, obs_df)


def simulate(bs: BatchSet, params: np.ndarray) -> np.ndarray:
    return _simulate(bs.y0, bs.v0, bs.feeds, bs.vsnap, np.asarray(params, dtype=np.float64))


def observe(traj: np.ndarray, batch_idx, steps) -> np.ndarray:
    """Map model states to the four measured outputs, shape (n, 4)."""
    s = traj[np.asarray(batch_idx, int), np.asarray(steps, int)]
    return np.column_stack([s[:, 0], s[:, 0] + s[:, 1], s[:, 4], s[:, 5]])


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def _fit_one_start(args) -> tuple[float, np.ndarray]:
    """One multi-start run: residual (pred - obs) / obs, soft-L1 loss, log-parameters."""
    bs, theta0, max_nfev = args
    y_obs = bs.obs[TARGETS].to_numpy(float)
    b_idx = bs.obs["batch_idx"].to_numpy(int)
    steps = bs.obs["step"].to_numpy(int)

    def resid(theta):
        r = ((observe(simulate(bs, np.exp(theta)), b_idx, steps) - y_obs) / y_obs).ravel()
        return r if np.all(np.isfinite(r)) else np.full(r.shape, 1e3)

    res = least_squares(resid, theta0, bounds=(np.log(P_LO), np.log(P_HI)), method="trf",
                        x_scale=1.0, max_nfev=max_nfev, loss="soft_l1", f_scale=0.05)  # fmt: skip
    return float(res.cost), np.exp(res.x)


def starting_points(n_starts: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    lo, hi = np.log(P_LO) + 1e-9, np.log(P_HI) - 1e-9
    base = np.log(P0)
    starts = [base]
    for _ in range(n_starts - 1):
        starts.append(np.clip(base + rng.uniform(-1.2, 1.2, base.size), lo, hi))
    return starts


def run_parallel(fn, tasks: list, n_jobs: int) -> list:
    """Map `fn` over `tasks` in `n_jobs` worker processes (in-process when n_jobs == 1).

    joblib's loky workers are fresh processes (no fork of a multi-threaded parent) that do not
    re-import the calling script, and each is limited to one BLAS thread.
    """
    if n_jobs > 1 and len(tasks) > 1:
        from joblib import Parallel, delayed, parallel_config

        with parallel_config(backend="loky", inner_max_num_threads=1):
            return Parallel(n_jobs=min(n_jobs, len(tasks)))(delayed(fn)(t) for t in tasks)
    return [fn(t) for t in tasks]


def fit_many(jobs: dict, n_starts: int, max_nfev: int, n_jobs: int, seed: int = 0) -> dict:
    """Fit several BatchSets (e.g. CV folds) with multi-start, in parallel.

    jobs: {name: BatchSet}. Returns {name: best parameter vector}.
    """
    tasks, keys = [], []
    for name, bs in jobs.items():
        for theta0 in starting_points(n_starts, seed):
            tasks.append((bs, theta0, max_nfev))
            keys.append(name)
    best: dict = {}
    for name, (cost, params) in zip(keys, run_parallel(_fit_one_start, tasks, n_jobs)):
        if name not in best or cost < best[name][0]:
            best[name] = (cost, params)
    return {name: params for name, (_, params) in best.items()}
