"""Sartorius-style CHO digital-twin model fitted to the DTB fed-batch data.

Model #4 from reports/Bioreactor model shortlist.md: the mechanistic ODE layer of
Richelle et al. 2026, "A Hybrid Modeling Framework for Predictive Digital Twins
of CHO Cell Culture", Comput Struct Biotechnol J 35:0078 (doi:10.34133/csbj.0078),
adapted to what this dataset measures.

States per batch (concentrations; time in days)
    Xv     viable cells            [1e6 cells/mL]   measured as VCD
    Xd     dead (unlysed) cells    [1e6 cells/mL]   TCD = Xv + Xd
    Xl     lysed cells             [1e6 cells/mL]   internal; trypan blue cannot see them
    gamma  "biomaterial"           [1e6 cells*d/mL] integral of Xv, diluted by feeds
    S      substrate (glucose)     [g/L]
    L      byproduct (lactate)     [g/L]

Kept from Richelle et al. 2026 (their equation numbers)
    dXv/dt = (mu_eff - mu_d - mu_l) * Xv                       (8, 14)
    dXd/dt = mu_d * Xv                                         (9)
    dXl/dt = mu_l * Xv                                         (10)
    dgamma/dt = Xv                                             (11)
    mu_d = k_d + k_Td * gamma                                  (12)
    mu_l = k_l + k_Tl * Xl                                     (13)
    v_S = Y_G * mu_eff * tanh(S/K)^2 / ((L/K_I)^3 + 1) + m_S   (20)
    v_L = Y_P * max(0, v_S - v_ox)
        - Y_C * max(0, v_ox - v_S) * tanh(L/K_L)^2             (21-23, overflow)
    dS/dt = -v_S * Xv,  dL/dt = v_L * Xv                       (15, 16)

Adapted to this dataset
    1. Growth. The paper predicts mu with a neural net fed 21 measured
       metabolites, which we do not have. Instead we use the smooth-switch
       growth law of the Sartorius sister model (Kuchemueller et al. 2026,
       doi:10.1007/s00449-026-03362-2):
           mu_eff = mu_max * tanh(S/K_S_mu)^2 / ((gamma/K_gamma_mu)^3 + 1)
       so growth stops as biomaterial accumulates. Feeding dilutes gamma and
       therefore extends growth.
    2. Feeds. Each bolus is an exact jump: V+ = V + sum(dV_j), every state is
       diluted by V/V+, and substrate gains sum(c_j * dV_j)/V+. The glucose
       concentrations of the substrate feed, feed B and feed D are fitted.
       Feed C is treated as volume only (tiny, pH-control-like doses).
       Volume is reset to the logged value at every volume stamp (sampling).
    3. Maintenance uptake is gated by tanh(S/K)^2 so substrate stays >= 0,
       and max() is replaced by a sharp softplus to keep the fit smooth.
    4. One joint fit (TRF, log-parameters, multi-start) instead of the paper's
       two sequential Nelder-Mead fits. By default it minimises relative error
       with a robust soft-L1 loss, because the leaderboard scores the mean of
       the four MAPEs (CV score 12.0 vs 12.5 for std-scaled squared error).
    5. Glutamine, glutamate, ammonia and the dFBA layer are dropped because
       they are not measured.

Evaluation
    Grouped K-fold cross-validation over training batches. Each held-out batch
    is simulated open-loop from its t=0 measurement plus its own feeds and
    volumes, exactly like the leaderboard holdouts. Then a final fit on all
    training batches produces predictions for the 12 holdout batches.

Run from the repo root:

    python src/sartorius_model.py                 # full run (CV + final fit)
    python src/sartorius_model.py --folds 0       # final fit only
"""

from __future__ import annotations

import os

# One BLAS/numba thread per process; parallelism comes from the process pool.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "input-data" / "Project 1 - DTB"
OUT_DIR = ROOT / "outputs" / "sartorius"

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
TARGET_LABELS = {
    "vcd": "VCD [1e6 cells/mL]",
    "tcd": "TCD [1e6 cells/mL]",
    "substrate": "Substrate [g/L]",
    "byproduct": "Byproduct [g/L]",
}
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


def kinetic_rates(Xv, Xd, Xl, gamma, S, L, p):
    """Specific rates. Works on floats (inside numba) and on numpy arrays."""
    S = np.maximum(S, 0.0)
    L = np.maximum(L, 0.0)
    gamma = np.maximum(gamma, 0.0)
    Xl = np.maximum(Xl, 0.0)
    mu_eff = p[0] * np.tanh(S / p[1]) ** 2 / ((gamma / p[2]) ** 3 + 1.0)
    mu_d = p[3] + p[4] * gamma
    mu_l = p[5] + p[6] * Xl
    v_S = (p[7] * mu_eff / ((L / p[8]) ** 3 + 1.0) + p[9]) * np.tanh(S / p[10]) ** 2
    lac_prod = _softplus(v_S - p[11])
    lac_cons = _softplus(p[11] - v_S) * np.tanh(L / p[12]) ** 2
    v_L = p[13] * lac_prod - p[14] * lac_cons
    return mu_eff, mu_d, mu_l, v_S, v_L


_softplus_nb = njit(_softplus)


@njit(cache=False)
def _rates_scalar(Xv, Xd, Xl, gamma, S, L, p):
    """Scalar copy of kinetic_rates for the numba integrator (keep in sync)."""
    S = max(S, 0.0)
    L = max(L, 0.0)
    gamma = max(gamma, 0.0)
    Xl = max(Xl, 0.0)
    mu_eff = p[0] * np.tanh(S / p[1]) ** 2 / ((gamma / p[2]) ** 3 + 1.0)
    mu_d = p[3] + p[4] * gamma
    mu_l = p[5] + p[6] * Xl
    v_S = (p[7] * mu_eff / ((L / p[8]) ** 3 + 1.0) + p[9]) * np.tanh(S / p[10]) ** 2
    lac_prod = _softplus_nb(v_S - p[11])
    lac_cons = _softplus_nb(p[11] - v_S) * np.tanh(L / p[12]) ** 2
    v_L = p[13] * lac_prod - p[14] * lac_cons
    return mu_eff, mu_d, mu_l, v_S, v_L


@njit(cache=False)
def _rhs(y, p, dy):
    Xv = max(y[0], 0.0)
    mu_eff, mu_d, mu_l, v_S, v_L = _rates_scalar(Xv, y[1], y[2], y[3], y[4], y[5], p)
    dy[0] = (mu_eff - mu_d - mu_l) * Xv
    dy[1] = mu_d * Xv
    dy[2] = mu_l * Xv
    dy[3] = Xv
    dy[4] = -v_S * Xv
    dy[5] = v_L * Xv


@njit(cache=False)
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
                if y[j] < 0.0:
                    y[j] = 0.0
    return out


# --------------------------------------------------------------------------
# Data handling
# --------------------------------------------------------------------------


@dataclass
class BatchSet:
    ids: list
    y0: np.ndarray  # (nb, 6)
    v0: np.ndarray  # (nb,)
    feeds: np.ndarray  # (nb, n_steps, 4) feed volumes logged at each grid step
    vsnap: np.ndarray  # (nb, n_steps) measured volume, NaN where none
    obs: pd.DataFrame  # measured rows after t=0: batch_idx, step, elapsed_time, targets
    to_predict: pd.DataFrame  # rows needing predictions: batch_idx, step, elapsed_time

    def subset(self, idx) -> "BatchSet":
        idx = np.asarray(idx)
        remap = {old: new for new, old in enumerate(idx)}
        obs = self.obs[self.obs.batch_idx.isin(idx)].copy()
        obs["batch_idx"] = obs.batch_idx.map(remap)
        tp = self.to_predict[self.to_predict.batch_idx.isin(idx)].copy()
        tp["batch_idx"] = tp.batch_idx.map(remap)
        return BatchSet(
            [self.ids[i] for i in idx],
            self.y0[idx],
            self.v0[idx],
            self.feeds[idx],
            self.vsnap[idx],
            obs.reset_index(drop=True),
            tp.reset_index(drop=True),
        )


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    remaining = pd.read_csv(DATA_DIR / "bioreaction_remaining.csv")
    template = pd.read_csv(DATA_DIR / "bioreaction_template.csv")
    remaining["batch_id"] = remaining["batch_id"].astype(str)
    template["batch_id"] = template["batch_id"].astype(str)
    return remaining, template


def build_batches(events: pd.DataFrame, batch_ids: list) -> BatchSet:
    n_steps = int(np.ceil(events["elapsed_time"].max() / DT)) + 2
    nb = len(batch_ids)
    y0 = np.zeros((nb, 6))
    v0 = np.zeros(nb)
    feeds = np.zeros((nb, n_steps, 4))
    vsnap = np.full((nb, n_steps), np.nan)
    obs_rows, pred_rows = [], []
    for i, bid in enumerate(batch_ids):
        g = events[events.batch_id == bid].sort_values("elapsed_time")
        measured = g[g[TARGETS].notna().all(axis=1)]
        first = measured.iloc[0]
        t0 = first.elapsed_time
        y0[i] = [first.vcd, max(first.tcd - first.vcd, 0.0), 0.0, 0.0, first.substrate, first.byproduct]
        v0[i] = first.volume
        steps = np.rint((g.elapsed_time.to_numpy() - t0) / DT).astype(int)
        for j, feed in enumerate(FEEDS):
            mask = g[feed].notna().to_numpy()
            np.add.at(feeds[i, :, j], steps[mask], g.loc[mask, feed].to_numpy())
        vol_mask = g.volume.notna().to_numpy()
        vsnap[i, steps[vol_mask]] = g.loc[vol_mask, "volume"].to_numpy()
        for (_, row), step in zip(g.iterrows(), steps):
            if row.elapsed_time > t0 and row[TARGETS].notna().all():
                obs_rows.append({"batch_idx": i, "step": step, "elapsed_time": row.elapsed_time,
                                 **{t: row[t] for t in TARGETS}})
            if bool(row.get("to_predict", False)):
                pred_rows.append({"batch_idx": i, "step": step, "elapsed_time": row.elapsed_time})
    obs = pd.DataFrame(obs_rows, columns=["batch_idx", "step", "elapsed_time", *TARGETS])
    tp = pd.DataFrame(pred_rows, columns=["batch_idx", "step", "elapsed_time"])
    return BatchSet(list(batch_ids), y0, v0, feeds, vsnap, obs, tp)


def simulate(bs: BatchSet, params: np.ndarray) -> np.ndarray:
    return _simulate(bs.y0, bs.v0, bs.feeds, bs.vsnap, np.asarray(params, dtype=np.float64))


def observe(traj: np.ndarray, batch_idx, steps) -> np.ndarray:
    """Map model states to the four measured outputs."""
    s = traj[np.asarray(batch_idx), np.asarray(steps)]
    return np.column_stack([s[:, 0], s[:, 0] + s[:, 1], s[:, 4], s[:, 5]])


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------


def residual_scale(obs: pd.DataFrame) -> np.ndarray:
    """Per-target scale so all four outputs carry similar weight."""
    return obs[TARGETS].std().to_numpy()


def _fit_one_start(args) -> tuple[float, np.ndarray]:
    """One multi-start run.

    objective "relative": residual = (pred - obs) / obs, which matches the
    leaderboard's equal-weight mean of four MAPEs. loss "soft_l1" makes the
    fit behave more like absolute (MAPE) than squared error.
    objective "scaled": residual = (pred - obs) / std(variable).
    """
    bs, theta0, scale, max_nfev, objective, loss = args
    y_obs = bs.obs[TARGETS].to_numpy()
    b_idx = bs.obs.batch_idx.to_numpy()
    steps = bs.obs.step.to_numpy()
    denom = y_obs if objective == "relative" else np.broadcast_to(scale, y_obs.shape)
    f_scale = 0.05 if objective == "relative" else 0.5

    def resid(theta):
        traj = simulate(bs, np.exp(theta))
        r = ((observe(traj, b_idx, steps) - y_obs) / denom).ravel()
        if not np.all(np.isfinite(r)):
            return np.full(r.shape, 1e3)
        return r

    res = least_squares(resid, theta0, bounds=(np.log(P_LO), np.log(P_HI)), method="trf",
                        x_scale=1.0, max_nfev=max_nfev, loss=loss, f_scale=f_scale)
    return float(res.cost), np.exp(res.x)


def starting_points(n_starts: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    lo, hi = np.log(P_LO) + 1e-9, np.log(P_HI) - 1e-9
    base = np.log(P0)
    starts = [base]
    for _ in range(n_starts - 1):
        starts.append(np.clip(base + rng.uniform(-1.2, 1.2, base.size), lo, hi))
    return starts


def fit_many(jobs: dict, n_starts: int, workers: int, max_nfev: int, objective: str = "relative",
             loss: str = "linear", seed: int = 0) -> dict:
    """Fit several BatchSets (e.g. CV folds) with multi-start, in parallel.

    jobs: {name: BatchSet}. Returns {name: (best_params, best_cost, all_costs)}.
    """
    tasks, keys = [], []
    for name, bs in jobs.items():
        scale = residual_scale(bs.obs)
        for s_i, theta0 in enumerate(starting_points(n_starts, seed)):
            tasks.append((bs, theta0, scale, max_nfev, objective, loss))
            keys.append((name, s_i))
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_fit_one_start, tasks))
    else:
        results = [_fit_one_start(t) for t in tasks]
    out = {}
    for (name, _), (cost, params) in zip(keys, results):
        best = out.get(name)
        costs = (best[2] if best else []) + [cost]
        if best is None or cost < best[1]:
            out[name] = (params, cost, costs)
        else:
            out[name] = (best[0], best[1], costs)
    return out


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def predictions_frame(bs: BatchSet, params: np.ndarray, split: str) -> pd.DataFrame:
    traj = simulate(bs, params)
    pred = observe(traj, bs.obs.batch_idx, bs.obs.step)
    frame = bs.obs[["batch_idx", "elapsed_time"]].copy()
    frame["batch_id"] = [bs.ids[i] for i in frame.batch_idx]
    frame["split"] = split
    long = []
    for j, t in enumerate(TARGETS):
        part = frame.copy()
        part["variable"] = t
        part["observed"] = bs.obs[t].to_numpy()
        part["predicted"] = pred[:, j]
        long.append(part)
    out = pd.concat(long, ignore_index=True).drop(columns="batch_idx")
    out["day"] = np.rint(out.elapsed_time).astype(int)
    out["error"] = out.predicted - out.observed
    out["abs_pct_error"] = 100.0 * out.error.abs() / out.observed.abs()
    return out


def metric_table(pred: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g):
        e = g.error.to_numpy()
        y = g.observed.to_numpy()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return pd.Series({
            "n": len(g),
            "rmse": np.sqrt(np.mean(e**2)),
            "mae": np.mean(np.abs(e)),
            "bias": np.mean(e),
            "mape_pct": g.abs_pct_error.mean(),
            "median_ape_pct": g.abs_pct_error.median(),
            "r2": 1.0 - (e**2).sum() / ss_tot if ss_tot > 0 else np.nan,
            "obs_mean": y.mean(),
        })
    return pred.groupby(by).apply(agg, include_groups=False).reset_index()


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

OBS_COLOR = "#4C78A8"
PRED_COLOR = "#E45756"
FIT_COLOR = "#72B7B2"


def plot_example_trajectories(bs, oof_params_by_batch, full_params, oof, path):
    """Observed points vs out-of-fold simulated trajectory for 6 example batches."""
    peak = oof[oof.variable == "vcd"].groupby("batch_id")["observed"].max().sort_values()
    picks = list(peak.index[np.linspace(0, len(peak) - 1, 6).round().astype(int)])
    t_grid = np.arange(bs.feeds.shape[1]) * DT
    fig, axes = plt.subplots(4, len(picks), figsize=(3.1 * len(picks), 10), sharex=True,
                             constrained_layout=True)
    for c, bid in enumerate(picks):
        b = bs.ids.index(bid)
        one = bs.subset([b])
        traj_oof = simulate(one, oof_params_by_batch[bid])[0]
        traj_fit = simulate(one, full_params)[0]
        curves_oof = np.column_stack([traj_oof[:, 0], traj_oof[:, 0] + traj_oof[:, 1], traj_oof[:, 4], traj_oof[:, 5]])
        curves_fit = np.column_stack([traj_fit[:, 0], traj_fit[:, 0] + traj_fit[:, 1], traj_fit[:, 4], traj_fit[:, 5]])
        obs_b = one.obs
        last = obs_b.elapsed_time.max() + 0.05
        keep = t_grid <= last
        for r, t in enumerate(TARGETS):
            ax = axes[r, c]
            ax.plot(t_grid[keep], curves_fit[keep, r], color=FIT_COLOR, lw=1, ls="--",
                    label="fit incl. this batch")
            ax.plot(t_grid[keep], curves_oof[keep, r], color=PRED_COLOR, lw=1.6,
                    label="prediction (batch held out)")
            ax.scatter([0.0], [[one.y0[0, 0], one.y0[0, 0] + one.y0[0, 1], one.y0[0, 4], one.y0[0, 5]][r]],
                       color="black", marker="s", s=22, zorder=3, label="t=0 input")
            ax.scatter(obs_b.elapsed_time, obs_b[t], color=OBS_COLOR, s=18, zorder=3, label="measured")
            sub = oof[(oof.batch_id == bid) & (oof.variable == t)]
            ax.text(0.03, 0.95, f"MAPE {sub.abs_pct_error.mean():.0f}%", transform=ax.transAxes,
                    va="top", fontsize=8)
            if c == 0:
                ax.set_ylabel(TARGET_LABELS[t])
            if r == 0:
                ax.set_title(f"{bid}  (peak VCD {peak[bid]:.0f})", fontsize=10)
            if r == 3:
                ax.set_xlabel("Time [d]")
    axes[0, 0].legend(fontsize=7, loc="center left")
    fig.suptitle("Sartorius-style model: open-loop predictions from t=0 + feeds for held-out batches "
                 "(6 batches spanning low to high peak VCD)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_population_and_error(oof, path):
    """Top: all batches observed vs predicted. Bottom: error distribution per day."""
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    for c, t in enumerate(TARGETS):
        sub = oof[oof.variable == t]
        ax = axes[0, c]
        for _, g in sub.groupby("batch_id"):
            ax.plot(g.elapsed_time, g.observed, color=OBS_COLOR, alpha=0.25, lw=0.8)
            ax.plot(g.elapsed_time, g.predicted, color=PRED_COLOR, alpha=0.25, lw=0.8)
        day = sub.groupby("day").agg(obs=("observed", "median"), pred=("predicted", "median"))
        ax.plot(day.index, day.obs, color=OBS_COLOR, lw=2.5, label="measured (median)")
        ax.plot(day.index, day.pred, color=PRED_COLOR, lw=2.5, label="predicted (median)")
        ax.set_title(TARGET_LABELS[t])
        ax.set_xlabel("Time [d]")
        if c == 0:
            ax.legend(fontsize=8)
            ax.set_ylabel("All held-out batches")

        ax = axes[1, c]
        days = sorted(sub.day.unique())
        ax.boxplot([sub[sub.day == d].error for d in days], positions=days, widths=0.6,
                   showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor="#F2CFCF", edgecolor=PRED_COLOR),
                   medianprops=dict(color=PRED_COLOR))
        ax.axhline(0, color="black", lw=0.8)
        ax.set_xlabel("Day")
        if c == 0:
            ax.set_ylabel("Signed error (pred - obs)")

        ax = axes[2, c]
        per_day = sub.groupby("day").abs_pct_error.agg(["mean", "median"])
        ax.bar(per_day.index, per_day["mean"], color="#F2CFCF", edgecolor=PRED_COLOR, label="mean APE")
        ax.plot(per_day.index, per_day["median"], color=PRED_COLOR, marker="o", label="median APE")
        ax.set_xlabel("Day")
        ax.set_ylim(bottom=0)
        if c == 0:
            ax.set_ylabel("Absolute % error")
            ax.legend(fontsize=8)
    fig.suptitle("Where the model works: held-out performance over culture time")
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_error_heatmap(oof, path):
    table = oof.pivot_table(index="variable", columns="day", values="abs_pct_error", aggfunc="mean")
    table = table.loc[TARGETS]
    fig, ax = plt.subplots(figsize=(13, 3.6), constrained_layout=True)
    im = ax.imshow(table.to_numpy(), cmap="RdYlGn_r", vmin=0, vmax=40, aspect="auto")
    ax.set_xticks(range(table.shape[1]), table.columns)
    ax.set_yticks(range(table.shape[0]), [TARGET_LABELS[t] for t in table.index])
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            v = table.iat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=9,
                    color="white" if v > 28 else "black")
    ax.set_xlabel("Culture day")
    fig.colorbar(im, ax=ax, label="Mean absolute % error (held-out batches)")
    ax.set_title("Held-out error by variable and day (green = good, red = poor)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_parity(oof, metrics, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), constrained_layout=True)
    for ax, t in zip(axes, TARGETS):
        sub = oof[oof.variable == t]
        sc = ax.scatter(sub.observed, sub.predicted, c=sub.elapsed_time, cmap="viridis", s=12, alpha=0.8)
        lo = min(sub.observed.min(), sub.predicted.min())
        hi = max(sub.observed.max(), sub.predicted.max())
        ax.plot([lo, hi], [lo, hi], color="black", lw=0.8)
        m = metrics[metrics.variable == t].iloc[0]
        ax.set_title(f"{TARGET_LABELS[t]}\nR² {m.r2:.2f} | RMSE {m.rmse:.2f} | MAPE {m.mape_pct:.1f}%")
        ax.set_xlabel("Measured")
        ax.set_ylabel("Predicted")
    fig.colorbar(sc, ax=axes, label="Time [d]")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_internal_states(bs, params, path):
    """Interpretability: hidden states and rates for the median batch."""
    b = len(bs.ids) // 2
    one = bs.subset([b])
    traj = simulate(one, params)[0]
    last = one.obs.elapsed_time.max() + 0.05
    t = np.arange(traj.shape[0]) * DT
    keep = t <= last
    t, traj = t[keep], traj[keep]
    mu_eff, mu_d, mu_l, v_S, v_L = kinetic_rates(*[traj[:, j] for j in range(6)], params)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    ax = axes[0]
    ax.plot(t, traj[:, 0], label="viable Xv")
    ax.plot(t, traj[:, 1], label="dead Xd (in TCD)")
    ax.plot(t, traj[:, 2], label="lysed Xl (invisible)")
    ax.set_ylabel("1e6 cells/mL")
    ax2 = ax.twinx()
    ax2.plot(t, traj[:, 3], color="grey", ls=":", label="biomaterial gamma")
    ax2.set_ylabel("biomaterial [1e6 cells*d/mL]")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="center right", fontsize=8)
    ax.set_title("Cell compartments")
    ax = axes[1]
    ax.plot(t, mu_eff, label="growth mu_eff")
    ax.plot(t, mu_d, label="death mu_d")
    ax.plot(t, mu_l, label="lysis mu_l")
    ax.plot(t, mu_eff - mu_d - mu_l, color="black", lw=1, ls="--", label="net")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("1/d")
    ax.legend(fontsize=8)
    ax.set_title("Specific cell rates")
    ax = axes[2]
    ax.plot(t, v_S, label="glucose uptake v_S")
    ax.plot(t, v_L, label="lactate rate v_L (+prod / -cons)")
    ax.axhline(params[PARAM_NAMES.index("v_ox")], color="grey", ls=":", label="oxidative capacity v_ox")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_ylabel("g/(1e9 cells*d)")
    ax.legend(fontsize=8)
    ax.set_title("Metabolic rates (overflow switch)")
    for ax in axes:
        ax.set_xlabel("Time [d]")
    fig.suptitle(f"Inside the model for {bs.ids[b]} (fit on all training batches)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=7, help="grouped CV folds (0 = skip CV)")
    parser.add_argument("--starts", type=int, default=8, help="multi-start count per fit")
    parser.add_argument("--max-nfev", type=int, default=300, help="max function evals per start")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--objective", choices=["relative", "scaled"], default="relative",
                        help="relative = percentage error (matches the MAPE leaderboard)")
    parser.add_argument("--loss", choices=["linear", "soft_l1"], default="soft_l1")
    parser.add_argument("--out", type=Path, default=OUT_DIR, help="output folder")
    args = parser.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"objective={args.objective} loss={args.loss} -> {out_dir}")
    events, template = load_data()
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    train = build_batches(events, train_ids)
    holdout = build_batches(events, holdout_ids)
    print(f"{len(train_ids)} training batches ({len(train.obs)} measurement rows), "
          f"{len(holdout_ids)} holdout batches ({len(holdout.to_predict)} rows to predict)")

    t_start = time.time()
    simulate(train, P0)  # compile
    t0 = time.time()
    simulate(train, P0)
    print(f"numba compiled in {t0 - t_start:.1f}s; one simulation of all training batches: "
          f"{1000 * (time.time() - t0):.1f} ms")

    jobs = {"full": train}
    folds = []
    if args.folds > 1:
        gkf = GroupKFold(n_splits=args.folds)
        idx = np.arange(len(train_ids))
        for f, (tr_idx, te_idx) in enumerate(gkf.split(idx, groups=idx)):
            jobs[f"fold{f}"] = train.subset(tr_idx)
            folds.append(te_idx)
    t0 = time.time()
    fits = fit_many(jobs, n_starts=args.starts, workers=args.workers, max_nfev=args.max_nfev,
                    objective=args.objective, loss=args.loss)
    print(f"fitted {len(jobs)} models x {args.starts} starts in {time.time() - t0:.0f}s")
    for name, (_, cost, costs) in fits.items():
        print(f"  {name:6s} best cost {cost:9.2f}   starts: {np.round(sorted(costs), 1).tolist()}")

    full_params = fits["full"][0]
    param_table = pd.DataFrame(PARAMS, columns=["name", "initial", "lower", "upper", "unit", "meaning"])
    param_table["fitted_all_training"] = full_params
    param_table["at_bound"] = np.isclose(full_params, P_LO, rtol=1e-3) | np.isclose(full_params, P_HI, rtol=1e-3)
    if folds:
        fold_params = np.array([fits[f"fold{f}"][0] for f in range(len(folds))])
        param_table["cv_fold_min"] = fold_params.min(axis=0)
        param_table["cv_fold_max"] = fold_params.max(axis=0)
    param_table.to_csv(out_dir / "parameters.csv", index=False)
    print("\nFitted parameters (all training batches):")
    print(param_table[["name", "fitted_all_training", "unit", "at_bound"]
                      + (["cv_fold_min", "cv_fold_max"] if folds else [])].to_string(index=False))

    insample = predictions_frame(train, full_params, "in-sample fit")
    frames = [insample]
    oof_params_by_batch = {}
    if folds:
        oof_parts = []
        for f, te_idx in enumerate(folds):
            params_f = fits[f"fold{f}"][0]
            oof_parts.append(predictions_frame(train.subset(te_idx), params_f, "held-out (CV)"))
            for i in te_idx:
                oof_params_by_batch[train_ids[i]] = params_f
        oof = pd.concat(oof_parts, ignore_index=True)
        frames.append(oof)
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(out_dir / "predictions_vs_measured.csv", index=False)

    overall = metric_table(preds, ["split", "variable"])
    by_day = metric_table(preds, ["split", "variable", "day"])
    overall.to_csv(out_dir / "metrics_overall.csv", index=False)
    by_day.to_csv(out_dir / "metrics_by_day.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nOverall metrics:")
    print(overall.round(3).to_string(index=False))

    # Leaderboard score: mean over the 4 variables of each variable's MAPE.
    score = overall.pivot(index="split", columns="variable", values="mape_pct")[TARGETS]
    score["leaderboard_score"] = score.mean(axis=1)
    score.to_csv(out_dir / "leaderboard_style_score.csv")
    print("\nLeaderboard-style score (MAPE %, lower is better):")
    print(score.round(2).to_string())

    if folds:
        oof_metrics = overall[overall.split == "held-out (CV)"]
        table = by_day[by_day.split == "held-out (CV)"].pivot(index="variable", columns="day", values="mape_pct")
        print("\nHeld-out mean absolute % error by day:")
        print(table.loc[TARGETS].round(1).to_string())
        plot_example_trajectories(train, oof_params_by_batch, full_params, oof, out_dir / "01_example_batches.png")
        plot_population_and_error(oof, out_dir / "02_performance_over_time.png")
        plot_error_heatmap(oof, out_dir / "03_error_heatmap_by_day.png")
        plot_parity(oof, oof_metrics, out_dir / "04_parity.png")
    plot_internal_states(train, full_params, out_dir / "05_internal_states.png")

    # Holdout predictions in leaderboard template format
    traj = simulate(holdout, full_params)
    pred = observe(traj, holdout.to_predict.batch_idx, holdout.to_predict.step)
    sub = pd.DataFrame({"batch_id": [holdout.ids[i] for i in holdout.to_predict.batch_idx],
                        "elapsed_time": holdout.to_predict.elapsed_time})
    for j, t in enumerate(TARGETS):
        sub[t] = pred[:, j]
    # Template and event-log times differ by ~1e-9, so match on rounded time.
    sub["t_key"] = sub.elapsed_time.round(6)
    sub = (template[["batch_id", "elapsed_time"]]
           .assign(t_key=template.elapsed_time.round(6))
           .merge(sub.drop(columns="elapsed_time"), on=["batch_id", "t_key"], how="left")
           .drop(columns="t_key"))
    missing = sub[TARGETS].isna().any(axis=1).sum()
    sub.to_csv(out_dir / "holdout_predictions.csv", index=False)
    print(f"\nWrote holdout predictions ({len(sub)} rows, {missing} missing) and figures to {out_dir}")


if __name__ == "__main__":
    main()
