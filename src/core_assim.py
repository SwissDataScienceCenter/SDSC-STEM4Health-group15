"""Sensor-conditioned particle ensemble on the Monod core (experiments_v6, idea 10).

Data assimilation instead of regression on the sensors:
1. Global Monod parameters fitted on the training batches (per CV fold).
2. Each training batch gets its own multipliers for mu_max, lactate yield and
   glucose yield, fitted to its full measurements -> a library of batch behaviours.
3. For a new batch, 400 particles are drawn from a Gaussian fitted to that library
   and simulated with the batch's t=0 state and feed log. Each particle is weighted
   by the likelihood of the batch's logged sensors:
     * glucose interval at every bolus decision (days 7-9, 11, 12), exact rule
       < 0.965 / < 1.96, lognormal assay noise sigma = 0.08;
     * base volume added by days 3, 4, 5 versus the particle's lactate produced,
       via a linear map fitted on the training batches.
4. Prediction = weighted geometric mean of the particle trajectories.

    python src/core_assim.py      # writes outputs/core_assim/{predictions_vs_measured,holdout_predictions}.csv
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_monod as cm  # noqa: E402
import core_monod_sens as cms  # noqa: E402  (simulator with per-batch multipliers)
import core_runner as cr  # noqa: E402
import hybrid_v3 as h3  # noqa: E402
import sartorius_model as sm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "core_assim"
TARGETS = sm.TARGETS
T1, T2, SIG_G = 0.965, 1.96, 0.08
N_PART = 400


def sim_mult(bs, p, mult):
    return cms._simulate(bs.y0, bs.v0, bs.feeds, bs.vsnap, p, mult)


def tile(bs, i, n):
    """BatchSet with batch i repeated n times."""
    return sm.BatchSet([bs.ids[i]] * n, np.repeat(bs.y0[i:i + 1], n, 0), np.repeat(bs.v0[i:i + 1], n, 0),
                       np.repeat(bs.feeds[i:i + 1], n, 0), np.repeat(bs.vsnap[i:i + 1], n, 0), bs.obs, bs.to_predict)


def sensor_table(events, ids):
    """Per batch: list of (step, lo, hi) glucose intervals and base_to3/4/5."""
    sb, _, _ = h3.sensor_features(events)
    out = {}
    for bid in ids:
        g = events[events.batch_id == bid].sort_values("elapsed_time")
        st = g[g.volume.notna()]
        t0 = st.elapsed_time.iloc[0]
        fs = g[g.feed_substrate.notna()]
        ints = []
        times = st.elapsed_time.to_numpy()
        for k, (t, V) in enumerate(zip(times, st.volume.to_numpy())):
            if k == 0:
                continue
            nxt = times[k + 1] if k + 1 < len(times) else 99
            A = fs[(fs.elapsed_time > t) & (fs.elapsed_time < nxt)].feed_substrate.sum() / V
            day = int(round(t))
            step = int(np.rint((t - t0) / sm.DT))
            if A > 0.005:
                ints.append((step, 0.0, T1))
            elif A > 0.001:
                ints.append((step, T1, T2))
            elif day in (7, 8, 9):
                ints.append((step, T2, np.inf))
        out[bid] = (ints, sb.loc[bid, ["base_to3", "base_to4", "base_to5"]].to_numpy(float))
    return out


def lactate_mass(traj_L, V_at, L0V0):
    return traj_L * V_at - L0V0


def base_map(bs, sens):
    """Linear map: base_toK ~ a_K + b_K * (observed lactate mass produced by day K)."""
    coefs = []
    for K in (3, 4, 5):
        xs, ys = [], []
        for i, bid in enumerate(bs.ids):
            o = bs.obs[bs.obs.batch_idx == i]
            r = o.iloc[(np.abs(o.elapsed_time - K)).argmin()]
            V = bs.vsnap[i, int(r.step)]
            xs.append(r.byproduct * V - bs.y0[i, 5] * bs.v0[i])
            ys.append(sens[bid][1][K - 3])
        b, a = np.polyfit(xs, ys, 1)
        sd = np.std(np.array(ys) - (a + b * np.array(xs)))
        coefs.append((a, b, sd))
    return coefs


def fit_library(bs, p):
    """Per-batch log-multipliers (mu_max, Y_LS, Y_XS) given global p."""
    lib = []
    for i in range(len(bs.ids)):
        one = bs.subset([i])
        y = one.obs[TARGETS].to_numpy()

        def resid(q):
            pred = _obs_mult(one, p, q)
            return np.r_[((np.maximum(pred, 1e-6) - y) / y).ravel(), 0.5 * q]

        r = least_squares(resid, np.zeros(3), bounds=(-1.5, 1.5), loss="soft_l1", f_scale=0.1, max_nfev=200)
        lib.append(r.x)
    return np.array(lib)


def _obs_mult(bs, p, q, rows=None):
    mult = np.repeat(np.exp(q)[None, :], len(bs.ids), 0) if np.ndim(q) == 1 else np.exp(q)
    sim = sim_mult(bs, p, mult)
    rows = bs.obs if rows is None else rows
    st = sim[rows.batch_idx.to_numpy(), rows.step.to_numpy()]
    return np.column_stack([st[:, 0], st[:, 0] + st[:, 1], st[:, 2], st[:, 3]])


def assimilate(bs, rows, p, lib, sens, bmap, seed=0):
    """Weighted particle prediction for every batch in bs at the given rows."""
    rng = np.random.default_rng(seed)
    mu, cov = lib.mean(0), np.cov(lib.T) + 1e-4 * np.eye(3)
    preds = np.zeros((len(rows), 4))
    ess = []
    for i, bid in enumerate(bs.ids):
        Q = np.vstack([np.zeros(3), rng.multivariate_normal(mu, cov, N_PART - 1)])
        T = tile(bs, i, N_PART)
        sim = sim_mult(T, p, np.exp(Q))  # (N_PART, ns, 5)
        logw = np.zeros(N_PART)
        ints, base = sens[bid]
        for step, lo, hi in ints:
            s = np.log(np.maximum(sim[:, step, 2], 1e-6))
            ph = norm.cdf((np.log(hi) - s) / SIG_G) if np.isfinite(hi) else 1.0
            pl = norm.cdf((np.log(lo) - s) / SIG_G) if lo > 0 else 0.0
            logw += np.log(np.maximum(ph - pl, 1e-12))
        ri = rows[rows.batch_idx == i]
        for K, (a, b, sd) in zip((3, 4, 5), bmap):
            step = int(ri.step.to_numpy()[np.abs(ri.elapsed_time.to_numpy() - K).argmin()])
            V = bs.vsnap[i, step] if np.isfinite(bs.vsnap[i, step]) else bs.v0[i]
            m = sim[:, step, 3] * V - bs.y0[i, 5] * bs.v0[i]
            logw += norm.logpdf(base[K - 3], a + b * m, sd * np.sqrt(3))
        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess.append(1.0 / np.sum(w**2))
        r = rows[rows.batch_idx == i]
        st = sim[:, r.step.to_numpy()]  # (N_PART, nr, 5)
        obs4 = np.stack([st[..., 0], st[..., 0] + st[..., 1], st[..., 2], st[..., 3]], -1)
        preds[(rows.batch_idx == i).to_numpy()] = np.exp(np.tensordot(w, np.log(np.maximum(obs4, 1e-6)), 1))
    return preds, np.array(ess)


def _fold(args):
    train, test, test_rows, sens, seed = args
    os.environ["NUMBA_NUM_THREADS"] = "1"
    p = cr.fit(cm, train, seed=0, starts=3)
    lib = fit_library(train, p)
    bmap = base_map(train, sens)
    pred, ess = assimilate(test, test_rows, p, lib, sens, bmap, seed)
    base_pred = cm.observe(test, p, test_rows)
    return pred, ess, base_pred


def main():
    events, template = sm.load_data()
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    train = sm.build_batches(events, train_ids)
    holdout = sm.build_batches(events, holdout_ids)
    sens = sensor_table(events, train_ids + holdout_ids)
    idx = np.arange(len(train_ids))
    folds = list(GroupKFold(7).split(idx, groups=idx))
    jobs = [(train.subset(tr), train.subset(te), train.subset(te).obs, sens, 0) for tr, te in folds]
    jobs.append((train, holdout, holdout.to_predict, sens, 0))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_fold, jobs))
    frames, frames0 = [], []
    for (tr, te), (pred, ess, base_pred) in zip(folds, res[:-1]):
        frames.append(cr.frame(train.subset(te), pred, "assim", "held-out (CV)"))
        frames0.append(cr.frame(train.subset(te), base_pred, "monod_refit", "held-out (CV)"))
        print("ESS median", np.median(ess).round(1), "min", ess.min().round(1))
    OUT.mkdir(parents=True, exist_ok=True)
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(OUT / "predictions_vs_measured.csv", index=False)
    for name, fr in (("assim", preds), ("monod (same fits, no assimilation)", pd.concat(frames0))):
        m = fr.groupby("variable").abs_pct_error.mean()[TARGETS]
        print(name, "held-out MAPE:", m.round(2).to_dict(), "mean", round(m.mean(), 2))
    tp = holdout.to_predict
    sub = pd.DataFrame({"batch_id": [holdout.ids[i] for i in tp.batch_idx], "t_key": tp.elapsed_time.round(6)})
    for j, t in enumerate(TARGETS):
        sub[t] = res[-1][0][:, j]
    sub = (template[["batch_id", "elapsed_time"]].assign(t_key=template.elapsed_time.round(6))
           .merge(sub, on=["batch_id", "t_key"], how="left").drop(columns="t_key"))
    assert sub[TARGETS].notna().all().all()
    sub.to_csv(OUT / "holdout_predictions.csv", index=False)


if __name__ == "__main__":
    main()
