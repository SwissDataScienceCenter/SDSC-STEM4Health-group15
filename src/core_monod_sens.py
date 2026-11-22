"""Sensor-informed variant of core_monod (idea 20): batch-specific parameters from soft sensors.

mu_max, Y_LS (lactate yield) and Y_XS (glucose yield) of batch b are multiplied by
exp(beta . z_b), z_b = standardised [base_tot, base_to3, A_early, A11, A12]
(volume-balance base sensor and bolus decisions, all available for holdout
batches). beta is fitted jointly with the kinetic parameters, then the ODE is
re-simulated per batch.

Original core:

A structurally different stage 1 from the Sartorius-style core, used as an
ensemble member in experiments_v5 (idea 17). States per batch:

    X  viable cells           D  dead (unlysed) cells
    S  glucose                L  lactate
    N  hidden nutrient pool (amino acids etc.), supplied only by feed B

    mu   = mu_max * S/(K_S+S) * N/(K_N+N) * K_IL/(K_IL+L)
    kd   = kd0 + kd_N * K_Nd/(K_Nd+N) + kd_L * L          (starvation + lactate toxicity)
    qS   = mu/Y_XS + m_S
    qLp  = Y_LS * qS * S/(K_SL+S)                         (overflow)
    qLc  = qLc_max * L/(K_Lc+L) * (1 - mu/mu_max)          (consumption once growth slows)
    dX = (mu - kd) X, dD = kd X - k_lys D, dS = -qS X, dL = (qLp - qLc) X, dN = -mu X / Y_XN

Feeds dilute all states; glucose comes with the substrate feed, feed B and feed
D; N comes with feed B. Volume is snapped to the logged value at every sample
(unlogged base addition). TCD = X + D.

    python src/core_monod.py      # writes outputs/core_monod/predictions_vs_measured.csv
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sartorius_model as sm  # noqa: E402  (data loading / BatchSet / event grid)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "core_monod_sens"
ZCOLS = ["base_tot", "base_to3", "A_early", "A11", "A12"]
NB = 3 * len(ZCOLS)  # beta for mu_max, Y_LS, Y_XS
TARGETS = sm.TARGETS
DT = sm.DT

# name, initial value, lower, upper
PARAMS = [
    ("mu_max", 0.9, 0.2, 3.0), ("K_S", 0.2, 1e-3, 5.0), ("K_N", 0.3, 1e-3, 10.0), ("K_IL", 5.0, 0.1, 100.0),
    ("Y_XN", 20.0, 0.5, 500.0), ("N0", 3.0, 0.05, 100.0), ("c_NB", 20.0, 0.1, 500.0),
    ("kd0", 0.01, 1e-4, 0.5), ("kd_N", 0.1, 1e-4, 2.0), ("K_Nd", 0.1, 1e-3, 10.0), ("kd_L", 0.01, 1e-5, 0.5),
    ("k_lys", 0.05, 1e-4, 1.0), ("Y_XS", 15.0, 1.0, 200.0), ("m_S", 0.01, 1e-5, 0.5),
    ("Y_LS", 1.0, 0.01, 5.0), ("K_SL", 0.5, 1e-3, 10.0), ("qLc_max", 0.05, 1e-4, 1.0), ("K_Lc", 0.5, 1e-3, 10.0),
    ("c_S", 300.0, 10.0, 1500.0), ("c_B", 40.0, 0.1, 300.0), ("c_D", 5.0, 1e-3, 1000.0),
]
P0 = np.array([p[1] for p in PARAMS])
LB = np.log([p[2] for p in PARAMS])
UB = np.log([p[3] for p in PARAMS])


@njit(cache=True)
def _rhs(y, p, m, dy):
    X, D, S, L, N = max(y[0], 0.0), max(y[1], 0.0), max(y[2], 0.0), max(y[3], 0.0), max(y[4], 0.0)
    mu_max, K_S, K_N, K_IL, Y_XN = p[0] * m[0], p[1], p[2], p[3], p[4]
    kd0, kd_N, K_Nd, kd_L, k_lys = p[7], p[8], p[9], p[10], p[11]
    Y_XS, m_S, Y_LS, K_SL, qLc_max, K_Lc = p[12] * m[2], p[13], p[14] * m[1], p[15], p[16], p[17]
    mu = mu_max * S / (K_S + S) * N / (K_N + N) * K_IL / (K_IL + L)
    kd = kd0 + kd_N * K_Nd / (K_Nd + N) + kd_L * L
    qS = mu / Y_XS + m_S * S / (0.05 + S)
    qLp = Y_LS * qS * S / (K_SL + S)
    qLc = qLc_max * L / (K_Lc + L) * (1.0 - mu / mu_max)
    dy[0] = (mu - kd) * X
    dy[1] = kd * X - k_lys * D
    dy[2] = -qS * X
    dy[3] = (qLp - qLc) * X
    dy[4] = -mu * X / Y_XN


@njit(cache=True)
def _simulate(y0, v0, feeds, vsnap, p, mult):
    nb, ns, _ = feeds.shape
    out = np.empty((nb, ns, 5))
    c_S, c_B, c_D, c_NB, N0 = p[18], p[19], p[20], p[6], p[5]
    h = DT
    y = np.empty(5)
    tmp = np.empty(5)
    k1 = np.empty(5)
    k2 = np.empty(5)
    k3 = np.empty(5)
    k4 = np.empty(5)
    for b in range(nb):
        y[0], y[1], y[2], y[3], y[4] = y0[b, 0], y0[b, 1], y0[b, 4], y0[b, 5], N0
        V = v0[b]
        for k in range(ns):
            for j in range(5):
                out[b, k, j] = y[j]
            if not np.isnan(vsnap[b, k]):
                if vsnap[b, k] > V:  # unlogged base addition dilutes
                    r = V / vsnap[b, k]
                    for j in range(5):
                        y[j] *= r
                V = vsnap[b, k]
            f_s, f_b, f_c, f_d = feeds[b, k, 0], feeds[b, k, 1], feeds[b, k, 2], feeds[b, k, 3]
            total = f_s + f_b + f_c + f_d
            if total > 0.0:
                v_new = V + total
                r = V / v_new
                for j in range(5):
                    y[j] *= r
                y[2] += (c_S * f_s + c_B * f_b + c_D * f_d) / v_new
                y[4] += c_NB * f_b / v_new
                V = v_new
            if k == ns - 1:
                break
            _rhs(y, p, mult[b], k1)
            for j in range(5):
                tmp[j] = y[j] + 0.5 * h * k1[j]
            _rhs(tmp, p, mult[b], k2)
            for j in range(5):
                tmp[j] = y[j] + 0.5 * h * k2[j]
            _rhs(tmp, p, mult[b], k3)
            for j in range(5):
                tmp[j] = y[j] + h * k3[j]
            _rhs(tmp, p, mult[b], k4)
            for j in range(5):
                y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
                if y[j] < 0.0:
                    y[j] = 0.0
    return out


def observe(bs: sm.BatchSet, p: np.ndarray, rows: pd.DataFrame, beta=None, Z=None) -> np.ndarray:
    beta = np.zeros(NB) if beta is None else beta
    mult = np.exp(Z @ beta.reshape(3, len(ZCOLS)).T)
    sim = _simulate(bs.y0, bs.v0, bs.feeds, bs.vsnap, p, mult)
    b, s = rows.batch_idx.to_numpy(), rows.step.to_numpy()
    st = sim[b, s]
    return np.column_stack([st[:, 0], st[:, 0] + st[:, 1], st[:, 2], st[:, 3]])


def fit(bs: sm.BatchSet, Z, seed: int = 0, max_nfev: int = 400, p_init=None):
    y = bs.obs[TARGETS].to_numpy()
    k = len(P0)

    def resid(q):
        pred = observe(bs, np.exp(q[:k]), bs.obs, q[k:], Z)
        r = (np.maximum(pred, 1e-6) - y) / y
        return np.r_[r.ravel(), 0.3 * q[k:]]  # ridge on beta

    rng = np.random.default_rng(seed)
    best = None
    base = np.log(P0 if p_init is None else p_init)
    for start in range(2):
        q0 = base + (0 if start == 0 else rng.normal(0, 0.2, len(P0)))
        q0 = np.r_[np.clip(q0, LB + 1e-6, UB - 1e-6), np.zeros(NB)]
        try:
            r = least_squares(resid, q0, bounds=(np.r_[LB, -np.ones(NB)], np.r_[UB, np.ones(NB)]), loss="soft_l1",
                              f_scale=0.1, max_nfev=max_nfev, x_scale=1.0)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return np.exp(best.x[:k]), best.x[k:]


def _fold_job(args):
    train, test, Ztr, Zte, p_init = args
    p, beta = fit(train, Ztr, 0, p_init=p_init)
    return (p, beta), observe(test, p, test.obs, beta, Zte)


def main() -> None:
    os.environ.setdefault("NUMBA_NUM_THREADS", "1")
    events, template = sm.load_data()
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    train = sm.build_batches(events, train_ids)
    import hybrid_v3 as h3
    sb, _, _ = h3.sensor_features(events)
    Zall = sb.loc[train_ids, ZCOLS].to_numpy(float)
    p_init = pd.read_csv(ROOT / "outputs" / "core_monod" / "params_full.csv", index_col=0).iloc[:, 0].to_numpy()
    idx = np.arange(len(train_ids))
    jobs, tests = [], []
    for tr, te in GroupKFold(7).split(idx, groups=idx):
        mu, sd = Zall[tr].mean(0), Zall[tr].std(0) + 1e-12
        jobs.append((train.subset(tr), train.subset(te), (Zall[tr] - mu) / sd, (Zall[te] - mu) / sd, p_init))
        tests.append(te)
    with ProcessPoolExecutor(7) as ex:
        res = list(ex.map(_fold_job, jobs))
    frames = []
    for (pb, pred), job in zip(res, jobs):
        frames.append(frame(job[1], pred, "held-out (CV)"))
        print("beta", np.round(pb[1], 2))
    p_full = pb[0]
    preds = pd.concat(frames, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    preds.to_csv(OUT / "predictions_vs_measured.csv", index=False)
    pd.Series(p_full, index=[q[0] for q in PARAMS]).to_csv(OUT / "params_full.csv")
    oof = preds[preds.split == "held-out (CV)"]
    m = oof.groupby("variable").abs_pct_error.mean()[TARGETS]
    print("held-out MAPE:", m.round(2).to_dict(), "mean", round(m.mean(), 2))


def frame(bs, pred, split):
    base = bs.obs[["batch_idx", "elapsed_time"]].copy()
    base["batch_id"] = [bs.ids[i] for i in base.batch_idx]
    parts = []
    for j, t in enumerate(TARGETS):
        part = base.copy()
        part["variable"], part["observed"], part["predicted"] = t, bs.obs[t].to_numpy(), pred[:, j]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).drop(columns="batch_idx")
    out["model"], out["split"] = "monod_sens", split
    out["day"] = np.rint(out.elapsed_time).astype(int)
    out["error"] = out.predicted - out.observed
    out["abs_pct_error"] = 100.0 * out.error.abs() / out.observed.abs()
    return out


if __name__ == "__main__":
    main()
