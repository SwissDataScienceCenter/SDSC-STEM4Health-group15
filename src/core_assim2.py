"""Configurable sensor-conditioned particle core (experiments_v7).

Extends core_assim.py (idea 10 of v6):
  * per-batch multipliers on up to 6 Monod parameters
      0 mu_max, 1 Y_LS (lactate yield), 2 Y_XS (glucose yield), 3 kd0 (death),
      4 qLc_max (lactate consumption), 5 Y_XN (nutrient yield)
  * optional initial-condition particle: VCD0/TCD0 multiplier (t=0 assay noise)
  * configurable glucose-interval noise, base-volume likelihood inflation, particles
  * global Monod fits per fold are cached (outputs/core_assim2_cache)
  * posterior mean of the per-batch log-multipliers is saved as batch features

Particles are drawn from a Gaussian fitted to the library of per-batch
multipliers of the training batches (each fitted to its own measurements), then
weighted by the likelihood of the batch's logged sensors (bolus-rule glucose
intervals on days 7-9/11/12, base volume added by days 3-5).

    python src/core_assim2.py --name core_assim2 --params 0,1,2,3,4 --ic
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy.optimize import least_squares
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_assim as ca  # noqa: E402  (sensor table, base map)
import core_monod as cm  # noqa: E402
import core_runner as cr  # noqa: E402
import sartorius_model as sm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "core_assim2_cache"
TARGETS = sm.TARGETS
DT = sm.DT
T1, T2 = ca.T1, ca.T2
NM = 6  # multiplier slots


@njit(cache=True)
def _rhs(y, p, m, dy):
    X, D, S, L, N = max(y[0], 0.0), max(y[1], 0.0), max(y[2], 0.0), max(y[3], 0.0), max(y[4], 0.0)
    mu_max = p[0] * m[0]
    K_S, K_N, K_IL, Y_XN = p[1], p[2], p[3], p[4] * m[5]
    kd0, kd_N, K_Nd, kd_L, k_lys = p[7] * m[3], p[8], p[9], p[10], p[11]
    Y_XS, m_S, Y_LS, K_SL, qLc_max, K_Lc = p[12] * m[2], p[13], p[14] * m[1], p[15], p[16] * m[4], p[17]
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
def _simulate(y0, v0, feeds, vsnap, p, mult, icm):
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
        m = mult[b]
        y[0], y[1], y[2], y[3], y[4] = y0[b, 0] * icm[b], y0[b, 1] * icm[b], y0[b, 4], y0[b, 5], N0
        V = v0[b]
        for k in range(ns):
            for j in range(5):
                out[b, k, j] = y[j]
            if not np.isnan(vsnap[b, k]):
                if vsnap[b, k] > V:
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
            _rhs(y, p, m, k1)
            for j in range(5):
                tmp[j] = y[j] + 0.5 * h * k1[j]
            _rhs(tmp, p, m, k2)
            for j in range(5):
                tmp[j] = y[j] + 0.5 * h * k2[j]
            _rhs(tmp, p, m, k3)
            for j in range(5):
                tmp[j] = y[j] + h * k3[j]
            _rhs(tmp, p, m, k4)
            for j in range(5):
                y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
                if y[j] < 0.0:
                    y[j] = 0.0
    return out


class Core:
    """Simulator + state layout of the core being assimilated."""

    def __init__(self, name):
        self.name = name
        if name == "monod":
            self.mod, self.sim = cm, _simulate
            self.iS, self.iL = 2, 3
        else:
            import core_twopop as ct
            self.mod, self.sim = ct, ct._simulate_m
            self.iS, self.iL = 3, 4

    def obs4(self, sim, steps):
        st = sim[:, steps]
        if self.name == "monod":
            vcd = st[..., 0]
            tcd = vcd + st[..., 1]
        else:
            vcd = st[..., 0] + st[..., 1]
            tcd = vcd + st[..., 2]
        return np.stack([vcd, tcd, st[..., self.iS], st[..., self.iL]], -1)


CORE = Core("monod")


class Cfg:
    def __init__(self, params, ic, sig, infl, npart, ridge, ic_sd, est):
        self.params, self.ic, self.sig, self.infl = list(params), ic, sig, infl
        self.npart, self.ridge, self.ic_sd, self.est = npart, ridge, ic_sd, est
        self.dim = len(self.params) + (1 if ic else 0)

    def expand(self, Q):
        """(n, dim) log-parameters -> (n, NM) multipliers, (n,) IC multipliers."""
        Q = np.atleast_2d(Q)
        mult = np.ones((len(Q), NM))
        for j, k in enumerate(self.params):
            mult[:, k] = np.exp(Q[:, j])
        icm = np.exp(Q[:, -1]) if self.ic else np.ones(len(Q))
        return mult, icm


def obs4(sim, steps):
    st = sim[:, steps]
    return np.stack([st[..., 0], st[..., 0] + st[..., 1], st[..., 2], st[..., 3]], -1)


def fit_library(bs, p, cfg):
    lib = []
    for i in range(len(bs.ids)):
        one = bs.subset([i])
        y = one.obs[TARGETS].to_numpy()
        steps = one.obs.step.to_numpy()

        def resid(q):
            mult, icm = cfg.expand(q)
            pred = CORE.obs4(CORE.sim(one.y0, one.v0, one.feeds, one.vsnap, p, mult, icm), steps)[0]
            pen = cfg.ridge * q.copy()
            if cfg.ic:
                pen[-1] = q[-1] / cfg.ic_sd * 0.1
            return np.r_[((np.maximum(pred, 1e-6) - y) / y).ravel(), pen]

        r = least_squares(resid, np.zeros(cfg.dim), bounds=(-1.5, 1.5), loss="soft_l1", f_scale=0.1, max_nfev=300)
        lib.append(r.x)
    return np.array(lib)


def assimilate(bs, rows, p, lib, sens, bmap, cfg, seed=0):
    rng = np.random.default_rng(seed)
    mu, cov = lib.mean(0), np.cov(lib.T) + 1e-4 * np.eye(cfg.dim)
    if cfg.ic:  # IC prior: t=0 assay noise, not the library spread
        mu[-1] = 0.0
        cov[-1, :] = cov[:, -1] = 0.0
        cov[-1, -1] = cfg.ic_sd ** 2
    preds = np.zeros((len(rows), 4))
    post = np.zeros((len(bs.ids), cfg.dim))
    ess = []
    for i, bid in enumerate(bs.ids):
        Q = np.vstack([mu, rng.multivariate_normal(mu, cov, cfg.npart - 1)])
        mult, icm = cfg.expand(Q)
        n = cfg.npart
        sim = CORE.sim(np.repeat(bs.y0[i:i + 1], n, 0), np.repeat(bs.v0[i:i + 1], n, 0),
                        np.repeat(bs.feeds[i:i + 1], n, 0), np.repeat(bs.vsnap[i:i + 1], n, 0), p, mult, icm)
        logw = np.zeros(n)
        ints, base = sens[bid]
        for step, lo, hi in ints:
            s = np.log(np.maximum(sim[:, step, CORE.iS], 1e-6))
            ph = norm.cdf((np.log(hi) - s) / cfg.sig) if np.isfinite(hi) else 1.0
            pl = norm.cdf((np.log(lo) - s) / cfg.sig) if lo > 0 else 0.0
            logw += np.log(np.maximum(ph - pl, 1e-12))
        ri = rows[rows.batch_idx == i]
        if cfg.infl > 0:
            for K, (a, b, sd) in zip((3, 4, 5), bmap):
                step = int(ri.step.to_numpy()[np.abs(ri.elapsed_time.to_numpy() - K).argmin()])
                V = bs.vsnap[i, step] if np.isfinite(bs.vsnap[i, step]) else bs.v0[i]
                mass = sim[:, step, CORE.iL] * V - bs.y0[i, 5] * bs.v0[i]
                logw += norm.logpdf(base[K - 3], a + b * mass, sd * cfg.infl)
        w = np.exp(logw - logw.max())
        w /= w.sum()
        ess.append(1.0 / np.sum(w ** 2))
        post[i] = w @ Q
        o4 = np.log(np.maximum(CORE.obs4(sim, ri.step.to_numpy()), 1e-6))  # (n, nr, 4)
        if cfg.est == "geo":
            pr = np.exp(np.tensordot(w, o4, 1))
        else:  # MAPE-optimal: weighted median of obs with weights w / obs
            pr = np.zeros(o4.shape[1:])
            for a in range(o4.shape[1]):
                for v in range(4):
                    z = np.exp(o4[:, a, v])
                    ww = w / z
                    order = np.argsort(z)
                    cw = np.cumsum(ww[order])
                    pr[a, v] = z[order][np.searchsorted(cw, 0.5 * cw[-1])]
        preds[(rows.batch_idx == i).to_numpy()] = pr
    return preds, np.array(ess), post


def global_fit(fold, train):
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (f"p_{fold}.npy" if CORE.name == "monod" else f"p_{CORE.name}_{fold}.npy")
    if path.exists():
        return np.load(path)
    p = cr.fit(CORE.mod, train, seed=0, starts=3)
    np.save(path, p)
    return p


def _job(args):
    fold, train, test, rows, sens, cfg, core = args
    os.environ["NUMBA_NUM_THREADS"] = "1"
    global CORE
    CORE = Core(core)
    p = global_fit(fold, train)
    lib = fit_library(train, p, cfg)
    bmap = ca.base_map(train, sens)
    return assimilate(test, rows, p, lib, sens, bmap, cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--params", default="0,1,2")
    ap.add_argument("--ic", action="store_true")
    ap.add_argument("--ic-sd", type=float, default=0.05)
    ap.add_argument("--sig", type=float, default=0.08)
    ap.add_argument("--infl", type=float, default=1.732)
    ap.add_argument("--npart", type=int, default=400)
    ap.add_argument("--ridge", type=float, default=0.5)
    ap.add_argument("--est", default="geo", choices=["geo", "wmed"])
    ap.add_argument("--core", default="monod", choices=["monod", "twopop"])
    a = ap.parse_args()
    cfg = Cfg([int(k) for k in a.params.split(",") if k != ""], a.ic, a.sig, a.infl, a.npart, a.ridge, a.ic_sd, a.est)
    out = ROOT / "outputs" / a.name
    out.mkdir(parents=True, exist_ok=True)

    events, template = sm.load_data()
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    train = sm.build_batches(events, train_ids)
    holdout = sm.build_batches(events, holdout_ids)
    sens = ca.sensor_table(events, train_ids + holdout_ids)
    idx = np.arange(len(train_ids))
    folds = list(GroupKFold(7).split(idx, groups=idx))
    jobs = [(f, train.subset(tr), train.subset(te), train.subset(te).obs, sens, cfg, a.core)
            for f, (tr, te) in enumerate(folds)]
    jobs.append(("full", train, holdout, holdout.to_predict, sens, cfg, a.core))
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(_job, jobs))
    frames, posts = [], []
    for (tr, te), (pred, ess, post) in zip(folds, res[:-1]):
        test = train.subset(te)
        frames.append(cr.frame(test, pred, a.name, "held-out (CV)"))
        posts.append(pd.DataFrame(post, index=test.ids))
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(out / "predictions_vs_measured.csv", index=False)
    posts.append(pd.DataFrame(res[-1][2], index=holdout.ids))
    P = pd.concat(posts)
    P.columns = [f"post{j}" for j in range(P.shape[1])]
    P.rename_axis("batch_id").to_csv(out / "posterior_params.csv")
    tp = holdout.to_predict
    sub = pd.DataFrame({"batch_id": [holdout.ids[i] for i in tp.batch_idx], "t_key": tp.elapsed_time.round(6)})
    for j, t in enumerate(TARGETS):
        sub[t] = res[-1][0][:, j]
    sub = (template[["batch_id", "elapsed_time"]].assign(t_key=template.elapsed_time.round(6))
           .merge(sub, on=["batch_id", "t_key"], how="left").drop(columns="t_key"))
    assert sub[TARGETS].notna().all().all()
    sub.to_csv(out / "holdout_predictions.csv", index=False)
    m = preds.groupby("variable").abs_pct_error.mean()[TARGETS]
    ess = np.concatenate([r[1] for r in res[:-1]])
    print(a.name, vars(a), "| stage-1 held-out MAPE:", m.round(2).to_dict(), "mean", round(m.mean(), 2),
          "| ESS median", round(float(np.median(ess)), 1))


if __name__ == "__main__":
    main()
