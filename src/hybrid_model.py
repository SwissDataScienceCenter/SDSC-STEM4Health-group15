"""Hybrid model #1: Sartorius-style mechanistic core + small neural net on the rates.

Shortlist option #1 (reports/Bioreactor model shortlist.md). This is a serial
grey-box hybrid in the style of Bayer et al. 2021 (one hidden layer of 4 tanh
neurons, trained with a Levenberg-Marquardt-type least-squares solver):

    mass balances, bolus feeds, death, lysis, biomaterial      -> mechanistic
    growth rate mu_eff and lactate rate v_L                    -> mechanistic
                                                                  x / + learned correction

    mu_eff = mu_eff,mech * exp(NN_mu(x))         multiplicative, stays >= 0
    v_L    = v_L,mech    + LAC_SCALE * NN_L(x)   additive, may flip sign

Glucose uptake is computed from the corrected mu_eff with the mechanistic law,
so a growth correction also changes glucose consumption and lactate overflow
consistently. The mechanistic parameters come from sartorius_model.py
(relative-error soft-L1 fit) and are frozen; only the network is trained, with
its output layer initialised at zero so training starts exactly at the
mechanistic solution, and a ridge penalty that pulls it back where data are
silent. Each fold uses an ensemble of networks (different random seeds) whose
predictions are averaged.

Network inputs (each roughly centred and scaled to ~[-1, 1]):
    variant "states":        S, L, Xv, biomaterial gamma, culture time
    variant "states_feeds":  the above + cumulative feed B / V,
                             cumulative feed C / V, initial VCD

Evaluation is identical to sartorius_model.py: the same 7 grouped folds over
the 49 training batches; each held-out batch is simulated open-loop from its
t=0 measurement plus its feeds and volumes; the score is the leaderboard's
mean of the four per-variable MAPEs.

Run from the repo root:

    python src/hybrid_model.py
    python src/hybrid_model.py --variants states_feeds --seeds 6
"""

from __future__ import annotations

import os

for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numba import njit  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

import sartorius_model as sm  # noqa: E402

OUT_DIR = sm.ROOT / "outputs" / "hybrid"
TARGETS = sm.TARGETS
DT = sm.DT
N_FEAT = 8
N_OUT = 3  # [growth, glucose (unused in option #1), lactate]
LAC_SCALE = 0.2  # [g/(1e9 cells d)] size of one unit of lactate correction
FEATURE_NAMES = ["S", "L", "Xv", "gamma", "time", "cumB/V", "cumC/V", "VCD0"]
VARIANTS = {
    "states": [0, 1, 2, 3, 4],
    "states_feeds": [0, 1, 2, 3, 4, 5, 6, 7],
}
OUTPUTS_USED = [0, 2]  # correct growth and lactate (option #1); index 1 = glucose


# --------------------------------------------------------------------------
# Hybrid simulator (numba)
# --------------------------------------------------------------------------


@njit(cache=False)
def _features(Xv, gamma, S, L, t, fb_frac, fc_frac, vcd0, x):
    x[0] = (S - 2.5) / 1.5
    x[1] = (L - 1.5) / 1.0
    x[2] = (Xv - 20.0) / 12.0
    x[3] = (gamma - 100.0) / 80.0
    x[4] = (t - 6.5) / 4.0
    x[5] = (fb_frac - 0.15) / 0.12
    x[6] = (fc_frac - 0.0008) / 0.0006
    x[7] = (vcd0 - 1.09) / 0.12


@njit(cache=False)
def _net(x, W1, b1, W2, b2, out):
    for o in range(N_OUT):
        out[o] = b2[o]
    for h in range(b1.shape[0]):
        a = b1[h]
        for i in range(N_FEAT):
            a += W1[h, i] * x[i]
        a = np.tanh(a)
        for o in range(N_OUT):
            out[o] += W2[o, h] * a


@njit(cache=False)
def _hybrid_rates(y, t, fb_frac, fc_frac, vcd0, p, W1, b1, W2, b2, x, o):
    """Mechanistic rates of sartorius_model.kinetic_rates with NN corrections."""
    Xv = max(y[0], 0.0)
    Xl = max(y[2], 0.0)
    gamma = max(y[3], 0.0)
    S = max(y[4], 0.0)
    L = max(y[5], 0.0)
    _features(Xv, gamma, S, L, t, fb_frac, fc_frac, vcd0, x)
    _net(x, W1, b1, W2, b2, o)
    mu_eff = p[0] * np.tanh(S / p[1]) ** 2 / ((gamma / p[2]) ** 3 + 1.0)
    mu_eff *= np.exp(min(max(o[0], -3.0), 3.0))
    mu_d = p[3] + p[4] * gamma
    mu_l = p[5] + p[6] * Xl
    v_S = (p[7] * mu_eff / ((L / p[8]) ** 3 + 1.0) + p[9]) * np.tanh(S / p[10]) ** 2
    v_S *= np.exp(min(max(o[1], -3.0), 3.0))
    lac_prod = sm._softplus_nb(v_S - p[11])
    lac_cons = sm._softplus_nb(p[11] - v_S) * np.tanh(L / p[12]) ** 2
    v_L = p[13] * lac_prod - p[14] * lac_cons + LAC_SCALE * o[2]
    if v_L < 0.0:  # lactate consumption must fade out as lactate runs out
        v_L *= np.tanh(L / 0.1) ** 2
    return Xv, mu_eff, mu_d, mu_l, v_S, v_L


@njit(cache=False)
def _rhs(y, t, fb_frac, fc_frac, vcd0, p, W1, b1, W2, b2, x, o, dy):
    Xv, mu_eff, mu_d, mu_l, v_S, v_L = _hybrid_rates(y, t, fb_frac, fc_frac, vcd0, p, W1, b1, W2, b2, x, o)
    dy[0] = (mu_eff - mu_d - mu_l) * Xv
    dy[1] = mu_d * Xv
    dy[2] = mu_l * Xv
    dy[3] = Xv
    dy[4] = -v_S * Xv
    dy[5] = v_L * Xv


@njit(cache=False)
def _simulate(y0, v0, feeds, vsnap, p, W1, b1, W2, b2):
    """Same event handling as sartorius_model._simulate, plus the NN.

    Returns states (nb, n_steps, 6) and aux (nb, n_steps, 2) = [cumB/V, cumC/V].
    """
    nb, ns, _ = feeds.shape
    out = np.empty((nb, ns, 6))
    aux = np.empty((nb, ns, 2))
    c_S, c_B, c_D = p[15], p[16], p[17]
    h = DT
    y = np.empty(6)
    tmp = np.empty(6)
    k1 = np.empty(6)
    k2 = np.empty(6)
    k3 = np.empty(6)
    k4 = np.empty(6)
    x = np.empty(N_FEAT)
    o = np.empty(N_OUT)
    for b in range(nb):
        for j in range(6):
            y[j] = y0[b, j]
        V = v0[b]
        vcd0 = y0[b, 0]
        cum_b = 0.0
        cum_c = 0.0
        for k in range(ns):
            for j in range(6):
                out[b, k, j] = y[j]
            aux[b, k, 0] = cum_b / V
            aux[b, k, 1] = cum_c / V
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
                cum_b += f_b
                cum_c += f_c
            if k == ns - 1:
                break
            t = k * h
            fbf = cum_b / V
            fcf = cum_c / V
            _rhs(y, t, fbf, fcf, vcd0, p, W1, b1, W2, b2, x, o, k1)
            for j in range(6):
                tmp[j] = y[j] + 0.5 * h * k1[j]
            _rhs(tmp, t + 0.5 * h, fbf, fcf, vcd0, p, W1, b1, W2, b2, x, o, k2)
            for j in range(6):
                tmp[j] = y[j] + 0.5 * h * k2[j]
            _rhs(tmp, t + 0.5 * h, fbf, fcf, vcd0, p, W1, b1, W2, b2, x, o, k3)
            for j in range(6):
                tmp[j] = y[j] + h * k3[j]
            _rhs(tmp, t + h, fbf, fcf, vcd0, p, W1, b1, W2, b2, x, o, k4)
            for j in range(6):
                y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
                if y[j] < 0.0:
                    y[j] = 0.0
    return out, aux


# --------------------------------------------------------------------------
# Network parameter packing
# --------------------------------------------------------------------------


class NetSpec:
    """Maps the trainable vector theta to full weight arrays (masked inputs/outputs = 0)."""

    def __init__(self, feat_idx, hidden, out_idx=OUTPUTS_USED):
        self.feat_idx = list(feat_idx)
        self.out_idx = list(out_idx)
        self.hidden = hidden
        self.sizes = [hidden * len(self.feat_idx), hidden, len(self.out_idx) * hidden, len(self.out_idx)]
        self.n = sum(self.sizes)

    def unpack(self, theta):
        H = self.hidden
        a, b, c, _ = np.cumsum(self.sizes)
        W1 = np.zeros((H, N_FEAT))
        W1[:, self.feat_idx] = theta[:a].reshape(H, len(self.feat_idx))
        b1 = theta[a:b].copy()
        W2 = np.zeros((N_OUT, H))
        W2[self.out_idx, :] = theta[b:c].reshape(len(self.out_idx), H)
        b2 = np.zeros(N_OUT)
        b2[self.out_idx] = theta[c:]
        return W1, b1, W2, b2

    def init(self, seed):
        rng = np.random.default_rng(seed)
        theta = np.zeros(self.n)
        a, b = self.sizes[0], self.sizes[0] + self.sizes[1]
        theta[:a] = rng.normal(0.0, 0.5, a)  # input weights
        theta[a:b] = rng.normal(0.0, 0.1, self.sizes[1])  # hidden biases
        return theta  # output layer = 0 -> starts at the mechanistic model


def simulate(bs: sm.BatchSet, mech_params, spec: NetSpec, theta):
    W1, b1, W2, b2 = spec.unpack(theta)
    return _simulate(bs.y0, bs.v0, bs.feeds, bs.vsnap, np.asarray(mech_params, dtype=np.float64), W1, b1, W2, b2)


def predict_obs(bs, mech_params, spec, thetas) -> np.ndarray:
    """Ensemble-mean prediction at the measured rows, shape (n_obs, 4)."""
    preds = [sm.observe(simulate(bs, mech_params, spec, th)[0], bs.obs.batch_idx, bs.obs.step) for th in thetas]
    return np.mean(preds, axis=0)


def predict_rows(bs, mech_params, spec, thetas, batch_idx, steps) -> np.ndarray:
    preds = [sm.observe(simulate(bs, mech_params, spec, th)[0], batch_idx, steps) for th in thetas]
    return np.mean(preds, axis=0)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _fit_net(args):
    """Train one network (one seed) on top of frozen mechanistic parameters."""
    bs, mech_params, feat_idx, hidden, lam, seed, max_nfev, out_idx = args
    spec = NetSpec(feat_idx, hidden, out_idx)
    y_obs = bs.obs[TARGETS].to_numpy()
    b_idx = bs.obs.batch_idx.to_numpy()
    steps = bs.obs.step.to_numpy()
    ridge = np.sqrt(lam)

    def resid(theta):
        traj, _ = simulate(bs, mech_params, spec, theta)
        r = ((sm.observe(traj, b_idx, steps) - y_obs) / y_obs).ravel()
        if not np.all(np.isfinite(r)):
            r = np.full(r.shape, 1e3)
        return np.concatenate([r, ridge * theta])

    res = least_squares(resid, spec.init(seed), method="trf", loss="soft_l1", f_scale=0.05,
                        x_scale=1.0, max_nfev=max_nfev)
    return float(res.cost), res.x


# --------------------------------------------------------------------------
# Evaluation helpers
# --------------------------------------------------------------------------


def frame_from_pred(bs: sm.BatchSet, pred: np.ndarray, model: str, split: str) -> pd.DataFrame:
    base = bs.obs[["batch_idx", "elapsed_time"]].copy()
    base["batch_id"] = [bs.ids[i] for i in base.batch_idx]
    parts = []
    for j, t in enumerate(TARGETS):
        part = base.copy()
        part["variable"] = t
        part["observed"] = bs.obs[t].to_numpy()
        part["predicted"] = pred[:, j]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True).drop(columns="batch_idx")
    out["model"] = model
    out["split"] = split
    out["day"] = np.rint(out.elapsed_time).astype(int)
    out["error"] = out.predicted - out.observed
    out["abs_pct_error"] = 100.0 * out.error.abs() / out.observed.abs()
    return out


def leaderboard_table(preds: pd.DataFrame) -> pd.DataFrame:
    mape = preds.groupby(["model", "split", "variable"]).abs_pct_error.mean().unstack("variable")[TARGETS]
    mape["leaderboard_score"] = mape.mean(axis=1)
    return mape.round(2)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

MECH_COLOR = "#9D9D9D"
HYB_COLOR = "#E45756"
OBS_COLOR = "#4C78A8"


def plot_examples(train, oof_curves, oof, best, path):
    peak = oof[(oof.variable == "vcd") & (oof.model == best)].groupby("batch_id")["observed"].max().sort_values()
    picks = list(peak.index[np.linspace(0, len(peak) - 1, 6).round().astype(int)])
    t_grid = np.arange(train.feeds.shape[1]) * DT
    fig, axes = plt.subplots(4, len(picks), figsize=(3.1 * len(picks), 10), sharex=True, constrained_layout=True)
    for c, bid in enumerate(picks):
        b = train.ids.index(bid)
        obs_b = train.obs[train.obs.batch_idx == b]
        keep = t_grid <= obs_b.elapsed_time.max() + 0.05
        for r, t in enumerate(TARGETS):
            ax = axes[r, c]
            ax.plot(t_grid[keep], oof_curves["mechanistic"][bid][keep, r], color=MECH_COLOR, lw=1.3,
                    ls="--", label="mechanistic (Sartorius)")
            ax.plot(t_grid[keep], oof_curves[best][bid][keep, r], color=HYB_COLOR, lw=1.6, label=f"hybrid ({best})")
            ax.scatter(obs_b.elapsed_time, obs_b[t], color=OBS_COLOR, s=16, zorder=3, label="measured")
            lines = []
            for model in ["mechanistic", best]:
                sub = oof[(oof.batch_id == bid) & (oof.variable == t) & (oof.model == model)]
                lines.append(f"{'mech' if model == 'mechanistic' else 'hyb'} {sub.abs_pct_error.mean():.0f}%")
            ax.text(0.03, 0.95, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7.5)
            if c == 0:
                ax.set_ylabel(sm.TARGET_LABELS[t])
            if r == 0:
                ax.set_title(f"{bid} (peak VCD {peak[bid]:.0f})", fontsize=10)
            if r == 3:
                ax.set_xlabel("Time [d]")
    axes[0, 0].legend(fontsize=7, loc="center left")
    fig.suptitle("Held-out batches: hybrid vs mechanistic (both predicted from t=0 + feeds only; MAPE per panel)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_mape_by_day(oof, models, path):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    colors = {"mechanistic": MECH_COLOR}
    palette = ["#E45756", "#F58518", "#54A24B"]
    for m, col in zip([m for m in models if m != "mechanistic"], palette):
        colors[m] = col
    for ax, t in zip(axes, TARGETS):
        for model in models:
            sub = oof[(oof.variable == t) & (oof.model == model)]
            per_day = sub.groupby("day").abs_pct_error.mean()
            ax.plot(per_day.index, per_day.values, marker="o", color=colors[model],
                    label=f"{model} ({sub.abs_pct_error.mean():.1f}%)")
        ax.set_title(sm.TARGET_LABELS[t])
        ax.set_xlabel("Day")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Held-out MAPE [%]")
    fig.suptitle("Held-out MAPE by culture day: mechanistic vs hybrid")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_corrections(train, mech_params, spec, thetas, path):
    """What the network learned: growth multiplier and lactate correction over time."""
    t_grid = np.arange(train.feeds.shape[1]) * DT
    x = np.empty(N_FEAT)
    o = np.empty(N_OUT)
    mu_fac = np.zeros((len(train.ids), len(t_grid)))
    lac_add = np.zeros_like(mu_fac)
    for th in thetas:
        W1, b1, W2, b2 = spec.unpack(th)
        traj, aux = simulate(train, mech_params, spec, th)
        for b in range(len(train.ids)):
            for k in range(0, len(t_grid)):
                y = traj[b, k]
                _features(max(y[0], 0), max(y[3], 0), max(y[4], 0), max(y[5], 0), t_grid[k],
                          aux[b, k, 0], aux[b, k, 1], train.y0[b, 0], x)
                _net(x, W1, b1, W2, b2, o)
                mu_fac[b, k] += np.exp(np.clip(o[0], -3, 3)) / len(thetas)
                lac_add[b, k] += LAC_SCALE * o[2] / len(thetas)
    keep = t_grid <= 13.0
    vcd0 = train.y0[:, 0]
    norm = plt.Normalize(vcd0.min(), vcd0.max())
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.4), constrained_layout=True)
    for b in range(len(train.ids)):
        axes[0].plot(t_grid[keep], mu_fac[b, keep], color=cmap(norm(vcd0[b])), lw=0.8, alpha=0.8)
        axes[1].plot(t_grid[keep], lac_add[b, keep], color=cmap(norm(vcd0[b])), lw=0.8, alpha=0.8)
    axes[0].axhline(1, color="black", lw=0.8)
    axes[0].set_ylabel("growth multiplier exp(NN_mu)")
    axes[0].set_title("Learned growth correction (1 = mechanistic)")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("lactate rate correction [g/(1e9 cells d)]")
    axes[1].set_title("Learned lactate-rate correction (0 = mechanistic)")
    for ax in axes:
        ax.set_xlabel("Time [d]")
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, label="initial VCD (t=0)")
    fig.suptitle("What the network learned (fit on all training batches, ensemble mean)")
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--variants", default="states,states_feeds")
    parser.add_argument("--hidden", type=int, default=4)
    parser.add_argument("--lam", type=float, default=1e-2, help="ridge penalty on network weights")
    parser.add_argument("--seeds", type=int, default=4, help="networks per ensemble")
    parser.add_argument("--mech-starts", type=int, default=6)
    parser.add_argument("--max-nfev", type=int, default=150)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--outputs", default="mu,lac",
                        help="rates the network corrects: any of mu, glc, lac (default mu,lac = option #1)")
    args = parser.parse_args()
    variants = [v.strip() for v in args.variants.split(",")]
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    events, template = sm.load_data()
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    train = sm.build_batches(events, train_ids)
    holdout = sm.build_batches(events, holdout_ids)

    jobs = {"full": train}
    folds = []
    gkf = GroupKFold(n_splits=args.folds)
    idx = np.arange(len(train_ids))
    for f, (tr_idx, te_idx) in enumerate(gkf.split(idx, groups=idx)):
        jobs[f"fold{f}"] = train.subset(tr_idx)
        folds.append(te_idx)

    # 1) Mechanistic core per fold (same objective as sartorius_model default)
    t0 = time.time()
    mech = sm.fit_many(jobs, n_starts=args.mech_starts, workers=args.workers, max_nfev=300,
                       objective="relative", loss="soft_l1")
    mech_params = {name: fit[0] for name, fit in mech.items()}
    print(f"mechanistic cores fitted in {time.time() - t0:.0f}s")

    # Timing of one hybrid simulation
    out_idx = [{"mu": 0, "glc": 1, "lac": 2}[o.strip()] for o in args.outputs.split(",")]
    spec0 = NetSpec(VARIANTS[variants[0]], args.hidden, out_idx)
    simulate(train, mech_params["full"], spec0, spec0.init(0))
    t1 = time.time()
    simulate(train, mech_params["full"], spec0, spec0.init(0))
    print(f"one hybrid simulation of all training batches: {1000 * (time.time() - t1):.1f} ms; "
          f"{spec0.n} network weights")

    # 2) Networks: variants x jobs x seeds, in parallel
    tasks, keys = [], []
    for v in variants:
        for name, bs in jobs.items():
            for s in range(args.seeds):
                tasks.append((bs, mech_params[name], VARIANTS[v], args.hidden, args.lam, 1000 + s, args.max_nfev,
                              out_idx))
                keys.append((v, name))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_fit_net, tasks))
    print(f"trained {len(tasks)} networks in {time.time() - t0:.0f}s")
    nets: dict = {}
    for key, (cost, theta) in zip(keys, results):
        nets.setdefault(key, []).append(theta)

    # 3) Predictions: in-sample (full fit) and held-out (CV)
    frames = []
    oof_curves = {"mechanistic": {}}
    for model in ["mechanistic", *variants]:
        oof_curves.setdefault(model, {})
        if model == "mechanistic":
            frames.append(frame_from_pred(train, sm.observe(sm.simulate(train, mech_params["full"]),
                                                            train.obs.batch_idx, train.obs.step),
                                          model, "in-sample fit"))
        else:
            spec = NetSpec(VARIANTS[model], args.hidden, out_idx)
            frames.append(frame_from_pred(train, predict_obs(train, mech_params["full"], spec, nets[(model, "full")]),
                                          model, "in-sample fit"))
        for f, te_idx in enumerate(folds):
            test = train.subset(te_idx)
            name = f"fold{f}"
            if model == "mechanistic":
                trajs = [sm.simulate(test, mech_params[name])]
            else:
                spec = NetSpec(VARIANTS[model], args.hidden, out_idx)
                trajs = [simulate(test, mech_params[name], spec, th)[0] for th in nets[(model, name)]]
            traj = np.mean(trajs, axis=0)
            pred = sm.observe(traj, test.obs.batch_idx, test.obs.step)
            frames.append(frame_from_pred(test, pred, model, "held-out (CV)"))
            for i, bid in enumerate(test.ids):
                tr = traj[i]
                oof_curves[model][bid] = np.column_stack([tr[:, 0], tr[:, 0] + tr[:, 1], tr[:, 4], tr[:, 5]])
    preds = pd.concat(frames, ignore_index=True)
    preds.to_csv(out_dir / "predictions_vs_measured.csv", index=False)

    scores = leaderboard_table(preds)
    scores.to_csv(out_dir / "leaderboard_style_score.csv")
    pd.set_option("display.width", 200)
    print("\nLeaderboard-style score (MAPE %, lower is better):")
    print(scores.to_string())

    oof = preds[preds.split == "held-out (CV)"]
    by_day = oof.pivot_table(index=["model", "variable"], columns="day", values="abs_pct_error", aggfunc="mean")
    by_day.round(2).to_csv(out_dir / "mape_by_day.csv")

    cv_scores = scores.xs("held-out (CV)", level="split")
    best = cv_scores.loc[variants, "leaderboard_score"].idxmin()
    print(f"\nBest hybrid variant by CV score: {best}")
    print("\nHeld-out MAPE by day (best hybrid):")
    print(by_day.loc[best].loc[TARGETS].round(1).to_string())

    best_spec = NetSpec(VARIANTS[best], args.hidden, out_idx)
    plot_examples(train, oof_curves, oof, best, out_dir / "01_example_batches.png")
    plot_mape_by_day(oof, ["mechanistic", *variants], out_dir / "02_mape_by_day.png")
    sm.plot_error_heatmap(oof[oof.model == best], out_dir / "03_error_heatmap_by_day.png")
    plot_corrections(train, mech_params["full"], best_spec, nets[(best, "full")], out_dir / "04_learned_corrections.png")

    # 4) Holdout predictions with the best variant (trained on all 49 batches)
    tp = holdout.to_predict
    pred = predict_rows(holdout, mech_params["full"], best_spec, nets[(best, "full")], tp.batch_idx, tp.step)
    sub = pd.DataFrame({"batch_id": [holdout.ids[i] for i in tp.batch_idx], "t_key": tp.elapsed_time.round(6)})
    for j, t in enumerate(TARGETS):
        sub[t] = pred[:, j]
    sub = (template[["batch_id", "elapsed_time"]].assign(t_key=template.elapsed_time.round(6))
           .merge(sub, on=["batch_id", "t_key"], how="left").drop(columns="t_key"))
    sub.to_csv(out_dir / "holdout_predictions.csv", index=False)
    print(f"\nWrote holdout predictions ({len(sub)} rows, {sub[TARGETS].isna().any(axis=1).sum()} missing) "
          f"for variant '{best}' and figures to {out_dir}")


if __name__ == "__main__":
    main()
