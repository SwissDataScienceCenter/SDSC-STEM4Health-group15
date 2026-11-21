"""Stage 1 of hybrid v3: the mechanistic ODE with small neural nets correcting two rates inside it.

Ported from branch hybrid_v3_tr, src/hybrid_model.py (variant "states", outputs "mu,lac").

The rates of `mechanistic` are kept, and a tanh network with `hidden` units, fed the current
state, multiplies the growth rate (exp of output 0) and adds to the lactate rate (output 2).
The mass balances still hold, so the corrected model stays a simulator: it forecasts a batch
open-loop from x(0) and the feed/volume log. Fitting is two steps:
  1. mechanistic parameters: multi-start least squares on relative error;
  2. networks on top of the frozen parameters, one per seed (ridge on the weights, output
     layer starting at zero = the mechanistic model); predictions average the seeds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time

from loguru import logger
from numba import njit
import numpy as np
from scipy.optimize import least_squares

from bioreactor_decoding.dataset import TARGETS
from bioreactor_decoding.modeling import mechanistic as mech
from bioreactor_decoding.modeling.mechanistic import DT, BatchSet, observe, softplus_nb

N_FEAT = 8
N_OUT = 3  # [growth, glucose uptake, lactate]
LAC_SCALE = 0.2  # [g/(1e9 cells d)] size of one unit of lactate correction
FEATURE_NAMES = ["S", "L", "Xv", "gamma", "time", "cumB/V", "cumC/V", "VCD0"]
VARIANTS = {"states": [0, 1, 2, 3, 4], "states_feeds": [0, 1, 2, 3, 4, 5, 6, 7]}
OUTPUT_INDEX = {"mu": 0, "glc": 1, "lac": 2}


# --------------------------------------------------------------------------
# Hybrid simulator (numba)
# --------------------------------------------------------------------------


@njit(cache=True)
def _features(Xv, gamma, S, L, t, fb_frac, fc_frac, vcd0, x):
    x[0] = (S - 2.5) / 1.5
    x[1] = (L - 1.5) / 1.0
    x[2] = (Xv - 20.0) / 12.0
    x[3] = (gamma - 100.0) / 80.0
    x[4] = (t - 6.5) / 4.0
    x[5] = (fb_frac - 0.15) / 0.12
    x[6] = (fc_frac - 0.0008) / 0.0006
    x[7] = (vcd0 - 1.09) / 0.12


@njit(cache=True)
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


@njit(cache=True)
def _hybrid_rates(y, t, fb_frac, fc_frac, vcd0, p, W1, b1, W2, b2, x, o):
    """Mechanistic rates with NN corrections."""
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
    lac_prod = softplus_nb(v_S - p[11])
    lac_cons = softplus_nb(p[11] - v_S) * np.tanh(L / p[12]) ** 2
    v_L = p[13] * lac_prod - p[14] * lac_cons + LAC_SCALE * o[2]
    if v_L < 0.0:  # lactate consumption must fade out as lactate runs out
        v_L *= np.tanh(L / 0.1) ** 2
    return Xv, mu_eff, mu_d, mu_l, v_S, v_L


@njit(cache=True)
def _rhs(y, t, fb_frac, fc_frac, vcd0, p, W1, b1, W2, b2, x, o, dy):
    Xv, mu_eff, mu_d, mu_l, v_S, v_L = _hybrid_rates(
        y, t, fb_frac, fc_frac, vcd0, p, W1, b1, W2, b2, x, o
    )
    dy[0] = (mu_eff - mu_d - mu_l) * Xv
    dy[1] = mu_d * Xv
    dy[2] = mu_l * Xv
    dy[3] = Xv
    dy[4] = -v_S * Xv
    dy[5] = v_L * Xv


@njit(cache=True)
def _simulate(y0, v0, feeds, vsnap, p, W1, b1, W2, b2):
    """Same event handling as mechanistic._simulate, plus the network. Returns (nb, n_steps, 6)."""
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
                y[j] = max(y[j], 0.0)
    return out


# --------------------------------------------------------------------------
# Network parameters
# --------------------------------------------------------------------------


class NetSpec:
    """Maps the trainable vector theta to full weight arrays (masked inputs/outputs = 0)."""

    def __init__(self, feat_idx, hidden, out_idx):
        self.feat_idx = list(feat_idx)
        self.out_idx = list(out_idx)
        self.hidden = hidden
        self.sizes = [hidden * len(self.feat_idx), hidden, len(self.out_idx) * hidden,
                      len(self.out_idx)]  # fmt: skip
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


def simulate(bs: BatchSet, mech_params, spec: NetSpec, theta) -> np.ndarray:
    W1, b1, W2, b2 = spec.unpack(theta)
    p = np.asarray(mech_params, dtype=np.float64)
    return _simulate(bs.y0, bs.v0, bs.feeds, bs.vsnap, p, W1, b1, W2, b2)


def _fit_net(args):
    """Train one network (one seed) on top of frozen mechanistic parameters."""
    bs, mech_params, spec, lam, seed, max_nfev = args
    y_obs = bs.obs[TARGETS].to_numpy(float)
    b_idx = bs.obs["batch_idx"].to_numpy(int)
    steps = bs.obs["step"].to_numpy(int)
    ridge = np.sqrt(lam)

    def resid(theta):
        r = (
            (observe(simulate(bs, mech_params, spec, theta), b_idx, steps) - y_obs) / y_obs
        ).ravel()
        if not np.all(np.isfinite(r)):
            r = np.full(r.shape, 1e3)
        return np.concatenate([r, ridge * theta])

    res = least_squares(resid, spec.init(seed), method="trf", loss="soft_l1", f_scale=0.05,
                        x_scale=1.0, max_nfev=max_nfev)  # fmt: skip
    return res.x


# --------------------------------------------------------------------------
# Stage-1 model
# --------------------------------------------------------------------------


@dataclass
class HybridODEConfig:
    """Hyperparameters of stage 1 (defaults = the settings behind the team's 7.67)."""

    variant: str = "states"
    outputs: tuple = ("mu", "lac")
    hidden: int = 4
    lam: float = 1e-2
    seeds: int = 4
    mech_starts: int = 6
    mech_max_nfev: int = 300
    net_max_nfev: int = 150

    def spec(self) -> NetSpec:
        return NetSpec(
            VARIANTS[self.variant], self.hidden, [OUTPUT_INDEX[o] for o in self.outputs]
        )


@dataclass
class HybridODE:
    """A fitted stage-1 model: mechanistic parameters plus one network per seed."""

    config: HybridODEConfig
    mech_params: np.ndarray
    thetas: list = field(default_factory=list)

    def predict(self, bs: BatchSet, batch_idx, steps) -> np.ndarray:
        """Seed-averaged prediction of the 4 targets at (batch_idx, step), shape (n, 4)."""
        spec = self.config.spec()
        preds = [observe(simulate(bs, self.mech_params, spec, th), batch_idx, steps)
                 for th in self.thetas]  # fmt: skip
        return np.mean(preds, axis=0)


def fit_hybrid_odes(jobs: dict, config: HybridODEConfig, n_jobs: int = 1) -> dict:
    """Fit one HybridODE per BatchSet in `jobs` ({name: BatchSet}); all fits run in parallel."""
    t0 = time.time()
    mech_params = mech.fit_many(jobs, config.mech_starts, config.mech_max_nfev, n_jobs)
    logger.info(f"stage 1: {len(jobs)} mechanistic fits in {time.time() - t0:.0f}s")
    t0 = time.time()
    spec = config.spec()
    tasks, keys = [], []
    for name, bs in jobs.items():
        for s in range(config.seeds):
            tasks.append((bs, mech_params[name], spec, config.lam, 1000 + s, config.net_max_nfev))
            keys.append(name)
    fitted = {name: HybridODE(config, mech_params[name]) for name in jobs}
    for name, theta in zip(keys, mech.run_parallel(_fit_net, tasks, n_jobs)):
        fitted[name].thetas.append(theta)
    logger.info(f"stage 1: {len(tasks)} networks in {time.time() - t0:.0f}s")
    return fitted
