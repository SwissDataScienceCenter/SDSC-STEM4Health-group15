"""Mechanistic fed-batch model.

States (per batch): Xv viable cells, Xd dead cells (TCD = Xv + Xd), S substrate,
P byproduct, N latent limiting nutrient (unmeasured, normalised to N(0)=1).

  mu   = mu_max * S/(Ks+S) * N/(KN+N) * KI/(KI+P) * LAG(t)  growth
  kd   = kd0 + kd1 * KdN/(KdN+N)                            death rises when N runs out
  dXv  = (mu - kd) Xv
  dXd  = kd Xv - klys Xd
  dS   = -(mu/Yxs + ms) Xv
  dP   = (Ypx mu * INH(P) - qPc * P/(KPc+P) * SW(mu)) Xv    byproduct production / consumption
         + Ybase * b(t) / V                                 measured acid production (option base_flux)
  dN   = -(mu/Yxn) Xv
Feeds are bolus additions: everything is diluted by V/(V+F), then
  S += (Sb*F_b + Ss*F_substrate)/V_new,  N += Nb*F_b/V_new.
Feeds C and D are only treated as volume here.

Optional terms (set_options; all OFF by default = the original "v1" model):
  lag            LAG(t) = 1 - exp(-t / t_lag)       cells adapt after inoculation (lag phase)
  byp_inhibition INH(P) = KPP / (KPP + P)           production slows as byproduct accumulates
                                                    (overflow metabolism -> plateau)
  byp_switch     SW(mu) = K_mu / (K_mu + mu)        consumption switches on when growth slows
                                                    ("lactate shift" after the growth peak)
  base_flux      b(t) = base added by the pH controller per day (volume not explained by the
                 feed log, days 2-6; see sensors.base_rate). Base neutralises the acid the cells
                 produce, so b(t) is a MEASURED byproduct-production flux: P += Ybase * b / V.
                 (Tobias' idea 3: put the soft sensor inside the ODE.) Needs arrays built with
                 to_arrays(..., base=sensors.base_rate(df, sample)).
When a term is off it is 1 and its parameter is removed from NAMES (so it is not fitted).

Solver: fixed-step forward Euler on the grid arr.t (dt = 0.005 d by default), all batches
vectorised. Feeds are applied at the grid step of their time, after the sample at that step.
"""
from __future__ import annotations

import numpy as np

PARAMS = {  # name: (initial guess, lower, upper)
    "mu_max": (1.0, 0.1, 3.0),
    "Ks":     (0.2, 1e-3, 5.0),
    "KI":     (3.0, 0.1, 50.0),
    "KN":     (0.1, 1e-3, 2.0),
    "Yxn":    (20.0, 0.5, 500.0),
    "Nb":     (5.0, 1e-3, 100.0),
    "kd0":    (0.01, 1e-4, 0.5),
    "kd1":    (0.1, 1e-4, 2.0),
    "KdN":    (0.1, 1e-3, 5.0),
    "klys":   (0.02, 1e-4, 1.0),
    "Yxs":    (10.0, 0.05, 200.0),
    "ms":     (0.01, 1e-5, 0.5),
    "Sb":     (10.0, 1e-3, 200.0),
    "Ss":     (100.0, 1e-3, 2000.0),
    "Ypx":    (0.3, 1e-3, 5.0),
    "qPc":    (0.02, 1e-5, 1.0),
    "KPc":    (0.5, 1e-3, 10.0),
    # optional terms
    "t_lag":  (0.5, 0.01, 3.0),      # lag-phase time constant [d]
    "KPP":    (2.0, 0.05, 50.0),     # byproduct level that halves its own production [g/L]
    "K_mu":   (0.2, 0.005, 5.0),     # growth rate below which consumption switches on [1/d]
    "Ybase":  (50.0, 1e-2, 2000.0),  # byproduct [g/L] per (base volume / culture volume)
}
OPTIONAL = {"lag": "t_lag", "byp_inhibition": "KPP", "byp_switch": "K_mu", "base_flux": "Ybase"}
OPTIONS = {"lag": False, "byp_inhibition": False, "byp_switch": False, "base_flux": False}

NAMES: list = []          # active parameter names (filled by set_options, mutated in place)
P0 = LB = UB = None       # active initial guess / bounds (arrays, rebuilt by set_options)


def set_options(**kw):
    """Switch optional model terms on/off, e.g. set_options(lag=False).
    Rebuilds NAMES / P0 / LB / UB. Parameter vectors fitted under other options do not fit.
    Returns the active options."""
    global P0, LB, UB
    for k, v in kw.items():
        if k not in OPTIONS:
            raise KeyError(f"unknown option {k}; choose from {list(OPTIONS)}")
        OPTIONS[k] = bool(v)
    off = {OPTIONAL[k] for k, v in OPTIONS.items() if not v}
    NAMES[:] = [n for n in PARAMS if n not in off]
    P0 = np.array([PARAMS[n][0] for n in NAMES])
    LB = np.array([PARAMS[n][1] for n in NAMES])
    UB = np.array([PARAMS[n][2] for n in NAMES])
    return dict(OPTIONS)


def model_tag():
    """Short name of the active model variant, e.g. for cache files: 'v1' (no optional terms)
    or e.g. 'v2-lag-base_flux' (the terms that are on)."""
    on = [k for k, v in OPTIONS.items() if v]
    if not on:
        return "v1"
    return "v2-" + "-".join(on)


set_options()


# ---------------------------------------------------------------- Euler (vectorised)
def simulate_euler(p, arr):
    """All batches at once on arr.t; returns (K, B, 4)."""
    if len(p) != len(NAMES):
        raise ValueError(f"parameter vector has {len(p)} entries, active model needs {len(NAMES)} "
                         f"({model_tag()}: {NAMES}) - fitted under different options?")
    q = dict(zip(NAMES, p))
    lag, inh, sw, bf = OPTIONS["lag"], OPTIONS["byp_inhibition"], OPTIONS["byp_switch"], OPTIONS["base_flux"]
    if bf and getattr(arr, "base", None) is None:
        raise ValueError("option base_flux needs arrays built with to_arrays(..., base=...)")
    t, F = arr.t, arr.feeds
    dt = t[1] - t[0]
    Xv = arr.x0[:, 0].copy(); Xd = np.maximum(arr.x0[:, 1] - arr.x0[:, 0], 0.0)
    S = arr.x0[:, 2].copy(); P = arr.x0[:, 3].copy()
    N = np.ones_like(Xv); V = arr.v0.copy()
    out = np.empty((len(t), len(Xv), 4))
    Fany = F.sum(axis=2) > 0
    lag_fac = 1.0 - np.exp(-t / q["t_lag"]) if lag else np.ones_like(t)
    for k in range(len(t)):
        out[k, :, 0] = Xv; out[k, :, 1] = Xv + Xd; out[k, :, 2] = S; out[k, :, 3] = P
        if Fany[k].any():
            f = F[k]; Vn = V + f.sum(axis=1); r = V / Vn
            Xv *= r; Xd *= r; P *= r
            S = S * r + (q["Ss"] * f[:, 0] + q["Sb"] * f[:, 1]) / Vn
            N = N * r + q["Nb"] * f[:, 1] / Vn
            V = Vn
        Sp = np.maximum(S, 0); Pp = np.maximum(P, 0); Np = np.maximum(N, 0)
        mu = (q["mu_max"] * Sp / (q["Ks"] + Sp) * Np / (q["KN"] + Np)
              * q["KI"] / (q["KI"] + Pp) * lag_fac[k])
        kd = q["kd0"] + q["kd1"] * q["KdN"] / (q["KdN"] + Np)
        prod = q["Ypx"] * mu * (q["KPP"] / (q["KPP"] + Pp) if inh else 1.0)
        cons = q["qPc"] * Pp / (q["KPc"] + Pp) * (q["K_mu"] / (q["K_mu"] + mu) if sw else 1.0)
        dXv = (mu - kd) * Xv; dXd = kd * Xv - q["klys"] * Xd
        dS = -(mu / q["Yxs"] + q["ms"]) * Xv
        dP = (prod - cons) * Xv
        if bf:
            dP = dP + q["Ybase"] * arr.base[k] / (V * dt)        # base volume added in this step
        dN = -(mu / q["Yxn"]) * Xv
        Xv = Xv + dt * dXv; Xd = Xd + dt * dXd
        S = np.maximum(S + dt * dS, 0); P = np.maximum(P + dt * dP, 0); N = np.maximum(N + dt * dN, 0)
    return out


# ---------------------------------------------------------------- public API
def simulate(p, arr):
    """Trajectories on the grid arr.t for all batches: (K, B, 4) = vcd, tcd, S, P."""
    return simulate_euler(p, arr)


def predict_at_obs(p, arr):
    """List (per batch) of (n_obs, 4) predictions at the sample times."""
    traj = simulate_euler(p, arr)
    return [traj[idx, j, :] for j, idx in enumerate(arr.obs_idx)]
