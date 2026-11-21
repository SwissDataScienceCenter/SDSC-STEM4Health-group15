"""Mechanistic fed-batch model, simulated for all batches at once (vectorised Euler).

States (per batch): Xv viable cells, Xd dead cells (TCD = Xv + Xd), S substrate,
P byproduct, N latent limiting nutrient (unmeasured, normalised to N(0)=1), V volume.

  mu   = mu_max * S/(Ks+S) * N/(KN+N) * KI/(KI+P)          growth
  kd   = kd0 + kd1 * KdN/(KdN+N)                            death rises when N runs out
  dXv  = (mu - kd) Xv
  dXd  = kd Xv - klys Xd
  dS   = -(mu/Yxs + ms) Xv
  dP   = (Ypx mu - qPc * P/(KPc+P)) Xv                      produced with growth, consumed later
  dN   = -(mu/Yxn) Xv
Feeds are bolus additions: everything is diluted by V/(V+F), then
  S += (Sb*F_b + Ss*F_substrate)/V_new,  N += Nb*F_b/V_new.
Feeds C and D are only treated as volume here (small, candidates for further study).
"""
from __future__ import annotations

import numpy as np

PARAMS = {  # name: (initial guess, lower, upper)
    "mu_max": (1.0, 0.1, 3.0),
    "Ks":     (0.2, 1e-3, 5.0),
    "KI":     (3.0, 0.1, 50.0),
    "KN":     (0.1, 1e-3, 2.0),
    "Yxn":    (20.0, 0.5, 500.0),
    "Nb":     (5.0, 0.0, 100.0),
    "kd0":    (0.01, 1e-4, 0.5),
    "kd1":    (0.1, 1e-4, 2.0),
    "KdN":    (0.1, 1e-3, 5.0),
    "klys":   (0.02, 1e-4, 1.0),
    "Yxs":    (10.0, 0.5, 200.0),
    "ms":     (0.01, 1e-5, 0.5),
    "Sb":     (10.0, 0.0, 100.0),
    "Ss":     (100.0, 0.0, 1000.0),
    "Ypx":    (0.3, 1e-3, 5.0),
    "qPc":    (0.02, 1e-5, 1.0),
    "KPc":    (0.5, 1e-3, 10.0),
}
NAMES = list(PARAMS)
P0 = np.array([PARAMS[n][0] for n in NAMES])
LB = np.array([PARAMS[n][1] for n in NAMES])
UB = np.array([PARAMS[n][2] for n in NAMES])


def simulate(p, arr, record_all=False):
    """Return trajectory (K, B, 4) of [vcd, tcd, S, P] (or only what's needed)."""
    q = dict(zip(NAMES, p))
    t, F = arr.t, arr.feeds
    dt = t[1] - t[0]
    Xv = arr.x0[:, 0].copy()
    Xd = np.maximum(arr.x0[:, 1] - arr.x0[:, 0], 0.0)
    S = arr.x0[:, 2].copy(); P = arr.x0[:, 3].copy()
    N = np.ones_like(Xv); V = arr.v0.copy()
    out = np.empty((len(t), len(Xv), 4))
    Fb_any = F.sum(axis=2) > 0
    for k in range(len(t)):
        out[k, :, 0] = Xv; out[k, :, 1] = Xv + Xd
        out[k, :, 2] = S;  out[k, :, 3] = P
        if Fb_any[k].any():
            f = F[k]; Vn = V + f.sum(axis=1); r = V / Vn
            Xv *= r; Xd *= r; P *= r; N *= r
            S = S * r + (q["Sb"] * f[:, 1] + q["Ss"] * f[:, 0]) / Vn
            N = N + q["Nb"] * f[:, 1] / Vn
            V = Vn
        Sp = np.maximum(S, 0.0); Np = np.maximum(N, 0.0); Pp = np.maximum(P, 0.0)
        mu = (q["mu_max"] * Sp / (q["Ks"] + Sp) * Np / (q["KN"] + Np)
              * q["KI"] / (q["KI"] + Pp))
        kd = q["kd0"] + q["kd1"] * q["KdN"] / (q["KdN"] + Np)
        dXv = (mu - kd) * Xv
        dXd = kd * Xv - q["klys"] * Xd
        dS = -(mu / q["Yxs"] + q["ms"]) * Xv
        dP = (q["Ypx"] * mu - q["qPc"] * Pp / (q["KPc"] + Pp)) * Xv
        dN = -(mu / q["Yxn"]) * Xv
        Xv = Xv + dt * dXv; Xd = Xd + dt * dXd
        S = np.maximum(S + dt * dS, 0.0); P = np.maximum(P + dt * dP, 0.0)
        N = np.maximum(N + dt * dN, 0.0)
    return out


def predict_at_obs(p, arr):
    traj = simulate(p, arr)
    return [traj[idx, j, :] for j, idx in enumerate(arr.obs_idx)]
