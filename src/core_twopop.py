"""Two-population mechanistic core (experiments_v6, idea 4).

Growing cells Xg switch to a non-growing, lactate-consuming state Xn. The
switch is driven by nutrient depletion and culture age; the lactate shift is
a population shift, not a kinetic switch.

    mu   = mu_max * S/(K_S+S) * N/(K_N+N) * K_IL/(K_IL+L)
    kt   = kt0 + kt_N * K_Nt/(K_Nt+N)                       (growing -> non-growing)
    kd_g = kd0,  kd_n = kd0 + kd_n0 + kd_L*L + kd_N * K_Nt/(K_Nt+N)
    dXg = (mu - kt - kd_g) Xg,  dXn = kt Xg - kd_n Xn,  dD = kd_g Xg + kd_n Xn - k_lys D
    dS  = -(mu/Y_XS + m_S) Xg - m_Sn Xn
    dL  = Y_LS (mu/Y_XS + m_S) S/(K_SL+S) Xg - qLc L/(K_Lc+L) Xn
    dN  = -mu Xg / Y_XN
    VCD = Xg + Xn, TCD = VCD + D
"""

from __future__ import annotations

import numpy as np
from numba import njit

import sartorius_model as sm

DT = sm.DT
PARAMS = [
    ("mu_max", 0.9, 0.2, 3.0), ("K_S", 0.2, 1e-3, 5.0), ("K_N", 0.3, 1e-3, 10.0), ("K_IL", 5.0, 0.1, 100.0),
    ("Y_XN", 20.0, 0.5, 500.0), ("N0", 3.0, 0.05, 100.0), ("c_NB", 20.0, 0.1, 500.0),
    ("kt0", 0.05, 1e-4, 2.0), ("kt_N", 0.3, 1e-4, 5.0), ("K_Nt", 0.2, 1e-3, 10.0),
    ("kd0", 0.01, 1e-4, 0.5), ("kd_n0", 0.02, 1e-5, 0.5), ("kd_L", 0.01, 1e-5, 0.5), ("kd_N", 0.05, 1e-5, 2.0),
    ("k_lys", 0.05, 1e-4, 1.0), ("Y_XS", 15.0, 1.0, 200.0), ("m_S", 0.01, 1e-5, 0.5), ("m_Sn", 0.01, 1e-5, 0.5),
    ("Y_LS", 1.0, 0.01, 5.0), ("K_SL", 0.5, 1e-3, 10.0), ("qLc", 0.05, 1e-4, 1.0), ("K_Lc", 0.5, 1e-3, 10.0),
    ("c_S", 300.0, 10.0, 1500.0), ("c_B", 40.0, 0.1, 300.0), ("c_D", 5.0, 1e-3, 1000.0),
]
P0 = np.array([p[1] for p in PARAMS])
LB = np.log([p[2] for p in PARAMS])
UB = np.log([p[3] for p in PARAMS])
NS = 6  # Xg, Xn, D, S, L, N


@njit(cache=True)
def _rhs(y, p, dy):
    Xg, Xn, D = max(y[0], 0.0), max(y[1], 0.0), max(y[2], 0.0)
    S, L, N = max(y[3], 0.0), max(y[4], 0.0), max(y[5], 0.0)
    mu = p[0] * S / (p[1] + S) * N / (p[2] + N) * p[3] / (p[3] + L)
    lim = p[9] / (p[9] + N)
    kt = p[7] + p[8] * lim
    kd_g = p[10]
    kd_n = p[10] + p[11] + p[12] * L + p[13] * lim
    qS = mu / p[15] + p[16]
    dy[0] = (mu - kt - kd_g) * Xg
    dy[1] = kt * Xg - kd_n * Xn
    dy[2] = kd_g * Xg + kd_n * Xn - p[14] * D
    dy[3] = -qS * Xg - p[17] * Xn
    dy[4] = p[18] * qS * S / (p[19] + S) * Xg - p[20] * L / (p[21] + L) * Xn
    dy[5] = -mu * Xg / p[4]


@njit(cache=True)
def _simulate(y0, v0, feeds, vsnap, p):
    nb, ns, _ = feeds.shape
    out = np.empty((nb, ns, NS))
    c_S, c_B, c_D, c_NB, N0 = p[22], p[23], p[24], p[6], p[5]
    h = DT
    y = np.empty(NS)
    tmp = np.empty(NS)
    k1 = np.empty(NS)
    k2 = np.empty(NS)
    k3 = np.empty(NS)
    k4 = np.empty(NS)
    for b in range(nb):
        y[0], y[1], y[2], y[3], y[4], y[5] = y0[b, 0], 0.0, y0[b, 1], y0[b, 4], y0[b, 5], N0
        V = v0[b]
        for k in range(ns):
            for j in range(NS):
                out[b, k, j] = y[j]
            if not np.isnan(vsnap[b, k]):
                if vsnap[b, k] > V:
                    r = V / vsnap[b, k]
                    for j in range(NS):
                        y[j] *= r
                V = vsnap[b, k]
            f_s, f_b, f_c, f_d = feeds[b, k, 0], feeds[b, k, 1], feeds[b, k, 2], feeds[b, k, 3]
            total = f_s + f_b + f_c + f_d
            if total > 0.0:
                v_new = V + total
                r = V / v_new
                for j in range(NS):
                    y[j] *= r
                y[3] += (c_S * f_s + c_B * f_b + c_D * f_d) / v_new
                y[5] += c_NB * f_b / v_new
                V = v_new
            if k == ns - 1:
                break
            _rhs(y, p, k1)
            for j in range(NS):
                tmp[j] = y[j] + 0.5 * h * k1[j]
            _rhs(tmp, p, k2)
            for j in range(NS):
                tmp[j] = y[j] + 0.5 * h * k2[j]
            _rhs(tmp, p, k3)
            for j in range(NS):
                tmp[j] = y[j] + h * k3[j]
            _rhs(tmp, p, k4)
            for j in range(NS):
                y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
                if y[j] < 0.0:
                    y[j] = 0.0
    return out


def observe(bs, p, rows):
    sim = _simulate(bs.y0, bs.v0, bs.feeds, bs.vsnap, p)
    st = sim[rows.batch_idx.to_numpy(), rows.step.to_numpy()]
    vcd = st[:, 0] + st[:, 1]
    return np.column_stack([vcd, vcd + st[:, 2], st[:, 3], st[:, 4]])


# ---- per-batch multipliers (for sensor assimilation, core_assim2 --core twopop) ----
# m: 0 mu_max, 1 Y_LS, 2 Y_XS, 3 kd0, 4 qLc, 5 Y_XN ; icm scales Xg0 and D0


@njit(cache=True)
def _rhs_m(y, p, m, dy):
    Xg, Xn, D = max(y[0], 0.0), max(y[1], 0.0), max(y[2], 0.0)
    S, L, N = max(y[3], 0.0), max(y[4], 0.0), max(y[5], 0.0)
    mu = p[0] * m[0] * S / (p[1] + S) * N / (p[2] + N) * p[3] / (p[3] + L)
    lim = p[9] / (p[9] + N)
    kt = p[7] + p[8] * lim
    kd_g = p[10] * m[3]
    kd_n = kd_g + p[11] + p[12] * L + p[13] * lim
    qS = mu / (p[15] * m[2]) + p[16]
    dy[0] = (mu - kt - kd_g) * Xg
    dy[1] = kt * Xg - kd_n * Xn
    dy[2] = kd_g * Xg + kd_n * Xn - p[14] * D
    dy[3] = -qS * Xg - p[17] * Xn
    dy[4] = p[18] * m[1] * qS * S / (p[19] + S) * Xg - p[20] * m[4] * L / (p[21] + L) * Xn
    dy[5] = -mu * Xg / (p[4] * m[5])


@njit(cache=True)
def _simulate_m(y0, v0, feeds, vsnap, p, mult, icm):
    nb, ns, _ = feeds.shape
    out = np.empty((nb, ns, NS))
    c_S, c_B, c_D, c_NB, N0 = p[22], p[23], p[24], p[6], p[5]
    h = DT
    y = np.empty(NS)
    tmp = np.empty(NS)
    k1 = np.empty(NS)
    k2 = np.empty(NS)
    k3 = np.empty(NS)
    k4 = np.empty(NS)
    for b in range(nb):
        m = mult[b]
        y[0], y[1], y[2], y[3], y[4], y[5] = y0[b, 0] * icm[b], 0.0, y0[b, 1] * icm[b], y0[b, 4], y0[b, 5], N0
        V = v0[b]
        for k in range(ns):
            for j in range(NS):
                out[b, k, j] = y[j]
            if not np.isnan(vsnap[b, k]):
                if vsnap[b, k] > V:
                    r = V / vsnap[b, k]
                    for j in range(NS):
                        y[j] *= r
                V = vsnap[b, k]
            f_s, f_b, f_c, f_d = feeds[b, k, 0], feeds[b, k, 1], feeds[b, k, 2], feeds[b, k, 3]
            total = f_s + f_b + f_c + f_d
            if total > 0.0:
                v_new = V + total
                r = V / v_new
                for j in range(NS):
                    y[j] *= r
                y[3] += (c_S * f_s + c_B * f_b + c_D * f_d) / v_new
                y[5] += c_NB * f_b / v_new
                V = v_new
            if k == ns - 1:
                break
            _rhs_m(y, p, m, k1)
            for j in range(NS):
                tmp[j] = y[j] + 0.5 * h * k1[j]
            _rhs_m(tmp, p, m, k2)
            for j in range(NS):
                tmp[j] = y[j] + 0.5 * h * k2[j]
            _rhs_m(tmp, p, m, k3)
            for j in range(NS):
                tmp[j] = y[j] + h * k3[j]
            _rhs_m(tmp, p, m, k4)
            for j in range(NS):
                y[j] += h / 6.0 * (k1[j] + 2.0 * k2[j] + 2.0 * k3[j] + k4[j])
                if y[j] < 0.0:
                    y[j] = 0.0
    return out
