"""Stage 2 of Tobias' hybrid v3 (src/hybrid_v3.py, branch hybrid_v3_tr), ported to this package,
plus the next ideas from his improving_strategies.md.

    stage 1  any base prediction s1 (here: our mechanistic model, optionally with the base flux
             inside the ODE, or Tobias' neural-net hybrid read from his CSV output)
    stage 2  hybrid branch:   s1 x exp(correction), corrections fitted on log(obs / s1):
                 cells      gradient boosting on day-0 state, feed-B/C/volume deviations, A_early, log s1
                 substrate  bolus soft sensor (rows followed by a bolus) + per-day ridge on
                            base_tot, A11, A12, A_early, log s1 (days 3-13)
                 byproduct  ridge with free per-day intercepts and sensor effects quadratic in day,
                            on base_cum, base_to3, base_to4, base_tot, log s1
                 then MAPE-optimal scaling per state and day (weighted median)
             data-driven branch: gradient boosting on log(target), bagged over batches
             final: 50/50 geometric blend, TCD floored at VCD
    idea 1   similarity member for cells: each batch's VCD/TCD correction curve = weighted mean of
             the curves of the k most similar training batches at day 0 (optional third member)

All corrections are fitted on training rows only. `rows` = one row per (batch, sample) with
columns batch_id, t, day, obs_<state> (NaN for test), s1_<state>, and the sensor features.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .data import STATES
from .sensors import BASE_COLS, BOLUS_COLS, T0_COLS

V2_CELL_COLS = T0_COLS + ["day", "fb_next_dev", "fb_prev_dev", "fc_next_dev", "fc_prev_dev", "V_dev"]
V2_DD_COLS = ["t", *T0_COLS, "cumB", "cumC", "cumS", "fb_next_dev", "fb_prev_dev", "fc_next_dev",
              "fc_prev_dev", "V_dev", "A"]
CELL_COLS = V2_CELL_COLS + ["A_early"]
S_COLS = ["base_tot", "A11", "A12", "A_early", "ls1_substrate"]
L_COLS = ["base_cum", "base_to3", "base_to4", "base_tot", "ls1_byproduct"]
DD_COLS = {"vcd": V2_DD_COLS + ["A_early"], "tcd": V2_DD_COLS + ["A_early"],
           "substrate": V2_DD_COLS + BASE_COLS + BOLUS_COLS, "byproduct": V2_DD_COLS + BASE_COLS}
SIM_COLS = ["vcd0", "s0", "l0", "via0", "fb_tot_dev", "A_early"]   # batch descriptors for idea 1


# ---------------------------------------------------------------- rows
def build_rows(F, sens_batch, sens_day, s1, obs):
    """Join feed features F (batch_id, t, ...), per-batch / per-day sensors, stage-1 predictions
    s1 (batch_id, t, s1_<state>) and observations obs (batch_id, t, obs_<state>) on (batch_id, t)."""
    key = lambda d: d.assign(t_key=d.t.round(6)).drop(columns="t")
    rows = key(s1).merge(key(F), on=["batch_id", "t_key"], how="left", validate="one_to_one")
    rows = rows.merge(key(obs), on=["batch_id", "t_key"], how="left", validate="one_to_one")
    rows = rows.merge(sens_batch.reset_index(names="batch_id"), on="batch_id", how="left")
    rows = rows.merge(sens_day, on=["batch_id", "day"], how="left").reset_index(drop=True)
    rows = rows.rename(columns={"t_key": "t"})
    for v in STATES:
        rows["ls1_" + v] = np.log(np.maximum(rows["s1_" + v], 1e-6))
    # feed-B total deviation per batch (for the similarity member)
    rows["fb_tot_dev"] = rows.groupby("batch_id").fb_prev_dev.transform("sum")
    return rows


def check_rows(rows):
    needed = sorted(set(CELL_COLS + S_COLS + L_COLS + sum(DD_COLS.values(), []) + ["A", "fs_group"]))
    missing = rows[needed].isna().sum()
    assert not missing.any(), f"feature merge failed: {missing[missing > 0].to_dict()}"


# ---------------------------------------------------------------- building blocks (Tobias)
def gbm(depth=2, lr=0.03, iters=200, leaf=30, seed=0):
    return HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=iters,
                                         min_samples_leaf=leaf, l2_regularization=1.0, random_state=seed)


class _Standardised:
    def _z(self, rows):
        return ((rows[self.cols] - self.mu) / self.sd).to_numpy()

    def _set_scale(self, rows):
        self.mu = rows[self.cols].mean()
        self.sd = rows[self.cols].std().replace(0, 1.0)


class DayRidge(_Standardised):
    """Independent ridge regression per culture day."""

    def __init__(self, cols, days=None, alpha=5.0):
        self.cols, self.days, self.alpha = cols, days, alpha

    def fit(self, rows, y):
        self._set_scale(rows)
        Z, self.coef = self._z(rows), {}
        for day in np.unique(rows.day):
            if self.days is not None and day not in self.days:
                continue
            m = (rows.day == day).to_numpy()
            Z1 = np.column_stack([np.ones(m.sum()), Z[m]])
            P = self.alpha * np.eye(Z1.shape[1]); P[0, 0] = 0.0
            self.coef[day] = np.linalg.solve(Z1.T @ Z1 + P, Z1.T @ y[m])
        return self

    def predict(self, rows):
        Z, out = self._z(rows), np.zeros(len(rows))
        for i, day in enumerate(rows.day.to_numpy()):
            c = self.coef.get(day)
            if c is not None:
                out[i] = c[0] + Z[i] @ c[1:]
        return out


class DaySmooth(_Standardised):
    """Free per-day intercepts; each feature's effect is quadratic in culture day."""

    def __init__(self, cols, alpha=5.0, degree=2):
        self.cols, self.alpha, self.degree = cols, alpha, degree

    def _design(self, rows):
        Z = self._z(rows)
        d = (rows.day.to_numpy() - 7.0) / 6.0
        Phi = np.column_stack([d ** k for k in range(self.degree + 1)])
        return np.column_stack([np.eye(15)[rows.day.to_numpy()]] + [Z[:, [j]] * Phi for j in range(Z.shape[1])])

    def fit(self, rows, y):
        self._set_scale(rows)
        D = self._design(rows)
        P = self.alpha * np.eye(D.shape[1]); P[:15, :15] = 1e-9 * np.eye(15)
        self.coef = np.linalg.solve(D.T @ D + P, D.T @ y)
        return self

    def predict(self, rows):
        return self._design(rows) @ self.coef


class SoftSensor:
    """Substrate bolus soft sensor: log(obs/s1) = a + b*A + c*log(s1), per bolus group."""

    def fit(self, rows, y):
        self.coef = {}
        for g in ["d11", "d12", "early"]:
            m = (rows.fs_group == g).to_numpy()
            if m.sum() >= 5:
                X = np.column_stack([np.ones(m.sum()), rows.A.to_numpy()[m], rows.ls1_substrate.to_numpy()[m]])
                self.coef[g] = np.linalg.lstsq(X, y[m], rcond=None)[0]
        return self

    def predict(self, rows):
        out = np.zeros(len(rows))
        for g, c in self.coef.items():
            m = (rows.fs_group == g).to_numpy()
            out[m] = c[0] + c[1] * rows.A.to_numpy()[m] + c[2] * rows.ls1_substrate.to_numpy()[m]
        return out


def mape_optimal_scale(pred, obs):
    """argmin_c sum |c*pred/obs - 1| = weighted median of obs/pred with weights pred/obs."""
    z, w = obs / pred, pred / obs
    order = np.argsort(z)
    cw = np.cumsum(w[order])
    return float(z[order][np.searchsorted(cw, 0.5 * cw[-1])])


class HybridBranch:
    """Stage-1 prediction x stacked soft-sensor corrections (+ MAPE-optimal scaling)."""

    def __init__(self, scale=True):
        self.use_scale = scale

    def fit(self, train):
        lr = {v: np.log(train["obs_" + v] / train["s1_" + v]).to_numpy() for v in STATES}
        self.cell = {v: gbm().fit(train[CELL_COLS + ["ls1_" + v]].to_numpy(), lr[v]) for v in ("vcd", "tcd")}
        self.soft = SoftSensor().fit(train, lr["substrate"])
        self.ridge_S = DayRidge(S_COLS, days=set(range(3, 14))).fit(train, lr["substrate"] - self.soft.predict(train))
        self.ridge_L = DaySmooth(L_COLS).fit(train, lr["byproduct"])
        raw = self._raw(train)
        self.scale = {}
        if self.use_scale:
            for v in STATES:
                for day in np.unique(train.day):
                    m = (train.day == day).to_numpy()
                    self.scale[(v, day)] = mape_optimal_scale(raw[v].to_numpy()[m], train["obs_" + v].to_numpy()[m])
        return self

    def _raw(self, rows):
        corr = {v: np.zeros(len(rows)) for v in STATES}
        for v in ("vcd", "tcd"):
            corr[v] = self.cell[v].predict(rows[CELL_COLS + ["ls1_" + v]].to_numpy())
        corr["substrate"] = self.soft.predict(rows) + self.ridge_S.predict(rows)
        corr["byproduct"] = self.ridge_L.predict(rows)
        return pd.DataFrame({v: rows["s1_" + v].to_numpy() * np.exp(corr[v]) for v in STATES}, index=rows.index)

    def predict(self, rows):
        out = self._raw(rows)
        for v in STATES:
            out[v] = out[v].to_numpy() * np.array([self.scale.get((v, d), 1.0) for d in rows.day])
        return out


class DataDrivenBranch:
    """Gradient boosting on log(target), bagged over batch-level bootstrap samples."""

    def __init__(self, n_bags=10, seed=0):
        self.n_bags, self.seed = n_bags, seed

    def fit(self, train):
        rng = np.random.default_rng(self.seed)
        ids = train.batch_id.unique()
        by_batch = {b: g for b, g in train.groupby("batch_id")}
        self.models = []
        for i in range(self.n_bags):
            boot = pd.concat([by_batch[b] for b in rng.choice(ids, len(ids), replace=True)], ignore_index=True)
            self.models.append({v: gbm(depth=3, lr=0.05, iters=400, leaf=15, seed=i).fit(
                boot[DD_COLS[v]].to_numpy(), np.log(boot["obs_" + v].to_numpy())) for v in STATES})
        return self

    def predict(self, rows):
        return pd.DataFrame({v: np.exp(np.mean([m[v].predict(rows[DD_COLS[v]].to_numpy()) for m in self.models],
                                               axis=0)) for v in STATES}, index=rows.index)


# ---------------------------------------------------------------- idea 1: similarity member
class CellSimilarity:
    """VCD/TCD: correction curve log(obs/s1) of a batch = inverse-distance-weighted mean of the
    curves of its k most similar training batches (distance on standardised SIM_COLS, i.e. day-0
    state, feed-B deviation, early bolus). Substrate and byproduct are passed through as s1."""

    def __init__(self, k=5, cols=SIM_COLS):
        self.k, self.cols = k, cols

    def _desc(self, rows):
        return rows.groupby("batch_id")[self.cols].first()

    def fit(self, train):
        D = self._desc(train)
        self.mu, self.sd = D.mean(), D.std().replace(0, 1.0)
        self.Dz = (D - self.mu) / self.sd
        lr = train.assign(**{f"lr_{v}": np.log(train["obs_" + v] / train["s1_" + v]) for v in ("vcd", "tcd")})
        self.curves = {v: lr.pivot_table(index="batch_id", columns="day", values=f"lr_{v}") for v in ("vcd", "tcd")}
        return self

    def predict(self, rows):
        Dq = (self._desc(rows) - self.mu) / self.sd
        out = pd.DataFrame({v: rows["s1_" + v].to_numpy() for v in STATES}, index=rows.index)
        for b, zq in Dq.iterrows():
            d = np.sqrt(((self.Dz - zq) ** 2).sum(axis=1))
            d = d.drop(b, errors="ignore").nsmallest(self.k)
            w = 1.0 / (d + 0.1)
            m = (rows.batch_id == b).to_numpy()
            days = rows.day.to_numpy()[m]
            for v in ("vcd", "tcd"):
                C = self.curves[v].loc[d.index]
                corr = np.array([np.nansum(w * C[dd]) / w[C[dd].notna()].sum() if dd in C else 0.0 for dd in days])
                out.loc[m, v] = rows.loc[m, "s1_" + v].to_numpy() * np.exp(corr)
        return out


# ---------------------------------------------------------------- final blend
def finalize(hyb, dd, sim=None, w_sim=1 / 3):
    """50/50 geometric blend of hybrid and data-driven branch. With `sim` (idea 1), VCD/TCD become
    the weighted geometric mean (1 - w_sim)/2, (1 - w_sim)/2, w_sim of hyb, dd, sim. TCD >= VCD."""
    out = pd.DataFrame({v: np.exp(0.5 * np.log(hyb[v].to_numpy()) + 0.5 * np.log(dd[v].to_numpy()))
                        for v in STATES}, index=hyb.index)
    if sim is not None:
        for v in ("vcd", "tcd"):
            w = (1 - w_sim) / 2
            out[v] = np.exp(w * np.log(hyb[v].to_numpy()) + w * np.log(dd[v].to_numpy())
                            + w_sim * np.log(sim[v].to_numpy()))
    out["tcd"] = np.maximum(out.tcd, out.vcd)
    return out


def score(rows, pred):
    """Challenge metric: per-state MAPE % and the mean of the four."""
    s = {v: 100 * float(np.mean(np.abs(pred[v].to_numpy() / rows["obs_" + v].to_numpy() - 1))) for v in STATES}
    s["score"] = float(np.mean([s[v] for v in STATES]))
    return s
