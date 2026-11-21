"""Hybrid v3 as one estimator: hybrid ODE (stage 1) + soft-sensor corrections + 50/50 blend.

Ported from branch hybrid_v3_tr (src/hybrid_v3.py, team CV score 7.67). See that branch's
outputs/hybrid_v3/improving_strategies.md for how each piece was chosen.

    stage 1     hybrid ODE (modeling.hybrid): open-loop forecast from x(0) + feed/volume log
    branch A    stage 1 x exp(correction of log(observed / stage 1)), stacked on log(stage 1):
                  cells      gradient boosting on t = 0 state, feed-B/C deviations, A_early
                  substrate  bolus soft sensor, then per-day ridge on base and bolus sensors
                  byproduct  varying-coefficient ridge on the base-addition (volume) sensor
                then MAPE-optimal scaling per variable and day
    branch B    gradient boosting on log(target), bagged over batch bootstrap samples
    final       50/50 geometric blend of A and B; TCD floored at VCD

Soft sensors (inputs only; they exist for test batches too):
  * volume balance: volume not explained by logged feeds (after adding back ~0.05 mL per
    sample) rises on days 2-5 = unlogged pH-control base, which tracks lactate produced;
  * glucose boluses are top-ups, so bolus volume per culture volume reveals how low glucose was.

Two leaks of the original are fixed here, so this is not bit-identical to it:
  * stacking: stage-2 training rows get stage-1 predictions from an inner GroupKFold over the
    training batches only (the original reused out-of-fold predictions from the same outer
    folds, so stage-1 models behind training features had seen the test fold);
  * pooled input statistics (per-day mean feed schedule, per-sample volume loss) are learned
    from the training batches in `fit`, instead of from all 61 batches.

Interface: X must carry `batch_id` and `t` (see train.labelled_data); every other feature is
rebuilt from the batch's feed and volume log, loaded with all measurements after t = 0 removed.
Y supplies the training observations.
"""

from __future__ import annotations

import itertools
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from bioreactor_decoding.config import MODELS_DIR
from bioreactor_decoding.dataset import FEEDS, RAW_PATH, TARGETS, load_tables
from bioreactor_decoding.modeling.hybrid import HybridODEConfig, fit_hybrid_odes
from bioreactor_decoding.modeling.lite import LiteODE, LiteODEConfig, build_arrays, fit_lite_odes
from bioreactor_decoding.modeling.mechanistic import DT, build_batches

T0_COLS = ["vcd0", "s0", "l0", "v0", "via0"]
DEV_COLS = ["fb_next_dev", "fb_prev_dev", "fc_next_dev", "fc_prev_dev", "V_dev"]
SCHEDULE_COLS = ["fb_next", "fb_prev", "fc_next", "fc_prev", "V"]
BASE_COLS = ["base_to3", "base_to4", "base_to5", "base_tot", "base_cum"]
BOLUS_COLS = ["A11", "A12", "A_early"]
CELL_COLS = [*T0_COLS, "day", *DEV_COLS, "A_early"]
S_COLS = ["base_tot", "A11", "A12", "A_early", "ls1_substrate"]
L_COLS = ["base_cum", "base_to3", "base_to4", "base_tot", "ls1_byproduct"]
_DD = ["t", *T0_COLS, "cumB", "cumC", "cumS", *DEV_COLS, "A"]
DD_COLS = {
    "vcd": [*_DD, "A_early"],
    "tcd": [*_DD, "A_early"],
    "substrate": [*_DD, *BASE_COLS, *BOLUS_COLS],
    "byproduct": [*_DD, *BASE_COLS],
}
SIM_COLS = ["vcd0", "s0", "l0", "via0", "fb_tot_dev", "A_early"]  # batch descriptors (idea 1)
N_DAYS = 14  # culture days 0..13


# --------------------------------------------------------------------------
# Inputs and soft-sensor features
# --------------------------------------------------------------------------


def input_tables(raw_path=RAW_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(samples, feeds) with every measurement after t = 0 removed: inputs only."""
    samples, feeds = load_tables(raw_path)
    samples = samples.copy()
    samples.loc[samples["elapsed_time"] > 0, TARGETS] = np.nan
    return samples, feeds


def stamp_features(samples: pd.DataFrame, feeds: pd.DataFrame, batch_ids) -> pd.DataFrame:
    """One row per sampling time after t = 0 of each batch, from that batch's inputs only."""
    rows = []
    s_by, f_by = dict(tuple(samples.groupby("batch_id"))), dict(tuple(feeds.groupby("batch_id")))
    empty = feeds.iloc[0:0]
    for b in batch_ids:
        s = s_by[b].sort_values("elapsed_time")
        f = f_by.get(b, empty)
        ft, fa = f["elapsed_time"].to_numpy(), f["amount"].to_numpy()
        kind = {name: (f["feed"] == name).to_numpy() for name in FEEDS}

        def total(name, lo, hi, kind=kind, ft=ft, fa=fa):
            """Amount of feed `name` dosed in the window (lo, hi]."""
            return fa[kind[name] & (ft > lo) & (ft <= hi)].sum()

        first = s.iloc[0]
        ts, vs = s["elapsed_time"].to_numpy(), s["volume"].to_numpy()
        for i in range(1, len(ts)):
            t, V = ts[i], vs[i]
            next_t = ts[i + 1] if i + 1 < len(ts) else t + 1.0
            hi_next = min(next_t, t + 0.9) - 1e-12  # window (t, min(next_t, t + 0.9))
            rows.append({
                "batch_id": b, "t": t, "day": int(np.rint(t)), "V": V,
                "cumB": total("feed_b", -np.inf, t) / V,
                "cumC": total("feed_c", -np.inf, t) / V,
                "cumS": total("feed_substrate", -np.inf, t) / V,
                "fb_next": total("feed_b", t, hi_next), "fb_prev": total("feed_b", ts[i - 1], t),
                "fc_next": total("feed_c", t, hi_next), "fc_prev": total("feed_c", ts[i - 1], t),
                "fs_next": total("feed_substrate", t, hi_next),
                "vcd0": first["vcd"], "s0": first["substrate"], "l0": first["byproduct"],
                "v0": first["volume"], "via0": first["vcd"] / first["tcd"],
            })  # fmt: skip
    F = pd.DataFrame(rows)
    F["A"] = F["fs_next"] / F["V"]  # substrate bolus volume per culture volume
    F["fs_group"] = np.where(
        F["A"] <= 0, "none",
        np.where(F["day"] == 11, "d11", np.where(F["day"] == 12, "d12", "early")),
    )  # fmt: skip
    return F


def unlogged_volume(samples: pd.DataFrame, feeds: pd.DataFrame, batch_ids) -> pd.DataFrame:
    """Batch x day: volume - V(0) - logged feeds so far (before adding back sample removal)."""
    R = {}
    s_by, f_by = dict(tuple(samples.groupby("batch_id"))), dict(tuple(feeds.groupby("batch_id")))
    for b in batch_ids:
        s = s_by[b].sort_values("elapsed_time")
        f = f_by.get(b)
        ft = f["elapsed_time"].to_numpy() if f is not None else np.array([])
        fa = f["amount"].to_numpy() if f is not None else np.array([])
        v0 = s["volume"].iat[0]
        R[b] = {
            int(np.rint(t)): v - v0 - fa[ft <= t].sum()
            for t, v in zip(s["elapsed_time"], s["volume"])
        }
    return pd.DataFrame(R).T.sort_index(axis=1)


def fit_sample_volume(R: pd.DataFrame) -> float:
    """Volume removed per sample: minus the slope of the mean unlogged volume on days >= 6."""
    late = [k for k in R.columns if k >= 6]
    return float(-np.polyfit(late, R[late].mean().to_numpy(), 1)[0])


def sensor_features(R: pd.DataFrame, sample: float, samples, feeds, batch_ids):
    """Per-batch (base_to3..5, base_tot, A11, A12, A_early) and per-(batch, day) base_cum."""
    A = R.add(sample * np.array(R.columns, dtype=float), axis=1)
    pre = A[[1, 2]].mean(axis=1)
    late = [k for k in A.columns if k >= 6]
    batch = pd.DataFrame({
        "base_to3": A[3] - pre, "base_to4": A[4] - pre, "base_to5": A[5] - pre,
        "base_tot": A[late].mean(axis=1) - pre,
    })  # fmt: skip
    s_by = dict(tuple(samples.groupby("batch_id")))
    fs = feeds[feeds["feed"] == "feed_substrate"]
    fs_by = dict(tuple(fs.groupby("batch_id")))
    bolus = {}
    for b in batch_ids:
        s = s_by[b].sort_values("elapsed_time")
        st, sv = s["elapsed_time"].to_numpy(), s["volume"].to_numpy()
        ev = fs_by.get(b, fs.iloc[0:0])

        def per_volume(lo, hi, ev=ev, st=st, sv=sv):
            x = ev[(ev["elapsed_time"] > lo) & (ev["elapsed_time"] < hi)]
            vol = sv[np.searchsorted(st, x["elapsed_time"].to_numpy(), side="right") - 1]
            return float((x["amount"].to_numpy() / vol).sum())

        bolus[b] = {
            "A11": per_volume(10.5, 12.0), "A12": per_volume(12.0, 14.0),
            "A_early": per_volume(6.0, 10.5),
        }  # fmt: skip
    batch = batch.join(pd.DataFrame(bolus).T.astype(float))
    day = A.sub(pre, axis=0).stack().rename("base_cum").reset_index()
    day.columns = ["batch_id", "day", "base_cum"]
    day = day.merge(batch[["base_tot"]].reset_index(names="batch_id"), on="batch_id")
    day.loc[day["day"] >= 6, "base_cum"] = day.loc[day["day"] >= 6, "base_tot"]
    day.loc[day["day"] <= 2, "base_cum"] = 0.0
    return batch, day[["batch_id", "day", "base_cum"]]


# --------------------------------------------------------------------------
# Stage-2 building blocks (unchanged from the original)
# --------------------------------------------------------------------------


def gbm(depth=2, lr=0.03, iters=200, leaf=30, random_state=0):
    return HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=iters,
                                         min_samples_leaf=leaf, l2_regularization=1.0,
                                         random_state=random_state)  # fmt: skip


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
        for day in np.unique(rows["day"]):
            if self.days is not None and day not in self.days:
                continue
            m = (rows["day"] == day).to_numpy()
            Z1 = np.column_stack([np.ones(m.sum()), Z[m]])
            P = self.alpha * np.eye(Z1.shape[1])
            P[0, 0] = 0.0
            self.coef[day] = np.linalg.solve(Z1.T @ Z1 + P, Z1.T @ y[m])
        return self

    def predict(self, rows):
        Z, out = self._z(rows), np.zeros(len(rows))
        for i, day in enumerate(rows["day"].to_numpy()):
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
        day = np.clip(rows["day"].to_numpy(), 0, N_DAYS - 1)
        d = (day - 7.0) / 6.0
        Phi = np.column_stack([d**k for k in range(self.degree + 1)])
        return np.column_stack(
            [np.eye(N_DAYS)[day]] + [Z[:, [j]] * Phi for j in range(Z.shape[1])]
        )

    def fit(self, rows, y):
        self._set_scale(rows)
        D = self._design(rows)
        P = self.alpha * np.eye(D.shape[1])
        P[:N_DAYS, :N_DAYS] = 1e-9 * np.eye(N_DAYS)
        self.coef = np.linalg.solve(D.T @ D + P, D.T @ y)
        return self

    def predict(self, rows):
        return self._design(rows) @ self.coef


class SoftSensor:
    """Substrate bolus soft sensor: log(obs/s1) = a + b*A + c*ls1, per day group."""

    def fit(self, rows, y):
        self.coef = {}
        for g in ["d11", "d12", "early"]:
            m = (rows["fs_group"] == g).to_numpy()
            if m.sum() >= 5:
                X = np.column_stack([np.ones(m.sum()), rows["A"].to_numpy()[m],
                                     rows["ls1_substrate"].to_numpy()[m]])  # fmt: skip
                self.coef[g] = np.linalg.lstsq(X, y[m], rcond=None)[0]
        return self

    def predict(self, rows):
        out = np.zeros(len(rows))
        for g, c in self.coef.items():
            m = (rows["fs_group"] == g).to_numpy()
            out[m] = (
                c[0] + c[1] * rows["A"].to_numpy()[m] + c[2] * rows["ls1_substrate"].to_numpy()[m]
            )
        return out


def mape_optimal_scale(pred: np.ndarray, obs: np.ndarray) -> float:
    """argmin_c sum |c*pred/obs - 1| = weighted median of obs/pred, weights pred/obs."""
    z, w = obs / pred, pred / obs
    order = np.argsort(z)
    cw = np.cumsum(w[order])
    return float(z[order][np.searchsorted(cw, 0.5 * cw[-1])])


class HybridBranch:
    """Branch A: stage-1 prediction x stacked soft-sensor corrections."""

    def fit(self, train):
        lr = {v: np.log(train["obs_" + v] / train["s1_" + v]).to_numpy() for v in TARGETS}
        self.cell = {v: gbm().fit(train[[*CELL_COLS, "ls1_" + v]].to_numpy(), lr[v])
                     for v in ("vcd", "tcd")}  # fmt: skip
        self.soft = SoftSensor().fit(train, lr["substrate"])
        self.ridge_S = DayRidge(S_COLS, days=set(range(3, N_DAYS))).fit(
            train, lr["substrate"] - self.soft.predict(train)
        )
        self.ridge_L = DaySmooth(L_COLS).fit(train, lr["byproduct"])
        raw = self._raw(train)
        self.scale = {}
        for v in TARGETS:
            for day in np.unique(train["day"]):
                m = (train["day"] == day).to_numpy()
                self.scale[(v, day)] = mape_optimal_scale(raw[v].to_numpy()[m],
                                                          train["obs_" + v].to_numpy()[m])  # fmt: skip
        return self

    def _raw(self, rows):
        corr = {v: np.zeros(len(rows)) for v in TARGETS}
        for v in ("vcd", "tcd"):
            corr[v] = self.cell[v].predict(rows[[*CELL_COLS, "ls1_" + v]].to_numpy())
        corr["substrate"] = self.soft.predict(rows) + self.ridge_S.predict(rows)
        corr["byproduct"] = self.ridge_L.predict(rows)
        return pd.DataFrame({v: rows["s1_" + v].to_numpy() * np.exp(corr[v]) for v in TARGETS},
                            index=rows.index)  # fmt: skip

    def predict(self, rows):
        out = self._raw(rows)
        for v in TARGETS:
            out[v] = out[v].to_numpy() * np.array(
                [self.scale.get((v, d), 1.0) for d in rows["day"]]
            )
        return out


class DataDrivenBranch:
    """Branch B: gradient boosting on log(target), bagged over batch-level bootstrap samples."""

    def __init__(self, n_bags=10, seed=0):
        self.n_bags, self.seed = n_bags, seed

    def fit(self, train):
        rng = np.random.default_rng(self.seed)
        ids = train["batch_id"].unique()
        by_batch = dict(tuple(train.groupby("batch_id")))
        self.models = []
        for _ in range(self.n_bags):
            boot = pd.concat([by_batch[b] for b in rng.choice(ids, len(ids), replace=True)],
                             ignore_index=True)  # fmt: skip
            self.models.append({
                v: gbm(depth=3, lr=0.05, iters=400, leaf=15).fit(
                    boot[DD_COLS[v]].to_numpy(), np.log(boot["obs_" + v].to_numpy()))
                for v in TARGETS
            })  # fmt: skip
        return self

    def predict(self, rows):
        return pd.DataFrame(
            {v: np.exp(np.mean([m[v].predict(rows[DD_COLS[v]].to_numpy()) for m in self.models],
                               axis=0)) for v in TARGETS},
            index=rows.index,
        )  # fmt: skip


class CellSimilarity:
    """Optional third member for VCD/TCD ("idea 1" of improving_strategies.md; ported from
    branch 06_hybrid_tr_based): a batch's correction curve log(observed / stage 1) is the
    inverse-distance-weighted mean of the curves of its k most similar training batches.
    Distance on SIM_COLS (t = 0 state, total feed-B deviation, early bolus), standardised with
    the training batches' mean and sd. Substrate and byproduct pass through as stage 1."""

    def __init__(self, k=5, cols=SIM_COLS):
        self.k, self.cols = k, cols

    def _desc(self, rows):
        return rows.groupby("batch_id")[self.cols].first()

    def fit(self, train):
        D = self._desc(train)
        self.mu, self.sd = D.mean(), D.std().replace(0, 1.0)
        self.Dz = (D - self.mu) / self.sd
        lr = train.assign(
            **{f"lr_{v}": np.log(train["obs_" + v] / train["s1_" + v]) for v in ("vcd", "tcd")}
        )
        self.curves = {
            v: lr.pivot_table(index="batch_id", columns="day", values=f"lr_{v}")
            for v in ("vcd", "tcd")
        }
        return self

    def predict(self, rows):
        Dq = (self._desc(rows) - self.mu) / self.sd
        out = pd.DataFrame({v: rows["s1_" + v].to_numpy() for v in TARGETS}, index=rows.index)
        days_all = rows["day"].to_numpy()
        for b, zq in Dq.iterrows():
            d = np.sqrt(((self.Dz - zq) ** 2).sum(axis=1))
            d = d.drop(b, errors="ignore").nsmallest(self.k)
            w = 1.0 / (d + 0.1)
            m = (rows["batch_id"] == b).to_numpy()
            for v in ("vcd", "tcd"):
                C = self.curves[v].loc[d.index]
                corr = np.array([
                    np.nansum(w * C[day]) / w[C[day].notna()].sum() if day in C else 0.0
                    for day in days_all[m]
                ])  # fmt: skip
                out.loc[m, v] = rows.loc[m, "s1_" + v].to_numpy() * np.exp(corr)
        return out


def finalize(hyb, dd, weight: float = 0.5, sim=None, sim_weight: float = 0.0) -> pd.DataFrame:
    """Geometric blend of the branches; TCD floored at VCD.

    Branch A gets `weight`, branch B `1 - weight`. With a similarity member, VCD and TCD give it
    `sim_weight` and scale the two branches by 1 - sim_weight.
    """
    out = pd.DataFrame(
        {v: np.exp(weight * np.log(hyb[v].to_numpy()) + (1 - weight) * np.log(dd[v].to_numpy()))
         for v in TARGETS},
        index=hyb.index,
    )  # fmt: skip
    if sim is not None and sim_weight > 0:
        for v in ("vcd", "tcd"):
            out[v] = np.exp((1 - sim_weight) * np.log(out[v].to_numpy())
                            + sim_weight * np.log(sim[v].to_numpy()))  # fmt: skip
    out["tcd"] = np.maximum(out["tcd"], out["vcd"])
    return out


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------


class HybridV3(BaseEstimator, RegressorMixin):
    """Hybrid v3 with the scikit-learn API: fit(X, Y) / predict(X) -> DataFrame of TARGETS.

    Parameters
    ----------
    inner_folds : batch-wise folds used inside `fit` to make stage-1 predictions for the
        stage-2 training rows (each from a stage-1 model that did not see that batch).
    n_bags : bootstrap bags of branch B.
    blend_weight : weight of branch A in the geometric blend (0.5 = the original).
    stage1 : stage-1 settings: HybridODEConfig (hybrid ODE with neural nets; None = the
        original settings) or LiteODEConfig (mechanistic ODE only, ~10x cheaper).
    similarity_weight : weight of the optional similarity member for VCD/TCD (0 = off).
    similarity_k : number of similar training batches it averages.
    n_jobs : processes for the stage-1 least-squares fits (None = all cores).
    cache_dir : joblib cache for stage-1 fits, keyed on the training data and config, so
        refitting on the same batches (figures, reruns) is free. None disables caching.
    """

    def __init__(self, inner_folds=5, n_bags=10, blend_weight=0.5, stage1=None,
                 similarity_weight=0.0, similarity_k=5, n_jobs=None, raw_path=RAW_PATH,
                 random_state=0, cache_dir=MODELS_DIR / "cache"):  # fmt: skip  # fmt: skip
        self.inner_folds = inner_folds
        self.n_bags = n_bags
        self.blend_weight = blend_weight
        self.stage1 = stage1
        self.similarity_weight = similarity_weight
        self.similarity_k = similarity_k
        self.n_jobs = n_jobs
        self.raw_path = raw_path
        self.random_state = random_state
        self.cache_dir = cache_dir

    # -- helpers -----------------------------------------------------------

    def _inputs(self):
        if not hasattr(self, "_tables"):
            self._tables = input_tables(self.raw_path)
        return self._tables

    @staticmethod
    def _batches(X):
        return list(dict.fromkeys(X["batch_id"]))

    def _config(self):
        return self.stage1 or HybridODEConfig()

    def _build(self, ids, obs=None):
        """Stage-1 inputs for `ids` in the format of the configured stage 1."""
        samples, feeds = self._inputs()
        if isinstance(self._config(), LiteODEConfig):
            return build_arrays(samples, feeds, ids, obs)
        return build_batches(samples, feeds, ids, obs)

    @staticmethod
    def _predict_s1(model, data, batch_idx, times) -> np.ndarray:
        if isinstance(model, LiteODE):
            return model.predict(data, batch_idx, times)
        return model.predict(data, batch_idx, np.rint(np.asarray(times) / DT).astype(int))

    def _rows(self, X, s1: np.ndarray) -> pd.DataFrame:
        """Stage-2 design rows for X (same order), given stage-1 predictions s1 (n, 4)."""
        samples, feeds = self._inputs()
        ids = self._batches(X)
        F = stamp_features(samples, feeds, ids)
        for c in SCHEDULE_COLS:
            mean = F["day"].map(self.schedule_[c]).fillna(self.schedule_[c].mean())
            F[c + "_dev"] = F[c] - mean
        batch, day = sensor_features(unlogged_volume(samples, feeds, ids), self.sample_volume_,
                                     samples, feeds, ids)  # fmt: skip
        F = F.merge(batch.reset_index(names="batch_id"), on="batch_id", how="left")
        F = F.merge(day, on=["batch_id", "day"], how="left")

        key = pd.DataFrame({"batch_id": X["batch_id"].to_numpy(),
                            "t_key": np.round(X["t"].to_numpy(), 6)})  # fmt: skip
        F["t_key"] = F["t"].round(6)
        rows = key.merge(F.drop(columns="t"), on=["batch_id", "t_key"], how="left",
                         validate="one_to_one")  # fmt: skip
        rows["t"] = X["t"].to_numpy()
        rows["fb_tot_dev"] = rows.groupby("batch_id")["fb_prev_dev"].transform("sum")
        for j, v in enumerate(TARGETS):
            rows["s1_" + v] = s1[:, j]
            rows["ls1_" + v] = np.log(s1[:, j])
        needed = sorted(
            {*CELL_COLS, *S_COLS, *L_COLS, *itertools.chain(*DD_COLS.values()), "fs_group"}
        )
        if rows[needed].isna().any().any():
            bad = rows[needed].columns[rows[needed].isna().any()].tolist()
            raise ValueError(f"Missing stage-2 features {bad}")
        rows.index = X.index
        return rows

    # -- API ---------------------------------------------------------------

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        samples, feeds = self._inputs()
        config = self._config()
        n_jobs = self.n_jobs or os.cpu_count() or 1
        ids = self._batches(X)
        times = X["t"].to_numpy(float)
        obs = pd.DataFrame({"batch_id": X["batch_id"].to_numpy(), "elapsed_time": times})
        obs[TARGETS] = Y[TARGETS].to_numpy(float)
        bs = self._build(ids, obs)
        b_idx = X["batch_id"].map({b: i for i, b in enumerate(ids)}).to_numpy()

        # Stage 1: one model on all training batches (for predict) plus one per inner fold
        # (for the stage-2 training features, each row predicted by a model blind to its batch).
        pos = np.arange(len(ids))
        inner = list(GroupKFold(self.inner_folds).split(pos, groups=pos))
        jobs = {"full": bs, **{f"inner{i}": bs.subset(tr) for i, (tr, _) in enumerate(inner)}}
        fit = fit_lite_odes if isinstance(config, LiteODEConfig) else fit_hybrid_odes
        self.stage1_from_cache_ = False
        if self.cache_dir is not None:
            fit = joblib.Memory(self.cache_dir, verbose=0).cache(fit, ignore=["n_jobs"])
            self.stage1_from_cache_ = fit.check_call_in_cache(jobs, config, n_jobs)
        fitted = fit(jobs, config, n_jobs)
        self.stage1_ = fitted["full"]
        s1 = np.empty((len(X), len(TARGETS)))
        for i, (_, te) in enumerate(inner):
            m = np.isin(b_idx, te)
            s1[m] = self._predict_s1(fitted[f"inner{i}"], bs, b_idx[m], times[m])

        # Pooled input statistics, learned from the training batches only.
        F = stamp_features(samples, feeds, ids)
        self.schedule_ = {c: F.groupby("day")[c].mean() for c in SCHEDULE_COLS}
        self.sample_volume_ = fit_sample_volume(unlogged_volume(samples, feeds, ids))

        rows = self._rows(X, s1)
        for v in TARGETS:
            rows["obs_" + v] = Y[v].to_numpy(float)
        self.branch_a_ = HybridBranch().fit(rows)
        self.branch_b_ = DataDrivenBranch(self.n_bags, self.random_state).fit(rows)
        self.similarity_ = None
        if self.similarity_weight > 0:
            self.similarity_ = CellSimilarity(self.similarity_k).fit(rows)
        return self

    def predict_stage1(self, X: pd.DataFrame) -> np.ndarray:
        ids = self._batches(X)
        b_idx = X["batch_id"].map({b: i for i, b in enumerate(ids)}).to_numpy()
        return self._predict_s1(self.stage1_, self._build(ids), b_idx, X["t"].to_numpy(float))

    def predict_members(self, X: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Predictions of each blend member: branch A, branch B and (if on) similarity."""
        rows = self._rows(X, self.predict_stage1(X))
        out = {"branch_a": self.branch_a_.predict(rows), "branch_b": self.branch_b_.predict(rows)}
        if self.similarity_ is not None:
            out["similarity"] = self.similarity_.predict(rows)
        return out

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        m = self.predict_members(X)
        return finalize(m["branch_a"], m["branch_b"], self.blend_weight,
                        m.get("similarity"), self.similarity_weight)  # fmt: skip

    def __getstate__(self):  # don't pickle the cached data tables
        state = super().__getstate__()
        state.pop("_tables", None)
        return state
