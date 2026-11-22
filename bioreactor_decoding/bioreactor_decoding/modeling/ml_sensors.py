"""ML model with soft sensors: ridge + XGBoost, no ODE (ported from branch 3_ML_models).

Source: notebooks/Average_v3_ml.ipynb, commit a52556f (the "ML model" half of the team's
average of v3 and ML). Label = log(value / per-day geometric mean curve); prediction = curve x
exp(model output). Two members, averaged geometrically:

  * row model: per target, the mean of a ridge and an XGBoost on every sample (day as a feature),
    with day-0 values, feeds so far / in the last 24 h / right after the sample, and the soft
    sensors below;
  * per-day model: per target and per culture day, one ridge on batch-level information.

Soft sensors (from the batch's own volume and feed log):
  * volume: volume not explained by the logged feeds, after adding back the sample removal
    (`sample_volume_`, the only pooled statistic, learned in `fit` from the training batches);
  * bolus: substrate bolus per culture volume on days 11 and 12, and on days 6-10.

Everything learned (mean curve, sample volume, ridge and XGBoost fits) is learned in `fit`;
the features of a batch depend on that batch's inputs only.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from bioreactor_decoding.dataset import FEEDS, RAW_PATH, TARGETS
from bioreactor_decoding.modeling.hybrid_v3 import input_tables

N_DAYS = 14
ROW_FEATURES = ["day", "vcd0", "viability0", "substrate0", "byproduct0", "volume0"] + [
    f"{p}_{f}" for f in FEEDS for p in ("cum", "last24h", "next")
]
VOL = ["base_cum", "base_to3", "base_to4", "base_to5", "base_tot"]
BOL = ["A11", "A12", "A_early"]
TOTALS = ["tot_sub", "tot_b", "tot_c", "tot_d"]
PERDAY = ["vcd0", "viability0", "substrate0", "byproduct0", "volume0", *TOTALS, *VOL, *BOL]
XGB_DEFAULTS = {
    "n_estimators": 400, "learning_rate": 0.03, "max_depth": 3, "min_child_weight": 3, "subsample": 0.8,
    "colsample_bytree": 0.8,
}  # fmt: skip


def batch_inputs(
    samples: pd.DataFrame, feeds: pd.DataFrame, batch_ids
) -> tuple[pd.DataFrame, dict]:
    """(rows, logs): one row per sample after t = 0 with the row features, and per batch the
    unlogged-volume curve and the bolus/total-feed numbers. Uses each batch's inputs only."""
    s_by, f_by = dict(tuple(samples.groupby("batch_id"))), dict(tuple(feeds.groupby("batch_id")))
    empty = feeds.iloc[0:0]
    rows, logs = [], {}
    for b in batch_ids:
        s = s_by[b].sort_values("elapsed_time")
        f = f_by.get(b, empty)
        ft, fa = f["elapsed_time"].to_numpy(), f["amount"].to_numpy()
        kind = {n: (f["feed"] == n).to_numpy() for n in FEEDS}
        t, V = s["elapsed_time"].to_numpy(), s["volume"].to_numpy()
        if len(t) != N_DAYS:
            raise ValueError(f"{b}: expected {N_DAYS} samples, got {len(t)}")
        d0 = s.iloc[0]

        def amount(name, lo, hi, kind=kind, ft=ft, fa=fa):  # feed `name` dosed in [lo, hi)
            return fa[kind[name] & (ft >= lo) & (ft < hi)].sum()

        for i in range(1, N_DAYS):
            t_next = t[i + 1] if i + 1 < N_DAYS else np.inf
            r = {"batch_id": b, "k": i, "t": t[i], "day": t[i], "vcd0": d0["vcd"],
                     "viability0": d0["vcd"] / d0["tcd"], "substrate0": d0["substrate"],
                     "byproduct0": d0["byproduct"], "volume0": d0["volume"]}  # fmt: skip
            for n in FEEDS:
                r[f"cum_{n}"] = amount(n, -np.inf, t[i])
                r[f"last24h_{n}"] = amount(n, t[i] - 1, t[i])
                r[f"next_{n}"] = amount(n, t[i], t_next)
            rows.append(r)
        cum = np.array([fa[ft < ti].sum() for ti in t])
        logs[b] = {
            "resid": V - V[0] - cum,  # volume not explained by the logged feeds
            "A11": amount("feed_substrate", 11, 12) / V[11],
            "A12": amount("feed_substrate", 12, 13) / V[12],
            "A_early": amount("feed_substrate", 6, 11) / V[8],
            "tot_sub": fa[kind["feed_substrate"]].sum(), "tot_b": fa[kind["feed_b"]].sum(),
            "tot_c": fa[kind["feed_c"]].sum(), "tot_d": fa[kind["feed_d"]].sum(),
        }  # fmt: skip
    return pd.DataFrame(rows), logs


def fit_sample_volume(logs: dict) -> float:
    """Volume removed per sample, from late culture (days 8-13) of the given batches."""
    return -float(np.median([(d["resid"][13] - d["resid"][8]) / 5 for d in logs.values()]))


def attach_sensors(rows: pd.DataFrame, logs: dict, sample_volume: float) -> pd.DataFrame:
    r = rows.copy()
    sens = {}
    for b, d in logs.items():
        A = d["resid"] + np.arange(N_DAYS) * sample_volume  # sample removal added back
        base = A[1:3].mean()  # early level (days 1-2)
        sens[b] = {"base_all": A - base, "base_to3": A[3] - base, "base_to4": A[4] - base,
                       "base_to5": A[5] - base, "base_tot": A[6:14].mean() - base}  # fmt: skip
    r["base_cum"] = [sens[b]["base_all"][k] for b, k in zip(r["batch_id"], r["k"])]
    for c in VOL[1:]:
        r[c] = r["batch_id"].map({b: v[c] for b, v in sens.items()})
    for c in BOL + TOTALS:
        r[c] = r["batch_id"].map({b: v[c] for b, v in logs.items()})
    return r


def _ridge(alphas):
    return make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))


class MLSensors(BaseEstimator, RegressorMixin):
    """Ridge + XGBoost with volume and bolus soft sensors (see module docstring)."""

    def __init__(self, xgb_params=None, raw_path=RAW_PATH, random_state=0):
        self.xgb_params = xgb_params
        self.raw_path = raw_path
        self.random_state = random_state

    def _inputs(self):
        if not hasattr(self, "_tables"):
            self._tables = input_tables(self.raw_path)
        return self._tables

    def _design(self, X):
        """Feature rows aligned with X (by batch and sample index k)."""
        ids = list(dict.fromkeys(X["batch_id"]))
        rows, logs = batch_inputs(*self._inputs(), ids)
        rows = attach_sensors(rows, logs, self.sample_volume_)
        key = pd.MultiIndex.from_arrays([X["batch_id"], X["k"].astype(int)])
        return rows.set_index(["batch_id", "k"]).loc[key].reset_index(), logs

    def _curve(self, t):
        days = self.curve_.index.to_numpy()
        return np.column_stack([np.interp(t, days, self.curve_[c].to_numpy()) for c in TARGETS])

    def _xgb(self):
        return XGBRegressor(**{**XGB_DEFAULTS, **(self.xgb_params or {})},
                            objective="reg:squarederror", n_jobs=4,
                            random_state=self.random_state)  # fmt: skip

    def fit(self, X: pd.DataFrame, Y: pd.DataFrame):
        ids = list(dict.fromkeys(X["batch_id"]))
        _, logs = batch_inputs(*self._inputs(), ids)
        self.sample_volume_ = fit_sample_volume(logs)
        R, _ = self._design(X)
        dayi = np.rint(R["t"]).astype(int).to_numpy()
        y = Y[TARGETS].to_numpy()
        self.curve_ = pd.DataFrame(np.log(y), columns=TARGETS).groupby(dayi).mean().apply(np.exp)
        lab = np.log(y / self._curve(R["t"].to_numpy()))
        feats = ROW_FEATURES + VOL + BOL
        self.row_models_ = [
            (_ridge(np.logspace(-2, 4, 30)).fit(R[feats], lab[:, j]),
             self._xgb().fit(R[feats], lab[:, j]))
            for j in range(len(TARGETS))
        ]  # fmt: skip
        self.day_models_ = {
            d: [_ridge(np.logspace(-1, 4, 30)).fit(R.loc[dayi == d, PERDAY], lab[dayi == d, j])
                for j in range(len(TARGETS))]
            for d in np.unique(dayi)
        }  # fmt: skip
        return self

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        R, _ = self._design(X)
        base = self._curve(R["t"].to_numpy())
        dayi = np.rint(R["t"]).astype(int).to_numpy()
        feats = ROW_FEATURES + VOL + BOL
        row = np.column_stack([(r.predict(R[feats]) + x.predict(R[feats])) / 2
                               for r, x in self.row_models_])  # fmt: skip
        day = np.zeros_like(row)
        for d in np.unique(dayi):
            m = dayi == d
            models = self.day_models_[min(self.day_models_, key=lambda k: abs(k - d))]
            day[m] = np.column_stack([mod.predict(R.loc[m, PERDAY]) for mod in models])
        return pd.DataFrame(base * np.exp((row + day) / 2), index=X.index, columns=TARGETS)

    def __getstate__(self):  # don't pickle the cached data tables
        state = self.__dict__.copy()
        state.pop("_tables", None)
        return state
