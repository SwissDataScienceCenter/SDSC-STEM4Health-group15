"""Mechanistic model as input to XGBoost (combination of notebook 02 and Philip's xgb).

For every sample row the mechanistic prediction (simulated from day 0 + feeds with parameters
fitted on the TRAINING batches only) is added as features:
    mech_log_<s> = log(mech prediction),  mech_lfc_<s> = log(mech) - log(day-0 value)

Targets (one XGBoost model per state, predictions in log space, always positive):
- "lfc":        z = log(y / y0)    (Philip's setup; mech enters only as a feature)
- "mech_resid": z = log(y / mech)  (XGBoost learns the correction to the mechanistic model)

Revision vs. Philip's version: the prediction is averaged over several seeds (in log space),
which removes most of the seed-to-seed noise of a single XGBoost fit on ~500 rows.
If xgboost is not installed, sklearn's HistGradientBoostingRegressor is used instead.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .data import STATES, to_arrays
from .model import predict_at_obs
from .xgb_features import feature_columns

EPS = 1e-3
XGB_PARAMS = dict(n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=3,
                  subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", n_jobs=4)

try:
    from xgboost import XGBRegressor

    def make_regressor(seed):
        return XGBRegressor(random_state=seed, **XGB_PARAMS)
except ImportError:  # e.g. local Python without xgboost
    from sklearn.ensemble import HistGradientBoostingRegressor
    warnings.warn("xgboost not installed -> using sklearn HistGradientBoostingRegressor")

    def make_regressor(seed):
        return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.03, max_depth=3,
                                             min_samples_leaf=10, random_state=seed)


# ---------------------------------------------------------------- challenge metric
def mape(y, yh):
    y, yh = np.asarray(y, float), np.asarray(yh, float)
    return float(np.mean(np.abs(y - yh) / np.abs(y)) * 100)


def challenge_score(Y, P):
    """Per-state MAPE % and 'score' = unweighted mean of the 4 (as in Philip's code)."""
    s = {c: mape(Y[c], P[c]) for c in STATES}
    s["score"] = float(np.mean([s[c] for c in STATES]))
    return s


# ---------------------------------------------------------------- mechanistic features
def add_mech_features(F, df, p, ids):
    """Return F restricted to `ids` with mech_* columns, simulated with parameters p.
    Rows are matched to the simulation by batch and sample index k (times checked)."""
    F = F[F.batch_id.isin(ids)].copy()
    lab = [b for b in ids if not F.loc[F.batch_id == b, "to_predict"].any()]
    tst = [b for b in ids if b not in lab]
    parts = []
    for bl, mask in [(lab, "measured"), (tst, "predict")]:
        if not bl:
            continue
        arr = to_arrays(df, bl, sample_mask=mask)
        preds = predict_at_obs(p, arr)
        for b, ts, ys in zip(arr.ids, arr.obs_t, preds):
            o = np.argsort(ts)
            d = pd.DataFrame(ys[o], columns=[f"mech_{s}" for s in STATES])
            d["batch_id"], d["k"], d["t_mech"] = b, np.arange(1, len(ts) + 1), ts[o]
            parts.append(d)
    M = pd.concat(parts)
    F = F.merge(M, on=["batch_id", "k"], how="left", validate="one_to_one")
    assert F[[f"mech_{s}" for s in STATES]].notna().all().all(), "missing mech prediction"
    assert np.allclose(F.t, F.t_mech, atol=1e-6), "sample times of features and simulation differ"
    for s in STATES:
        F[f"mech_log_{s}"] = np.log(np.maximum(F[f"mech_{s}"], 0) + EPS)
        F[f"mech_lfc_{s}"] = F[f"mech_log_{s}"] - F[f"log0_{s}"]
    return F.drop(columns=["t_mech"])


# ---------------------------------------------------------------- model
class MechXGB:
    """One gradient-boosting model per state, averaged over seeds.

    target: "lfc" or "mech_resid" (see module docstring)
    use_mech: include the mech_log_* / mech_lfc_* columns as features
    """

    def __init__(self, target="mech_resid", use_mech=True, seeds=(0, 1, 2)):
        if target == "mech_resid" and not use_mech:
            raise ValueError("mech_resid needs the mechanistic prediction")
        self.target, self.use_mech, self.seeds = target, use_mech, seeds

    def _cols(self, F):
        cols = feature_columns(F)
        if not self.use_mech:
            cols = [c for c in cols if not c.startswith("mech_")]
        else:  # raw mech_<s> duplicates mech_log_<s>; keep the log versions only
            cols = [c for c in cols if c.startswith(("mech_log_", "mech_lfc_")) or not c.startswith("mech_")]
        return cols

    def _offset(self, F, s):
        return F[f"log0_{s}"].to_numpy() if self.target == "lfc" else F[f"mech_log_{s}"].to_numpy()

    def fit(self, F):
        self.cols_ = self._cols(F)
        X = F[self.cols_]
        self.models_ = {}
        for s in STATES:
            z = np.log(F[f"y_{s}"].to_numpy()) - self._offset(F, s)
            self.models_[s] = [make_regressor(seed).fit(X, z) for seed in self.seeds]
        return self

    def predict(self, F):
        X = F[self.cols_]
        out = {}
        for s in STATES:
            z = np.mean([m.predict(X) for m in self.models_[s]], axis=0)
            out[s] = np.exp(z + self._offset(F, s))
        return pd.DataFrame(out, index=F.index)

    def importances(self):
        """Mean feature importance per state (only for models exposing feature_importances_)."""
        imp = {s: np.mean([m.feature_importances_ for m in ms], axis=0)
               for s, ms in self.models_.items() if hasattr(ms[0], "feature_importances_")}
        return pd.DataFrame(imp, index=self.cols_)


def per_day_median(F_tr, F_va):
    """Null model: median of each state per sampling day over the training rows."""
    d_tr, d_va = np.rint(F_tr.t).astype(int), np.rint(F_va.t).astype(int)
    med = F_tr[[f"y_{s}" for s in STATES]].groupby(d_tr.to_numpy()).median()
    med.columns = STATES
    d_va = np.clip(d_va, med.index.min(), med.index.max())
    return pd.DataFrame(med.reindex(d_va).to_numpy(), columns=STATES, index=F_va.index)


def fill_template(tmpl, F_test, P_test):
    """Write predictions into the template, aligned on batch_id + sorted time (times in the
    template are rounded, so an exact merge would fail)."""
    sub = tmpl.copy()
    sub[STATES] = sub[STATES].astype(float)          # template ships integer zeros
    t_tmpl = sub.elapsed_time.astype(float)
    order = sub.assign(_t=t_tmpl).sort_values(["batch_id", "_t"]).index
    pred = pd.concat([F_test[["batch_id", "t"]], P_test], axis=1).sort_values(["batch_id", "t"])
    ok = (len(order) == len(pred)
          and (sub.loc[order, "batch_id"].to_numpy() == pred.batch_id.to_numpy()).all()
          and np.allclose(t_tmpl.loc[order].to_numpy(), pred.t.to_numpy(), atol=1e-4))
    if not ok:
        raise ValueError("predictions do not align with the template rows")
    sub.loc[order, STATES] = pred[STATES].to_numpy()
    if (sub[STATES] <= 0).any().any():
        raise ValueError("non-positive predictions")
    return sub
