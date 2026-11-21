"""Leak-free per-sample features for forecasting from x(0) + known feeds and volume.

Ported from Philip's `bioreactor_decoding/features.py` (branch philip/analysis), same logic and
names, adapted to this repo's data loading. Every feature of row (batch b, sample time t) uses
only: batch b's day-0 measurement, its sampling times and volumes, and its feed events -
exactly what is known for a test batch. Nothing is pooled across batches, so the table can be
built once outside the CV loop.

Feature groups
- time:    t, k (sample index), dt, log_volume, vol_ratio, dvol
- day 0:   log0_<state>, viab0
- feeds:   per feed: cum_ (added up to t), cum_*_per_vol, int_ (added since previous sample),
           n_ (number of events), tsince_first_, next_ (added between t and the next sample;
           reflects the operator's reaction to sample t - legitimate because the test feed
           schedules are given)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEEDS, STATES

META = ["batch_id", "elapsed_time"]   # k (sample index) stays a feature, as in Philip's code


def _tables(df):
    """(samples, feeds_long). Samples = rows with volume (incl. to_predict rows and day 0)."""
    samples = df[df.volume.notna()].sort_values(["batch_id", "elapsed_time"], kind="stable")
    feeds = (df[df.volume.isna()][["batch_id", "elapsed_time", *FEEDS]]
             .melt(id_vars=["batch_id", "elapsed_time"], var_name="feed", value_name="amount")
             .dropna(subset=["amount"]).sort_values(["batch_id", "elapsed_time"], kind="stable"))
    return samples, feeds


def _batch_features(s, f, use_future_feeds):
    t = s.elapsed_time.to_numpy()
    if not np.isclose(t[0], 0.0):
        raise ValueError(f"{s.batch_id.iat[0]}: no t = 0 sample")
    x0 = s.iloc[0]
    vol = s.volume.to_numpy()
    out = pd.DataFrame({
        "batch_id": s.batch_id.to_numpy(), "elapsed_time": t, "k": np.arange(len(t)),
        "t": t, "dt": np.diff(t, prepend=np.nan), "log_volume": np.log(vol),
        "vol_ratio": vol / vol[0], "dvol": np.diff(vol, prepend=np.nan),
    })
    for c in STATES:
        out[f"log0_{c}"] = np.log(x0[c])
    out["viab0"] = x0["vcd"] / x0["tcd"]
    t_prev = np.r_[-np.inf, t[:-1]]
    t_next = np.r_[t[1:], np.inf]
    for name in FEEDS:
        ev = f[f.feed == name]
        et, amt = ev.elapsed_time.to_numpy(), ev.amount.to_numpy()
        cum = np.r_[0.0, np.cumsum(amt)]
        n_upto = np.searchsorted(et, t, side="right")
        n_prev = np.searchsorted(et, t_prev, side="right")
        out[f"cum_{name}"] = cum[n_upto]
        out[f"cum_{name}_per_vol"] = cum[n_upto] / vol
        out[f"int_{name}"] = cum[n_upto] - cum[n_prev]
        out[f"n_{name}"] = n_upto
        out[f"tsince_first_{name}"] = np.where(n_upto > 0, t - (et[0] if len(et) else 0), np.nan)
        if use_future_feeds:
            out[f"next_{name}"] = cum[np.searchsorted(et, t_next, side="right")] - cum[n_upto]
    # targets (NaN for rows to predict)
    for c in STATES:
        out[f"y_{c}"] = s[c].to_numpy()
    out["to_predict"] = s.to_predict.to_numpy()
    return out


def build_features(df, use_future_feeds=True):
    """One row per sample with t > 0 (k >= 1), all 61 batches."""
    samples, feeds = _tables(df)
    fb = dict(tuple(feeds.groupby("batch_id")))
    empty = feeds.iloc[0:0]
    parts = [_batch_features(s, fb.get(b, empty), use_future_feeds)
             for b, s in samples.groupby("batch_id", sort=False)]
    F = pd.concat(parts, ignore_index=True)
    return F[F.k > 0].reset_index(drop=True)


def feature_columns(F):
    return [c for c in F.columns
            if c not in META and not c.startswith("y_") and c != "to_predict"]
