"""Soft sensors from the feed / volume log (ported from Tobias' src/hybrid_v3.py, branch hybrid_v3_tr).

None of these read a measurement after day 0; they only use the feed events and the logged
volume, which are given for the test batches too.

1. Feed-schedule features (feed_features), one row per sample after day 0:
   cumulative feeds per volume, feed B / feed C right before and right after the sample,
   their deviation from the typical schedule of that day, day-0 state, and
   A = substrate bolus right after the sample / culture volume ("bolus soft sensor":
   a top-up to a target, so a larger bolus means glucose was lower before it).

2. Volume balance (volume_balance): unlogged volume = V(t) - V(0) - all logged feeds. Late in
   the run it falls by a constant amount per sample (sample removal); adding that back, the
   remaining extra volume on days 2-5 is base added by the pH controller (not in the log).
   Base neutralises acid -> proxy for the byproduct (lactate) produced.
       base_to3/4/5  base added up to day 3/4/5   (relative to days 1-2)
       base_tot      total base (plateau days 6-13 relative to days 1-2)
       base_cum      running total per day (0 up to day 2, base_tot from day 6)

3. Bolus features per batch: A11, A12 (bolus per volume around day 11 / 12), A_early (days 6-10.5).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEEDS

T0_COLS = ["vcd0", "s0", "l0", "v0", "via0"]
BASE_COLS = ["base_to3", "base_to4", "base_to5", "base_tot", "base_cum"]
BOLUS_COLS = ["A11", "A12", "A_early"]


def feed_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row per volume stamp after t=0 (measured sample or time to predict)."""
    rows = []
    for bid, g in df.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        stamps = g[g.volume.notna()]
        first = g[g.vcd.notna()].iloc[0]
        t_st = stamps.elapsed_time.to_numpy()
        for i, (_, s) in enumerate(stamps.iterrows()):
            t = s.elapsed_time
            if t <= first.elapsed_time:
                continue
            prev_t = t_st[i - 1]
            next_t = t_st[i + 1] if i + 1 < len(t_st) else t + 1.0
            prev_win = g[(g.elapsed_time > prev_t) & (g.elapsed_time <= t)]
            next_win = g[(g.elapsed_time > t) & (g.elapsed_time < min(next_t, t + 0.9))]
            so_far = g[g.elapsed_time <= t]
            rows.append({
                "batch_id": bid, "t": t, "day": int(round(t)), "V": s.volume,
                "cumB": so_far.feed_b.sum() / s.volume, "cumC": so_far.feed_c.sum() / s.volume,
                "cumS": so_far.feed_substrate.sum() / s.volume,
                "fb_next": next_win.feed_b.sum(), "fb_prev": prev_win.feed_b.sum(),
                "fc_next": next_win.feed_c.sum(), "fc_prev": prev_win.feed_c.sum(),
                "fs_next": next_win.feed_substrate.sum(),
                "vcd0": first.vcd, "s0": first.substrate, "l0": first.byproduct, "v0": first.volume,
                "via0": first.vcd / first.tcd,
            })
    F = pd.DataFrame(rows)
    # deviation of each batch from the typical schedule of that day (pooled over all batches;
    # uses only feeds/volume, no measurements)
    for c in ["fb_next", "fb_prev", "fc_next", "fc_prev", "V"]:
        F[c + "_dev"] = F[c] - F.groupby("day")[c].transform("mean")
    F["A"] = F.fs_next / F.V
    F["fs_group"] = np.where(F.A <= 0, "none", np.where(F.day == 11, "d11", np.where(F.day == 12, "d12", "early")))
    return F


def volume_balance(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Cumulative unlogged volume + k * (sample volume), per batch (rows) and sample day (columns)."""
    R = {}
    for bid, g in df.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        st = g[g.volume.notna()]
        v0 = st.volume.iloc[0]
        cum_feed = g[FEEDS].fillna(0).sum(axis=1).cumsum()
        R[bid] = {int(round(s.elapsed_time)): s.volume - v0 - cum_feed.loc[i] for i, s in st.iterrows()}
    R = pd.DataFrame(R).T.sort_index(axis=1)
    late = [k for k in R.columns if k >= 6]
    sample = -np.polyfit(late, R[late].mean().to_numpy(), 1)[0]     # volume removed per sample
    return R.add(sample * np.array(R.columns, dtype=float), axis=1), sample


def sensor_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """(per-batch features, per-(batch, day) base_cum, sample volume)."""
    A, sample = volume_balance(df)
    pre = A[[1, 2]].mean(axis=1)
    late = [k for k in A.columns if k >= 6]
    batch = pd.DataFrame({"base_to3": A[3] - pre, "base_to4": A[4] - pre, "base_to5": A[5] - pre,
                          "base_tot": A[late].mean(axis=1) - pre})
    bolus = {}
    for bid, g in df.groupby("batch_id"):
        st = g[g.volume.notna()].sort_values("elapsed_time")
        fs = g[g.feed_substrate.notna()]

        def per_volume(lo, hi):
            x = fs[(fs.elapsed_time > lo) & (fs.elapsed_time < hi)]
            return sum(r.feed_substrate / st[st.elapsed_time <= r.elapsed_time].volume.iloc[-1]
                       for _, r in x.iterrows())
        bolus[bid] = {"A11": per_volume(10.5, 12.0), "A12": per_volume(12.0, 14.0), "A_early": per_volume(6.0, 10.5)}
    batch = batch.join(pd.DataFrame(bolus).T.astype(float))
    day = A.sub(pre, axis=0).stack().rename("base_cum").reset_index()
    day.columns = ["batch_id", "day", "base_cum"]
    day = day.merge(batch[["base_tot"]].reset_index(names="batch_id"), on="batch_id")
    day.loc[day.day >= 6, "base_cum"] = day.loc[day.day >= 6, "base_tot"]
    day.loc[day.day <= 2, "base_cum"] = 0.0
    return batch, day[["batch_id", "day", "base_cum"]], sample


def base_rate(df: pd.DataFrame, sample: float) -> dict:
    """Per batch: (sample times, base volume added in each interval between samples), for using the
    base addition as a measured flux inside the ODE (idea 3). Only days 2-6 carry base; elsewhere 0."""
    out = {}
    for bid, g in df.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        st = g[g.volume.notna()]
        cum_feed = g[FEEDS].fillna(0).sum(axis=1).cumsum()
        t = st.elapsed_time.to_numpy()
        unlogged = np.array([s.volume - st.volume.iloc[0] - cum_feed.loc[i] for i, s in st.iterrows()])
        unlogged = unlogged + sample * np.arange(len(t))         # put the removed sample volume back
        inc = np.diff(unlogged, prepend=unlogged[0])
        day = np.rint(t).astype(int)
        inc = np.where((day >= 2) & (day <= 6), np.maximum(inc, 0.0), 0.0)
        out[bid] = (t, inc)
    return out
