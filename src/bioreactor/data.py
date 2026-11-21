"""Load the challenge data and turn each batch into arrays for vectorised simulation.

Data conventions (inferred from the files):
- one row per event: an offline sample (vcd, tcd, substrate, byproduct, volume)
  or a feed addition (feed_substrate, feed_b, feed_c, feed_d = added volume, L).
- feeds explain the volume increase (dV ~ 0.95*feed_b + 0.78*feed_substrate ...).
- test batches (to_predict == True) only have the day-0 sample; the rest must be predicted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
STATES = ["vcd", "tcd", "substrate", "byproduct"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]


def load(data_dir: Path = DATA):
    df = pd.read_csv(data_dir / "bioreaction_remaining.csv")
    tmpl = pd.read_csv(data_dir / "bioreaction_template.csv")
    df = df.sort_values(["batch_id", "elapsed_time"]).reset_index(drop=True)
    test_ids = sorted(df.loc[df.to_predict, "batch_id"].unique(), key=_num)
    train_ids = sorted(set(df.batch_id) - set(test_ids), key=_num)
    return df, tmpl, train_ids, test_ids


def _num(b: str) -> int:
    return int("".join(c for c in b if c.isdigit()))


@dataclass
class BatchArrays:
    ids: list[str]
    t: np.ndarray            # (K,) simulation grid [days]
    feeds: np.ndarray        # (K, B, 4) volume of each feed added at step k [L]
    x0: np.ndarray           # (B, 4) day-0 vcd, tcd, substrate, byproduct
    v0: np.ndarray           # (B,) initial volume [L]
    obs_idx: list[np.ndarray]   # per batch: grid indices of samples
    obs_t: list[np.ndarray]     # per batch: sample times
    obs_y: list[np.ndarray]     # per batch: (n, 4) observed states (NaN for test)
    base: np.ndarray | None = None   # (K, B) base volume added per grid step (option base_flux)


def to_arrays(df: pd.DataFrame, ids: list[str], dt: float = 0.005,
              t_end: float = 13.2, sample_mask: str = "measured", base: dict | None = None) -> BatchArrays:
    """sample_mask: 'measured' -> rows with vcd; 'predict' -> rows with to_predict.
    base: optional {batch_id: (sample_times, base_volume_per_interval)} from sensors.base_rate;
          each interval's base is spread evenly over the grid steps of that interval."""
    t = np.arange(0.0, t_end + dt, dt)
    K, B = len(t), len(ids)
    feeds = np.zeros((K, B, 4))
    x0 = np.zeros((B, 4)); v0 = np.zeros(B)
    oi, ot, oy = [], [], []
    for j, b in enumerate(ids):
        g = df[df.batch_id == b]
        first = g[g.elapsed_time == g.elapsed_time.min()].iloc[0]
        x0[j] = first[STATES].values.astype(float)
        v0[j] = first.volume
        for c_i, c in enumerate(FEEDS):
            ev = g[g[c].notna()]
            # feeds come after the sample at the same grid step -> ceil
            k = np.minimum(np.ceil(ev.elapsed_time.values / dt).astype(int), K - 1)
            np.add.at(feeds[:, j, c_i], k, ev[c].values)
        if sample_mask == "measured":
            s = g[g.vcd.notna() & (g.elapsed_time > 0)]
        else:
            s = g[g.to_predict]
        oi.append(np.round(s.elapsed_time.values / dt).astype(int))
        ot.append(s.elapsed_time.values)
        oy.append(s[STATES].values.astype(float))
    B_arr = None
    if base is not None:
        B_arr = np.zeros((K, B))
        for j, b in enumerate(ids):
            ts, inc = base[b]
            for t0_, t1_, v in zip(ts[:-1], ts[1:], inc[1:]):
                k = (t > t0_) & (t <= t1_)
                if v > 0 and k.any():
                    B_arr[k, j] = v / k.sum()
    return BatchArrays(ids, t, feeds, x0, v0, oi, ot, oy, B_arr)
