"""Leak-free feature construction for forecasting from x(0) plus known feeds and volume.

Every feature of a row (batch b, sample time t) is computed only from:
  * batch b's measurements at t = 0,
  * batch b's sampling times and volumes, and
  * batch b's feed events.
These are exactly the inputs available for a test batch. Measurements after t = 0 are never
read, and no statistic is pooled across batches, so features can be built once outside the CV
loop without leaking between folds.
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from bioreactor_decoding.config import PROCESSED_DATA_DIR
from bioreactor_decoding.dataset import FEEDS, RAW_PATH, TARGETS, load_tables

META_COLS = ["batch_id", "elapsed_time"]

app = typer.Typer()


def _batch_features(s: pd.DataFrame, f: pd.DataFrame, use_future_feeds: bool) -> pd.DataFrame:
    s = s.sort_values("elapsed_time")
    t = s["elapsed_time"].to_numpy()
    if not np.isclose(t[0], 0.0):
        raise ValueError(f"Batch {s['batch_id'].iat[0]} has no t = 0 sample")
    x0 = s.iloc[0]
    vol = s["volume"].to_numpy()

    out = pd.DataFrame(
        {
            "batch_id": s["batch_id"].to_numpy(),
            "elapsed_time": t,
            "t": t,
            "k": np.arange(len(t)),
            "dt": np.diff(t, prepend=np.nan),
            "log_volume": np.log(vol),
            "vol_ratio": vol / vol[0],
            "dvol": np.diff(vol, prepend=np.nan),
        },
        index=s.index,
    )
    for c in TARGETS:
        out[f"log0_{c}"] = np.log(x0[c])
    out["viab0"] = x0["vcd"] / x0["tcd"]

    t_prev = np.r_[-np.inf, t[:-1]]
    t_next = np.r_[t[1:], np.inf]
    for name in FEEDS:
        ev = f.loc[f["feed"] == name]
        et, amt = ev["elapsed_time"].to_numpy(), ev["amount"].to_numpy()
        cum = np.r_[0.0, np.cumsum(amt)]
        # number of events with time <= x
        n_upto = np.searchsorted(et, t, side="right")
        n_prev = np.searchsorted(et, t_prev, side="right")
        out[f"cum_{name}"] = cum[n_upto]
        out[f"cum_{name}_per_vol"] = cum[n_upto] / vol
        out[f"int_{name}"] = cum[n_upto] - cum[n_prev]
        out[f"n_{name}"] = n_upto
        out[f"tsince_first_{name}"] = np.where(n_upto > 0, t - (et[0] if len(et) else 0), np.nan)
        if use_future_feeds:
            # Feeds dosed right after sample t reflect the operator's reading of sample t
            # (adaptive feeding). Legitimate here because test feed schedules are given.
            n_next = np.searchsorted(et, t_next, side="right")
            out[f"next_{name}"] = cum[n_next] - cum[n_upto]
    return out


def build_features(
    samples: pd.DataFrame, feeds: pd.DataFrame, use_future_feeds: bool = True
) -> pd.DataFrame:
    """One row per sample with t > 0. Returns META_COLS + feature columns, indexed like samples."""
    feeds_by_batch = dict(tuple(feeds.groupby("batch_id")))
    empty = feeds.iloc[0:0]
    parts = [
        _batch_features(s, feeds_by_batch.get(b, empty), use_future_feeds)
        for b, s in samples.groupby("batch_id", sort=False)
    ]
    feats = pd.concat(parts).sort_index()
    return feats.loc[feats["k"] > 0]


def feature_columns(feats: pd.DataFrame) -> list[str]:
    return [c for c in feats.columns if c not in META_COLS]


@app.command()
def main(
    input_path: Path = RAW_PATH,
    output_path: Path = PROCESSED_DATA_DIR / "features.csv",
    use_future_feeds: bool = True,
):
    samples, feeds = load_tables(input_path)
    feats = build_features(samples, feeds, use_future_feeds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(output_path, index=False)
    logger.success(f"{len(feats)} rows x {len(feature_columns(feats))} features -> {output_path}")


if __name__ == "__main__":
    app()
