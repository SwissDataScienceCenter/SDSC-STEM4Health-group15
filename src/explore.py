"""Data-quality exploration for Decode the Bioreactor.

Run:  python src/explore.py      -> prints a report, writes figures to outputs/explore/
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bioreactor.data import load, STATES, FEEDS, ROOT  # noqa: E402

OUT = ROOT / "outputs" / "explore"
OUT.mkdir(parents=True, exist_ok=True)
UNITS = {"vcd": "VCD [1e6 cells/mL]", "tcd": "TCD [1e6 cells/mL]",
         "substrate": "substrate [g/L]", "byproduct": "byproduct [g/L]"}


def section(title):
    print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70)


def main():
    df, tmpl, train_ids, test_ids = load()
    tr = df[df.batch_id.isin(train_ids)]
    samp = tr[tr.vcd.notna()].copy()

    section("1. Batches")
    print(f"total batches: {df.batch_id.nunique()}")
    print(f"  with full measurements (usable for learning/validation): {len(train_ids)}")
    print(f"  hidden test batches (only day-0 sample given): {len(test_ids)} -> {test_ids}")
    print(f"template rows to fill: {len(tmpl)} ({len(tmpl)//len(test_ids)} time points x {len(test_ids)} batches)")

    section("2. Sampling")
    n = samp.groupby("batch_id").size()
    print(f"samples per training batch: min {n.min()}, median {n.median():.0f}, max {n.max()}")
    print(f"sampling days: ~daily, day 0 to {samp.elapsed_time.max():.2f}")
    samp["day"] = samp.elapsed_time.round().astype(int)
    dup = samp.groupby(["batch_id", "day"]).size()
    print(f"batch-days with >1 sample: {(dup > 1).sum()}; missing batch-days (0..13): "
          f"{len(train_ids)*14 - dup.size}")
    jitter = (samp.elapsed_time - samp.day).abs()
    print(f"sampling time jitter vs whole day: median {jitter.median()*24:.1f} h, max {jitter.max()*24:.1f} h")

    section("3. Missing values inside sample rows")
    print(samp[STATES + ["volume"]].isna().sum().to_string())

    section("4. Physical consistency")
    bad_tcd = samp[samp.tcd < samp.vcd - 1e-9]
    print(f"rows with TCD < VCD: {len(bad_tcd)}")
    samp["viability"] = samp.vcd / samp.tcd
    print(f"viability: min {samp.viability.min():.3f}, median {samp.viability.median():.3f}")
    neg = (samp[STATES] < 0).sum()
    print(f"negative values: {neg.to_dict()}")
    # volume balance: dV between samples vs total feed volume added in between
    rows = []
    for b, g in tr.groupby("batch_id"):
        v = g[g.volume.notna()]
        for (t0, v0), (t1, v1) in zip(v[["elapsed_time", "volume"]].values[:-1],
                                      v[["elapsed_time", "volume"]].values[1:]):
            m = (g.elapsed_time > t0) & (g.elapsed_time <= t1)
            rows.append([b, t1, v1 - v0, g.loc[m, FEEDS].sum().sum()])
    vb = pd.DataFrame(rows, columns=["b", "t", "dV", "feed"])
    gap = vb.dV - vb.feed
    print(f"volume balance dV - sum(feeds): median {gap.median()*1000:.1f} mL, "
          f"IQR [{gap.quantile(.25)*1000:.1f}, {gap.quantile(.75)*1000:.1f}] mL "
          f"(negative ~ sampling/evaporation)")

    section("5. Outliers (robust z vs. the across-batch median at each day)")
    flags = []
    for s in STATES:
        med = samp.groupby("day")[s].transform("median")
        mad = samp.groupby("day")[s].transform(lambda x: (x - x.median()).abs().median())
        z = (samp[s] - med) / (1.4826 * mad + 1e-9)
        f = samp[z.abs() > 4][["batch_id", "elapsed_time", s]].assign(z=z[z.abs() > 4].round(1))
        print(f"{s:10s}: {len(f)} points with |z|>4" +
              (f"  e.g. {f.head(4).to_dict('records')}" if len(f) else ""))
        flags.append(f.assign(var=s))
    # within-batch spikes: point far from mean of its neighbours
    spikes = []
    for b, g in samp.groupby("batch_id"):
        for s in ["vcd", "tcd"]:
            y = g[s].values
            if len(y) < 3:
                continue
            ly = np.log(y)  # growth is exponential -> compare in log space
            rel = np.exp(ly[1:-1] - (ly[:-2] + ly[2:]) / 2) - 1
            for i in np.where(np.abs(rel) > 0.25)[0]:
                spikes.append((b, s, round(g.elapsed_time.values[i+1], 2), round(rel[i], 2)))
    print(f"within-batch spikes in cell counts (>25% off neighbour mean): {len(spikes)} {spikes[:8]}")

    section("6. Batch-to-batch variability (training)")
    ic = df[df.elapsed_time == 0].set_index("batch_id")[STATES + ["volume"]]
    ft = df.groupby("batch_id")[FEEDS].sum()
    nfe = df.groupby("batch_id")[FEEDS].count().add_prefix("n_")
    kpi = samp.groupby("batch_id").agg(peak_vcd=("vcd", "max"),
                                       t_peak=("vcd", lambda x: samp.loc[x.idxmax(), "elapsed_time"]),
                                       final_vcd=("vcd", "last"), peak_byp=("byproduct", "max"),
                                       min_S=("substrate", "min"))
    print("outcome KPIs:\n" + kpi.describe().loc[["mean", "std", "min", "max"]].round(2).to_string())
    print("\nCV% of inputs: day-0 " + ", ".join(f"{c} {100*ic[c].std()/ic[c].mean():.1f}%" for c in ic) )
    print("CV% of total feed volume: " + ", ".join(f"{c} {100*ft[c].std()/ft[c].mean():.1f}%" for c in ft))
    print("number of feed events: " + ", ".join(f"{c} {nfe[c].min()}-{nfe[c].max()}" for c in nfe))

    section("7. Do the hidden test batches look like the training batches?")
    X = pd.concat([ic.drop(columns="volume"), ft], axis=1)
    X["set"] = np.where(X.index.isin(test_ids), "test", "train")
    cmp = X.groupby("set").agg(["mean", "min", "max"]).T.unstack(level=1).round(4)
    print(cmp.to_string())
    zt = (X[X.set == "test"].drop(columns="set") - X[X.set == "train"].drop(columns="set").mean()) \
        / X[X.set == "train"].drop(columns="set").std()
    print(f"\nmax |z| of any test-batch input vs train distribution: {zt.abs().max().max():.2f}")
    print("test batches outside train range on some input:",
          [b for b in zt.index if (zt.loc[b].abs() > 3).any()])

    section("8. Which inputs relate to outcomes? (Spearman, training batches)")
    J = pd.concat([X[X.set == "train"].drop(columns="set").add_prefix("in_"), kpi], axis=1)
    corr = J.corr(method="spearman").loc[[c for c in J if c.startswith("in_")], kpi.columns]
    print(corr.round(2).to_string())

    # ---------------- figures ----------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for a, s in zip(ax.ravel(), STATES):
        for b, g in samp.groupby("batch_id"):
            a.plot(g.elapsed_time, g[s], "-", lw=0.7, alpha=0.5, color="tab:blue")
        med = samp.groupby("day")[s].median()
        a.plot(med.index, med.values, "k-", lw=2, label="median")
        ff = [f for f in flags if s in f.columns][0]
        if len(ff):
            a.plot(ff.elapsed_time, ff[s], "rx", ms=8, label="|z|>4")
        a.set_xlabel("time [days]"); a.set_ylabel(UNITS[s]); a.legend(fontsize=8)
    fig.suptitle(f"All {len(train_ids)} training batches")
    fig.tight_layout(); fig.savefig(OUT / "01_trajectories.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 5, figsize=(18, 3.6))
    for a, c in zip(ax, FEEDS + ["volume"]):
        for b, g in df.groupby("batch_id"):
            g = g[g[c].notna()]
            col = "tab:red" if b in test_ids else "tab:blue"
            if c == "volume":
                a.plot(g.elapsed_time, g[c], "-", color=col, lw=0.7, alpha=0.6)
            else:
                a.plot(g.elapsed_time, g[c], "o", color=col, ms=3, alpha=0.5)
        a.set_title(c); a.set_xlabel("time [days]")
    ax[0].plot([], [], "o", color="tab:blue", label="train"); ax[0].plot([], [], "o", color="tab:red", label="test")
    ax[0].legend(fontsize=8)
    fig.suptitle("Feed events (volume added per event, L) and reactor volume")
    fig.tight_layout(); fig.savefig(OUT / "02_feeds.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    tr_ = X[X.set == "train"]; te_ = X[X.set == "test"]
    pairs = [("vcd", "feed_b"), ("feed_substrate", "feed_c")]
    for a, (cx, cy) in zip(ax, pairs):
        a.scatter(tr_[cx], tr_[cy], label="train", alpha=0.7)
        a.scatter(te_[cx], te_[cy], label="test", color="tab:red", marker="s")
        a.set_xlabel(cx + (" (day 0)" if cx in STATES else " (total L)"))
        a.set_ylabel(cy + " (total L)"); a.legend()
    fig.tight_layout(); fig.savefig(OUT / "03_train_vs_test_inputs.png", dpi=120); plt.close(fig)
    corr.round(3).to_csv(OUT / "input_outcome_spearman.csv")
    kpi.join(X).to_csv(OUT / "batch_summary.csv")
    print(f"\nfigures + tables written to {OUT}")


if __name__ == "__main__":
    main()
