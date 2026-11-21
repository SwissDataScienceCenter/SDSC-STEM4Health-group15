"""Hybrid v3: the best model. Hybrid ODE + soft-sensor corrections + blend.

Held-out score (7-fold grouped CV, leaderboard metric = mean of 4 MAPEs): 7.67
(VCD 6.97, TCD 6.95, substrate 8.02, byproduct 8.76).

Stage 1 is the hybrid ODE model in src/hybrid_model.py (mechanistic core +
small neural net correcting growth and lactate rates). It forecasts open-loop
from t=0 and the feeds, and its out-of-fold and holdout predictions are read
from outputs/hybrid/. Run it first:

    python src/hybrid_model.py
    python src/hybrid_v3.py

Stage 2 corrects stage 1 with information in the event log that the ODE does
not use. All of it exists for the holdout batches too:

0. Feed-log features. The t=0 state, deviations of feed B / feed C / volume
   from the typical daily schedule, cumulative feeds, and the substrate-bolus
   soft sensor: the glucose bolus is a top-up, so its volume per culture
   volume (A) reveals how low glucose was before it.

1. Volume-balance sensor (lactate, glucose). The culture volume is logged at
   every sampling time. The unlogged volume, V(t) - V(0) - (logged feeds), falls
   by a constant ~0.05 mL per sample late in culture (sample removal). Between
   days 2 and 5 it rises by about 0.18 mL more than the feeds explain, and that
   extra volume tracks how much lactate the batch produced (within-day
   r = +0.56) and how much glucose it consumed (r = -0.50). This is consistent
   with unlogged pH-control base addition starting once the medium's buffer is
   used up.
       A_k       cumulative unlogged volume at sample k plus k * sample volume
       base_cumK base added up to day K = A_K - mean(A_1, A_2)
       base_tot  total base = mean(A_6..A_13) - mean(A_1, A_2)

2. Bolus-decision sensor (glucose, cells). Glucose boluses top glucose up when
   it runs low, so their sizes (per volume) reveal the glucose level around
   them: the day-11 bolus reflects glucose on days 9-12, the day-12 bolus
   reflects days 11-13. Early boluses (days 7-9) happen only in batches that
   consumed more glucose, which also had more cells.
       A11, A12  bolus volume per culture volume on day ~11.4 and ~12.4
       A_early   bolus volume per culture volume between days 6 and 10.5

How they are used (all fitted on training folds only; "ls1" = log of the
stage-1 prediction, which every correction may use, i.e. stacking)
    hybrid branch, corrections of log(observed / stage-1):
        cells      gradient boosting on t=0 state, feed-B deviations, A_early, ls1
        substrate  bolus soft sensor (A, ls1) for rows followed by a bolus, then
                   per-day ridge on base_tot, A11, A12, A_early, ls1 (days 3-13)
        byproduct  varying-coefficient ridge (free per-day intercepts, sensor
                   effects quadratic in day) on base_cum, base_to3, base_to4,
                   base_tot, ls1
        then MAPE-optimal scaling per variable and day
    data-driven branch: gradient boosting on log(target), bagged over 10
        batch-level bootstrap samples; cells get t=0, time, feeds and A_early;
        substrate gets all sensors; byproduct gets the volume (lactate) sensor
    final: fixed 50/50 geometric blend of the two branches; TCD floored at VCD

Run from the repo root (after src/hybrid_model.py):

    python src/hybrid_v3.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "hybrid_v3"
DATA_DIR = ROOT / "input-data" / "Project 1 - DTB"
STAGE1_DIR = ROOT / "outputs" / "hybrid"  # output of src/hybrid_model.py
STAGE1_MODEL = "states"
N_FOLDS = 7
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
T0_COLS = ["vcd0", "s0", "l0", "v0", "via0"]
V2_CELL_COLS = T0_COLS + ["day", "fb_next_dev", "fb_prev_dev", "fc_next_dev", "fc_prev_dev", "V_dev"]
V2_DD_COLS = ["t", *T0_COLS, "cumB", "cumC", "cumS", "fb_next_dev", "fb_prev_dev", "fc_next_dev", "fc_prev_dev",
              "V_dev", "A"]
FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]
BASE_COLS = ["base_to3", "base_to4", "base_to5", "base_tot", "base_cum"]
BOLUS_COLS = ["A11", "A12", "A_early"]


# --------------------------------------------------------------------------
# Soft-sensor features (inputs only)
# --------------------------------------------------------------------------


def feed_features(events: pd.DataFrame) -> pd.DataFrame:
    """One row per volume stamp after t=0 (measurement or prediction time)."""
    rows = []
    for bid, g in events.groupby("batch_id"):
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
                "batch_id": bid, "t_key": round(t, 6), "day": int(round(t)), "t": t, "V": s.volume,
                "cumB": so_far.feed_b.sum() / s.volume, "cumC": so_far.feed_c.sum() / s.volume,
                "cumS": so_far.feed_substrate.sum() / s.volume,
                "fb_next": next_win.feed_b.sum(), "fb_prev": prev_win.feed_b.sum(),
                "fc_next": next_win.feed_c.sum(), "fc_prev": prev_win.feed_c.sum(),
                "fs_next": next_win.feed_substrate.sum(),
                "vcd0": first.vcd, "s0": first.substrate, "l0": first.byproduct, "v0": first.volume,
                "via0": first.vcd / first.tcd,
            })
    F = pd.DataFrame(rows)
    # deviation of each batch from the typical schedule of that day
    for c in ["fb_next", "fb_prev", "fc_next", "fc_prev", "V"]:
        F[c + "_dev"] = F[c] - F.groupby("day")[c].transform("mean")
    F["A"] = F.fs_next / F.V  # substrate bolus volume per culture volume
    F["fs_group"] = np.where(F.A <= 0, "none", np.where(F.day == 11, "d11", np.where(F.day == 12, "d12", "early")))
    return F



def volume_balance(events: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Cumulative unlogged volume + k * sample volume, per batch and sample day."""
    R = {}
    for bid, g in events.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        st = g[g.volume.notna()]
        v0 = st.volume.iloc[0]
        cum_feed = g[FEEDS].fillna(0).sum(axis=1).cumsum()
        R[bid] = {int(round(s.elapsed_time)): s.volume - v0 - cum_feed.loc[i] for i, s in st.iterrows()}
    R = pd.DataFrame(R).T.sort_index(axis=1)
    late = [k for k in R.columns if k >= 6]
    sample = -np.polyfit(late, R[late].mean().to_numpy(), 1)[0]
    return R.add(sample * np.array(R.columns, dtype=float), axis=1), sample


def sensor_features(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Per-batch and per-(batch, day) soft-sensor features."""
    A, sample = volume_balance(events)
    pre = A[[1, 2]].mean(axis=1)
    late = [k for k in A.columns if k >= 6]
    batch = pd.DataFrame({"base_to3": A[3] - pre, "base_to4": A[4] - pre, "base_to5": A[5] - pre,
                          "base_tot": A[late].mean(axis=1) - pre})
    bolus = {}
    for bid, g in events.groupby("batch_id"):
        st = g[g.volume.notna()].sort_values("elapsed_time")
        fs = g[g.feed_substrate.notna()]

        def per_volume(lo, hi):
            x = fs[(fs.elapsed_time > lo) & (fs.elapsed_time < hi)]
            return sum(r.feed_substrate / st[st.elapsed_time <= r.elapsed_time].volume.iloc[-1] for _, r in x.iterrows())
        bolus[bid] = {"A11": per_volume(10.5, 12.0), "A12": per_volume(12.0, 14.0), "A_early": per_volume(6.0, 10.5)}
    batch = batch.join(pd.DataFrame(bolus).T.astype(float))
    day = A.sub(pre, axis=0).stack().rename("base_cum").reset_index()
    day.columns = ["batch_id", "day", "base_cum"]
    day = day.merge(batch[["base_tot"]].reset_index(names="batch_id"), on="batch_id")
    day.loc[day.day >= 6, "base_cum"] = day.loc[day.day >= 6, "base_tot"]
    day.loc[day.day <= 2, "base_cum"] = 0.0
    return batch, day[["batch_id", "day", "base_cum"]], sample


# --------------------------------------------------------------------------
# Stage-2 building blocks
# --------------------------------------------------------------------------


def gbm(depth=2, lr=0.03, iters=200, leaf=30):
    return HistGradientBoostingRegressor(max_depth=depth, learning_rate=lr, max_iter=iters, min_samples_leaf=leaf,
                                         l2_regularization=1.0, random_state=0)


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
            P = self.alpha * np.eye(Z1.shape[1])
            P[0, 0] = 0.0
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
        return np.column_stack([np.eye(14)[rows.day.to_numpy()]] + [Z[:, [j]] * Phi for j in range(Z.shape[1])])

    def fit(self, rows, y):
        self._set_scale(rows)
        D = self._design(rows)
        P = self.alpha * np.eye(D.shape[1])
        P[:14, :14] = 1e-9 * np.eye(14)
        self.coef = np.linalg.solve(D.T @ D + P, D.T @ y)
        return self

    def predict(self, rows):
        return self._design(rows) @ self.coef


class SoftSensor:
    """Substrate bolus soft sensor: log(obs/s1) = a + b*A + c*ls1, per day group."""

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


CELL_COLS = V2_CELL_COLS + ["A_early"]
S_COLS = ["base_tot", "A11", "A12", "A_early", "ls1_substrate"]
L_COLS = ["base_cum", "base_to3", "base_to4", "base_tot", "ls1_byproduct"]
DD_COLS = {"vcd": V2_DD_COLS + ["A_early"], "tcd": V2_DD_COLS + ["A_early"],
           "substrate": V2_DD_COLS + BASE_COLS + BOLUS_COLS, "byproduct": V2_DD_COLS + BASE_COLS}


class HybridBranch:
    """Stage-1 hybrid prediction x stacked soft-sensor corrections."""

    def fit(self, train):
        lr = {v: np.log(train["obs_" + v] / train["s1_" + v]).to_numpy() for v in TARGETS}
        self.cell = {v: gbm().fit(train[CELL_COLS + ["ls1_" + v]].to_numpy(), lr[v]) for v in ("vcd", "tcd")}
        self.soft = SoftSensor().fit(train, lr["substrate"])
        self.ridge_S = DayRidge(S_COLS, days=set(range(3, 14))).fit(train, lr["substrate"] - self.soft.predict(train))
        self.ridge_L = DaySmooth(L_COLS).fit(train, lr["byproduct"])
        raw = self._raw(train)
        self.scale = {}
        for v in TARGETS:
            for day in np.unique(train.day):
                m = (train.day == day).to_numpy()
                self.scale[(v, day)] = mape_optimal_scale(raw[v].to_numpy()[m], train["obs_" + v].to_numpy()[m])
        return self

    def _raw(self, rows):
        corr = {v: np.zeros(len(rows)) for v in TARGETS}
        for v in ("vcd", "tcd"):
            corr[v] = self.cell[v].predict(rows[CELL_COLS + ["ls1_" + v]].to_numpy())
        corr["substrate"] = self.soft.predict(rows) + self.ridge_S.predict(rows)
        corr["byproduct"] = self.ridge_L.predict(rows)
        return pd.DataFrame({v: rows["s1_" + v].to_numpy() * np.exp(corr[v]) for v in TARGETS}, index=rows.index)

    def predict(self, rows):
        out = self._raw(rows)
        for v in TARGETS:
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
        for _ in range(self.n_bags):
            boot = pd.concat([by_batch[b] for b in rng.choice(ids, len(ids), replace=True)], ignore_index=True)
            self.models.append({v: gbm(depth=3, lr=0.05, iters=400, leaf=15).fit(
                boot[DD_COLS[v]].to_numpy(), np.log(boot["obs_" + v].to_numpy())) for v in TARGETS})
        return self

    def predict(self, rows):
        return pd.DataFrame({v: np.exp(np.mean([m[v].predict(rows[DD_COLS[v]].to_numpy()) for m in self.models], axis=0))
                             for v in TARGETS}, index=rows.index)


def mape_optimal_scale(pred: np.ndarray, obs: np.ndarray) -> float:
    """argmin_c sum |c*pred/obs - 1|  = weighted median of obs/pred, weights pred/obs."""
    z = obs / pred
    w = pred / obs
    order = np.argsort(z)
    cw = np.cumsum(w[order])
    return float(z[order][np.searchsorted(cw, 0.5 * cw[-1])])



def blend(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fixed 50/50 geometric mean."""
    return np.exp(0.5 * np.log(a) + 0.5 * np.log(b))


def mape_table(frame: pd.DataFrame, pred_cols: dict) -> pd.Series:
    vals = {v: 100 * np.mean(np.abs(frame[pred_cols[v]] / frame["obs_" + v] - 1)) for v in TARGETS}
    s = pd.Series(vals)
    s["leaderboard_score"] = s.mean()
    return s



def finalize(hyb: pd.DataFrame, dd: pd.DataFrame) -> pd.DataFrame:
    """Fixed 50/50 geometric blend of the two branches; TCD floored at VCD."""
    out = pd.DataFrame({v: blend(hyb[v].to_numpy(), dd[v].to_numpy()) for v in TARGETS}, index=hyb.index)
    out["tcd"] = np.maximum(out.tcd, out.vcd)
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def build_rows(F, sens_batch, sens_day, stage1_wide):
    rows = stage1_wide.merge(F, on=["batch_id", "t_key"], how="left")
    rows = rows.merge(sens_batch.reset_index(names="batch_id"), on="batch_id", how="left")
    rows = rows.merge(sens_day, on=["batch_id", "day"], how="left").reset_index(drop=True)
    for v in TARGETS:
        rows["ls1_" + v] = np.log(rows["s1_" + v])
    return rows


def load_stage1(stage1_dir: Path) -> pd.DataFrame:
    s1 = pd.read_csv(stage1_dir / "predictions_vs_measured.csv", dtype={"batch_id": str})
    s1 = s1[(s1.split == "held-out (CV)") & (s1.model == STAGE1_MODEL)].copy()
    s1["t_key"] = s1.elapsed_time.round(6)
    wide = s1.pivot_table(index=["batch_id", "t_key"], columns="variable", values=["predicted", "observed"]).reset_index()
    wide.columns = [{"predicted": "s1_", "observed": "obs_"}.get(a, a) + (b or "") for a, b in wide.columns]
    return wide


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1", type=Path, default=STAGE1_DIR, help="output folder of hybrid_model.py")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    events = pd.read_csv(DATA_DIR / "bioreaction_remaining.csv", dtype={"batch_id": str})
    template = pd.read_csv(DATA_DIR / "bioreaction_template.csv", dtype={"batch_id": str})
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    F = feed_features(events)
    sens_batch, sens_day, sample = sensor_features(events)
    print(f"stage 1: {args.stage1}; pooled sample volume {sample:.4f} mL; base added mean "
          f"{sens_batch.base_tot.mean():.3f} mL (sd {sens_batch.base_tot.std():.3f}); "
          f"batches with early glucose bolus: {(sens_batch.A_early > 0).sum()}")

    rows = build_rows(F, sens_batch, sens_day, load_stage1(args.stage1))
    needed = sorted(set(CELL_COLS + S_COLS + L_COLS + sum(DD_COLS.values(), []) + ["A", "fs_group"]))
    assert rows[needed].notna().all().all(), "feature merge failed"
    fold_of = {}
    for f, (_, te) in enumerate(GroupKFold(N_FOLDS).split(np.arange(len(train_ids)), groups=np.arange(len(train_ids)))):
        for i in te:
            fold_of[train_ids[i]] = f
    rows["fold"] = rows.batch_id.map(fold_of)

    oof = rows.copy()
    parts = {k: pd.DataFrame(index=rows.index, columns=TARGETS, dtype=float) for k in ("hyb", "dd", "fin")}
    for f in range(N_FOLDS):
        tr, te = (rows.fold != f).to_numpy(), (rows.fold == f).to_numpy()
        h = HybridBranch().fit(rows[tr]).predict(rows[te])
        d = DataDrivenBranch().fit(rows[tr]).predict(rows[te])
        parts["hyb"].loc[te] = h.to_numpy()
        parts["dd"].loc[te] = d.to_numpy()
        parts["fin"].loc[te] = finalize(h, d).to_numpy()
    names = {"hyb": "v3 hybrid branch", "dd": "v3 data-driven branch", "fin": "v3 final (50/50 blend)"}
    for v in TARGETS:
        oof["stage 1 hybrid|" + v] = oof["s1_" + v]
        for k, name in names.items():
            oof[f"{name}|{v}"] = parts[k][v].astype(float)
    labels = ["stage 1 hybrid", *names.values()]
    table = pd.DataFrame({lab: mape_table(oof, {v: f"{lab}|{v}" for v in TARGETS}) for lab in labels}).T.round(2)
    table.to_csv(out_dir / "leaderboard_style_score.csv")
    pd.set_option("display.width", 220)
    print("Held-out (7-fold grouped CV) leaderboard-style score, MAPE %:")
    print(table.to_string())
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)

    long = []
    for v in TARGETS:
        for lab in ["stage 1 hybrid", names["fin"]]:
            long.append(pd.DataFrame({"variable": v, "model": lab, "day": oof.day,
                                      "ape": 100 * np.abs(oof[f"{lab}|{v}"] / oof["obs_" + v] - 1)}))
    long = pd.concat(long)
    by_day = long.pivot_table(index=["variable", "model"], columns="day", values="ape", aggfunc="mean").round(1)
    by_day.to_csv(out_dir / "mape_by_day.csv")
    print("\nHeld-out MAPE by day:")
    print(by_day.to_string())
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), constrained_layout=True)
    for ax, v in zip(axes, TARGETS):
        for lab, color in [("stage 1 hybrid", "#9D9D9D"), (names["fin"], "#E45756")]:
            s = by_day.loc[(v, lab)]
            ax.plot(s.index, s.values, marker="o", color=color,
                    label=f"{lab} ({long[(long.variable == v) & (long.model == lab)].ape.mean():.1f}%)")
        ax.set_title(v)
        ax.set_xlabel("Day")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Held-out MAPE [%]")
    fig.suptitle("Held-out MAPE by day: hybrid vs v3 (feed, bolus and volume-balance soft sensors, stacked + blended)")
    fig.savefig(out_dir / "01_mape_by_day.png", dpi=140)
    plt.close(fig)

    # Final models on all 49 training batches -> holdout predictions
    hyb, dd = HybridBranch().fit(rows), DataDrivenBranch().fit(rows)
    h1 = pd.read_csv(args.stage1 / "holdout_predictions.csv", dtype={"batch_id": str})
    h1["t_key"] = h1.elapsed_time.round(6)
    hrows = build_rows(F, sens_batch, sens_day, h1.rename(columns={v: "s1_" + v for v in TARGETS}))
    assert len(hrows) == len(template) and hrows[needed].notna().all().all()
    final = finalize(hyb.predict(hrows), dd.predict(hrows))
    sub = template[["batch_id", "elapsed_time"]].copy()
    for v in TARGETS:
        sub[v] = final[v].to_numpy()
    assert (sub.batch_id.to_numpy() == hrows.batch_id.to_numpy()).all()
    sub.to_csv(out_dir / "holdout_predictions.csv", index=False)
    print(f"\nWrote {len(sub)} holdout predictions ({sub[TARGETS].isna().any(axis=1).sum()} missing) to {out_dir}")


if __name__ == "__main__":
    main()
