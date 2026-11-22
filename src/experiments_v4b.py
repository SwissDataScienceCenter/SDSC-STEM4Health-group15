"""v4 experiments: same 7-fold grouped CV as hybrid_v3, with pluggable extras.

Imports hybrid_v3 (module-level column lists are looked up at call time, so
patching them changes what the branches see) and adds candidate features that
are legal on holdout batches (t=0 assay, full feed/volume log, stage-1 curve).

New features tested:
  dev_{v}_prev / dev_{v} / dev_{v}_next : stage-1 predicted log level of a
      variable at this batch's day minus the day's mean over all batches —
      i.e. the batch's predicted rank around days d-1, d, d+1.
  g_prev / g_next : hours from this sample to the previous / next substrate
      bolus (capped 2 d). Sample-to-bolus timing is logged for every batch.
  t0 assays (via0, v0, l0) added to the substrate / byproduct corrections.

    python src/experiments_v4b.py [config ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_v3 as h3  # noqa: E402

ROOT = h3.ROOT
TARGETS = h3.TARGETS


# --------------------------------------------------------------------------


def add_features(rows: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    # predicted rank deviations around neighbouring days
    for v in TARGETS:
        piv = rows.pivot_table(index="batch_id", columns="day", values="ls1_" + v)
        dev = piv - piv.mean()
        for lag, nm in [(-1, "prev"), (0, "cur"), (1, "next")]:
            m = dev.shift(-lag).stack().rename(f"dev_{v}_{nm}")
            rows = rows.merge(m.reset_index().rename(columns={"level_1": "day"}), on=["batch_id", "day"], how="left")
    # distance to previous / next substrate bolus
    gaps = []
    for bid, g in events.groupby("batch_id"):
        bol = np.sort(g[g.feed_substrate.notna()].elapsed_time.to_numpy())
        st = g[g.volume.notna()].sort_values("elapsed_time")
        for t in st.elapsed_time:
            nxt = bol[bol > t]
            prv = bol[bol < t]
            gaps.append({"batch_id": bid, "t_key": round(t, 6),
                         "g_next": min(nxt[0] - t, 2.0) if len(nxt) else 2.0,
                         "g_prev": min(t - prv[-1], 2.0) if len(prv) else 2.0})
    rows = rows.merge(pd.DataFrame(gaps), on=["batch_id", "t_key"], how="left")
    rows[[c for c in rows.columns if c.startswith("dev_")]] = (
        rows[[c for c in rows.columns if c.startswith("dev_")]].fillna(0.0))
    return rows


# --------------------------------------------------------------------------


BASE = dict(
    CELL=h3.CELL_COLS, DD=dict(h3.DD_COLS), S=h3.S_COLS, L=h3.L_COLS,
    ridge_alpha=5.0, smooth_alpha=5.0,
)


def variant(name, cell=None, dd=None, s=None, l=None, ridge_alpha=None, smooth_alpha=None):
    c = dict(BASE)
    c.update({k: v for k, v in dict(cell=cell, s=s, l=l, ridge_alpha=ridge_alpha,
                                     smooth_alpha=smooth_alpha).items() if v is not None})
    if dd is not None:
        c["DD"] = {**h3.DD_COLS, **dd}
    c["name"] = name
    return c


NEW_SUB = ["dev_substrate_prev", "dev_substrate_cur", "dev_substrate_next", "g_prev", "g_next", "via0", "v0"]
NEW_LAC = ["dev_byproduct_prev", "dev_byproduct_cur", "dev_byproduct_next", "via0", "v0", "l0"]
NEW_CELL = ["dev_vcd_prev", "dev_vcd_next", "dev_tcd_prev", "dev_tcd_next"]

CONFIGS = {
    "base": BASE,
    "f_sub": variant("f_sub", s=h3.S_COLS + NEW_SUB, ridge_alpha=10.0),
    "f_lac": variant("f_lac", l=h3.L_COLS + NEW_LAC, smooth_alpha=10.0),
    "f_cells": variant("f_cells", cell=h3.CELL_COLS + NEW_CELL),
    "f_sub_dd": variant("f_sub_dd", dd={"substrate": h3.DD_COLS["substrate"] + NEW_SUB}),
    "f_lac_dd": variant("f_lac_dd", dd={"byproduct": h3.DD_COLS["byproduct"] + NEW_LAC}),
    "f_cells_dd": variant("f_cells_dd", dd={"vcd": h3.DD_COLS["vcd"] + NEW_CELL,
                                            "tcd": h3.DD_COLS["tcd"] + NEW_CELL}),
}


# --------------------------------------------------------------------------
# Loss-function variants: MAPE's optimum is a weighted median of obs (weight
# 1/obs), both v3 branches currently fit squared error on log(obs).
# --------------------------------------------------------------------------

_GBM = h3.HistGradientBoostingRegressor
_ORIG_HYB, _ORIG_DD = h3.HybridBranch, h3.DataDrivenBranch


class DDl1(h3.DataDrivenBranch):
    """Data-driven branch with absolute-error loss, optionally 1/obs weighted."""

    def __init__(self, n_bags=10, seed=0, loss="absolute_error", weight=False):
        super().__init__(n_bags, seed)
        self.loss, self.weight = loss, weight

    def fit(self, train):
        rng = np.random.default_rng(self.seed)
        ids = train.batch_id.unique()
        by_batch = {b: g for b, g in train.groupby("batch_id")}
        self.models = []
        for _ in range(self.n_bags):
            boot = pd.concat([by_batch[b] for b in rng.choice(ids, len(ids), replace=True)], ignore_index=True)
            ms = {}
            for v in TARGETS:
                y = np.log(boot["obs_" + v].to_numpy())
                sw = (1.0 / boot["obs_" + v].to_numpy()) if self.weight else None
                m = _GBM(max_depth=3, learning_rate=0.05, max_iter=400, min_samples_leaf=15,
                         l2_regularization=1.0, loss=self.loss, random_state=0)
                m.fit(boot[h3.DD_COLS[v]].to_numpy(), y, sample_weight=sw)
                ms[v] = m
            self.models.append(ms)
        return self


class HYBl1(h3.HybridBranch):
    """Cell correction fitted with absolute-error loss."""

    def fit(self, train):
        lr = {v: np.log(train["obs_" + v] / train["s1_" + v]).to_numpy() for v in TARGETS}
        self.cell = {v: _GBM(max_depth=2, learning_rate=0.03, max_iter=200, min_samples_leaf=30,
                             l2_regularization=1.0, loss="absolute_error", random_state=0).fit(
                             train[h3.CELL_COLS + ["ls1_" + v]].to_numpy(), lr[v]) for v in ("vcd", "tcd")}
        self.soft = h3.SoftSensor().fit(train, lr["substrate"])
        self.ridge_S = h3.DayRidge(h3.S_COLS, days=set(range(3, 14))).fit(train, lr["substrate"] - self.soft.predict(train))
        self.ridge_L = h3.DaySmooth(h3.L_COLS).fit(train, lr["byproduct"])
        raw = self._raw(train)
        self.scale = {}
        for v in TARGETS:
            for day in np.unique(train.day):
                m = (train.day == day).to_numpy()
                self.scale[(v, day)] = h3.mape_optimal_scale(raw[v].to_numpy()[m], train["obs_" + v].to_numpy()[m])
        return self


CONFIGS.update({
    "dd_l1": {**BASE, "name": "dd_l1", "dd_cls": lambda: DDl1(loss="absolute_error")},
    "dd_w": {**BASE, "name": "dd_w", "dd_cls": lambda: DDl1(loss="squared_error", weight=True)},
    "dd_l1_w": {**BASE, "name": "dd_l1_w", "dd_cls": lambda: DDl1(loss="absolute_error", weight=True)},
    "hyb_l1": {**BASE, "name": "hyb_l1", "hyb_cls": HYBl1},
})


_RIDGE, _SMOOTH = h3.DayRidge, h3.DaySmooth


def run_cv(cfg: dict, rows: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    h3.CELL_COLS = cfg["CELL"]
    h3.DD_COLS = cfg["DD"]
    h3.S_COLS = cfg["S"]
    ra, sa = cfg["ridge_alpha"], cfg["smooth_alpha"]
    h3.DayRidge = lambda cols, days=None, alpha=5.0: _RIDGE(cols, days, ra)
    h3.DaySmooth = lambda cols, alpha=5.0, degree=2: _SMOOTH(cols, sa, degree)
    h3.HybridBranch = cfg.get("hyb_cls", _ORIG_HYB)
    dd_cls = cfg.get("dd_cls", _ORIG_DD)
    h3.DataDrivenBranch = lambda: dd_cls()
    needed = sorted(set(cfg["CELL"] + cfg["S"] + cfg["L"] + sum(cfg["DD"].values(), []) + ["A", "fs_group"]))
    assert rows[needed].notna().all().all(), f"missing features for {cfg['name']}"
    parts = {k: pd.DataFrame(index=rows.index, columns=TARGETS, dtype=float) for k in ("hyb", "dd", "fin")}
    for f in range(h3.N_FOLDS):
        tr, te = (rows.fold != f).to_numpy(), (rows.fold == f).to_numpy()
        h = h3.HybridBranch().fit(rows[tr]).predict(rows[te])
        d = h3.DataDrivenBranch().fit(rows[tr]).predict(rows[te])
        parts["hyb"].loc[te] = h.to_numpy()
        parts["dd"].loc[te] = d.to_numpy()
        parts["fin"].loc[te] = h3.finalize(h, d).to_numpy()
    fin = parts["fin"].add_prefix("fin|")
    obs_cols = ["obs_" + v for v in TARGETS]
    out = pd.concat([rows[["batch_id", "day", "fold"] + obs_cols], fin], axis=1)
    return h3.mape_table(out, {v: f"fin|{v}" for v in TARGETS}), out


def main(argv: list[str]) -> None:
    names = argv or list(CONFIGS)
    events = pd.read_csv(h3.DATA_DIR / "bioreaction_remaining.csv", dtype={"batch_id": str})
    template = pd.read_csv(h3.DATA_DIR / "bioreaction_template.csv", dtype={"batch_id": str})
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    F = h3.feed_features(events)
    sens_batch, sens_day, _ = h3.sensor_features(events)
    rows = add_features(h3.build_rows(F, sens_batch, sens_day, h3.load_stage1(h3.STAGE1_DIR)), events)
    fold_of = {}
    for f, (_, te) in enumerate(GroupKFold(h3.N_FOLDS).split(np.arange(len(train_ids)), groups=np.arange(len(train_ids)))):
        for i in te:
            fold_of[train_ids[i]] = f
    rows["fold"] = rows.batch_id.map(fold_of)

    results, oofs = {}, {}
    for n in names:
        sc, oof = run_cv(CONFIGS[n], rows)
        results[n] = sc
        oofs[n] = oof
        print(f"{n:12s} " + "  ".join(f"{v}={sc[v]:.2f}" for v in TARGETS) + f"  SCORE={sc['leaderboard_score']:.2f}", flush=True)

    # ensemble with the saved glc stage-1 variant, if all runs done
    if "base" in names:
        glc_path = ROOT / "outputs" / "hybrid_v3_glc" / "oof_predictions.csv"
        if glc_path.exists():
            glc = pd.read_csv(glc_path, dtype={"batch_id": str})
            key = ["batch_id", "t_key"]
            g = glc[key + [f"v3 final (50/50 blend)|{v}" for v in TARGETS]].rename(
                columns={f"v3 final (50/50 blend)|{v}": "glc|" + v for v in TARGETS})
            m = rows[key].merge(g, on=key)
            for w, tag in [(0.5, "ens-glc50"), (0.25, "ens-glc25")]:
                o = rows[["batch_id", "day", "fold"] + ["obs_" + v for v in TARGETS]].copy()
                for v in TARGETS:
                    o[v] = np.exp((1 - w) * np.log(oofs["base"][f"fin|{v}"]) + w * np.log(m["glc|" + v]))
                o["tcd"] = np.maximum(o["tcd"], o["vcd"])
                results[tag] = h3.mape_table(o, {v: v for v in TARGETS})
                sc = results[tag]
                print(f"{tag:12s} " + "  ".join(f"{v}={sc[v]:.2f}" for v in TARGETS) + f"  SCORE={sc['leaderboard_score']:.2f}")

    table = pd.DataFrame(results).T.round(3)
    out = ROOT / "outputs" / "experiments_v4"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "scores_v4b.csv")
    print(table.to_string())


if __name__ == "__main__":
    main(sys.argv[1:])
