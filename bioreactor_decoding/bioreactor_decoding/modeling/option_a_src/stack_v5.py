"""v5 stack: holdout predictions for the best model from experiments_v5.

Members (each fitted on the 49 training batches):
  * v3 (hybrid branch + data-driven branch) built on 5 stage-1 cores:
      states (hybrid ODE), glc (hybrid ODE with glucose correction),
      mech (pure Sartorius-style ODE), monod (two-substrate Monod core),
      msens (sensor-informed Monod core)
  * functional-PCA trajectory member on each core (K=5, ridge 10)
  * raw functional-PCA member (no stage-1 offset)
Blend: geometric, non-negative weights per variable and per segment
(days 1-6 / 7-13), fitted on the out-of-fold training predictions. Then TCD >=
VCD and glucose clipped into the interval implied by the bolus controller.

The states_feeds core from the 7.364 CV stack is left out: its stage-1 holdout
predictions were never saved. The nested CV score of exactly this member set is
printed and saved.

    python src/stack_v5_stage1_holdout.py monod monod_sens mech
    python src/stack_v5.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_v5 as x  # noqa: E402
import experiments_v5_ideas as ideas  # noqa: E402
import hybrid_v3 as h3  # noqa: E402

ROOT = h3.ROOT
OUT = ROOT / "outputs" / "stack_v5"
TARGETS = h3.TARGETS
CORES = {  # name: (stage-1 OOF dir, stage-1 model, stage-1 holdout file, members_oof prefix)
    "states": ("hybrid", "states", ROOT / "outputs/hybrid/holdout_predictions.csv", "states"),
    "glc": ("hybrid_glc", "states", ROOT / "outputs/hybrid_glc/holdout_predictions.csv", "glc"),
    "mech": ("hybrid", "mechanistic", OUT / "stage1_mech/holdout_predictions.csv", "mech"),
    "monod": ("core_monod", "monod", OUT / "stage1_monod/holdout_predictions.csv", "monod"),
    "monod_sens": ("core_monod_sens", "monod_sens", OUT / "stage1_monod_sens/holdout_predictions.csv", "msens"),
}
K, ALPHA = 5, 10.0


def load_rows(core):
    d, m, _, _ = CORES[core]
    saved = (h3.STAGE1_DIR, h3.STAGE1_MODEL)
    h3.STAGE1_DIR, h3.STAGE1_MODEL = ROOT / "outputs" / d, m
    try:
        rows, events = x.build()
    finally:
        h3.STAGE1_DIR, h3.STAGE1_MODEL = saved
    return rows, events


def holdout_rows(core, events, template):
    F = h3.feed_features(events)
    sb, sd, _ = h3.sensor_features(events)
    h1 = pd.read_csv(CORES[core][2], dtype={"batch_id": str})
    h1["t_key"] = h1.elapsed_time.round(6)
    hr = h3.build_rows(F, sb, sd, h1.rename(columns={v: "s1_" + v for v in TARGETS}))
    assert len(hr) == len(template) and (hr.batch_id.to_numpy() == template.batch_id.to_numpy()).all()
    hr["day"] = np.rint(hr.t_key).astype(int)
    return x.add_censor(hr)


def batch_X(rows, ids):
    feats = rows.groupby("batch_id").agg(**{c: (c, "first") for c in ideas.BATCH_FEATS},
                                         **{f"m_{c}": (c, "mean") for c in ["fb_next_dev", "fc_next_dev", "V_dev"]})
    feats = feats.loc[ids]
    S = {v: rows.pivot_table(index="batch_id", columns="day", values="ls1_" + v).loc[ids].to_numpy() for v in TARGETS}
    return np.c_[feats.to_numpy(float), np.hstack([S[v] for v in TARGETS])], S


def fpca_full(rows, hrows, oof_unscaled, use_offset=True):
    """FPCA member fitted on all training batches -> holdout; per-day MAPE-optimal
    scale taken from the member's own out-of-fold training predictions."""
    tr_ids = list(dict.fromkeys(rows.batch_id))
    ho_ids = list(dict.fromkeys(hrows.batch_id))
    Xtr, Str = batch_X(rows, tr_ids)
    Xho, Sho = batch_X(hrows, ho_ids)
    out = {}
    for v in TARGETS:
        Y = np.log(rows.pivot_table(index="batch_id", columns="day", values="obs_" + v).loc[tr_ids].to_numpy())
        off_tr = Str[v] if use_offset else 0 * Y
        off_ho = Sho[v] if use_offset else 0 * Sho[v]
        R = Y - off_tr
        mu = R.mean(0)
        _, _, Vt = np.linalg.svd(R - mu, full_matrices=False)
        sc = (R - mu) @ Vt[:K].T
        fn = [x.ridge_fit(Xtr, sc[:, k], ALPHA) for k in range(K)]
        P = np.exp(off_ho + mu + np.column_stack([q(Xho) for q in fn]) @ Vt[:K])
        pred = ideas._unstack(hrows, P, pd.Index(ho_ids))
        scale = np.array([h3.mape_optimal_scale(oof_unscaled[v][(rows.day == d).to_numpy()],
                                                rows["obs_" + v].to_numpy()[(rows.day == d).to_numpy()])
                          for d in hrows.day])
        out[v] = pred * scale
    out["tcd"] = np.maximum(out["tcd"], out["vcd"])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    template = pd.read_csv(h3.DATA_DIR / "bioreaction_template.csv", dtype={"batch_id": str})
    x.OUT = OUT / "cv"  # cached out-of-fold predictions per core
    x.OUT.mkdir(parents=True, exist_ok=True)
    mem = None
    hold = {}
    hr_states = None
    for core, (_, _, _, pref) in CORES.items():
        rows, events = load_rows(core)
        ideas.CORES.setdefault(core, CORES[core][:2])
        _, parts = ideas.core_parts(core, events)
        assert (parts.batch_id.to_numpy() == rows.batch_id.to_numpy()).all()
        if mem is None:
            mem = parts[["batch_id", "t_key", "day", "fold"] + ["obs_" + v for v in TARGETS]].copy()
        for v in TARGETS:
            mem[f"{pref}|{v}"] = parts[f"fin|{v}"].to_numpy()
            mem[f"{pref}_fp|{v}"] = parts[f"fps|{v}"].to_numpy()
        hr = holdout_rows(core, events, template)
        # v3 on this core, fitted on all training rows
        fin = h3.finalize(h3.HybridBranch().fit(rows).predict(hr), h3.DataDrivenBranch().fit(rows).predict(hr))
        for v in TARGETS:
            hold[f"{pref}|{v}"] = fin[v].to_numpy()
        # FPCA member (unscaled OOF from the same CV, for the full-data scale)
        fo = ideas.fpca_member(rows, parts, K=K, alpha=ALPHA)
        fp = fpca_full(rows, hr, {v: fo[f"fp|{v}"].to_numpy() for v in TARGETS})
        for v in TARGETS:
            hold[f"{pref}_fp|{v}"] = fp[v]
        if core == "states":
            ro = ideas.fpca_bagged(rows, parts, K=K, alpha=ALPHA, n_bags=1, use_s1_offset=False)
            raw = fpca_full(rows, hr, {v: ro[f"fb|{v}"].to_numpy() for v in TARGETS}, use_offset=False)
            for v in TARGETS:
                hold[f"raw|{v}"] = raw[v]
                mem[f"raw|{v}"] = ro[f"fbs|{v}"].to_numpy()
            hr_states = hr
        print(f"core {core}: done", flush=True)
    hold = pd.DataFrame(hold)
    mem.to_csv(x.OUT / "members_oof.csv", index=False)

    srcs = ["states", "glc", "mech", "monod", "msens", "states_fp", "glc_fp", "mech_fp", "monod_fp", "msens_fp", "raw"]
    # honest CV score of exactly this member set (weights fitted on the other folds)
    rows_s, _ = load_rows("states")
    o = ideas.i18_segment_stack(rows_s, None, None, srcs=srcs, tag="stack submitted (no states_feeds)")

    # weights on all training OOF predictions
    seg_tr = np.where(mem.day.to_numpy() <= 6, 0, 1)
    seg_ho = np.where(hr_states.day.to_numpy() <= 6, 0, 1)
    final = {}
    wtab = {}
    for v in TARGETS:
        P = np.column_stack([mem[f"{s}|{v}"].to_numpy() for s in srcs])
        H = np.column_stack([hold[f"{s}|{v}"].to_numpy() for s in srcs])
        y = mem["obs_" + v].to_numpy()
        out = np.zeros(len(H))
        for sg in (0, 1):
            w = ideas._nested_weights(P[seg_tr == sg], y[seg_tr == sg], None)
            wtab[(v, "days 1-6" if sg == 0 else "days 7-13")] = w
            out[seg_ho == sg] = np.exp(np.log(H[seg_ho == sg]) @ w)
        final[v] = out
    final["tcd"] = np.maximum(final["tcd"], final["vcd"])
    lo, hi = hr_states.g_lo.to_numpy(), hr_states.g_hi.to_numpy()
    final["substrate"] = np.clip(final["substrate"], lo, np.minimum(hi, 1e9) - 1e-6)

    sub = template[["batch_id", "elapsed_time"]].copy()
    for v in TARGETS:
        sub[v] = final[v]
    assert sub[TARGETS].notna().all().all() and (sub[TARGETS] > 0).all().all()
    sub.to_csv(OUT / "holdout_predictions.csv", index=False)
    pd.DataFrame(wtab, index=srcs).T.round(3).to_csv(OUT / "stack_weights.csv")
    hold.assign(batch_id=template.batch_id.to_numpy(), elapsed_time=template.elapsed_time.to_numpy()).to_csv(
        OUT / "holdout_members.csv", index=False)
    v3 = pd.read_csv(ROOT / "outputs/hybrid_v3/holdout_predictions.csv", dtype={"batch_id": str})
    ratio = sub[TARGETS].to_numpy() / v3[TARGETS].to_numpy()
    print("stack / v3 holdout ratio: median", np.median(ratio, 0).round(3), "p5", np.percentile(ratio, 5, 0).round(3),
          "p95", np.percentile(ratio, 95, 0).round(3))
    print(pd.DataFrame(wtab, index=srcs).T.round(2).to_string())
    print("wrote", OUT / "holdout_predictions.csv")


if __name__ == "__main__":
    main()
