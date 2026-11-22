"""Option A: one mechanistic core, two ML views (simplified hybrid_v7).

Core: the sensor-conditioned particle Monod core (src/core_assim.py,
outputs/core_assim): global Monod fit + per-batch multipliers on mu_max,
lactate yield and glucose yield, drawn as 400 particles and weighted by the
bolus-rule glucose intervals and the base volume added by days 3-5.

Two members, both on that core:
  as     hybrid_v3 recipe (hybrid branch + data-driven branch, 30 bags, 50/50)
  as_fp  functional-PCA trajectory model (K=5, ridge 10)

Blend: geometric, non-negative weights per variable and per day segment
(1-6 / 7-13), fitted on out-of-fold training predictions.
Finish: glucose = MAPE-optimal point inside the bolus-rule interval; TCD >= VCD.
Dropped vs hybrid_v7: Sartorius+NN core, two-population core, 5-multiplier
core with posterior features, TCD ratio model.

    python src/core_assim.py     # only if outputs/core_assim is missing
    python src/option_a.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_v3 as h3  # noqa: E402

if not (h3.DATA_DIR / "bioreaction_remaining.csv").exists() and (h3.ROOT / "data" / "bioreaction_remaining.csv").exists():
    h3.DATA_DIR = h3.ROOT / "data"  # local copy of the challenge data

import experiments_v5 as x  # noqa: E402
import experiments_v5_ideas as ideas  # noqa: E402
import experiments_v6 as e  # noqa: E402
import stack_v5 as s5  # noqa: E402

ROOT = h3.ROOT
OUT = ROOT / "outputs" / "option_a"
TARGETS = h3.TARGETS
MEMBERS = ["as", "as_fp"]
CORE, CORE_DIR, CORE_MODEL = "assim", "core_assim", "assim"
DD_BAGS = 30  # matches the out-of-fold 'as' member in members_oof_v7c.csv


def blend_weights(L, y, day):
    W = {}
    for seg in e.SEG2:
        m = np.isin(day, list(seg))
        W[seg] = e.weights_simplex(L[m], y[m])
    return W


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x.OUT = e.CV
    s5.CORES = {CORE: (CORE_DIR, CORE_MODEL, ROOT / f"outputs/{CORE_DIR}/holdout_predictions.csv", "as")}
    template = pd.read_csv(h3.DATA_DIR / "bioreaction_template.csv", dtype={"batch_id": str})
    mem = pd.read_csv(e.CV / "members_oof_v7c.csv", dtype={"batch_id": str})

    # members on the 12 holdout batches (fitted on all 49 training batches)
    rows, events = s5.load_rows(CORE)
    ideas.CORES.setdefault(CORE, (CORE_DIR, CORE_MODEL))
    _, parts = ideas.core_parts(CORE, events)
    hr = s5.holdout_rows(CORE, events, template)
    fin = h3.finalize(h3.HybridBranch().fit(rows).predict(hr), h3.DataDrivenBranch(n_bags=DD_BAGS).fit(rows).predict(hr))
    fo = ideas.fpca_member(rows, parts, K=s5.K, alpha=s5.ALPHA)
    fp = s5.fpca_full(rows, hr, {v: fo[f"fp|{v}"].to_numpy() for v in TARGETS})
    hold = pd.DataFrame({**{f"as|{v}": fin[v].to_numpy() for v in TARGETS}, **{f"as_fp|{v}": fp[v] for v in TARGETS}})
    assert (rows.batch_id.to_numpy() == mem.batch_id.to_numpy()).all()

    # honest CV of exactly this configuration (weights nested inside the folds)
    o_cv = e.stack(mem, MEMBERS, rows=rows, clip=False)
    c_cv = x.idea_censor(rows, o_cv, src="st", dst="f", days_pool=13)
    c_cv["f|tcd"] = np.maximum(c_cv["f|tcd"], c_cv["f|vcd"])
    sc = x.score(c_cv, "f")
    print("Option A nested CV:", sc.round(3).to_dict())

    # weights on all training OOF predictions -> holdout blend
    day_tr, day_ho = mem.day.to_numpy(), hr.day.to_numpy()
    st, wtab = {}, {}
    for v in TARGETS:
        L = np.column_stack([np.log(mem[f"{s}|{v}"].to_numpy()) for s in MEMBERS])
        H = np.column_stack([np.log(hold[f"{s}|{v}"].to_numpy()) for s in MEMBERS])
        out = np.zeros(len(H))
        for seg, w in blend_weights(L, mem["obs_" + v].to_numpy(), day_tr).items():
            wtab[(v, f"days {seg.start}-{seg.stop - 1}")] = w
            mh = np.isin(day_ho, list(seg))
            out[mh] = np.exp(H[mh] @ w)
        st[v] = out
    st["tcd"] = np.maximum(st["tcd"], st["vcd"])

    # glucose: MAPE-optimal point inside the bolus-rule interval
    r_pool = np.log(o_cv.obs_substrate.to_numpy() / o_cv["st|substrate"].to_numpy())
    lo, hi = hr.g_lo.to_numpy(), hr.g_hi.to_numpy()
    for i in np.where((lo > 0) | np.isfinite(hi))[0]:
        st["substrate"][i] = x.censored_point(st["substrate"][i], lo[i], hi[i], r_pool)

    sub = template[["batch_id", "elapsed_time"]].copy()
    for v in TARGETS:
        sub[v] = st[v]
    assert sub[TARGETS].notna().all().all() and (sub[TARGETS] > 0).all().all()
    sub.to_csv(OUT / "holdout_predictions.csv", index=False)
    sub.to_csv(OUT / "leaderboard_submission.csv", index=False)
    hold.insert(0, "elapsed_time", template.elapsed_time.to_numpy())
    hold.insert(0, "batch_id", template.batch_id.to_numpy())
    hold.to_csv(OUT / "holdout_members.csv", index=False)
    c_cv.to_csv(OUT / "oof_predictions.csv", index=False)
    pd.DataFrame(wtab, index=MEMBERS).T.round(3).to_csv(OUT / "blend_weights.csv")
    pd.DataFrame([sc]).round(3).to_csv(OUT / "cv_score.csv", index=False)
    print(pd.DataFrame(wtab, index=MEMBERS).T.round(2).to_string())
    for ref in ("hybrid_v3", "hybrid_v7"):
        prev = pd.read_csv(ROOT / f"outputs/{ref}/holdout_predictions.csv", dtype={"batch_id": str})
        ratio = sub[TARGETS].to_numpy() / prev[TARGETS].to_numpy()
        print(f"option_a / {ref} holdout ratio: median", np.median(ratio, 0).round(3),
              "p5", np.percentile(ratio, 5, 0).round(3), "p95", np.percentile(ratio, 95, 0).round(3))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
