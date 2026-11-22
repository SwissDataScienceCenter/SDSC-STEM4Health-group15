"""Hybrid v7: hybrid_v3 improved with sensor assimilation, blended from 4 models.

Final prediction = geometric blend of exactly 4 models (weights per variable and
per day segment 1-6 / 7-13, fitted on out-of-fold training predictions):

  1. v3 on the sensor-conditioned Monod core (outputs/a2_p5ic, src/core_assim2.py):
     particles over per-batch growth, yield, death and lactate-consumption
     multipliers plus a t=0 cell-count factor, weighted by the bolus-rule glucose
     intervals and the base volume. Its posterior parameters are extra features of
     the data-driven branch.
  2. Functional-PCA trajectory model on the assimilation core (outputs/a2_p3ic).
  3. hybrid_v3 (hybrid ODE core, outputs/hybrid), data-driven branch with 30 bags.
  4. v3 on the two-population core (outputs/core_twopop, src/core_twopop.py), 30 bags.

Finish (uses only these 4 models): glucose = MAPE-optimal point inside the
interval implied by the bolus controller; TCD = VCD x per-day ratio model whose
inputs are the 4 models' own TCD/VCD ratios.

Cross-validated (7-fold, whole batches, weights nested): 7.187
(VCD 6.77, TCD 6.71, glucose 7.17, lactate 8.10); hybrid_v3 alone: 7.674.

    python src/core_runner.py --core twopop --name core_twopop
    python src/core_assim2.py --name a2_p5ic --params 0,1,2,3,4 --ic
    python src/core_assim2.py --name a2_p3ic --params 0,1,2 --ic
    python src/hybrid_v7.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_v5 as x  # noqa: E402
import experiments_v5_ideas as ideas  # noqa: E402
import experiments_v6 as e  # noqa: E402
import experiments_v7_ideas as iv7  # noqa: E402
import hybrid_v3 as h3  # noqa: E402
import stack_v5 as s5  # noqa: E402

ROOT = h3.ROOT
OUT = ROOT / "outputs" / "hybrid_v7"
TARGETS = h3.TARGETS
MEMBERS = ["pf3a", "a3i_fp", "states", "tp"]
CORE_DIR = {"pf3a": ("a2_p5ic", "a2_p5ic"), "a3i_fp": ("a2_p3ic", "a2_p3ic"),
            "states": ("hybrid", "states"), "tp": ("core_twopop", "core_twopop")}
HOLD = {"a2_p5ic": ROOT / "outputs/a2_p5ic/holdout_predictions.csv",
        "a2_p3ic": ROOT / "outputs/a2_p3ic/holdout_predictions.csv",
        "hybrid": ROOT / "outputs/hybrid/holdout_predictions.csv",
        "core_twopop": ROOT / "outputs/core_twopop/holdout_predictions.csv"}


def holdout_rows(d, events, template):
    F = h3.feed_features(events)
    sb, sd, _ = h3.sensor_features(events)
    h1 = pd.read_csv(HOLD[d], dtype={"batch_id": str})
    h1["t_key"] = h1.elapsed_time.round(6)
    hr = h3.build_rows(F, sb, sd, h1.rename(columns={v: "s1_" + v for v in TARGETS}))
    assert len(hr) == len(template) and (hr.batch_id.to_numpy() == template.batch_id.to_numpy()).all()
    hr["day"] = np.rint(hr.t_key).astype(int)
    return x.add_censor(hr)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    x.OUT = e.CV
    template = pd.read_csv(h3.DATA_DIR / "bioreaction_template.csv", dtype={"batch_id": str})
    events = pd.read_csv(h3.DATA_DIR / "bioreaction_remaining.csv", dtype={"batch_id": str})
    mem = pd.read_csv(e.CV / "members_oof_v7c.csv", dtype={"batch_id": str})
    hold = {}
    rows_states = hrows_states = None
    for name in MEMBERS:
        d, model = CORE_DIR[name]
        rows = iv7.rows_for(d, d, model)
        hr = holdout_rows(d, events, template)
        if name == "pf3a":  # v3 with posterior parameters in the data-driven branch
            rows, pc = iv7.with_post(rows, d)
            hr, _ = iv7.with_post(hr, d)
            saved = dict(h3.DD_COLS)
            h3.DD_COLS = {v: saved[v] + pc for v in TARGETS}
            try:
                fin = h3.finalize(h3.HybridBranch().fit(rows).predict(hr), h3.DataDrivenBranch().fit(rows).predict(hr))
            finally:
                h3.DD_COLS = saved
        elif name == "a3i_fp":  # functional-PCA trajectory model
            ideas.CORES.setdefault(d, (d, model))
            _, parts = ideas.core_parts(d, events)
            fo = ideas.fpca_member(rows, parts, K=s5.K, alpha=s5.ALPHA)
            fin = s5.fpca_full(rows, hr, {v: fo[f"fp|{v}"].to_numpy() for v in TARGETS})
        else:  # plain v3; its out-of-fold predictions (members_oof_v7c) use a 30-bag data-driven branch
            fin = h3.finalize(h3.HybridBranch().fit(rows).predict(hr), h3.DataDrivenBranch(n_bags=30).fit(rows).predict(hr))
        for v in TARGETS:
            hold[f"{name}|{v}"] = np.asarray(fin[v])
        if name == "states":
            rows_states, hrows_states = rows, hr
        print(f"model {name}: done", flush=True)
    hold = pd.DataFrame(hold)

    # honest CV of exactly this configuration
    c_cv, sc = iv7.blend4(mem, rows_states, MEMBERS, "hybrid_v7 final (4 models)")
    o_cv = e.stack(mem, MEMBERS, rows=rows_states, clip=False)

    # weights on all training OOF predictions
    day_tr, day_ho = mem.day.to_numpy(), hrows_states.day.to_numpy()
    st, wtab = {}, {}
    for v in TARGETS:
        L = np.column_stack([np.log(mem[f"{s}|{v}"].to_numpy()) for s in MEMBERS])
        H = np.column_stack([np.log(hold[f"{s}|{v}"].to_numpy()) for s in MEMBERS])
        y = mem["obs_" + v].to_numpy()
        out = np.zeros(len(H))
        for seg in e.SEG2:
            mt, mh = np.isin(day_tr, list(seg)), np.isin(day_ho, list(seg))
            w = e.weights_simplex(L[mt], y[mt])
            wtab[(v, f"days {seg.start}-{seg.stop - 1}")] = w
            out[mh] = np.exp(H[mh] @ w)
        st[v] = out
    st["tcd"] = np.maximum(st["tcd"], st["vcd"])

    # glucose: MAPE-optimal point inside the bolus-rule interval
    r_pool = np.log(o_cv.obs_substrate.to_numpy() / o_cv["st|substrate"].to_numpy())
    lo, hi = hrows_states.g_lo.to_numpy(), hrows_states.g_hi.to_numpy()
    for i in np.where((lo > 0) | np.isfinite(hi))[0]:
        st["substrate"][i] = x.censored_point(st["substrate"][i], lo[i], hi[i], r_pool)

    # TCD = VCD x ratio model (inputs: blend ratio, t=0 / sensor features, the 4 models' ratios)
    def ratio_X(r_blend, rws, mm):
        return np.c_[r_blend, rws[["via0", "ls1_vcd", "base_tot", "A_early"]].to_numpy(float),
                     *[np.log(mm[f"{s}|tcd"] / mm[f"{s}|vcd"]) for s in MEMBERS]]

    Xtr = ratio_X(np.log(o_cv["st|tcd"] / o_cv["st|vcd"]).to_numpy(), rows_states, mem)
    Xho = ratio_X(np.log(st["tcd"] / st["vcd"]), hrows_states, hold)
    ytr = np.log(mem.obs_tcd / mem.obs_vcd).to_numpy()
    pr = np.zeros(len(hold))
    for d in range(1, 14):
        mt, mh = day_tr == d, day_ho == d
        pr[mh] = x.ridge_fit(Xtr[mt], ytr[mt], 5.0)(Xho[mh])
    st["tcd"] = np.maximum(st["vcd"] * np.exp(pr), st["vcd"])

    sub = template[["batch_id", "elapsed_time"]].copy()
    for v in TARGETS:
        sub[v] = st[v]
    assert sub[TARGETS].notna().all().all() and (sub[TARGETS] > 0).all().all()
    sub.to_csv(OUT / "holdout_predictions.csv", index=False)
    pd.DataFrame(wtab, index=MEMBERS).T.round(3).to_csv(OUT / "blend_weights.csv")
    pd.DataFrame([sc]).round(3).to_csv(OUT / "cv_score.csv", index=False)
    print(pd.DataFrame(wtab, index=MEMBERS).T.round(2).to_string())
    for ref in ("hybrid_v3", "stack_v5"):
        prev = pd.read_csv(ROOT / f"outputs/{ref}/holdout_predictions.csv", dtype={"batch_id": str})
        ratio = sub[TARGETS].to_numpy() / prev[TARGETS].to_numpy()
        print(f"v7 / {ref} holdout ratio: median", np.median(ratio, 0).round(3),
              "p5", np.percentile(ratio, 5, 0).round(3), "p95", np.percentile(ratio, 95, 0).round(3))
    print("wrote", OUT / "holdout_predictions.csv")


if __name__ == "__main__":
    main()
