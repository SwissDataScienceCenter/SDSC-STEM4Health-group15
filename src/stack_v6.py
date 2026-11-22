"""v6 stack: holdout predictions for the best model from experiments_v6.

Members = the 11 v5 members + v3 and functional-PCA members on three new
stage-1 cores (two-population core, consensus of six cores, sensor-conditioned
particle core). Weights: simplex search (zeros allowed), per variable and per
segment (days 1-6 / 7-13), fitted on out-of-fold training predictions. Finish:
interval-aware glucose posterior (bolus rule) and TCD = VCD x modelled ratio.
The data-driven branch of every v3 member uses 30 batch-bootstrap bags.

    python src/core_runner.py --core twopop --name core_twopop
    python src/core_avg.py
    python src/core_assim.py
    python src/stack_v6.py
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
import experiments_v6_ideas as iv6  # noqa: E402
import hybrid_v3 as h3  # noqa: E402
import stack_v5 as s5  # noqa: E402

ROOT = h3.ROOT
OUT = ROOT / "outputs" / "stack_v6"
TARGETS = h3.TARGETS
CORES = dict(s5.CORES)
CORES.update({
    "twopop": ("core_twopop", "core_twopop", ROOT / "outputs/core_twopop/holdout_predictions.csv", "tp"),
    "avg": ("core_avg", "avg", ROOT / "outputs/core_avg/holdout_predictions.csv", "avg"),
    "assim": ("core_assim", "assim", ROOT / "outputs/core_assim/holdout_predictions.csv", "as"),
})
DD_BAGS = 30
SRCS = e.BASE_SRCS + ["tp", "tp_fp", "avg", "avg_fp", "as", "as_fp"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s5.CORES = CORES
    template = pd.read_csv(h3.DATA_DIR / "bioreaction_template.csv", dtype={"batch_id": str})
    x.OUT = e.CV
    mem = pd.read_csv(e.CV / "members_oof_v6h.csv", dtype={"batch_id": str})  # data-driven branch: 30 bags
    hold, hrows = {}, {}
    for core, (d, model, _, pref) in CORES.items():
        rows, events = s5.load_rows(core)
        ideas.CORES.setdefault(core, (d, model))
        _, parts = ideas.core_parts(core, events)
        p30 = pd.read_csv(e.CV / f"parts30_{core}.csv", dtype={"batch_id": str})
        assert np.allclose(p30["fin|vcd"].to_numpy(), mem[f"{pref}|vcd"].to_numpy())
        hr = s5.holdout_rows(core, events, template)
        fin = h3.finalize(h3.HybridBranch().fit(rows).predict(hr), h3.DataDrivenBranch(n_bags=DD_BAGS).fit(rows).predict(hr))
        fo = ideas.fpca_member(rows, parts, K=s5.K, alpha=s5.ALPHA)
        fp = s5.fpca_full(rows, hr, {v: fo[f"fp|{v}"].to_numpy() for v in TARGETS})
        for v in TARGETS:
            hold[f"{pref}|{v}"] = fin[v].to_numpy()
            hold[f"{pref}_fp|{v}"] = fp[v]
        if core == "states":
            ro = ideas.fpca_bagged(rows, parts, K=s5.K, alpha=s5.ALPHA, n_bags=1, use_s1_offset=False)
            raw = s5.fpca_full(rows, hr, {v: ro[f"fb|{v}"].to_numpy() for v in TARGETS}, use_offset=False)
            for v in TARGETS:
                hold[f"raw|{v}"] = raw[v]
            rows_s, hrows = rows, hr
        print(f"core {core}: done", flush=True)
    hold = pd.DataFrame(hold)

    # honest CV of exactly this configuration
    o_cv = e.stack(mem, SRCS, rows=rows_s, clip=False)
    c_cv = iv6.finish_all(mem, o_cv, rows_s)
    sc = x.score(c_cv, "f")
    print("nested CV:", sc.round(3).to_dict())

    # stack weights on all training OOF predictions
    day_tr, day_ho = mem.day.to_numpy(), hrows.day.to_numpy()
    st = {}
    wtab = {}
    for v in TARGETS:
        L = np.column_stack([np.log(mem[f"{s}|{v}"].to_numpy()) for s in SRCS])
        H = np.column_stack([np.log(hold[f"{s}|{v}"].to_numpy()) for s in SRCS])
        y = mem["obs_" + v].to_numpy()
        out = np.zeros(len(H))
        for si, seg in enumerate(e.SEG2):
            mt, mh = np.isin(day_tr, list(seg)), np.isin(day_ho, list(seg))
            w = e.weights_simplex(L[mt], y[mt])
            wtab[(v, f"days {seg.start}-{seg.stop - 1}")] = w
            out[mh] = np.exp(H[mh] @ w)
        st[v] = out
    st["tcd"] = np.maximum(st["tcd"], st["vcd"])

    # glucose: interval-aware posterior, residual pool = nested OOF residuals of the stack
    r_pool = np.log(o_cv.obs_substrate.to_numpy() / o_cv["st|substrate"].to_numpy())
    lo, hi = hrows.g_lo.to_numpy(), hrows.g_hi.to_numpy()
    glc = st["substrate"].copy()
    for i in np.where((lo > 0) | np.isfinite(hi))[0]:
        glc[i] = x.censored_point(glc[i], lo[i], hi[i], r_pool)
    st["substrate"] = glc

    # TCD = VCD x ratio model fitted on training (OOF inputs)
    def ratio_X(r_st, rws, mm):
        return np.c_[r_st, rws[["via0", "ls1_vcd", "base_tot", "A_early"]].to_numpy(float),
                     np.log(mm["mech|tcd"] / mm["mech|vcd"]), np.log(mm["msens|tcd"] / mm["msens|vcd"])]

    Xtr = ratio_X(np.log(o_cv["st|tcd"] / o_cv["st|vcd"]).to_numpy(), rows_s, mem)
    Xho = ratio_X(np.log(st["tcd"] / st["vcd"]), hrows, hold)
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
    pd.DataFrame(wtab, index=SRCS).T.round(3).to_csv(OUT / "stack_weights.csv")
    pd.DataFrame([sc]).round(3).to_csv(OUT / "cv_score.csv", index=False)
    prev = pd.read_csv(ROOT / "outputs/stack_v5/holdout_predictions.csv", dtype={"batch_id": str})
    ratio = sub[TARGETS].to_numpy() / prev[TARGETS].to_numpy()
    print("v6 / v5 holdout ratio: median", np.median(ratio, 0).round(3), "p5", np.percentile(ratio, 5, 0).round(3),
          "p95", np.percentile(ratio, 95, 0).round(3))
    print("wrote", OUT / "holdout_predictions.csv")


if __name__ == "__main__":
    main()
