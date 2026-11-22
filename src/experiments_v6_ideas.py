"""Idea implementations for experiments_v6.py."""

from __future__ import annotations

import numpy as np
import pandas as pd

import experiments_v5 as x
import experiments_v6 as e
import hybrid_v3 as h3

TARGETS = e.TARGETS
CORE_PREF = {"states": "states", "glc": "glc", "mech": "mech", "monod": "monod", "monod_sens": "msens"}


def base(mem, rows, events):
    o = e.stack(mem, e.BASE_SRCS, wfun=e.weights_old, rows=rows)
    e.log_trial("0 v5 stack (reference)", x.score(o, "st"), "old weight search (min weight ~0.04)")


def i01_simplex(mem, rows, events):
    """Weight search that can drop members completely."""
    o, W = e.stack(mem, e.BASE_SRCS, rows=rows, return_w=True)
    e.log_trial("1 simplex weights (zeros allowed)", x.score(o, "st"))
    for k, w in W.items():
        print(k, dict((s, round(float(a), 2)) for s, a in zip(e.BASE_SRCS, w) if a > 0.01))


def i02_branches(mem, rows, events):
    """Hybrid branch and data-driven branch of every core as separate members."""
    m = mem.copy()
    srcs = list(e.BASE_SRCS)
    for core, pref in CORE_PREF.items():
        m = e.add_parts(m, core, pref, cols=("hyb", "dd"))
        srcs += [f"{pref}_hyb", f"{pref}_dd"]
    o = e.stack(m, srcs, rows=rows)
    e.log_trial("2 hyb + dd branches as members", x.score(o, "st"), f"{len(srcs)} members")
    m.to_csv(e.CV / "members_oof_v6.csv", index=False)


def i03_segments(mem, rows, events):
    """Finer time segments for the stack weights."""
    m = pd.read_csv(e.CV / "members_oof_v6.csv", dtype={"batch_id": str}) if (e.CV / "members_oof_v6.csv").exists() else mem
    srcs = [c.split("|")[0] for c in m.columns if c.endswith("|vcd") and not c.startswith("obs")]
    for name, segs in {"1 segment": [range(1, 14)],
                       "3 segments (1-4, 5-9, 10-13)": [range(1, 5), range(5, 10), range(10, 14)],
                       "4 segments (1-3, 4-7, 8-10, 11-13)": [range(1, 4), range(4, 8), range(8, 11), range(11, 14)]}.items():
        o = e.stack(m, srcs, segments=segs, rows=rows)
        e.log_trial(f"3 stack {name}", x.score(o, "st"), f"{len(srcs)} members")


def add_core(mem, events, core, d, model, pref):
    """Build v3 + FPCA on a new stage-1 core (cached in stack_v5/cv) and add as members."""
    import experiments_v5_ideas as ideas
    ideas.CORES[core] = (d, model)
    _, p = ideas.core_parts(core, events)
    assert (p.batch_id.to_numpy() == mem.batch_id.to_numpy()).all()
    m = mem.copy()
    for v in TARGETS:
        m[f"{pref}|{v}"] = p[f"fin|{v}"].to_numpy()
        m[f"{pref}_fp|{v}"] = p[f"fps|{v}"].to_numpy()
    e.log_trial(f"   v3 on core {core}", x.score(m, pref))
    return m


def i04_twopop(mem, rows, events):
    m = add_core(mem, events, "twopop", "core_twopop", "core_twopop", "tp")
    o = e.stack(m, e.BASE_SRCS + ["tp", "tp_fp"], rows=rows)
    e.log_trial("4 + two-population core", x.score(o, "st"), "simplex weights")
    m.to_csv(e.CV / "members_oof_v6b.csv", index=False)


def _stack_tp(rows, clip=True):
    m = pd.read_csv(e.CV / "members_oof_v6b.csv", dtype={"batch_id": str})
    return m, e.stack(m, e.BASE_SRCS + ["tp", "tp_fp"], rows=rows, clip=clip)


def i06_censor(mem, rows, events):
    """Glucose: interval-aware MAPE-optimal posterior instead of a clip."""
    m, o = _stack_tp(rows, clip=False)
    for pool in (2, 13):
        c = x.idea_censor(rows, o, src="st", dst="cen", days_pool=pool)
        e.log_trial(f"6 censored glucose posterior p{pool}", x.score(c, "cen"), "on stack incl. two-pop")


def i07_tcd_ratio(mem, rows, events):
    """TCD = stacked VCD x modelled TCD/VCD ratio (per-day ridge, nested)."""
    m, o = _stack_tp(rows)
    y = np.log(o.obs_tcd / o.obs_vcd).to_numpy()
    r_st = np.log(o["st|tcd"] / o["st|vcd"]).to_numpy()
    X = np.c_[r_st, rows[["via0", "ls1_vcd", "base_tot", "A_early"]].to_numpy(float),
              np.log(m["mech|tcd"] / m["mech|vcd"]), np.log(m["msens|tcd"] / m["msens|vcd"])]
    pr = np.zeros(len(o))
    for f in range(h3.N_FOLDS):
        tr, te = (o.fold != f).to_numpy(), (o.fold == f).to_numpy()
        for d in range(1, 14):
            mtr, mte = tr & (o.day == d).to_numpy(), te & (o.day == d).to_numpy()
            pr[mte] = x.ridge_fit(X[mtr], y[mtr], 5.0)(X[mte])
    for w in (1.0, 0.5):
        c = o.copy()
        c["st|tcd"] = np.maximum(c["st|vcd"] * np.exp(w * pr + (1 - w) * r_st), c["st|vcd"])
        e.log_trial(f"7 TCD via stacked VCD x ratio w{w}", x.score(c, "st"))


def i08_consensus(mem, rows, events):
    m = pd.read_csv(e.CV / "members_oof_v6b.csv", dtype={"batch_id": str})
    m = add_core(m, events, "avg", "core_avg", "avg", "avg")
    srcs = e.BASE_SRCS + ["tp", "tp_fp", "avg", "avg_fp"]
    o = e.stack(m, srcs, rows=rows)
    e.log_trial("8 + consensus stage-1 core", x.score(o, "st"), "simplex weights")
    m.to_csv(e.CV / "members_oof_v6c.csv", index=False)


def i05_bagged_monod(mem, rows, events):
    m = pd.read_csv(e.CV / "members_oof_v6c.csv", dtype={"batch_id": str})
    m = add_core(m, events, "monod_bag5", "core_monod_bag5", "core_monod_bag5", "mbag")
    base_srcs = e.BASE_SRCS + ["tp", "tp_fp", "avg", "avg_fp"]
    o = e.stack(m, base_srcs + ["mbag", "mbag_fp"], rows=rows)
    e.log_trial("5 + bootstrap-bagged Monod core", x.score(o, "st"), "simplex weights")
    m.to_csv(e.CV / "members_oof_v6d.csv", index=False)


ALL_CORES = {"states": ("hybrid", "states", "states"), "glc": ("hybrid_glc", "states", "glc"),
             "mech": ("hybrid", "mechanistic", "mech"), "monod": ("core_monod", "monod", "monod"),
             "monod_sens": ("core_monod_sens", "monod_sens", "msens"), "twopop": ("core_twopop", "core_twopop", "tp"),
             "avg": ("core_avg", "avg", "avg"), "assim": ("core_assim", "assim", "as")}


def i09_dd30(mem, rows, events):
    """Data-driven branch with 30 instead of 10 batch-bootstrap bags in every core."""
    import experiments_v5_ideas as ideas
    m = pd.read_csv(e.CV / "members_oof_v6c.csv", dtype={"batch_id": str})
    for core, (d, model, pref) in ALL_CORES.items():
        path = e.CV / f"parts30_{core}.csv"
        if path.exists():
            p = pd.read_csv(path, dtype={"batch_id": str})
        else:
            saved = (h3.STAGE1_DIR, h3.STAGE1_MODEL)
            h3.STAGE1_DIR, h3.STAGE1_MODEL = h3.ROOT / "outputs" / d, model
            try:
                r, _ = x.build()
            finally:
                h3.STAGE1_DIR, h3.STAGE1_MODEL = saved
            p = x.run_parts(r, dd_cls=lambda: h3.DataDrivenBranch(n_bags=30))
            p.to_csv(path, index=False)
        for v in TARGETS:
            m[f"{pref}|{v}"] = p[f"fin|{v}"].to_numpy()
        e.log_trial(f"   v3 (dd 30 bags) on core {core}", x.score(m, pref))
    if "as_fp|vcd" not in m:
        f = pd.read_csv(e.CV / "members_oof_v6f.csv", dtype={"batch_id": str})
        for v in TARGETS:
            m[f"as_fp|{v}"] = f[f"as_fp|{v}"].to_numpy()
    srcs = e.BASE_SRCS + ["tp", "tp_fp", "avg", "avg_fp"]
    o = e.stack(m, srcs, rows=rows)
    e.log_trial("9 data-driven branch 30 bags, all cores", x.score(o, "st"), "simplex weights")
    m.to_csv(e.CV / "members_oof_v6h.csv", index=False)
    final(m, rows, events, member_file="members_oof_v6h.csv", extra=("as", "as_fp"), tag="9+10 dd30 everywhere + assim")


def finish_all(m, o, rows, pool=13):
    """Glucose interval posterior (instead of clip) + TCD through the VCD ratio model."""
    c = x.idea_censor(rows, o, src="st", dst="f", days_pool=pool)
    y = np.log(c.obs_tcd / c.obs_vcd).to_numpy()
    r_st = np.log(c["f|tcd"] / c["f|vcd"]).to_numpy()
    X = np.c_[r_st, rows[["via0", "ls1_vcd", "base_tot", "A_early"]].to_numpy(float),
              np.log(m["mech|tcd"] / m["mech|vcd"]), np.log(m["msens|tcd"] / m["msens|vcd"])]
    pr = np.zeros(len(c))
    for f in range(h3.N_FOLDS):
        tr, te = (c.fold != f).to_numpy(), (c.fold == f).to_numpy()
        for d in range(1, 14):
            mtr, mte = tr & (c.day == d).to_numpy(), te & (c.day == d).to_numpy()
            pr[mte] = x.ridge_fit(X[mtr], y[mtr], 5.0)(X[mte])
    c["f|tcd"] = np.maximum(c["f|vcd"] * np.exp(pr), c["f|vcd"])
    return c


def final(mem, rows, events, member_file="members_oof_v6c.csv", extra=(), tag="final"):
    m = pd.read_csv(e.CV / member_file, dtype={"batch_id": str})
    srcs = e.BASE_SRCS + ["tp", "tp_fp", "avg", "avg_fp"] + list(extra)
    o = e.stack(m, srcs, rows=rows, clip=False)
    c = finish_all(m, o, rows)
    e.log_trial(f"{tag}: stack + glucose posterior + TCD ratio", x.score(c, "f"), f"{len(srcs)} members")
    return c


def i10_assim(mem, rows, events):
    m = pd.read_csv(e.CV / "members_oof_v6c.csv", dtype={"batch_id": str})
    m = add_core(m, events, "assim", "core_assim", "assim", "as")
    o = e.stack(m, e.BASE_SRCS + ["tp", "tp_fp", "avg", "avg_fp", "as", "as_fp"], rows=rows)
    e.log_trial("10 + sensor-conditioned particle core", x.score(o, "st"), "simplex weights")
    m.to_csv(e.CV / "members_oof_v6f.csv", index=False)
    final(m, rows, events, member_file="members_oof_v6f.csv", extra=("as", "as_fp"), tag="final+assim")
