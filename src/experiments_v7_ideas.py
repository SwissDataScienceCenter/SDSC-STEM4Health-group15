"""Idea implementations for experiments_v7.py (max 4 models per blend)."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

import experiments_v5 as x
import experiments_v6 as e
import experiments_v6_ideas as iv6
import experiments_v7 as s7
import hybrid_v3 as h3

TARGETS = h3.TARGETS
ALL = e.BASE_SRCS + ["tp", "tp_fp", "avg", "avg_fp", "as", "as_fp"]


def base(mem, rows, events):
    p = pd.read_csv(e.CV / "parts_states.csv", dtype={"batch_id": str})
    s7.log_trial("0 hybrid_v3 (reference)", x.score(p, "fin"), "1 model")
    s7.log_trial("0b v6 stack, 17 models (reference)", x.score(iv6.final(mem, rows, events, "members_oof_v6h.csv", ("as", "as_fp"), tag="   ref"), "f"), "exceeds the 4-model cap")


def i01_select4(mem, rows, events):
    for k in (1, 2, 3, 4):
        o, picks, pre = s7.stack_k(mem, ALL, rows, k=k)
        c = Counter(tuple(p) for p in picks).most_common(2)
        s7.log_trial(f"1 best {k}-model blend (nested selection)", x.score(o, pre), f"picks: {c}")


def add(mem, events, core, pref, d=None, model=None):
    """v3 + FPCA on a stage-1 core in outputs/<core> (cached in stack_v5/cv)."""
    m = iv6.add_core(mem, events, core, d or core, model or core, pref)
    return m


def i02_assim_variants(mem, rows, events):
    m = mem.copy()
    for core, pref in (("a2_p3ic", "a3i"), ("a2_p5ic", "a5i"), ("a2_p5ic_s12", "a5s")):
        m = add(m, events, core, pref)
    m.to_csv(e.CV / "members_oof_v7a.csv", index=False)
    cands = ALL + ["a3i", "a3i_fp", "a5i", "a5i_fp", "a5s", "a5s_fp"]
    o, picks, pre = s7.stack_k(m, cands, rows, k=4)
    s7.log_trial("2 best 4-model blend incl. assim variants", x.score(o, pre), f"picks: {Counter(tuple(p) for p in picks).most_common(2)}")


def rows_for(core, d=None, model=None):
    saved = (h3.STAGE1_DIR, h3.STAGE1_MODEL)
    h3.STAGE1_DIR, h3.STAGE1_MODEL = h3.ROOT / "outputs" / (d or core), model or core
    try:
        r, ev = x.build()
    finally:
        h3.STAGE1_DIR, h3.STAGE1_MODEL = saved
    return r


def with_post(r, core):
    P = pd.read_csv(h3.ROOT / "outputs" / core / "posterior_params.csv", dtype={"batch_id": str})
    cols = [c for c in P.columns if c.startswith("post")]
    r = r.merge(P, on="batch_id", how="left")
    assert r[cols].notna().all().all()
    return r, cols


def run_feat(r, cell=None, s=None, l=None, dd=None, dd_bags=10):
    import experiments_v5_ideas as ideas
    saved = (h3.CELL_COLS, h3.S_COLS, h3.L_COLS, dict(h3.DD_COLS))
    try:
        if cell is not None:
            h3.CELL_COLS = cell
        if s is not None:
            h3.S_COLS = s
        if l is not None:
            h3.L_COLS = l
        if dd is not None:
            h3.DD_COLS = {**h3.DD_COLS, **dd}
        return x.run_parts(r, dd_cls=lambda: h3.DataDrivenBranch(n_bags=dd_bags))
    finally:
        h3.CELL_COLS, h3.S_COLS, h3.L_COLS, h3.DD_COLS = saved


def i03_post_feats(mem, rows, events):
    core = "a2_p5ic"
    r, pc = with_post(rows_for(core), core)
    variants = {
        "3a post -> DD only": dict(dd={v: h3.DD_COLS[v] + pc for v in TARGETS}),
        "3b post -> DD + cells": dict(dd={v: h3.DD_COLS[v] + pc for v in TARGETS}, cell=h3.CELL_COLS + pc),
        "3c post -> everywhere": dict(dd={v: h3.DD_COLS[v] + pc for v in TARGETS}, cell=h3.CELL_COLS + pc,
                                      s=h3.S_COLS + pc, l=h3.L_COLS + pc),
    }
    m = pd.read_csv(e.CV / "members_oof_v7a.csv", dtype={"batch_id": str})
    for name, kw in variants.items():
        p = run_feat(r, **kw)
        s7.log_trial(f"   v3 on {core}, {name}", x.score(p, "fin"))
        tag = name.split()[0]
        for v in TARGETS:
            m[f"pf{tag}|{v}"] = p[f"fin|{v}"].to_numpy()
    m.to_csv(e.CV / "members_oof_v7b.csv", index=False)
    cands = ALL + ["a3i", "a3i_fp", "a5i", "a5i_fp", "a5s", "a5s_fp", "pf3a", "pf3b", "pf3c"]
    o, picks, pre = s7.stack_k(m, cands, rows, k=4)
    s7.log_trial("3 best 4-model blend incl. posterior-feature v3", x.score(o, pre), f"picks: {Counter(tuple(p) for p in picks).most_common(2)}")


def finish4(m, o, rows, members, pool=13):
    """Finish using only the blend's own members: glucose interval posterior,
    TCD = VCD x ratio model whose inputs are the members' own TCD/VCD ratios."""
    c = x.idea_censor(rows, o, src="st", dst="f", days_pool=pool)
    y = np.log(c.obs_tcd / c.obs_vcd).to_numpy()
    X = np.c_[np.log(c["f|tcd"] / c["f|vcd"]).to_numpy(), rows[["via0", "ls1_vcd", "base_tot", "A_early"]].to_numpy(float),
              *[np.log(m[f"{s}|tcd"] / m[f"{s}|vcd"]).to_numpy() for s in members]]
    pr = np.zeros(len(c))
    for f in range(h3.N_FOLDS):
        tr, te = (c.fold != f).to_numpy(), (c.fold == f).to_numpy()
        for d in range(1, 14):
            mtr, mte = tr & (c.day == d).to_numpy(), te & (c.day == d).to_numpy()
            pr[mte] = x.ridge_fit(X[mtr], y[mtr], 5.0)(X[mte])
    c["f|tcd"] = np.maximum(c["f|vcd"] * np.exp(pr), c["f|vcd"])
    return c


def blend4(m, rows, members, tag, note=""):
    assert len(members) <= s7.MAX_MODELS
    o = e.stack(m, members, rows=rows, clip=False)
    c = finish4(m, o, rows, members)
    s = x.score(c, "f")
    s7.log_trial(tag, s, note or "+".join(members))
    return c, s


def i04_fixed_sets(mem, rows, events):
    m = pd.read_csv(e.CV / "members_oof_v7b.csv", dtype={"batch_id": str})
    sets = {
        "4a assim-v3(post) + assim-FPCA + v3 + Monod-v3": ["pf3a", "a3i_fp", "states", "monod"],
        "4b assim-v3(post) + assim-FPCA + v3 + raw FPCA": ["pf3a", "a3i_fp", "states", "raw"],
        "4c assim-v3(post) + assim-FPCA + v3 + 2pop-v3": ["pf3a", "a3i_fp", "states", "tp"],
        "4d assim-v3 + assim-FPCA + v3 + Monod-v3": ["a5i", "a3i_fp", "states", "monod"],
        "4e assim-v3(post) + assim-FPCA + mech-v3 + Monod-v3": ["pf3a", "a3i_fp", "mech", "monod"],
        "4f assim-v3(post) + v3 + mech-v3 + Monod-v3": ["pf3a", "states", "mech", "monod"],
    }
    for tag, mem_ in sets.items():
        blend4(m, rows, mem_, tag)


def i05_dd30(mem, rows, events):
    m = pd.read_csv(e.CV / "members_oof_v7b.csv", dtype={"batch_id": str})
    path = e.CV / "parts30_pf3a.csv"
    if path.exists():
        p = pd.read_csv(path, dtype={"batch_id": str})
    else:
        core = "a2_p5ic"
        r, pc = with_post(rows_for(core), core)
        p = run_feat(r, dd={v: h3.DD_COLS[v] + pc for v in TARGETS}, dd_bags=30)
        p.to_csv(path, index=False)
    for v in TARGETS:
        m[f"pf3a30|{v}"] = p[f"fin|{v}"].to_numpy()
    for core, pref in (("states", "states"), ("twopop", "tp"), ("monod", "monod")):
        q = pd.read_csv(e.CV / f"parts30_{core}.csv", dtype={"batch_id": str})
        for v in TARGETS:
            m[f"{pref}30|{v}"] = q[f"fin|{v}"].to_numpy()
    s7.log_trial("   v3 on assim + post feats, dd 30 bags", x.score(m, "pf3a30"))
    m.to_csv(e.CV / "members_oof_v7c.csv", index=False)
    blend4(m, rows, ["pf3a30", "a3i_fp", "states30", "tp30"], "5a = 4c with 30-bag data-driven branches")
    blend4(m, rows, ["pf3a30", "a3i_fp", "states30", "monod30"], "5b = 4a with 30-bag data-driven branches")


def i06_robust(mem, rows, events):
    m = pd.read_csv(e.CV / "members_oof_v7c.csv", dtype={"batch_id": str})
    for fp in ("a3i_fp", "a5i_fp", "as_fp"):
        blend4(m, rows, ["pf3a", fp, "states", "tp"], f"6 set 4c with FPCA member {fp}")
    saved = e.SEG2
    for name, segs in {"1 segment": [range(1, 14)], "3 segments": [range(1, 5), range(5, 10), range(10, 14)]}.items():
        e.SEG2 = segs
        blend4(m, rows, ["pf3a", "a3i_fp", "states", "tp"], f"6 set 4c, weights with {name}")
    e.SEG2 = saved


def i07_tp_assim(mem, rows, events):
    m = pd.read_csv(e.CV / "members_oof_v7c.csv", dtype={"batch_id": str})
    m = add(m, events, "a2tp_p5ic", "atp")
    # also with posterior features in the data-driven branch
    r, pc = with_post(rows_for("a2tp_p5ic"), "a2tp_p5ic")
    p = run_feat(r, dd={v: h3.DD_COLS[v] + pc for v in TARGETS})
    for v in TARGETS:
        m[f"atpf|{v}"] = p[f"fin|{v}"].to_numpy()
    s7.log_trial("   v3 on assimilated two-pop core + post feats", x.score(m, "atpf"))
    m.to_csv(e.CV / "members_oof_v7d.csv", index=False)
    blend4(m, rows, ["pf3a", "a3i_fp", "states", "atp"], "7a set 4c with assimilated two-pop core")
    blend4(m, rows, ["pf3a", "a3i_fp", "states", "atpf"], "7b set 4c with assim two-pop + post feats")
    blend4(m, rows, ["pf3a", "atp_fp", "states", "tp"], "7c set 4c with two-pop assim FPCA")


def blend4_seg(m, rows, members, segs, tag):
    o = e.stack(m, members, segments=segs, rows=rows, clip=False)
    c = finish4(m, o, rows, members)
    s7.log_trial(tag, x.score(c, "f"), "+".join(members))
    return c


def i08_blend_structure(mem, rows, events):
    m = pd.read_csv(e.CV / "members_oof_v7c.csv", dtype={"batch_id": str})
    M = ["pf3a", "a3i_fp", "states", "tp"]
    for name, segs in {"1 segment": [range(1, 14)], "3 segments": [range(1, 5), range(5, 10), range(10, 14)],
                       "per-day weights": [range(d, d + 1) for d in range(1, 14)]}.items():
        blend4_seg(m, rows, M, segs, f"8a set 4c, {name}")
    # post-blend per-day MAPE-optimal scaling (fitted on other folds)
    o = e.stack(m, M, rows=rows, clip=False)
    c = finish4(m, o, rows, M)
    sc = x.per_day_scale(c.rename(columns={f"f|{v}": f"f|{v}" for v in TARGETS}), "f", "g")
    s7.log_trial("8b set 4c + per-day MAPE-optimal scaling", x.score(sc, "g"))
    # glucose posterior with residual pools restricted to nearby days
    for pool in (1, 3):
        o = e.stack(m, M, rows=rows, clip=False)
        c = finish4(m, o, rows, M, pool=pool)
        s7.log_trial(f"8c set 4c, glucose posterior pool +-{pool} d", x.score(c, "f"))
