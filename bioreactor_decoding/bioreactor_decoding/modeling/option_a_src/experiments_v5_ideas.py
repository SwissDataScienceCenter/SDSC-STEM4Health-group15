"""Idea implementations for experiments_v5.py. Each function logs its own trials."""

from __future__ import annotations

import numpy as np
import pandas as pd

import experiments_v5 as x
import hybrid_v3 as h3

TARGETS = x.TARGETS
BATCH_FEATS = ["vcd0", "s0", "l0", "v0", "via0", "base_to3", "base_to4", "base_to5", "base_tot", "A11", "A12", "A_early"]


def i01_censor(rows, base, events):
    """Glucose: exact bolus-controller interval (clip, then censored posterior)."""
    o = x.idea_clip(rows, base)
    x.log_trial("1a glucose interval clip", x.score(o, "clip"))
    for pool in (1, 2, 13):
        o = x.idea_censor(rows, base, days_pool=pool)
        x.log_trial(f"1b censored posterior p{pool}", x.score(o, "cen"), f"residual pool +-{pool} d")


def i02_lac_bolus(rows, base, events):
    """Lactate late days: glucose-limitation (bolus sizes) drives lactate consumption."""
    cols = ["A11", "A12", "A_early", "base_tot"]
    for a in (10.0, 30.0, 100.0):
        o = x.idea_stage3_ridge(rows, base, "byproduct", cols, days=range(9, 14), alpha=a)
        x.log_trial(f"2 lactate ~ bolus a{a:g}", x.score(o, "s3"), "stage-3 ridge, days 9-13")


def i03_gp(rows, base, events):
    """GP smoother of residuals across similar batches and neighbouring days."""
    for var in ("substrate", "byproduct", "vcd"):
        best = None
        for ls_x in (1.5, 3.0):
            for ls_t in (1.5, 4.0):
                for noise in (1.0, 3.0):
                    o = x.idea_gp_residual(rows, base, var, BATCH_FEATS, ls_t=ls_t, ls_x=ls_x, noise=noise)
                    s = x.score(o, "gp")
                    if best is None or s[var] < best[0][var]:
                        best = (s, (ls_x, ls_t, noise))
        x.log_trial(f"3 GP residual {var}", best[0], f"best (ls_x, ls_t, noise)={best[1]} (grid on CV: optimistic)")


def _merge_other(rows, path, name):
    o = pd.read_csv(path, dtype={"batch_id": str})
    cols = {f"v3 final (50/50 blend)|{v}": f"{name}|{v}" for v in TARGETS}
    return rows[["batch_id", "t_key"]].merge(o[["batch_id", "t_key"] + list(cols)].rename(columns=cols),
                                             on=["batch_id", "t_key"], how="left")


def i04_glc_ens(rows, base, events):
    """Ensemble with v3 rebuilt on the glucose-uptake stage-1 variant."""
    g = _merge_other(rows, x.ROOT / "outputs" / "hybrid_v3_glc" / "oof_predictions.csv", "glc")
    o = base.copy()
    for v in TARGETS:
        o[f"glc|{v}"] = g[f"glc|{v}"].to_numpy()
    for w in (0.5,):
        o2 = x.geo_blend(o, ["fin", "glc"], [1 - w, w], "ens")
        x.log_trial("4 stage-1 ensemble (+glc)", x.score(o2, "ens"), f"geometric, w_glc={w}")
        o3 = x.idea_clip(rows, o2, src="ens", dst="ensc")
        x.log_trial("4+1 ensemble + glucose clip", x.score(o3, "ensc"))


def i05_loss(rows, base, events):
    """MAPE-aligned losses (absolute error, 1/obs weights) in both branches."""
    import experiments_v4b as v4b
    for n in ("dd_l1", "dd_w", "dd_l1_w", "hyb_l1"):
        sc, _ = v4b.run_cv(v4b.CONFIGS[n], rows)
        x.log_trial(f"5 loss {n}", sc)
    v4b.run_cv(v4b.CONFIGS["base"], rows)  # restore module globals


def i06_fpca(rows, base, events):
    """New member: functional PCA. Each batch's log-trajectory (13 days) =
    mean + K principal components; PC scores regressed (ridge) on batch-level
    features (t=0 state, sensors, mean feed deviations, stage-1 PC scores)."""
    feats = rows.groupby("batch_id").agg(**{c: (c, "first") for c in BATCH_FEATS},
                                         **{f"m_{c}": (c, "mean") for c in ["fb_next_dev", "fc_next_dev", "V_dev"]})
    o = base.copy()
    for v in TARGETS:
        Y = rows.pivot_table(index="batch_id", columns="day", values="obs_" + v).loc[feats.index]
        S = rows.pivot_table(index="batch_id", columns="day", values="ls1_" + v).loc[feats.index]
        Y, S = np.log(Y.to_numpy()), S.to_numpy()
        fold = rows.groupby("batch_id").fold.first().loc[feats.index].to_numpy()
        pred = np.zeros_like(Y)
        for f in range(h3.N_FOLDS):
            tr, te = fold != f, fold == f
            R = Y[tr] - S[tr]  # residual trajectory vs stage 1
            mu = R.mean(0)
            U, s, Vt = np.linalg.svd(R - mu, full_matrices=False)
            K = 3
            scores = (R - mu) @ Vt[:K].T
            Xb = np.c_[feats.to_numpy(float), S]
            fn = [x.ridge_fit(Xb[tr], scores[:, k], 30.0) for k in range(K)]
            sc = np.column_stack([g(Xb[te]) for g in fn])
            pred[te] = S[te] + mu + sc @ Vt[:K]
        P = pd.DataFrame(np.exp(pred), index=feats.index, columns=range(1, 14)).stack().rename("p").reset_index()
        P.columns = ["batch_id", "day", "p"]
        o[f"fpca|{v}"] = rows[["batch_id", "day"]].merge(P, on=["batch_id", "day"], how="left").p.to_numpy()
    o = x.floor_tcd(o, "fpca")
    o = x.per_day_scale(o, "fpca", "fpcas")
    x.log_trial("6a FPCA member alone", x.score(o, "fpcas"))
    for w in (0.2, 0.33):
        o2 = x.geo_blend(o, ["fin", "fpcas"], [1 - w, w], "b")
        x.log_trial(f"6b blend v3 + FPCA w{w}", x.score(o2, "b"))


def i07_tcd_ratio(rows, base, events):
    """TCD = predicted VCD x predicted TCD/VCD ratio (shares the cell-count noise)."""
    o = base.copy()
    r = np.log(o["fin|tcd"] / o["fin|vcd"])
    # ratio model: per-day ridge on log(TCD/VCD) obs with stage-1 ratio and t=0 viability
    rows2 = rows.copy()
    rows2["ls1_ratio"] = rows.ls1_tcd - rows.ls1_vcd
    y = np.log(o.obs_tcd / o.obs_vcd).to_numpy()
    X = rows2[["ls1_ratio", "via0", "ls1_vcd", "base_tot", "A_early"]].to_numpy(float)
    pr = np.zeros(len(o))
    for f in range(h3.N_FOLDS):
        tr, te = (o.fold != f).to_numpy(), (o.fold == f).to_numpy()
        for d in range(1, 14):
            mtr, mte = tr & (o.day == d).to_numpy(), te & (o.day == d).to_numpy()
            pr[mte] = x.ridge_fit(X[mtr], y[mtr], 5.0)(X[mte])
    for w in (1.0, 0.5):
        o[f"rat|tcd"] = o["fin|vcd"] * np.exp(w * pr + (1 - w) * r)
        for v in ("vcd", "substrate", "byproduct"):
            o[f"rat|{v}"] = o[f"fin|{v}"]
        x.log_trial(f"7 TCD via ratio w{w}", x.score(o, "rat"))


def _batch_table(rows):
    feats = rows.groupby("batch_id").agg(**{c: (c, "first") for c in BATCH_FEATS},
                                         **{f"m_{c}": (c, "mean") for c in ["fb_next_dev", "fc_next_dev", "V_dev"]})
    fold = rows.groupby("batch_id").fold.first().loc[feats.index].to_numpy()
    return feats, fold


def _traj(rows, col, index):
    return rows.pivot_table(index="batch_id", columns="day", values=col).loc[index].to_numpy()


def _unstack(rows, mat, index):
    P = pd.DataFrame(mat, index=index, columns=range(1, 14)).stack().rename("p").reset_index()
    P.columns = ["batch_id", "day", "p"]
    return rows[["batch_id", "day"]].merge(P, on=["batch_id", "day"], how="left").p.to_numpy()


def fpca_member(rows, base, K=3, alpha=30.0, use_s1=True, joint=False, extra_day_feats=False):
    """FPCA/PLS-style member. joint=True: one PCA over the concatenated
    4-variable residual trajectory (52 dims), so variables share components."""
    feats, fold = _batch_table(rows)
    idx = feats.index
    Ys = {v: np.log(_traj(rows, "obs_" + v, idx)) for v in TARGETS}
    Ss = {v: _traj(rows, "ls1_" + v, idx) for v in TARGETS}
    Xb = feats.to_numpy(float)
    if use_s1:
        Xb = np.c_[Xb] if False else np.c_[Xb, np.hstack([Ss[v] for v in TARGETS])]
    if extra_day_feats:
        Xb = np.c_[Xb, _traj(rows, "g_lo", idx), np.minimum(_traj(rows, "g_hi", idx), 3.0)]
    o = base.copy()
    groups = [TARGETS] if joint else [[v] for v in TARGETS]
    preds = {}
    for g in groups:
        R = np.hstack([Ys[v] - Ss[v] for v in g])
        P = np.zeros_like(R)
        for f in range(h3.N_FOLDS):
            tr, te = fold != f, fold == f
            mu, sd = R[tr].mean(0), R[tr].std(0) if joint else np.ones(R.shape[1])
            Z = (R[tr] - mu) / sd
            _, _, Vt = np.linalg.svd(Z, full_matrices=False)
            sc = Z @ Vt[:K].T
            fn = [x.ridge_fit(Xb[tr], sc[:, k], alpha) for k in range(K)]
            P[te] = mu + (np.column_stack([q(Xb[te]) for q in fn]) @ Vt[:K]) * sd
        for j, v in enumerate(g):
            preds[v] = P[:, 13 * j:13 * (j + 1)] + Ss[v]
    for v in TARGETS:
        o[f"fp|{v}"] = _unstack(rows, np.exp(preds[v]), idx)
    o = x.floor_tcd(o, "fp")
    return x.per_day_scale(o, "fp", "fps")


def i08_fpca_tune(rows, base, events):
    res = []
    for K in (2, 3, 5):
        for alpha in (10.0, 30.0, 100.0):
            o = fpca_member(rows, base, K=K, alpha=alpha)
            o2 = x.geo_blend(o, ["fin", "fps"], [0.75, 0.25], "b")
            res.append((x.score(o2, "b")["leaderboard_score"], K, alpha, x.score(o, "fps")["leaderboard_score"]))
    res.sort()
    print(res[:5])
    best = res[0]
    o = fpca_member(rows, base, K=best[1], alpha=best[2])
    o2 = x.geo_blend(o, ["fin", "fps"], [0.75, 0.25], "b")
    x.log_trial("8a FPCA tuned blend w.25", x.score(o2, "b"), f"K={best[1]} alpha={best[2]} (picked on CV)")
    for K, a in ((3, 30.0), (5, 30.0)):
        o = fpca_member(rows, base, K=K, alpha=a, joint=True)
        o2 = x.geo_blend(o, ["fin", "fps"], [0.75, 0.25], "b")
        x.log_trial(f"8b joint 4-var FPCA K{K}", x.score(o2, "b"), f"alone {x.score(o, 'fps')['leaderboard_score']:.3f}")
    o = fpca_member(rows, base, K=3, alpha=30.0, extra_day_feats=True)
    o2 = x.geo_blend(o, ["fin", "fps"], [0.75, 0.25], "b")
    x.log_trial("8c FPCA + interval feats", x.score(o2, "b"), f"alone {x.score(o, 'fps')['leaderboard_score']:.3f}")


def i09_fpca_grid(rows, base, events):
    res = []
    for K in (3, 5, 8, 13):
        for alpha in (1.0, 3.0, 10.0):
            o = fpca_member(rows, base, K=K, alpha=alpha)
            for w in (0.25, 0.35, 0.5):
                o2 = x.geo_blend(o, ["fin", "fps"], [1 - w, w], "b")
                s = x.score(o2, "b")
                res.append((round(s["leaderboard_score"], 3), K, alpha, w, round(x.score(o, "fps")["leaderboard_score"], 3)))
    res.sort()
    for r in res[:8]:
        print(r)
    _, K, a, w, alone = res[0]
    o = fpca_member(rows, base, K=K, alpha=a)
    o2 = x.geo_blend(o, ["fin", "fps"], [1 - w, w], "b")
    x.log_trial("9 FPCA grid best", x.score(o2, "b"), f"K={K} alpha={a} w={w}, alone {alone} (picked on CV)")


LS1 = ["ls1_vcd", "ls1_tcd", "ls1_substrate", "ls1_byproduct"]


def i10_cross_s1(rows, base, events):
    """Every correction sees all four stage-1 predictions (cross-variable stacking)."""
    import experiments_v4b as v4b
    V = v4b.variant
    cfgs = {
        "10a DD + all ls1": V("a", dd={v: h3.DD_COLS[v] + LS1 for v in TARGETS}),
        "10b cells + other ls1": V("b", cell=h3.CELL_COLS + ["ls1_substrate", "ls1_byproduct"]),
        "10c S/L + other ls1": V("c", s=h3.S_COLS + ["ls1_byproduct", "ls1_vcd"], l=h3.L_COLS + ["ls1_substrate", "ls1_vcd"]),
    }
    for name, cfg in cfgs.items():
        sc, _ = v4b.run_cv(cfg, rows)
        x.log_trial(name, sc)
    v4b.run_cv(v4b.CONFIGS["base"], rows)


def run_cols(rows, cell=None, s=None, l=None, dd=None):
    """Run the v3 CV with patched feature lists (restored afterwards)."""
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
        return x.run_parts(rows)
    finally:
        h3.CELL_COLS, h3.S_COLS, h3.L_COLS, h3.DD_COLS = saved


def i10b_cross_s1(rows, base, events):
    for name, kw in {
        "10b cells + other ls1": dict(cell=h3.CELL_COLS + ["ls1_substrate", "ls1_byproduct"]),
        "10c S + other ls1": dict(s=h3.S_COLS + ["ls1_byproduct", "ls1_vcd"]),
        "10d L + other ls1": dict(l=h3.L_COLS + ["ls1_substrate", "ls1_vcd"]),
        "10e v4b dev feats (S,L,cells)": dict(s=h3.S_COLS + ["dev_substrate_prev", "dev_substrate_next"],
                                             l=h3.L_COLS + ["dev_byproduct_prev", "dev_byproduct_next"],
                                             cell=h3.CELL_COLS + ["dev_vcd_prev", "dev_vcd_next"]),
    }.items():
        p = run_cols(rows, **kw)
        x.log_trial(name, x.score(p))


def _members(rows, base):
    o = base.copy()
    g = _merge_other(rows, x.ROOT / "outputs" / "hybrid_v3_glc" / "oof_predictions.csv", "glc")
    for v in TARGETS:
        o[f"glc|{v}"] = g[f"glc|{v}"].to_numpy()
    f = fpca_member(rows, base, K=5, alpha=10.0)
    for v in TARGETS:
        o[f"fps|{v}"] = f[f"fps|{v}"].to_numpy()
    return o


def finish(rows, o, src, dst):
    """TCD through ratio to VCD (w=1 version of idea 7 on the blended VCD) is
    not reused here; only the glucose interval clip is applied."""
    return x.idea_clip(rows, o, src=src, dst=dst)


def i11_combo(rows, base, events):
    o = _members(rows, base)
    for wf, wg, wp in ((0.75, 0.0, 0.25), (0.375, 0.375, 0.25), (0.4, 0.3, 0.3)):
        b = x.geo_blend(o, ["fin", "glc", "fps"], [wf, wg, wp], "b")
        c = finish(rows, b, "b", "c")
        x.log_trial(f"11 combo v3/glc/fpca {wf}/{wg}/{wp}", x.score(c, "c"), "+ glucose clip")


def _member_sklearn(rows, base, make, cols_fn, name):
    o = base.copy()
    for v in TARGETS:
        cols = cols_fn(v)
        X = rows[cols].to_numpy(float)
        y = np.log(rows["obs_" + v].to_numpy() / rows["s1_" + v].to_numpy())
        p = np.zeros(len(rows))
        for f in range(h3.N_FOLDS):
            tr, te = (rows.fold != f).to_numpy(), (rows.fold == f).to_numpy()
            mu, sd = X[tr].mean(0), X[tr].std(0)
            sd[sd == 0] = 1
            m = make().fit((X[tr] - mu) / sd, y[tr])
            p[te] = m.predict((X[te] - mu) / sd)
        o[f"{name}|{v}"] = rows["s1_" + v].to_numpy() * np.exp(p)
    o = x.floor_tcd(o, name)
    return x.per_day_scale(o, name, name + "s")


def i12_members(rows, base, events):
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.kernel_ridge import KernelRidge
    cols = lambda v: ["day"] + BATCH_FEATS + ["fb_next_dev", "fc_next_dev", "V_dev", "A", "base_cum"] + LS1
    o = _members(rows, base)
    et = _member_sklearn(rows, base, lambda: ExtraTreesRegressor(300, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=4), cols, "et")
    kr = _member_sklearn(rows, base, lambda: KernelRidge(alpha=1.0, kernel="rbf", gamma=0.02), cols, "kr")
    for v in TARGETS:
        o[f"ets|{v}"], o[f"krs|{v}"] = et[f"ets|{v}"], kr[f"krs|{v}"]
    x.log_trial("12a ExtraTrees alone", x.score(o, "ets"))
    x.log_trial("12b KernelRidge alone", x.score(o, "krs"))
    for name, srcs, w in [("12c v3+fpca+ET", ["fin", "fps", "ets"], [0.6, 0.25, 0.15]),
                          ("12d v3+fpca+KR", ["fin", "fps", "krs"], [0.6, 0.25, 0.15]),
                          ("12e v3+glc+fpca+ET+KR", ["fin", "glc", "fps", "ets", "krs"], [0.3, 0.3, 0.2, 0.1, 0.1])]:
        b = x.geo_blend(o, srcs, w, "b")
        x.log_trial(name, x.score(finish(rows, b, "b", "c"), "c"), f"w={w} + clip")


CORES = {"states": ("hybrid", "states"), "glc": ("hybrid_glc", "states"),
         "feeds": ("hybrid", "states_feeds"), "mech": ("hybrid", "mechanistic")}


def core_parts(core, events):
    """Rows + v3 out-of-fold parts + FPCA member built on one stage-1 core (cached)."""
    path = x.OUT / f"parts_{core}.csv"
    d, m = CORES[core]
    saved = (h3.STAGE1_DIR, h3.STAGE1_MODEL)
    h3.STAGE1_DIR, h3.STAGE1_MODEL = x.ROOT / "outputs" / d, m
    try:
        rows, _ = x.build()
    finally:
        h3.STAGE1_DIR, h3.STAGE1_MODEL = saved
    if path.exists():
        p = pd.read_csv(path, dtype={"batch_id": str})
    else:
        p = x.run_parts(rows)
        f = fpca_member(rows, p, K=5, alpha=10.0)
        for v in TARGETS:
            p[f"fps|{v}"] = f[f"fps|{v}"].to_numpy()
        p.to_csv(path, index=False)
    return rows, p


def i13_cores(rows, base, events):
    parts = {c: core_parts(c, events)[1] for c in CORES}
    key = parts["states"][["batch_id", "t_key"]]
    o = parts["states"][["batch_id", "t_key", "day", "fold"] + ["obs_" + v for v in TARGETS]].copy()
    for c, p in parts.items():
        assert (p.batch_id.to_numpy() == key.batch_id.to_numpy()).all()
        for v in TARGETS:
            o[f"{c}|{v}"] = p[f"fin|{v}"].to_numpy()
            o[f"{c}_fp|{v}"] = p[f"fps|{v}"].to_numpy()
        x.log_trial(f"13 v3 on core '{c}'", x.score(o, c))
    combos = {
        "13a v3 x4 cores": (["states", "glc", "feeds", "mech"], [1, 1, 1, 1]),
        "13b v3 x3 + fpca x3": (["states", "glc", "feeds", "states_fp", "glc_fp", "feeds_fp"], [0.25, 0.25, 0.25, 0.083, 0.083, 0.083]),
        "13c v3 x4 + fpca x4": (["states", "glc", "feeds", "mech", "states_fp", "glc_fp", "feeds_fp", "mech_fp"],
                                [0.1875] * 4 + [0.0625] * 4),
    }
    for name, (srcs, w) in combos.items():
        b = x.geo_blend(o, srcs, w, "b")
        x.log_trial(name, x.score(finish(rows, b, "b", "c"), "c"), "+ clip")
    o.to_csv(x.OUT / "members_oof.csv", index=False)


def fpca_bagged(rows, base, K=5, alpha=10.0, n_bags=20, use_s1_offset=True, seed=0):
    """Bootstrap-bagged FPCA member (batches resampled within training folds)."""
    feats, fold = _batch_table(rows)
    idx = feats.index
    rng = np.random.default_rng(seed)
    o = base.copy()
    Ss = {v: _traj(rows, "ls1_" + v, idx) for v in TARGETS}
    Xb = np.c_[feats.to_numpy(float), np.hstack([Ss[v] for v in TARGETS])]
    for v in TARGETS:
        Y = np.log(_traj(rows, "obs_" + v, idx))
        S = Ss[v] if use_s1_offset else np.zeros_like(Y)
        R = Y - S
        P = np.zeros_like(R)
        for f in range(h3.N_FOLDS):
            tr_i, te = np.where(fold != f)[0], fold == f
            acc = []
            for _ in range(n_bags):
                bi = rng.choice(tr_i, len(tr_i), replace=True) if n_bags > 1 else tr_i
                mu = R[bi].mean(0)
                _, _, Vt = np.linalg.svd(R[bi] - mu, full_matrices=False)
                sc = (R[bi] - mu) @ Vt[:K].T
                fn = [x.ridge_fit(Xb[bi], sc[:, k], alpha) for k in range(K)]
                acc.append(mu + np.column_stack([q(Xb[te]) for q in fn]) @ Vt[:K])
            P[te] = np.mean(acc, 0)
        o[f"fb|{v}"] = _unstack(rows, np.exp(P + S), idx)
    o = x.floor_tcd(o, "fb")
    return x.per_day_scale(o, "fb", "fbs")


def i14_15_fpca_variants(rows, base, events):
    mem = pd.read_csv(x.OUT / "members_oof.csv", dtype={"batch_id": str})
    srcs = ["states", "glc", "feeds", "mech", "states_fp", "glc_fp", "feeds_fp", "mech_fp"]
    w = [0.1875] * 4 + [0.0625] * 4
    fb = fpca_bagged(rows, base)
    x.log_trial("14a bagged FPCA alone", x.score(fb, "fbs"))
    o = mem.copy()
    for v in TARGETS:
        o[f"fbs|{v}"] = fb[f"fbs|{v}"].to_numpy()
    b = x.geo_blend(o, ["fin_states" if False else "states", "fbs"], [0.75, 0.25], "b")
    x.log_trial("14b v3 + bagged FPCA w.25", x.score(finish(rows, b, "b", "c"), "c"), "+ clip")
    raw = fpca_bagged(rows, base, K=5, alpha=10.0, n_bags=1, use_s1_offset=False)
    x.log_trial("15a raw FPCA (no stage-1 offset) alone", x.score(raw, "fbs"))
    for v in TARGETS:
        o[f"raw|{v}"] = raw[f"fbs|{v}"].to_numpy()
    b = x.geo_blend(o, srcs + ["raw"], w + [0.1], "b")
    x.log_trial("15b 13c + raw FPCA 0.1", x.score(finish(rows, b, "b", "c"), "c"), "+ clip")
    o.to_csv(x.OUT / "members_oof.csv", index=False)


def _nested_weights(P, y, fold_tr, grid_step=0.05, n_iter=200, seed=0):
    """Non-negative blend weights (sum 1) minimising MAPE of the geometric blend,
    by coordinate search on the simplex."""
    L = np.log(P)
    k = L.shape[1]
    w = np.full(k, 1.0 / k)
    best = np.mean(np.abs(np.exp(L @ w) / y - 1))
    for _ in range(n_iter):
        improved = False
        for i in range(k):
            for j in range(k):
                if i == j or w[j] < grid_step - 1e-12:
                    continue
                w2 = w.copy()
                w2[i] += grid_step
                w2[j] -= grid_step
                e = np.mean(np.abs(np.exp(L @ w2) / y - 1))
                if e < best - 1e-9:
                    best, w, improved = e, w2, True
        if not improved:
            break
    return w


def i16_nested_stack(rows, base, events):
    """Blend weights per variable learned on the other folds only (honest)."""
    mem = pd.read_csv(x.OUT / "members_oof.csv", dtype={"batch_id": str})
    srcs = ["states", "glc", "feeds", "mech", "states_fp", "glc_fp", "feeds_fp", "mech_fp", "raw"]
    o = mem.copy()
    for v in TARGETS:
        P = np.column_stack([o[f"{s}|{v}"].to_numpy() for s in srcs])
        y = o["obs_" + v].to_numpy()
        out = np.zeros(len(o))
        ws = []
        for f in range(h3.N_FOLDS):
            tr, te = (o.fold != f).to_numpy(), (o.fold == f).to_numpy()
            w = _nested_weights(P[tr], y[tr], None)
            ws.append(w)
            out[te] = np.exp(np.log(P[te]) @ w)
        o[f"st|{v}"] = out
        print(v, dict(zip(srcs, np.round(np.mean(ws, 0), 2))))
    o = x.floor_tcd(o, "st")
    x.log_trial("16 nested stacking weights", x.score(finish(rows, o, "st", "c"), "c"), "weights fitted on other folds + clip")


def i18_segment_stack(rows, base, events, srcs=None, tag="18 nested stacking, early/late weights"):
    """Nested stacking with separate weights for days 1-6 and 7-13."""
    mem = pd.read_csv(x.OUT / "members_oof.csv", dtype={"batch_id": str})
    srcs = srcs or ["states", "glc", "feeds", "mech", "states_fp", "glc_fp", "feeds_fp", "mech_fp", "raw"]
    o = mem.copy()
    seg = np.where(o.day.to_numpy() <= 6, 0, 1)
    for v in TARGETS:
        P = np.column_stack([o[f"{s}|{v}"].to_numpy() for s in srcs])
        y = o["obs_" + v].to_numpy()
        out = np.zeros(len(o))
        for f in range(h3.N_FOLDS):
            for sg in (0, 1):
                tr = (o.fold != f).to_numpy() & (seg == sg)
                te = (o.fold == f).to_numpy() & (seg == sg)
                w = _nested_weights(P[tr], y[tr], None)
                out[te] = np.exp(np.log(P[te]) @ w)
        o[f"st|{v}"] = out
    o = x.floor_tcd(o, "st")
    x.log_trial(tag, x.score(finish(rows, o, "st", "c"), "c"), "weights fitted on other folds + clip")
    return o


CORES["monod"] = ("core_monod", "monod")


def i19_monod(rows, base, events):
    """New mechanistic core (two-substrate Monod, hidden feed-B nutrient) through v3 + FPCA."""
    r_m, p = core_parts("monod", events)
    mem = pd.read_csv(x.OUT / "members_oof.csv", dtype={"batch_id": str})
    assert (p.batch_id.to_numpy() == mem.batch_id.to_numpy()).all()
    for v in TARGETS:
        mem[f"monod|{v}"] = p[f"fin|{v}"].to_numpy()
        mem[f"monod_fp|{v}"] = p[f"fps|{v}"].to_numpy()
    x.log_trial("19a v3 on Monod core", x.score(mem, "monod"))
    mem.to_csv(x.OUT / "members_oof.csv", index=False)
    srcs = ["states", "glc", "feeds", "mech", "monod", "states_fp", "glc_fp", "feeds_fp", "mech_fp", "monod_fp", "raw"]
    i16 = i18_segment_stack(rows, base, events, srcs=srcs, tag="19b stacking incl. Monod core")


CORES["monod_sens"] = ("core_monod_sens", "monod_sens")


def i20_monod_sens(rows, base, events):
    """Sensor-informed core: per-batch mu_max / lactate yield / glucose yield from
    base-volume and bolus sensors, re-simulated (mixed-effects style)."""
    _, p = core_parts("monod_sens", events)
    mem = pd.read_csv(x.OUT / "members_oof.csv", dtype={"batch_id": str})
    for v in TARGETS:
        mem[f"msens|{v}"] = p[f"fin|{v}"].to_numpy()
        mem[f"msens_fp|{v}"] = p[f"fps|{v}"].to_numpy()
    x.log_trial("20a v3 on sensor-informed core", x.score(mem, "msens"))
    mem.to_csv(x.OUT / "members_oof.csv", index=False)
    srcs = ["states", "glc", "feeds", "mech", "monod", "msens", "states_fp", "glc_fp", "feeds_fp", "mech_fp",
            "monod_fp", "msens_fp", "raw"]
    o = i18_segment_stack(rows, base, events, srcs=srcs, tag="20b final stack (all cores)")
    o.to_csv(x.OUT / "final_stack_oof.csv", index=False)
