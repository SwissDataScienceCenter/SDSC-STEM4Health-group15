"""v5 experiments: up to 20 new ideas on top of hybrid v3, all scored with the
same 7-fold grouped CV over whole batches (leaderboard metric = mean of 4 MAPEs).

Every trial is appended to outputs/experiments_v5/trials.csv.

    python src/experiments_v5.py IDEA [IDEA ...]

Post-hoc ideas (stage 3) are fitted per fold on the other folds' out-of-fold
predictions only, so the held-out fold never informs its own correction.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import experiments_v4b as v4b  # noqa: E402
import hybrid_v3 as h3  # noqa: E402

ROOT = h3.ROOT
OUT = ROOT / "outputs" / "experiments_v5"
TARGETS = h3.TARGETS
T1, T2 = 0.965, 1.96  # exact glucose thresholds of the bolus controller (see censor rule below)


# ----------------------------------------------------------------- data
def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    events = pd.read_csv(h3.DATA_DIR / "bioreaction_remaining.csv", dtype={"batch_id": str})
    template = pd.read_csv(h3.DATA_DIR / "bioreaction_template.csv", dtype={"batch_id": str})
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    F = h3.feed_features(events)
    sens_batch, sens_day, _ = h3.sensor_features(events)
    rows = v4b.add_features(h3.build_rows(F, sens_batch, sens_day, h3.load_stage1(h3.STAGE1_DIR)), events)
    fold_of = {}
    for f, (_, te) in enumerate(GroupKFold(h3.N_FOLDS).split(np.arange(len(train_ids)), groups=np.arange(len(train_ids)))):
        for i in te:
            fold_of[train_ids[i]] = f
    rows["fold"] = rows.batch_id.map(fold_of)
    rows = add_censor(rows)
    return rows.reset_index(drop=True), events


def add_censor(rows: pd.DataFrame) -> pd.DataFrame:
    """Glucose interval implied by the bolus controller.

    The controller checks glucose on days 7-9 and 11-12 and adds a bolus about
    0.5 d later: none if glucose >= T2 (days 7-9 only), 0.004*V if T1 <= glucose < T2,
    0.006*V if glucose < T1. Every training sample obeys this rule exactly.
    """
    lo = np.zeros(len(rows))
    hi = np.full(len(rows), np.inf)
    A, day = rows.A.to_numpy(), rows.day.to_numpy()
    big, small = A > 0.005, (A > 0.001) & (A <= 0.005)
    hi[big] = T1
    lo[small], hi[small] = T1, T2
    lo[np.isin(day, [7, 8, 9]) & (A <= 0.001)] = T2
    rows = rows.copy()
    rows["g_lo"], rows["g_hi"] = lo, hi
    return rows


# ----------------------------------------------------------------- base CV
def run_parts(rows: pd.DataFrame, hyb_cls=None, dd_cls=None) -> pd.DataFrame:
    """Out-of-fold predictions of both v3 branches and the final blend."""
    hyb_cls = hyb_cls or v4b._ORIG_HYB
    dd_cls = dd_cls or v4b._ORIG_DD
    out = rows[["batch_id", "t_key", "day", "fold"] + ["obs_" + v for v in TARGETS]].copy()
    for k in ("hyb", "dd", "fin"):
        for v in TARGETS:
            out[f"{k}|{v}"] = np.nan
    for f in range(h3.N_FOLDS):
        tr, te = (rows.fold != f).to_numpy(), (rows.fold == f).to_numpy()
        h = hyb_cls().fit(rows[tr]).predict(rows[te])
        d = dd_cls().fit(rows[tr]).predict(rows[te])
        fin = h3.finalize(h, d)
        for v in TARGETS:
            out.loc[te, f"hyb|{v}"] = h[v].to_numpy()
            out.loc[te, f"dd|{v}"] = d[v].to_numpy()
            out.loc[te, f"fin|{v}"] = fin[v].to_numpy()
    return out


def score(out: pd.DataFrame, prefix: str = "fin") -> pd.Series:
    s = pd.Series({v: 100 * np.mean(np.abs(out[f"{prefix}|{v}"] / out["obs_" + v] - 1)) for v in TARGETS})
    s["leaderboard_score"] = s.mean()
    return s


def log_trial(idea: str, s: pd.Series, note: str = "") -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "trials.csv"
    row = pd.DataFrame([{"idea": idea, **{k: round(float(v), 3) for k, v in s.items()}, "note": note}])
    row.to_csv(path, mode="a", header=not path.exists(), index=False)
    print(f"{idea:28s} " + "  ".join(f"{v}={s[v]:.3f}" for v in TARGETS)
          + f"  SCORE={s['leaderboard_score']:.3f}  {note}", flush=True)


def base_parts(rows: pd.DataFrame) -> pd.DataFrame:
    path = OUT / "base_parts.csv"
    if path.exists():
        return pd.read_csv(path, dtype={"batch_id": str})
    OUT.mkdir(parents=True, exist_ok=True)
    p = run_parts(rows)
    p.to_csv(path, index=False)
    return p


def floor_tcd(out: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out[f"{prefix}|tcd"] = np.maximum(out[f"{prefix}|tcd"], out[f"{prefix}|vcd"])
    return out


def copy_pred(out: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    for v in TARGETS:
        out[f"{dst}|{v}"] = out[f"{src}|{v}"]
    return out


# ----------------------------------------------------------------- ideas
def censored_point(pred, lo, hi, resid, obs_weighted=True):
    """MAPE-optimal point given pred, an interval for obs, and an empirical
    distribution of log(obs/pred) residuals: weighted median (weights 1/obs)
    of the residual-implied obs values that fall inside the interval."""
    cand = pred * np.exp(resid)
    m = (cand >= lo) & (cand < hi)
    if m.sum() < 5:
        return float(np.clip(pred, lo, np.nextafter(hi, 0) if np.isfinite(hi) else pred))
    c = np.sort(cand[m])
    w = 1.0 / c if obs_weighted else np.ones_like(c)
    cw = np.cumsum(w)
    return float(c[np.searchsorted(cw, 0.5 * cw[-1])])


def idea_censor(rows, out, src="fin", dst="cen", days_pool=2):
    """Glucose: combine the prediction with the exact bolus-rule interval."""
    out = copy_pred(out.copy(), src, dst)
    p = out[f"{src}|substrate"].to_numpy()
    y = out.obs_substrate.to_numpy()
    r_all = np.log(y / p)
    lo, hi = rows.g_lo.to_numpy(), rows.g_hi.to_numpy()
    day = out.day.to_numpy()
    newp = p.copy()
    for f in range(h3.N_FOLDS):
        tr, te = (out.fold != f).to_numpy(), (out.fold == f).to_numpy()
        for i in np.where(te & ((lo > 0) | np.isfinite(hi)))[0]:
            pool = tr & (np.abs(day - day[i]) <= days_pool)
            newp[i] = censored_point(p[i], lo[i], hi[i], r_all[pool])
    out[f"{dst}|substrate"] = newp
    return out


def idea_clip(rows, out, src="fin", dst="clip"):
    out = copy_pred(out.copy(), src, dst)
    p = out[f"{src}|substrate"].to_numpy()
    out[f"{dst}|substrate"] = np.clip(p, rows.g_lo.to_numpy(), np.minimum(rows.g_hi.to_numpy(), 1e9) - 1e-6)
    return out


def ridge_fit(X, y, alpha):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1
    Z = (X - mu) / sd
    Z1 = np.c_[np.ones(len(Z)), Z]
    P = alpha * np.eye(Z1.shape[1])
    P[0, 0] = 0
    b = np.linalg.solve(Z1.T @ Z1 + P, Z1.T @ y)
    return lambda Xn: np.c_[np.ones(len(Xn)), (Xn - mu) / sd] @ b


def idea_stage3_ridge(rows, out, var, cols, days, src="fin", dst="s3", alpha=20.0, per_day=True):
    """Post-hoc ridge correction of log(obs/pred) for one variable on some days."""
    out = copy_pred(out.copy(), src, dst)
    p = out[f"{src}|{var}"].to_numpy()
    r = np.log(out["obs_" + var].to_numpy() / p)
    X = rows[cols].to_numpy(float)
    day = out.day.to_numpy()
    corr = np.zeros(len(out))
    for f in range(h3.N_FOLDS):
        tr, te = (out.fold != f).to_numpy(), (out.fold == f).to_numpy()
        groups = [[d] for d in days] if per_day else [list(days)]
        for g in groups:
            mtr, mte = tr & np.isin(day, g), te & np.isin(day, g)
            if mte.sum() == 0:
                continue
            Xd = np.c_[X, day] if not per_day else X
            fn = ridge_fit(Xd[mtr], r[mtr], alpha)
            corr[mte] = fn(Xd[mte])
    out[f"{dst}|{var}"] = p * np.exp(corr)
    return floor_tcd(out, dst)


def idea_gp_residual(rows, out, var, feat_cols, src="fin", dst="gp", ls_t=2.0, ls_x=2.0, noise=1.0, shrink=1.0):
    """Gaussian-process smoother of the residual log(obs/pred) across batches:
    k((x,t),(x',t')) = exp(-|x-x'|^2/2ls_x^2) * exp(-(t-t')^2/2ls_t^2)."""
    out = copy_pred(out.copy(), src, dst)
    p = out[f"{src}|{var}"].to_numpy()
    r = np.log(out["obs_" + var].to_numpy() / p)
    X = rows[feat_cols].to_numpy(float)
    t = out.day.to_numpy(float)
    corr = np.zeros(len(out))
    for f in range(h3.N_FOLDS):
        tr, te = (out.fold != f).to_numpy(), (out.fold == f).to_numpy()
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1
        Z = (X - mu) / sd
        rm = r[tr].mean()
        var_r = r[tr].var()

        def K(a, b, ta, tb):
            d2 = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
            return np.exp(-0.5 * d2 / ls_x**2) * np.exp(-0.5 * (ta[:, None] - tb[None, :]) ** 2 / ls_t**2)

        Ktr = K(Z[tr], Z[tr], t[tr], t[tr]) + noise * np.eye(tr.sum())
        alpha = np.linalg.solve(Ktr, r[tr] - rm)
        corr[te] = rm + shrink * K(Z[te], Z[tr], t[te], t[tr]) @ alpha
    out[f"{dst}|{var}"] = p * np.exp(corr)
    return floor_tcd(out, dst)


def per_day_scale(out, src, dst):
    """MAPE-optimal per-(variable, day) rescaling, fitted on the other folds."""
    out = copy_pred(out.copy(), src, dst)
    for v in TARGETS:
        p = out[f"{src}|{v}"].to_numpy()
        y = out["obs_" + v].to_numpy()
        newp = p.copy()
        for f in range(h3.N_FOLDS):
            tr, te = (out.fold != f).to_numpy(), (out.fold == f).to_numpy()
            for d in np.unique(out.day):
                mtr, mte = tr & (out.day == d).to_numpy(), te & (out.day == d).to_numpy()
                newp[mte] = p[mte] * h3.mape_optimal_scale(p[mtr], y[mtr])
        out[f"{dst}|{v}"] = newp
    return floor_tcd(out, dst)


def geo_blend(out, srcs, weights, dst):
    out = out.copy()
    w = np.asarray(weights, float) / np.sum(weights)
    for v in TARGETS:
        out[f"{dst}|{v}"] = np.exp(sum(wi * np.log(out[f"{s}|{v}"].to_numpy()) for s, wi in zip(srcs, w)))
    return floor_tcd(out, dst)


# ----------------------------------------------------------------- main
def main(argv):
    t0 = time.time()
    rows, events = build()
    base = base_parts(rows)
    assert (base.batch_id.to_numpy() == rows.batch_id.to_numpy()).all()
    if "base" in argv:
        log_trial("0 baseline v3", score(base), "reference")
    import experiments_v5_ideas as ideas  # noqa: E402
    for name in argv:
        if name == "base":
            continue
        getattr(ideas, name)(rows, base, events)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
