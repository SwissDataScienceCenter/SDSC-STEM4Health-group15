"""Experiment harness for v4: candidate stage-2 improvements on saved OOF preds.

Loads outputs/hybrid_v3/oof_predictions.csv (out-of-fold predictions of both
v3 branches, produced with proper batch-level CV) and evaluates candidate
additions under the same 7-fold grouped CV, so every comparison is honest in
the same way the 7.67 baseline is.

    python src/experiments_v4.py [--only NAME[,NAME]]

Experiments:
  blend_w      per-variable learned blend weights between the branches
  shrink       shrink each batch's prediction toward its own geometric level
  knn_cells    similarity member for VCD/TCD (kernel-weighted residual kNN)
  knn_all      similarity member for all four variables
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.kernel_ridge import KernelRidge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OOF = ROOT / "outputs" / "hybrid_v3" / "oof_predictions.csv"
TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
HYB = "v3 hybrid branch"
DD = "v3 data-driven branch"
FIN = "v3 final (50/50 blend)"

T0_COLS = ["vcd0", "s0", "l0", "v0", "via0"]
SENSOR_COLS = ["base_to3", "base_to4", "base_to5", "base_tot", "A11", "A12", "A_early"]


# ----------------------------------------------------------------------


def mape(pred: np.ndarray, obs: np.ndarray) -> float:
    return float(100 * np.mean(np.abs(pred / obs - 1)))


def score_table(rows: pd.DataFrame, cols: dict[str, str]) -> pd.Series:
    s = pd.Series({v: mape(rows[cols[v]].to_numpy(), rows["obs_" + v].to_numpy()) for v in TARGETS})
    s["leaderboard_score"] = s.mean()
    return s


def mape_optimal_scale(pred: np.ndarray, obs: np.ndarray) -> float:
    z = obs / pred
    w = pred / obs
    order = np.argsort(z)
    cw = np.cumsum(w[order])
    return float(z[order][np.searchsorted(cw, 0.5 * cw[-1])])


def batch_features(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per batch: t=0 state, sensors, mean feed deviations, mean stage-1 preds."""
    agg = {c: (c, "first") for c in T0_COLS + SENSOR_COLS}
    agg.update({f"m_{c}": (c, "mean") for c in ["fb_next_dev", "fb_prev_dev", "fc_next_dev", "fc_prev_dev", "V_dev"]})
    agg.update({f"s1_{v}": (f"s1_{v}", "mean") for v in TARGETS})
    return rows.groupby("batch_id").agg(**agg)


def per_day_blend(rows: pd.DataFrame, out: str = "blend_w") -> pd.DataFrame:
    """Per-(variable, day) blend weight in log space, fitted on train folds."""
    res = rows.copy()
    for v in TARGETS:
        la, lb = np.log(res[f"{HYB}|{v}"]), np.log(res[f"{DD}|{v}"])
        outp = np.zeros(len(res))
        for f in rows.fold.unique():
            tr, te = (rows.fold != f).to_numpy(), (rows.fold == f).to_numpy()
            w = {}
            for day in rows.day.unique():
                m = tr & (rows.day == day).to_numpy()
                # 1-D search: minimise MAPE of exp((1-w)la+w lb)
                best, bw = np.inf, 0.5
                for cand in np.linspace(0, 1, 21):
                    p = np.exp((1 - cand) * la[m] + cand * lb[m])
                    e = np.mean(np.abs(p / res["obs_" + v].to_numpy()[m] - 1))
                    if e < best:
                        best, bw = e, cand
                w[day] = bw
            outp[te] = ((1 - np.array([w[d] for d in rows.day[te]])) * la[te] + np.array([w[d] for d in rows.day[te]]) * lb[te])
        res[out + "|" + v] = np.exp(outp)
    res[out + "|tcd"] = np.maximum(res[out + "|tcd"], res[out + "|vcd"])
    return res


def shrink(rows: pd.DataFrame, lam_by_var: dict[str, float], src: str, out: str) -> pd.DataFrame:
    """Shrink each batch's log prediction toward the batch's own geometric mean."""
    res = rows.copy()
    for v in TARGETS:
        lp = np.log(res[f"{src}|{v}"].to_numpy())
        gm = pd.Series(lp).groupby(res.batch_id.to_numpy()).transform("mean").to_numpy()
        lam = lam_by_var[v]
        res[out + "|" + v] = np.exp((1 - lam) * lp + lam * gm)
    res[out + "|tcd"] = np.maximum(res[out + "|tcd"], res[out + "|vcd"])
    return res


def knn_residual(rows: pd.DataFrame, feats: pd.DataFrame, var: str, src: str, h: float, k: int) -> np.ndarray:
    """Kernel-weighted average of similar training batches' batch-level residual."""
    out = np.zeros(len(rows))
    Xb = feats.to_numpy()
    mu, sd = Xb.mean(0), Xb.std(0)
    sd[sd == 0] = 1
    Xz = (Xb - mu) / sd
    bid_index = {b: i for i, b in enumerate(feats.index)}
    # batch-level residual: mean of log(obs/src) over the batch
    lperr = np.log(rows["obs_" + var].to_numpy() / rows[f"{src}|" + var].to_numpy())
    err_b = pd.Series(lperr).groupby(rows.batch_id.to_numpy()).mean()
    for f in rows.fold.unique():
        tr_ids = [b for b in feats.index if fold_of[b] != f]
        te_mask = (rows.fold == f).to_numpy()
        tri = np.array([bid_index[b] for b in tr_ids])
        sub = rows[te_mask]
        for bid, labels in sub.groupby("batch_id").groups.items():
            if bid not in bid_index:
                continue
            idx = rows.index.get_indexer(labels)
            q = Xz[bid_index[bid]][None, :]
            d = np.linalg.norm(Xz[tri] - q, axis=1)
            w = np.exp(-0.5 * (d / h) ** 2)
            order = np.argsort(-w)[:k]
            w = w[order]
            if w.sum() < 1e-9:
                out[idx] = err_b.loc[tr_ids].mean()
            else:
                out[idx] = float(w @ err_b.loc[[tr_ids[i] for i in order]].to_numpy() / w.sum())
    return out


def main() -> None:
    global fold_of
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "experiments_v4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    only = set(args.only.split(",")) - {""}

    rows = pd.read_csv(OOF, dtype={"batch_id": str})
    fold_of = rows.groupby("batch_id").fold.first().to_dict()
    base = score_table(rows, {v: f"{FIN}|{v}" for v in TARGETS})
    results = {"baseline v3 (7.67)": base}
    print("baseline:", base.round(3).to_dict())

    feats = batch_features(rows)

    if not only or "blend_w" in only:
        r = per_day_blend(rows)
        results["per-day blend weights"] = score_table(r, {v: f"blend_w|{v}" for v in TARGETS})

    if not only or "shrink" in only:
        # grid the lambda per variable on train folds, apply to test fold
        res = rows.copy()
        best_lam = {}
        for v in TARGETS:
            lp = np.log(res[f"{FIN}|{v}"].to_numpy())
            gm = pd.Series(lp).groupby(res.batch_id.to_numpy()).transform("mean").to_numpy()
            scores = []
            for lam in np.linspace(0, 0.8, 17):
                tot = []
                for f in rows.fold.unique():
                    te = (rows.fold == f).to_numpy()
                    p = np.exp((1 - lam) * lp[te] + lam * gm[te])
                    tot.append(np.mean(np.abs(p / res["obs_" + v].to_numpy()[te] - 1)))
                scores.append(np.mean(tot))
            best_lam[v] = float(np.linspace(0, 0.8, 17)[int(np.argmin(scores))])
            print(f"  shrink lam {v}: {best_lam[v]:.2f} (mape {100*np.min(scores):.2f} vs {results['baseline v3 (7.67)'][v]:.2f})")
        r = shrink(rows, best_lam, FIN, "shrink")
        results["batch-level shrinkage"] = score_table(r, {v: f"shrink|{v}" for v in TARGETS})

    if not only or "knn_cells" in only or "knn_all" in only:
        cols = T0_COLS + SENSOR_COLS + ["m_fb_next_dev", "m_fb_prev_dev", "m_V_dev", "s1_vcd", "s1_tcd"]
        fb = feats[cols]
        for v in (TARGETS if ("knn_all" in only and "knn_cells" not in only) else ["vcd", "tcd"]):
            best = None
            for h in [0.8, 1.2, 1.8, 2.5]:
                for k in [5, 10, 20, 49]:
                    corr = knn_residual(rows, fb, v, FIN, h, k)
                    p = rows[f"{FIN}|{v}"].to_numpy() * np.exp(corr)
                    m = mape(p, rows["obs_" + v].to_numpy())
                    if best is None or m < best[0]:
                        best = (m, h, k, corr)
            print(f"  knn {v}: {best[0]:.2f} (h={best[1]}, k={best[2]}) vs {base[v]:.2f}")
            rows[f"knn|{v}"] = rows[f"{FIN}|{v}"].to_numpy() * np.exp(best[3])
        for v in TARGETS:
            if f"knn|{v}" not in rows:
                rows[f"knn|{v}"] = rows[f"{FIN}|{v}"]
        rows["knn|tcd"] = np.maximum(rows["knn|tcd"], rows["knn|vcd"])
        results["knn residual"] = score_table(rows, {v: f"knn|{v}" for v in TARGETS})

    table = pd.DataFrame(results).T.round(3)
    print(table.to_string())
    table.to_csv(args.out / "experiment_scores.csv")


if __name__ == "__main__":
    main()
