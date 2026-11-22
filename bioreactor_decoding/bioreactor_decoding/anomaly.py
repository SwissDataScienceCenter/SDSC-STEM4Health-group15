"""Anomaly detection from normal batches only (Triantafyllia, branch anomalies_detection).

A port of notebooks/anomaly_v7.ipynb, which is the narrative and calls these functions. The
computations are the notebook's, unchanged; only state the notebook kept in globals is passed
explicitly. Hybrid v7 is the model of normal behaviour: its out-of-fold residuals
log(measured / predicted) describe how far a normal batch deviates from its forecast.

  1. residuals      the v7 blend of its four members' out-of-fold predictions (7 folds)
  2. batch level    D1 (RMS of robust z) and PCA T2 / SPE, each batch against the other 48
  3. cell level     spikes and persistent deviations, at a standard and a strict level
  4. injected faults  detection rates, since no rejected batch exists
  5. alarm share    share of real alarms when bad batches are rare
  6. test inputs    the 12 test batches have no measurements: their inputs only

The v7 files (members_oof_v7c.csv, blend_weights.csv) are read as committed at hybrid_v7.ref in
params.yaml, the commit the hybrid_v7 stage runs.

    python -m bioreactor_decoding.anomaly
-> reports/anomaly/*.csv, reports/metrics/anomaly.json, reports/figures/anomaly_v7_*.png
"""

import io
import json
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from bioreactor_decoding.config import FIGURES_DIR, METRICS_DIR, REPORTS_DIR, load_params
from bioreactor_decoding.dataset import RAW_PATH, TARGETS, TEMPLATE_PATH
from bioreactor_decoding.modeling.external import git_show

app = typer.Typer()
ANOMALY_DIR = REPORTS_DIR / "anomaly"
T = TARGETS
MEM = ["pf3a", "a3i_fp", "states", "tp"]  # the four members of hybrid v7
V7_MEMBERS = "outputs/stack_v5/cv/members_oof_v7c.csv"
V7_WEIGHTS = "outputs/hybrid_v7/blend_weights.csv"


# --- 1. residuals of the normal-behaviour model -------------------------------------------


def load_v7(ref: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """v7 members' out-of-fold predictions and blend weights, as committed at `ref`."""
    ref = ref or str(load_params()["hybrid_v7"]["ref"])
    m = pd.read_csv(io.StringIO(git_show(ref, V7_MEMBERS)), dtype={"batch_id": str})
    W = pd.read_csv(io.StringIO(git_show(ref, V7_WEIGHTS)), index_col=[0, 1])
    return m, W


def blend(m: pd.DataFrame, W: pd.DataFrame) -> pd.DataFrame:
    """The v7 blend (weighted geometric mean per variable and day range), no finishing steps."""
    P = pd.DataFrame(index=m.index)
    for v in T:
        out = np.zeros(len(m))
        for seg, days in (("days 1-6", range(1, 7)), ("days 7-13", range(7, 14))):
            w = W.loc[(v, seg), MEM].to_numpy(float)
            mk = m.day.isin(days).to_numpy()
            out[mk] = np.exp(np.column_stack([np.log(m.loc[mk, f"{s}|{v}"]) for s in MEM]) @ w)
        P[v] = out
    P["tcd"] = np.maximum(P.tcd, P.vcd)
    return P


def blend_mape(m: pd.DataFrame, P: pd.DataFrame) -> dict[str, float]:
    return {v: 100 * np.mean(np.abs(P[v] / m["obs_" + v] - 1)) for v in T}


def residuals(m: pd.DataFrame, P: pd.DataFrame):
    """(ids, Obs, Pred, R, RV): log measured and predicted, 49 batches x 52 (4 variables, 13
    days); R = Obs - Pred, and RV the 13 residual columns of each variable."""
    ids = list(dict.fromkeys(m.batch_id))

    def to_matrix(values):
        df = pd.DataFrame({"batch_id": m.batch_id, "day": m.day, **values})
        return np.column_stack(
            [df.pivot(index="batch_id", columns="day", values=v).loc[ids].to_numpy() for v in T]
        )

    Obs = np.log(to_matrix({v: m["obs_" + v] for v in T}))
    Pred = np.log(to_matrix({v: P[v] for v in T}))
    R = Obs - Pred
    RV = {v: R[:, 13 * j : 13 * (j + 1)] for j, v in enumerate(T)}
    return ids, Obs, Pred, R, RV


# --- 2. batch-level detectors -------------------------------------------------------------


def fit_norm(Rm, k=3):
    mu = np.median(Rm, 0)
    sd = np.maximum(1.4826 * np.median(np.abs(Rm - mu), 0), 0.5 * Rm.std(0) + 1e-6)
    Z = (Rm - mu) / sd
    _, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    return {"mu": mu, "sd": sd, "zm": Z.mean(0), "V": Vt[:k], "lam": (S[:k] ** 2) / (len(Rm) - 1)}


def score(M, r):
    z = (r - M["mu"]) / M["sd"]
    zc = z - M["zm"]
    t = zc @ M["V"].T
    return (
        np.sqrt(np.mean(np.clip(z, -6, 6) ** 2, axis=1)),
        np.sum(t**2 / M["lam"], 1),
        np.sum((zc - t @ M["V"]) ** 2, 1),
    )


def batch_scores(R, ids):
    """D1, T2, SPE of each batch against the other 48, the 95% thresholds, and the flags."""
    S_batch = np.zeros((len(ids), 3))
    for i in range(len(ids)):
        a = score(fit_norm(np.delete(R, i, 0)), R[i : i + 1])
        S_batch[i] = [a[0][0], a[1][0], a[2][0]]
    THR = np.quantile(S_batch, 0.95, axis=0)
    flag = (S_batch > THR).any(axis=1)
    table = pd.DataFrame(S_batch, index=ids, columns=["D1", "T2", "SPE"]).assign(flagged=flag)
    return table, THR


# --- 3. cell-level rules --------------------------------------------------------------------

LEVELS = {"strict": (6.5, 3.5, 3), "standard": (5.5, 3.0, 3)}  # (spike z, persistent z, days)
PERS_VARS = ["vcd", "substrate", "byproduct"]


def loo_z(Rv, i, row=None):
    others = np.delete(Rv, i, 0)
    mu = np.median(others, 0)
    sd = np.maximum(1.4826 * np.median(np.abs(others - mu), 0), 0.5 * others.std(0))
    return ((Rv[i] if row is None else row) - mu) / sd


def find_deviations(z, spike, pers, ndays):
    """z: dict variable -> 13 z-values. Returns (type, variable, first day, last day, mean z)."""
    found = []
    for v in T:
        for d in range(13):
            nb = [abs(z[v][k]) for k in (d - 1, d + 1) if 0 <= k < 13]
            if abs(z[v][d]) >= spike and max(nb) < pers:
                found.append(("spike", v, d + 1, d + 1, z[v][d]))
    for v in PERS_VARS:
        d = 0
        while d < 13:
            if abs(z[v][d]) >= pers:
                e = d
                while (
                    e + 1 < 13
                    and abs(z[v][e + 1]) >= pers
                    and np.sign(z[v][e + 1]) == np.sign(z[v][d])
                ):
                    e += 1
                if e - d + 1 >= ndays:
                    found.append(("persistent", v, d + 1, e + 1, float(np.mean(z[v][d : e + 1]))))
                d = e + 1
            else:
                d += 1
    return found


def cell_deviations(RV, ids) -> pd.DataFrame:
    """Every deviation at the standard level, and whether the strict level keeps it."""
    rows = []
    for i, b in enumerate(ids):
        z = {v: loo_z(RV[v], i) for v in T}
        for typ, v, d0, d1, zv in find_deviations(z, *LEVELS["standard"]):
            strict = any(
                x[0] == typ and x[1] == v and x[2] <= d1 and x[3] >= d0
                for x in find_deviations(z, *LEVELS["strict"])
            )
            rows.append({
                "batch": b, "type": typ, "variable": v,
                "days": str(d0) if d0 == d1 else f"{d0}-{d1}",
                "mean z": round(zv, 1),
                "measured vs predicted (%)": round(100 * (np.exp(RV[v][i, d0 - 1 : d1].mean()) - 1)),
                "strict level": "yes" if strict else "no",
            })  # fmt: skip
    dev = pd.DataFrame(rows)
    # sorted for display as in the notebook (the original key used dtype == object, which is
    # not true for pandas >= 3 string columns; this is the same ordering)
    return dev.sort_values(
        ["type", "mean z"],
        key=lambda s: s if not pd.api.types.is_numeric_dtype(s) else -s.abs(),
    ).reset_index(drop=True)


# --- 4. evaluation with injected faults -----------------------------------------------------

KINDS = ["growth stall", "lactate overproduction", "glucose feed failure", "single assay outlier"]
SEVS = [0.1, 0.2, 0.3, 0.5]
COLS = ["batch score", "cell rules, standard", "cell rules, strict"]


def fault(kind, sev, rng):
    f = np.zeros(52)

    def ramp(d0):
        return np.clip((np.arange(1, 14) - d0) / (13 - d0), 0, 1)

    if kind == "growth stall":
        f[0:13] = -sev * ramp(4)
        f[13:26] = -sev * ramp(4)
        f[26:39] = 0.6 * sev * ramp(4)
    if kind == "lactate overproduction":
        f[39:52] = sev * ramp(3)
    if kind == "glucose feed failure":
        f[26:39] = -sev * (np.arange(1, 14) >= 8)
    if kind == "single assay outlier":
        f[rng.integers(0, 52)] = sev * rng.choice([-1, 1])
    return f


def alarms(i, f, R, RV, THR):
    a = score(fit_norm(np.delete(R, i, 0)), (R[i] + f)[None])
    batch = bool(a[0][0] > THR[0] or a[1][0] > THR[1] or a[2][0] > THR[2])
    z = {v: loo_z(RV[v], i, RV[v][i] + f[13 * j : 13 * (j + 1)]) for j, v in enumerate(T)}
    return [
        batch,
        bool(find_deviations(z, *LEVELS["standard"])),
        bool(find_deviations(z, *LEVELS["strict"])),
    ]


def fault_table(R, RV, THR, ids) -> pd.DataFrame:
    """Share of batches that raise an alarm, per fault and size; 'no fault' = false alarms."""
    rng = np.random.default_rng(0)
    res = {
        ("no fault", ""): np.mean(
            [alarms(i, np.zeros(52), R, RV, THR) for i in range(len(ids))], axis=0
        )
    }
    for kind in KINDS:
        for sev in SEVS:
            reps = 3 if kind == "single assay outlier" else 1
            res[(kind, sev)] = np.mean(
                [alarms(i, fault(kind, sev, rng), R, RV, THR)
                 for _ in range(reps) for i in range(len(ids))],
                axis=0,
            )  # fmt: skip
    return pd.DataFrame(res, index=COLS).T


# --- 5. share of real alarms when bad batches are rare --------------------------------------


def alarm_share(tab: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in COLS:
        tpr, fpr = np.mean([tab.loc[(k, 0.5), col] for k in KINDS]), tab.loc[("no fault", ""), col]
        for p in (0.001, 0.01, 0.05):
            real, false = 1000 * p * tpr, 1000 * (1 - p) * fpr
            rows.append({
                "detector": col, "bad batches": f"{p:.1%}", "detection": f"{tpr:.0%}",
                "false-alarm rate": f"{fpr:.0%}", "alarms per 1000 batches": round(real + false),
                "real alarms": round(real, 1),
                "share of alarms that are real": f"{real / (real + false):.0%}",
            })  # fmt: skip
    return pd.DataFrame(rows)


# --- 6. input check of the 12 test batches --------------------------------------------------

FEEDS = ["feed_substrate", "feed_b", "feed_c", "feed_d"]


def input_features(raw_path: Path = RAW_PATH, template_path: Path = TEMPLATE_PATH):
    """(feat, train, test): day-0 values, feeds, volumes and the v3 soft sensors per batch."""
    ev = pd.read_csv(raw_path, dtype={"batch_id": str})
    tp = pd.read_csv(template_path, dtype={"batch_id": str})
    test = list(dict.fromkeys(tp.batch_id))
    allb = list(dict.fromkeys(ev.batch_id))
    train = [b for b in allb if b not in test]

    Vol = {}  # unlogged volume (adapted from hybrid_v3.volume_balance)
    for bid, g in ev.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        st = g[g.volume.notna()]
        cum = g[FEEDS].fillna(0).sum(axis=1).cumsum()
        Vol[bid] = {
            int(round(s.elapsed_time)): s.volume - st.volume.iloc[0] - cum.loc[i]  # noqa: RUF046
            for i, s in st.iterrows()
        }
    Vol = pd.DataFrame(Vol).T.sort_index(axis=1)
    late = [k for k in Vol.columns if k >= 6]
    sample = -np.polyfit(late, Vol[late].mean().to_numpy(), 1)[0]
    A = Vol.add(sample * np.array(Vol.columns, dtype=float), axis=1)
    pre = A[[1, 2]].mean(axis=1)
    base = pd.DataFrame({
        "base_to3": A[3] - pre, "base_to4": A[4] - pre, "base_to5": A[5] - pre,
        "base_tot": A[late].mean(axis=1) - pre,
    })  # fmt: skip

    rows = {}
    for bid, g in ev.groupby("batch_id"):
        g = g.sort_values("elapsed_time")
        st = g[g.volume.notna()]
        fs = g[g.feed_substrate.notna()]
        first = g[g.vcd.notna()].iloc[0]

        def per_volume(lo, hi):  # called in this iteration only, so fs and st are current
            x = fs[(fs.elapsed_time > lo) & (fs.elapsed_time < hi)]  # noqa: B023
            return sum(
                r.feed_substrate / st[st.elapsed_time <= r.elapsed_time].volume.iloc[-1]  # noqa: B023
                for _, r in x.iterrows()
            )

        rows[bid] = {
            "vcd0": first.vcd, "s0": first.substrate, "l0": first.byproduct, "v0": first.volume,
            "via0": first.vcd / first.tcd, **{"tot_" + f: g[f].fillna(0).sum() for f in FEEDS},
            "V_end": st.volume.iloc[-1], "A_early": per_volume(6.0, 10.5),
            "A11": per_volume(10.5, 12.0), "A12": per_volume(12.0, 14.0),
        }  # fmt: skip
    feat = pd.DataFrame(rows).T.astype(float).join(base).loc[allb]
    return feat, train, test


def fit_pca(X, k=4):
    m_, s_ = X.mean(), X.std().replace(0, 1)
    _, S, Vt = np.linalg.svd(((X - m_) / s_).to_numpy(), full_matrices=False)
    return m_, s_, Vt[:k], (S[:k] ** 2) / (len(X) - 1)


def pca_score(M, x):
    m_, s_, V, lam = M
    z = ((x - m_) / s_).to_numpy()
    t = z @ V.T
    return float(np.sum(t**2 / lam)), float(np.sum((z - t @ V) ** 2))


def input_check(feat, train, test):
    """(table, thresholds T2/SPE, top-3 largest |z| among the labelled batches)."""
    Xtr = feat.loc[train]
    Zf = (feat - Xtr.mean()) / Xtr.std().replace(0, 1)
    lo, hi = Xtr.min(), Xtr.max()
    loo = np.array([pca_score(fit_pca(Xtr.drop(b)), Xtr.loc[b]) for b in train])
    thr2 = np.quantile(loo, 0.95, axis=0)
    M_all = fit_pca(Xtr)

    tab_in = []
    for b in test:
        j = Zf.loc[b].abs().idxmax()
        t2, spe = pca_score(M_all, feat.loc[b])
        outside = [
            col for col in feat.columns
            if feat.loc[b, col] < lo[col] - 1e-12 or feat.loc[b, col] > hi[col] + 1e-12
        ]  # fmt: skip
        tab_in.append({
            "batch": b, "largest abs z": round(abs(Zf.loc[b, j]), 2), "feature": j,
            "outside training range": ", ".join(outside) or "-", "T2": round(t2, 1),
            "SPE": round(spe, 1), "flag": "yes" if (t2 > thr2[0] or spe > thr2[1]) else "",
        })  # fmt: skip
    table = pd.DataFrame(tab_in).sort_values("largest abs z", ascending=False)
    top3 = Zf.loc[train].abs().max(axis=1).sort_values(ascending=False).head(3).round(2)
    return table.reset_index(drop=True), thr2, top3.to_dict()


# --- figures --------------------------------------------------------------------------------

EXAMPLES = [("Batch34", "byproduct"), ("Batch29", "byproduct"), ("Batch46", "substrate"),
            ("Batch4", "byproduct")]  # fmt: skip


def cell_level_figure(Obs, Pred, ids):
    OBS = {v: np.exp(Obs[:, 13 * j : 13 * (j + 1)]) for j, v in enumerate(T)}
    PRD = {v: np.exp(Pred[:, 13 * j : 13 * (j + 1)]) for j, v in enumerate(T)}
    days = np.arange(1, 14)
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.3))
    for ax, (b, v) in zip(axs, EXAMPLES):
        for i in range(len(ids)):
            ax.plot(days, OBS[v][i], color="#dddddd", lw=0.7)
        ax.fill_between(days, np.percentile(OBS[v], 5, 0), np.percentile(OBS[v], 95, 0),
                        color="#9ecae1", alpha=0.5)  # fmt: skip
        i = ids.index(b)
        ax.plot(days, OBS[v][i], color="#c0392b", lw=2, label="measured")
        ax.plot(days, PRD[v][i], "k--", lw=1.2, label="v7 prediction")
        ax.set_title(f"{b}, {v}", fontsize=10)
        ax.set_xlabel("day")
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    return fig


def detection_figure(tab):
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.2), sharey=True)
    style = {"batch score": ("#7f8c8d", "--"), "cell rules, standard": ("#c0392b", "-"),
             "cell rules, strict": ("#2c7fb8", "-")}  # fmt: skip
    for ax, kind in zip(axs, KINDS):
        for col, (colour, ls) in style.items():
            ax.plot(SEVS, [tab.loc[(kind, s), col] for s in SEVS], "o" + ls, color=colour,
                    label=col)  # fmt: skip
        ax.set_title(kind, fontsize=10)
        ax.set_xticks(SEVS)
        ax.set_xlabel("fault size (log change)")
    axs[0].set_ylabel("share of faulty batches flagged")
    axs[0].legend(fontsize=8)
    plt.tight_layout()
    return fig


# --- the whole analysis ---------------------------------------------------------------------


def run(ref: str | None = None) -> dict:
    """Every result of the analysis, in the notebook's order."""
    m, W = load_v7(ref)
    P = blend(m, W)
    ids, Obs, Pred, R, RV = residuals(m, P)
    batch, THR = batch_scores(R, ids)
    tab = fault_table(R, RV, THR, ids)
    feat, train, test = input_features()
    inputs, thr2, top3 = input_check(feat, train, test)
    return {
        "m": m, "P": P, "mape": blend_mape(m, P), "ids": ids, "Obs": Obs, "Pred": Pred,
        "R": R, "RV": RV, "batch": batch, "THR": THR, "dev": cell_deviations(RV, ids),
        "tab": tab, "share": alarm_share(tab), "inputs": inputs, "thr2": thr2, "top3": top3,
    }  # fmt: skip


def metrics(r: dict) -> dict:
    tab, dev = r["tab"], r["dev"]
    return {
        "blend_mape": {
            **{v: round(float(x), 3) for v, x in r["mape"].items()},
            "score": round(float(np.mean(list(r["mape"].values()))), 3),
        },
        "residual_spread_log": round(float(r["R"].std(0).mean()), 4),
        "batch_level": {
            "thresholds": dict(zip(["D1", "T2", "SPE"], np.round(r["THR"], 3).tolist())),
            "flagged": int(r["batch"]["flagged"].sum()),
            "batches": len(r["ids"]),
        },
        "cell_level": {
            "flagged_standard": int(dev["batch"].nunique()),
            "flagged_strict": int(dev.loc[dev["strict level"] == "yes", "batch"].nunique()),
        },
        "false_alarm_rate": {c: round(float(tab.loc[("no fault", ""), c]), 3) for c in COLS},
        "detection_size_0.5": {
            c: round(float(np.mean([tab.loc[(k, 0.5), c] for k in KINDS])), 3) for c in COLS
        },
        "test_inputs": {
            "thresholds": dict(zip(["T2", "SPE"], np.round(r["thr2"], 2).tolist())),
            "flagged": r["inputs"].loc[r["inputs"]["flag"] == "yes", "batch"].tolist(),
        },
    }


@app.command()
def main(output_dir: Path = ANOMALY_DIR, figures_dir: Path = FIGURES_DIR):
    r = run()
    output_dir.mkdir(parents=True, exist_ok=True)
    r["batch"].rename_axis("batch").round(4).to_csv(output_dir / "batch_scores.csv")
    r["dev"].to_csv(output_dir / "cell_deviations.csv", index=False)
    r["tab"].rename_axis(["fault", "size"]).round(4).to_csv(output_dir / "fault_detection.csv")
    r["share"].to_csv(output_dir / "alarm_share.csv", index=False)
    r["inputs"].to_csv(output_dir / "test_input_check.csv", index=False)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in (
        ("anomaly_v7_cell_level.png", cell_level_figure(r["Obs"], r["Pred"], r["ids"])),
        ("anomaly_v7_detection.png", detection_figure(r["tab"])),
    ):
        fig.savefig(figures_dir / name, dpi=130)
        plt.close(fig)
    out = metrics(r)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / "anomaly.json").write_text(json.dumps(out, indent=2) + "\n")
    logger.success(f"anomaly: {json.dumps(out)}")


if __name__ == "__main__":
    app()
