"""Pictures for the 5-minute Option A talk -> outputs/slides/*.png

  fig1_sensors.png     slide 2: the two hidden sensors in the event log
  fig2_particles.png   slide 3: particles of the Monod core, weighted by the sensors
  fig3_ml_views.png    slide 4: core -> two ML views -> final, plus the FPCA curve shapes
  fig4_results.png     slide 5: held-out error per target, hybrid_v3 vs Option A

Uses the saved CV folds; the example batch in fig2/fig3 is predicted without
its own measurements (fitted on the other folds only).

    python src/plot_slides.py [--batch Batch41]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_v3 as h3  # noqa: E402
import sartorius_model as sm  # noqa: E402

for _m in (h3, sm):
    if not (_m.DATA_DIR / "bioreaction_remaining.csv").exists() and (_m.ROOT / "data" / "bioreaction_remaining.csv").exists():
        _m.DATA_DIR = _m.ROOT / "data"
import core_assim as ca  # noqa: E402
import core_monod_id as cid  # noqa: E402
import experiments_v5 as x  # noqa: E402

ROOT = h3.ROOT
OUT = ROOT / "outputs" / "slides"
TARGETS = h3.TARGETS
INK, INK2, GRID = "#1d1d1b", "#6b6a66", "#e6e5e1"
C_CORE, C_ML, C_FIN, C_OBS, C_V3 = "#eb6834", "#1baf7a", "#2a78d6", "#1d1d1b", "#a9a8a4"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 14,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})


def tidy(ax):
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def load_rows():
    h3.STAGE1_DIR, h3.STAGE1_MODEL = ROOT / "outputs" / "core_assim", "assim"
    rows, events = x.build()
    fmap = pd.read_csv(cid.FOLD_FILE, dtype={"batch_id": str}).groupby("batch_id").fold.first()
    rows["fold"] = rows.batch_id.map(fmap)
    return rows, events


# ---------------------------------------------------------------- slide 2
def fig_sensors(rows):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1.05, 1]})
    # (a) bolus rule
    ax = axs[0]
    d = rows[rows.day.isin([7, 8, 9, 11, 12])].copy()
    cat = np.where(d.A > 0.005, 2, np.where(d.A > 0.001, 1, 0))
    labels = ["no top-up", "small top-up", "big top-up"]
    cols = ["#9ec5f4", "#f5a37c", C_CORE]
    rng = np.random.default_rng(0)
    for k in range(3):
        g = d.obs_substrate.to_numpy()[cat == k]
        ax.scatter(k + rng.uniform(-0.22, 0.22, len(g)), g, s=26, color=cols[k], alpha=0.85, edgecolor="white", lw=0.4)
    for th, lab in ((ca.T1, "0.965 g/L"), (ca.T2, "1.96 g/L")):
        ax.axhline(th, color=INK, lw=1.4, ls=(0, (5, 3)))
        ax.text(2.62, th + 0.06, lab, va="bottom", ha="center", fontsize=12, color=INK)
    ax.set_xticks(range(3), labels)
    ax.set_xlim(-0.5, 2.95)
    ax.set_ylabel("measured glucose before decision [g/L]")
    ax.set_title("1  The top-up size tells us the glucose range", loc="left", fontweight="bold")
    ok = ((d.obs_substrate >= d.g_lo) & (d.obs_substrate < d.g_hi)).mean()
    ax.text(0.02, 0.97, f"{len(d)} controller decisions\n{ok:.0%} follow the rule exactly", transform=ax.transAxes,
            va="top", fontsize=12, color=INK2)
    tidy(ax)
    # (b) unlogged volume vs lactate
    ax = axs[1]
    b = rows.groupby("batch_id").first()
    lac = []
    for bid, g in rows.groupby("batch_id"):
        r = g.iloc[(g.t - 5).abs().argmin()]
        lac.append(r.obs_byproduct * r.V - r.l0 * r.v0)
    b["lac5"] = lac
    rr = np.corrcoef(b.lac5, b.base_to5)[0, 1]
    ax.scatter(b.lac5, b.base_to5, s=60, color=C_ML, edgecolor="white", lw=0.6)
    k, c = np.polyfit(b.lac5, b.base_to5, 1)
    xx = np.linspace(b.lac5.min(), b.lac5.max(), 10)
    ax.plot(xx, k * xx + c, color=INK, lw=1.4)
    ax.set_xlabel("lactate produced by day 5 [mg]")
    ax.set_ylabel("unlogged volume added, days 2–5 [mL]")
    ax.set_title("2  Hidden volume tracks lactate (pH base)", loc="left", fontweight="bold")
    ax.text(0.02, 0.97, f"one dot = one batch\ncorrelation r = {rr:.2f}", transform=ax.transAxes, va="top",
            fontsize=12, color=INK2)
    tidy(ax)
    fig.tight_layout(w_pad=3)
    fig.savefig(OUT / "fig1_sensors.png")
    plt.close(fig)
    return ok, rr, len(d)


# ---------------------------------------------------------------- slide 3
def fig_particles(bid):
    events, template = sm.load_data()
    holdout_ids = list(dict.fromkeys(template.batch_id))
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in set(holdout_ids)]
    train = sm.build_batches(events, train_ids)
    sens = ca.sensor_table(events, train_ids)
    folds = cid.saved_folds(train_ids)
    i_all = train_ids.index(bid)
    tr, te = next((tr, te) for tr, te in folds if i_all in te)
    trs, tes = train.subset(tr), train.subset(te)
    p, _, _, _ = cid.fit(trs, "V0 baseline")
    lib = ca.fit_library(trs, p)
    bmap = ca.base_map(trs, sens)
    i = list(tes.ids).index(bid)
    rng = np.random.default_rng(0)
    mu, cov = lib.mean(0), np.cov(lib.T) + 1e-4 * np.eye(3)
    Q = np.vstack([np.zeros(3), rng.multivariate_normal(mu, cov, ca.N_PART - 1)])
    sim = ca.sim_mult(ca.tile(tes, i, ca.N_PART), p, np.exp(Q))
    from scipy.stats import norm
    logw = np.zeros(ca.N_PART)
    ints, base = sens[bid]
    for step, lo, hi in ints:
        s = np.log(np.maximum(sim[:, step, 2], 1e-6))
        ph = norm.cdf((np.log(hi) - s) / ca.SIG_G) if np.isfinite(hi) else 1.0
        pl = norm.cdf((np.log(lo) - s) / ca.SIG_G) if lo > 0 else 0.0
        logw += np.log(np.maximum(ph - pl, 1e-12))
    ri = tes.obs[tes.obs.batch_idx == i]
    for K, (a, b_, sd) in zip((3, 4, 5), bmap):
        step = int(ri.step.to_numpy()[np.abs(ri.elapsed_time.to_numpy() - K).argmin()])
        V = tes.vsnap[i, step] if np.isfinite(tes.vsnap[i, step]) else tes.v0[i]
        m = sim[:, step, 3] * V - tes.y0[i, 5] * tes.v0[i]
        logw += norm.logpdf(base[K - 3], a + b_ * m, sd * np.sqrt(3))
    w = np.exp(logw - logw.max())
    w /= w.sum()
    t = np.arange(sim.shape[1]) * sm.DT
    keep = t <= 13.6
    order = np.argsort(w)
    fig, axs = plt.subplots(1, 3, figsize=(15, 5.0))
    spec = [(0, "viable cells (VCD)", "10⁶ cells/mL", "vcd"), (2, "glucose", "g/L", "substrate"),
            (3, "lactate", "g/L", "byproduct")]
    wn = w / w.max()
    for ax, (j, title, unit, v) in zip(axs, spec):
        for k in order[::2]:  # noqa
            ax.plot(t[keep], sim[k, keep, j], color=C_CORE if wn[k] > 0.05 else "#cfcdc8",
                    alpha=0.15 + 0.85 * wn[k] if wn[k] > 0.05 else 0.25, lw=0.6 + 1.4 * wn[k])
        mean = np.exp(np.tensordot(w, np.log(np.maximum(sim[:, keep, j], 1e-6)), 1))
        ax.plot(t[keep], mean, color=INK, lw=2.6, label="weighted forecast")
        ax.scatter(ri.elapsed_time, ri[v], s=34, color="white", edgecolor=INK, lw=1.4, zorder=5,
                   label="measured (not used)")
        if v == "substrate":
            for step, lo, hi in ints:
                hi_ = min(hi, 4.4)
                ax.add_patch(plt.Rectangle((step * sm.DT - 0.18, lo), 0.36, hi_ - lo, color=C_FIN, alpha=0.45, lw=0,
                                           zorder=4))
            ax.set_xlim(5.6, 13.4)
            ax.set_ylim(0, 4.5)
            title = "glucose (days 6–13)"
            ax.plot([], [], color=C_FIN, alpha=0.5, lw=10, label="glucose range from top-up decision")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("day")
        ax.set_ylabel(unit)
        tidy(ax)
    h, l_ = axs[1].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower center", ncol=3, frameon=False, fontsize=12)
    ess = 1 / np.sum(w ** 2)
    fig.suptitle(f"{bid}: 400 Monod particles · grey = rejected, orange = kept (darker = higher weight) · "
                 f"effective particles {ess:.0f}", fontsize=13, color=INK2, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95), w_pad=2.5)
    fig.savefig(OUT / "fig2_particles.png")
    plt.close(fig)
    return ess


# ---------------------------------------------------------------- slide 4
def fig_ml_views(rows, bid):
    a = ROOT / "outputs" / "option_a3" / "v0_baseline"
    mem = pd.read_csv(a / "members_oof.csv", dtype={"batch_id": str})
    fin = pd.read_csv(a / "oof_predictions.csv", dtype={"batch_id": str})
    core = rows[["batch_id", "t_key"] + [f"s1_{v}" for v in TARGETS]]
    m = mem.merge(core, on=["batch_id", "t_key"]).merge(fin[["batch_id", "t_key"] + [f"f|{v}" for v in TARGETS]],
                                                        on=["batch_id", "t_key"])
    g = m[m.batch_id == bid].sort_values("t_key")
    fig = plt.figure(figsize=(15, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.05], wspace=0.32)
    for k, (v, title) in enumerate((("vcd", "viable cells (VCD)"), ("byproduct", "lactate"))):
        ax = fig.add_subplot(gs[0, k])
        ax.plot(g.t_key, g[f"s1_{v}"], color=C_CORE, lw=2.2, ls=(0, (5, 3)), label="Monod core")
        ax.plot(g.t_key, g[f"as|{v}"], color=C_ML, lw=1.6, marker="s", ms=4, label="view 1: day-by-day")
        ax.plot(g.t_key, g[f"as_fp|{v}"], color="#9a5ec7", lw=1.6, marker="^", ms=4, label="view 2: whole curve")
        ax.plot(g.t_key, g[f"f|{v}"], color=C_FIN, lw=2.8, label="final blend")
        ax.scatter(g.t_key, g[f"obs_{v}"], s=40, color="white", edgecolor=INK, lw=1.4, zorder=5, label="measured")
        e_core = 100 * np.mean(np.abs(g[f"s1_{v}"] / g[f"obs_{v}"] - 1))
        e_fin = 100 * np.mean(np.abs(g[f"f|{v}"] / g[f"obs_{v}"] - 1))
        ax.set_title(f"{title}: error {e_core:.0f}% → {e_fin:.0f}%", loc="left", fontweight="bold")
        ax.set_xlabel("day")
        tidy(ax)

    # FPCA shapes for VCD
    ax = fig.add_subplot(gs[0, 2])
    R = (np.log(rows.pivot_table(index="batch_id", columns="day", values="obs_vcd"))
         - rows.pivot_table(index="batch_id", columns="day", values="ls1_vcd"))
    Z = R.to_numpy() - R.to_numpy().mean(0)
    _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    share = sv ** 2 / np.sum(sv ** 2)
    names = ("level from day 5 on", "early days vs. mid-run")
    for k, col in zip(range(2), (C_FIN, C_ML)):
        sgn = np.sign(Vt[k][np.argmax(np.abs(Vt[k]))])
        ax.plot(R.columns, sgn * Vt[k], color=col, lw=2.8, marker="o", ms=4,
                label=f"shape {k + 1}: {names[k]} ({share[k]:.0%})")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_title("view 2: error shapes (VCD)", loc="left", fontweight="bold")
    ax.set_xlabel("day")
    ax.set_yticks([])
    ax.legend(fontsize=11.5, frameon=False, loc="upper right", bbox_to_anchor=(1.02, -0.17))
    tidy(ax)
    fig.suptitle(f"{bid}, predicted without its own measurements", fontsize=13, color=INK2, x=0.01, ha="left")
    h, l_ = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="lower left", ncol=5, frameon=False, fontsize=12, bbox_to_anchor=(0.04, 0.0))
    fig.subplots_adjust(left=0.05, right=0.98, top=0.86, bottom=0.22)
    fig.savefig(OUT / "fig3_ml_views.png")
    plt.close(fig)
    return share[:3]


# ---------------------------------------------------------------- slide 5
def fig_results():
    v3 = pd.read_csv(ROOT / "outputs" / "stack_v5" / "cv" / "members_oof_v7c.csv")
    fa = pd.read_csv(ROOT / "outputs" / "option_a" / "oof_predictions.csv")
    s3 = {v: 100 * np.mean(np.abs(v3[f"states|{v}"] / v3[f"obs_{v}"] - 1)) for v in TARGETS}
    sa = {v: 100 * np.mean(np.abs(fa[f"f|{v}"] / fa[f"obs_{v}"] - 1)) for v in TARGETS}
    s3["mean"], sa["mean"] = np.mean(list(s3.values())), np.mean(list(sa.values()))
    lab = {"vcd": "VCD", "tcd": "TCD", "substrate": "glucose", "byproduct": "lactate", "mean": "MEAN"}
    fig, ax = plt.subplots(figsize=(11, 5.6))
    xs = np.arange(5)
    keys = list(lab)
    b1 = ax.bar(xs - 0.2, [s3[k] for k in keys], 0.38, color=C_V3, label="hybrid_v3 (ODE + neural net + ML)")
    b2 = ax.bar(xs + 0.2, [sa[k] for k in keys], 0.38, color=C_CORE, label="Option A (Monod + sensors + 2 ML views)")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, f"{b.get_height():.2f}", ha="center", fontsize=12)
    for k, key in enumerate(keys):
        d = sa[key] - s3[key]
        ax.text(k, 9.25, f"{d:+.2f}", ha="center", fontsize=13, fontweight="bold", color=C_ML if d < -0.2 else INK2)
    ax.set_xticks(xs, [lab[k] for k in keys])
    ax.set_ylim(6, 9.7)
    ax.set_ylabel("held-out error, MAPE [%]")
    ax.text(-0.45, 9.55, "change:", fontsize=12, color=INK2)
    ax.legend(frameon=False, fontsize=12, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_results.png")
    plt.close(fig)
    return s3, sa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="Batch41")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rows, _ = load_rows()
    print("sensors:", fig_sensors(rows))
    print("ml views, FPCA shares:", fig_ml_views(rows, a.batch))
    print("results:", fig_results())
    print("particles ESS:", fig_particles(a.batch))


if __name__ == "__main__":
    main()
