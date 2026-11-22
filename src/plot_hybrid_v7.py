"""Explainer figure for hybrid_v7 -> outputs/hybrid_v7/model_explained.png

    python src/plot_hybrid_v7.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "hybrid_v7" / "model_explained.png"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_CORE, C_ASSIM, C_S2 = "#2a78d6", "#eb6834", "#1baf7a"  # validated categorical slots 1-3
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]  # blue ramp
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK})

fig = plt.figure(figsize=(18, 21), facecolor=SURF)
fig.suptitle("How hybrid v7 predicts VCD, TCD, substrate and byproduct for a new batch\n"
             "Held-out mean MAPE 7.674 (hybrid_v3) → 7.187 with 4 models.  All terms are explained in panel D.",
             fontsize=16.5, fontweight="bold", x=0.03, ha="left", y=0.985)

# ------------------------------------------------------------- A pipeline
ax = fig.add_axes([0.03, 0.63, 0.94, 0.31])
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")
ax.text(0, 49.5, "A   The pipeline", fontsize=13, fontweight="bold", va="top")


def box(x, y, w, h, title, body, color, fill=True, fs=9.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                fc=color + ("1f" if fill else "00"), ec=color, lw=1.8))
    ax.text(x + 0.9, y + h - 0.9, title, fontsize=10.8, fontweight="bold", va="top")
    ax.text(x + 0.9, y + h - 4.4, body, fontsize=fs, va="top", color=INK2, linespacing=1.4)


def arrow(p, q, style="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, color=INK2, lw=1.4,
                                 connectionstyle=style))


box(0.3, 14, 12.5, 30, "Inputs",
    "• t=0 values: VCD,\n  TCD, substrate,\n  byproduct, volume\n• feed log (feed_\n  substrate, b, c, d)\n• volume log\n\nSensors (D) read\nfrom the logs:\n• bolus size →\n  substrate range\n• unlogged volume\n  → byproduct made", INK2,
    fill=False, fs=9.0)

# stage-1 cores
box(16, 35, 18, 11, "Hybrid ODE core", "mechanistic ODE + small\nneural net on growth and\nbyproduct rates", C_CORE)
box(16, 20.5, 18, 12, "Monod core +\nassimilation",
    "\n400 parameter draws per\nbatch, kept in proportion\nto how well they match\nthe batch's sensors", C_ASSIM)
box(16, 6, 18, 12, "Two-population core", "growing cells switch to a\nnon-growing state that\nconsumes byproduct", C_CORE)

# 4 models
box(39, 38, 20, 9, "M3  hybrid_v3 (baseline)", "hybrid ODE core +\nv3 corrections", C_S2, fs=9.0)
box(39, 26.5, 20, 9.5, "M1  Assimilated Monod",
    "assimilated Monod core +\nv3 corrections", C_S2, fs=9.0)
box(39, 15.5, 20, 9, "M2  Trajectory model", "whole 13-day curve from\nthe assimilated Monod core", C_S2, fs=9.0)
box(39, 4, 20, 9, "M4  Two-population", "two-population core +\nv3 corrections", C_S2, fs=9.0)

box(63.5, 14, 16, 30, "Geometric blend",
    "prediction =\nM1^w1 · M2^w2 ·\nM3^w3 · M4^w4\n\nweights ≥ 0, sum 1,\none set per target\nand per day range\n(1–6, 7–13)\n\n(weights: panel C)", INK, fill=False)
box(83, 14, 16.4, 30, "Finish",
    "substrate: kept inside\nthe range implied by\nthe bolus rule\n\nTCD = VCD × predicted\nTCD/VCD ratio\n\n→ VCD, TCD,\n   substrate, byproduct\n   for days 1–13", INK, fill=False)

arrow((13.3, 40), (15.6, 40.5))
arrow((13.3, 27), (15.6, 26.5))
arrow((13.3, 13), (15.6, 12), "arc3,rad=0")
arrow((34.4, 40.5), (38.6, 42.5))
arrow((34.4, 27), (38.6, 31))
arrow((34.4, 25), (38.6, 20))
arrow((34.4, 12), (38.6, 8.5))
for yy in (42.5, 31.2, 20, 8.5):
    arrow((59.4, yy), (63.1, 29 + (yy - 25) * 0.25))
arrow((79.9, 29), (82.6, 29))
ax.text(0.3, 1.0, "Every model predicts a batch only from its t=0 values and its feed and volume logs, "
        "the same information available for the 12 leaderboard batches.", fontsize=9, color=INK2, style="italic")

# ------------------------------------------------------------- B build-up
bx = fig.add_axes([0.25, 0.335, 0.30, 0.25])
fig.text(0.03, 0.605, "B   Held-out mean MAPE as each piece is added (7-fold CV over whole batches)",
         fontsize=13, fontweight="bold")
steps = [("M3 hybrid_v3 alone (baseline)", 7.674, INK2),
         ("M1 assimilated Monod alone", 7.550, C_ASSIM),
         ("M1 + M2 trajectory model", 7.339, C_S2),
         ("+ M3 hybrid_v3", 7.289, C_CORE),
         ("+ M4 two-population", 7.220, C_CORE),
         ("+ finish (substrate range, TCD ratio)", 7.187, INK)]
y = np.arange(len(steps))[::-1]
for yi, (lab, val, col) in zip(y, steps):
    bx.hlines(yi, 6.5, val, color=GRID, lw=1.5, zorder=1)
    bx.scatter([val], [yi], s=150, color=col, edgecolor=SURF, linewidth=2, zorder=3)
    bx.text(val + 0.03, yi, f"{val:.3f}", va="center", fontsize=10.5, fontweight="bold")
prev = None
for yi, (_, val, _) in zip(y, steps):
    if prev is not None:
        bx.text(7.92, yi, f"{val - prev:+.3f}", va="center", ha="right", fontsize=9.5, color=INK2)
    prev = val
bx.text(7.92, len(steps) - 0.45, "change", ha="right", fontsize=9, color=INK2)
bx.set_yticks(y, [s[0] for s in steps], fontsize=10)
bx.set_xlim(6.5, 7.95)
bx.set_ylim(-0.6, len(steps) - 0.3)
bx.axvline(7.0, color=INK2, lw=1.1, ls=(0, (4, 3)))
bx.text(7.005, -0.45, "target 7", fontsize=9, color=INK2)
bx.set_xlabel("Mean MAPE over VCD, TCD, substrate, byproduct (%)", fontsize=9.5)
bx.grid(axis="x", color=GRID, lw=0.8)
bx.set_axisbelow(True)
for s in ("top", "right", "left"):
    bx.spines[s].set_visible(False)
bx.spines["bottom"].set_color(GRID)
bx.tick_params(axis="y", length=0)

# ------------------------------------------------------------- C weights
W = pd.read_csv(ROOT / "outputs" / "hybrid_v7" / "blend_weights.csv", index_col=[0, 1])
names = {"pf3a": "M1\nassimilated\nMonod", "a3i_fp": "M2\ntrajectory\nmodel", "states": "M3\nhybrid_v3", "tp": "M4\ntwo-\npopulation"}
W = W[list(names)]
cx = fig.add_axes([0.68, 0.335, 0.29, 0.22])
fig.text(0.60, 0.605, "C   Blend weight of each model", fontsize=13, fontweight="bold")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ramp", RAMP)
cx.imshow(W.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        v = W.iat[i, j]
        cx.text(j, i, f"{v:.2f}" if v >= 0.005 else "–", ha="center", va="center", fontsize=10,
                color="#ffffff" if v > 0.55 else INK)
var_lab = {"vcd": "VCD", "tcd": "TCD", "substrate": "substrate", "byproduct": "byproduct"}
cx.set_yticks(range(W.shape[0]), [f"{var_lab[a]}, {b}" for a, b in W.index], fontsize=10)
cx.set_xticks(range(W.shape[1]), list(names.values()), fontsize=9.5)
cx.xaxis.tick_top()
for s in cx.spines.values():
    s.set_visible(False)
cx.tick_params(length=0)
for k in range(2, W.shape[0], 2):
    cx.axhline(k - 0.5, color=SURF, lw=3)


# ------------------------------------------------------------- D glossary
fig.text(0.03, 0.275, "D   Glossary", fontsize=13, fontweight="bold")
G = [
    ("Targets", "VCD = viable cell density, TCD = total (viable + dead) cell density, substrate and byproduct = the two\n"
                "measured concentrations. Substrate behaves like glucose (consumed, topped up by feed_substrate) and\n"
                "byproduct like lactate (produced, later consumed), so the cores model them that way."),
    ("MAPE, held-out, CV", "Mean absolute percentage error, averaged over the 4 targets (the leaderboard score). Held-out = each\n"
                           "batch is predicted by models fitted without it (7-fold cross-validation, CV, over the 49 training batches)."),
    ("Core (mechanistic)", "A set of differential equations (ODE) for cells, substrate and byproduct, run forward from the t=0\n"
                           "values while replaying every logged feed. It is the first-stage forecast of every model."),
    ("Hybrid ODE core", "Mechanistic ODE (viable, dead, lysed cells, substrate uptake, byproduct overflow) with a small neural\n"
                        "network that corrects the growth and byproduct rates. Used by hybrid_v3."),
    ("Monod core", "Alternative ODE: growth is limited by substrate and by a hidden nutrient that only feed_b supplies."),
    ("Two-population core", "Alternative ODE: growing cells switch to a non-growing state that consumes byproduct."),
    ("Sensors (soft sensors)", "Not measurements, but logged actions that reveal one. (1) Bolus size: feed_substrate is added ~0.5 d\n"
                               "after a sample, 0.006·V if substrate < 0.965 and 0.004·V if < 1.96 (days 7–9, 11, 12), so the size gives a\n"
                               "substrate range. (2) Unlogged volume: volume gained on days 2–5 beyond the logged feeds (pH-control base),\n"
                               "which tracks how much byproduct the batch has made."),
    ("Assimilation", "Run the core 400 times per batch with different growth, yield and death parameters, then weight each\n"
                     "run by how well it matches that batch's sensors. The weighted average is the assimilated forecast."),
    ("v3 corrections", "The second stage of hybrid_v3: corrects a core's forecast using the t=0 values, deviations of the feed\n"
                       "log from the usual schedule and the sensors, then averages with gradient-boosted trees (50/50)."),
    ("Trajectory model", "Predicts a batch's whole 13-day curve as the core forecast plus a few typical curve shapes, whose\n"
                         "amounts are regressed on batch features (functional PCA)."),
    ("Geometric blend, finish", "Weighted product of the 4 models' predictions; weights per target and day range, fitted on\n"
                                "held-out predictions. Finish: substrate is moved into its bolus-rule range, TCD = VCD × ratio."),
]
gx = fig.add_axes([0.03, 0.03, 0.94, 0.235])
gx.axis("off")
yy = 1.0
for term, text in G:
    n = text.count("\n") + 1
    gx.text(0.0, yy, term, fontsize=9.6, fontweight="bold", va="top", transform=gx.transAxes)
    gx.text(0.17, yy, text, fontsize=9.3, color=INK2, va="top", linespacing=1.3, transform=gx.transAxes)
    yy -= 0.034 * n + 0.012

fig.text(0.03, 0.008, "Final per target: VCD 6.77 · TCD 6.71 · substrate 7.17 · byproduct 8.10 (CV). "
         "The 4-model set was chosen among six candidate sets on the same folds (≈0.03 optimism). "
         "Code: src/hybrid_v7.py, src/core_assim2.py, src/core_twopop.py.", fontsize=9.3, color=INK2)
fig.savefig(OUT, dpi=150, facecolor=SURF)
print("saved", OUT)
