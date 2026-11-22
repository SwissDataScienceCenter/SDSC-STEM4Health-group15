"""Explainer figure for Option A, compared with hybrid_v3 and hybrid_v7
-> outputs/option_a/model_explained.png

    python src/option_a.py
    python src/plot_option_a.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hybrid_v3 as h3  # noqa: E402

if not (h3.DATA_DIR / "bioreaction_remaining.csv").exists() and (h3.ROOT / "data" / "bioreaction_remaining.csv").exists():
    h3.DATA_DIR = h3.ROOT / "data"
import experiments_v5 as x  # noqa: E402
import experiments_v6 as e  # noqa: E402
import experiments_v7_ideas as iv7  # noqa: E402

ROOT = h3.ROOT
OUT = ROOT / "outputs" / "option_a" / "model_explained.png"
TARGETS = h3.TARGETS
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C_CORE, C_ASSIM, C_S2 = "#2a78d6", "#eb6834", "#1baf7a"
C_V3, C_A, C_V7 = "#9d9c98", "#eb6834", "#2a78d6"
RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"]
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK})

# ------------------------------------------------------------- data
x.OUT = e.CV
mem = pd.read_csv(e.CV / "members_oof_v7c.csv", dtype={"batch_id": str})
oof_a = pd.read_csv(ROOT / "outputs" / "option_a" / "oof_predictions.csv", dtype={"batch_id": str})
rows_states = iv7.rows_for("hybrid", "hybrid", "states")
V7 = ["pf3a", "a3i_fp", "states", "tp"]
oof_v7 = iv7.finish4(mem, e.stack(mem, V7, rows=rows_states, clip=False), rows_states, V7)


def ape(pred, obs):
    return 100 * np.abs(pred / obs - 1)


models = {"hybrid_v3": (mem, "states", C_V3), "Option A": (oof_a, "f", C_A), "hybrid_v7": (oof_v7, "f", C_V7)}
score = {k: {v: ape(d[f"{p}|{v}"], d["obs_" + v]).mean() for v in TARGETS} for k, (d, p, _) in models.items()}
for k in score:
    score[k]["mean"] = np.mean([score[k][v] for v in TARGETS])

fig = plt.figure(figsize=(18, 21), facecolor=SURF)
fig.suptitle("Option A: one Monod core, two ML views — a simpler hybrid_v7\n"
             f"Held-out mean MAPE {score['hybrid_v3']['mean']:.2f} (hybrid_v3) → {score['Option A']['mean']:.2f} "
             f"(Option A, 2 models) vs {score['hybrid_v7']['mean']:.2f} (hybrid_v7, 4 models)",
             fontsize=16.5, fontweight="bold", x=0.03, ha="left", y=0.985)

# ------------------------------------------------------------- A pipeline
ax = fig.add_axes([0.03, 0.66, 0.94, 0.28])
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)
ax.axis("off")
ax.text(0, 49.5, "A   The pipeline", fontsize=13, fontweight="bold", va="top")


def box(x0, y0, w, h, title, body, color, fill=True, fs=9.4):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                fc=color + ("1f" if fill else "00"), ec=color, lw=1.8))
    ax.text(x0 + 0.9, y0 + h - 0.9, title, fontsize=10.8, fontweight="bold", va="top")
    ax.text(x0 + 0.9, y0 + h - 4.4, body, fontsize=fs, va="top", color=INK2, linespacing=1.4)


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=15, color=INK2, lw=1.4))


box(0.3, 8, 13, 36, "Inputs",
    "• t=0 values: VCD,\n  TCD, glucose,\n  lactate, volume\n• feed log\n• volume log\n\nSoft sensors:\n• bolus size →\n  glucose range\n  (days 7–9, 11, 12)\n• unlogged volume\n  (base, days 3–5)\n  → lactate made", INK2,
    fill=False, fs=9.0)
box(17, 8, 21, 36, "Monod core + assimilation",
    "μ = μmax·S/(KS+S)·N/(KN+N)·KIL/(KIL+L)\nN = hidden nutrient (feed B)\n\n1. global Monod fit\n2. per training batch: multipliers\n    on μmax, Y_LS, Y_XS\n3. new batch: 400 particles\n    from that library\n4. weight each particle by\n    • glucose inside bolus range\n    • lactate ↔ base volume\n→ batch-specific trajectory s1",
    C_ASSIM, fs=9.0)
box(42.5, 27, 20, 17, "M1  day-by-day view (v3)",
    "hybrid branch: s1 × correction\n  from the soft sensors\ndata-driven branch: boosted\n  trees, 30 bags\n→ 50/50 geometric blend", C_S2, fs=9.0)
box(42.5, 8, 20, 16, "M2  whole-curve view (FPCA)",
    "residual curve log(obs/s1),\n  days 1–13\n5 typical shapes; amounts\n  predicted by ridge from\n  batch features", C_S2, fs=9.0)
box(67, 8, 14.5, 36, "Geometric blend",
    "pred = M1^w · M2^(1−w)\n\none w per target\nand day range\n(1–6, 7–13)\n\n(weights: panel C)", INK, fill=False)
box(85, 8, 14.5, 36, "Finish",
    "glucose: moved to\nthe best point\ninside the bolus-\nrule range\n\nTCD ≥ VCD\n\n→ VCD, TCD,\n   glucose, lactate\n   days 1–13", INK, fill=False)
arrow((13.7, 26), (16.6, 26))
arrow((38.4, 30), (42.1, 34))
arrow((38.4, 22), (42.1, 17))
arrow((62.9, 35), (66.6, 28))
arrow((62.9, 16), (66.6, 22))
arrow((81.9, 26), (84.6, 26))
ax.text(0.3, 3.0, "Removed vs hybrid_v7: Sartorius core + neural net, two-population core, 5-multiplier core with "
        "posterior features, TCD/VCD ratio model.  1 mechanistic core instead of 3, 2 models instead of 4, no neural net.",
        fontsize=9.5, color=INK2, style="italic")

# ------------------------------------------------------------- B per-target comparison
bx = fig.add_axes([0.06, 0.40, 0.42, 0.2])
fig.text(0.03, 0.625, "B   Held-out MAPE per target (7-fold CV over whole batches)", fontsize=13, fontweight="bold")
cats = TARGETS + ["mean"]
lab = {"vcd": "VCD", "tcd": "TCD", "substrate": "glucose", "byproduct": "lactate", "mean": "MEAN\n(leaderboard)"}
xw = np.arange(len(cats))
for i, (k, (_, _, col)) in enumerate(models.items()):
    vals = [score[k][c] for c in cats]
    bars = bx.bar(xw + (i - 1) * 0.26, vals, 0.24, color=col, label=k)
    for b_, v in zip(bars, vals):
        bx.text(b_.get_x() + b_.get_width() / 2, v + 0.05, f"{v:.2f}", ha="center", fontsize=8.3, color=INK)
bx.set_xticks(xw, [lab[c] for c in cats], fontsize=10)
bx.set_ylim(6, 9.2)
bx.set_ylabel("MAPE (%)")
bx.legend(frameon=False, fontsize=10, ncol=3, loc="upper left")
bx.grid(axis="y", color=GRID, lw=0.8)
bx.set_axisbelow(True)
for s in ("top", "right"):
    bx.spines[s].set_visible(False)

# ------------------------------------------------------------- C weights
W = pd.read_csv(ROOT / "outputs" / "option_a" / "blend_weights.csv", index_col=[0, 1])
cx = fig.add_axes([0.70, 0.40, 0.17, 0.2])
fig.text(0.58, 0.625, "C   Option A blend weights", fontsize=13, fontweight="bold")
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ramp", RAMP)
cx.imshow(W.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        v = W.iat[i, j]
        cx.text(j, i, f"{v:.2f}" if v >= 0.005 else "–", ha="center", va="center", fontsize=10,
                color="#ffffff" if v > 0.55 else INK)
cx.set_yticks(range(W.shape[0]), [f"{lab[a]}, {b}" for a, b in W.index], fontsize=10)
cx.set_xticks(range(2), ["M1\nday-by-day", "M2\nwhole-curve"], fontsize=9.5)
cx.xaxis.tick_top()
for s in cx.spines.values():
    s.set_visible(False)
cx.tick_params(length=0)
for k in range(2, W.shape[0], 2):
    cx.axhline(k - 0.5, color=SURF, lw=3)

# ------------------------------------------------------------- D by day
fig.text(0.03, 0.355, "D   Held-out MAPE by culture day — where Option A gains and where it loses",
         fontsize=13, fontweight="bold")
for j, v in enumerate(TARGETS):
    dx = fig.add_axes([0.05 + j * 0.24, 0.19, 0.2, 0.14])
    for k, (d, p, col) in models.items():
        s = pd.Series(ape(d[f"{p}|{v}"], d["obs_" + v]).to_numpy()).groupby(d.day.to_numpy()).mean()
        dx.plot(s.index, s.values, marker="o", ms=3.5, lw=1.8, color=col, label=k)
    dx.set_title(lab[v], fontsize=11)
    dx.set_xlabel("day", fontsize=9)
    dx.set_ylim(bottom=0)
    dx.grid(color=GRID, lw=0.8)
    for s_ in ("top", "right"):
        dx.spines[s_].set_visible(False)
    if j == 0:
        dx.set_ylabel("MAPE (%)")
        dx.legend(frameon=False, fontsize=8.5)

# ------------------------------------------------------------- E takeaways
fig.text(0.03, 0.135, "E   Takeaways", fontsize=13, fontweight="bold")
d_a = score["Option A"]["mean"] - score["hybrid_v3"]["mean"]
d_7 = score["hybrid_v7"]["mean"] - score["Option A"]["mean"]
T = [
    f"• Option A keeps {abs(d_a) / (abs(d_a) + abs(d_7)) * 100:.0f}% of v7's gain over hybrid_v3 "
    f"({d_a:+.2f} of {d_a + d_7:+.2f}) with 2 models on 1 mechanistic core and no neural net.",
    "• Gains come from glucose and lactate: the particles are weighted by the bolus ranges and base volume, "
    "so the Monod parameters are informed by each batch's own sensors.",
    "• Cells (VCD, TCD) improve only slightly over hybrid_v3; the late death phase is where hybrid_v7's "
    "two-population core earned its weight.",
    "• The 2 members were chosen by nested selection (best 2-model blend), so this CV is less optimistic than v7's "
    "hand-picked 4-model set.",
]
for i, t in enumerate(T):
    fig.text(0.03, 0.11 - i * 0.022, t, fontsize=10.5, color=INK)
fig.text(0.03, 0.012, "Code: src/option_a.py, src/core_assim.py, src/hybrid_v3.py, src/experiments_v5_ideas.py (fpca_member). "
         "Leaderboard file: outputs/option_a/leaderboard_submission.csv", fontsize=9.3, color=INK2)
fig.savefig(OUT, dpi=150, facecolor=SURF)
print("saved", OUT, {k: round(v["mean"], 3) for k, v in score.items()})
