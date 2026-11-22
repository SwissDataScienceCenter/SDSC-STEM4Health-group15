"""Whole-model overview picture (Option A) in two rows, Arial 15 pt
-> outputs/slides/model_overview.png and .svg (the SVG keeps Arial as live text)

    python src/plot_model_overview.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "slides"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 15, "svg.fonttype": "none"})
INK, SOFT, GREY = "#14213D", "#4A5568", "#9C9A93"
STYLE = {"in": ("#F6F5F1", "#C9C5BA"), "sens": ("#FFF6E0", "#D9A21B"), "phys": ("#FDF0EA", "#EB6834"),
         "ml": ("#EAF7F1", "#1BAF7A"), "fin": ("#EBF1FA", "#2A68C2")}

fig, ax = plt.subplots(figsize=(17.6, 6.8))
ax.set_xlim(0, 176)
ax.set_ylim(0, 68)
ax.axis("off")


def box(x, y, w, h, title, body, kind):
    fc, ec = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=15, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 8.6, body, ha="center", va="top", fontsize=15, color=SOFT, linespacing=1.35)


def arrow(p, q, style="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=20, color=GREY, lw=2, connectionstyle=style))


# row 1: from the log to the personalized physics forecast
y1, h1 = 36, 24
r1 = [(1, 22, "Input", "day-0 values\n+ feed and\nvolume log", "in"),
      (29, 36, "Soft sensors", "top-up size → glucose range\nextra volume → lactate", "sens"),
      (71, 32, "1. Physics", "Monod model:\ngrowth, glucose, lactate,\nhidden nutrient, death", "phys"),
      (109, 28, "2. Personalize", "own growth speed\nand yields:\n400 versions", "phys"),
      (143, 32, "3. Data assimilation", "keep versions that\nagree with the\nsoft sensors", "phys")]
for x, w, t, b, k in r1:
    box(x, y1, w, h1, t, b, k)
for (x, w, *_), (nx, *_) in zip(r1[:-1], r1[1:]):
    arrow((x + w + 0.6, y1 + h1 / 2), (nx - 0.6, y1 + h1 / 2))
arrow((47, y1 + h1 + 0.6), (159, y1 + h1 + 0.6), "arc3,rad=-0.06")
ax.text(103, 67.6, "soft sensors steer which versions are kept", ha="center", va="top", fontsize=15, color=SOFT,
        style="italic")

# row 2: learned corrections and finish
y2, h2 = 3, 24
box(1, y2, 26, h2, "Physics forecast", "weighted average\nof the kept versions", "phys")
box(34, y2 + 13.5, 42, 12, "View 1 · fix each day", "learned correction per day", "ml")
box(34, y2 - 1.5, 42, 12, "View 2 · fix the whole curve", "functional PCA: error shapes", "ml")
box(83, y2, 26, h2, "Average", "per variable,\nearly / late phase", "fin")
box(116, y2, 28, h2, "Physical rules", "glucose in its range\nTCD ≥ VCD", "fin")
box(151, y2, 24, h2, "Forecast", "VCD, TCD,\nglucose, lactate\ndays 1–13", "fin")
# wrap from the end of row 1 to the start of row 2, through the gap between rows
ax.plot([159, 159, 14], [y1 - 0.6, 31.5, 31.5], color=GREY, lw=2, solid_capstyle="round")
arrow((14, 31.5), (14, y2 + h2 + 0.6))
arrow((27.6, y2 + h2 / 2), (33.4, y2 + 19.5))
arrow((27.6, y2 + h2 / 2), (33.4, y2 + 4.5))
arrow((76.6, y2 + 19.5), (82.4, y2 + h2 / 2))
arrow((76.6, y2 + 4.5), (82.4, y2 + h2 / 2))
arrow((109.6, y2 + h2 / 2), (115.4, y2 + h2 / 2))
arrow((144.6, y2 + h2 / 2), (150.4, y2 + h2 / 2))

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"model_overview.{ext}", dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("saved", OUT / "model_overview.png")
