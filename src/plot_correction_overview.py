"""Second half of the model only: physics forecast -> two views -> average -> physical rules.
Arial 15 pt -> outputs/slides/correction_overview.png and .svg

    python src/plot_correction_overview.py
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
STYLE = {"phys": ("#FDF0EA", "#EB6834"), "ml": ("#EAF7F1", "#1BAF7A"), "fin": ("#EBF1FA", "#2A68C2")}

fig, ax = plt.subplots(figsize=(14.6, 3.3))
ax.set_xlim(0, 146)
ax.set_ylim(0, 33)
ax.axis("off")


def box(x, y, w, h, title, body, kind):
    fc, ec = STYLE[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=2))
    ax.text(x + w / 2, y + h - 3.0, title, ha="center", va="top", fontsize=15, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h - 8.4, body, ha="center", va="top", fontsize=15, color=SOFT, linespacing=1.35)


def arrow(p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=20, color=GREY, lw=2))


y, h = 3, 26
box(1, y, 27, h, "Physics forecast", "weighted average\nof the kept versions", "phys")
box(35, y + 14, 44, 13.5, "View 1 · fix each day", "learned correction per day", "ml")
box(35, y - 1.5, 44, 13.5, "View 2 · fix the whole curve", "functional PCA: error shapes", "ml")
box(86, y, 26, h, "Average", "per variable,\nearly / late phase", "fin")
box(119, y, 26, h, "Physical rules", "glucose in its range\nTCD ≥ VCD", "fin")
arrow((28.6, y + h / 2), (34.4, y + 20.75))
arrow((28.6, y + h / 2), (34.4, y + 5.25))
arrow((79.6, y + 20.75), (85.4, y + h / 2))
arrow((79.6, y + 5.25), (85.4, y + h / 2))
arrow((112.6, y + h / 2), (118.4, y + h / 2))

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(OUT / f"correction_overview.{ext}", dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.15)
print("saved")
