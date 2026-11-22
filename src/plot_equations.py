"""LaTeX-style equation pictures for the slides -> outputs/slides/eq_*.png

    python src/plot_equations.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "outputs" / "slides"
INK, SOFT, ORANGE, GREEN, BLUE = "#14213D", "#4A5568", "#B8441A", "#12805A", "#2A68C2"
plt.rcParams.update({"mathtext.fontset": "cm", "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def monod():
    fig = plt.figure(figsize=(9.6, 2.3))
    fig.text(0.5, 0.66, r"$\mu \;=\; \mu_{max}\;\cdot\;\dfrac{S}{K_S+S}\;\cdot\;\dfrac{N}{K_N+N}\;\cdot\;\dfrac{K_{IL}}{K_{IL}+L}$",
             ha="center", va="center", fontsize=34, color=INK)
    for x, lab, col in ((0.2, "max growth\nspeed", SOFT), (0.395, "glucose", ORANGE), (0.64, "hidden nutrient\n(feed B)", GREEN),
                        (0.885, "lactate brake", BLUE)):
        fig.text(x, 0.13, lab, ha="center", va="center", fontsize=15, color=col)
    save(fig, "eq_monod.png")


def fpca():
    fig = plt.figure(figsize=(9.6, 1.5))
    fig.text(0.5, 0.5, r"$\log\dfrac{\mathrm{measured}(t)}{\mathrm{core}(t)} \;\approx\; \overline{r}(t) \;+\; a_1\,\varphi_1(t) \;+\; a_2\,\varphi_2(t) \;+\;\dots\;+\; a_5\,\varphi_5(t)$",
             ha="center", va="center", fontsize=28, color=INK)
    save(fig, "eq_fpca.png")


def full():
    lines = [
        (r"$\mu = \mu_{max}\,\dfrac{S}{K_S+S}\,\dfrac{N}{K_N+N}\,\dfrac{K_{IL}}{K_{IL}+L}$", "growth"),
        (r"$k_d = k_{d0} + k_{dN}\,\dfrac{K_{Nd}}{K_{Nd}+N} + k_{dL}\,L$", "death: basal, starvation, lactate"),
        (r"$q_S = \mu / Y_{XS} + m_S$", "glucose uptake"),
        (r"$q_L = Y_{LS}\,q_S\,\dfrac{S}{K_{SL}+S} \;-\; q_{Lc}\,\dfrac{L}{K_{Lc}+L}\,\left(1-\dfrac{\mu}{\mu_{max}}\right)$", "lactate: made, then eaten"),
        (r"$\dfrac{dX}{dt}=(\mu-k_d)\,X \qquad \dfrac{dD}{dt}=k_d X - k_{lys} D$", "viable, dead cells"),
        (r"$\dfrac{dS}{dt}=-q_S X \qquad \dfrac{dL}{dt}=q_L X \qquad \dfrac{dN}{dt}=-\mu X / Y_{XN}$", "glucose, lactate, nutrient"),
    ]
    fig = plt.figure(figsize=(13, 7.2))
    for i, (eq, lab) in enumerate(lines):
        y = 0.93 - i * 0.165
        fig.text(0.02, y, eq, ha="left", va="center", fontsize=25, color=INK)
        fig.text(0.98, y, lab, ha="right", va="center", fontsize=16, color=SOFT)
    save(fig, "eq_full.png")


def weight():
    fig = plt.figure(figsize=(12, 1.7))
    fig.text(0.5, 0.5, r"$w_i \;\propto\; \prod_{k}\mathrm{Pr}\left(S_i(t_k)\in[\ell_k,h_k]\right)\;\cdot\;\prod_{K=3}^{5}\mathcal{N}\!\left(\mathrm{base}_K \,|\, a_K + b_K\,\Delta L_i(K),\;\sigma_K\right)$",
             ha="center", va="center", fontsize=26, color=INK)
    save(fig, "eq_weight.png")


def fpca_full():
    fig = plt.figure(figsize=(12, 1.7))
    fig.text(0.5, 0.5, r"$\widehat{y}_b(t) \;=\; \mathrm{core}_b(t)\,\cdot\,\exp\left(\overline{r}(t) + \sum_{k=1}^{5}\widehat{a}_{bk}\,\varphi_k(t)\right),\qquad \widehat{a}_{bk} = \mathrm{ridge}_k(\mathrm{batch\ facts})$",
             ha="center", va="center", fontsize=26, color=INK)
    save(fig, "eq_fpca_full.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (monod, fpca, full, weight, fpca_full):
        f()
    print("saved", sorted(p.name for p in OUT.glob("eq_*.png")))
