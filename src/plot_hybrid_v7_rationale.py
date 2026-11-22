"""Why each hybrid_v7 step helped -> outputs/hybrid_v7/why_each_step_helped.png

One row per step, each with the evidence for its gain (all held-out / out-of-fold):
  1 M1 assimilated Monod: sensors select plausible parameter draws (example batch)
    and cut the late-day substrate / byproduct error of the core.
  2 M2 trajectory model: its errors differ from M1's, so the blend cancels part of them.
  3 M3 hybrid_v3: more accurate on days 1-6 for cells and byproduct.
  4 M4 two-population: more accurate late for cells.
  5 Finish: substrate pulled into the bolus-rule range; TCD/VCD ratio modelled directly.

    python src/plot_hybrid_v7_rationale.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core_assim as ca  # noqa: E402
import core_assim2 as c2  # noqa: E402
import experiments_v5 as x  # noqa: E402
import experiments_v6 as e  # noqa: E402
import experiments_v7_ideas as iv7  # noqa: E402
import sartorius_model as sm  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "hybrid_v7" / "why_each_step_helped.png"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
C1, C2, C3, C4 = "#eb6834", "#1baf7a", "#2a78d6", "#eda100"  # M1 orange, M2 aqua, M3 blue, M4 yellow
GRAY = "#9a9994"
EXAMPLE = "Batch8"
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "font.size": 10})
TARGETS = sm.TARGETS


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def mape_by_day(df, col, obs):
    return (100 * (df[col] / df[obs] - 1).abs()).groupby(df.day).mean()


# ---------------------------------------------------------------- data
mem = pd.read_csv(e.CV / "members_oof_v7c.csv", dtype={"batch_id": str})
rows, _ = x.build()
M = ["pf3a", "a3i_fp", "states", "tp"]


def particles_for_example():
    events, template = sm.load_data()
    hold = set(template.batch_id)
    train_ids = [b for b in dict.fromkeys(events.batch_id) if b not in hold]
    train = sm.build_batches(events, train_ids)
    idx = np.arange(len(train_ids))
    for f, (tr, te) in enumerate(GroupKFold(7).split(idx, groups=idx)):
        if train_ids.index(EXAMPLE) in te:
            break
    p = np.load(c2.CACHE / f"p_{f}.npy")
    cfg = c2.Cfg([0, 1, 2, 3, 4], True, 0.08, 1.732, 400, 0.5, 0.05, "geo")
    trs = train.subset(tr)
    lib = c2.fit_library(trs, p, cfg)
    sens = ca.sensor_table(events, train_ids)
    bmap = ca.base_map(trs, sens)
    i = train_ids.index(EXAMPLE)
    one = train.subset([i])
    rng = np.random.default_rng(0)
    mu, cov = lib.mean(0), np.cov(lib.T) + 1e-4 * np.eye(cfg.dim)
    mu[-1] = 0.0
    cov[-1, :] = cov[:, -1] = 0.0
    cov[-1, -1] = cfg.ic_sd ** 2
    Q = np.vstack([mu, rng.multivariate_normal(mu, cov, cfg.npart - 1)])
    mult, icm = cfg.expand(Q)
    n = cfg.npart
    sim = c2._simulate(np.repeat(one.y0, n, 0), np.repeat(one.v0, n, 0), np.repeat(one.feeds, n, 0),
                       np.repeat(one.vsnap, n, 0), p, mult, icm)
    logw = np.zeros(n)
    ints, base = sens[EXAMPLE]
    for step, lo, hi in ints:
        s = np.log(np.maximum(sim[:, step, 2], 1e-6))
        ph = norm.cdf((np.log(hi) - s) / cfg.sig) if np.isfinite(hi) else 1.0
        pl = norm.cdf((np.log(lo) - s) / cfg.sig) if lo > 0 else 0.0
        logw += np.log(np.maximum(ph - pl, 1e-12))
    ri = one.obs
    for K, (a, b, sd) in zip((3, 4, 5), bmap):
        step = int(ri.step.to_numpy()[np.abs(ri.elapsed_time.to_numpy() - K).argmin()])
        V = one.vsnap[0, step]
        logw += norm.logpdf(base[K - 3], a + b * (sim[:, step, 3] * V - one.y0[0, 5] * one.v0[0]), sd * cfg.infl)
    w = np.exp(logw - logw.max())
    w /= w.sum()
    t = np.arange(sim.shape[1]) * sm.DT
    return t, sim, w, one.obs, ints


t, sim, w, obs_ex, ints = particles_for_example()

fig = plt.figure(figsize=(17, 27), facecolor=SURF)
TOP, *SUB = fig.subfigures(6, 1, hspace=0.03, height_ratios=[0.16, 1, 1, 1, 1, 1])
TOP.set_facecolor(SURF)
TOP.text(0.03, 0.5, "Why each step of hybrid v7 lowers the error (all numbers held-out, 7-fold CV over whole batches)",
         fontsize=16, fontweight="bold", va="center")
AX = []
for sf in SUB:
    sf.set_facecolor(SURF)
    AX.append(sf.subplots(1, 2, gridspec_kw=dict(left=0.06, right=0.98, top=0.80, bottom=0.12, wspace=0.18)))


def row_title(r, text, sub):
    SUB[r].text(0.03, 0.965, text, fontsize=13, fontweight="bold", va="top")
    SUB[r].text(0.03, 0.905, sub, fontsize=10, color=INK2, va="top")


# ---------------------------------------------------------------- 1 M1
row_title(0, "1   M1 assimilated Monod (7.674 → 7.550): the logs say which parameter draws are plausible",
          "The bolus rule gives a substrate range on days 7–9, 11, 12, and the unlogged volume gives the byproduct made "
          "on days 2–5. Draws that contradict them get almost no weight.")
ax = AX[0][0]
style(ax)
order = np.argsort(w)
wn = w / w.max()
for k in order:
    ax.plot(t, sim[k, :, 2], color=C1 if wn[k] > 0.2 else GRAY, alpha=0.05 + 0.5 * wn[k] if wn[k] > 0.2 else 0.08,
            lw=0.8)
mean_all = np.exp(np.log(np.maximum(sim[:, :, 2], 1e-6)).mean(0))
mean_w = np.exp(w @ np.log(np.maximum(sim[:, :, 2], 1e-6)))
ax.plot(t, mean_all, color=INK2, lw=2, ls=(0, (4, 2)), label="all draws equally (no sensors)")
ax.plot(t, mean_w, color=C1, lw=2.6, label="draws weighted by the sensors (M1 core)")
for step, lo, hi in ints:
    tt = step * sm.DT
    ax.add_patch(plt.Rectangle((tt - 0.13, lo), 0.26, min(hi, 3.6) - lo, color=C2, alpha=0.35, lw=0))
ax.scatter(obs_ex.elapsed_time, obs_ex.substrate, color=INK, s=28, zorder=5, label="measured (not used)")
ax.add_patch(plt.Rectangle((0, 0), 0, 0, color=C2, alpha=0.35, label="substrate range from the bolus rule"))
ax.set_xlim(6.6, 13.2)
ax.set_ylim(0, 5.4)
ax.set_xlabel("day (zoom on days 7–13, where the bolus rule applies)")
ax.set_ylabel("substrate")
ax.set_title(f"Example {EXAMPLE}: 400 parameter draws of the Monod core (orange = kept, grey = down-weighted)",
             fontsize=10.5, loc="left")
lg = ax.legend(fontsize=8.6, frameon=True, loc="upper right", ncol=2, framealpha=0.95, facecolor=SURF, edgecolor=GRID)
lg.set_zorder(10)

ax = AX[0][1]
style(ax)
mo = pd.read_csv(ROOT / "outputs/core_monod/predictions_vs_measured.csv")
mo = mo[mo.split == "held-out (CV)"]
asd = pd.read_csv(ROOT / "outputs/a2_p5ic/predictions_vs_measured.csv")
for var, ls in (("substrate", "-"), ("byproduct", (0, (4, 2)))):
    a = mo[mo.variable == var].groupby("day").abs_pct_error.mean()
    b = asd[asd.variable == var].groupby("day").abs_pct_error.mean()
    ax.plot(a.index, a.values, color=GRAY, lw=2, ls=ls, marker="o", ms=4, label=f"{var}: Monod core alone")
    ax.plot(b.index, b.values, color=C1, lw=2, ls=ls, marker="o", ms=4, label=f"{var}: assimilated")
ax.axvspan(1.5, 5.5, color=C2, alpha=0.08)
ax.axvspan(6.5, 12.5, color=C2, alpha=0.08)
ax.text(3.5, 1.5, "unlogged-volume\ninformation", ha="center", fontsize=8.5, color=INK2)
ax.text(9.5, 1.5, "bolus-rule ranges", ha="center", fontsize=8.5, color=INK2)
ax.set_xlabel("day")
ax.set_ylabel("held-out MAPE of the core (%)")
ax.set_ylim(0, None)
ax.set_title("Core error by day: the drop sits where the sensors have information", fontsize=10.5, loc="left")
ax.legend(fontsize=8.5, frameon=False, ncol=2, loc="upper left")

# ---------------------------------------------------------------- 2 M2
row_title(1, "2   + M2 trajectory model (7.550 → 7.339): a differently built model makes different mistakes",
          "M2 predicts the whole 13-day curve at once from batch features. Its errors are only partly correlated with "
          "M1's, so the geometric average cancels part of both.")
ax = AX[1][0]
style(ax)
v = "substrate"
e1 = np.log(mem[f"pf3a|{v}"] / mem["obs_" + v])
e2 = np.log(mem[f"a3i_fp|{v}"] / mem["obs_" + v])
ax.scatter(100 * e1, 100 * e2, s=10, color=C2, alpha=0.45, lw=0)
lim = 45
ax.plot([-lim, lim], [-lim, lim], color=INK2, lw=1, ls=(0, (4, 3)))
ax.axhline(0, color=GRID)
ax.axvline(0, color=GRID)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("M1 error on substrate (log %, + = too high)")
ax.set_ylabel("M2 error on substrate (log %)")
r = np.corrcoef(e1, e2)[0, 1]
ax.set_title(f"Errors of M1 and M2 per sample: correlation {r:.2f} (1 = identical mistakes)", fontsize=10.5, loc="left")
ax.text(lim - 2, -lim + 4, "points off the diagonal: the two\nmodels disagree, so their average\nis closer to the measurement", fontsize=9,
        color=INK2, ha="right")

ax = AX[1][1]
style(ax)
o12 = e.stack(mem, ["pf3a", "a3i_fp"], rows=rows, clip=False)
xs = np.arange(4)
for j, (lab, col, src) in enumerate((("M1 alone", C1, "pf3a"), ("M2 alone", C2, "a3i_fp"), ("M1 + M2 blend", INK, "st"))):
    frame = mem if src != "st" else o12
    vals = [100 * np.mean(np.abs(frame[f"{src}|{vv}"] / frame["obs_" + vv] - 1)) for vv in TARGETS]
    ax.bar(xs + (j - 1) * 0.26, vals, width=0.24, color=col, edgecolor=SURF, lw=1.5, label=lab)
    for xx, vv in zip(xs + (j - 1) * 0.26, vals):
        ax.text(xx, vv + 0.1, f"{vv:.1f}", ha="center", fontsize=8.2, color=INK2)
ax.set_xticks(xs, ["VCD", "TCD", "substrate", "byproduct"])
ax.set_ylabel("held-out MAPE (%)")
ax.set_ylim(0, 11)
ax.set_title("The blend beats both on every target, most on substrate and byproduct", fontsize=10.5, loc="left")
ax.legend(fontsize=9, frameon=False, ncol=3, loc="upper left")

# ---------------------------------------------------------------- 3 M3
row_title(2, "3   + M3 hybrid_v3 (7.339 → 7.289): its neural-net core is better in the first days",
          "hybrid_v3's core has a neural net trained on early growth and byproduct production. On days 1–6 it is the "
          "most accurate of the three for VCD and byproduct, so the blend gives it most of the weight there.")
for k, var in enumerate(("vcd", "byproduct")):
    ax = AX[2][k]
    style(ax)
    for src, col, lab in (("pf3a", C1, "M1 assimilated Monod"), ("a3i_fp", C2, "M2 trajectory"),
                          ("states", C3, "M3 hybrid_v3")):
        d = mape_by_day(mem, f"{src}|{var}", "obs_" + var)
        ax.plot(d.index, d.values, color=col, lw=2.2 if src == "states" else 1.6, marker="o", ms=4, label=lab)
    ax.axvspan(0.5, 6.5, color=C3, alpha=0.07)
    ax.text(3.5, 0.6, "days 1–6: M3 weight 0.73–0.93", ha="center", fontsize=9, color=INK2)
    ax.set_xlabel("day")
    ax.set_ylabel("held-out MAPE (%)")
    ax.set_ylim(0, None)
    ax.set_title({"vcd": "VCD", "byproduct": "byproduct"}[var] + " error by day", fontsize=10.5, loc="left")
    ax.legend(fontsize=9, frameon=False, loc="upper left")

# ---------------------------------------------------------------- 4 M4
row_title(3, "4   + M4 two-population (7.289 → 7.220): late cells follow a switch to a non-growing state",
          "In the two-population core, growing cells stop dividing and consume byproduct, matching the late plateau and "
          "decline. It is the best cell model on days 7–13, where it gets weight 0.68–0.73.")
ax = AX[3][0]
style(ax)
for src, col, lab in (("states", C3, "M3 hybrid_v3"), ("tp", C4, "M4 two-population")):
    d = mape_by_day(mem, f"{src}|vcd", "obs_vcd")
    ax.plot(d.index, d.values, color=col, lw=2.2, marker="o", ms=4, label=lab)
ax.axvspan(6.5, 13.5, color=C4, alpha=0.10)
ax.text(10, 0.6, "days 7–13: M4 weight ≈ 0.7", ha="center", fontsize=9, color=INK2)
ax.set_xlabel("day")
ax.set_ylabel("held-out MAPE (%)")
ax.set_ylim(0, None)
ax.set_title("VCD error by day", fontsize=10.5, loc="left")
ax.legend(fontsize=9, frameon=False, loc="upper left")

ax = AX[3][1]
style(ax)
tt = np.linspace(0, 13, 1301)
xg_, xn_ = [1.0], [0.0]
for t_ in tt[1:]:
    kt = 0.9 / (1 + np.exp(-(t_ - 6.5) * 1.4))  # switching rate rises as nutrients run out
    g, n = xg_[-1], xn_[-1]
    xg_.append(g + 0.01 * (0.75 - kt - 0.02) * g)
    xn_.append(n + 0.01 * (kt * g - 0.06 * n))
xg, xn = np.array(xg_), np.array(xn_)
xtot = xg + xn
ax.stackplot(tt, xg, xn, colors=[C3 + "99", C4 + "cc"], labels=["growing cells", "non-growing cells"])
ax.plot(tt, xtot, color=INK, lw=2, label="VCD = growing + non-growing")
ax.set_xlabel("day")
ax.set_ylabel("cells (schematic)")
ax.set_yticks([])
ax.set_title("Sketch of the two-population idea (not data)", fontsize=10.5, loc="left")
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.annotate("growth stops: cells switch\nto the non-growing state,\nwhich consumes byproduct", xy=(8.0, xg[800]), xytext=(9.0, xtot.max() * 0.3),
            fontsize=8.8, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

# ---------------------------------------------------------------- 5 finish
row_title(4, "5   Finish (7.220 → 7.187): use what is known exactly, and keep TCD consistent with VCD",
          "The bolus rule fixes a range for substrate on days 7–9, 11, 12, and predictions outside it are moved in. "
          "TCD and VCD come from the same count, so TCD is predicted as VCD × a separately modelled TCD/VCD ratio.")
o4 = e.stack(mem, M, rows=rows, clip=False)
f4 = iv7.finish4(mem, o4, rows, M)
ax = AX[4][0]
style(ax)
lo, hi = rows.g_lo.to_numpy(), rows.g_hi.to_numpy()
msk = (lo > 0) | np.isfinite(hi)
obs = mem.obs_substrate.to_numpy()[msk]
before = o4["st|substrate"].to_numpy()[msk]
after = f4["f|substrate"].to_numpy()[msk]
moved = np.abs(after / before - 1) > 0.01
ax.scatter(obs[~moved], after[~moved], s=12, color=GRAY, alpha=0.45, lw=0, label="already inside its range (unchanged)")
ax.scatter(obs[moved], before[moved], s=22, facecolor="none", edgecolor=C1, lw=1, label="before finish (outside its range)")
ax.scatter(obs[moved], after[moved], s=22, color=C1, lw=0, label="after finish (moved to the range)")
for o_, b_, a_ in zip(obs[moved], before[moved], after[moved]):
    ax.annotate("", xy=(o_, a_), xytext=(o_, b_), arrowprops=dict(arrowstyle="->", color=C1, lw=0.9))
ax.plot([0, 3.1], [0, 3.1], color=INK2, lw=1, ls=(0, (4, 3)))
ax.set_xlim(0, 3.1)
ax.set_ylim(0, 3.1)
ax.set_xlabel("measured substrate")
ax.set_ylabel("predicted substrate")
e_b = 100 * np.mean(np.abs(before / obs - 1))
e_a = 100 * np.mean(np.abs(after / obs - 1))
ax.set_title(f"Substrate on bolus-rule days: MAPE {e_b:.1f} → {e_a:.1f} % ({moved.sum()} of {msk.sum()} samples moved)",
             fontsize=10.5, loc="left")
ax.legend(fontsize=9, frameon=False, loc="upper left")

ax = AX[4][1]
style(ax)
r_obs = (mem.obs_tcd / mem.obs_vcd).to_numpy()
r_b = (o4["st|tcd"] / o4["st|vcd"]).to_numpy()
r_a = (f4["f|tcd"] / f4["f|vcd"]).to_numpy()
days = mem.day.to_numpy()
for arr, col, lab in ((r_obs, INK, "measured"), (r_b, GRAY, "before finish (TCD and VCD blended separately)"),
                      (r_a, C3, "after finish (ratio modelled)")):
    m_ = pd.Series(arr).groupby(days).median()
    ax.plot(m_.index, m_.values, color=col, lw=2.2 if col != GRAY else 1.6, marker="o", ms=4, label=lab)
tb = 100 * np.mean(np.abs(o4["st|tcd"] / mem.obs_tcd - 1))
ta = 100 * np.mean(np.abs(f4["f|tcd"] / mem.obs_tcd - 1))
ax.set_xlabel("day")
ax.set_ylabel("TCD / VCD (median over batches)")
ax.set_title(f"Dead-cell share rises late; TCD MAPE {tb:.2f} → {ta:.2f} %", fontsize=10.5, loc="left")
ax.legend(fontsize=9, frameon=False, loc="upper left")

fig.savefig(OUT, dpi=140, facecolor=SURF)
print("saved", OUT)
