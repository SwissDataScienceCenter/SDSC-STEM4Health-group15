"""Build the 5-minute deck: presentation/slides/presentation.pptx (`make slides`).

One idea per slide; the titles alone tell the story. Figures are drawn here in the hero
video's palette, and every number is read from the DVC-tracked metrics, so rerunning after
`dvc repro` keeps the deck current. The last slide plays the hero video on loop
(`make hero-video` first).

Figures and the deck show measured values, so they are not in git (like the videos).
"""

import copy
import json
from pathlib import Path

from lxml import etree
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).resolve().parent
PRES = HERE.parent
ROOT = PRES.parent
DATA, METRICS = PRES / "data", ROOT / "reports" / "metrics"
FIG = HERE / "figures"
HERO = PRES / "media" / "videos" / "hero" / "1080p60" / "hero.mp4"
OUT = HERE / "presentation.pptx"

TARGETS = ["vcd", "tcd", "substrate", "byproduct"]
LABELS = {"vcd": "VCD", "tcd": "TCD", "substrate": "Substrate", "byproduct": "Byproduct"}
FEED_ROWS = {"feed_b": "Feed B", "feed_c": "Feed C", "feed_d": "Feed D",
             "feed_substrate": "Substrate"}  # fmt: skip
# Same palette as the hero video (validated: CVD and contrast checks pass on this surface).
SURFACE, TEXT, TEXT_2, GRID, TRUTH = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37", "#f0efec"
MODELS = {"mechanistic_lite": ("Mechanistic", "#d95926"),
          "ml_sensors": ("Machine learning", "#3987e5"),
          "hybrid_blend": ("Hybrid", "#199e70")}  # fmt: skip
BATCH = "Batch10"  # the typical holdout batch of the hero video
FONT = "Arial"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 13, "text.color": TEXT,
    "axes.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT_2, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False,
})  # fmt: skip


def metric(name, key):
    return json.loads((METRICS / name).read_text())[key]


def rgb(hex_):
    return RGBColor.from_string(hex_.lstrip("#"))


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def batch_traces():
    return {m: pd.read_csv(DATA / f"{m}.csv").query("batch_id == @BATCH").sort_values("t")
            for m in MODELS}  # fmt: skip


def panels(n_rows, height):
    fig, axes = plt.subplots(n_rows, 1, figsize=(8.6, height), sharex=True,
                             gridspec_kw={"hspace": 0.25})  # fmt: skip
    return fig, axes


def fig_task(path):
    """Day 0 known, the feed plan known, everything after day 0 to be predicted."""
    tr = batch_traces()["hybrid_blend"]
    feeds = pd.read_csv(DATA / "feeds" / "feeds.csv").query("batch_id == @BATCH")
    fig = plt.figure(figsize=(8.6, 5.2))
    gs = fig.add_gridspec(5, 1, height_ratios=[1, 1, 1, 1, 0.9], hspace=0.3)
    for i, c in enumerate(TARGETS):
        ax = fig.add_subplot(gs[i])
        top = 1.15 * max(tr[f"{c}_actual"].max(), tr[f"{c}_pred"].max())
        ax.axvspan(0.4, 14, color="#24241f", zorder=0)
        ax.plot(0, tr[f"{c}_actual"].iat[0], "o", color=TRUTH, markersize=8, zorder=3)
        ax.text(7.2, top * 0.45, "?", color=TEXT_2, fontsize=22, ha="center", va="center")
        ax.set_ylim(-0.08 * top, top)  # room for the day-0 dots
        ax.set_xlim(-0.3, 14)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylabel(LABELS[c], rotation=0, ha="right", va="center", color=TEXT, fontsize=13)
    ax = fig.add_subplot(gs[4])
    for k, (name, label) in enumerate(FEED_ROWS.items()):
        f = feeds[feeds["feed"] == name]
        base = 3 - k
        ax.vlines(f["t"], base, base + 0.75 * np.maximum(f["rel"], 0.15), color=TEXT_2, lw=2)
        ax.text(-0.5, base + 0.3, label, ha="right", va="center", color=TEXT_2, fontsize=10)
    ax.set_xlim(-0.3, 14)
    ax.set_ylim(-0.1, 4)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Culture day")
    ax.set_xticks(range(0, 15, 2))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_prediction(model, path):
    """One holdout batch: actual (off-white dots) vs. the model's prediction."""
    tr = batch_traces()
    d = tr[model]
    fig, axes = panels(4, 5.2)
    for ax, c in zip(axes, TARGETS):
        top = 1.12 * max(max(t[f"{c}_actual"].max(), t[f"{c}_pred"].max()) for t in tr.values())
        ax.plot(d["t"], d[f"{c}_actual"], "-o", color=TRUTH, lw=1.6, markersize=4.5)
        ax.plot(d["t"], d[f"{c}_pred"], "-", color=MODELS[model][1], lw=3.2)
        ax.set_ylim(0, top)
        ax.set_yticks([])
        ax.set_ylabel(LABELS[c], rotation=0, ha="right", va="center", color=TEXT, fontsize=13)
    axes[-1].set_xlabel("Culture day")
    axes[-1].set_xticks(range(0, 15, 2))
    axes[0].plot([], [], "-o", color=TRUTH, label="Actual", markersize=4.5)
    axes[0].plot([], [], "-", color=MODELS[model][1], lw=3.2, label="Predicted")
    axes[0].legend(frameon=False, loc="lower right", ncol=2, fontsize=11, labelcolor=TEXT)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_per_variable(path):
    """CV MAPE per variable for the three approaches (lower is better)."""
    fig, ax = plt.subplots(figsize=(8.8, 4.6))
    y = np.arange(len(TARGETS))[::-1]
    offs = np.linspace(0.22, -0.22, len(MODELS))
    for o, (m, (name, color)) in zip(offs, MODELS.items()):
        cv = metric(f"cv_{m}.json", "cv_mape")
        vals = [cv[v] for v in TARGETS]
        ax.hlines(y + o, 0, vals, color=color, lw=3)
        ax.plot(vals, y + o, "o", color=color, markersize=9, label=name,
                markeredgecolor=SURFACE, markeredgewidth=1.5)  # fmt: skip
        for v, yy in zip(vals, y + o):
            ax.text(v + 0.25, yy, f"{v:.1f}", va="center", color=TEXT_2, fontsize=10)
    ax.set_yticks(y, [LABELS[v] for v in TARGETS], color=TEXT, fontsize=14)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 19)
    ax.set_xlabel("Cross-validated error, MAPE (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17),
              fontsize=12, labelcolor=TEXT)  # fmt: skip
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def first_frame(video, path, at=6.0):
    """Poster image for the video slide (the behaviour phase of the hero video)."""
    import subprocess

    ffmpeg = "/layers/paketo-buildpacks_conda-env-update/conda-env/bin/ffmpeg"
    subprocess.run([ffmpeg, "-loglevel", "error", "-y", "-ss", str(at), "-i", str(video),
                    "-frames:v", "1", str(path)], check=True)  # fmt: skip


# --------------------------------------------------------------------------------------
# Slides
# --------------------------------------------------------------------------------------

W, H = Inches(13.333), Inches(7.5)


def new_slide(prs, title=None, notes=""):
    s = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = rgb(SURFACE)
    if title:
        text(s, title, Inches(0.6), Inches(0.45), Inches(12.1), Inches(1.0), 28, TEXT, bold=True)
    if notes:
        s.notes_slide.notes_text_frame.text = notes
    return s


def text(slide, s, x, y, w, h, size, color, bold=False, align=PP_ALIGN.LEFT, font=FONT):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(s.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.name = Pt(size), bold, font
        r.font.color.rgb = rgb(color)
    return box


def big_number(slide, value, caption, x, y, color, sub=None):
    text(slide, value, x, y, Inches(3.6), Inches(1.3), 66, color, bold=True)
    text(slide, caption, x, y + Inches(1.3), Inches(3.6), Inches(0.9), 18, TEXT_2)
    if sub:
        text(slide, sub, x, y + Inches(2.4), Inches(3.6), Inches(1.2), 15, TEXT_2)


def box(slide, label, x, y, w, h, color, size=20):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shp.fill.background()
    shp.line.color.rgb = rgb(color)
    shp.line.width = Pt(2.5)
    tf = shp.text_frame
    tf.word_wrap = True
    for i, line in enumerate(label.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = line
        r.font.size, r.font.name = Pt(size if i == 0 else size - 6), FONT
        r.font.bold = i == 0
        r.font.color.rgb = rgb(TEXT if i == 0 else TEXT_2)
    return shp


def add_looping_video(slide, video, poster):
    """Full-slide video that starts automatically and loops until the slide is left."""
    movie = slide.shapes.add_movie(str(video), 0, 0, W, H, poster_frame_image=str(poster),
                                   mime_type="video/mp4")  # fmt: skip
    spid = movie.shape_id
    ns = "http://schemas.openxmlformats.org/presentationml/2006/main"
    timing = f"""
<p:timing xmlns:p="{ns}"><p:tnLst><p:par>
 <p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot"><p:childTnLst>
  <p:seq concurrent="1" nextAc="seek"><p:cTn id="2" dur="indefinite" nodeType="mainSeq">
   <p:childTnLst><p:par><p:cTn id="3" fill="hold"><p:stCondLst>
    <p:cond delay="indefinite"/><p:cond evt="onBegin" delay="0"><p:tn val="2"/></p:cond>
   </p:stCondLst><p:childTnLst><p:par><p:cTn id="4" fill="hold"><p:stCondLst>
    <p:cond delay="0"/></p:stCondLst><p:childTnLst><p:par>
     <p:cTn id="5" presetID="1" presetClass="mediacall" presetSubtype="0" fill="hold"
            nodeType="afterEffect"><p:stCondLst><p:cond delay="0"/></p:stCondLst>
      <p:childTnLst><p:cmd type="call" cmd="playFrom(0.0)"><p:cBhvr>
       <p:cTn id="6" dur="1" fill="hold"/><p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>
      </p:cBhvr></p:cmd></p:childTnLst></p:cTn></p:par>
    </p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn></p:par></p:childTnLst></p:cTn>
   <p:prevCondLst><p:cond evt="onPrev" delay="0"><p:tgtEl><p:sldTgt/></p:tgtEl></p:cond>
   </p:prevCondLst><p:nextCondLst><p:cond evt="onNext" delay="0"><p:tgtEl><p:sldTgt/>
   </p:tgtEl></p:cond></p:nextCondLst></p:seq>
  <p:video><p:cMediaNode vol="0"><p:cTn id="7" repeatCount="indefinite" fill="hold"
   display="0"><p:stCondLst><p:cond delay="indefinite"/></p:stCondLst></p:cTn>
   <p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cMediaNode></p:video>
 </p:childTnLst></p:cTn></p:par></p:tnLst></p:timing>"""
    sld = slide._element
    for old in sld.findall(f"{{{ns}}}timing"):
        sld.remove(old)
    new = etree.fromstring(timing)
    ext = sld.find(f"{{{ns}}}extLst")  # timing must come before extLst
    if ext is not None:
        ext.addprevious(copy.deepcopy(new))
    else:
        sld.append(new)


def build():
    FIG.mkdir(exist_ok=True)
    fig_task(FIG / "task.png")
    for m in MODELS:
        fig_prediction(m, FIG / f"pred_{m}.png")
    fig_per_variable(FIG / "per_variable.png")
    first_frame(HERO, FIG / "hero_poster.png")

    cv = {m: metric(f"cv_{m}.json", "cv_mape")["score"] for m in [*MODELS, "null_per_day"]}
    v7 = json.loads((METRICS / "hybrid_v7.json").read_text())

    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H

    # 1 ---------------------------------------------------------------------------------
    s = new_slide(
        prs,
        notes=(
            "We were asked to predict how a CHO cell culture evolves over 13 days, from its "
            "first measurement and its feeding plan. Here is what we learned in one sentence: "
            "physics and machine learning fail in different ways, and combining them works best."
        ),
    )
    text(s, "Decode the Bioreactor", Inches(0.9), Inches(2.4), Inches(11.5), Inches(1.4),
         54, TEXT, bold=True)  # fmt: skip
    text(s, "Forecasting 13 days of cell culture from day 0 and the feed plan",
         Inches(0.9), Inches(3.75), Inches(11.5), Inches(0.8), 24, TEXT_2)  # fmt: skip
    text(s, "Team 15 · SDSC STEM4Health hackathon", Inches(0.9), Inches(5.6), Inches(11.5),
         Inches(0.6), 18, TEXT_2)  # fmt: skip

    # 2 ---------------------------------------------------------------------------------
    s = new_slide(
        prs,
        "We forecast a whole culture run from its first day",
        notes=(
            "For each new batch we get only the day-0 measurements, four numbers, and the complete "
            "feed and volume log. We must predict viable and total cell density, substrate and "
            "byproduct on every day until day 13. The score is the mean absolute percent error. "
            "One subtlety: operators dose feed B according to the cell count, so the feed plan "
            "itself carries information about how the batch grew."
        ),
    )
    s.shapes.add_picture(str(FIG / "task.png"), Inches(0.6), Inches(1.65), width=Inches(8.6))
    text(s, "Given\nday-0 values\nthe feed plan\n\nPredict\n13 days × 4 variables",
         Inches(9.6), Inches(2.3), Inches(3.3), Inches(4), 22, TEXT)  # fmt: skip

    # 3 ---------------------------------------------------------------------------------
    m = "mechanistic_lite"
    s = new_slide(
        prs,
        "Physics alone gets the shape but misses the level",
        notes=(
            "Our first approach is a mechanistic model: differential equations for growth, death, "
            "substrate uptake and byproduct, with dilution by every feed. It gets the shape right "
            "and stays physically consistent, but a few shared parameters cannot capture how much "
            "batches differ. On its own it is worse than simply averaging past runs day by day "
            f"({cv['null_per_day']:.1f}%)."
        ),
    )
    s.shapes.add_picture(str(FIG / f"pred_{m}.png"), Inches(0.5), Inches(1.75), width=Inches(8.5))
    big_number(s, f"{cv[m]:.1f}%", "cross-validated error", Inches(9.3), Inches(2.2),
               MODELS[m][1], f"average past run: {cv['null_per_day']:.1f}%")  # fmt: skip

    # 4 ---------------------------------------------------------------------------------
    m = "ml_sensors"
    s = new_slide(
        prs,
        "Machine learning alone reads the batch but knows no physics",
        notes=(
            "Our second approach is pure machine learning: ridge regression and gradient boosting "
            "on features from the feed and volume logs. The key features are soft sensors, for "
            "example the volume not explained by the logged feeds. It captures batch differences "
            "well, but it has no notion of mass balance and partly learns that feeding follows "
            "growth, which would not hold for a new feeding strategy."
        ),
    )
    s.shapes.add_picture(str(FIG / f"pred_{m}.png"), Inches(0.5), Inches(1.75), width=Inches(8.5))
    big_number(s, f"{cv[m]:.1f}%", "cross-validated error", Inches(9.3), Inches(2.2),
               MODELS[m][1])  # fmt: skip

    # 5 ---------------------------------------------------------------------------------
    m = "hybrid_blend"
    s = new_slide(
        prs,
        "The hybrid lets physics draw the curve and learning correct it",
        notes=(
            "The hybrid keeps the mechanistic simulation as the backbone and learns only its "
            "errors: the ratio between the measurement and the simulation, from the same soft "
            "sensors. Learning a correction is a much easier problem than learning the whole "
            "trajectory, which is why it works with only 49 labelled batches."
        ),
    )
    y0, h = Inches(2.7), Inches(1.6)
    box(
        s,
        "Mechanistic ODE\nthe shape",
        Inches(0.7),
        y0,
        Inches(3.2),
        h,
        MODELS["mechanistic_lite"][1],
    )
    text(s, "×", Inches(3.95), y0 + Inches(0.3), Inches(0.7), Inches(1), 44, TEXT,
         align=PP_ALIGN.CENTER)  # fmt: skip
    box(
        s,
        "Learned correction\nthe level",
        Inches(4.7),
        y0,
        Inches(3.2),
        h,
        MODELS["ml_sensors"][1],
    )
    text(s, "=", Inches(7.95), y0 + Inches(0.3), Inches(0.7), Inches(1), 44, TEXT,
         align=PP_ALIGN.CENTER)  # fmt: skip
    box(s, "Forecast", Inches(8.7), y0, Inches(2.0), h, MODELS[m][1])
    text(s, "The correction is learned from the feed and volume logs (soft sensors)",
         Inches(0.7), Inches(4.7), Inches(10), Inches(0.6), 17, TEXT_2)  # fmt: skip
    text(s, f"{cv[m]:.2f}%", Inches(11.0), Inches(2.75), Inches(2.2), Inches(1), 40,
         MODELS[m][1], bold=True)  # fmt: skip
    text(s, "cross-validated\nerror", Inches(11.0), Inches(3.6), Inches(2.2), Inches(1), 15,
         TEXT_2)  # fmt: skip

    # 6 ---------------------------------------------------------------------------------
    s = new_slide(
        prs,
        "Each approach covers the other's weakness",
        notes=(
            "Per variable, the picture is clear. The mechanistic model is weakest everywhere, but "
            "it gives the hybrid its strength on substrate, which is governed by mass balance. "
            "Machine learning is strongest on byproduct, which is why, in the video at the end, "
            "it sometimes matches the hybrid."
        ),
    )
    s.shapes.add_picture(
        str(FIG / "per_variable.png"), Inches(1.97), Inches(1.65), width=Inches(9.39)
    )

    # 7 ---------------------------------------------------------------------------------
    lb = v7["leaderboard_mape"]["score"]
    s = new_slide(
        prs,
        "Stacking more models bought the last points, at a cost",
        notes=(
            "Our best leaderboard score came from Tobias' v7: four hybrids with different "
            "mechanistic cores, blended per variable and per phase of the run. The biggest single "
            "gain came from an idea, feeding the sensors into the ODE itself. The rest came from "
            "stacking, and that has costs: more code and compute, a submitted file that a rerun "
            "no longer reproduces exactly, and a model that no longer explains anything. Most of "
            "the remaining error is a per-batch offset that the inputs do not predict; more layers "
            "cannot recover it."
        ),
    )
    text(s, f"7.14 → {lb:.2f}%", Inches(0.9), Inches(2.0), Inches(6.5), Inches(1.3), 60,
         MODELS["hybrid_blend"][1], bold=True)  # fmt: skip
    text(s, "leaderboard error, 12 unseen test batches", Inches(0.9), Inches(3.3), Inches(6.5),
         Inches(0.6), 18, TEXT_2)  # fmt: skip
    text(s, "4 sub-models, ~2,000 lines of code\nharder to reproduce\nnothing left to interpret",
         Inches(8.1), Inches(2.1), Inches(4.8), Inches(2.5), 22, TEXT_2)  # fmt: skip

    # 8 ---------------------------------------------------------------------------------
    s = new_slide(
        prs,
        "Every number is reproducible with one command",
        notes=(
            "Everything we showed comes out of one DVC pipeline: data, frozen batch-wise split, "
            "code and parameters are versioned, models are registered with semantic versions, and "
            "a change is adopted only if it is better in at least four of five folds. Rejected "
            "ideas are logged too."
        ),
    )
    text(s, "dvc repro", Inches(0.9), Inches(2.1), Inches(11), Inches(1.4), 72, TEXT,
         bold=True, font="Courier New")  # fmt: skip
    text(s, "data, split, code and parameters versioned · models in a registry · "
         "changes kept only if better in 4 of 5 folds",
         Inches(0.9), Inches(3.7), Inches(11.5), Inches(1.2), 20, TEXT_2)  # fmt: skip

    # 9 ---------------------------------------------------------------------------------
    s = new_slide(
        prs,
        notes=(
            "To close: three holdout batches the models never saw. First the hybrid forecasting "
            "the run from day 0 and the feed plan; then the same picture as percent error, with "
            "the mechanistic and machine-learning models for comparison. Plays on loop."
        ),
    )
    add_looping_video(s, HERO, FIG / "hero_poster.png")

    prs.save(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
