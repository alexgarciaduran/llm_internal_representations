"""Build figures/EMOTIONS.png: is the affective circumplex in there?

287 emotion-bearing words scored against the NRC VAD lexicon (Mohammad, ACL 2018)
-- the only ground truth in this repo that I did not write myself. People rated
~20,000 English words for valence (unpleasant to pleasant), arousal (calm to
excited) and dominance.

The circumplex is two-dimensional, like the periodic table, and the same design
rule applies: the two axes have to be independent or recovering one implies the
other. Across the whole lexicon valence and arousal correlate at -0.27, so the
stimulus set is drawn evenly from a grid of valence x arousal cells, which brings
that to +0.02 within the set.

Dominance is kept but is NOT independent: it correlates +0.55 with valence even
after the grid sampling, so a good dominance score is largely a valence score.

Scoring: out-of-fold ridge predictions from 10-fold CV, Spearman r, with the same
probe on shuffled labels as the null.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

from llmprobe import quiet  # noqa: F401
from llmprobe import geometry as G, variables as V
from llmprobe.embed import extract_reps
from llmprobe.models import load, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
MODELS = ["mbert", "gpt2", "qwen0.5b"]
COLOUR = {"mbert": BLUE, "gpt2": ORANGE, "qwen0.5b": AQUA}
SHORT = {"mbert": "mBERT", "gpt2": "GPT-2", "qwen0.5b": "Qwen2.5-0.5B"}

# valence runs unpleasant <-> pleasant through a neutral middle, so it is
# diverging: two hues with a grey midpoint. Arousal is a magnitude, so one hue.
VAL_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "valence", [ORANGE, "#dcdbd6", BLUE])
ARO_CMAP = matplotlib.colors.ListedColormap(
    plt.get_cmap("Greens")(np.linspace(0.2, 1.0, 256)))

TARGETS = ["valence", "arousal", "dominance"]
ALPHAS = np.logspace(-1, 4, 12)
ANCHORS = {"suicide", "funeral", "sick", "cruel", "healthy", "great", "abundant",
           "education", "table", "ceiling", "dry", "sexual", "atomic", "paper"}


def axes_style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.set_axisbelow(True)
    for g in (["x", "y"] if grid == "both" else [grid]):
        getattr(ax, f"{g}axis").grid(True, color=GRID, lw=0.8)
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_color(INK2)
        lab.set_fontsize(8.5)


def title(ax, head, sub=None, size=11):
    ax.text(0, 1.10 if sub else 1.035, head, transform=ax.transAxes, fontsize=size,
            color=INK, fontweight="semibold", va="bottom")
    if sub:
        ax.text(0, 1.022, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def legend(ax, **kw):
    for t in ax.legend(frameon=False, **kw).get_texts():
        t.set_color(INK2)


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_color(GRID)


def oof(X, y):
    """Out-of-fold ridge predictions, 10-fold."""
    return cross_val_predict(RidgeCV(alphas=ALPHAS), X, y,
                             cv=KFold(10, shuffle=True, random_state=0))


def curve(model, var, vals):
    f = RES / f"emotions_{model}.csv"
    if f.exists():
        return pd.read_csv(f)
    reps = extract_reps(load(model), var)
    rng = np.random.default_rng(0)
    rows = []
    for L in range(len(reps)):
        X = G.center(G.item_states(reps[L], var.groups))
        row = {"L": L}
        for k, y in zip(TARGETS, vals):
            row[k] = spearmanr(oof(X, y), y).statistic
            row[k + "_shuf"] = np.mean(
                [abs(spearmanr(oof(X, p := rng.permutation(y)), p).statistic)
                 for _ in range(3)])
        rows.append(row)
        print(f"    {model} L{L}: " + "  ".join(f"{k}={row[k]:+.3f}" for k in TARGETS),
              flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(f, index=False)
    return t


def main():
    set_seed(0)
    var = V.build("emotions")
    words, valence, arousal, dominance = V.emotion_table()
    vals = [valence, arousal, dominance]
    print("computing...", flush=True)
    print(f"  {len(words)} words · rho(valence, arousal) = "
          f"{spearmanr(valence, arousal).statistic:+.3f} in the set, "
          f"-0.268 across the whole lexicon", flush=True)
    print(f"  rho(valence, dominance) = {spearmanr(valence, dominance).statistic:+.3f}",
          flush=True)

    curves = {m: curve(m, var, vals) for m in MODELS}
    best = {m: int(curves[m][["valence", "arousal"]].mean(axis=1).idxmax()) for m in MODELS}
    for m in MODELS:
        r = curves[m].iloc[best[m]]
        print(f"  {SHORT[m]:13s} L{best[m]:<3d} " + "  ".join(
            f"{k}={r[k]:+.3f} (shuf {r[k + '_shuf']:.3f})" for k in TARGETS), flush=True)

    ref = max(MODELS, key=lambda m: curves[m][["valence", "arousal"]].mean(axis=1).max())
    reps = extract_reps(load(ref), var)
    X = G.center(G.item_states(reps[best[ref]], var.groups))
    P = np.column_stack([oof(X, valence), oof(X, arousal)])
    U = G.project_2d(X, method="umap", seed=0)

    fig = plt.figure(figsize=(21, 11), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.28,
                          left=0.045, right=0.985, top=0.81, bottom=0.07)

    def label_some(ax, x, y):
        for w, xi, yi in zip(words, x, y):
            if w in ANCHORS:
                ax.annotate(w, (xi, yi), fontsize=6.4, color=INK2, ha="center",
                            va="bottom", xytext=(0, 7), textcoords="offset points",
                            zorder=4)

    r = curves[ref].iloc[best[ref]]
    for col, (x, y, head, sub) in enumerate([
            (valence, arousal, "The rated circumplex",
             f"{len(words)} words · axes independent by design (rho = "
             f"{spearmanr(valence, arousal).statistic:+.2f})"),
            (P[:, 0], P[:, 1], f"Predicted from {SHORT[ref]}",
             f"layer {best[ref]}, held out · valence r={r.valence:+.2f}, "
             f"arousal r={r.arousal:+.2f}")]):
        ax = fig.add_subplot(gs[0, col], facecolor=SURFACE)
        ax.scatter(x, y, c=valence, cmap=VAL_CMAP, s=42, edgecolor="white",
                   linewidth=0.6, zorder=3)
        label_some(ax, x, y)
        axes_style(ax, grid="both")
        ax.set_xlabel("valence   (unpleasant → pleasant)")
        ax.set_ylabel("arousal   (calm → excited)")
        title(ax, head, sub)

    for col, (v, nm, cmap) in enumerate([(valence, "valence", VAL_CMAP),
                                         (arousal, "arousal", ARO_CMAP)]):
        ax = fig.add_subplot(gs[0, 2 + col], facecolor=SURFACE)
        sc = ax.scatter(U[:, 0], U[:, 1], c=v, cmap=cmap, s=42, edgecolor="white",
                        linewidth=0.6, zorder=3)
        label_some(ax, U[:, 0], U[:, 1])
        bare(ax)
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045,
                          pad=0.06, aspect=26)
        cb.set_label(nm, color=INK2, fontsize=8.5)
        cb.ax.tick_params(colors=MUTED, labelsize=7.5)
        cb.outline.set_edgecolor(GRID)
        best_r = lambda t: max(abs(spearmanr(U[:, a], t).statistic) for a in (0, 1))
        rv = best_r(v)
        rng = np.random.default_rng(0)
        null = np.array([best_r(rng.permutation(v)) for _ in range(2000)])
        title(ax, f"UMAP, coloured by {nm}",
              f"unsupervised · |r| with best axis = {rv:.2f} · "
              f"shuffled {np.median(null):.2f} · p = {(null >= rv).mean():.3f}")

    ax = fig.add_subplot(gs[1, 0:2], facecolor=SURFACE)
    w, xs = 0.26, np.arange(len(TARGETS))
    for i, m in enumerate(MODELS):
        r = curves[m].iloc[best[m]]
        ax.bar(xs + (i - 1) * w, [r[k] for k in TARGETS], width=w, color=COLOUR[m],
               label=SHORT[m], zorder=3)
        ax.bar(xs + (i - 1) * w, [r[k + "_shuf"] for k in TARGETS], width=w,
               color="none", edgecolor=MUTED, lw=1.1, ls=":", zorder=4,
               label="shuffled labels" if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(["valence", "arousal", "dominance\n(r = +0.55 with valence)"],
                       fontsize=8.5)
    ax.set_ylabel("held-out correlation")
    ax.set_ylim(0, 0.95)                     # headroom, so the legend clears the bars
    ax.axhline(0, color=MUTED, lw=0.9)
    legend(ax, fontsize=8, loc="upper right", ncol=4)
    axes_style(ax)
    title(ax, "What is recovered",
          "arousal is the real test: it is uncorrelated with valence in this set")

    ax = fig.add_subplot(gs[1, 2:4], facecolor=SURFACE)
    for m in MODELS:
        t = curves[m]
        d = np.linspace(0, 1, len(t))
        ax.plot(d, t.valence, "-", color=COLOUR[m], lw=2.2, label=SHORT[m], zorder=3)
        ax.plot(d, t.arousal, "--", color=COLOUR[m], lw=1.8, zorder=3)
        ax.plot(best[m] / (len(t) - 1), t.valence.iloc[best[m]], "o", color=COLOUR[m],
                ms=8, mec="white", mew=1.4, zorder=4)
    h = [matplotlib.lines.Line2D([], [], color=MUTED, lw=2.2),
         matplotlib.lines.Line2D([], [], color=MUTED, lw=1.8, ls="--")]
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xlabel("relative depth  (0 = first layer, 1 = last)")
    ax.set_ylabel("held-out correlation")
    axes_style(ax)
    for t in ax.legend(ax.get_legend_handles_labels()[0][:0] + h
                       + [matplotlib.lines.Line2D([], [], color=COLOUR[m], lw=2.2)
                          for m in MODELS],
                       ["valence", "arousal"] + [SHORT[m] for m in MODELS],
                       frameon=False, fontsize=8, loc="lower right", ncol=2).get_texts():
        t.set_color(INK2)
    title(ax, "Where in the network", "dot = layer used above")

    fig.text(0.045, 0.95, "Does a model place words on the emotional map?",
             fontsize=19, color=INK, fontweight="semibold")
    fig.text(0.045, 0.9,
             "287 words, rated by people for valence and arousal in the NRC VAD lexicon "
             "— the only ground truth here that I did not write. The set is sampled so "
             "the two axes are independent, making arousal a separate claim from valence.",
             fontsize=10.5, color=INK2)
    fig.text(0.045, 0.015,
             "Lexicon: Mohammad, ACL 2018, downloaded on first run and cached, not "
             "redistributed. Words are restricted to those mBERT encodes as a single "
             "wordpiece, a crude frequency filter. Frames are part-of-speech agnostic "
             "because the lexicon mixes adjectives, nouns and verbs.",
             fontsize=8, color=MUTED)

    out = ROOT / "figures" / "EMOTIONS.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
