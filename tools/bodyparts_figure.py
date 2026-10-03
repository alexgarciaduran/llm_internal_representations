"""Build figures/BODYPARTS.png: is there a map of the body in there?

90 body parts with a vertical coordinate, 0 at the crown and 1 at the toes. The
question is whether a model that has only read about bodies lays them out in
order from head to foot.

Two controls, of unequal quality:

  internal   organs and bones against surfaces. Balanced (40/90) and essentially
             uncorrelated with height (rho = -0.10), so it is a real second axis.
  paired     two of them against one. Correlates +0.39 with height, because limbs
             are both lower down and paired, so it is a WEAK control and is
             labelled as such rather than presented as independent.

The ground truth here is one I wrote, unlike the periodic table or the NRC
lexicon. Coarse ordering only: whether liver sits above or below stomach in this
table is not worth defending.

Vertical position: 10-fold out-of-fold ridge, Spearman r. Categorical targets: 10-fold
stratified CV with balanced class weights and balanced accuracy -- without the
weighting an imbalanced target is "predicted" by always guessing the majority.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, cross_val_score

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

VERT_CMAP = matplotlib.colors.ListedColormap(
    plt.get_cmap("Blues")(np.linspace(0.22, 1.0, 256)))
CLASS = ["internal", "paired", "region"]
LABEL = {"vertical": "vertical position", "internal": "internal\n(organ / surface)",
         "paired": "paired\n(weak control)", "region": "region\n(7 classes)"}
ALPHAS = np.logspace(-1, 4, 12)
ANCHORS = {"hair", "brain", "eye", "tongue", "throat", "shoulder", "heart", "lung",
           "liver", "stomach", "navel", "hand", "finger", "hip", "thigh", "knee",
           "shin", "ankle", "foot", "toe"}


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


def oof_pred(X, y):
    """Out-of-fold ridge predictions, 10-fold.

    NOT leave-one-out. Under LOO a null target does not score zero: dropping the
    held-out point shifts the training mean away from it, so the predictions come
    out systematically ANTI-correlated -- the signed null mean here was -0.49 for
    mBERT. Ten folds shrink that, and the null is reported signed rather than as
    |r|, which is what turned a -0.49 artifact into a fake 0.43 "chance level".
    """
    return cross_val_predict(RidgeCV(alphas=ALPHAS), X, y,
                             cv=KFold(10, shuffle=True, random_state=0))


def clf_acc(X, y):
    return cross_val_score(RidgeClassifierCV(alphas=ALPHAS, class_weight="balanced"),
                           X, y, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                           scoring="balanced_accuracy").mean()


def curve(model, var, vertical, cats):
    f = RES / f"bodyparts_{model}.csv"
    if f.exists():
        return pd.read_csv(f)
    reps = extract_reps(load(model), var)
    rng = np.random.default_rng(0)
    rows = []
    for L in range(len(reps)):
        X = G.center(G.item_states(reps[L], var.groups))
        row = {"L": L, "vertical": spearmanr(oof_pred(X, vertical), vertical).statistic}
        null = np.array([spearmanr(oof_pred(X, p := rng.permutation(vertical)), p).statistic
                         for _ in range(8)])
        row["vertical_shuf"] = np.quantile(null, 0.95)       # signed, 95th percentile
        for k in CLASS:
            row[k] = clf_acc(X, cats[k])
            row[k + "_shuf"] = np.quantile(
                [clf_acc(X, rng.permutation(cats[k])) for _ in range(8)], 0.95)
        rows.append(row)
        print(f"    {model} L{L}: vertical={row['vertical']:+.3f}  "
              f"internal={row['internal']:.3f}  region={row['region']:.3f}", flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(f, index=False)
    return t


def main():
    set_seed(0)
    var = V.build("bodyparts")
    names, vertical, internal, paired, region = V.bodypart_table()
    cats = {"internal": internal, "paired": paired, "region": region}
    print("computing...", flush=True)
    print(f"  {len(names)} parts · rho(vertical, internal) = "
          f"{spearmanr(vertical, internal).statistic:+.3f} · rho(vertical, paired) = "
          f"{spearmanr(vertical, paired).statistic:+.3f}", flush=True)

    curves = {m: curve(m, var, vertical, cats) for m in MODELS}
    best = {m: int(curves[m].vertical.idxmax()) for m in MODELS}
    for m in MODELS:
        r = curves[m].iloc[best[m]]
        print(f"  {SHORT[m]:13s} L{best[m]:<3d} " + "  ".join(
            f"{k}={r[k]:+.3f} (shuf {r[k + '_shuf']:.3f})"
            for k in ["vertical"] + CLASS), flush=True)

    ref = max(MODELS, key=lambda m: curves[m].vertical.max())
    reps = extract_reps(load(ref), var)
    X = G.center(G.item_states(reps[best[ref]], var.groups))
    pred = oof_pred(X, vertical)
    U = G.project_2d(X, method="umap", seed=0)

    fig = plt.figure(figsize=(17, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.28,
                          left=0.055, right=0.985, top=0.79, bottom=0.07)

    def label_some(ax, x, y):
        for nm, xi, yi in zip(names, x, y):
            if nm in ANCHORS:
                ax.annotate(nm, (xi, yi), fontsize=6.4, color=INK2, ha="center",
                            va="bottom", xytext=(0, 6), textcoords="offset points",
                            zorder=4)

    r = curves[ref].iloc[best[ref]]
    ax = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    ax.scatter(vertical, pred, c=vertical, cmap=VERT_CMAP, s=60, edgecolor="white",
               linewidth=0.7, zorder=3)
    label_some(ax, vertical, pred)
    lim = [min(vertical.min(), pred.min()) - 0.03, max(vertical.max(), pred.max()) + 0.03]
    ax.plot(lim, lim, "--", color=MUTED, lw=1.1, zorder=2)
    ax.invert_xaxis()
    ax.invert_yaxis()
    axes_style(ax, grid="both")
    ax.set_xlabel("true position   (0 = crown, 1 = toes)")
    ax.set_ylabel("predicted position")
    title(ax, f"Head to foot, predicted from {SHORT[ref]}",
          f"layer {best[ref]}, held out · r = {r.vertical:+.2f} · "
          f"shuffled 95th pct {r.vertical_shuf:+.2f}")

    ax = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    sc = ax.scatter(U[:, 0], U[:, 1], c=vertical, cmap=VERT_CMAP, s=60,
                    edgecolor="white", linewidth=0.7, zorder=3)
    label_some(ax, U[:, 0], U[:, 1])
    bare(ax)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.06,
                      aspect=26)
    cb.set_label("vertical position", color=INK2, fontsize=8.5)
    cb.ax.tick_params(colors=MUTED, labelsize=7.5)
    cb.outline.set_edgecolor(GRID)
    rng = np.random.default_rng(0)
    best_r = lambda t: max(abs(spearmanr(U[:, a], t).statistic) for a in (0, 1))
    rv = best_r(vertical)
    null = np.array([best_r(rng.permutation(vertical)) for _ in range(2000)])
    title(ax, "UMAP, coloured by position",
          f"unsupervised · |r| with best axis = {rv:.2f} · "
          f"shuffled {np.median(null):.2f} · p = {(null >= rv).mean():.3f}")

    ax = fig.add_subplot(gs[0, 2], facecolor=SURFACE)
    keys = ["vertical"] + CLASS
    w, xs = 0.26, np.arange(len(keys))
    for i, m in enumerate(MODELS):
        rr = curves[m].iloc[best[m]]
        ax.bar(xs + (i - 1) * w, [rr[k] for k in keys], width=w, color=COLOUR[m],
               label=SHORT[m], zorder=3)
        ax.bar(xs + (i - 1) * w, [rr[k + "_shuf"] for k in keys], width=w, color="none",
               edgecolor=MUTED, lw=1.1, ls=":", zorder=4,
               label="shuffled labels" if i == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k] for k in keys], fontsize=7.6)
    ax.set_ylabel("r  (position)   /   balanced accuracy")
    ax.set_ylim(0, 1.05)
    legend(ax, fontsize=7.5, loc="upper right", ncol=2)
    axes_style(ax)
    title(ax, "What is recovered",
          "dotted outline = 95th percentile of the same probe on shuffled labels")

    for col, k in enumerate(["vertical", "internal", "region"]):
        ax = fig.add_subplot(gs[1, col], facecolor=SURFACE)
        for m in MODELS:
            t = curves[m]
            d = np.linspace(0, 1, len(t))
            ax.plot(d, t[k], "-", color=COLOUR[m], lw=2.2, label=SHORT[m], zorder=3)
            ax.plot(d, t[k + "_shuf"], ":", color=COLOUR[m], lw=1.4, zorder=2)
            ax.plot(best[m] / (len(t) - 1), t[k].iloc[best[m]], "o", color=COLOUR[m],
                    ms=7, mec="white", mew=1.4, zorder=4)
        ax.axhline(0.5 if k != "vertical" else 0, ls="--", color=MUTED, lw=1.1)
        ax.set_xlabel("relative depth  (0 = first layer, 1 = last)")
        ax.set_ylabel("held-out r" if k == "vertical" else "balanced accuracy")
        axes_style(ax)
        if col == 0:
            h = [matplotlib.lines.Line2D([], [], color=COLOUR[m], lw=2.2) for m in MODELS]
            h.append(matplotlib.lines.Line2D([], [], color=MUTED, lw=1.4, ls=":"))
            for t in ax.legend(h, [SHORT[m] for m in MODELS] + ["shuffled"],
                               frameon=False, fontsize=8, loc="lower right").get_texts():
                t.set_color(INK2)
        title(ax, LABEL[k].replace("\n", " "), "dot = layer used above")

    fig.text(0.055, 0.945, "Is there a map of the body in there?",
             fontsize=19, color=INK, fontweight="semibold")
    fig.text(0.055, 0.895,
             "90 body parts, each with a vertical coordinate from crown to toes. "
             "Whether organs and bones separate from surfaces is the second axis, and "
             "it is nearly uncorrelated with height, so it is a real one.",
             fontsize=10.5, color=INK2)
    fig.text(0.055, 0.015,
             "The coordinates are ones I wrote, unlike the periodic table or the NRC "
             "lexicon, so only coarse ordering is meant. “Paired” is a weak control: it "
             "correlates +0.39 with height, since limbs are both lower and paired. Frames "
             "force the anatomical sense of palm, sole, temple, arch and calf.",
             fontsize=8, color=MUTED)

    out = ROOT / "figures" / "BODYPARTS.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
