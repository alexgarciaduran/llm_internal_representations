"""Build figures/NUMBERS.png: the models know what numbers look like, not what they are.

The numbers 1-400. Magnitude is the axis any frequency effect would also produce,
so on its own it proves little. Last digit and parity are the real surface tests:
both are uncorrelated with magnitude by arithmetic, so no amount of knowing the
number line produces them. Divisibility and primality are the arithmetic tests.

Three controls do the work here:

  shuffled labels  the same probe on permuted targets, to show the CV is honest
  the last-digit sieve  every prime above 5 ends in 1, 3, 7 or 9, so knowing only
                   the final digit already scores 0.857 on primality. Primality is
                   therefore also scored on numbers ending in 1, 3, 7, 9 only,
                   where that cue is worth nothing.
  layer 0          the static embedding table. If a property is already maximal
                   there, the network did not compute it.

Qwen2.5-0.5B is excluded: its tokenizer splits numerals into single digits
("47" -> "4", "7"), so the last digit is literally a token inside the span and
decoding it would measure the tokenizer rather than the model.

Scoring: magnitude by 10-fold out-of-fold ridge predictions (Spearman r); the
categorical targets by 10-fold stratified CV with balanced class weights, which
matters because the positive classes are small and an unweighted classifier just
predicts the majority and scores exactly 0.500.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV, RidgeClassifierCV
from sklearn.metrics import balanced_accuracy_score
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
MODELS = ["mbert", "gpt2"]
COLOUR = {"mbert": BLUE, "gpt2": ORANGE}
SHORT = {"mbert": "mBERT", "gpt2": "GPT-2"}

# magnitude is a magnitude, so one hue light-to-dark. Last digit is CYCLIC, so it
# gets a cyclic map -- the one case where a ramp returning to its start is the
# honest choice, since 9 really is adjacent to 0.
MAG_CMAP = matplotlib.colors.ListedColormap(
    plt.get_cmap("Blues")(np.linspace(0.22, 1.0, 256)))
DIG_CMAP = plt.get_cmap("twilight")

BARS = ["parity", "last_digit", "div3", "prime", "prime_hard"]
CHANCE = {"parity": 0.5, "last_digit": 0.1, "div3": 0.5, "prime": 0.5, "prime_hard": 0.5}
LABEL = {"magnitude": "magnitude", "parity": "parity", "last_digit": "last digit\n(10 classes)",
         "div3": "divisible\nby 3", "prime": "prime\n(all)",
         "prime_hard": "prime\n(cue removed)"}
SURFACE_KEYS = {"parity", "last_digit"}        # what a tokenizer could tell you
ALPHAS = np.logspace(-1, 4, 12)


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


def ridge_r(X, y):
    """Out-of-fold ridge predictions, scored as Spearman r."""
    pred = cross_val_predict(RidgeCV(alphas=ALPHAS), X, y,
                             cv=KFold(10, shuffle=True, random_state=0))
    return spearmanr(pred, y).statistic


def clf_acc(X, y):
    """10-fold stratified CV. Balanced weights and balanced scoring, because the
    positive classes are small and an unweighted fit just predicts the majority."""
    scoring = "accuracy" if len(np.unique(y)) > 2 else "balanced_accuracy"
    return cross_val_score(RidgeClassifierCV(alphas=ALPHAS, class_weight="balanced"),
                           X, y, cv=StratifiedKFold(10, shuffle=True, random_state=0),
                           scoring=scoring).mean()


def curve(model, var, value, targets, masks):
    f = RES / f"numbers_{model}.csv"
    if f.exists():
        return pd.read_csv(f)
    reps = extract_reps(load(model), var)
    rng = np.random.default_rng(0)
    rows = []
    for L in range(len(reps)):
        X = G.center(G.item_states(reps[L], var.groups))
        row = {"L": L, "magnitude": ridge_r(X, value)}
        for k in BARS:
            m = masks.get(k, np.ones(len(value), bool))
            Xk, yk = X[m], targets[k][m]
            row[k] = clf_acc(Xk, yk)
            row[k + "_shuf"] = np.mean([clf_acc(Xk, rng.permutation(yk)) for _ in range(3)])
        rows.append(row)
        print(f"    {model} L{L}: " + "  ".join(
            f"{k}={row[k]:.3f}" for k in ("magnitude", "last_digit", "prime", "prime_hard")),
            flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(f, index=False)
    return t


def main():
    set_seed(0)
    var = V.build("numbers")
    names, value, targets, masks = V.number_table()
    sieve = np.isin(value.astype(int) % 10, [1, 3, 7, 9]).astype(int)
    sieve_score = balanced_accuracy_score(targets["prime"], sieve)
    print("computing...", flush=True)
    print(f"  last-digit sieve alone on primality = {sieve_score:.3f}", flush=True)

    curves = {m: curve(m, var, value, targets, masks) for m in MODELS}
    best = {m: int(curves[m].last_digit.idxmax()) for m in MODELS}
    for m in MODELS:
        r = curves[m].iloc[best[m]]
        print(f"  {SHORT[m]:7s} L{best[m]:<3d} " + "  ".join(
            f"{k}={r[k]:.3f}" for k in ["magnitude"] + BARS), flush=True)

    ref = max(MODELS, key=lambda m: curves[m].last_digit.max())
    reps = extract_reps(load(ref), var)
    X = G.center(G.item_states(reps[best[ref]], var.groups))
    U = G.project_2d(X, method="umap", seed=0)

    fig = plt.figure(figsize=(17, 10.5), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.28,
                          left=0.055, right=0.985, top=0.79, bottom=0.07)

    for col, (v, nm, cmap, sub) in enumerate([
            (value, "magnitude", MAG_CMAP, "pale = small, dark = large"),
            (targets["last_digit"], "last digit", DIG_CMAP, "cyclic: 9 and 0 are neighbours")]):
        ax = fig.add_subplot(gs[0, col], facecolor=SURFACE)
        sc = ax.scatter(U[:, 0], U[:, 1], c=v, cmap=cmap, s=26,
                        edgecolor="white", linewidth=0.4, zorder=3)
        bare(ax)
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.045,
                          pad=0.06, aspect=28)
        cb.set_label(nm, color=INK2, fontsize=8.5)
        cb.ax.tick_params(colors=MUTED, labelsize=7.5)
        cb.outline.set_edgecolor(GRID)
        title(ax, f"UMAP, coloured by {nm}",
              f"{SHORT[ref]} layer {best[ref]}, unsupervised · {sub}")

    ax = fig.add_subplot(gs[0, 2], facecolor=SURFACE)
    w, xs = 0.3, np.arange(len(BARS))
    for i, m in enumerate(MODELS):
        r = curves[m].iloc[best[m]]
        bars = ax.bar(xs + (i - 0.5) * w, [r[k] for k in BARS], width=w,
                      color=COLOUR[m], label=SHORT[m], zorder=3)
        for k, b in zip(BARS, bars):          # surface properties get a hatch
            if k in SURFACE_KEYS:
                b.set_hatch("///")
                b.set_edgecolor("white")
    for j, k in enumerate(BARS):
        ax.plot([xs[j] - 1.1 * w, xs[j] + 1.1 * w], [CHANCE[k]] * 2, "--",
                color=MUTED, lw=1.3, zorder=4, label="chance" if j == 0 else None)
    ax.axhline(sieve_score, color=INK2, lw=1.1, ls=":", zorder=4)
    ax.text(len(BARS) - 0.55, sieve_score + 0.015, "last-digit sieve alone",
            fontsize=7.5, color=INK2, ha="right")
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[k] for k in BARS], fontsize=7.2)
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0, 1.05)
    legend(ax, fontsize=8, loc="upper right", ncol=3)
    axes_style(ax)
    title(ax, "What is decodable", "hatched = readable from the numeral itself")

    for col, k in enumerate(["magnitude", "last_digit", "prime_hard"]):
        ax = fig.add_subplot(gs[1, col], facecolor=SURFACE)
        for m in MODELS:
            t = curves[m]
            d = np.linspace(0, 1, len(t))
            ax.plot(d, t[k], "-", color=COLOUR[m], lw=2.2, label=SHORT[m], zorder=3)
            if k != "magnitude":
                ax.plot(d, t[k + "_shuf"], ":", color=COLOUR[m], lw=1.5, zorder=2,
                        label=f"{SHORT[m]}, shuffled")
            ax.plot(0, t[k].iloc[0], "o", color=COLOUR[m], ms=7, mec="white", mew=1.4,
                    zorder=4)
        ax.axhline(CHANCE.get(k, 0), ls="--", color=MUTED, lw=1.2)
        ax.set_xlabel("relative depth  (0 = first layer, 1 = last)")
        ax.set_ylabel("held-out r" if k == "magnitude" else "held-out accuracy")
        if k != "magnitude":
            ax.set_ylim(0, 1.05)
        axes_style(ax)
        if col == 0:          # one legend, in this panel's empty middle band
            h = [matplotlib.lines.Line2D([], [], color=COLOUR[m], lw=2.2) for m in MODELS]
            h.append(matplotlib.lines.Line2D([], [], color=MUTED, lw=1.5, ls=":"))
            for t in ax.legend(h, [SHORT[m] for m in MODELS] + ["shuffled labels"],
                               frameon=False, fontsize=8.5, loc="center").get_texts():
                t.set_color(INK2)
        title(ax, LABEL[k].replace("\n", " "),
              "dot = layer 0, the embedding table")

    fig.text(0.055, 0.945, "What a model knows about a number",
             fontsize=19, color=INK, fontweight="semibold")
    fig.text(0.055, 0.895,
             "The numbers 1–400. Everything readable off the numeral itself is "
             "decodable almost perfectly — and is already maximal at layer 0, the "
             "embedding table. Everything that needs arithmetic is at chance.",
             fontsize=10.5, color=INK2)
    fig.text(0.055, 0.015,
             "Qwen2.5-0.5B is excluded: it tokenises “47” as “4”, “7”, so its last "
             "digit is an input token, not a representation. Magnitude: out-of-fold ridge, "
             "Spearman r. Categorical targets: 10-fold stratified CV, balanced class weights "
             "and balanced accuracy, except last digit (10 classes, plain accuracy).",
             fontsize=8, color=MUTED)

    out = ROOT / "figures" / "NUMBERS.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
