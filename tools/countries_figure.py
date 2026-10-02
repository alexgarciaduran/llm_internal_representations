"""Build figures/COUNTRIES.png: what the country embedding actually tracks.

Top row    scatter of representational distance against each ground truth (BERT)
Bottom row marginal correlations, and multiple-regression weights

Marginal correlation asks "does the model track geography at all?". The
regression asks the sharper question: "does geography still explain anything once
language, economy and spelling are accounted for?". With predictors this weakly
correlated the two mostly agree, and where they disagree the regression is the
one to believe.

Significance for both comes from permuting country labels, not from OLS standard
errors: pairwise distances are not independent observations (each country appears
in n-1 pairs), so the usual errors would be far too small.
"""

from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from llmprobe import quiet  # noqa: F401
from llmprobe import geometry as G, variables as V
from llmprobe.embed import extract_reps
from llmprobe.geometry import mantel
from llmprobe.models import load, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9992"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
MODELS = ["bert", "gpt2", "qwen0.5b"]
COLOUR = {"bert": BLUE, "gpt2": ORANGE, "qwen0.5b": AQUA}
SHORT = {"bert": "BERT", "gpt2": "GPT-2", "qwen0.5b": "Qwen2.5-0.5B"}
KEYS = ["geo", "ling", "econ", "lex"]
LABEL = {"geo": "Geographic distance", "ling": "Language family distance",
         "econ": "GDP gap (log scale)", "lex": "Name spelling (control)"}
UNIT = {"geo": "km between capitals", "ling": "unshared branches (0–3)",
        "econ": "|Δ log₁₀ GDP per capita|", "lex": "1 − string similarity"}


def axes_style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(False)
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.set_axisbelow(True)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_color(INK2)
        lab.set_fontsize(8.5)


def title(ax, head, sub=None, size=10.5):
    ax.text(0, 1.11 if sub else 1.04, head, transform=ax.transAxes, fontsize=size,
            color=INK, fontweight="semibold", va="bottom")
    if sub:
        ax.text(0, 1.025, sub, transform=ax.transAxes, fontsize=8, color=MUTED,
                va="bottom")


def legend(ax, **kw):
    for t in ax.legend(frameon=False, **kw).get_texts():
        t.set_color(INK2)



def country_layer(curve):
    """One layer per model, chosen neutrally: the layer with the highest mean
    agreement across geography, language and economy.

    Letting each ground truth pick its own best layer inflates it -- that is a
    maximum over ~13-25 chances. Reading all of them at one layer keeps them
    comparable.
    """
    return int(curve[["geo", "ling", "econ"]].mean(axis=1).idxmax())


def rep_distances(model, var, n_items):
    """Pairwise representational distance at the model's best geography layer."""
    t = pd.read_csv(RES / f"countries_{model}.csv")
    L = int(t.loc[country_layer(t), "L"])
    reps = extract_reps(load(model), var)
    D = squareform(pdist(G.center(G.item_states(reps[L], var.groups)), "euclidean"))
    return D, L


def regression_weights(D, truths, n_perm=500, seed=0):
    """Standardised OLS weights, with a label-permutation null for each.

    Permuting the country labels of the representational matrix preserves the
    structure of the predictors and destroys only their link to the model, which
    is exactly the null we want.
    """
    n = D.shape[0]
    iu = np.triu_indices(n, 1)
    z = lambda v: (v - v.mean()) / v.std()
    X = np.column_stack([z(truths[k][iu]) for k in KEYS])
    beta = np.linalg.lstsq(X, z(D[iu]), rcond=None)[0]

    rng = np.random.default_rng(seed)
    null = np.empty((n_perm, len(KEYS)))
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = np.linalg.lstsq(X, z(D[np.ix_(p, p)][iu]), rcond=None)[0]
    pvals = [(np.sum(np.abs(null[:, j]) >= abs(beta[j])) + 1) / (n_perm + 1)
             for j in range(len(KEYS))]
    return beta, np.array(pvals), null.std(0)


def main():
    set_seed(0)
    var = V.build("countries")
    names, _, _, _, _, _ = V.country_table()
    geo, ling, econ = V.country_distances()
    lex = np.array([[1 - SequenceMatcher(None, a.lower(), b.lower()).ratio()
                     for b in names] for a in names])
    truths = {"geo": geo, "ling": ling, "econ": econ, "lex": lex}
    iu = np.triu_indices(len(names), 1)
    print("computing...", flush=True)

    D, betas, pvals, errs, marg = {}, {}, {}, {}, {}
    for m in MODELS:
        D[m], L = rep_distances(m, var, len(names))
        betas[m], pvals[m], errs[m] = regression_weights(D[m], truths)
        marg[m] = [mantel(D[m], truths[k], n_perm=1000)[0] for k in KEYS]
        print(f"  {m:9s} L{L}  weights " +
              "  ".join(f"{k}={b:+.2f}{'*' if p < .05 else ''}"
                        for k, b, p in zip(KEYS, betas[m], pvals[m])), flush=True)

    fig = plt.figure(figsize=(17, 9.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.34,
                          left=0.05, right=0.985, top=0.83, bottom=0.085)

    # ---- top: one scatter per ground truth, for BERT
    ref = "bert"
    y = (D[ref][iu] - D[ref][iu].mean()) / D[ref][iu].std()
    for col, k in enumerate(KEYS):
        ax = fig.add_subplot(gs[0, col], facecolor=SURFACE)
        x = truths[k][iu]
        ax.scatter(x, y, s=5, color=GREY, alpha=0.28, linewidth=0, zorder=2)
        # binned mean, so the trend is visible through the cloud
        if k == "ling":
            bins = np.unique(x)
            cx = bins
            cy = [y[x == b].mean() for b in bins]
            se = [y[x == b].std() / np.sqrt((x == b).sum()) for b in bins]
        else:
            q = np.quantile(x, np.linspace(0, 1, 9))
            idx = np.clip(np.digitize(x, q[1:-1]), 0, 7)
            cx = [x[idx == i].mean() for i in range(8)]
            cy = [y[idx == i].mean() for i in range(8)]
            se = [y[idx == i].std() / np.sqrt(max((idx == i).sum(), 1)) for i in range(8)]
        ax.errorbar(cx, cy, yerr=se, fmt="o-", color=COLOUR[ref], lw=2, ms=6,
                    capsize=3, mec="white", mew=1, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(UNIT[k])
        if col == 0:
            ax.set_ylabel("distance in the model (z)")
        axes_style(ax)
        title(ax, LABEL[k], f"ρ = {marg[ref][col]:+.2f}", size=10)

    # ---- bottom left: marginal correlations
    ax = fig.add_subplot(gs[1, 0:2], facecolor=SURFACE)
    w = 0.26
    for i, m in enumerate(MODELS):
        ax.bar(np.arange(4) + (i - 1) * w, marg[m], width=w, color=COLOUR[m],
               label=SHORT[m], zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LABEL[k].replace(" (control)", "\n(control)").replace(
        " (log scale)", "") for k in KEYS], fontsize=8)
    ax.set_ylabel("Mantel ρ")
    ax.axhline(0, color=MUTED, lw=0.9)
    legend(ax, fontsize=8, loc="upper left", ncol=3)
    axes_style(ax)
    title(ax, "Marginal correlation", "each ground truth on its own")

    # ---- bottom right: regression weights
    ax = fig.add_subplot(gs[1, 2:4], facecolor=SURFACE)
    for i, m in enumerate(MODELS):
        ax.bar(np.arange(4) + (i - 1) * w, betas[m], width=w, yerr=errs[m],
               color=COLOUR[m], label=SHORT[m], capsize=2.5,
               error_kw=dict(lw=0.9, ecolor=MUTED), zorder=3)
        for j, (b, p) in enumerate(zip(betas[m], pvals[m])):
            if p < 0.05:
                ax.text(j + (i - 1) * w, b + (0.012 if b >= 0 else -0.028), "*",
                        ha="center", fontsize=10, color=INK2)
    ax.set_xticks(range(4))
    ax.set_xticklabels([LABEL[k].replace(" (control)", "\n(control)").replace(
        " (log scale)", "") for k in KEYS], fontsize=8)
    ax.set_ylabel("standardised weight")
    ax.axhline(0, color=MUTED, lw=0.9)
    legend(ax, fontsize=8, loc="upper left", ncol=3)
    axes_style(ax)
    title(ax, "Regression weights", "all four together · * = p < 0.05, error bars = permutation sd")

    fig.text(0.05, 0.945, "What the country embedding tracks", fontsize=19,
             color=INK, fontweight="semibold")
    fig.text(0.05, 0.902,
             "Representational distance between 48 countries, against four external "
             "ground truths. Scatters are BERT; bars compare all three models.",
             fontsize=10.5, color=INK2)
    fig.text(0.05, 0.016,
             "Each point is one country pair (1,128 of them). Significance comes from "
             "permuting country labels, not OLS errors — pairwise distances are not "
             "independent observations. The four ground truths correlate with each "
             "other at |ρ| < 0.2, so the weights are close to the marginal correlations.",
             fontsize=8, color=MUTED)

    out = ROOT / "figures" / "COUNTRIES.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
