"""Build figures/COUNTRIES.png: what the country embedding tracks.

Row 1  scatter of representational distance against the four clearest predictors
Row 2  marginal correlations, and one regression with all seven together
Row 3  how each weight moves with depth

## Choosing a layer

The layer is picked from the PARALLEL CORPUS, not from the country data: the layer
where sentences cluster most by meaning and least by language. That criterion
knows nothing about countries, so it cannot be tuned to the result.

It only works for a multilingual model, which is why mBERT is used here rather
than English-only BERT -- for BERT "meaning" never beats "language" and the rule
degenerates to L0, the embedding layer. GPT-2 is kept as the monolingual case and
its layer is the least-bad point of the same curve; it lands mid-network, which is
what matters.

That matters because an earlier rule picked GPT-2's L1, where rare country names
have large vectors and sit on the periphery (rho between familiarity and vector
norm is -0.44 at L1, -0.00 by L6). That is token-frequency geometry, not
geopolitics, and it flipped the sign of the familiarity weight. `lexical_score`
is reported so the chosen layer can be checked against it, and row 3 shows the
whole trajectory so no single layer carries the argument.

Significance comes from permuting country labels, never OLS standard errors:
pairwise distances are not independent observations.
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
from llmprobe.curves import cached_familiarity, semantic_layer
from llmprobe.embed import extract_reps
from llmprobe.models import load, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9992"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
MODELS = ["mbert", "gpt2", "qwen0.5b"]
COLOUR = {"mbert": BLUE, "gpt2": ORANGE, "qwen0.5b": AQUA}
SHORT = {"mbert": "mBERT", "gpt2": "GPT-2", "qwen0.5b": "Qwen2.5-0.5B"}

KEYS = ["econ", "ling", "geo", "fampair", "famdiff", "pop", "lex"]
LABEL = {"econ": "GDP gap", "ling": "Language family", "geo": "Geographic distance",
         "fampair": "How well known (pair mean)", "famdiff": "Difference in how well known",
         "pop": "Population gap", "lex": "Name spelling (control)"}
UNIT = {"econ": "|Δ log₁₀ GDP per capita|", "ling": "unshared branches (0–3)",
        "geo": "km between capitals", "famdiff": "|Δ log P(name)|"}
SCATTERS = ["econ", "ling", "geo", "famdiff"]
TRACES = ["econ", "ling", "geo", "famdiff"]


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
        ax.text(0, 1.025, sub, transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")


def legend(ax, **kw):
    for t in ax.legend(frameon=False, **kw).get_texts():
        t.set_color(INK2)




def lexical_score(reps, groups, fam):
    """Per layer: rho(familiarity, ||vector||). Strongly negative means rare names
    sit on the periphery -- token-frequency geometry rather than semantics."""
    return np.array([spearmanr(fam, np.linalg.norm(
        G.center(G.item_states(reps[L], groups)), axis=1)).statistic
        for L in range(len(reps))])



def predictors(model, names):
    """The seven pairwise predictors. Five are model-independent; the two
    familiarity terms are measured from that model itself."""
    geo, ling, econ = V.country_distances()
    lex = np.array([[1 - SequenceMatcher(None, a.lower(), b.lower()).ratio()
                     for b in names] for a in names])
    f = cached_familiarity(model, names)
    return {"econ": econ, "ling": ling, "geo": geo,
            "fampair": (f[:, None] + f[None, :]) / 2,
            "famdiff": np.abs(f[:, None] - f[None, :]),
            "pop": V.population_distance(), "lex": lex}


def _z(v):
    return (v - v.mean()) / v.std()


def weights(D, T, iu, n_perm=400, seed=0):
    """Standardised OLS weights with a label-permutation null, plus R²."""
    X = np.column_stack([_z(T[k][iu]) for k in KEYS])
    y = _z(D[iu])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum()
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    null = np.array([np.linalg.lstsq(X, _z(D[np.ix_(p, p)][iu]), rcond=None)[0]
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    pv = np.array([(np.sum(np.abs(null[:, j]) >= abs(b[j])) + 1) / (n_perm + 1)
                   for j in range(len(KEYS))])
    return b, pv, null.std(0), r2


def weights_by_layer(reps, groups, T, iu):
    """Standardised weights at every layer. (n_layers, n_predictors)"""
    X = np.column_stack([_z(T[k][iu]) for k in KEYS])
    return np.array([np.linalg.lstsq(
        X, _z(squareform(pdist(G.center(G.item_states(reps[L], groups)),
                               "euclidean"))[iu]), rcond=None)[0]
        for L in range(len(reps))])


def main():
    set_seed(0)
    names, *_ = V.country_table()
    var = V.build("countries")
    iu = np.triu_indices(len(names), 1)
    print("computing...", flush=True)

    T, D, marg, beta, pval, err, r2, traj, sel = {}, {}, {}, {}, {}, {}, {}, {}, {}
    for m in MODELS:
        T[m] = predictors(m, names)
        reps = extract_reps(load(m), var)
        L = sel[m] = semantic_layer(m)
        lex_s = lexical_score(reps, var.groups, cached_familiarity(m, names))
        D[m] = squareform(pdist(G.center(G.item_states(reps[L], var.groups)), "euclidean"))
        marg[m] = [G.mantel(D[m], T[m][k], n_perm=1000)[0] for k in KEYS]
        beta[m], pval[m], err[m], r2[m] = weights(D[m], T[m], iu)
        traj[m] = weights_by_layer(reps, var.groups, T[m], iu)
        print(f"  {SHORT[m]:13s} L{L}/{len(reps)-1}  R²={r2[m]:.2f}  "
              f"lexical score at this layer {lex_s[L]:+.2f}")
        print(f"  {'':13s} " + "  ".join(f"{k}={b:+.2f}{'*' if p < .05 else ''}"
                                         for k, b, p in zip(KEYS, beta[m], pval[m])), flush=True)

    j = KEYS.index("famdiff")
    for m in MODELS:
        print(f"  famdiff trace {SHORT[m]:13s} " +
              " ".join(f"{v:+.2f}" for v in traj[m][:, j]), flush=True)

    fig = plt.figure(figsize=(17.5, 14.0), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 4, hspace=0.5, wspace=0.32,
                          left=0.05, right=0.985, top=0.855, bottom=0.055)

    # ---- row 1: scatters
    ref = "mbert"
    y = _z(D[ref][iu])
    for col, k in enumerate(SCATTERS):
        ax = fig.add_subplot(gs[0, col], facecolor=SURFACE)
        x = T[ref][k][iu]
        ax.scatter(x, y, s=5, color=GREY, alpha=0.26, linewidth=0, zorder=2)
        if k == "ling":
            cx = np.unique(x)
            cy = [y[x == b].mean() for b in cx]
            se = [y[x == b].std() / np.sqrt((x == b).sum()) for b in cx]
        else:
            q = np.quantile(x, np.linspace(0, 1, 9))
            i = np.clip(np.digitize(x, q[1:-1]), 0, 7)
            cx = [x[i == j].mean() for j in range(8)]
            cy = [y[i == j].mean() for j in range(8)]
            se = [y[i == j].std() / np.sqrt(max((i == j).sum(), 1)) for j in range(8)]
        ax.errorbar(cx, cy, yerr=se, fmt="o-", color=COLOUR[ref], lw=2, ms=6,
                    capsize=3, mec="white", mew=1, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xlabel(UNIT[k])
        if col == 0:
            ax.set_ylabel("distance in the model (z)")
        axes_style(ax)
        title(ax, LABEL[k], f"ρ = {marg[ref][KEYS.index(k)]:+.2f}", size=10)

    # ---- row 2: marginal and joint
    w = 0.26
    xs = np.arange(len(KEYS))
    short = [LABEL[k].replace(" (control)", "\n(control)").replace(" (pair mean)", "\n(pair mean)")
             .replace("Difference in how well known", "Difference in\nhow well known")
             .replace("Geographic distance", "Geographic\ndistance")
             .replace("Language family", "Language\nfamily")
             .replace("Population gap", "Population\ngap") for k in KEYS]

    ax = fig.add_subplot(gs[1, 0:2], facecolor=SURFACE)
    for i, m in enumerate(MODELS):
        ax.bar(xs + (i - 1) * w, marg[m], width=w, color=COLOUR[m], label=SHORT[m], zorder=3)
    ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=7.2)
    ax.set_ylabel("Mantel ρ"); ax.axhline(0, color=MUTED, lw=0.9)
    legend(ax, fontsize=8, loc="lower left", ncol=3)
    axes_style(ax)
    title(ax, "Marginal correlation", "each predictor on its own")

    ax = fig.add_subplot(gs[1, 2:4], facecolor=SURFACE)
    for i, m in enumerate(MODELS):
        ax.bar(xs + (i - 1) * w, beta[m], width=w, yerr=err[m], color=COLOUR[m],
               label=f"{SHORT[m]}  (R²={r2[m]:.2f})", capsize=2.5,
               error_kw=dict(lw=0.9, ecolor=MUTED), zorder=3)
        for j, (b, p) in enumerate(zip(beta[m], pval[m])):
            if p < 0.05:
                ax.text(j + (i - 1) * w, b + (0.012 if b >= 0 else -0.03), "*",
                        ha="center", fontsize=10, color=INK2)
    ax.set_xticks(xs); ax.set_xticklabels(short, fontsize=7.2)
    ax.set_ylabel("standardised weight"); ax.axhline(0, color=MUTED, lw=0.9)
    legend(ax, fontsize=8, loc="lower left", ncol=3)
    axes_style(ax)
    title(ax, "All seven together", "* = p < 0.05 · error bars = permutation sd")

    # ---- row 3: weight against depth
    for col, k in enumerate(TRACES):
        ax = fig.add_subplot(gs[2, col], facecolor=SURFACE)
        j = KEYS.index(k)
        for m in MODELS:
            d = np.linspace(0, 1, traj[m].shape[0])
            ax.plot(d, traj[m][:, j], "-", color=COLOUR[m], lw=2, label=SHORT[m], zorder=3)
            x0 = sel[m] / (traj[m].shape[0] - 1)
            ax.plot(x0, traj[m][sel[m], j], "o", color=COLOUR[m], ms=8, mec="white",
                    mew=1.4, zorder=4)
        ax.axhline(0, color=MUTED, lw=0.9)
        ax.set_xlabel("relative depth")
        if col == 0:
            ax.set_ylabel("standardised weight")
            legend(ax, fontsize=7.4, loc="best")
        axes_style(ax)
        title(ax, LABEL[k], "dot = chosen layer", size=10)

    fig.text(0.05, 0.955, "What the country embedding tracks", fontsize=19,
             color=INK, fontweight="semibold")
    fig.text(0.05, 0.922,
             "Distance between 48 countries against seven predictors, including two "
             "measuring how well the model knows each country — the control for "
             "“it is just training-data volume”.",
             fontsize=10.5, color=INK2)
    fig.text(0.05, 0.012,
             "Each point is one country pair (1,128 of them); scatters are mBERT. The layer "
             "is chosen from the parallel corpus — where sentences cluster most by meaning "
             "and least by language — so it is picked without reference to the country data. "
             "Significance from permuting country labels, not OLS errors.",
             fontsize=8, color=MUTED)

    out = ROOT / "figures" / "COUNTRIES.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
