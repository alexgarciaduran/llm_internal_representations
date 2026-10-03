"""Build figures/COUNTRIES.png: what the country embedding tracks.

Top    scatter of representational distance against the four clearest predictors
Bottom marginal correlations, and one regression with all seven together

Marginal correlation asks "does the model track this at all?". The regression asks
the sharper question: "does it still explain anything once the others are
accounted for?" -- which is the only way to tell a real effect from a proxy for
national prominence.

Significance comes from permuting country labels, not from OLS standard errors:
pairwise distances are not independent observations (each country appears in n-1
pairs), so the usual errors would be far too small.
"""

from difflib import SequenceMatcher
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from llmprobe import quiet  # noqa: F401
from llmprobe import geometry as G, variables as V
from llmprobe.embed import extract_reps
from llmprobe.familiarity import familiarity
from llmprobe.models import load, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9992"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
MODELS = ["bert", "gpt2", "qwen0.5b"]
COLOUR = {"bert": BLUE, "gpt2": ORANGE, "qwen0.5b": AQUA}
SHORT = {"bert": "BERT", "gpt2": "GPT-2", "qwen0.5b": "Qwen2.5-0.5B"}

KEYS = ["econ", "ling", "geo", "fampair", "famdiff", "pop", "lex"]
LABEL = {"econ": "GDP gap", "ling": "Language family", "geo": "Geographic distance",
         "fampair": "How well known (pair mean)", "famdiff": "Difference in how well known",
         "pop": "Population gap", "lex": "Name spelling (control)"}
UNIT = {"econ": "|Δ log₁₀ GDP per capita|", "ling": "unshared branches (0–3)",
        "geo": "km between capitals",
        "famdiff": "|Δ log P(name)| — one well known, one not"}
SCATTERS = ["famdiff", "econ", "ling", "geo"]          # the four clearest, by |rho|


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


def country_layer(curve):
    """One layer per model: the highest mean agreement over geo/ling/econ. Letting
    each predictor pick its own best layer inflates it -- that is a maximum over
    13-25 chances."""
    return int(curve.loc[curve[["geo", "ling", "econ"]].mean(axis=1).idxmax(), "L"])


def cached_familiarity(model, names):
    f = RES / f"familiarity_{model}.csv"
    if f.exists():
        return pd.read_csv(f).value.values
    v = familiarity(load(model), names)
    pd.DataFrame({"country": names, "value": v}).to_csv(f, index=False)
    return v


def predictors(model, names):
    """All seven pairwise predictors for one model. Five are model-independent;
    the two familiarity terms are measured from that model itself."""
    geo, ling, econ = V.country_distances()
    lex = np.array([[1 - SequenceMatcher(None, a.lower(), b.lower()).ratio()
                     for b in names] for a in names])
    f = cached_familiarity(model, names)
    return {"econ": econ, "ling": ling, "geo": geo,
            "fampair": (f[:, None] + f[None, :]) / 2,
            "famdiff": np.abs(f[:, None] - f[None, :]),
            "pop": V.population_distance(), "lex": lex}


def weights(D, T, iu, n_perm=400, seed=0):
    """Standardised OLS weights with a label-permutation null, plus R²."""
    z = lambda v: (v - v.mean()) / v.std()
    X = np.column_stack([z(T[k][iu]) for k in KEYS])
    y = z(D[iu])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r2 = 1 - ((y - X @ b) ** 2).sum() / (y ** 2).sum()
    rng = np.random.default_rng(seed)
    n = D.shape[0]
    null = np.array([np.linalg.lstsq(X, z(D[np.ix_(p, p)][iu]), rcond=None)[0]
                     for p in (rng.permutation(n) for _ in range(n_perm))])
    pv = [(np.sum(np.abs(null[:, j]) >= abs(b[j])) + 1) / (n_perm + 1)
          for j in range(len(KEYS))]
    return b, np.array(pv), null.std(0), r2


def main():
    set_seed(0)
    names, *_ = V.country_table()
    var = V.build("countries")
    iu = np.triu_indices(len(names), 1)
    print("computing...", flush=True)

    T, D, marg, beta, pval, err, r2 = {}, {}, {}, {}, {}, {}, {}
    for m in MODELS:
        T[m] = predictors(m, names)
        L = country_layer(pd.read_csv(RES / f"countries_{m}.csv"))
        reps = extract_reps(load(m), var)
        D[m] = squareform(pdist(G.center(G.item_states(reps[L], var.groups)), "euclidean"))
        marg[m] = [G.mantel(D[m], T[m][k], n_perm=1000)[0] for k in KEYS]
        beta[m], pval[m], err[m], r2[m] = weights(D[m], T[m], iu)
        print(f"  {SHORT[m]:13s} L{L}  R²={r2[m]:.2f}  " +
              "  ".join(f"{k}={b:+.2f}{'*' if p < .05 else ''}"
                        for k, b, p in zip(KEYS, beta[m], pval[m])), flush=True)

    fig = plt.figure(figsize=(17.5, 9.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.32,
                          left=0.05, right=0.985, top=0.82, bottom=0.095)

    ref = "bert"
    y = (D[ref][iu] - D[ref][iu].mean()) / D[ref][iu].std()
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

    w = 0.26
    xs = np.arange(len(KEYS))
    short_lab = [LABEL[k].replace(" (control)", "\n(control)").replace(" (pair mean)", "\n(pair mean)")
                 .replace("Difference in how well known", "Difference in\nhow well known")
                 .replace("Geographic distance", "Geographic\ndistance")
                 .replace("Language family", "Language\nfamily")
                 .replace("Population gap", "Population\ngap") for k in KEYS]

    ax = fig.add_subplot(gs[1, 0:2], facecolor=SURFACE)
    for i, m in enumerate(MODELS):
        ax.bar(xs + (i - 1) * w, marg[m], width=w, color=COLOUR[m], label=SHORT[m], zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(short_lab, fontsize=7.2)
    ax.set_ylabel("Mantel ρ")
    ax.axhline(0, color=MUTED, lw=0.9)
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
    ax.set_xticks(xs)
    ax.set_xticklabels(short_lab, fontsize=7.2)
    ax.set_ylabel("standardised weight")
    ax.axhline(0, color=MUTED, lw=0.9)
    legend(ax, fontsize=8, loc="lower left", ncol=3)
    axes_style(ax)
    title(ax, "All seven together", "* = p < 0.05 · error bars = permutation sd")

    fig.text(0.05, 0.945, "What the country embedding tracks", fontsize=19,
             color=INK, fontweight="semibold")
    fig.text(0.05, 0.9,
             "Distance between 48 countries against seven predictors, including two "
             "measuring how well the model knows each country — the control for "
             "“it is just training-data volume”.",
             fontsize=10.5, color=INK2)
    fig.text(0.05, 0.016,
             "Each point is one country pair (1,128 of them); scatters are BERT. "
             "“How well known” is the model's own log-probability of the country name, "
             "a proxy for how much training text mentions it. Significance from "
             "permuting country labels, not OLS errors.",
             fontsize=8, color=MUTED)

    out = ROOT / "figures" / "COUNTRIES.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
