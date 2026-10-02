"""Build figures/SUMMARY.png. Results are cached to results/ on first run."""

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
from llmprobe.embed import extract_reps, sentence_states
from llmprobe.geometry import mantel
from llmprobe.models import load, set_seed

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
RES.mkdir(exist_ok=True)

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # validated palette slots 1-3
GREY = "#9a9992"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
MODELS = ["bert", "gpt2", "qwen0.5b"]
COLOUR = {"bert": BLUE, "gpt2": ORANGE, "qwen0.5b": AQUA, "mbert": BLUE}
SHORT = {"bert": "BERT", "gpt2": "GPT-2", "qwen0.5b": "Qwen2.5-0.5B", "mbert": "mBERT"}


def axes_style(ax, scatter=False, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(not scatter)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(False)
    if scatter:
        ax.set_xticks([])
        ax.set_yticks([])
        return
    ax.tick_params(colors=MUTED, labelsize=8, length=3, width=0.8)
    ax.set_axisbelow(True)
    getattr(ax, f"{grid}axis").grid(True, color=GRID, lw=0.8)
    for lab in (ax.xaxis.label, ax.yaxis.label):
        lab.set_color(INK2)
        lab.set_fontsize(9)


def title(ax, head, sub=None, size=11.5):
    ax.text(0, 1.10 if sub else 1.04, head, transform=ax.transAxes, fontsize=size,
            color=INK, fontweight="semibold", va="bottom")
    if sub:
        ax.text(0, 1.022, sub, transform=ax.transAxes, fontsize=8.5, color=MUTED,
                va="bottom")


def legend(ax, **kw):
    for t in ax.legend(frameon=False, **kw).get_texts():
        t.set_color(INK2)


def cached(name, build):
    f = RES / f"{name}.csv"
    if f.exists():
        return pd.read_csv(f)
    df = build()
    df.to_csv(f, index=False)
    return df


def tax_curve(model, tax, D_tax):
    def build():
        reps = extract_reps(load(model), tax)
        return pd.DataFrame([{"L": L, "tax_rho": mantel(squareform(pdist(
            G.center(G.item_states(reps[L], tax.groups)), "euclidean")),
            D_tax, n_perm=500)[0]} for L in range(len(reps))])
    return cached(f"taxonomy_{model}", build)


def meaning_curve(model, pc, pool):
    def build():
        reps = sentence_states(load(model), pc["texts"], pool=pool)
        return pd.DataFrame([
            {"layer": L, **{lab: G.cluster_separation(G.center(reps[L]), pc[lab],
                                                      n_perm=300)[0]
                            for lab in ("language", "meaning", "topic")}}
            for L in range(len(reps))])
    return cached(f"meaning_{model}_{pool}", build)


def country_curve(model, var, truths):
    def build():
        reps = extract_reps(load(model), var)
        rows = []
        for L in range(len(reps)):
            D = squareform(pdist(G.center(G.item_states(reps[L], var.groups)),
                                 "euclidean"))
            row = {"L": L}
            for nm, T in truths.items():
                row[nm], row[nm + "_p"] = mantel(D, T, n_perm=1000)
            rows.append(row)
        return pd.DataFrame(rows)
    return cached(f"countries_{model}", build)



def country_layer(curve):
    """One layer per model, chosen neutrally: the layer with the highest mean
    agreement across geography, language and economy.

    Letting each ground truth pick its own best layer inflates it -- that is a
    maximum over ~13-25 chances. Reading all of them at one layer keeps them
    comparable.
    """
    return int(curve[["geo", "ling", "econ"]].mean(axis=1).idxmax())


def main():
    set_seed(0)
    print("building panels...", flush=True)

    # ---- living things
    tax = V.build("taxonomy")
    D_tax = V.taxonomy_distance()
    org_names, ranks, _ = V.taxonomy_table()
    kingdom = ranks[:, 2]
    treps = extract_reps(load("bert"), tax)
    tc = tax_curve("bert", tax, D_tax)
    xbest = int(tc.loc[tc.tax_rho.idxmax(), "L"])
    Xtax = G.project_2d(G.center(G.item_states(treps[xbest], tax.groups)),
                        method="umap", seed=0)
    print(f"  living things  BERT L{xbest}", flush=True)

    # ---- meaning
    pc = V.parallel_corpus()
    mt = meaning_curve("mbert", pc, "mean")
    mbest = int(mt.meaning.idxmax())
    mreps = sentence_states(load("mbert"), pc["texts"], pool="mean")
    Xm = G.project_2d(G.center(mreps[mbest]), method="umap", seed=0)
    print(f"  meaning        mBERT L{mbest}", flush=True)

    # ---- countries, with three independent ground truths
    cvar = V.build("countries")
    cnames, _, _, region, _, _ = V.country_table()
    geo, ling, econ = V.country_distances()
    lex = np.array([[1 - SequenceMatcher(None, a.lower(), b.lower()).ratio()
                     for b in cnames] for a in cnames])
    truths = {"geo": geo, "ling": ling, "econ": econ, "lex": lex}
    curves = {m: country_curve(m, cvar, truths) for m in MODELS}
    creps = extract_reps(load("bert"), cvar)
    cbest = int(curves["bert"].loc[country_layer(curves["bert"]), "L"])
    Xc = G.project_2d(G.center(G.item_states(creps[cbest], cvar.groups)),
                      method="umap", seed=0)
    print(f"  countries      BERT L{cbest}", flush=True)

    fig = plt.figure(figsize=(18.5, 10.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 6, hspace=0.5, wspace=0.85,
                          left=0.045, right=0.985, top=0.84, bottom=0.085)

    # --- living things
    ax = fig.add_subplot(gs[0, 0:2], facecolor=SURFACE)
    for k, c, mk, lbl in [("animal", BLUE, "o", "animals"), ("plant", ORANGE, "o", "plants"),
                          ("fungus", AQUA, "^", "fungi"),
                          (("protist", "bacteria"), GREY, "s", "protists & bacteria")]:
        m = np.isin(kingdom, k if isinstance(k, tuple) else [k])
        ax.scatter(Xtax[m, 0], Xtax[m, 1], color=c, marker=mk, s=62,
                   edgecolor="white", linewidth=0.9, label=lbl, zorder=3)
    for w in ["dog", "wolf", "whale", "eagle", "shark", "bee", "oak", "pine",
              "rose", "wheat", "mushroom", "yeast", "amoeba", "bacterium"]:
        ax.annotate(w, Xtax[org_names.index(w)], fontsize=6.6, color=INK2,
                    xytext=(4, 3), textcoords="offset points")
    legend(ax, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.015), ncol=4,
           handletextpad=0.1, columnspacing=0.8)
    axes_style(ax, scatter=True)
    title(ax, "Living things")

    # --- meaning: two layers side by side, to show the flip
    early = 1          # language still outranks meaning here; it flips by L4
    inner_m = gs[0, 2:4].subgridspec(1, 2, wspace=0.08)
    marks = ["o", "s", "^", "D", "v", "P"]

    ax = fig.add_subplot(inner_m[0], facecolor=SURFACE)
    Xe = G.project_2d(G.center(mreps[early]), method="umap", seed=0)
    for i, lg in enumerate(pc["languages"]):
        m = pc["language"] == lg
        ax.scatter(Xe[m, 0], Xe[m, 1], color=[BLUE, ORANGE, AQUA][i % 3],
                   marker=marks[i], s=34, edgecolor="white", linewidth=0.4,
                   label=lg, zorder=3)
    legend(ax, fontsize=6.4, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.015),
           handletextpad=0.05, columnspacing=0.35)
    axes_style(ax, scatter=True)
    title(ax, f"Layer {early}: by language", size=10)

    ax = fig.add_subplot(inner_m[1], facecolor=SURFACE)
    topics = sorted(set(pc["topic"]))
    meanings = sorted(set(pc["meaning"]))
    topic_of = {m: pc["topic"][list(pc["meaning"]).index(m)] for m in meanings}
    rank = {m: i for i, m in enumerate(
        sorted(meanings, key=lambda k: (topics.index(topic_of[k]), k)))}
    cvals = np.array([rank[m] for m in pc["meaning"]], dtype=float)
    for lg, mk in zip(pc["languages"], marks):
        m = pc["language"] == lg
        ax.scatter(Xm[m, 0], Xm[m, 1], c=cvals[m], cmap="viridis", vmin=0,
                   vmax=len(rank) - 1, marker=mk, s=34, edgecolor="white",
                   linewidth=0.4, zorder=3)
    axes_style(ax, scatter=True)
    title(ax, f"Layer {mbest}: by meaning", size=10)

    # --- countries
    ax = fig.add_subplot(gs[0, 4:6], facecolor=SURFACE)
    regions = ["europe", "americas", "asia", "mideast", "africa", "oceania"]
    for i, rg in enumerate(regions):
        m = region == rg
        ax.scatter(Xc[m, 0], Xc[m, 1], color=[BLUE, ORANGE, AQUA][i % 3],
                   marker=["o", "o", "o", "^", "^", "^"][i], s=62,
                   edgecolor="white", linewidth=0.9, label=rg, zorder=3)
    for w in ["Portugal", "Spain", "Brazil", "Japan", "China", "Egypt", "Nigeria",
              "Russia", "Australia", "United States"]:
        ax.annotate(w, Xc[cnames.index(w)], fontsize=6.4, color=INK2,
                    xytext=(4, 3), textcoords="offset points")
    legend(ax, fontsize=7.6, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.015),
           handletextpad=0.1, columnspacing=0.5)
    axes_style(ax, scatter=True)
    title(ax, "Countries")

    # --- taxonomic distance
    ax = fig.add_subplot(gs[1, 0:2], facecolor=SURFACE)
    iu = np.triu_indices(len(org_names), 1)
    d = D_tax[iu]
    nb = int(d.max()) + 1
    for m in MODELS:
        tt = tax_curve(m, tax, D_tax)
        L = int(tt.loc[tt.tax_rho.idxmax(), "L"])
        r2 = extract_reps(load(m), tax)
        rep = squareform(pdist(G.center(G.item_states(r2[L], tax.groups))))[iu]
        rep = (rep - rep.mean()) / rep.std()
        ax.errorbar(range(nb), [rep[d == k].mean() for k in range(nb)],
                    yerr=[rep[d == k].std() / np.sqrt((d == k).sum()) for k in range(nb)],
                    fmt="o-", color=COLOUR[m], lw=2.2, ms=6, capsize=3, mec="white",
                    mew=1.1, label=f"{SHORT[m]} (ρ={tt.tax_rho.max():.2f})", zorder=3)
    ax.set_xticks(range(nb))
    ax.set_xticklabels(["same\ngroup", "same\nclass", "same\nkingdom",
                        "same\nsupergroup", "same\ndomain", "diff.\ndomain"],
                       fontsize=7)
    ax.set_xlabel("relatedness   (dog–wolf → dog–bacterium)")
    ax.set_ylabel("distance in the model (z across all pairs)")
    ax.annotate("fungi break it: mushrooms sit\nnearer plants than their kin",
                xy=(3, 0.70), xytext=(2.25, -1.25), fontsize=7.2, color=MUTED,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8))
    legend(ax, fontsize=7.6, loc="upper left")
    axes_style(ax)
    title(ax, "Distance tracks relatedness")

    # --- shared space for meaning
    ax = fig.add_subplot(gs[1, 2:4], facecolor=SURFACE)
    for m, pool in [("mbert", "mean"), ("qwen0.5b", "last")]:
        t = meaning_curve(m, pc, pool)
        x = np.linspace(0, 1, len(t))
        ax.plot(x, t.meaning, "-", color=COLOUR[m], lw=2.6,
                label=f"{SHORT[m]} — meaning", zorder=3)
        ax.plot(x, t.language, "--", color=COLOUR[m], lw=1.5, alpha=0.8,
                label=f"{SHORT[m]} — language", zorder=2)
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xlabel("relative depth  (0 = first layer, 1 = last)")
    ax.set_ylabel("cluster separation")
    legend(ax, fontsize=8, loc="upper left")
    axes_style(ax)
    title(ax, "A shared space for meaning")

    # --- what the country embedding tracks: one small panel per ground truth
    inner = gs[1, 4:6].subgridspec(4, 1, hspace=0.95)
    panels = [("geo", "Geographic distance between capitals"),
              ("ling", "Shared branches of the language family tree"),
              ("econ", "Gap in GDP per capita  (log scale)"),
              ("lex", "Spelling of the country names  (control)")]
    at = {m: country_layer(c) for m, c in curves.items()}
    hi = max(curves[m].loc[at[m], k] for m in MODELS for k, _ in panels)
    for row, (key, head) in enumerate(panels):
        ax = fig.add_subplot(inner[row], facecolor=SURFACE)
        vals = [curves[m].loc[at[m], key] for m in MODELS]
        bars = ax.barh(range(3), vals, height=0.6,
                       color=[GREY if key == "lex" else COLOUR[m] for m in MODELS],
                       zorder=3)
        for b, v in zip(bars, vals):
            ax.text(v + 0.012, b.get_y() + b.get_height() / 2, f"{v:.2f}",
                    va="center", fontsize=7.2, color=INK2)
        ax.set_yticks(range(3))
        ax.set_yticklabels([SHORT[m] for m in MODELS], fontsize=7, color=INK2)
        ax.invert_yaxis()
        ax.set_xlim(0, hi * 1.3)
        axes_style(ax, grid="x")
        ax.tick_params(axis="y", length=0)
        title(ax, head, size=9)
        if row == 3:
            ax.set_xlabel("agreement with the model's distances (ρ)")

    fig.text(0.045, 0.945, "Inside the representations of three language models",
             fontsize=20, color=INK, fontweight="semibold")
    fig.text(0.045, 0.903, "What the internal layers encode about living things, "
             "meaning and places — and where in the network it lives.",
             fontsize=11, color=INK2)
    fig.text(0.045, 0.016,
             f"UMAP of internal activity: {len(org_names)} organisms at BERT L{xbest}, "
             f"180 parallel sentences at mBERT L{early} and L{mbest}, "
             f"{len(cnames)} countries at BERT L{cbest}. The four country ground truths "
             "are nearly independent of one another (all |ρ| < 0.2), so each is credited "
             "separately. All statistics tested against permutation nulls.",
             fontsize=8, color=MUTED)

    out = ROOT / "figures" / "SUMMARY.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
