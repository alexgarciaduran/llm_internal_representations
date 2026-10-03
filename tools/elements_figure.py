"""Build figures/ELEMENTS.png: does a language model recover the periodic table?

The periodic table is the only two-dimensional ground truth in this repo. Every
other concept is a scale, a circle or a tree. Each element has a period (row) and
a group (column), and the two are independent of each other (rho = 0.03), so a
model could easily learn the list of elements in order without learning the grid.

Top    the real table, the predicted table, and the same activations as a UMAP
Bottom how well each axis is recovered, and how that changes with depth

Predictions are LEAVE-ONE-OUT: every element's position is predicted by a ridge
model that never saw it. Without that, 56 elements and hundreds of features would
reproduce any layout asked for.

Colour is atomic number throughout, except the last panel. Atomic number tracks
period almost perfectly (rho = +0.97) so it stands in for it, but it is
uncorrelated with group (rho = +0.00), so columns need a panel of their own.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut

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

# one sequential ramp per quantity, light to dark, never a rainbow. The palest
# end is cut off so low values stay visible on a near-white surface.
def ramp(name, lo=0.22):
    return matplotlib.colors.ListedColormap(
        plt.get_cmap(name)(np.linspace(lo, 1.0, 256)))


Z_CMAP, G_CMAP = ramp("Blues"), ramp("Oranges")
# a readable subset; 56 labels at a legible size do not fit
ANCHORS = {"hydrogen", "helium", "carbon", "oxygen", "neon", "sodium",
           "chlorine", "iron", "copper", "zinc", "silver", "iodine",
           "gold", "mercury", "lead", "tungsten"}


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


def scatter(ax, x, y, names, c, cmap):
    sc = ax.scatter(x, y, c=c, cmap=cmap,
                    s=95, edgecolor="white", linewidth=0.8, zorder=3)
    for nm, xi, yi in zip(names, x, y):
        if nm in ANCHORS:
            ax.annotate(nm, (xi, yi), fontsize=6.4, color=INK2, ha="center",
                        va="bottom", xytext=(0, 7), textcoords="offset points", zorder=4)
    return sc


def bare(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_color(GRID)


def loo_predict(X, targets):
    """Leave-one-out ridge predictions of each target. (n, n_targets)"""
    P = np.zeros((X.shape[0], len(targets)))
    for tr, te in LeaveOneOut().split(X):
        for k, t in enumerate(targets):
            P[te, k] = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X[tr], t[tr]).predict(X[te])
    return P


def recovery_curve(model, var, period, group, z):
    """Held-out correlation for period, group and atomic number at every layer."""
    f = RES / f"elements_{model}.csv"
    if f.exists():
        return pd.read_csv(f)
    reps = extract_reps(load(model), var)
    rows = []
    for L in range(len(reps)):
        X = G.center(G.item_states(reps[L], var.groups))
        P = loo_predict(X, [period, group, z])
        rows.append({"L": L,
                     "period": spearmanr(P[:, 0], period).statistic,
                     "group": spearmanr(P[:, 1], group).statistic,
                     "z": spearmanr(P[:, 2], z).statistic})
    t = pd.DataFrame(rows)
    t.to_csv(f, index=False)
    return t


def main():
    set_seed(0)
    var = V.build("elements")
    names, z, period, group, block, cat = V.element_table()
    print("computing...", flush=True)
    print(f"  rho(atomic number, period) = {spearmanr(z, period).statistic:+.3f}   "
          f"rho(atomic number, group) = {spearmanr(z, group).statistic:+.3f}", flush=True)

    curves = {m: recovery_curve(m, var, period, group, z) for m in MODELS}
    best = {m: int(curves[m][["period", "group"]].mean(axis=1).idxmax()) for m in MODELS}
    for m in MODELS:
        r = curves[m].iloc[best[m]]
        print(f"  {SHORT[m]:13s} L{best[m]:<3d} period r={r.period:+.3f}  "
              f"group r={r['group']:+.3f}  atomic number r={r.z:+.3f}", flush=True)

    ref = max(MODELS, key=lambda m: curves[m][["period", "group"]].mean(axis=1).max())
    reps = extract_reps(load(ref), var)
    X = G.center(G.item_states(reps[best[ref]], var.groups))
    P = loo_predict(X, [period, group])

    fig = plt.figure(figsize=(21, 11), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 4, hspace=0.55, wspace=0.26,
                          left=0.045, right=0.985, top=0.82, bottom=0.07)

    ax1 = fig.add_subplot(gs[0, 0], facecolor=SURFACE)
    sc = scatter(ax1, group, period, names, z, Z_CMAP)
    ax2 = fig.add_subplot(gs[0, 1], facecolor=SURFACE)
    scatter(ax2, P[:, 1], P[:, 0], names, z, Z_CMAP)
    r = curves[ref].iloc[best[ref]]
    for ax, head, sub in [
            (ax1, "The real periodic table", f"{len(names)} elements, f-block excluded"),
            (ax2, f"Predicted from {SHORT[ref]}",
             f"layer {best[ref]}, held out, period r={r.period:+.2f}, "
             f"group r={r['group']:+.2f}")]:
        ax.invert_yaxis()                  # period 1 at the top, as in the table
        axes_style(ax, grid="both")
        ax.set_xlabel("group  (column)")
        ax.set_ylabel("period  (row)")
        title(ax, head, sub)

    # UMAP of the same activations: unsupervised, so the layout is not told what
    # to look for. What is read off it is whether colour organises across the map,
    # so the control is a label shuffle on that same map, not a noise matrix.
    # |r| because UMAP axis orientation is arbitrary; best-of-two-axes inflates it,
    # which is exactly what the null absorbs.
    U = G.project_2d(X, method="umap", seed=0)
    rng = np.random.default_rng(0)
    best_r = lambda v: max(abs(spearmanr(U[:, a], v).statistic) for a in (0, 1))

    heads = ["The same activations, as a UMAP", "Coloured by group instead"]
    panels = []
    for col, (v, nm, cmap) in enumerate([(z, "atomic number", Z_CMAP),
                                         (group, "group", G_CMAP)]):
        rv = best_r(v)
        null = np.array([best_r(rng.permutation(v)) for _ in range(2000)])
        ax = fig.add_subplot(gs[0, 2 + col], facecolor=SURFACE)
        s = scatter(ax, U[:, 0], U[:, 1], names, v, cmap)
        bare(ax)
        title(ax, heads[col],
              f"unsupervised · |r| = {rv:.2f}, shuffled {np.median(null):.2f}, "
              f"p = {(null >= rv).mean():.3f}")
        panels.append((ax, s))
        print(f"  UMAP {nm:14s} |r|={rv:.3f}  shuffled median={np.median(null):.3f}  "
              f"95th={np.quantile(null, .95):.3f}  p={(null >= rv).mean():.4f}", flush=True)

    cb = fig.colorbar(sc, ax=[ax1, ax2, panels[0][0]], orientation="horizontal",
                      fraction=0.045, pad=0.13, aspect=70)
    cb.set_label("atomic number  (one colour code, these three panels)",
                 color=INK2, fontsize=8.5)
    cbg = fig.colorbar(panels[1][1], ax=panels[1][0], orientation="horizontal",
                       fraction=0.045, pad=0.13, aspect=22)
    cbg.set_label("group  (column)", color=INK2, fontsize=8.5)
    for c in (cb, cbg):
        c.ax.tick_params(colors=MUTED, labelsize=7.5)
        c.outline.set_edgecolor(GRID)

    ax = fig.add_subplot(gs[1, 0:2], facecolor=SURFACE)
    w, xs = 0.26, np.arange(3)
    for i, m in enumerate(MODELS):
        r = curves[m].iloc[best[m]]
        ax.bar(xs + (i - 1) * w, [r.period, r["group"], r.z], width=w,
               color=COLOUR[m], label=SHORT[m], zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(["period\n(row)", "group\n(column)", "atomic number\n(the 1-D list)"],
                       fontsize=8.5)
    ax.set_ylabel("held-out correlation")
    ax.axhline(0, color=MUTED, lw=0.9)
    legend(ax, fontsize=8, loc="upper left", ncol=3)
    axes_style(ax)
    title(ax, "Which axis is recovered",
          "group is independent of atomic number, so it is the real test")

    ax = fig.add_subplot(gs[1, 2:4], facecolor=SURFACE)
    for m in MODELS:
        t = curves[m]
        d = np.linspace(0, 1, len(t))
        ax.plot(d, t[["period", "group"]].mean(axis=1), "-", color=COLOUR[m], lw=2.2,
                label=SHORT[m], zorder=3)
        ax.plot(best[m] / (len(t) - 1), t[["period", "group"]].mean(axis=1).iloc[best[m]],
                "o", color=COLOUR[m], ms=8, mec="white", mew=1.4, zorder=4)
    ax.axhline(0, color=MUTED, lw=0.9)
    ax.set_xlabel("relative depth  (0 = first layer, 1 = last)")
    ax.set_ylabel("mean of period and group r")
    legend(ax, fontsize=8, loc="lower center")
    axes_style(ax)
    title(ax, "Where in the network", "dot = best layer")

    fig.text(0.045, 0.955, "Does a language model recover the periodic table?",
             fontsize=19, color=INK, fontweight="semibold")
    fig.text(0.045, 0.908,
             "Each element's position is predicted from internal activations by a model "
             "that never saw that element. The table has two independent axes, so learning "
             "the list of elements in order would not produce this.",
             fontsize=10.5, color=INK2)
    fig.text(0.045, 0.015,
             "Atomic number stands in for period (rho = +0.97 between them) but says nothing "
             "about group (rho = +0.00), which is why columns keep a panel of their own. "
             "Lanthanides and actinides are excluded: they have no standard group.",
             fontsize=8, color=MUTED)

    out = ROOT / "figures" / "ELEMENTS.png"
    fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
