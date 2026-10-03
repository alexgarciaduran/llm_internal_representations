"""Per-layer curves, cached to results/, and the rule for picking one layer.

Both figures use these, so the layer rule is defined once. Keeping two copies is
what let the two figures drift onto different rules in the first place.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from . import geometry as G
from . import variables as V
from .embed import sentence_states
from .familiarity import familiarity
from .models import load

RESULTS = Path(__file__).resolve().parents[2] / "results"
RESULTS.mkdir(exist_ok=True)


def _cached(name, build):
    f = RESULTS / f"{name}.csv"
    if f.exists():
        return pd.read_csv(f)
    df = build()
    df.to_csv(f, index=False)
    return df


def meaning_curve(model):
    """Silhouette by language / meaning / topic at every layer."""
    lm = load(model)
    pool = "last" if lm.kind == "causal" else "mean"

    def build():
        pc = V.parallel_corpus()
        reps = sentence_states(lm, pc["texts"], pool=pool)
        return pd.DataFrame([
            {"layer": L,
             **{lab: G.cluster_separation(G.center(reps[L]), pc[lab], n_perm=300)[0]
                for lab in ("language", "meaning", "topic")}}
            for L in range(len(reps))])

    return _cached(f"meaning_{model}_{pool}", build)


def semantic_layer(model):
    """The layer where sentences cluster most by meaning and least by language.

    Chosen from the parallel corpus, so it knows nothing about whatever is being
    measured with it. Only well defined for a multilingual model: for an
    English-only one "meaning" never beats "language" and this degenerates to the
    least-bad point of the curve.
    """
    t = meaning_curve(model)
    return int(np.argmax((t["meaning"] - t["language"]).values))


def taxonomy_curve(model, tax, D_tax):
    """Mantel rho against taxonomic distance at every layer."""
    from scipy.spatial.distance import pdist, squareform
    from .embed import extract_reps

    def build():
        reps = extract_reps(load(model), tax)
        return pd.DataFrame([
            {"L": L, "tax_rho": G.mantel(squareform(pdist(
                G.center(G.item_states(reps[L], tax.groups)), "euclidean")),
                D_tax, n_perm=500)[0]} for L in range(len(reps))])

    return _cached(f"taxonomy_{model}", build)


def cached_familiarity(model, names, concept):
    """How well the model knows each name. Cached; see familiarity.py.

    The cache key includes `concept`: keying on the model alone handed the 48
    countries back for a 56-element query.
    """
    f = RESULTS / f"familiarity_{concept}_{model}.csv"
    if f.exists():
        d = pd.read_csv(f)
        if list(d.name) == list(names):
            return d.value.values
    v = familiarity(load(model), names)
    pd.DataFrame({"name": names, "value": v}).to_csv(f, index=False)
    return v
