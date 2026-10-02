"""Measuring the geometry of a set of activations."""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def item_states(reps, groups):
    """Average each item's sentence frames into one vector. (n_items, d)

    We want the geometry of items, not of sentences, and averaging also cancels
    most of the template's own contribution.
    """
    return np.stack([reps[groups == g].mean(0) for g in np.unique(groups)])


def center(X):
    """Subtract the centre of mass.

    Transformer states sit in a narrow off-origin cone, so the dominant direction
    is just "where the cloud is". We do NOT L2-normalise afterwards: that maps a
    1-D line onto two antipodal directions and destroys the metric.
    """
    return X - X.mean(0, keepdims=True)


def mantel(a, b, n_perm=1000, seed=0):
    """Correlate two distance matrices, with a permutation test over item labels.

    The entries of a distance matrix are not independent of each other, so an
    ordinary correlation p-value would be badly wrong here.
    """
    n = a.shape[0]
    iu = np.triu_indices(n, 1)
    av, bv = a[iu], b[iu]
    obs = spearmanr(av, bv).statistic
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        p = rng.permutation(n)
        null[i] = spearmanr(av, b[np.ix_(p, p)][iu]).statistic
    return float(obs), float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1))


def cluster_separation(X, labels, n_perm=300, seed=0):
    """Silhouette score with a shuffled-label null. Returns (score, p, null mean).

    Silhouette per point is (b - a) / max(a, b), where a is the mean distance to
    its own group and b the mean distance to the nearest other group: +1 is tight
    separate groups, 0 is complete overlap. Raw values drift with dimensionality
    and class count, so the null is the number that matters.

    Distances are computed once and reused across permutations -- rebuilding them
    inside the loop is what made this slow.
    """
    from sklearn.metrics import silhouette_score

    D = squareform(pdist(X))
    obs = float(silhouette_score(D, labels, metric="precomputed"))
    rng = np.random.default_rng(seed)
    null = np.array([silhouette_score(D, rng.permutation(labels), metric="precomputed")
                     for _ in range(n_perm)])
    return obs, float((np.sum(null >= obs) + 1) / (n_perm + 1)), float(null.mean())


def project_2d(X, method="umap", seed=0):
    """2-D view. PCA is linear and invents nothing; UMAP reveals curved structure
    but will produce tidy arcs and clusters from noise, especially with few
    points."""
    if method == "pca":
        from sklearn.decomposition import PCA
        return PCA(n_components=2, random_state=seed).fit_transform(X)
    import umap
    k = max(2, min(10, X.shape[0] - 1))
    return umap.UMAP(n_components=2, n_neighbors=k, min_dist=0.3,
                     random_state=seed, n_jobs=1).fit_transform(X)
