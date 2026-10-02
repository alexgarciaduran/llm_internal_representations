"""The three concepts, each as (texts, labels, item ids) plus its ground truths."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import yaml

STIM = Path(__file__).parent / "stimuli"


@lru_cache(maxsize=None)
def _load(name):
    return yaml.safe_load((STIM / f"{name}.yaml").read_text(encoding="utf-8"))


@dataclass
class Variable:
    name: str
    texts: list
    y: np.ndarray
    groups: np.ndarray        # item identity, so items can be averaged
    level: str                # "word" | "sentence"
    targets: list = None


# ------------------------------------------------------------------ taxonomy

def taxonomy_table():
    t = _load("taxonomy")["organisms"]
    names = list(t)
    return names, np.array([t[n] for n in names]), _load("taxonomy")["ranks"]


def taxonomy_distance():
    """5 minus the number of leading taxonomic ranks two organisms share.

    0 dog-wolf, 1 dog-mouse, 2 dog-eagle, 3 dog-mushroom, 4 dog-oak,
    5 dog-bacterium. Opisthokonta groups animals with fungi, so a mushroom is
    genuinely closer to a dog than an oak is. A coarse proxy for genetic
    distance, but an external one.
    """
    _, ranks, _ = taxonomy_table()
    n, depth = ranks.shape
    shared = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k = 0
            while k < depth and ranks[i, k] == ranks[j, k]:
                k += 1
            shared[i, j] = k
    return depth - shared


def _taxonomy():
    cfg = _load("taxonomy")
    names, ranks, levels = taxonomy_table()
    texts, targets, y, groups = [], [], [], []
    for i, nm in enumerate(names):
        for t in cfg["templates"]:
            texts.append(t.format(nm))
            targets.append(nm)
            y.append(ranks[i, levels.index("class")])
            groups.append(i)
    return Variable("taxonomy", texts, np.array(y), np.array(groups), "word", targets)


# ----------------------------------------------------------------- countries

def country_table():
    c = _load("countries")["countries"]
    names = list(c)
    arr = lambda i, t=float: np.array([c[n][i] for n in names], dtype=t)
    return (names, arr(0), arr(1), np.array([c[n][2] for n in names]),
            np.array([c[n][3:6] for n in names]), arr(6))


def country_distances():
    """(geographic km, linguistic, economic) distance matrices.

    Geography is the great-circle distance between capitals. Linguistic distance
    is 3 minus the number of leading language-family ranks shared, a standard
    proxy for cultural distance. Economic distance is |Δ log10 GDP per capita| --
    log because GDP spans two orders of magnitude.

    These three correlate with each other at |rho| < 0.2, so each can be credited
    separately.
    """
    _, lat, lon, _, langs, gdp = country_table()
    la, lo = np.radians(lat), np.radians(lon)
    a = (np.sin((la[:, None] - la[None, :]) / 2) ** 2
         + np.cos(la)[:, None] * np.cos(la)[None, :]
         * np.sin((lo[:, None] - lo[None, :]) / 2) ** 2)
    geo = 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

    n, depth = langs.shape
    shared = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            k = 0
            while k < depth and langs[i, k] == langs[j, k]:
                k += 1
            shared[i, j] = k

    lg = np.log10(gdp)
    return geo, depth - shared, np.abs(lg[:, None] - lg[None, :])


def _countries():
    cfg = _load("countries")
    names, _, _, region, _, _ = country_table()
    texts, targets, y, groups = [], [], [], []
    for i, nm in enumerate(names):
        for t in cfg["templates"]:
            texts.append(t.format(nm))
            targets.append(nm)
            y.append(region[i])
            groups.append(i)
    return Variable("countries", texts, np.array(y), np.array(groups), "word", targets)


# ------------------------------------------------------------ parallel corpus

def parallel_corpus():
    """180 sentences with three labellings of the same points.

    language (6) is surface form; meaning (30) is translation equivalence, an
    identity relation; topic (6) adds the graded structure that meaning labels
    cannot express.
    """
    v = _load("parallel")
    langs, par = v["languages"], v["parallel"]
    texts, language, meaning, topic = [], [], [], []
    for i, row in enumerate(par):
        for lg in langs:
            texts.append(row[lg])
            language.append(lg)
            meaning.append(f"m{i:02d}")
            topic.append(row["topic"])
    return {"texts": texts, "language": np.array(language),
            "meaning": np.array(meaning), "topic": np.array(topic),
            "languages": langs, "n_meanings": len(par)}



BUILDERS = {"taxonomy": _taxonomy, "countries": _countries}


def build(name):
    return BUILDERS[name]()
