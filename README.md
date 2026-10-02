# Inside the representations of language models

![summary](figures/SUMMARY.png)

What the **internal layers** of open-weight language models encode about living
things, meaning and places — and where in the network it lives.

Runs on **CPU**, uses **open-weight models only**, and is **deterministic**: no
sampling anywhere, and every statistic is tested against a permutation null.

## Reproduce

```bash
pip install -e .
python tools/summary_figure.py
python tools/countries_figure.py
```

First run downloads four models (~2 GB, cached in `~/.cache/huggingface`) and
writes per-layer statistics to `results/`. Later runs reuse that cache. No data
files to fetch.

## What it shows

**Living things.** 75 organisms across five kingdoms. Representational distance
rises with taxonomic distance, from *dog–wolf* to *dog–bacterium* — **except for
fungi**, which the models place nearer plants than their actual relatives.
Opisthokonta makes a mushroom genuinely closer to a dog than an oak is; the models
disagree, organising life by everyday similarity rather than phylogeny.

**Meaning across languages.** 30 meanings × 6 languages. Mid-network, each tight
cluster is one *meaning* holding all six languages — a shared semantic space.
Language identity is strongest at the bottom and top of the network and weakest in
the middle.

**Countries.** Distances between 48 countries are compared against four external
ground truths that correlate with each other at |ρ| < 0.2, so each can be credited
separately: great-circle distance between capitals, shared branches of the
language family tree, the gap in GDP per capita, and — as a control — how
similarly the names are spelled.

![countries](figures/COUNTRIES.png)

Geography, language and wealth all score ~0.2–0.4; spelling scores much lower.
The multiple regression asks the sharper question — whether each still explains
anything once the others are accounted for — and the answer is yes for geography
and GDP in every model.

## How it works

Each item goes into fixed sentence frames; the sentence is run through the model;
the target word's tokens are located by **character offset** and mean-pooled into
one vector per layer; items are averaged over their frames and centred. Then
either representational **distances** are correlated against a ground truth
(Mantel test, permutation null) or categories are scored by **silhouette** against
a shuffled-label null.

Three details that decide whether any of it is real:

- **Character offsets, not token counting.** Comparing token counts of `prefix`
  and `prefix + word` is off by one whenever the prefix ends in a space, and every
  word then silently returns the vector of the following token.
- **Pooling matters for causal models.** Their early tokens have seen almost
  nothing. On Qwen, meaning clustering peaks at 0.00 with mean pooling and 0.18
  with last-token pooling.
- **Permutation nulls everywhere.** Pairwise distances are not independent
  observations, so ordinary p-values and OLS standard errors would be far too
  small.

## Layout

```
src/llmprobe/
  models.py      model registry, CPU-safe cached loading
  embed.py       per-layer activations for a word or a whole sentence
  geometry.py    distances, Mantel test, silhouette, 2-D projection
  variables.py   the three concepts and their ground truths
  stimuli/       taxonomy.yaml, countries.yaml, parallel.yaml
tools/
  summary_figure.py     -> figures/SUMMARY.png
  countries_figure.py   -> figures/COUNTRIES.png
```

Stimuli are plain YAML — adding an organism, a country or a sentence is an edit,
not a code change.

## Limitations

Four models differing in many ways at once, so architectural explanations are
hypotheses rather than demonstrated causes. Small stimulus sets (48–180 items).
Taxonomic rank depth and language-family depth are coarse proxies for genetic and
cultural distance. GDP figures are approximate, rounded, around 2023. Probes show
that information is *available* in a representation, not that the model *uses* it.
