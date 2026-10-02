"""Getting internal activations out of a model, one vector per layer."""

import numpy as np
import torch


@torch.no_grad()
def contextual_embeddings(lm, word, template, layer="all"):
    """Hidden state of `word` inside `template`, at every layer. (L+1, d)

    The word is located by CHARACTER offsets. Comparing token counts of the
    prefix and prefix+word instead is off by one whenever the prefix ends in a
    space, because GPT-2 style tokenizers emit that space as its own token -- the
    span then points at the token after the word, and every word silently returns
    the same vector.
    """
    if not getattr(lm.tokenizer, "is_fast", False):
        raise RuntimeError(f"{lm.name} needs a fast tokenizer for offset mapping")
    text = template.format(word)
    start = len(template.split("{}")[0])
    end = start + len(word)

    enc = lm.tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
    offsets = enc.pop("offset_mapping")[0].tolist()
    # tokens overlapping the word; e > s drops specials, which carry (0, 0)
    idx = [i for i, (s, e) in enumerate(offsets) if e > s and s < end and e > start]
    if not idx:
        raise ValueError(f"could not locate {word!r} in {text!r}")

    hs = torch.stack(lm.model(**enc, output_hidden_states=True).hidden_states)
    return hs[:, 0, idx, :].mean(1).numpy()


@torch.no_grad()
def sentence_states(lm, sentences, pool="mean"):
    """Whole-sentence representation at every layer. (L+1, n, d)

    pool matters for causal models: their early tokens have seen almost nothing,
    so averaging buries the content and only the last position has read the whole
    sentence. Measured on Qwen, meaning clustering peaks at 0.00 with mean pooling
    and 0.18 with last-token pooling.
    """
    reps = []
    for s in sentences:
        enc = lm.tokenizer(s, return_tensors="pt", truncation=True, max_length=256)
        hs = torch.stack(lm.model(**enc, output_hidden_states=True).hidden_states)[:, 0]
        reps.append((hs.mean(1) if pool == "mean" else hs[:, -1]).numpy())
    return np.stack(reps).transpose(1, 0, 2)


def extract_reps(lm, var):
    """Activations for a word-level Variable at every layer. (L+1, n_samples, d)"""
    reps = []
    for text, target in zip(var.texts, var.targets):
        i = text.find(target)
        tmpl = text[:i] + "{}" + text[i + len(target):]
        reps.append(contextual_embeddings(lm, target, tmpl))
    return np.stack(reps).transpose(1, 0, 2)
