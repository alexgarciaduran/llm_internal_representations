"""Loading models. CPU-first, eval mode, cached per process.

Nothing here samples. Every measurement is a hidden state from a single forward
pass, so results are exactly reproducible.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch

REGISTRY = {
    "bert": ("bert-base-uncased", "masked"),
    "mbert": ("bert-base-multilingual-cased", "masked"),
    "gpt2": ("gpt2", "causal"),
    "qwen0.5b": ("Qwen/Qwen2.5-0.5B-Instruct", "causal"),
}


@dataclass
class LoadedModel:
    name: str
    hf_id: str
    kind: str
    model: object
    tokenizer: object

    @property
    def n_layers(self):
        cfg = self.model.config
        return getattr(cfg, "num_hidden_layers", None) or cfg.n_layer

    def __repr__(self):
        return f"<{self.name}: {self.hf_id}, {self.kind}, {self.n_layers}L>"


def set_seed(seed=0):
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@lru_cache(maxsize=8)
def load(name):
    from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

    if name not in REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(REGISTRY)}")
    hf_id, kind = REGISTRY[name]
    tok = AutoTokenizer.from_pretrained(hf_id)
    cls = AutoModelForCausalLM if kind == "causal" else AutoModel
    model = cls.from_pretrained(hf_id, dtype=torch.float32).eval().to("cpu")
    if kind == "causal" and tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for p in model.parameters():
        p.requires_grad_(False)
    return LoadedModel(name, hf_id, kind, model, tok)


@lru_cache(maxsize=4)
def load_mlm(name):
    """Masked-LM head version, needed to score a string under a masked model."""
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    hf_id, kind = REGISTRY[name]
    if kind != "masked":
        raise ValueError(f"{name} is {kind}, not a masked LM")
    tok = AutoTokenizer.from_pretrained(hf_id)
    model = AutoModelForMaskedLM.from_pretrained(hf_id, dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok
