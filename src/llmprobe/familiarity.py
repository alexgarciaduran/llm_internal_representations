"""How well the model knows a name -- a proxy for how much training text mentions it.

familiarity(name) = total log P(name tokens | "The")

An LM's probability for a string is its estimate of that string's corpus
frequency, which is exactly the confound we need to control: a model may separate
two countries simply because it has read far more about one of them.

Two variants were tried and rejected, both diagnosable from their rankings:
  * mean log-prob per token inside a frame -- rewards multi-word names (correlates
    +0.52 with token count; "Saudi Arabia" came top, "Netherlands" last because the
    frame "I travelled to ___" is ungrammatical without an article);
  * PMI against a contentless prefix -- inverts the ranking, because PMI divides
    out the base rate and here the base rate IS the signal.
The variant below correlates -0.10 with token count and ranks the United States,
China, Canada and Russia highest, which is the expected answer.
"""

import numpy as np
import torch
import torch.nn.functional as F

PREFIX = "The"


@torch.no_grad()
def _familiarity_masked(lm, names, prefix=PREFIX):
    """Pseudo-log-likelihood: mask each token of the name in turn and sum the log
    probability of the true token. A masked model has no left-to-right
    probability, so this is the standard substitute (Salazar et al., 2020)."""
    from .models import load_mlm

    model, tok = load_mlm(lm.name)
    out = []
    for name in names:
        ids = tok(prefix + " " + name, return_tensors="pt")["input_ids"]
        n_p = tok(prefix, return_tensors="pt")["input_ids"].shape[1] - 1
        total = 0.0
        for i in range(n_p, ids.shape[1] - 1):
            masked = ids.clone()
            true = ids[0, i].item()
            masked[0, i] = tok.mask_token_id
            total += float(F.log_softmax(model(masked).logits[0, i].float(), dim=-1)[true])
        out.append(total)
    return np.array(out)


@torch.no_grad()
def familiarity(lm, names, prefix=PREFIX):
    """Total log P(name | prefix) for each name. (n,) -- higher = better known."""
    if lm.kind == "masked":
        return _familiarity_masked(lm, names, prefix)
    out = []
    ip = lm.tokenizer(prefix, return_tensors="pt")["input_ids"]
    n_p = ip.shape[1]
    for name in names:
        ids = lm.tokenizer(prefix + " " + name, return_tensors="pt")["input_ids"]
        lp = F.log_softmax(lm.model(ids).logits.float(), dim=-1)
        tgt = ids[0, n_p:]
        out.append(float(lp[0, n_p - 1:-1, :].gather(-1, tgt.unsqueeze(-1)).sum()))
    return np.array(out)


def familiarity_distance(lm, names):
    """|Δ familiarity| -- pairs where one country is well known and the other is not.

    This is the pairwise form that slots into the distance framework. Note it does
    not capture the other possible effect, that two equally obscure countries both
    get generic representations and so sit close together; `familiarity` itself is
    returned so that can be checked separately.
    """
    f = familiarity(lm, names)
    return np.abs(f[:, None] - f[None, :]), f
