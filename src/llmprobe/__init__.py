"""Probing what the internal layers of language models encode.

    variables.build("taxonomy" | "countries")   -> stimuli + item ids
    embed.extract_reps(model, var)              -> (layers, samples, dims)
    geometry.*                                  -> distances, clustering, 2-D views
"""

__version__ = "1.0.0"

from . import embed, geometry, models, variables
from .models import load, set_seed

__all__ = ["embed", "geometry", "models", "variables", "load", "set_seed"]
