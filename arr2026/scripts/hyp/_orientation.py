"""One definition of the model->human orientation map, imported everywhere.

The released corpus stores human answers reverse-recoded on 55 of the 120
IPIP-NEO items so that a higher number always means more of the trait; the
simulator is shown the literal item and answers raw. Any model-vs-human
comparison must put the two on one scale first.

Variance-shaped statistics (VR, the within/between decomposition) are invariant
to a per-item sign flip, which is why they were unaffected and why the defect
went unnoticed for so long. Correlations, Wasserstein distances, classifier
inputs and absolute-error scores are NOT invariant.

Every analysis that compares model answers to human answers imports from here,
so the transform cannot drift between scripts again.
"""
import csv
from pathlib import Path

import numpy as np

_KEY = Path(__file__).resolve().parents[3] / "data/IPIP-NEO/120/item_key.csv"


def reverse_keyed_mask(n_items: int = 120) -> np.ndarray:
    """Boolean mask of the items the corpus stores reverse-recoded."""
    neg = np.zeros(n_items, dtype=bool)
    for r in csv.DictReader(open(_KEY)):
        if str(r["reverse"]).strip().lower() == "true":
            neg[int(r["item"]) - 1] = True
    return neg


NEG = reverse_keyed_mask()


def to_human_orientation(model_answers: np.ndarray) -> np.ndarray:
    """Map raw model answers onto the corpus's recoded scale (5-point items)."""
    out = np.asarray(model_answers, dtype=float).copy()
    out[:, NEG] = 6.0 - out[:, NEG]
    return out
