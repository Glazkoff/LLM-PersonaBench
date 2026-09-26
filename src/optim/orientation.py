"""One orientation map for icml2027. The corpus stores human answers reverse-recoded on 55/120
items; the model answers the literal item. Every model-vs-human comparison goes through here."""
import csv
from pathlib import Path
import numpy as np

_KEY = Path(__file__).resolve().parents[2] / "data/IPIP-NEO/120/item_key.csv"


def _mask() -> np.ndarray:
    m = np.zeros(120, dtype=bool)
    with open(_KEY) as f:
        for r in csv.DictReader(f):
            if str(r["reverse"]).strip().lower() == "true":
                m[int(r["item"]) - 1] = True
    return m


REVERSE_MASK_120 = _mask()


def _sel(item_ids):
    return np.array([REVERSE_MASK_120[i - 1] for i in item_ids], dtype=bool)


def flip_simplex(P, item_ids):
    """Map a model simplex over raw answers onto the corpus's recoded scale."""
    out = np.array(P, dtype=float, copy=True)
    s = _sel(item_ids)
    out[..., s, :] = out[..., s, ::-1]
    return out


def flip_answers(X, item_ids):
    out = np.array(X, dtype=float, copy=True)
    s = _sel(item_ids)
    out[..., s] = 6.0 - out[..., s]
    return out
