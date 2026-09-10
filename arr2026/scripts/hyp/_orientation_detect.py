r"""Decide, from the data alone, whether a corpus stores recoded or raw answers.

This project's worst bug was an orientation error: 55 of 120 IPIP items are
stored reverse-recoded while the model answers the literal item, and the
correction was missing. Variance is invariant to per-item sign flips, so it
survived every variance check; correlations, Wasserstein distances and every
absolute-error score did not.

The fix is not a bigger keying table -- a hand-maintained table is exactly what
went wrong -- but a test the data itself answers. Within one scale, reverse-keyed
items correlate NEGATIVELY with forward-keyed ones when the corpus is raw, and
POSITIVELY once it has been recoded. So:

    detect_recoded()   sign of the mean within-scale item-item correlation
    reverse_keyed()    which items oppose their own scale's dominant direction

Both are validated in test_orientation_detect.py against IPIP-NEO-120 (known
recoded) and SD3 (known raw) before being trusted on a new corpus.
"""
from __future__ import annotations

import numpy as np


def _scale_blocks(scale):
    out = {}
    for j, s in enumerate(scale):
        out.setdefault(s, []).append(j)
    return out


def item_scale_corr(Y, scale):
    """Per-item correlation with the mean of the OTHER items in its scale.

    Excluding the item itself matters: a part-whole correlation is positive by
    construction and would report every corpus as recoded.
    """
    Y = np.asarray(Y, dtype=float)
    r = np.full(Y.shape[1], np.nan)
    for _, cols in _scale_blocks(scale).items():
        if len(cols) < 2:
            continue
        for j in cols:
            rest = [c for c in cols if c != j]
            a = Y[:, j]
            with np.errstate(invalid="ignore"):
                # A respondent who skipped every other item in the scale gives
                # an all-NaN slice; that is a legitimately missing value here,
                # not an error, and the mask below drops it.
                sub = Y[:, rest]
                cnt = np.sum(~np.isnan(sub), axis=1)
                b = np.where(cnt > 0, np.nansum(sub, axis=1) / np.maximum(cnt, 1), np.nan)
            ok = ~(np.isnan(a) | np.isnan(b))
            if ok.sum() > 30 and np.std(a[ok]) > 0 and np.std(b[ok]) > 0:
                r[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return r


def detect_recoded(Y, scale):
    """True if the corpus appears already reverse-recoded.

    Returns (recoded, fraction_negative, mean_r) so a caller can see how
    decisive the evidence is instead of only a boolean.
    """
    r = item_scale_corr(Y, scale)
    v = r[~np.isnan(r)]
    if v.size == 0:
        raise ValueError("no scale has two or more usable items")
    frac_neg = float((v < 0).mean())
    # A recoded corpus has essentially no negatively loading items; a raw one
    # has roughly however many the instrument reverse-keys, typically 30-50%.
    return bool(frac_neg < 0.10), frac_neg, float(v.mean())


def reverse_keyed(Y, scale):
    """Items opposing their own scale's dominant direction -- empty if recoded."""
    r = item_scale_corr(Y, scale)
    return set(np.where(np.nan_to_num(r, nan=1.0) < 0)[0].tolist())
