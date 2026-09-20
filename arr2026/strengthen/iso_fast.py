"""Exact vectorised isotonic (monotone) projection of ordinal CDF rows.

Validated identical to `h15_analyse.py:isotonic_rows` and to a brute-force L2
projection on adversarial inputs (uniform, strictly decreasing, already
monotone, ties, out-of-range, K-1=6). ~36x faster on a (256,60,4) array, which
is what makes hyperparameter-tuned budget curves affordable. See iso_check.py.
"""
import numpy as np


def iso_fast(D, lo=0.0, hi=1.0):
    """max_{j<=i} min_{k>=i} mean(y_j..y_k), then clip. Exact L2 isotonic."""
    L = D.shape[-1]
    X = D.reshape(-1, L)
    cs = np.concatenate([np.zeros((len(X), 1)), np.cumsum(X, axis=1)], axis=1)
    # M[:, j, k] = mean of y_j..y_k for j<=k, +inf elsewhere so min ignores it
    j = np.arange(L)[:, None]; k = np.arange(L)[None, :]
    cnt = (k - j + 1).astype(float)
    M = (cs[:, k + 1] - cs[:, j]) / np.where(cnt > 0, cnt, 1.0)
    M = np.where((k >= j)[None], M, np.inf)
    out = np.empty_like(X)
    for i in range(L):
        # min over k>=i, then max over j<=i
        out[:, i] = M[:, : i + 1, i:].min(axis=2).max(axis=1)
    return np.clip(out, lo, hi).reshape(D.shape)
