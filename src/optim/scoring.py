"""Proper ordinal scoring and population calibration. RPS is lower-is-better, normalised by K-1,
identical to arr2026/scripts/hyp/_scoring.rps."""
import itertools
import numpy as np


def cdf(P):
    return np.cumsum(P, axis=-1)[..., :-1]


def cdf_to_simplex(F):
    F = np.asarray(F, float)
    z = np.zeros(F.shape[:-1] + (1,))
    return np.clip(np.diff(np.concatenate([z, F, z + 1.0], axis=-1), axis=-1), 0, None)


def rps(P, Y):
    K = P.shape[-1]
    Yf = np.nan_to_num(Y, nan=1.0)
    ind = np.stack([(Yf <= t).astype(float) for t in range(1, K)], axis=-1)
    out = np.sum((cdf(P) - ind) ** 2, axis=-1) / (K - 1.0)
    return np.where(np.isnan(Y), np.nan, out)


def s0(P, Y):
    """The audited metric as a loss (negated similarity), lower is better."""
    K = P.shape[-1]
    lv = np.arange(1, K + 1, dtype=float)
    d = np.abs(lv - np.nan_to_num(Y, nan=1.0)[..., None]) / (K - 1.0)
    return np.where(np.isnan(Y), np.nan, -(1.0 - np.sum(P * d, axis=-1)))


def pop_cdf(Y_train, K=5):
    out = []
    for t in range(1, K):
        v = np.where(np.isnan(Y_train), np.nan, (Y_train <= t).astype(float))
        out.append(np.nanmean(v, axis=0))
    return np.stack(out, axis=-1)


def _partitions(m):
    """All ways to cut m ordered points into contiguous blocks, as block-id vectors."""
    res = []
    for cuts in itertools.product([0, 1], repeat=m - 1):
        ids = [0]
        for c in cuts:
            ids.append(ids[-1] + c)
        res.append(np.array(ids))
    return res


_PARTS = {}


def iso_project(F):
    """Exact L2 projection onto non-decreasing sequences in [0,1] along the last axis.
    Enumerates the 2^(m-1) contiguous partitions (m = K-1 = 4 -> 8), vectorised over rows."""
    F = np.clip(np.asarray(F, float), 0, 1)
    m = F.shape[-1]
    if m not in _PARTS:
        _PARTS[m] = _partitions(m)
    flat = F.reshape(-1, m)
    best = flat.copy()
    best_sse = np.full(flat.shape[0], np.inf)
    ok_any = np.zeros(flat.shape[0], bool)
    for ids in _PARTS[m]:
        nb = ids.max() + 1
        fit = np.empty_like(flat)
        means = []
        for b in range(nb):
            cols = ids == b
            mu = flat[:, cols].mean(axis=1)
            means.append(mu)
            fit[:, cols] = mu[:, None]
        M = np.stack(means, axis=1)
        mono = np.all(np.diff(M, axis=1) >= -1e-12, axis=1) if nb > 1 else np.ones(flat.shape[0], bool)
        sse = ((fit - flat) ** 2).sum(axis=1)
        take = mono & (sse < best_sse)
        best[take] = fit[take]
        best_sse[take] = sse[take]
        ok_any |= mono
    return best.reshape(F.shape)


def calibrate(Q, F0, alpha, Qbar=None):
    """Population calibration: prior F0 plus alpha times the person-specific deviation of the model CDF.
    Qbar (the model's mean CDF) comes from the panel alpha was fitted on when given."""
    Qbar = Q.mean(axis=0, keepdims=True) if Qbar is None else Qbar[None] if Qbar.ndim == Q.ndim - 1 else Qbar
    return iso_project(F0[None] + alpha * (Q - Qbar))


def cal_cells(Q, F0, Y, alpha, Qbar=None):
    return rps(cdf_to_simplex(calibrate(Q, F0, alpha, Qbar)), Y)


def _cal_score(Q, F0, Y, alpha):
    return float(np.nanmean(cal_cells(Q, F0, Y, alpha)))


def select_alpha(Q, F0, Y, alphas):
    best = min((_cal_score(Q, F0, Y, a), float(a)) for a in alphas)
    return best[1], best[0]


def crossfit_rps_cal(Q, F0, Y, alphas, rng):
    """alpha selected on one half, scored on the other, both ways; returns (score, (aA, aB))."""
    n = Q.shape[0]
    perm = rng.permutation(n)
    A, B = perm[: n // 2], perm[n // 2:]
    aA, _ = select_alpha(Q[A], F0, Y[A], alphas)
    aB, _ = select_alpha(Q[B], F0, Y[B], alphas)
    sB = _cal_score(Q[B], F0, Y[B], aA)
    sA = _cal_score(Q[A], F0, Y[A], aB)
    return (sA * len(A) + sB * len(B)) / n, (aA, aB)


def per_respondent(cells):
    return np.nanmean(cells, axis=-1)
