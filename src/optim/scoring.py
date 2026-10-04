"""Proper ordinal scoring and population calibration. RPS is lower-is-better, normalised by K-1,
identical to arr2026/scripts/hyp/_scoring.rps."""
import itertools
import numpy as np


SCORER_VERSION = "v2"


def cdf(P):
    _check_simplex(P)
    return np.cumsum(P, axis=-1)[..., :-1]


def cdf_to_simplex(F):
    F = np.asarray(F, float)
    _check_cdf(F)
    z = np.zeros(F.shape[:-1] + (1,))
    return np.diff(np.concatenate([z, F, z + 1.0], axis=-1), axis=-1)


def _check_simplex(P):
    P = np.asarray(P, float)
    if P.ndim < 1 or P.shape[-1] < 2 or not np.all(np.isfinite(P)):
        raise ValueError("forecast must be a finite categorical simplex")
    if np.any(P < -1e-8) or np.any(P > 1 + 1e-8) or not np.allclose(P.sum(-1), 1, atol=1e-6, rtol=0):
        raise ValueError("forecast is not a categorical simplex")


def _check_cdf(F):
    F = np.asarray(F, float)
    if F.ndim < 1 or F.shape[-1] < 1 or not np.all(np.isfinite(F)):
        raise ValueError("CDF must be finite")
    if np.any(F < -1e-8) or np.any(F > 1 + 1e-8) or np.any(np.diff(F, axis=-1) < -1e-8):
        raise ValueError("CDF must be bounded and nondecreasing")


def _check_targets(Y, shape, K):
    Y = np.asarray(Y, float)
    if Y.shape != shape:
        raise ValueError("targets must match the forecast panel shape")
    observed = ~np.isnan(Y)
    if np.any(~np.isfinite(Y[observed])) or np.any((Y[observed] < 1) | (Y[observed] > K)) or np.any(Y[observed] != np.floor(Y[observed])):
        raise ValueError("observed targets must be integer categories in 1..K")
    return Y


def rps(P, Y):
    P = np.asarray(P, float)
    _check_simplex(P)
    K = P.shape[-1]
    Y = _check_targets(Y, P.shape[:-1], K)
    Yf = np.nan_to_num(Y, nan=1.0)
    ind = np.stack([(Yf <= t).astype(float) for t in range(1, K)], axis=-1)
    out = np.sum((cdf(P) - ind) ** 2, axis=-1) / (K - 1.0)
    return np.where(np.isnan(Y), np.nan, out)


def s0(P, Y):
    """The audited metric as a loss (negated similarity), lower is better."""
    P = np.asarray(P, float)
    _check_simplex(P)
    K = P.shape[-1]
    Y = _check_targets(Y, P.shape[:-1], K)
    lv = np.arange(1, K + 1, dtype=float)
    d = np.abs(lv - np.nan_to_num(Y, nan=1.0)[..., None]) / (K - 1.0)
    return np.where(np.isnan(Y), np.nan, -(1.0 - np.sum(P * d, axis=-1)))


def pop_cdf(Y_train, K=5):
    Y_train = np.asarray(Y_train, float)
    if Y_train.ndim != 2 or Y_train.shape[0] == 0:
        raise ValueError("training targets must be a nonempty respondent-by-item matrix")
    _check_targets(Y_train, Y_train.shape, K)
    if np.any(np.all(np.isnan(Y_train), axis=0)):
        raise ValueError("each target item needs an observed training label")
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
    F = np.asarray(F, float)
    if F.ndim < 1 or F.shape[-1] < 1 or not np.all(np.isfinite(F)):
        raise ValueError("isotonic input must be finite and nonempty")
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
    return np.clip(best.reshape(F.shape), 0, 1)


def calibrate(Q, F0, alpha, Qbar=None):
    """Population calibration: prior F0 plus alpha times the person-specific deviation of the model CDF.
    Qbar (the model's mean CDF) comes from the panel alpha was fitted on when given."""
    Q = np.asarray(Q, float)
    F0 = np.asarray(F0, float)
    _check_cdf(Q)
    _check_cdf(F0)
    if Q.ndim != 3 or F0.shape != Q.shape[1:]:
        raise ValueError("Q and F0 must have matching item and threshold axes")
    Qbar = Q.mean(axis=0, keepdims=True) if Qbar is None else np.asarray(Qbar, float)
    if Qbar.ndim == Q.ndim - 1:
        Qbar = Qbar[None]
    if Qbar.shape != (1,) + Q.shape[1:]:
        raise ValueError("Qbar must match the item and threshold axes")
    _check_cdf(Qbar)
    return iso_project(F0[None] + alpha * (Q - Qbar))


def cal_cells(Q, F0, Y, alpha, Qbar=None):
    return rps(cdf_to_simplex(calibrate(Q, F0, alpha, Qbar)), Y)


def _cal_score(Q, F0, Y, alpha, Qbar=None):
    cells = cal_cells(Q, F0, Y, alpha, Qbar)
    if not np.any(np.isfinite(cells)):
        raise ValueError("calibration fold has no observed targets")
    return float(np.nanmean(per_respondent(cells)))


def select_alpha(Q, F0, Y, alphas):
    if not alphas:
        raise ValueError("at least one alpha is required")
    best = min((_cal_score(Q, F0, Y, a), float(a)) for a in alphas)
    return best[1], best[0]


def crossfit_rps_cal(Q, F0, Y, alphas, rng, *, folds=None, return_cells=False):
    """Fit alpha and Qbar on each half; score the opposite half out of fold.

    ``folds`` freezes the partition across candidate prompts. ``return_cells`` adds
    the held-out cell losses without changing the legacy two-value return.
    """
    n = Q.shape[0]
    if n < 2:
        raise ValueError("crossfit needs at least two respondents")
    if folds is None:
        perm = rng.permutation(n)
        A, B = perm[: n // 2], perm[n // 2:]
    else:
        A, B = (np.asarray(f, int) for f in folds)
        if A.ndim != 1 or B.ndim != 1 or not len(A) or not len(B) or not np.array_equal(np.sort(np.r_[A, B]), np.arange(n)):
            raise ValueError("folds must partition all respondents exactly once")
    aA, _ = select_alpha(Q[A], F0, Y[A], alphas)
    aB, _ = select_alpha(Q[B], F0, Y[B], alphas)
    cells = np.empty_like(Y, dtype=float)
    cells[B] = cal_cells(Q[B], F0, Y[B], aA, Q[A].mean(axis=0))
    cells[A] = cal_cells(Q[A], F0, Y[A], aB, Q[B].mean(axis=0))
    score = float(np.nanmean(per_respondent(cells)))
    result = (score, (aA, aB))
    return result + (cells,) if return_cells else result


def per_respondent(cells):
    cells = np.asarray(cells, float)
    observed = np.isfinite(cells)
    counts = observed.sum(axis=-1)
    out = np.full(counts.shape, np.nan, dtype=float)
    return np.divide(np.where(observed, cells, 0).sum(axis=-1), counts, out=out, where=counts > 0)
