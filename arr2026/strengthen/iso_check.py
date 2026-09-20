"""Exact vectorised isotonic projection, validated against the incumbent loop.

Every Track A arm projects an (n_respondents, n_items, K-1) array of CDF values
onto the monotone non-decreasing simplex. The incumbent `isotonic_rows` in
h15_analyse.py is a pure-Python pool-adjacent-violators loop over n*items rows;
at 256x60 that is 15360 Python loops per arm, and a budget curve evaluates
hundreds of arms. It is the wall-clock bottleneck of the whole campaign.

For K-1 = 4 (or 6 for HEXACO) the L2 isotonic solution has a closed form:
    yhat_i = max_{j<=i} min_{k>=i} mean(y_j..y_k)
which is exact and fully vectorisable. This validates the fast version against
the incumbent on adversarial inputs before anything depends on it.
"""
import numpy as np

def iso_loop(D):                      # the incumbent, verbatim from h15_analyse.py
    out = D.copy(); flat = out.reshape(-1, D.shape[-1])
    for r in range(flat.shape[0]):
        vals, wts = [], []
        for x in flat[r]:
            vals.append(float(x)); wts.append(1.0)
            while len(vals) > 1 and vals[-2] > vals[-1]:
                b, wb = vals.pop(), wts.pop(); a_, wa = vals.pop(), wts.pop()
                vals.append((a_*wa + b*wb)/(wa+wb)); wts.append(wa+wb)
        o = []
        for v, w in zip(vals, wts): o.extend([v]*int(w))
        flat[r] = np.clip(np.array(o[:D.shape[-1]]), 0, 1)
    return flat.reshape(D.shape)

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

rng = np.random.default_rng(0)
cases = {
    "uniform random":      rng.random((2000, 4)),
    "strictly decreasing": np.tile(np.array([0.9, 0.7, 0.4, 0.1]), (500, 1)),
    "already monotone":    np.sort(rng.random((500, 4)), axis=1),
    "ties":                rng.integers(0, 3, (500, 4)).astype(float) / 2,
    "out of range":        rng.normal(0.5, 0.8, (1000, 4)),
    "hexaco K-1=6":        rng.random((1000, 6)),
}
ok = True
for name, X in cases.items():
    a, b = iso_loop(X.copy()), iso_fast(X.copy())
    same = np.allclose(a, b, atol=1e-9)
    # independent check: is the fast output actually monotone and in range?
    mono = bool((np.diff(b, axis=1) >= -1e-12).all())
    rng_ok = bool(((b >= -1e-12) & (b <= 1 + 1e-12)).all())
    print(f"{name:22s} loop==fast {str(same):5s}  monotone {str(mono):5s}  in[0,1] {str(rng_ok):5s}"
          f"  maxdiff {np.abs(a-b).max():.2e}")
    ok &= same and mono and rng_ok

# is the LOOP version even correct? compare both to a brute-force L2 projection
from itertools import combinations_with_replacement
def brute(y):
    L = len(y); best, bv = None, np.inf
    for blocks in range(1, L + 1):
        for cut in combinations_with_replacement(range(1, L), blocks - 1):
            idx = [0] + sorted(set(cut)) + [L]
            seg = [y[idx[t]:idx[t+1]] for t in range(len(idx)-1) if idx[t] < idx[t+1]]
            m = np.concatenate([[s.mean()]*len(s) for s in seg])
            if (np.diff(m) >= -1e-12).all():
                v = ((m - y)**2).sum()
                if v < bv: bv, best = v, m
    return best
bad_loop = bad_fast = 0
for _ in range(400):
    y = rng.random(4)
    t = brute(y)
    if not np.allclose(iso_loop(y[None].copy())[0], t, atol=1e-9): bad_loop += 1
    if not np.allclose(iso_fast(y[None].copy())[0], t, atol=1e-9): bad_fast += 1
print(f"\nvs brute-force L2 projection over 400 random rows: "
      f"loop wrong {bad_loop}, fast wrong {bad_fast}")

import time
X = rng.random((256, 60, 4))
t0 = time.time(); iso_loop(X.copy()); t1 = time.time(); iso_fast(X.copy()); t2 = time.time()
print(f"speed on (256,60,4): loop {t1-t0:.3f}s  fast {t2-t1:.4f}s  "
      f"speedup {(t1-t0)/max(t2-t1,1e-9):.0f}x")
print("ALL CHECKS PASS" if ok and bad_fast == 0 else "MISMATCH -- investigate")
