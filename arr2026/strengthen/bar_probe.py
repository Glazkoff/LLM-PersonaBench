"""How strong is a conditional decoder that sees the RAW input-half answers?

The incumbent decoder in h15_analyse.py consumes only the binned per-scale
percentile scores (onehot of 5 bins x n_scales). The input half holds 60 raw
IPIP answers; binning them into 30 facet scores x 5 bins is lossy. If a decoder
on the raw answers is much better than 0.1136, that -- not 0.1136 -- is the bar
the LLM has to clear, and the whole campaign is harder than the review assumes.

Reproduces the h15 split exactly: seed 260909, n_test 256, n_train 20000.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/glazkov/Development/personality-twins-arr/LLM-PersonaBench")
sys.path.insert(0, str(ROOT / "arr2026/scripts/hyp"))
THRESH = [1, 2, 3, 4]


def crps(Q, Y):
    ind = np.stack([(Y <= t).astype(float) for t in THRESH], axis=2)
    return np.nanmean((Q - ind) ** 2, axis=2)


def ridge(X, Y, Xp, lam=1e-2):
    Xc = np.hstack([X, np.ones((len(X), 1))])
    Xpc = np.hstack([Xp, np.ones((len(Xp), 1))])
    return Xpc @ np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Y)


def iso(D):
    """Row-wise monotone projection onto a non-decreasing sequence in [0,1]."""
    out = D.copy().reshape(-1, D.shape[-1])
    for r in range(out.shape[0]):
        v, w = [], []
        for x in out[r]:
            v.append(float(x)); w.append(1.0)
            while len(v) > 1 and v[-2] > v[-1]:
                b, wb = v.pop(), w.pop(); a, wa = v.pop(), w.pop()
                v.append((a * wa + b * wb) / (wa + wb)); w.append(wa + wb)
        o = []
        for val, wt in zip(v, w):
            o.extend([val] * int(wt))
        out[r] = np.clip(np.array(o[:D.shape[-1]]), 0, 1)
    return out.reshape(D.shape)


def to_cdf(flat, n, n_items):
    return iso(np.transpose(flat.reshape(n, len(THRESH), n_items), (0, 2, 1)))


def split_items(ids, scale, n_input):
    inp, tgt = [], []
    for s in sorted(set(scale.values())):
        mem = [i for i in ids if scale[i] == s]
        inp += mem[:n_input]; tgt += mem[n_input:]
    return sorted(inp), sorted(tgt)


df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
key = pd.read_csv(ROOT / "data/IPIP-NEO/120/item_key.csv")
scale = {int(r.item): r.facet_key for r in key.itertuples()}
ids = list(range(1, 121))
Y = df[[f"i{i}" for i in ids]].to_numpy(float)
inp, tgt = split_items(ids, scale, 2)
idx = {i: k for k, i in enumerate(ids)}

rng = np.random.default_rng(260909)
perm = rng.permutation(len(Y))
test, train = perm[:256], perm[256:256 + 20000]
Yi_tr, Yi_te = Y[np.ix_(train, [idx[i] for i in inp])], Y[np.ix_(test, [idx[i] for i in inp])]
Yt_tr, Yt_te = Y[np.ix_(train, [idx[j] for j in tgt])], Y[np.ix_(test, [idx[j] for j in tgt])]
n, nt = len(test), len(tgt)
print(f"train {Yi_tr.shape} -> {Yt_tr.shape} | test {Yi_te.shape} -> {Yt_te.shape}")

targ = np.concatenate([(Yt_tr <= t).astype(float) for t in THRESH], axis=1)

# 0. empirical prior
prior = np.broadcast_to(
    np.stack([(Yt_tr <= t).mean(axis=0) for t in THRESH], axis=1)[None], (n, nt, 4))

# 1. incumbent: 30 facet scores, binned to 5 levels, one-hot  (150 features)
def facet_scores(Yin):
    cols = []
    for s in sorted(set(scale.values())):
        mem = [k for k, i in enumerate(inp) if scale[i] == s]
        cols.append((Yin[:, mem].mean(axis=1) - 1.0) / 4.0 * 100.0)
    return np.column_stack(cols)

def onehot(S, L=5):
    b = np.clip((S / 20.0).astype(int), 0, L - 1)
    m = np.zeros((len(S), S.shape[1] * L))
    for k in range(S.shape[1]):
        m[np.arange(len(S)), k * L + b[:, k]] = 1.0
    return m

Str, Ste = facet_scores(Yi_tr), facet_scores(Yi_te)
inc = to_cdf(np.clip(ridge(onehot(Str), targ, onehot(Ste)), 0, 1), n, nt)

# 2. continuous facet scores, no binning  (30 features)
cont = to_cdf(np.clip(ridge(Str / 100.0, targ, Ste / 100.0), 0, 1), n, nt)

# 3. RAW 60 input answers, one-hot over 5 levels  (300 features)
def oh_items(Yin, K=5):
    m = np.zeros((len(Yin), Yin.shape[1] * K))
    b = np.clip(Yin.astype(int) - 1, 0, K - 1)
    for k in range(Yin.shape[1]):
        m[np.arange(len(Yin)), k * K + b[:, k]] = 1.0
    return m
raw = to_cdf(np.clip(ridge(oh_items(Yi_tr), targ, oh_items(Yi_te)), 0, 1), n, nt)

# 4. RAW answers, linear (60 features)
rawlin = to_cdf(np.clip(ridge((Yi_tr - 3.0) / 2.0, targ, (Yi_te - 3.0) / 2.0), 0, 1), n, nt)

names = ["prior", "incumbent(binned facets)", "continuous facets",
         "RAW 60 answers one-hot", "RAW 60 answers linear"]
mats = [prior, inc, cont, raw, rawlin]
per = [np.nanmean(crps(M, Yt_te), axis=1) for M in mats]
print()
for nm, v in zip(names, per):
    print(f"{nm:28s} {v.mean():.4f}")

base = per[1]
print("\npaired vs incumbent (positive = better than incumbent), 95% CI:")
bs = np.random.default_rng(7)
for nm, v in zip(names, per):
    d = base - v
    b = np.percentile([np.mean(d[bs.integers(0, n, n)]) for _ in range(4000)], [2.5, 97.5])
    print(f"  {nm:28s} {d.mean():+.4f} [{b[0]:+.4f},{b[1]:+.4f}]")
