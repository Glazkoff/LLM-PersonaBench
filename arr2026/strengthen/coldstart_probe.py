"""The item cold-start bar.

The conditional decoder that wins at 0.1087 is fitted per TARGET ITEM on 20000
human responses to that exact item. On a genuinely unseen item -- a new
questionnaire, a new task -- no such labels exist. The review says this
directly: "For an unseen question, its empirical response distribution is
unavailable: do not silently estimate it from test labels."

So there are two different bars, and only one of them is the LLM's real
competition:

  WARM bar : target item has human training labels   -> decoder ~0.109
  COLD bar : target item has NONE                    -> ?

This measures the cold bar honestly. For each target item j we refit the
decoder using ONLY the other items of the same facet as surrogate supervision
(leave-one-item-out within facet), then apply it to item j. That is the best a
label-free statistical model can do without reading item text.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/glazkov/Development/personality-twins-arr/LLM-PersonaBench")
THRESH = [1, 2, 3, 4]


def crps(Q, Y):
    ind = np.stack([(Y <= t).astype(float) for t in THRESH], axis=2)
    return np.nanmean((Q - ind) ** 2, axis=2)


def ridge(X, Y, Xp, lam=1e-2):
    Xc = np.hstack([X, np.ones((len(X), 1))]); Xpc = np.hstack([Xp, np.ones((len(Xp), 1))])
    return Xpc @ np.linalg.solve(Xc.T @ Xc + lam * np.eye(Xc.shape[1]), Xc.T @ Y)


def iso(D):
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


df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
key = pd.read_csv(ROOT / "data/IPIP-NEO/120/item_key.csv")
scale = {int(r.item): r.facet_key for r in key.itertuples()}
ids = list(range(1, 121))
Y = df[[f"i{i}" for i in ids]].to_numpy(float)
idx = {i: k for k, i in enumerate(ids)}
inp, tgt = [], []
for s in sorted(set(scale.values())):
    mem = [i for i in ids if scale[i] == s]
    inp += mem[:2]; tgt += mem[2:]
inp, tgt = sorted(inp), sorted(tgt)

rng = np.random.default_rng(260909)
perm = rng.permutation(len(Y))
test, train = perm[:256], perm[256:256 + 20000]
Yi_tr = Y[np.ix_(train, [idx[i] for i in inp])]
Yi_te = Y[np.ix_(test, [idx[i] for i in inp])]
Yt_tr = Y[np.ix_(train, [idx[j] for j in tgt])]
Yt_te = Y[np.ix_(test, [idx[j] for j in tgt])]
n, nt = len(test), len(tgt)

def oh(Yin, K=5):
    m = np.zeros((len(Yin), Yin.shape[1] * K))
    b = np.clip(Yin.astype(int) - 1, 0, K - 1)
    for k in range(Yin.shape[1]):
        m[np.arange(len(Yin)), k * K + b[:, k]] = 1.0
    return m
Xtr, Xte = oh(Yi_tr), oh(Yi_te)

# ---- WARM: per-item supervision on the item itself -------------------------
warm = np.zeros((n, nt, 4))
for j in range(nt):
    t = np.column_stack([(Yt_tr[:, j] <= c).astype(float) for c in THRESH])
    warm[:, j, :] = np.clip(ridge(Xtr, t, Xte), 0, 1)
warm = iso(warm)

# ---- COLD: item j has NO human labels; borrow its facet siblings -----------
# Each target facet has 2 target items. Holding one out leaves exactly one
# sibling, which is the most favourable label-free surrogate available.
fac_of_tgt = [scale[j] for j in tgt]
cold = np.zeros((n, nt, 4))
prior_cold = np.zeros((n, nt, 4))
for j in range(nt):
    sib = [k for k in range(nt) if fac_of_tgt[k] == fac_of_tgt[j] and k != j]
    ysib = Yt_tr[:, sib].mean(axis=1)          # surrogate target, item j unseen
    t = np.column_stack([(ysib <= c).astype(float) for c in THRESH])
    cold[:, j, :] = np.clip(ridge(Xtr, t, Xte), 0, 1)
    prior_cold[:, j, :] = t.mean(axis=0)[None]
cold = iso(cold)
prior_cold = iso(prior_cold)

# ---- WARM prior (uses item j's own labels) --------------------------------
prior_warm = np.broadcast_to(
    np.stack([(Yt_tr <= c).mean(axis=0) for c in THRESH], axis=1)[None], (n, nt, 4))

rows = [("prior  WARM (item j labels)", prior_warm),
        ("decoder WARM (item j labels)", warm),
        ("prior  COLD (sibling only)", prior_cold),
        ("decoder COLD (sibling only)", cold)]
per = {k: np.nanmean(crps(M, Yt_te), axis=1) for k, M in rows}
print(f"{'arm':32s} {'CRPS':>8s}")
for k, _ in rows:
    print(f"{k:32s} {per[k].mean():8.4f}")

print("\nThe LLM's recorded score on this exact panel: 0.2510 (Qwen2.5-7B), 0.3578 (Granite).")
b = np.random.default_rng(11)
d = per["decoder COLD (sibling only)"] - per["decoder WARM (item j labels)"]
ci = np.percentile([np.mean(d[b.integers(0, n, n)]) for _ in range(4000)], [2.5, 97.5])
print(f"cold-minus-warm penalty: {d.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]")
