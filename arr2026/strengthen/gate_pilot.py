"""PILOT of the gating question, on saved probabilities only. No inference.

Splits the 256 scored h15 respondents into 128 CALIBRATION / 128 EVAL. Every
alpha, temperature and stacking weight is chosen on the calibration half and
applied frozen to the eval half. Statistical baselines are fitted on the 20000
training respondents, who are disjoint from both halves by construction.

This is a pilot, not the experiment: 128 eval respondents is small, one model,
one instrument, one split. It exists to tell us whether the campaign has a
chance before a single GPU-hour is spent.
"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/glazkov/Development/personality-twins-arr/LLM-PersonaBench")
RUN = ROOT / "arr2026/results/h15/h15_qwen25_7b_ipip"
THRESH = [1, 2, 3, 4]


def crps_pt(Q, Y):
    ind = np.stack([(Y <= t).astype(float) for t in THRESH], axis=2)
    return (Q - ind) ** 2


def score(Q, Y):
    return np.nanmean(crps_pt(Q, Y), axis=(1, 2))


def logloss(P, Y):
    p = np.clip(P, 1e-6, None); p = p / p.sum(axis=2, keepdims=True)
    k = np.clip(Y.astype(int) - 1, 0, 4)
    return -np.log(np.take_along_axis(p, k[:, :, None], axis=2))[:, :, 0].mean(axis=1)


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


meta = json.loads((RUN / "summary.json").read_text())
P = np.load(RUN / "belief_probs.npy")
# MANDATORY reorientation: corpus is recoded, model answers the literal item.
rk = set(meta["reverse_keyed_targets"])
flip = np.array([j in rk for j in meta["target_ids"]], dtype=bool)
P[:, flip, :] = P[:, flip, ::-1]
print(f"reoriented {flip.sum()}/{len(flip)} targets")

Yt = np.load(RUN / "target_answers.npy")
Ytr = np.load(RUN / "train_answers.npy")

# raw observed input-half answers for the same panel, rebuilt from the corpus
df = pd.read_csv(ROOT / "data/raw/df_ipipneo_120_clusters")
key = pd.read_csv(ROOT / "data/IPIP-NEO/120/item_key.csv")
scale = {int(r.item): r.facet_key for r in key.itertuples()}
ids = list(range(1, 121)); idx = {i: k for k, i in enumerate(ids)}
Yall = df[[f"i{i}" for i in ids]].to_numpy(float)
inp = sorted(sum([[i for i in ids if scale[i] == s][:2] for s in sorted(set(scale.values()))], []))
rng = np.random.default_rng(260909); perm = rng.permutation(len(Yall))
test, train = perm[:256], perm[256:256 + 20000]
assert np.allclose(Yt, Yall[np.ix_(test, [idx[j] for j in meta["target_ids"]])], equal_nan=True)
Xin_te = Yall[np.ix_(test, [idx[i] for i in inp])]
Xin_tr = Yall[np.ix_(train, [idx[i] for i in inp])]

def oh(Y, K=5):
    m = np.zeros((len(Y), Y.shape[1] * K)); b = np.clip(Y.astype(int) - 1, 0, K - 1)
    for k in range(Y.shape[1]):
        m[np.arange(len(Y)), k * K + b[:, k]] = 1.0
    return m

nt = Yt.shape[1]
targ = np.concatenate([(Ytr <= t).astype(float) for t in THRESH], axis=1)
DEC = iso(np.transpose(np.clip(ridge(oh(Xin_tr), targ, oh(Xin_te)), 0, 1)
                       .reshape(256, 4, nt), (0, 2, 1)))          # the 0.1087 bar
PRIOR = np.broadcast_to(np.stack([(Ytr <= t).mean(axis=0) for t in THRESH], axis=1)[None],
                        (256, nt, 4))
Q = np.cumsum(P, axis=2)[:, :, :4]                                 # raw LLM CDF

# --- respondent split: calibration vs eval, disjoint -------------------------
sp = np.random.default_rng(20260918).permutation(256)
CAL, EV = sp[:128], sp[128:]

F0 = PRIOR[0]                                                      # (nt,4) train-fitted
Qbar = np.nanmean(Q[CAL], axis=0)                                  # calibration pool only

def repair(alpha, rows):
    return iso(np.clip(F0[None] + alpha * (Q[rows] - Qbar[None]), 0, 1))

def stack(w, alpha, rows):
    return iso(np.clip((1 - w) * DEC[rows] + w * repair(alpha, rows), 0, 1))

ALPHAS = np.round(np.arange(0.0, 2.01, 0.1), 2)
WS = np.round(np.arange(0.0, 1.01, 0.05), 2)

a_star = min(ALPHAS, key=lambda a: score(repair(a, CAL), Yt[CAL]).mean())
best = min(((w, a) for w in WS for a in ALPHAS),
           key=lambda t: score(stack(t[0], t[1], CAL), Yt[CAL]).mean())
w_star, a_stack = best
print(f"selected on CAL: alpha*={a_star}   stack w*={w_star} (alpha={a_stack})")

arms = {
    "population prior":              PRIOR[EV],
    "decoder (raw answers)":         DEC[EV],
    "LLM raw (reoriented)":          Q[EV],
    f"LLM recentered a={a_star}":    repair(a_star, EV),
    f"stack w={w_star}":             stack(w_star, a_stack, EV),
}
print(f"\n{'arm':32s} {'CRPS':>8s}  {'delta vs decoder':>26s}")
base = score(DEC[EV], Yt[EV])
b = np.random.default_rng(3)
for k, M in arms.items():
    v = score(M, Yt[EV]); d = base - v
    ci = np.percentile([np.mean(d[b.integers(0, len(EV), len(EV))]) for _ in range(4000)],
                       [2.5, 97.5])
    star = " *" if ci[0] > 0 else ""
    print(f"{k:32s} {v.mean():8.4f}  {d.mean():+8.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]{star}")

print(f"\nw*=0 would mean the LLM adds nothing. Selected w* = {w_star}.")
