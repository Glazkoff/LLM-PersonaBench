"""Does the LLM add value where the statistical model CANNOT be fitted?

The warm pilot selected w*=0: with 20000 human answers to the target item, the
LLM is redundant. That is the regime the decoder owns. The interesting question
is the other one -- as human supervision for the target item is removed, does
the LLM's weight rise off zero?

Four supervision regimes for the target item j, increasingly cold:
  WARM      20000 training respondents answered item j
  N=100     only 100 did
  N=25      only 25 did
  COLD      none did; only item j's facet sibling supervises
Everything is selected on a disjoint 128-respondent calibration half and frozen.
Per-item stacking weights are also tested, since a single global weight cannot
express "the LLM helps on some items and not others".
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/glazkov/Development/personality-twins-arr/LLM-PersonaBench")
RUN = ROOT / "arr2026/results/h15/h15_qwen25_7b_ipip"
THRESH = [1, 2, 3, 4]

def crps_i(Q, Y):                      # per (respondent, item)
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
                v.append((a*wa + b*wb)/(wa+wb)); w.append(wa+wb)
        o = []
        for val, wt in zip(v, w): o.extend([val]*int(wt))
        out[r] = np.clip(np.array(o[:D.shape[-1]]), 0, 1)
    return out.reshape(D.shape)

meta = json.loads((RUN/"summary.json").read_text())
P = np.load(RUN/"belief_probs.npy")
rk = set(meta["reverse_keyed_targets"])
flip = np.array([j in rk for j in meta["target_ids"]], bool)
P[:, flip, :] = P[:, flip, ::-1]
Yt = np.load(RUN/"target_answers.npy"); Ytr = np.load(RUN/"train_answers.npy")
Q = np.cumsum(P, axis=2)[:, :, :4]

df = pd.read_csv(ROOT/"data/raw/df_ipipneo_120_clusters")
key = pd.read_csv(ROOT/"data/IPIP-NEO/120/item_key.csv")
scale = {int(r.item): r.facet_key for r in key.itertuples()}
ids = list(range(1,121)); idx = {i:k for k,i in enumerate(ids)}
Yall = df[[f"i{i}" for i in ids]].to_numpy(float)
inp = sorted(sum([[i for i in ids if scale[i]==s][:2] for s in sorted(set(scale.values()))],[]))
rng = np.random.default_rng(260909); perm = rng.permutation(len(Yall))
test, train = perm[:256], perm[256:256+20000]
Xtr_raw = Yall[np.ix_(train,[idx[i] for i in inp])]
Xte_raw = Yall[np.ix_(test ,[idx[i] for i in inp])]
def oh(Y,K=5):
    m=np.zeros((len(Y),Y.shape[1]*K)); b=np.clip(Y.astype(int)-1,0,K-1)
    for k in range(Y.shape[1]): m[np.arange(len(Y)),k*K+b[:,k]]=1.0
    return m
Xtr, Xte = oh(Xtr_raw), oh(Xte_raw)
nt = Yt.shape[1]
tgt_ids = meta["target_ids"]; fac = [scale[j] for j in tgt_ids]

def decoder(nlab):
    """nlab = number of training respondents with labels on item j; None = cold."""
    D = np.zeros((256, nt, 4)); PR = np.zeros((256, nt, 4))
    r = np.random.default_rng(4242)
    for j in range(nt):
        if nlab is None:
            sib = [k for k in range(nt) if fac[k]==fac[j] and k!=j]
            y = Ytr[:, sib].mean(axis=1); sub = np.arange(len(Ytr))
        else:
            sub = r.choice(len(Ytr), nlab, replace=False); y = Ytr[sub, j]
        t = np.column_stack([(y <= c).astype(float) for c in THRESH])
        D[:, j, :] = np.clip(ridge(Xtr[sub], t, Xte), 0, 1)
        PR[:, j, :] = t.mean(axis=0)[None]
    return iso(D), iso(PR)

sp = np.random.default_rng(20260918).permutation(256); CAL, EV = sp[:128], sp[128:]
ALPHAS = np.round(np.arange(0.0, 2.01, 0.1), 2)
WS = np.round(np.arange(0.0, 1.001, 0.05), 3)

print(f"{'regime':10s} {'decoder':>8s} {'+LLM glob':>10s} {'w*':>5s} "
      f"{'+LLM per-item':>14s} {'mean w_j':>9s} {'gain':>9s} {'95% CI':>20s}")
b = np.random.default_rng(9)
for name, nlab in [("WARM", None_ := len(Ytr)), ("N=100", 100), ("N=25", 25), ("COLD", None)]:
    D, PR = decoder(None if name == "COLD" else nlab)
    F0 = PR[0]
    Qbar = np.nanmean(Q[CAL], axis=0)
    a = min(ALPHAS, key=lambda a: crps_i(iso(np.clip(F0[None]+a*(Q[CAL]-Qbar[None]),0,1)),
                                         Yt[CAL]).mean())
    R = lambda rows: iso(np.clip(F0[None] + a*(Q[rows]-Qbar[None]), 0, 1))
    Rc, Re = R(CAL), R(EV)
    # global weight
    wg = min(WS, key=lambda w: crps_i(iso(np.clip((1-w)*D[CAL]+w*Rc,0,1)), Yt[CAL]).mean())
    Sg = crps_i(iso(np.clip((1-wg)*D[EV]+wg*Re,0,1)), Yt[EV]).mean(axis=1)
    # per-item weight, each selected on CAL
    wj = np.zeros(nt)
    for j in range(nt):
        wj[j] = min(WS, key=lambda w: crps_i(
            iso(np.clip((1-w)*D[CAL][:,[j]]+w*Rc[:,[j]],0,1)), Yt[CAL][:,[j]]).mean())
    Sp = crps_i(iso(np.clip((1-wj)[None,:,None]*D[EV]+wj[None,:,None]*Re,0,1)),
                Yt[EV]).mean(axis=1)
    base = crps_i(D[EV], Yt[EV]).mean(axis=1)
    d = base - Sp
    ci = np.percentile([np.mean(d[b.integers(0,len(EV),len(EV))]) for _ in range(4000)],[2.5,97.5])
    star = " *" if ci[0] > 0 else ""
    print(f"{name:10s} {base.mean():8.4f} {Sg.mean():10.4f} {wg:5.2f} "
          f"{Sp.mean():14.4f} {wj.mean():9.3f} {d.mean():+9.4f} "
          f"[{ci[0]:+.4f},{ci[1]:+.4f}]{star}")
