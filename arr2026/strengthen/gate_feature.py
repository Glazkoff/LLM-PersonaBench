"""Feature-level combination: does the LLM carry information ORTHOGONAL to the decoder?

A scalar convex weight between two CDFs can only interpolate. If the LLM's signal is
orthogonal to the decoder's, a convex weight is blind to it. This fits a real combiner.

Constraint that dictates the design: belief_probs exist ONLY for the 256 scored
respondents, never for the 20000 training respondents. So the combiner is fitted on the
128 CALIBRATION respondents (128 x 60 = 7680 pooled rows) and evaluated frozen on the 128
EVAL respondents. The decoder itself is still fitted on the 20000 training respondents
and enters as a feature.

The control is the SAME combiner, same capacity, same fitting rows, WITHOUT the LLM
features. That is the apples-to-apples test of "does adding the LLM improve the strongest
statistical predictor".
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/Users/glazkov/Development/personality-twins-arr/LLM-PersonaBench")
TH = [1, 2, 3, 4]

def iso_fast(D, lo=0.0, hi=1.0):
    L = D.shape[-1]; X = D.reshape(-1, L)
    cs = np.concatenate([np.zeros((len(X), 1)), np.cumsum(X, axis=1)], axis=1)
    j = np.arange(L)[:, None]; k = np.arange(L)[None, :]
    cnt = (k - j + 1).astype(float)
    M = (cs[:, k + 1] - cs[:, j]) / np.where(cnt > 0, cnt, 1.0)
    M = np.where((k >= j)[None], M, np.inf)
    out = np.empty_like(X)
    for i in range(L):
        out[:, i] = M[:, : i + 1, i:].min(axis=2).max(axis=1)
    return np.clip(out, lo, hi).reshape(D.shape)

def crps_i(Q, Y):
    ind = np.stack([(Y <= t).astype(float) for t in TH], axis=2)
    return np.nanmean((Q - ind) ** 2, axis=2)

def ridge(X, Y, Xp, lam):
    Xc = np.hstack([X, np.ones((len(X),1))]); Xpc = np.hstack([Xp, np.ones((len(Xp),1))])
    return Xpc @ np.linalg.solve(Xc.T@Xc + lam*np.eye(Xc.shape[1]), Xc.T@Y)

def load(run):
    d = ROOT / run
    meta = json.loads((d/"summary.json").read_text())
    P = np.load(d/"belief_probs.npy")
    if meta.get("corpus_recoded"):
        rk = set(meta["reverse_keyed_targets"])
        fl = np.array([j in rk for j in meta["target_ids"]], bool)
        if fl.any(): P[:, fl, :] = P[:, fl, ::-1]
    return meta, P, np.load(d/"target_answers.npy"), np.load(d/"train_answers.npy")

def run_one(tag, run, corpus):
    meta, P, Yt, Ytr = load(run)
    Q = np.cumsum(P, axis=2)[:, :, :4]
    n, nt = Yt.shape

    # rebuild the observed input-half answers for this panel
    if corpus == "ipip":
        df = pd.read_csv(ROOT/"data/raw/df_ipipneo_120_clusters")
        key = pd.read_csv(ROOT/"data/IPIP-NEO/120/item_key.csv")
        scale = {int(r.item): r.facet_key for r in key.itertuples()}
        ids = list(range(1,121)); nin = 2
        Yall = df[[f"i{i}" for i in ids]].to_numpy(float); ntest = 256
    else:
        import csv as _csv
        rows = list(_csv.DictReader(open(ROOT/"data/PAlign/Dark-Triad.csv"), delimiter="\t"))
        keys = [f"{t}{i}" for t in "MNP" for i in range(1,10)]
        Yall = np.array([[float(r[k]) for k in keys] for r in rows])
        Yall = Yall[(Yall >= 1).all(axis=1)]
        ids = list(range(1,28)); scale = {i+1: keys[i][0] for i in range(len(keys))}; nin = 4
        ntest = 256
    idx = {i:k for k,i in enumerate(ids)}
    inp = sorted(sum([[i for i in ids if scale[i]==s][:nin]
                      for s in sorted(set(scale.values()))], []))
    rng = np.random.default_rng(260909); perm = rng.permutation(len(Yall))
    test, train = perm[:ntest], perm[ntest:ntest+20000]
    def oh(Y,K=5):
        m=np.zeros((len(Y),Y.shape[1]*K)); b=np.clip(Y.astype(int)-1,0,K-1)
        for k in range(Y.shape[1]): m[np.arange(len(Y)),k*K+b[:,k]]=1.0
        return m
    Xtr = oh(Yall[np.ix_(train,[idx[i] for i in inp])])
    Xte = oh(Yall[np.ix_(test ,[idx[i] for i in inp])])

    # decoder fitted on the 20000 training respondents, lambda tuned on CAL
    sp = np.random.default_rng(20260918).permutation(n); CAL, EV = sp[:n//2], sp[n//2:]
    targ = np.concatenate([(Ytr <= t).astype(float) for t in TH], axis=1)
    def dec(lam):
        return iso_fast(np.transpose(np.clip(ridge(Xtr,targ,Xte,lam),0,1)
                                     .reshape(n,4,nt),(0,2,1)))
    LAMS=[1e-2,1e-1,1,10,100,1000]
    lam = min(LAMS, key=lambda l: crps_i(dec(l)[CAL], Yt[CAL]).mean())
    D = dec(lam)
    PR = np.broadcast_to(np.stack([(Ytr<=t).mean(axis=0) for t in TH],axis=1)[None],(n,nt,4))

    ind = np.stack([(Yt <= t).astype(float) for t in TH], axis=2)      # (n,nt,4)
    def rows(sel, with_llm):
        f = [D[sel].reshape(-1,4), PR[sel].reshape(-1,4)]
        if with_llm: f.append(Q[sel].reshape(-1,4))
        return np.nan_to_num(np.hstack(f), nan=0.5)
    ytab = lambda sel: ind[sel].reshape(-1,4)

    out = {"decoder": crps_i(D[EV], Yt[EV]).mean(axis=1)}
    for with_llm in (False, True):
        Xc, Xe = rows(CAL, with_llm), rows(EV, with_llm)
        yc = ytab(CAL)
        # linear combiner, lambda chosen by 2-fold on CAL rows
        half = len(Xc)//2
        lc = min([1e-3,1e-2,1e-1,1,10], key=lambda l: (
            (np.clip(ridge(Xc[:half],yc[:half],Xc[half:],l),0,1)-yc[half:])**2).mean())
        lin = iso_fast(np.clip(ridge(Xc,yc,Xe,lc),0,1).reshape(len(EV),nt,4))
        # nonlinear combiner
        gb = np.column_stack([HistGradientBoostingRegressor(
                max_iter=200, learning_rate=0.06, max_depth=4, random_state=0
             ).fit(Xc, yc[:,t]).predict(Xe) for t in range(4)])
        gbm = iso_fast(np.clip(gb,0,1).reshape(len(EV),nt,4))
        k = "with LLM" if with_llm else "no LLM  "
        out[f"linear  {k}"] = crps_i(lin, Yt[EV]).mean(axis=1)
        out[f"GBM     {k}"] = crps_i(gbm, Yt[EV]).mean(axis=1)

    print(f"\n=== {tag}  (lambda*={lam}, n_eval={len(EV)}, items={nt}) ===")
    b = np.random.default_rng(5)
    for fam in ("linear", "GBM"):
        base = out[f"{fam}  no LLM  "] if fam=="linear" else out[f"{fam}     no LLM  "]
        key  = f"{fam}  with LLM" if fam=="linear" else f"{fam}     with LLM"
        d = base - out[key]
        ci = np.percentile([np.mean(d[b.integers(0,len(EV),len(EV))]) for _ in range(4000)],
                           [2.5,97.5])
        star = " *SIGNIFICANT*" if ci[0] > 0 else ""
        print(f"  {fam:7s} no-LLM {base.mean():.4f} -> with-LLM {out[key].mean():.4f}  "
              f"gain {d.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}]  "
              f"rel {100*d.mean()/base.mean():+.1f}%{star}")
    print(f"  reference: decoder alone {out['decoder'].mean():.4f}")

run_one("Qwen2.5-7B / IPIP",    "arr2026/results/h15/h15_qwen25_7b_ipip",  "ipip")
run_one("Granite-4.1-8B / IPIP","arr2026/results/h15/h15_granite_8b_ipip", "ipip")
run_one("Qwen2.5-7B / SD3",     "arr2026/results/h15/h15_qwen25_7b_sd3",   "sd3")
run_one("Granite-4.1-8B / SD3", "arr2026/results/h15/h15_granite_8b_sd3",  "sd3")
