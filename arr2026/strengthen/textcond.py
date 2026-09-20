"""Held-out CONSTRUCT transfer: can item TEXT carry a response model to unseen facets?

Setup. A person's observed answers are the 60 input items (2 per facet, all 30 facets).
The 60 target items are split by FACET: 24 facets' targets are training items, 6 facets'
targets are TEST items that no supervised arm has ever seen a human label for. Labels for
training items come from the 20000 training respondents. Evaluation is on the 128 EVAL
respondents' answers to the 6 held-out facets' target items.

This is the regime the review says matters: "A transferable version must predict
calibration from item text and training tasks". A tabular decoder cannot be fitted on an
item with no labels; a text-conditioned model can, and so can the LLM.

Arms
  global prior        marginal over training items -- the floor
  structural transfer person's own 2 observed answers in the SAME facet, with the
                      observed->target mapping learned on training facets only.
                      This is the strong label-free baseline and the one to beat.
  text-conditioned    structural features PLUS an embedding of the item text
  LLM readout         the model's own saved distribution, reoriented
  text + LLM          both

Rotates all 5 disjoint facet folds so every facet is held out exactly once.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/Users/glazkov/Development/personality-twins-arr/LLM-PersonaBench")
TH = [1, 2, 3, 4]

def iso_fast(D):
    L = D.shape[-1]; X = D.reshape(-1, L)
    cs = np.concatenate([np.zeros((len(X),1)), np.cumsum(X,axis=1)], axis=1)
    j = np.arange(L)[:,None]; k = np.arange(L)[None,:]
    cnt = (k-j+1).astype(float)
    M = (cs[:,k+1]-cs[:,j]) / np.where(cnt>0,cnt,1.0)
    M = np.where((k>=j)[None], M, np.inf)
    out = np.empty_like(X)
    for i in range(L): out[:,i] = M[:,:i+1,i:].min(axis=2).max(axis=1)
    return np.clip(out,0,1).reshape(D.shape)

def crps_i(Q,Y):
    ind = np.stack([(Y<=t).astype(float) for t in TH],axis=2)
    return np.nanmean((Q-ind)**2, axis=2)

RUN = ROOT/"arr2026/results/h15/h15_qwen25_7b_ipip"
meta = json.loads((RUN/"summary.json").read_text())
P = np.load(RUN/"belief_probs.npy")
rk = set(meta["reverse_keyed_targets"])
fl = np.array([j in rk for j in meta["target_ids"]], bool); P[:,fl,:] = P[:,fl,::-1]
Q_llm = np.cumsum(P,axis=2)[:,:,:4]
Yt = np.load(RUN/"target_answers.npy"); Ytr = np.load(RUN/"train_answers.npy")

df = pd.read_csv(ROOT/"data/raw/df_ipipneo_120_clusters")
key = pd.read_csv(ROOT/"data/IPIP-NEO/120/item_key.csv")
scale = {int(r.item): r.facet_key for r in key.itertuples()}
text  = {int(r.item): r.text for r in key.itertuples()}
rev   = {int(r.item) for r in key.itertuples() if str(r.reverse).lower()=="true"}
ids = list(range(1,121)); idx={i:k for k,i in enumerate(ids)}
Yall = df[[f"i{i}" for i in ids]].to_numpy(float)
facets = sorted(set(scale.values()))
inp = sorted(sum([[i for i in ids if scale[i]==s][:2] for s in facets],[]))
tgt = meta["target_ids"]
rng = np.random.default_rng(260909); perm = rng.permutation(len(Yall))
test, train = perm[:256], perm[256:256+20000]
Xin_te = Yall[np.ix_(test ,[idx[i] for i in inp])]
Xin_tr = Yall[np.ix_(train,[idx[i] for i in inp])]
inp_pos = {i:k for k,i in enumerate(inp)}
nt = len(tgt)

# item text embedding
try:
    from sentence_transformers import SentenceTransformer
    emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    E = emb_model.encode([text[j] for j in tgt], normalize_embeddings=True)
    src = "MiniLM"
except Exception as e:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    V = TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), min_df=1)
    E = TruncatedSVD(24, random_state=0).fit_transform(V.fit_transform([text[j] for j in tgt]))
    E = E/np.maximum(np.linalg.norm(E,axis=1,keepdims=True),1e-9)
    src = f"TF-IDF+SVD (no embedder: {type(e).__name__})"
from sklearn.decomposition import PCA
E = PCA(min(16, E.shape[1]-1, len(tgt)-1), random_state=0).fit_transform(E)
print(f"item embedding: {src}, shape {E.shape}")

sp = np.random.default_rng(20260918).permutation(256); CAL, EV = sp[:128], sp[128:]

def feats(Yin, jpos, with_text):
    """Per (respondent, item) features that are DEFINED for an unseen item."""
    j = tgt[jpos]; f = scale[j]
    sib = [inp_pos[i] for i in inp if scale[i]==f]          # observed same-facet answers
    a = (Yin[:, sib] - 3.0)/2.0                              # (n,2)
    prof = (Yin.mean(axis=1, keepdims=True) - 3.0)/2.0       # overall acquiescence
    sd   = Yin.std(axis=1, keepdims=True)/2.0
    r = np.full((len(Yin),1), 1.0 if j in rev else 0.0)
    cols = [a, prof, sd, r]
    if with_text:
        cols.append(np.repeat(E[jpos][None], len(Yin), axis=0))
    return np.hstack(cols)

folds = [facets[i::5] for i in range(5)]
res = {k: [] for k in ["global prior","structural","text-cond","LLM readout","text+LLM"]}
for fi, hold in enumerate(folds):
    tr_pos = [k for k,j in enumerate(tgt) if scale[j] not in hold]
    te_pos = [k for k,j in enumerate(tgt) if scale[j] in hold]
    # --- supervised rows from TRAINING items x TRAINING respondents ---------
    def build(rows_pos, Yin, Ylab, with_text, sub=None):
        X, y = [], []
        for jp in rows_pos:
            s = np.arange(len(Yin)) if sub is None else sub
            X.append(feats(Yin[s], jp, with_text))
            y.append(np.column_stack([(Ylab[s, jp] <= t).astype(float) for t in TH]))
        return np.vstack(X), np.vstack(y)
    sub = np.random.default_rng(7).choice(len(Xin_tr), 3000, replace=False)
    gp = np.array([[(Ytr[:, jp] <= t).mean() for t in TH] for jp in tr_pos]).mean(axis=0)
    for name, wt in (("structural", False), ("text-cond", True)):
        Xc, yc = build(tr_pos, Xin_tr, Ytr, wt, sub)
        pred = np.zeros((len(EV), len(te_pos), 4))
        models = [HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                  max_depth=5, random_state=0).fit(Xc, yc[:, t]) for t in range(4)]
        for c, jp in enumerate(te_pos):
            Xe = feats(Xin_te[EV], jp, wt)
            pred[:, c, :] = np.column_stack([m.predict(Xe) for m in models])
        res[name].append(crps_i(iso_fast(np.clip(pred,0,1)), Yt[np.ix_(EV, te_pos)]))
    res["global prior"].append(crps_i(
        np.broadcast_to(gp[None,None], (len(EV), len(te_pos), 4)), Yt[np.ix_(EV, te_pos)]))
    Ql = Q_llm[np.ix_(EV, te_pos)]
    res["LLM readout"].append(crps_i(Ql, Yt[np.ix_(EV, te_pos)]))
    # text + LLM: refit text-cond with the LLM CDF appended, fitted on CAL respondents
    Xc2 = np.hstack([feats(Xin_te[CAL], 0, True)[:0], ])  # placeholder
    blocks_c, y_c = [], []
    for jp in tr_pos:
        blocks_c.append(np.hstack([feats(Xin_te[CAL], jp, True),
                                   np.nan_to_num(Q_llm[np.ix_(CAL,[jp])][:,0,:], nan=0.5)]))
        y_c.append(np.column_stack([(Yt[np.ix_(CAL,[jp])][:,0] <= t).astype(float) for t in TH]))
    Xc2, yc2 = np.vstack(blocks_c), np.vstack(y_c)
    ms = [HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=5,
          random_state=0).fit(Xc2, yc2[:, t]) for t in range(4)]
    pr = np.zeros((len(EV), len(te_pos), 4))
    for c, jp in enumerate(te_pos):
        Xe = np.hstack([feats(Xin_te[EV], jp, True),
                        np.nan_to_num(Q_llm[np.ix_(EV,[jp])][:,0,:], nan=0.5)])
        pr[:, c, :] = np.column_stack([m.predict(Xe) for m in ms])
    res["text+LLM"].append(crps_i(iso_fast(np.clip(pr,0,1)), Yt[np.ix_(EV, te_pos)]))
    print(f"  fold {fi+1}/5 held-out facets: {len(hold)}, items {len(te_pos)}", flush=True)

print(f"\n{'arm':16s} {'CRPS':>8s}   (held-out facets, mean over 5 folds, n_eval=128)")
per = {k: np.concatenate(v, axis=1).mean(axis=1) for k, v in res.items()}
for k in ["global prior","LLM readout","structural","text-cond","text+LLM"]:
    print(f"{k:16s} {per[k].mean():8.4f}")
b = np.random.default_rng(13)
print("\npaired vs structural transfer (positive = better):")
base = per["structural"]
for k in ["text-cond","text+LLM","LLM readout"]:
    d = base - per[k]
    ci = np.percentile([np.mean(d[b.integers(0,128,128)]) for _ in range(4000)],[2.5,97.5])
    star = " *SIGNIFICANT*" if ci[0] > 0 else ""
    print(f"  {k:14s} {d.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] "
          f"rel {100*d.mean()/base.mean():+.1f}%{star}")
