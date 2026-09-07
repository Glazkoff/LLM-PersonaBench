"""
Select alpha under a criterion that can SEE the within/between trade.

The pilot showed CRPS improves monotonically in alpha while VR_within moves away
from human -- a marginal proper score is blind to the decomposition. So alpha must
not be selected on CRPS. Here we select on held-out C2ST-AUC, which reads the
joint respondent-level structure and is therefore sensitive to over-sharpened,
degenerate per-item beliefs.

Protocol, to avoid the circularity the novelty check flagged:
  - alpha chosen on DEV facets by C2ST against held-out humans
  - reported on HELD-OUT facets, never seen during selection
  - reported against the human ceiling and constant floor on the same split
  - VR_within / VR_between reported but NOT optimised

No GPU. Runs on the saved belief vectors.
"""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ITEMS=[f"i{i}" for i in range(1,121)]; VALS=np.arange(1,6,dtype=float); SEED=2026
df=pd.read_csv("data/raw/df_ipipneo_120_clusters")
KEY=pd.read_csv("data/IPIP-NEO/120/item_key.csv").sort_values("item")
FACET=KEY["facet_key"].to_numpy(); facets=sorted(set(FACET))
DEV=np.array([j for j in range(120) if FACET[j] in facets[:24]])
HLD=np.array([j for j in range(120) if FACET[j] in facets[24:]])
MODELS=[("qwen25_7b","hse"),("qwen25_32b","hse"),("qwen25_72b","hse"),
        ("mistral_24b","hse"),("qwq_32b","hse"),("qwen3_235b","euler")]
ALPHAS=[1.0,2.0,4.0,6.0,8.0,12.0,16.0,24.0]

def c2st(a,b,seed=SEED):
    if len(a)<8 or len(b)<8: return float("nan")
    X=np.vstack([a,b]); y=np.r_[np.zeros(len(a)),np.ones(len(b))]; au=[]
    for tr,te in StratifiedKFold(4,shuffle=True,random_state=seed).split(X,y):
        if len(np.unique(y[te]))<2: continue
        m=HistGradientBoostingClassifier(max_iter=100,random_state=seed).fit(X[tr],y[tr])
        au.append(roc_auc_score(y[te],m.predict_proba(X[te])[:,1]))
    return float(max(np.mean(au),1-np.mean(au))) if au else float("nan")

def sample(P,cols,rng):
    out=np.empty((P.shape[0],len(cols)))
    for i in range(P.shape[0]):
        for k,j in enumerate(cols): out[i,k]=rng.choice(VALS,p=P[i,j])
    return out

rows=[]
for tag,src in MODELS:
    per=[]
    for cl in range(4):
        f=pathlib.Path(f"arr2026/results_{src}/h4_{tag}/readout_cluster_{cl}/belief_probs.npy")
        if not f.exists(): continue
        P=np.load(f); cli=df[df.clusters==cl]
        used=set(pd.read_csv(f"arr2026/results_{src}/h4_{tag}/readout_cluster_{cl}/test_case_ids.csv")["case"])
        pool=cli[~cli.case.isin(used)][ITEMS].to_numpy(float)
        per.append((P,pool,np.nanstd(cli[ITEMS].to_numpy(float),axis=0)))
    if not per: continue
    rng=np.random.default_rng(SEED)

    def eval_alpha(a,cols):
        cs=[]
        for P,pool,hs in per:
            L=np.log(np.clip(P,1e-12,None)); Lb=L.mean(axis=0,keepdims=True)
            Q=np.exp(Lb+a*(L-Lb)); Q/=Q.sum(axis=2,keepdims=True)
            sim=sample(Q,cols,rng); ref=pool[rng.choice(len(pool),len(sim),replace=False)][:,cols]
            cs.append(c2st(sim,ref))
        return float(np.nanmean(cs))

    dev={a:eval_alpha(a,DEV) for a in ALPHAS}
    a_star=min(dev,key=dev.get)                      # C2ST closest to chance = lowest
    # held-out report
    c_model=eval_alpha(a_star,HLD); c_base=eval_alpha(1.0,HLD)
    ch=[];cc=[]
    for P,pool,hs in per:
        idx=rng.choice(len(pool),2*P.shape[0],replace=False)
        ch.append(c2st(pool[idx[:P.shape[0]]][:,HLD],pool[idx[P.shape[0]:]][:,HLD]))
        const=np.tile(np.round(pool.mean(axis=0)),(P.shape[0],1))
        cc.append(c2st(const[:,HLD],pool[idx[:P.shape[0]]][:,HLD]))
    # VR components at a_star (reported, not optimised)
    W=B=0.0
    for P,pool,hs in per:
        L=np.log(np.clip(P,1e-12,None)); Lb=L.mean(axis=0,keepdims=True)
        Q=np.exp(Lb+a_star*(L-Lb)); Q/=Q.sum(axis=2,keepdims=True)
        ok=hs[HLD]>0
        mu=np.nansum(Q[:,HLD,:]*VALS,axis=2); ex2=np.nansum(Q[:,HLD,:]*VALS**2,axis=2)
        var=np.clip(ex2-mu**2,0,None)
        W+=np.nanmean(np.sqrt(np.nanmean(var,axis=0))[ok]/hs[HLD][ok])
        B+=np.nanmean(np.sqrt(np.nanvar(mu,axis=0))[ok]/hs[HLD][ok])
    n=len(per)
    rows.append({"model":tag,"alpha_star":a_star,"c2st_base":c_base,"c2st_cal":c_model,
                 "c2st_human_ceiling":float(np.nanmean(ch)),"c2st_constant_floor":float(np.nanmean(cc)),
                 "VR_within_at_star":W/n,"VR_between_at_star":B/n})
    print(f"{tag:<13} a*={a_star:<5} C2ST {c_base:.3f} -> {c_model:.3f} "
          f"(human {np.nanmean(ch):.3f}, const {np.nanmean(cc):.3f}) | "
          f"within {W/n:.3f} between {B/n:.3f}", flush=True)

t=pd.DataFrame(rows); out=pathlib.Path("idea-stage"); t.to_csv(out/"exp_alpha_c2st_results.csv",index=False)
summary={"n_models":len(t),"mean_alpha_star":float(t.alpha_star.mean()),
         "mean_c2st_base":round(float(t.c2st_base.mean()),4),
         "mean_c2st_calibrated":round(float(t.c2st_cal.mean()),4),
         "mean_c2st_human_ceiling":round(float(t.c2st_human_ceiling.mean()),4),
         "mean_VR_between_at_star":round(float(t.VR_between_at_star.mean()),4),
         "note":"alpha selected on DEV facets by C2ST; all numbers on HELD-OUT facets."}
summary["verdict"]=("C2ST IMPROVES: amplification helps on a criterion we did not design around"
                    if t.c2st_cal.mean() < t.c2st_base.mean()-0.01 else
                    "C2ST FLAT/WORSE: amplification does not survive a joint-structure criterion")
(out/"exp_alpha_c2st_summary.json").write_text(json.dumps(summary,indent=2))
print("\n=== SUMMARY ==="); print(json.dumps(summary,indent=2))
