"""Same budget curve, but the decoder is given every advantage.

The first pass fixed ridge lambda=1e-2 at every label budget. With 300 features
and 100 labelled respondents that decoder is overfitting, so beating it proves
nothing. Here lambda is tuned per budget on the SAME calibration respondents
that select the LLM weight, over a wide grid, and the decoder is additionally
allowed a shrink-to-prior coefficient -- the strongest cheap low-data
statistical model. If the LLM's weight still rises off zero, the effect is real.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/glazkov/Development/personality-twins-arr/LLM-PersonaBench")
RUN = ROOT / "arr2026/results/h15/h15_qwen25_7b_ipip"
TH = [1, 2, 3, 4]

def crps_i(Q, Y):
    ind = np.stack([(Y <= t).astype(float) for t in TH], axis=2)
    return np.nanmean((Q - ind) ** 2, axis=2)

def ridge(X, Y, Xp, lam):
    Xc = np.hstack([X, np.ones((len(X),1))]); Xpc = np.hstack([Xp, np.ones((len(Xp),1))])
    return Xpc @ np.linalg.solve(Xc.T@Xc + lam*np.eye(Xc.shape[1]), Xc.T@Y)

def iso(D):
    out = D.copy().reshape(-1, D.shape[-1])
    for r in range(out.shape[0]):
        v,w = [],[]
        for x in out[r]:
            v.append(float(x)); w.append(1.0)
            while len(v)>1 and v[-2]>v[-1]:
                b,wb=v.pop(),w.pop(); a,wa=v.pop(),w.pop()
                v.append((a*wa+b*wb)/(wa+wb)); w.append(wa+wb)
        o=[]
        for val,wt in zip(v,w): o.extend([val]*int(wt))
        out[r]=np.clip(np.array(o[:D.shape[-1]]),0,1)
    return out.reshape(D.shape)

meta=json.loads((RUN/"summary.json").read_text())
P=np.load(RUN/"belief_probs.npy")
rk=set(meta["reverse_keyed_targets"]); flip=np.array([j in rk for j in meta["target_ids"]],bool)
P[:,flip,:]=P[:,flip,::-1]
Yt=np.load(RUN/"target_answers.npy"); Ytr=np.load(RUN/"train_answers.npy")
Q=np.cumsum(P,axis=2)[:,:,:4]

df=pd.read_csv(ROOT/"data/raw/df_ipipneo_120_clusters")
key=pd.read_csv(ROOT/"data/IPIP-NEO/120/item_key.csv")
scale={int(r.item):r.facet_key for r in key.itertuples()}
ids=list(range(1,121)); idx={i:k for k,i in enumerate(ids)}
Yall=df[[f"i{i}" for i in ids]].to_numpy(float)
inp=sorted(sum([[i for i in ids if scale[i]==s][:2] for s in sorted(set(scale.values()))],[]))
rng=np.random.default_rng(260909); perm=rng.permutation(len(Yall))
test,train=perm[:256],perm[256:256+20000]
def oh(Y,K=5):
    m=np.zeros((len(Y),Y.shape[1]*K)); b=np.clip(Y.astype(int)-1,0,K-1)
    for k in range(Y.shape[1]): m[np.arange(len(Y)),k*K+b[:,k]]=1.0
    return m
Xtr=oh(Yall[np.ix_(train,[idx[i] for i in inp])])
Xte=oh(Yall[np.ix_(test ,[idx[i] for i in inp])])
nt=Yt.shape[1]
sp=np.random.default_rng(20260918).permutation(256); CAL,EV=sp[:128],sp[128:]
LAMS=[1e-2,1e-1,1,3,10,30,100,300,1000,3000]
SHRINK=[0.0,0.1,0.25,0.5,0.75,0.9,1.0]     # 1.0 = pure prior
ALPHAS=np.round(np.arange(0.0,2.01,0.1),2)
WS=np.round(np.arange(0.0,1.001,0.05),3)

print(f"{'budget':8s} {'lam*':>7s} {'shr*':>5s} {'decoder':>8s} {'w*':>5s} "
      f"{'+LLM':>8s} {'gain':>9s} {'rel':>7s} {'95% CI':>20s}")
b=np.random.default_rng(9)
for name,nlab in [("N=25",25),("N=100",100),("N=400",400),("N=2000",2000),("N=20000",len(Ytr))]:
    r=np.random.default_rng(4242)
    sub=np.arange(len(Ytr)) if nlab>=len(Ytr) else r.choice(len(Ytr),nlab,replace=False)
    PRm=np.stack([(Ytr[sub]<=c).mean(axis=0) for c in TH],axis=1)      # (nt,4)
    # tune (lambda, shrink) on CAL respondents
    cache={}
    def build(lam,s,rows):
        k=(lam,s)
        if k not in cache:
            D=np.zeros((256,nt,4))
            for j in range(nt):
                t=np.column_stack([(Ytr[sub,j]<=c).astype(float) for c in TH])
                D[:,j,:]=ridge(Xtr[sub],t,Xte,lam)
            cache[k]=iso(np.clip((1-s)*D+s*PRm[None],0,1))
        return cache[k][rows]
    lam,s=min(((l,s) for l in LAMS for s in SHRINK),
              key=lambda p: crps_i(build(p[0],p[1],CAL),Yt[CAL]).mean())
    Dc,De=build(lam,s,CAL),build(lam,s,EV)
    F0=PRm; Qbar=np.nanmean(Q[CAL],axis=0)
    a=min(ALPHAS,key=lambda a: crps_i(iso(np.clip(F0[None]+a*(Q[CAL]-Qbar[None]),0,1)),Yt[CAL]).mean())
    R=lambda rows: iso(np.clip(F0[None]+a*(Q[rows]-Qbar[None]),0,1))
    Rc,Re=R(CAL),R(EV)
    w=min(WS,key=lambda w: crps_i(iso(np.clip((1-w)*Dc+w*Rc,0,1)),Yt[CAL]).mean())
    base=crps_i(De,Yt[EV]).mean(axis=1)
    comb=crps_i(iso(np.clip((1-w)*De+w*Re,0,1)),Yt[EV]).mean(axis=1)
    d=base-comb
    ci=np.percentile([np.mean(d[b.integers(0,len(EV),len(EV))]) for _ in range(4000)],[2.5,97.5])
    star=" *" if ci[0]>0 else ""
    print(f"{name:8s} {lam:7g} {s:5.2f} {base.mean():8.4f} {w:5.2f} {comb.mean():8.4f} "
          f"{d.mean():+9.4f} {100*d.mean()/base.mean():6.1f}% [{ci[0]:+.4f},{ci[1]:+.4f}]{star}")
