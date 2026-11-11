import json,os,sys,hashlib
from pathlib import Path
import numpy as np
import audit_result as a
root=Path(sys.argv[1]);lane=root/'twin-20261106-stimulus-r2';gate=root/'ops-20261106/postprobe-r2-review-2'
a.require(a.sha(Path(a.__file__))=='cc4497b569e9f2df5bb135c3deabdf4bbae826b28c01f5190b6e34c377f24f74','auditor drift')
_,alloc,_,_,manifest,ep,labels,_,cfg,fr,official,methods=a.checked_result(root/'bundle-20261106-stimulus-r2',lane,gate/'COMPARE_SCORE_VERIFIED.json',gate/'verify_readout.py','ae38a2e7119709181c0b8b30e884fb180d6e9ae3da153fe0b265e4b4b610e49c')
ids=manifest['split'];keys=list(ep);n=658;nc=200;E=49;grid=np.array(fr['grid']);a.require(len(keys)==E and len(ids['cal'])==nc and len(ids['test'])==n,'shape')
cal={'decoder1200':a.forecasts(cfg['decoder1200']['cal'],ids['cal'],ep,'cal')}
for arm in ('neutral','t0'):cal[arm]=a.forecasts(cfg['models']['granite'][arm]['cal'],ids['cal'],ep,'cal')
stim={}
for line in (lane/'private/prepared/STIMULI.jsonl').open():
 r=json.loads(line);stim[(r['respondent_id'],r['endpoint_id'])]=json.dumps([r['question_text'],r['options']],sort_keys=True)
def loss(p,y,kind):
 p=np.asarray(p); y=np.asarray(y)
 if kind=='ordinal':return ((p.cumsum(axis=-1)[...,:-1]-(y[...,None]<=np.arange(p.shape[-1]-1)))**2).mean(axis=-1)
 with np.errstate(divide='ignore'):return -np.log(np.take_along_axis(p,y[...,None],axis=-1)[...,0])
den=np.array([fr['prior_normalization'][k]['cal_prior_mean_primary_loss'] for k in keys]);fam={f:np.array([ep[k]['family']==f for k in keys]) for f in a.FAMILIES};w=sum(m/m.sum()/2 for m in fam.values());kindm={str(kind):np.array([ep[k]['kind']==kind for k in keys]) for kind in set(e['kind'] for e in ep.values())}
testloss={name:np.column_stack([loss(methods[name][k],[labels[p][k] for p in ids['test']],ep[k]['kind']) for k in keys]) for name in methods}
weights={};selected={};calgrid={};testgrid={};fallbacks={};library=[]
names=['prior','uniform','stimulus_prior','neutral','t0','neutral_proxy','t0_proxy']
for name in names:calgrid[name]=np.empty((nc,E,len(grid)));testgrid[name]=np.empty((n,E,len(grid)));weights[name]=[];selected[name]=np.empty((n,E))
priorloss=np.empty((n,E));calchoice=[]
for j,k in enumerate(keys):
 spec=ep[k];K=len(spec['options']);yc=np.array([labels[p][k] for p in ids['cal']]);yt=np.array([labels[p][k] for p in ids['test']]);yf=np.array([labels[p][k] for p in ids['fit']]);cp=np.array(cal['decoder1200'][k]);tp=np.array(methods['decoder1200'][k]);prior=np.array(fr['prior_normalization'][k]['fit1000_prior_probabilities']);priorloss[:,j]=loss(np.broadcast_to(prior,(n,K)),yt,spec['kind'])
 groups={}
 for i,p in enumerate(ids['fit']):groups.setdefault(stim[(p,k)],[]).append(yf[i])
 def sp(person):
  vals=groups.get(stim[(person,k)])
  return (np.bincount(vals,minlength=K)+1)/(len(vals)+K) if vals else prior
 ca={'prior':np.broadcast_to(prior,(nc,K)),'uniform':np.full((nc,K),1/K),'stimulus_prior':np.array([sp(p) for p in ids['cal']])}
 ta={'prior':np.broadcast_to(prior,(n,K)),'uniform':np.full((n,K),1/K),'stimulus_prior':np.array([sp(p) for p in ids['test']])}
 for arm in ('neutral','t0'):
  ca[arm]=np.array(cal[arm][k]);ta[arm]=np.array(methods['granite__'+arm][k]);cgroups={}
  for i,p in enumerate(ids['cal']):cgroups.setdefault(stim[(p,k)],[]).append(i)
  fallback={'cal':0,'test':0};pc=[];pt=[]
  for i,p in enumerate(ids['cal']):
   ii=[h for h in cgroups.get(stim[(p,k)],[]) if h!=i]
   if not ii:ii=[h for h in range(nc) if h!=i];fallback['cal']+=1
   pc.append(ca[arm][ii].mean(axis=0))
  for p in ids['test']:
   ii=cgroups.get(stim[(p,k)],[])
   if not ii:ii=list(range(nc));fallback['test']+=1
   pt.append(ca[arm][ii].mean(axis=0))
  ca[arm+'_proxy']=np.array(pc);ta[arm+'_proxy']=np.array(pt);fallbacks[arm+'__'+k]=fallback
 for name in names:
  for gi,g in enumerate(grid):
   calgrid[name][:,j,gi]=loss((1-g)*cp+g*ca[name],yc,spec['kind'])
   testgrid[name][:,j,gi]=loss((1-g)*tp+g*ta[name],yt,spec['kind'])
  means=calgrid[name][:,j,:].mean(axis=0);idx=int(np.argmin(means));weights[name].append(float(grid[idx]));selected[name][:,j]=testgrid[name][:,j,idx]
  if name in ('neutral','t0'):
   a.require(grid[idx]==fr['selected']['granite__'+name][k]['weight_persona'],'original CAL selection drift')
   a.require(np.max(np.abs(selected[name][:,j]-testloss['granite__'+name+'__cal_selected'][:,j]))<1e-12,'original TEST drift')
 choices=[(float(calgrid[z][:,j,g].mean()),float(grid[g]),i,g) for i,z in enumerate(names[:3]) for g in range(len(grid))]
 _,_,ni,gi=min(choices);library.append(testgrid[names[ni]][:,j,gi]);calchoice.append({'endpoint':k,'control':names[ni],'weight':float(grid[gi])})
selected['stat_library']=np.column_stack(library);testloss.update({'control__'+name:arr for name,arr in selected.items() if name not in ('neutral','t0')});testloss['fit_prior']=priorloss
for v in testloss.values():a.require(np.isfinite(v).all(),'nonfinite test loss')
rng=np.random.default_rng(20261110);draw=rng.integers(0,n,size=(10000,n))
def summary(v):return {'estimate':float(v.mean()),'ci95':np.quantile(v[draw].mean(axis=1),[.025,.975]).tolist()}
contrasts=[]
for arm in ('neutral','t0'):
 s='granite__'+arm+'__cal_selected'
 for r in ['decoder1200','full1400',*['control__'+q for q in ['prior','uniform','stimulus_prior','stat_library',arm+'_proxy']]]:
  diff=(testloss[s]-testloss[r])/den
  contrasts.append({'system':s,'reference':r,'equal_family':summary(diff@w),'family_estimates':{f:float(diff[:,m].mean()) for f,m in fam.items()}})
absolute={}
for name,v in testloss.items():
 absolute[name]={'normalized':float((v/den@w).mean()),'economic_logloss':float(v[:,fam[a.FAMILIES[0]]].mean()),'ordinal_RPS':float(v[:,[ep[k]['kind']=='ordinal' for k in keys]].mean()),'other_logloss':float(v[:,[ep[k]['family']==a.FAMILIES[1] and ep[k]['kind']!='ordinal' for k in keys]].mean())}
# Joint participant resampling: keep endpoint correlation in both independent CAL and TEST draws.
B=2000;rng=np.random.default_rng(20261111);cc=rng.multinomial(nc,np.full(nc,1/nc),size=B)/nc;tt=rng.multinomial(n,np.full(n,1/n),size=B)/n
joint={};jointvals={};hist={}
for arm in ('neutral','t0'):
 cube=calgrid[arm];scores=cc@cube.reshape(nc,-1);choice=scores.reshape(B,E,len(grid)).argmin(axis=2)
 tv=tt@testgrid[arm].reshape(n,-1);tv=tv.reshape(B,E,len(grid));v=np.take_along_axis(tv,choice[:,:,None],axis=2)[:,:,0];paired=(v/den*w).sum(axis=1);jointvals[arm]=paired
 joint[arm]={}
 for ref in ('decoder1200','full1400'):
  d=paired-tt@(testloss[ref]/den@w);joint[arm][ref]={'bootstrap_ci95':np.quantile(d,[.025,.975]).tolist(),'bootstrap_mean':float(d.mean())}
 hist[arm]=[(np.bincount(choice[:,j],minlength=len(grid))/B).tolist() for j in range(E)]
d=jointvals['t0']-jointvals['neutral'];joint['t0_minus_neutral']={'bootstrap_ci95':np.quantile(d,[.025,.975]).tolist(),'bootstrap_mean':float(d.mean())}
np.savez_compressed('private-analysis.npz',**testloss)
res={'status':'PASS','job_id':os.environ['SLURM_JOB_ID'],'allocation':alloc,'scope':'exploratory post-review, exposed R2, no independent confirmation','people':n,'cal_people':nc,'endpoints':E,'spec_sha256':a.sha(Path('SPEC.md')),'source_sha256':a.sha(Path(__file__)),'freeze_sha256':a.sha(lane/'private/compare/FROZEN.json'),'absolute_losses':absolute,'contrasts':contrasts,'selected_weights':{name:dict(zip(keys,v)) for name,v in weights.items()},'stat_library_selection':calchoice,'proxy_fallback_counts':fallbacks,'joint_cal_test_bootstrap':{'draws':B,'seed':20261111,'results':joint,'weight_frequencies':dict(zip(('neutral','t0'),[dict(zip(keys,hist[z])) for z in ('neutral','t0')]))},'portable_instruction':a.read(lane/'private/instruction/portable.json'),'baseline_selection':a.read(lane/'private/baseline/SELECTION.json') if (lane/'private/baseline/SELECTION.json').exists() else None}
# Selection metadata may carry file paths, but never raw participant rows. Keep it remote; export scientific summaries only.
res.pop('baseline_selection')
with open('RESULT.json','x') as f:json.dump(res,f,indent=2,allow_nan=False)
