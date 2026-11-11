from pathlib import Path
import json,math
p=Path(__file__).parent
r=json.loads((p/'controls.json').read_text());checks=[]
def ck(ok,label):
 if not ok:raise AssertionError(label)
 checks.append(label)
for c in r['contrasts']:
 d=c['equal_family'];a=r['absolute_losses'];x=a[c['system']]['normalized']-a[c['reference']]['normalized']
 ck(abs(x-d['estimate'])<1e-12,'risk-difference '+c['system']+'/'+c['reference'])
 ck(abs(sum(c['family_estimates'].values())/2-x)<1e-12,'family mean '+c['system']+'/'+c['reference'])
 ck(d['ci95'][0]<=d['estimate']<=d['ci95'][1],'conditional interval '+c['system']+'/'+c['reference'])
for k,v in r['selected_weights'].items():
 ck(len(v)==49 and all(abs(x*10-round(x*10))<1e-12 and 0<=x<=1 for x in v.values()),'grid '+k)
for arm,eps in r['joint_cal_test_bootstrap']['weight_frequencies'].items():
 for k,v in eps.items():ck(len(v)==11 and abs(sum(v)-1)<1e-12,'resampled frequency '+arm+'/'+k)
ck(all(sum(v.values())==0 for v in r['proxy_fallback_counts'].values()),'zero proxy fallback')
ck(r['scope'].startswith('exploratory'),'scope')
print(json.dumps({'status':'PASS','checks':len(checks),'scope':'aggregate arithmetic and schema only; no independent bootstrap reconstruction'},indent=2))
