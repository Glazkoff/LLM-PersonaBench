"""Portable entry point: public cell checks or explicitly requested restricted replay.

Default restricted mode emits a plan only. Never submits jobs or downloads data.
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
PINS={'audit_result.py':'cc4497b569e9f2df5bb135c3deabdf4bbae826b28c01f5190b6e34c377f24f74',
      'audit_uncertainty.py':'0bcc43d0d0d62c73d83f77e33a3e9f2099c28b4b68637299c5e804f007a8ee76'}
GATE='ae38a2e7119709181c0b8b30e884fb180d6e9ae3da153fe0b265e4b4b610e49c'
CONTROL='b7ab995259050dc67a931a28d459e26f02d9302de86906fafadd38ce9c3cc34e'
CONTROL_SPEC='7807404cca436ff7ce512afc086a21ee2bf889e8da40b9ebdf97aaa0d930d775'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(ok,msg):
    if not ok: raise ValueError(msg)
def get(value,path):
    for key in path: value=value[key]
    return value
def cells(package):
    package=Path(package)
    mapping=json.loads((package/'replay/cells.json').read_text()); cache={}; checked=[]
    for c in mapping['cells']:
        source=c['source']
        if source not in cache: cache[source]=json.loads((package/source).read_text())
        value=get(cache[source],c['path'])
        rendered=format(value,c['format'])
        if c.get('strip_zero'): rendered=rendered.replace('0.','.',1)
        require(rendered==c['expected'],f"source cell changed: {c['id']}")
        text=(package/c['manuscript']).read_text()
        if c.get('section'):
            require(c['section'] in text,'section missing')
            text=text.split(c['section'],1)[1].split('\\subsection',1)[0]
        require(c['needle'] in text,f"manuscript cell missing: {c['id']}")
        checked.append(c['id'])
    return {'status':'PASS','checked_cells':len(checked),'ids':checked,
            'scope':'source-key/rounding/manuscript checks; no bootstrap recomputation',
            'unmapped':mapping['unmapped']}
def finite_cpu():
    job=os.environ.get('SLURM_JOB_ID')
    require(job and job.isdigit(),'restricted execution requires Slurm')
    raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
    f=dict(x.split('=',1) for x in raw.split() if '=' in x)
    require(f.get('TimeLimit') not in (None,'UNLIMITED','INFINITE'),'finite walltime required')
    require('gpu' not in f.get('AllocTRES','').lower(),'CPU-only allocation required')
    return f
def restricted(args):
    require(args.root and args.out,'--root and fresh --out required')
    root=Path(args.root).resolve(); out=Path(args.out).resolve()
    require(not out.exists(),'output exists; refusing overwrite')
    for name,pin in PINS.items():require(sha(HERE/'vendor'/name)==pin,'vendored audit drift')
    require(sha(PACKAGE/'analysis/controls.py')==CONTROL,'controls source drift')
    require(sha(HERE/'vendor/controls-SPEC.md')==CONTROL_SPEC,'controls frozen spec drift')
    gate=root/'ops-20261106/postprobe-r2-review-2'
    inputs=[root/'bundle-20261106-stimulus-r2',root/'twin-20261106-stimulus-r2',gate/'verify_readout.py',gate/'COMPARE_SCORE_VERIFIED.json']
    require(all(p.exists() for p in inputs),'restricted tree unavailable; no data download attempted')
    require(sha(inputs[2])==GATE,'gate source drift')
    if args.mode=='r2':
        cmd=[sys.executable,str(HERE/'vendor/audit_result.py'),'--bundle',str(inputs[0]),'--lane',str(inputs[1]),
             '--gate-script',str(inputs[2]),'--score-gate',str(inputs[3]),'--out',str(out/'aggregates'),
             '--gate-script-sha256',GATE,'--uncertainty-source-sha256',PINS['audit_uncertainty.py']]
    else:
        cmd=[sys.executable,str(out/'controls.py'),str(root)]
    plan={'mode':args.mode,'command':cmd,'output':str(out),'execute':args.execute,
          'working_directory':str(out),
          'environment_overrides':{'PYTHONPATH':str(HERE/'vendor'),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'},
          'staged_files':({'controls.py':str(PACKAGE/'analysis/controls.py'),'SPEC.md':str(HERE/'vendor/controls-SPEC.md')} if args.mode=='controls' else {}),
          'execution_readiness':'NOT_VALIDATED_BY_PLAN_ONLY',
          'relocation_boundary':'Package paths are resolved from launcher location. Signed input configs may retain absolute paths; no rewrite is performed. Restricted scoring also requires sibling twin-20261106-v6 and historical Slurm accounting.',
          'scope':'restricted outputs stay restricted; includes possible per-person arrays',
          'finite_envelope':'r2:4CPU/32GB/2h; controls:4CPU/24GB/20min'}
    if not args.execute:return plan
    plan['allocation']=finite_cpu()
    out.mkdir(parents=True,mode=0o700,exist_ok=False); os.chmod(out,0o700)
    env=os.environ.copy(); env.update(plan['environment_overrides'])
    if args.mode=='controls':
        shutil.copy2(PACKAGE/'analysis/controls.py',out/'controls.py')
        shutil.copy2(HERE/'vendor/controls-SPEC.md',out/'SPEC.md')
    (out/'LAUNCH.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (out/'replay.log').open('x') as log:
        result=subprocess.run(cmd,cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/'EXIT.json').write_text(json.dumps({'returncode':result.returncode})+'\n')
    require(result.returncode==0,'replay failed; preserve log, no automatic retry')
    return plan
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['cells','r2','controls']);p.add_argument('--root');p.add_argument('--out');p.add_argument('--execute',action='store_true')
    a=p.parse_args()
    print(json.dumps(cells(PACKAGE) if a.mode=='cells' else restricted(a),indent=2))
if __name__=='__main__':main()
