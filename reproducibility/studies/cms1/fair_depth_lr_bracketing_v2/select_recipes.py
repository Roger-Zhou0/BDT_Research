"""Selection for a declared development amendment; reporting never used."""
from pathlib import Path
import fcntl,json,math
import protocol as p
from parent_evidence import verify_parent

def read_new_grid(project):
    rows=[];missing=[]
    expected_sources={q.relative_to(p.HERE).as_posix():p.file_sha(q) for q in sorted(p.HERE.rglob('*.py')) if '__pycache__' not in q.parts}
    for seed in p.DEV_SEEDS:
      for method in p.METHODS:
       for lr in p.NEW_LRS:
        d=p.trial_dir(project,'development',seed,method,lr)
        paths=[q for q in (d/'result.json',d/'diverged.json') if q.exists()]
        if not paths:missing.append(str(d));continue
        if len(paths)!=1:raise RuntimeError('Both result and divergence exist: '+str(d))
        path=paths[0];obj=json.loads(path.read_text());ct=obj['contract']
        if (ct['protocol_sha'],ct['stage'],ct['seed'],ct['method'],ct['lr'])!=(p.PROTOCOL_SHA,'development',seed,method,lr):
            raise RuntimeError('New-trial registration mismatch: '+str(path))
        if obj['contract_sha256']!=p.digest(ct) or json.loads((d/'contract.json').read_text())!=ct:
            raise RuntimeError('New-trial contract digest mismatch: '+str(path))
        if ct['source_sha256']!=expected_sources or ct['data_hashes']!=p.PARENT_EVIDENCE['data_hashes']:
            raise RuntimeError('New-trial source/cache mismatch: '+str(path))
        cfg=dict(ct['config']);cfg.pop('learning_rate')
        if cfg!=p.PARENT_EVIDENCE['config_template'] or ct['runtime']!=p.PARENT_EVIDENCE['runtime_by_seed'][str(seed)]:
            raise RuntimeError('New trial changed non-LR config/runtime: '+str(path))
        ini=json.loads((d/'initialization.json').read_text())
        if ini!=p.PARENT_EVIDENCE['initializations'][str(seed)]:
            raise RuntimeError('New vs inherited initialization mismatch: '+str(path))
        value=None
        if path.name=='result.json':
            if obj['state']!='complete' or obj['step']!=p.STEPS or obj.get('report') is not None:
                raise RuntimeError('Invalid or reporting-contaminated DEV result: '+str(path))
            if obj['initialization']!=ini:raise RuntimeError('Result initialization mismatch')
            dv=obj['dev'];value=dv['loss']
            if not math.isfinite(value) or (dv['partition_start'],dv['partition_stop'],dv['tokens'])!=(p.DEV_OFFSET,p.DEV_OFFSET+p.EVAL_SEQUENCES*513,p.EVAL_SEQUENCES*512):
                raise RuntimeError('Invalid development evaluation: '+str(path))
        elif obj['state']!='diverged':raise RuntimeError('Invalid numerical-divergence marker')
        rows.append(dict(seed=seed,method=method,lr=lr,dev_loss=value,result=str(path),sha256=p.file_sha(path),stage_origin='amendment_v2'))
    return dict(ready=not missing,rows=rows,missing=missing)

def summarize(rows):
    chosen={};scores={}
    for method in p.METHODS:
        scores[method]=[]
        for lr in p.SEARCH_LRS:
            rr=sorted([r for r in rows if r['method']==method and r['lr']==lr],key=lambda r:r['seed'])
            if [r['seed'] for r in rr]!=list(p.DEV_SEEDS):raise RuntimeError('Incomplete or duplicated search cell')
            vals=[r['dev_loss'] for r in rr]
            score=None if any(v is None for v in vals) else sum(vals)/len(vals)
            scores[method].append(dict(lr=lr,mean_dev=score,per_seed=vals))
        valid=[r for r in scores[method] if r['mean_dev'] is not None]
        if not valid:raise RuntimeError('No valid two-seed recipe for '+method)
        chosen[method]=min(valid,key=lambda r:(r['mean_dev'],r['lr']))['lr']
    return dict(chosen_lr=chosen,scores=scores)

def verify_locked(obj):
    if obj['protocol_sha']!=p.PROTOCOL_SHA:raise RuntimeError('Locked amendment protocol changed')
    for r in obj['development_results']:
        if p.file_sha(r['result'])!=r['sha256']:raise RuntimeError('Locked development evidence changed')

def choose(project):
    root=p.campaign(project);root.mkdir(parents=True,exist_ok=True)
    with (root/'.selection_writer.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        # Check that old campaign has not changed, even when re-entering the new lock.
        prior=verify_parent(project)
        lock=root/'selection.lock.json'
        if lock.exists():
            obj=json.loads(lock.read_text());verify_locked(obj)
            return dict(state='locked',selection=obj)
        if (root/'report').exists() and any(q.is_file() for q in (root/'report').rglob('*')):
            raise RuntimeError('Reporting files exist without a selection lock.')
        new=read_new_grid(project)
        if not new['ready']:return dict(state='waiting_bracketing',missing=len(new['missing']))
        rows=prior+new['rows'];chosen=summarize(rows)
        if any(lr in (min(p.SEARCH_LRS),max(p.SEARCH_LRS)) for lr in chosen['chosen_lr'].values()):
            obj=dict(state='unbracketed',protocol_sha=p.PROTOCOL_SHA,**chosen,
                     reason='Outer boundary persists; no more automatic expansion and no reporting.')
            p.atomic_json(root/'selection.unbracketed.json',obj);return obj
        obj=dict(protocol_sha=p.PROTOCOL_SHA,**chosen,lrs_searched=list(p.SEARCH_LRS),
                 inherited_lrs=list(p.INHERITED_LRS),new_lrs=list(p.NEW_LRS),
                 development_seeds=list(p.DEV_SEEDS),development_results=rows,
                 source_note='Declared AFTER v1 development, BEFORE amendment trials or reporting. Reporting never used for selection.',
                 budget_per_method=dict(configurations=10,seeds=2,full_trials=20,inherited_full_trials=14,new_full_trials=6,scheduled_token_exposures=20*p.STEPS*p.TOKENS_PER_STEP),
                 backend_hashes=json.loads((p.HERE/'FROZEN_KERNELS.json').read_text()))
        p.atomic_json(lock,obj);return dict(state='locked',selection=obj)

if __name__=='__main__':
 import argparse
 a=argparse.ArgumentParser();a.add_argument('--project',default='.');x=a.parse_args()
 print(json.dumps(choose(x.project),indent=2))
