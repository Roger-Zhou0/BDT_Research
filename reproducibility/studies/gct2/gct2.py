"""Shared-parent confirmation: gates, an immutable cohort barrier, and deferred REPORT."""
from __future__ import annotations
import csv, gc, json, math, sys
from pathlib import Path
import core as io

def stages(project,seed):
    d=io.root(project)/f'seed_{seed}'
    out=[dict(seed=seed,stage='parent',mode='adamw',end=7630,parent=None,directory=str(d/'parent'),gate=False),
         dict(seed=seed,stage='grown',mode='adamw',end=0,parent=str(d/'parent'),directory=str(d/'grown'),gate=False),
         dict(seed=seed,stage='prefix',mode='adamw',end=200,parent=str(d/'grown'),directory=str(d/'prefix'),gate=False)]
    for m in io.GCT2['arm_order'][str(seed)]:
        out.append(dict(seed=seed,stage='arm',mode=m,end=3000,parent=str(d/'prefix'),directory=str(d/'arms'/m),gate=False))
    return out

def execute(project,j,data,resume=False):
    tr,rt,_=io.engine(project,'gct2');rt.configure(4)
    opts=dict(data=data,gate=j['gate'],tiny=j.get('tiny',False),resume=resume)
    if j['stage']=='grown': return tr.grow(project,j['directory'],j['seed'],j['parent'],**opts)
    return tr.train(project,j['directory'],j['seed'],j['stage'],j['mode'],j['end'],j.get('parent'),interrupt=j.get('interrupt'),**opts)

def gate_plan(directory,tiny=False):
    d=Path(directory)
    def j(name,stage,mode,end,parent=None,interrupt=None):
        return dict(seed=127,stage=stage,mode=mode,end=end,parent=str(d/parent) if parent else None,
                    directory=str(d/name),gate=True,tiny=tiny,interrupt=interrupt)
    jobs=[j('parent','parent','adamw',200),j('parent_reference','parent','reference_adamw',200),
          j('grown','grown','adamw',0,'parent'),j('prefix','prefix','adamw',200,'grown'),
          j('prefix_repeat','prefix','adamw',200,'grown')]
    for m in io.GCT2['arms']:
        jobs.append(j(m+'_0','arm',m,420,'prefix'))
        if m=='scrambled_matched':
            jobs.append(j(m+'_1','arm',m,420,'prefix',310))
        jobs.append(j(m+'_1','arm',m,420,'prefix'))
    jobs.extend([j('reference_adam','arm','reference_adamw',420,'prefix'),j('reference_bdt','arm','reference_bdt',420,'prefix')])
    return jobs

def compare_gate(directory):
    d=Path(directory);compare=io.load_module('gct_analysis',io.HERE/'frozen/gct1/gct_analysis.py').compare
    pairs=[('parent','parent_reference'),('prefix','prefix_repeat'),('adamw_0','reference_adam'),('bdt_0','reference_bdt')]
    pairs += [(m+'_0',m+'_1') for m in io.GCT2['arms']]
    checks={a+'__'+b:compare(d/a,d/b) for a,b in pairs}
    first=[]
    for m in io.GCT2['arms']:
        r=io.read(d/(m+'_0')/'result.json');z=next(x for x in r['proof'] if x['step']==201)
        first.append({k:z[k] for k in ('gradient','raw','batch','rng_before','rng_after')})
    if first[1:]!=first[:-1]: raise RuntimeError('First active inputs differ')
    checks['first_active_inputs_equal']=True
    return checks

@__import__('contextlib').contextmanager
def restored_for_evaluation(project,directory,tiny=False):
    import torch
    tr,rt,mods=io.engine(project,'gct2');rt.configure(4);native=mods['native'];device=torch.device('cpu' if tiny else 'cuda:0')
    before=rt.cpu(rt.rng());ck,meta=io.load_ck(directory,torch)
    try:
        if ck['contract']['runtime']!=rt.runtime(device): raise RuntimeError('Evaluation runtime differs from training')
        cfg=native.ExperimentConfig(**ck['contract']['config']);m=tr.build(native,cfg,ck['contract']['seed']).to(device)
        from gct_policy import Policy
        opt=Policy(m,cfg,mods,'adamw',device,ck['contract']['seed'],True,ck['contract']['offset'])
        m.load_state_dict(ck['model'],strict=True);opt.load(ck['opt']);rt.restore_rng(ck['rng'])
        if tr.identity(m,opt)!=meta['identity']: raise RuntimeError('Full final model/Adam/history identity mismatch')
        del opt,ck;gc.collect()
        yield m,cfg,native,device,rt,meta
    finally:
        rt.restore_rng(before)
        if 'm' in locals(): del m
        gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()

def evaluate_slice(model,path,cfg,native,device,rt,offset=2500000,tiny=False):
    import torch,numpy as np
    import torch.nn.functional as F
    state=rt.cpu(rt.rng());old=model.training;model.eval();nseq=4 if tiny else 2048;ctx=cfg.context_length
    data=np.memmap(path,dtype=np.uint16,mode='r');end=offset+nseq*(ctx+1)
    if end>len(data): raise RuntimeError('REPORT slice outside cache')
    total=0.;tokens=0
    try:
        with torch.no_grad():
            for i in range(0,nseq,cfg.micro_batch):
                n=min(cfg.micro_batch,nseq-i);start=offset+i*(ctx+1)
                a=np.array(data[start:start+n*(ctx+1)],dtype=np.int64).reshape(n,ctx+1)
                x=torch.from_numpy(a[:,:-1].copy()).to(device);y=torch.from_numpy(a[:,1:].copy()).to(device)
                with native.autocast_context(device,cfg.precision):
                    z,_=model(x);loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1))
                total+=float(loss)*y.numel();tokens+=y.numel()
        value=total/tokens
        if not math.isfinite(value):raise FloatingPointError('Nonfinite REPORT loss')
        return dict(loss=value,tokens=tokens,offset=offset,stop=end)
    finally:model.train(old);rt.restore_rng(state)

def report_gate(project,directory,data,tiny=False):
    tr,rt,_=io.engine(project,'gct2');before=rt.treehash(rt.rng())
    with restored_for_evaluation(project,Path(directory)/'bdt_0',tiny) as (m,cfg,native,device,rt,meta):
        inner=rt.treehash(rt.rng());path=Path(data['path'])/'val.bin'
        a=evaluate_slice(m,path,cfg,native,device,rt,tiny=tiny)
        b=evaluate_slice(m,path,cfg,native,device,rt,tiny=tiny)
        if a!=b or rt.treehash(rt.rng())!=inner: raise RuntimeError('REPORT gate repeatability/RNG failure')
        # Independent offset-zero reproduction against the unchanged DEV evaluator.
        dev=evaluate_slice(m,path,cfg,native,device,rt,offset=0,tiny=tiny)
        if dev!=rt.evaluate(m,path,cfg,native,device,tiny):raise RuntimeError('REPORT evaluator disagrees with legacy DEV at offset zero')
        # Dispatch evidence, deliberately not a timer.
        if not tiny:
            import torch,numpy as np
            ctx=cfg.context_length;a0=np.memmap(path,dtype=np.uint16,mode='r')[:cfg.micro_batch*(ctx+1)]
            xx=torch.tensor(np.array(a0,dtype=np.int64).reshape(cfg.micro_batch,ctx+1)[:,:-1].copy(),device=device)
            saved=rt.cpu(rt.rng());m.train();m.zero_grad(set_to_none=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                with native.autocast_context(device,cfg.precision):zz,_=m(xx);loss=zz.float().square().mean()
                loss.backward();torch.cuda.synchronize(device)
            ops=sorted({v.key for v in prof.key_averages() if 'attention' in v.key.lower()});rt.restore_rng(saved)
        else:ops=['CPU fixture: dispatch not tested on CUDA']
    if rt.treehash(rt.rng())!=before:raise RuntimeError('Evaluation gate leaked RNG')
    return dict(repeatable=True,rng_unchanged=True,offset_zero_matches_dev=True,partition=a,attention_ops=ops)

def verify_checkpoint_payload(directory):
    """Reload one committed payload on CPU and reproduce every stored tensor identity."""
    import torch
    from gct_runtime import treehash
    ck,meta=io.load_ck(directory,torch);o=ck['opt']
    actual=dict(model=treehash(ck['model']),rng=treehash(ck['rng']),adam=treehash(o['adam']),
                canonical=treehash(o['canonical']),scrambled=treehash(o['scrambled']),
                permutations=treehash(o['permutations']),phase=o['phase'],offset=o['offset'])
    if actual!=meta['identity']:raise RuntimeError('Checkpoint payload identity does not reproduce: '+str(directory))
    if o['adam']['age']!=o['offset']+o['phase']:raise RuntimeError('Payload optimizer age inconsistent')
    for name,t in ck['model'].items():
        if not bool(torch.isfinite(t).all()):raise FloatingPointError('Nonfinite checkpoint parameter '+name)
    for name,state in o['adam']['states'].items():
        tensors=[state['m'],state['v']] if state['kind']=='projection' else list(state['state'].values())
        if any(not bool(torch.isfinite(t).all()) for t in tensors):raise FloatingPointError('Nonfinite checkpoint optimizer '+name)
    del ck,o;gc.collect()
    return meta

def seal_cohort(project,runtime):
    root=io.root(project);path=root/'COHORT_LOCK.json';source=io.verify()[0]
    io.engine(project,'gct2')
    endpoints=[];records={};missing=[]
    for seed in io.GCT2['candidate_seeds']:
        for j in stages(project,seed):
            r=io.verify_completed(j['directory'],{'protocol':io.compact(io.GCT2),'seed':seed,'stage':j['stage'],'mode':j['mode'],'runtime':runtime,'gate':False,'tiny':False})
            if r is None:missing.append(j['directory']);continue
            records[(seed,j['stage'],j['mode'])]=r
            if not path.exists():verify_checkpoint_payload(j['directory'])
    if missing:return dict(state='waiting_for_complete_cohort',missing=missing)
    # Reuse all unchanged GCT1 lineage, matched-control, and whole-batch-history checks.
    io.engine(project,'gct2')
    validation=io.load_module('gct_analysis',io.HERE/'frozen/gct1/gct_analysis.py').analyze(project,write=False)
    if validation['state']!='complete':raise RuntimeError('Cohort lineage validation incomplete')
    parents=[]
    for seed in io.GCT2['candidate_seeds']:
        p=records[(seed,'parent','adamw')];prefix=records[(seed,'prefix','adamw')];parents.append(p['final']['model'])
        grown=records[(seed,'grown','adamw')];checks=grown['checks']
        for name in ('weights_exact','named_moments_exact','independent_parameter_storage','independent_moment_storage','both_covariance_histories_zero','rng_preserved'):
            if checks.get(name) is not True:raise RuntimeError('Growth mapping audit failed: '+name)
        if any(checks.get(k)!=v for k,v in dict(parent_age=7630,grown_age=7630,phase=0,next_global_data_step=7631,covariance_depth_dimension=11).items()):raise RuntimeError('Growth age/covariance mismatch')
        for m in io.GCT2['arms']:
            metrics=records[(seed,'arm',m)]['matching']
            if metrics['matched_groups']+metrics['identity_groups']+metrics['numerical_zero_groups']!=8400:raise RuntimeError('Matching accounting mismatch')
            for k,v in metrics.items():
                if isinstance(v,(int,float)) and not math.isfinite(v):raise RuntimeError('Nonfinite matching diagnostics')
        for m in io.GCT2['arms']:
            d=root/f'seed_{seed}'/'arms'/m;r=records[(seed,'arm',m)]
            endpoints.append(dict(seed=seed,arm=m,step=3000,adam_age=10630,parent_steps=7630,prefix_steps=200,
                training_result_sha256=io.sha(d/'result.json'),checkpoint_sha256=r['checkpoint']['sha256'],
                prefix_sha256=prefix['checkpoint']['sha256'],small_parent_sha256=p['checkpoint']['sha256'],
                completed_at=io.read(d/'status.json')['time']))
    if len(set(parents))!=6:raise RuntimeError('Parent models are not independent/distinct')
    gates=[]
    for lane in io.GCT2['lanes']:
        f=root/'lanes'/lane/'training_complete.json';v=io.read(f)
        if v['source']!=source or v['runtime']!=runtime:raise RuntimeError('Lane runtime/source changed')
        for gp in v['gates']:
            g=io.read(io.safe_file(root,gp))
            if g['state']!='passed' or g['source']!=source or g['runtime']!=runtime:raise RuntimeError('Invalid target gate')
            for rel,h in g['evidence'].items():
                if io.sha(io.safe_file(root,rel))!=h:raise RuntimeError('Gate evidence changed')
            gates.append(dict(path=gp,sha256=io.sha(root/gp)))
    fixed=dict(state='sealed',source=source,protocol=io.compact(io.GCT2),runtime=runtime,endpoints=endpoints,gates=gates)
    if path.exists():
        locked=io.read(path)
        if {k:v for k,v in locked.items() if k!='sealed_at'}!=fixed:raise RuntimeError('Sealed cohort changed')
    else:
        locked=dict(fixed,sealed_at=io.now());io.atomic(path,locked)
    return locked

def report_one(project,seed,arm,data):
    root=io.root(project);lock=root/'COHORT_LOCK.json';cohort=io.read(lock)
    if cohort['state']!='sealed' or len(cohort['endpoints'])!=18 or cohort.get('source')!=io.verify()[0] or cohort.get('protocol')!=io.compact(io.GCT2):raise RuntimeError('REPORT forbidden before valid complete cohort seal')
    entry=next(r for r in cohort['endpoints'] if r['seed']==seed and r['arm']==arm)
    directory=root/f'seed_{seed}'/'arms'/arm
    if io.sha(directory/'result.json')!=entry['training_result_sha256']:raise RuntimeError('Training record changed after seal')
    destination=root/'reports'/f'seed_{seed}'/(arm+'.json')
    linkage=dict(seed=seed,arm=arm,training_result_sha256=entry['training_result_sha256'],checkpoint_sha256=entry['checkpoint_sha256'],
                 cohort_sha256=io.sha(lock),training_source_sha256=io.verify()[0],collector_source_sha256=io.sha(Path(__file__)),partition='REPORT')
    if destination.exists():
        prior=io.read(destination)
        if any(prior.get(k)!=v for k,v in linkage.items()):raise RuntimeError('Existing REPORT link changed')
        return prior
    io.guard_cache(data);started=io.now()
    with restored_for_evaluation(project,directory) as (m,cfg,native,device,rt,meta):
        if meta['sha256']!=entry['checkpoint_sha256']:raise RuntimeError('Locked checkpoint mismatch')
        metric=evaluate_slice(m,Path(data['path'])/'val.bin',cfg,native,device,rt)
    io.guard_cache(data)
    result=dict(linkage,**metric,started_at=started,completed_at=io.now(),evaluator='gct2.evaluate_slice',runtime=cohort['runtime'])
    io.atomic(destination,result);return result

def analyze_reports(project):
    root=io.root(project);cohort=io.read(root/'COHORT_LOCK.json');reg=io.read(root/'registration.json')
    # Repeat the finite cohort audit before publishing inference; this verifies actual records, not just flags.
    if seal_cohort(project,cohort['runtime'])!=cohort:raise RuntimeError('Cohort changed before analysis')
    inventory=io.read(root/'seed_inventory.json')
    if not inventory['complete'] or inventory['collisions'] or io.sha(root/'seed_inventory.json')!=reg['inventory_sha256']:raise RuntimeError('Inventory changed')
    for lane in io.GCT2['lanes']:io.guard_cache(io.read(root/'lanes'/lane/'cache.json'))
    sys.path.insert(0,str(io.HERE/'design/tools'));from analyze_gct2 import analyze
    checks={k:True for k in ('source','cache','growth_mapping','optimizer_age','all_parents_distinct','lineage','first_active_inputs','full_batch_rng_histories','matching','final_checkpoints','target_gates')}
    reports=[io.read(root/'reports'/f'seed_{s}'/(a+'.json')) for s in io.GCT2['candidate_seeds'] for a in io.GCT2['arms']]
    if any(r['cohort_sha256']!=io.sha(root/'COHORT_LOCK.json') or r['stop']!=3550624 or r['runtime']!=cohort['runtime'] for r in reports):raise RuntimeError('REPORT cohort/partition/runtime mismatch')
    bundle=dict(schema='gct2_endpoint_bundle_v1',protocol_digest=io.compact(io.GCT2),seeds=io.GCT2['candidate_seeds'],cache_sha256=io.GCT2['cache_sha256'],
                report_partition=io.GCT2['report'],parent_cost_counted_per_path=True,
                inventory=dict(complete=True,collisions=[],manifest_sha256=reg['inventory_sha256']),checks=checks,
                training_source_sha256=io.verify()[0],collector_source_sha256=io.sha(Path(__file__)),cohort_lock=cohort,reports=reports)
    result=analyze(bundle);io.atomic(root/'normalized_reports.json',bundle);io.atomic(root/'analysis.json',result)
    with (root/'endpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['seed',*io.GCT2['arms']]);w.writeheader();w.writerows(result['rows'])
    lines=['# GCT2 final confirmation',result['scope'],'','| Seed | AdamW | BDT | Matched scrambled |','|---|---:|---:|---:|']
    lines += [f"| {r['seed']} | {r['adamw']:.9f} | {r['bdt']:.9f} | {r['scrambled_matched']:.9f} |" for r in result['rows']]
    lines += ['',json.dumps(result['comparisons'],indent=2),f"Joint confirmation: {result['joint_confirmation']}",
              'Established REPORT text; six new parents, pilot excluded. No GS/scratch superiority or time-to-quality claim.']
    (root/'READOUT.md').write_text('\n'.join(lines)+'\n');return result
