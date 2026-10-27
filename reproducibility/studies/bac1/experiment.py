"""Prospective follow-ups. Historical files are read-only dependencies."""
from __future__ import annotations
import json, math, statistics, copy, gc, os, sys, subprocess, time
from pathlib import Path
from dataclasses import asdict
import common as io
import _legacy_experiment as legacy
from paired_analysis import six_parent_contrast, first_crossing_bracket


def job(d, seed, stage, mode, end, parent=None, **extra):
    return dict(directory=str(d), seed=seed, stage=stage, mode=mode, end=end,
                parent=str(parent) if parent else None, gate=False, **extra)


def science_jobs(project, seeds=None):
    jobs = []
    for seed in io.SPEC['seeds'] if seeds is None else seeds:
        d = io.root(project) / f'seed_{seed}'
        if io.SPEC['kind'] == 'ancestry_confirmation':
            jobs += [job(d/'parent', seed, 'parent', 'adamw', 7630),
                     job(d/'grown', seed, 'grown', 'adamw', 0, d/'parent')]
        else:
            jobs += [job(d/'grown', seed, 'import', 'adamw', 0)]
        jobs += [job(d/'prefix', seed, 'prefix', 'adamw', 200, d/'grown')]
        jobs += [job(d/'arms'/m, seed, 'arm', m, io.SPEC['post_updates'], d/'prefix')
                 for m in io.SPEC['arm_order'][str(seed)]]
    return jobs


def gate_modes():
    return tuple(io.SPEC['arms'])


def gate_plan(directory, tiny=False):
    # Reference parent/growth/prefix and all four active policy branches.
    # The window policy's true late boundaries additionally need boundary_tests.py.
    d = Path(directory)
    def j(name, stage, mode, end, parent=None, reference=False, interrupt=None):
        return dict(directory=str(d/name), seed=127, stage=stage, mode=mode,
                    end=end, parent=str(d/parent) if parent else None,
                    reference=reference, gate=True, tiny=tiny, interrupt=interrupt)
    out = [j('ref_parent','parent','adamw',200,reference=True),
           j('ref_grown','grown','adamw',0,'ref_parent',reference=True),
           j('ref_prefix','prefix','adamw',200,'ref_grown',reference=True),
           j('imported','import','adamw',0,'ref_grown'),
           j('prefix','prefix','adamw',200,'imported'),
           j('prefix_repeat','prefix','adamw',200,'imported')]
    for mode in gate_modes():
        out += [j(mode+'_0','arm',mode,420,'prefix')]
        # Every policy is checked across an actual saved-file interruption.
        out += [j(mode+'_1','arm',mode,420,'prefix',interrupt=310),
                j(mode+'_1','arm',mode,420,'prefix')]
    out += [j('ref_adam','arm','adamw',420,'ref_prefix',reference=True),
            j('ref_ffn','arm','bdt',420,'ref_prefix',reference=True)]
    return out


def import_window_parent(project, j, data):
    """New horizon at phase zero only; preserve learned weights/Adam age/RNG.

    A changed cosine horizon is explicit in the new contract. No historical
    checkpoint or contract is rewritten, and no cooled endpoint is extended.
    """
    import torch
    from scope_policy import Policy
    tr, rt, mods = io.engine(project)
    tiny, gate = j.get('tiny', False), j['gate']
    device = torch.device('cpu' if tiny else 'cuda:0')
    seed, d = j['seed'], Path(j['directory'])
    original = (Path(j['parent']) if gate else
                Path(project)/'runs'/io.SPEC['source_parent_study']/f'seed_{seed}'/'grown')
    old = io.read(original/'result.json'); ck, meta = io.load_ck(original, torch)
    if old['state'] != 'complete' or old['step'] != 0 or old['checkpoint'] != meta:
        raise RuntimeError('Source is not a complete phase-zero grown state')
    if legacy.old_identity(ck, rt) != meta['identity'] or ck['contract'] != old['contract']:
        raise RuntimeError('Source payload/contract identity mismatch')
    cfg = tr.config(mods['native'], data['path'], True, tiny)
    expected_cfg = asdict(cfg)
    if not gate:
        expected_cfg['train_steps'] = 3000
    c = old['contract']
    for key, value in {'seed':seed, 'post':True, 'offset':200 if gate else 7630,
                       'gate':gate, 'tiny':tiny, 'config':expected_cfg,
                       'cache':data['hashes'], 'runtime':rt.runtime(device)}.items():
        if c.get(key) != value:
            raise RuntimeError('Unexpected source grown field: '+key)
    if not gate:
        for key in ('source','protocol'):
            if c[key] != io.SPEC['source_parent_'+key+'_sha256']:
                raise RuntimeError('Wrong parent-source registration')
        pr = io.read(original.parent/'parent/result.json')
        if pr['step'] != 7630 or c['parent_sha'] != pr['checkpoint']['sha256']:
            raise RuntimeError('Small-parent lineage mismatch')
    ct = dict(c, version=io.SPEC['version'], protocol=io.digest(io.SPEC),
              source=io.verify()[0], dependencies=__import__('gct_protocol').DEPENDENCIES,
              config=asdict(cfg), parent_sha=meta['sha256'], mode='adamw', stage='grown')
    with io.lock(d/'.writer.lock'):
        io.immutable(d/'contract.json',ct)
        if (d/'result.json').exists():
            return io.completed(d, ct)
        model = tr.build(mods['native'],cfg,seed).to(device)
        model.load_state_dict(ck['model'],strict=True)
        opt = Policy(model,cfg,mods,'adamw',device,seed,True,c['offset'])
        opt.load_legacy(ck['opt']); rt.restore_rng(ck['rng'])
        ident=tr.identity(model,opt)
        for key in ('model','adam','rng','phase','offset','canonical','scrambled','permutations'):
            if ident[key] != meta['identity'][key]:
                raise RuntimeError('Phase-zero import changed '+key)
        dev=rt.evaluate(model,Path(data['path'])/'val.bin',cfg,mods['native'],device,tiny)
        if dev != old['grown_dev']:
            raise RuntimeError('Phase-zero evaluation failed exact reproduction')
        payload=dict(step=0,contract=ct,model=rt.cpu(model.state_dict()),opt=opt.state(),
                     rng=rt.cpu(rt.rng()),initial=ident,curve=[],history=[],proof=[],
                     timing=dict(training_step_seconds=0.,evaluation_seconds=0.,checkpoint_seconds=0.,updates=0))
        new=io.save_ck(d,payload,torch,ident,keep=True)
        result=dict(state='complete',step=0,contract=ct,contract_sha=io.digest(ct),
                    checkpoint=new,initial=ident,final=ident,grown_dev=dev,
                    imported=dict(path=str(original),result_sha256=io.sha(original/'result.json'),
                                  checkpoint_sha256=meta['sha256'],old_horizon=c['config']['train_steps'],
                                  new_horizon=cfg.train_steps,phase_zero_only=True,
                                  discarded_unused_zero_attention_histories=True))
        io.atomic(d/'initialization.json',ident);io.atomic(d/'result.json',result)
        return result


def execute(project,j,data,resume=False):
    if j['stage']=='import':
        if io.SPEC['kind']=='window_pilot':
            return import_window_parent(project,j,data)
        # Confirmation uses this only in engineering; science pretrains new parents.
        if not j['gate']:
            raise RuntimeError('Fresh confirmation must not import a development parent')
        return legacy.import_grown(project,j['seed'],j['directory'],data,
                                   j.get('tiny',False),True,j['parent'])
    tr,rt,_=io.engine(project,reference=j.get('reference',False));rt.configure(4)
    kw=dict(gate=j['gate'],tiny=j.get('tiny',False),data=data,resume=resume)
    if j['stage']=='grown':
        return tr.grow(project,j['directory'],j['seed'],j['parent'],**kw)
    return tr.train(project,j['directory'],j['seed'],j['stage'],j['mode'],j['end'],
                    j.get('parent'),interrupt=j.get('interrupt'),**kw)


def compare_gate(d):
    import torch
    _,rt,_=io.engine(Path(d).parents[2]);d=Path(d);out={}
    pairs=[('prefix','prefix_repeat',False),('prefix','ref_prefix',True),
           ('adamw_0','ref_adam',True),('bdt_0','ref_ffn',True)]
    pairs += [(m+'_0',m+'_1',False) for m in gate_modes()]
    for a,b,ref in pairs:
        x=io.read(d/a/'result.json');y=io.read(d/b/'result.json')
        ca,ma=io.load_ck(d/a,torch);cb,mb=io.load_ck(d/b,torch)
        if not x['proof'] or len(x['proof']) != len(y['proof']):
            raise RuntimeError('Missing gate proof')
        for key in ('model','rng'):
            if rt.treehash(ca[key]) != rt.treehash(cb[key]):
                raise RuntimeError('Gate payload mismatch '+a+'/'+b+'/'+key)
        if rt.treehash(ca['opt']['adam']) != rt.treehash(cb['opt']['adam']):
            raise RuntimeError('Gate moments/counters mismatch')
        for u,v in zip(x['proof'],y['proof']):
            keys=('step','global_step','loss','grad_norm','lr','batch','rng_before','rng_after','gradient_sample','parameter_sample','dev')
            if any(u.get(k)!=v.get(k) for k in keys):
                raise RuntimeError('Gate trace mismatch at '+str(u['step']))
        if not ref and rt.treehash(ca['opt']) != rt.treehash(cb['opt']):
            raise RuntimeError('Full optimizer repeat/resume mismatch')
        out[a+'__'+b]=dict(model_adam_rng_exact=True,full_optimizer_exact=not ref,
                           deliberate_reference_difference='unused proposal histories' if ref else None)
    rows=[next(z for z in io.read(d/(m+'_0')/'result.json')['curve'] if z['step']==201) for m in gate_modes()]
    for key in ('gradient','raw','batch','rng_before','rng_after'):
        if len({z[key] for z in rows})!=1:
            raise RuntimeError('First-active inputs differ')
    out['first_active_inputs_equal']=True
    return out


def check_original_prefix(project,seed,directory):
    # Each follow-up starts a newly declared prefix. No historical horizon is reused.
    r=io.completed(directory,{'seed':seed,'stage':'prefix','end':200})
    if r is None:
        raise RuntimeError('Missing new prefix')


def validated_rows(project):
    root=io.root(project);rows=[];hashes={};parents=[]
    for seed in io.SPEC['seeds']:
        d=root/f'seed_{seed}';prefix=io.completed(d/'prefix',{'seed':seed,'stage':'prefix','end':200,'gate':False,'tiny':False})
        grown=io.completed(d/'grown',{'seed':seed,'stage':'grown','end':0,'gate':False,'tiny':False})
        if prefix is None or grown is None or prefix['initial']!=grown['final']:
            raise RuntimeError('Invalid grown/prefix chain')
        if io.SPEC['kind']=='ancestry_confirmation':
            parent=io.completed(d/'parent',{'seed':seed,'stage':'parent','end':7630,'gate':False,'tiny':False})
            if parent is None or grown['contract']['parent_sha']!=parent['checkpoint']['sha256']:
                raise RuntimeError('Invalid fresh-parent lineage')
            parents.append(parent['final']['model'])
        z={'seed':seed};history=None;inputs=None
        for mode in io.SPEC['arms']:
            f=d/'arms'/mode;v=io.completed(f,{'seed':seed,'stage':'arm','mode':mode,'end':io.SPEC['post_updates'],'gate':False,'tiny':False})
            if v is None or v['initial']!=prefix['final'] or v['contract']['parent_sha']!=prefix['checkpoint']['sha256']:
                raise RuntimeError('Incomplete or nonshared branch')
            expected_steps=list(range(201,io.SPEC['post_updates']+1))
            if [x['step'] for x in v['history']]!=expected_steps or any(x['global_step']!=7630+x['step'] for x in v['history']):
                raise RuntimeError('Invalid phase/data history')
            first=next(x for x in v['curve'] if x['step']==201)
            identity={k:first[k] for k in ('gradient','raw','batch','rng_before','rng_after')}
            if history is not None and (history!=v['history'] or inputs!=identity):
                raise RuntimeError('Branch history/first-active mismatch')
            history,inputs=v['history'],identity
            if v['dev']['offset']!=0 or v['dev']['tokens']!=1048576 or not math.isfinite(v['dev']['loss']):
                raise RuntimeError('Wrong development evaluation')
            z[mode]=v['dev']['loss'];hashes[str((f/'result.json').relative_to(root))]=io.sha(f/'result.json')
            if io.SPEC['kind']=='ancestry_confirmation':
                a,b=(v['chart_design']['draw0_'+k] for k in ('preserved','reassigned'))
                if a['chart_multiset_sha256']!=b['chart_multiset_sha256'] or a['chart_multiplicities']!=[2]*6 or b['chart_multiplicities']!=[2]*6 or a['preserved_ancestry_pairs']!=6 or b['preserved_ancestry_pairs']!=0:
                    raise RuntimeError('Balanced ancestry assignment changed')
        rows.append(z)
    if io.SPEC['kind']=='ancestry_confirmation' and len(set(parents))!=6:
        raise RuntimeError('Six independently pretrained parent states required')
    return rows,hashes


def evaluate_report(project,seed,mode):
    import torch
    root=io.root(project);lock=io.read(root/'COHORT_LOCK.json')
    if len(lock['endpoints']) != 24 or lock['source']!=io.verify()[0]:
        raise RuntimeError('Full confirmation REPORT barrier not satisfied')
    rel=f'seed_{seed}/arms/{mode}/result.json'
    if rel not in lock['endpoints'] or io.sha(root/rel)!=lock['endpoints'][rel]:
        raise RuntimeError('Unregistered or changed REPORT endpoint')
    tr,rt,mods=io.engine(project);rt.configure(4)
    d=root/f'seed_{seed}'/'arms'/mode;v=io.completed(d);ck,meta=io.load_ck(d,torch)
    data=io.read(root/'lanes'/io.SPEC['evaluation_lane']/'cache.json');io.guard_cache(data)
    device=torch.device('cuda:0')
    if rt.runtime(device)!=v['contract']['runtime']:
        raise RuntimeError('REPORT runtime differs')
    cfg=tr.config(mods['native'],data['path'],True,False);model=tr.build(mods['native'],cfg,seed).to(device)
    model.load_state_dict(ck['model'],strict=True)
    from scope_policy import Policy
    opt=Policy(model,cfg,mods,mode,device,seed,True,7630);opt.load(ck['opt']);rt.restore_rng(ck['rng'])
    if tr.identity(model,opt)!=meta['identity']:
        raise RuntimeError('REPORT state identity failed')
    from report_slice import evaluate_slice
    before=rt.treehash(rt.rng());scored=evaluate_slice(model,Path(data['path'])/'val.bin',cfg,mods['native'],device,2500000,2048)
    if rt.treehash(rt.rng())!=before:
        raise RuntimeError('Evaluation consumed training RNG')
    out=dict(scored,seed=seed,mode=mode,partition='REPORT',source=io.verify()[0],
             cohort_sha256=io.sha(root/'COHORT_LOCK.json'),result_sha256=io.sha(d/'result.json'),
             checkpoint_sha256=meta['sha256'])
    io.immutable(root/'reports'/f'seed_{seed}'/(mode+'.json'),out)
    return out


def analyze(project):
    root=io.root(project);dev,hashes=validated_rows(project)
    io.immutable(root/'COHORT_LOCK.json',dict(source=io.verify()[0],protocol=io.digest(io.SPEC),
                                           endpoints=hashes,seeds=io.SPEC['seeds'],arms=io.SPEC['arms']))
    if io.SPEC['kind']=='ancestry_confirmation':
        # Evaluation remains deferred to the designated account-B-good lane, even if
        # the other lane produces the last endpoint. No early REPORT peeking.
        rows=[]
        for seed in io.SPEC['seeds']:
            z={'seed':seed}
            for mode in io.SPEC['arms']:
                f=root/'reports'/f'seed_{seed}'/(mode+'.json')
                if not f.exists():
                    return dict(state='awaiting_locked_report',candidate=False)
                v=io.read(f)
                if v['cohort_sha256']!=io.sha(root/'COHORT_LOCK.json') or v['result_sha256']!=hashes[f'seed_{seed}/arms/{mode}/result.json'] or v['partition']!='REPORT' or v['offset']!=2500000 or v['tokens']!=1048576:
                    raise RuntimeError('REPORT binding mismatch')
                z[mode]=v['loss']
            rows.append(z)
        primary=six_parent_contrast(rows,io.SPEC['seeds'],'draw0_reassigned','draw0_preserved')
        out=dict(state='complete',rows=rows,primary=primary,
                 candidate=primary['ci95'][0]>0 and not primary['zero_variance_review'],
                 inference=io.SPEC['inference'])
    else:
        summaries=[]
        for z in dev:
            seed=z['seed'];d=root/f'seed_{seed}';pref=io.read(d/'prefix/result.json')
            row=dict(seed=seed,losses={m:z[m] for m in io.SPEC['arms']},policies={})
            for m in io.SPEC['arms']:
                v=io.read(d/'arms'/m/'result.json')
                prefix_seconds=pref['timing']['training_step_seconds']
                points=[dict(step=x['step'],loss=x['dev']['loss'],training_seconds=prefix_seconds+x['cumulative_training_seconds']) for x in v['curve'] if isinstance(x.get('dev'),dict)]
                row['policies'][m]=dict(timing=v['timing'],shared_prefix_training_seconds=prefix_seconds,
                    crossings={str(t):first_crossing_bracket(points,t) for t in io.SPEC['quality_targets']},
                    curve=points,loss=z[m])
            row['early_minus_adam']=z['early_bdt']-z['adamw'];row['early_minus_continuous']=z['early_bdt']-z['bdt']
            summaries.append(row)
        out=dict(state='complete',rows=dev,details=summaries,candidate=False,
                 inference=io.SPEC['inference'],automatic_extension=False,
                 note='Two-family development. No automatic winner, noninferiority claim or speed recommendation.')
    out.update(source=io.verify()[0],protocol=io.digest(io.SPEC),time=io.now(),automatic_extension=False)
    io.atomic(root/'analysis.json',out)
    return out
