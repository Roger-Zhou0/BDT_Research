#!/usr/bin/env python3
"""Local access to the original study trainers; inspection is the default.

CPU tests use synthetic tokens. GPU gate/job/report actions require explicit
consent and a new workspace with an exact external cache. No cluster connection,
notebook management, dataset download, or historical result mutation occurs.
The serial workspace adapter is new packaging code, not the original scheduler.
"""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCES={'cms1':ROOT/'studies/cms1/correspondence_matched_s1_v1',
         'gct2':ROOT/'studies/gct2','bac1':ROOT/'studies/bac1'}

def dump(x): print(json.dumps(x,indent=2,default=str,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_id(study):
    d=SOURCES[study];m=json.loads((d/'SOURCE_MANIFEST.json').read_text())
    for name,h in m.items():
        p=d/name
        if p.is_symlink() or not p.resolve().is_relative_to(d.resolve()) or not p.is_file() or sha(p)!=h:raise RuntimeError('Source mismatch: '+name)
    kw={'sort_keys':True,'allow_nan':False}
    if study=='cms1':kw['separators']=(',',':')
    return hashlib.sha256(json.dumps(m,**kw).encode()).hexdigest()

def plan(study):
    d=SOURCES[study]
    if study=='cms1':
        spec=json.loads((ROOT/'reference/native_cms1/PROTOCOL.json').read_text());seeds=spec['seeds']
        jobs=[]
        for s in seeds:
            jobs.append(dict(seed=s,stage='parent',mode='adamw',end=200,parent=None,rel=f'seed_{s}/parent'))
            for a in spec['order'][str(s)]:jobs.append(dict(seed=s,stage='arm',mode=a,end=3000,parent=f'seed_{s}/parent',rel=f'seed_{s}/arms/{a}'))
    else:
        spec=json.loads((d/('design/protocols/GCT2.json' if study=='gct2' else 'PROTOCOL.json')).read_text())
        seeds=spec['candidate_seeds' if study=='gct2' else 'seeds'];jobs=[]
        for s in seeds:
            for stage,end,parent in [('parent',7630,None),('grown',0,f'seed_{s}/parent'),('prefix',200,f'seed_{s}/grown')]:
                jobs.append(dict(seed=s,stage=stage,mode='adamw',end=end,parent=parent,rel=f'seed_{s}/{stage}'))
            for a in spec['arm_order'][str(s)]:jobs.append(dict(seed=s,stage='arm',mode=a,end=3000,parent=f'seed_{s}/prefix',rel=f'seed_{s}/arms/{a}'))
    return {'study':study,'public_source':source_id(study),'jobs':[dict(index=i,**j) for i,j in enumerate(jobs)],
            'action':'inspection only','cohort_units':len(seeds),'scientific_gpu_validation_in_package':False,
            'note':'Original scientific schedules and native functions; new local serial dispatch. A gate is required before any job, and all endpoints before REPORT.'}

def cpu_test(study,out):
    if out.exists():raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True);env=dict(os.environ,CUDA_VISIBLE_DEVICES='',CUBLAS_WORKSPACE_CONFIG=':4096:8',PYTHONPATH='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    d=SOURCES[study];report=out/'result.json'
    if study=='cms1':
        # Only small source files are staged; the source tree itself remains read-only.
        with tempfile.TemporaryDirectory(prefix='cms1_reviewer_') as td:
            project=Path(td);shutil.copytree(d.parent/'fair_depth_lr_bracketing_v2',project/'fair_depth_lr_bracketing_v2')
            cmd=[sys.executable,str(d/'tests.py'),'--project',str(project),'--output',str(report)]
            with (out/'console.txt').open('x') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=out,timeout=900)
    else:
        env['FOLLOWUP_TEST_OUTPUT']=str(report)
        cmd=([sys.executable,str(d/'tests/run_cpu.py'),'--study','gct2','--output',str(report)] if study=='gct2' else [sys.executable,str(d/'tests/test_suite.py')])
        with (out/'console.txt').open('x') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,cwd=out,timeout=900)
    if r.returncode:raise RuntimeError('Native test failed; retained console: '+str(out/'console.txt'))
    dump(json.loads(report.read_text()))

def gpu_action(args):
    if not args.allow_gpu or args.work is None:raise RuntimeError('GPU actions require --allow-gpu and --work')
    p=args.work.resolve()
    if p==ROOT or p.is_relative_to(ROOT):raise RuntimeError('Use a separate workspace, not the release tree')
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU never substitutes for scientific training')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 support required')
    if torch.cuda.device_count()!=1:raise RuntimeError('Expose exactly one GPU via CUDA_VISIBLE_DEVICES')
    host=p.parent
    if not host.is_dir():raise RuntimeError('Workspace parent must exist')
    if shutil.disk_usage(host).free<80*(1<<30):raise RuntimeError('Require 80 GiB free on compute storage; never stage this on a constrained laptop')
    sid=source_id(args.study)
    if args.action=='init':
        if args.cache is None or not args.cache.resolve().is_dir():raise RuntimeError('--cache must identify the existing exact S1-B cache')
        if p.exists():raise FileExistsError('Initialization requires a NEW workspace')
        p.mkdir();(p/'runs').mkdir();(p/'data_s1').mkdir()
        (p/'data_s1/fineweb_edu_3b').symlink_to(args.cache.resolve(),target_is_directory=True)
        if args.study=='cms1':shutil.copytree(SOURCES['cms1'].parent/'fair_depth_lr_bracketing_v2',p/'fair_depth_lr_bracketing_v2')
        (p/'PORTABLE_WORKSPACE.json').write_text(json.dumps({'study':args.study,'source':sid,'dispatch':'new serial adapter','initialized_only':True},indent=2))
    if not (p/'PORTABLE_WORKSPACE.json').is_file():raise RuntimeError('Not a workspace created by this adapter')
    marker=json.loads((p/'PORTABLE_WORKSPACE.json').read_text())
    if (marker['study'],marker['source'])!=(args.study,sid):raise RuntimeError('Workspace identity differs')
    sys.path.insert(0,str(SOURCES[args.study]))
    if args.study=='cms1':
        import cms_io as io,cms_runtime as rt,cms_protocol as protocol,cms_train as tr,cms_analysis as analysis
        rt.configure(4);native_root=protocol.root(p);verify=io.verify_data
        read=io.read_json;atomic=io.atomic_json;cache_guard=io.cache_guard;source=lambda:io.verify_sources(p)
    else:
        if args.study=='gct2':import core as io,gct2 as experiment
        else:import common as io,experiment
        tr,rt,_=io.engine(p);rt.configure(4);native_root=io.root(p)
        verify=(lambda p:io.cache(p,'gct2')) if args.study=='gct2' else io.cache
        read=io.read;atomic=io.atomic;cache_guard=io.guard_cache;source=lambda:io.verify()[0]
    runtime=(rt.signature(torch,rt.backend(p)['model_support'],torch.device('cuda:0')) if args.study=='cms1' else rt.runtime(torch.device('cuda:0')))
    data_file=p/'PORTABLE_CACHE.json'
    if args.action=='init':
        data=verify(p);atomic(data_file,data)
        if args.study=='cms1':atomic(native_root/'cache_identity.json',data)
        elif args.study=='gct2':
            for lane in io.GCT2['lanes']:atomic(native_root/'lanes'/lane/'cache.json',data)
        else:atomic(native_root/'lanes'/io.SPEC['evaluation_lane']/'cache.json',data)
        dump({'state':'initialized_cache_verified','source':sid,'bytes_copied_from_cache':0});return
    data=read(data_file);cache_guard(data)
    gate_file=p/'PORTABLE_GPU_GATE.json'
    if args.action=='gate':
        d=native_root/'gates/portable';d.mkdir(parents=True,exist_ok=True)
        if args.study=='cms1':
            from cms_analysis import compare_jobs
            common=d/'common';reference=d/'reference_parent'
            opts=dict(tiny=False,gate=True,data_identity=data,resume=args.resume)
            tr.train_job(p,common,127,'adamw',200,**opts);tr.train_job(p,reference,127,'reference_adamw',200,**opts)
            checks={'prefix':compare_jobs(common,reference,reference=True)}
            for a in protocol.ARMS:
                tr.train_job(p,d/(a+'_0'),127,a,420,parent=common,**opts)
                tr.train_job(p,d/(a+'_1'),127,a,420,parent=common,interrupt_at=310,**opts)
                tr.train_job(p,d/(a+'_1'),127,a,420,parent=common,**dict(opts,resume=True))
                checks[a]=compare_jobs(d/(a+'_0'),d/(a+'_1'))
            first=[]
            for a in protocol.ARMS:
                row=next(z for z in read(d/(a+'_0')/'result.json')['comparison_rows'] if z['step']==201)
                first.append({k:row[k] for k in ('batch_hash','rng_before','rng_after_backward','gradient_sha','raw_proposal_sha')})
            if any(x!=first[0] for x in first):raise RuntimeError('First-active inputs differ')
            checks['first_active_inputs_equal']=True
            for a,ref in [('adamw','reference_adamw'),('bdt','reference_bdt')]:
                tr.train_job(p,d/ref,127,ref,420,parent=common,**opts);checks[ref]=compare_jobs(d/(a+'_0'),d/ref,reference=True)
        else:
            for j in experiment.gate_plan(d,False):experiment.execute(p,j,data,resume=True)
            checks=experiment.compare_gate(d)
            if args.study=='gct2':checks['report_evaluator']=experiment.report_gate(p,d,data,False)
        evidence={str(f.relative_to(native_root)):sha(f) for f in d.rglob('*') if f.is_file() and f.name in {'result.json','samples.npz','initialization.json','contract.json'}}
        value={'state':'passed','source':source(),'runtime':runtime,'checks':checks,'evidence':evidence,'scope':'New full-sized local engineering gate; not historical scientific efficacy'}
        atomic(gate_file,value);atomic(native_root/'gates/portable_receipt.json',value);dump(value);return
    gate=read(gate_file)
    if gate['state']!='passed' or gate['source']!=source() or gate['runtime']!=runtime:raise RuntimeError('Current full-sized gate required')
    for n,h in gate['evidence'].items():
        if sha(native_root/n)!=h:raise RuntimeError('Gate evidence changed')
    if args.action=='job':
        jobs=plan(args.study)['jobs']
        if args.index is None or not 0<=args.index<len(jobs):raise ValueError('Valid --index required; see plan')
        j=jobs[args.index];directory=native_root/j['rel'];parent=native_root/j['parent'] if j['parent'] else None
        if args.study=='cms1':out=tr.train_job(p,directory,j['seed'],j['mode'],j['end'],parent=parent,resume=args.resume,data_identity=data)
        else:out=experiment.execute(p,dict(seed=j['seed'],stage=j['stage'],mode=j['mode'],end=j['end'],directory=str(directory),parent=str(parent) if parent else None,gate=False),data,args.resume)
        dump({'state':out.get('state'),'step':out.get('step'),'job':j});return
    if args.action=='report':
        if args.study=='cms1':
            analysis.freeze_cohort(p)
            for s in protocol.SEEDS:
                for a in protocol.ARMS:tr.report_job(p,analysis.trial_dir(p,s,a))
            result=analysis.analyze(p)
        elif args.study=='bac1':
            experiment.analyze(p) # validates all24; writes actual cohort hash before evaluation
            for s in io.SPEC['seeds']:
                for a in io.SPEC['arms']:experiment.evaluate_report(p,s,a)
            result=experiment.analyze(p)
        else:
            # Serial local dispatch supplies actual same-device gate provenance to
            # both logical seed lanes. Never synthesizes historical gate results.
            for lane,config in io.GCT2['lanes'].items():
                for s in config['seeds']:
                    for j in experiment.stages(p,s):
                        if io.verify_completed(j['directory']) is None:raise RuntimeError('Incomplete cohort: REPORT remains closed')
                atomic(native_root/'lanes'/lane/'training_complete.json',{'source':source(),'runtime':runtime,'gates':['gates/portable_receipt.json']})
            inventory={'complete':True,'collisions':[],'seeds':io.GCT2['candidate_seeds'],'scope':'New exclusive workspace; no old study imported'}
            atomic(native_root/'seed_inventory.json',inventory);atomic(native_root/'registration.json',{'inventory_sha256':sha(native_root/'seed_inventory.json'),'source':source(),'dispatch':'new serial adapter'})
            seal=experiment.seal_cohort(p,runtime)
            if seal['state']!='sealed':raise RuntimeError('Incomplete cohort')
            for s in io.GCT2['candidate_seeds']:
                for a in io.GCT2['arms']:experiment.report_one(p,s,a,data)
            result=experiment.analyze_reports(p)
        dump(result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True,choices=SOURCES);p.add_argument('--action',choices=['plan','cpu-test','init','gate','job','report'],default='plan')
    p.add_argument('--out',type=Path);p.add_argument('--work',type=Path);p.add_argument('--cache',type=Path);p.add_argument('--allow-gpu',action='store_true');p.add_argument('--index',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.action=='plan':dump(plan(a.study))
    elif a.action=='cpu-test':
        if a.out is None:raise ValueError('--out required')
        cpu_test(a.study,a.out.resolve())
    else:
        if not a.allow_gpu or a.work is None:raise RuntimeError('GPU actions require --allow-gpu and --work')
        import fcntl
        parent=a.work.resolve().parent
        if not parent.is_dir():raise RuntimeError('Workspace parent must exist')
        with (parent/('.'+a.work.name+'.portable.lock')).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError as exc:raise RuntimeError('Workspace already owned by another process') from exc
            try:gpu_action(a)
            finally:fcntl.flock(lock,fcntl.LOCK_UN)
if __name__=='__main__':
    try:main()
    except (OSError,ValueError,RuntimeError) as e:raise SystemExit(str(e))
