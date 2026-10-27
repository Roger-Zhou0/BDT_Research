#!/usr/bin/env python3
"""Detached GCT2 lanes and BCO1 collector. Importing this file starts no work."""
from __future__ import annotations
import argparse,contextlib,getpass,json,os,re,shutil,signal,socket,subprocess,sys,tempfile,time,zipfile
from pathlib import Path
import core as io
STOP=False
CHILD=None
TARGET_UUID=None

def say(*args):print(io.now(),*args,flush=True)
def target(study):
    account=getpass.getuser();good='good' in socket.gethostname()
    if study=='gct2' and account=='account_b':return 'B-good' if good else 'B-default'
    if study=='bco1' and account=='account_a' and good:return 'A-good'
    raise RuntimeError('GCT2 requires account_b/{good,default}; BCO1 requires account_a/good. No lane spoofing.')
def gpu():
    r=subprocess.run(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    if r.returncode or len(r.stdout.strip().splitlines())!=1:raise RuntimeError('GPU query failed: '+r.stderr)
    fields=[s.strip() for s in r.stdout.strip().split(',')]
    if len(fields)!=4 or 'H200' not in fields[0] or not fields[1].startswith('GPU-'):raise RuntimeError('Require H200 at device0')
    return dict(name=fields[0],uuid=fields[1],free_mib=int(fields[2]),utilization=int(fields[3]))
def runtime_probe(project,study):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='0';env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    r=subprocess.run([sys.executable,str(io.HERE/'study.py'),'_runtime','--study',study,'--project',str(project)],env=env,capture_output=True,text=True,timeout=120)
    if r.returncode:raise RuntimeError('Runtime probe failed: '+r.stderr+r.stdout)
    return json.loads(r.stdout.strip().splitlines()[-1])
def live_work():
    jobs=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if int(proc.name)==os.getpid() or proc.stat().st_uid!=os.getuid():continue
            words=[b.decode(errors='replace') for b in (proc/'cmdline').read_bytes().split(b'\0') if b]
            scripts=[Path(w).name for w in words[1:4] if w.endswith('.py')]
            if any(s in ('scheduler.py','coordinator.py','replay.py') for s in scripts) or any(s in ('study.py','overnight.py') for s in scripts) and any(w in words for w in ('_worker','_coordinate')):
                jobs.append(dict(pid=int(proc.name),command=words))
        except (OSError,ValueError):continue
    return jobs

def memory_wait(gid):
    good=0;start=time.monotonic()
    while good<2:
        if STOP:raise InterruptedError('Stop requested')
        g=gpu()
        if g['uuid']!=gid:raise RuntimeError('GPU changed within coordinator lifetime')
        good=good+1 if g['free_mib']>=32768 else 0
        if good<2:say('GPU ADMISSION',g['free_mib'],'MiB free; require32768');time.sleep(10)
        if time.monotonic()-start>86400:raise RuntimeError('24-hour admission wait exhausted')

def prerequisite(project,study,lane):
    relative='runs/growth_scratch_reference_v1/scheduler_status.json' if study=='gct2' and lane=='B-default' else 'runs/growth_correspondence_transfer_v1/scheduler_status.json' if study=='gct2' else 'runs/optimizer_cost_benchmark_v1/scheduler_status.json'
    f=Path(project)/relative;start=time.monotonic()
    while True:
        if STOP:raise InterruptedError('Stopped while waiting')
        if not f.exists():raise RuntimeError('Prerequisite status missing: '+str(f))
        s=io.read(f)
        if s.get('state')=='finished':return
        if s.get('state') in ('failed','interrupted'):raise RuntimeError('Prerequisite requires inspection: '+str(f))
        say('WAIT_PREREQUISITE',relative,s.get('state'));time.sleep(30)
        if time.monotonic()-start>86400:raise RuntimeError('24-hour prerequisite wait exhausted')

def register(project,study,rt):
    root=io.root(project,study);source=io.verify()[0];spec=io.GCT2 if study=='gct2' else io.BCO1
    with io.lock(Path(project)/'logs'/('.'+study+'_registration.lock'),blocking=True):
        path=root/'registration.json';fixed=dict(protocol=io.compact(spec),source=source,runtime=rt)
        if path.exists():
            reg=io.read(path)
            if reg['fixed']!=fixed:raise RuntimeError('Registered source/protocol/runtime changed')
            if study=='gct2' and io.sha(root/'seed_inventory.json')!=reg['inventory_sha256']:raise RuntimeError('Inventory record changed')
            return reg
        if study=='gct2':
            sys.path.insert(0,str(io.HERE/'design/tools'));from admission import scan
            audit=scan(Path(project))
            if not audit['complete'] or audit['collisions']:raise RuntimeError('Seed collision/incomplete inventory: '+json.dumps(audit['errors']+audit['collisions'])[:1800])
            if shutil.disk_usage(project).free<128*1024**3:raise RuntimeError('GCT2 needs128GiB free shared filesystem at cohort admission')
            extra={}
        else:
            if shutil.disk_usage(project).free<32*1024**3:raise RuntimeError('BCO1 needs32GiB free filesystem')
            extra={}
        # Publish the complete registration directory atomically. A killed initial
        # registration must not leave a half-created study that blocks its own inventory scan.
        if root.exists():raise RuntimeError('Unregistered study directory exists; inspect before registering')
        root.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.'+study+'_registration_',dir=Path(project)))
        try:
            if study=='gct2':
                io.atomic(staging/'seed_inventory.json',audit);extra=dict(inventory_sha256=io.sha(staging/'seed_inventory.json'))
            reg=dict(fixed=fixed,time=io.now(),**extra)
            io.atomic(staging/'registration.json',reg);io.atomic(staging/'PROTOCOL.json',spec)
            os.replace(staging,root);io.fsync_dir(root.parent)
        finally:
            if staging.exists():shutil.rmtree(staging)
        return reg

def expected(j,rt,study):
    if study=='gct2':return dict(seed=j['seed'],stage=j['stage'],mode=j['mode'],end=j['end'],gate=j['gate'],tiny=j.get('tiny',False),runtime=rt,protocol=io.compact(io.GCT2))
    return dict(seed=127,stage='bco_gate',mode=j['condition'],end=j['end'],gate=True,tiny=j.get('tiny',False),runtime=rt,protocol=io.compact(io.BCO1))

def run_process(project,study,lane,label,action,job,identity,rt,data_path,resume=False,output_result=None):
    """Durable per-logical-job OOM budget; immutable registration; no empty skip logs."""
    global CHILD
    root=io.root(project,study);base=root/'jobs'/identity;registration=base/'job.json'
    request=dict(action=action,job=job,source=io.verify()[0],runtime=rt,data_path=str(data_path),study=study)
    # An engineering interruption is an attempt directive, not a different scientific job.
    stable=json.loads(json.dumps(request));stable['job'].pop('interrupt',None)
    io.immutable(registration,stable)
    if action in ('gct_job','bco_job'):
        r=io.verify_completed(job['directory'],expected(job,rt,study))
        if r:
            say('SKIP_COMPLETE',label,io.sha(Path(job['directory'])/'result.json'));return r
        latest=Path(job['directory'])/'latest.json'
        if job.get('interrupt') and latest.exists() and io.read(latest)['step']>=job['interrupt']:
            say('SKIP_COMPLETED_TEST_INTERRUPTION',label);return dict(state='test_paused',step=io.read(latest)['step'])
    if output_result and Path(output_result).exists():
        r=io.read(output_result)
        if r.get('source')!=io.verify()[0] or r.get('runtime')!=rt:raise RuntimeError('Completed engineering artifact changed')
        for rel,h in r.get('evidence',{}).items():
            if io.sha(io.safe_file(Path(output_result).parent,rel))!=h:raise RuntimeError('Completed artifact evidence changed')
        say('SKIP_COMPLETE',label,io.sha(output_result));return r
    ledger=base/'attempts.json';state=io.read(ledger) if ledger.exists() else dict(oom_failures=0,attempts=[])
    if state.get('fatal'):raise RuntimeError('Fatal job blocked; inspect '+str(ledger))
    while True:
        if STOP:raise InterruptedError('Stop requested')
        if state['oom_failures']>=3:raise RuntimeError('Durable OOM limit exhausted; no further attempts')
        io.verify();memory_wait(TARGET_UUID or gpu()['uuid'])
        if shutil.disk_usage(project).free<12*1024**3:raise RuntimeError('Less than12GiB free disk headroom')
        logs=Path(project)/'logs'/study/lane;logs.mkdir(parents=True,exist_ok=True)
        log=logs/(label+'_'+str(time.time_ns())+'.log');instruction=base/('attempt_'+str(time.time_ns())+'.json')
        active=dict(request,resume=resume or bool(state['attempts']));io.atomic(instruction,active)
        env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['CUDA_VISIBLE_DEVICES']='' if action=='cpu' else '0'
        attempt=dict(started=io.now(),log=str(log),gpu_uuid=gpu()['uuid'],request_sha=io.sha(instruction),state='started')
        state['attempts'].append(attempt);io.atomic(ledger,state)
        with log.open('xb',buffering=0) as f:
            CHILD=subprocess.Popen([sys.executable,'-u',str(io.HERE/'study.py'),'_worker','--study',study,'--project',str(project),'--request',str(instruction)],
                stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,cwd=project,env=env)
            io.atomic(root/'lanes'/lane/'status.json',dict(state='running',pid=os.getpid(),child_pid=CHILD.pid,child_log=str(log),label=label,time=io.now(),gpu_uuid=attempt['gpu_uuid']))
            (Path(project)/'logs'/f'{study}_{lane}_child_latest.txt').write_text(str(log)+'\n')
            say('START',label,CHILD.pid,log);code=CHILD.wait()
        CHILD=None;attempt.update(exit_code=code,finished=io.now(),state='complete' if code==0 else 'interrupted' if code<0 or code==76 else 'failed')
        if code==75:state['oom_failures']+=1
        elif code not in (0,76) and code>0:state['fatal']=True
        io.atomic(ledger,state)
        if code==0:
            say('DONE',label)
            if action in ('gct_job','bco_job'):
                if job.get('interrupt'):return dict(state='test_paused')
                completed=io.verify_completed(job['directory'],expected(job,rt,study))
                if completed is None:raise RuntimeError('Child exited without a committed complete result')
                return completed
            return io.read(output_result) if output_result else dict(state='complete')
        if code==75 and state['oom_failures']<3:say('OOM_SAME_RECIPE_RETRY',label);time.sleep(30);continue
        if code<0 or code==76 or STOP:raise InterruptedError('Infrastructure interruption; --resume required')
        raise RuntimeError('Child failed: '+str(log))

def gate_key(lane,gid,rt):return lane+'_'+gid+'_'+io.compact(rt)[:16]+'_'+io.verify()[0][:16]
def verify_gate(root,path,rt):
    g=io.read(path)
    if g['state']!='passed' or g['source']!=io.verify()[0] or g['runtime']!=rt:raise RuntimeError('Invalid prior gate')
    for rel,h in g['evidence'].items():
        if io.sha(io.safe_file(root,rel))!=h:raise RuntimeError('Gate proof changed')
    return g

def gct_gate(project,lane,gid,rt,data_path,resume):
    import gct2
    io.engine(project,'gct2');root=io.root(project);key=gate_key(lane,gid,rt);d=root/'gates'/key;done=d/'gate.json'
    if done.exists():verify_gate(root,done,rt);say('SKIP_PASSED_GPU_GATE',key);return done
    for j in gct2.gate_plan(d):
        name=Path(j['directory']).name+('_interrupt' if j.get('interrupt') else '')
        run_process(project,'gct2',lane,'gate_'+name,'gct_job',j,'gate/'+key+'/'+Path(j['directory']).name,rt,data_path,resume=True)
    checks=gct2.compare_gate(d)
    j=dict(directory=str(d),output=str(d/'report_gate.json'))
    report=run_process(project,'gct2',lane,'gate_report_evaluator','report_gate',j,'gate/'+key+'/report_eval',rt,data_path,resume,output_result=d/'report_gate.json')
    # Measure actual payload sizes before deleting only temporary engineering payloads.
    parent_bytes=(d/'parent'/io.read(d/'parent/latest.json')['path']).stat().st_size
    target_bytes=(d/'bdt_0'/io.read(d/'bdt_0/latest.json')['path']).stat().st_size
    anticipated=6*parent_bytes+6*11*target_bytes
    already=sum(f.stat().st_size for f in root.glob('seed_*/**/checkpoints/*.pt') if f.is_file() and not f.is_symlink())
    if shutil.disk_usage(project).free<max(0,anticipated-already)+8*1024**3:raise RuntimeError('Measured checkpoint footprint exceeds remaining GCT2 space')
    evidence={str(f.relative_to(root)):io.sha(f) for f in d.rglob('*.json') if f.name in ('result.json','contract.json','report_gate.json')}
    g=dict(state='passed',source=io.verify()[0],runtime=rt,gpu_uuid=gid,checks=checks,report_check=report,evidence=evidence,
           measured_payloads=dict(parent=parent_bytes,target=target_bytes,cohort_estimate=anticipated),time=io.now())
    io.atomic(done,g);say('GCT2 FULL-SIZE GROWTH + REPORT GATE: PASS',lane)
    removed=[]
    for f in d.rglob('checkpoints/*.pt'):
        if f.is_symlink() or not f.resolve().is_relative_to(d.resolve()):raise RuntimeError('Unsafe engineering cleanup')
        removed.append(dict(path=str(f.relative_to(root)),sha256=io.sha(f)));f.unlink()
    io.atomic(d/'engineering_cleanup.json',dict(removed=removed,time=io.now()));return done

def gct_coordinate(project,lane,gid,rt,data_path,resume):
    import gct2
    root=io.root(project);done=gct_gate(project,lane,gid,rt,data_path,resume)
    gate_log=root/'lanes'/lane/'gate_history.json';gates=io.read(gate_log) if gate_log.exists() else []
    rel=str(done.relative_to(root))
    if rel not in gates:gates.append(rel);io.atomic(gate_log,gates)
    for seed in io.GCT2['lanes'][lane]['seeds']:
        for j in gct2.stages(project,seed):
            name=f"seed_{seed}_{j['stage']}_{j['mode']}"
            run_process(project,'gct2',lane,name,'gct_job',j,'science/'+name,rt,data_path,resume)
    record=dict(source=io.verify()[0],runtime=rt,gates=gates,seeds=io.GCT2['lanes'][lane]['seeds'])
    training_done=root/'lanes'/lane/'training_complete.json'
    if training_done.exists():
        old=io.read(training_done)
        if any(old[k]!=record[k] for k in ('source','runtime','seeds')):raise RuntimeError('Completed lane identity changed')
        for rel in old['gates']:verify_gate(root,io.safe_file(root,rel),rt)
    else:io.immutable(training_done,record)
    # The first lane exits without holding its GPU waiting for the other. The last finalizes.
    other=[l for l in io.GCT2['lanes'] if not (root/'lanes'/l/'training_complete.json').exists()]
    if other:return dict(state='training_complete_waiting_for_other_lane',waiting_for=other)
    with io.lock(root/'.finalizer.lock',blocking=True):
        lock=gct2.seal_cohort(project,rt)
        if lock['state']!='sealed':return lock
        for seed in io.GCT2['candidate_seeds']:
            for arm in io.GCT2['arms']:
                dest=root/'reports'/f'seed_{seed}'/(arm+'.json')
                if dest.exists():
                    r=io.read(dest)
                    entry=next(x for x in lock['endpoints'] if x['seed']==seed and x['arm']==arm)
                    if r['cohort_sha256']!=io.sha(root/'COHORT_LOCK.json') or r['training_result_sha256']!=entry['training_result_sha256'] or r['checkpoint_sha256']!=entry['checkpoint_sha256']:raise RuntimeError('Existing report cohort mismatch')
                    say('SKIP_COMPLETE REPORT',seed,arm);continue
                j=dict(seed=seed,arm=arm)
                run_process(project,'gct2',lane,f'report_{seed}_{arm}','report',j,f'report/{seed}/{arm}',rt,data_path,resume)
        result=gct2.analyze_reports(project);io.atomic(root/'scheduler_status.json',dict(state='finished',joint_confirmation=result['joint_confirmation'],time=io.now()))
        return dict(state='finished',joint_confirmation=result['joint_confirmation'])

def bco_coordinate(project,lane,gid,rt,data_path,resume):
    root=io.root(project,'bco1');key=gate_key(lane,gid,rt);d=root/'sessions'/key
    others=[f for f in (root/'sessions').glob('*/blocks/*/*.json') if not f.is_relative_to(d)]
    if others:raise RuntimeError('Prior timing blocks on another target are preserved. A new full timing session requires explicit review; no automatic device shopping.')
    io.immutable(d/'SESSION.json',dict(source=io.verify()[0],protocol=io.compact(io.BCO1),runtime=rt,gpu_uuid=gid))
    # Profile original update500 first, without contaminating primary timers.
    run_process(project,'bco1',lane,'profile_reference','profile',dict(directory=str(d/'profile')),'session/'+key+'/profile',rt,data_path,resume,output_result=d/'profile/profile_result.json')
    if (d/'gate.json').exists():verify_gate(root,d/'gate.json',rt)
    else:
        gd=d/'validation'
        jobs=[dict(condition='split_adam',end=200,parent=None,directory=str(gd/'prefix'))]
        for name,condition in [('reference','reference_bdt'),('candidate','candidate_bdt'),('resumed','candidate_bdt')]:
            if name=='resumed':jobs.append(dict(condition=condition,end=1000,parent=str(gd/'prefix'),directory=str(gd/name),interrupt=310))
            jobs.append(dict(condition=condition,end=1000,parent=str(gd/'prefix'),directory=str(gd/name)))
        for j in jobs:
            name=Path(j['directory']).name
            run_process(project,'bco1',lane,'gate_'+name,'bco_job',j,'session/'+key+'/gate/'+name,rt,data_path,True)
        # Comparison in its own process keeps the coordinator free of CUDA state.
        run_process(project,'bco1',lane,'compare_candidate','bco_compare',dict(directory=str(gd),output=str(d/'gate.json'),gpu_uuid=gid),
                    'session/'+key+'/compare',rt,data_path,resume,output_result=d/'gate.json')
    say('BCO1 FULL-SIZE EXACT-EQUIVALENCE GATE: PASS')
    run_process(project,'bco1',lane,'balanced_timing','timing',dict(directory=str(d),gpu_uuid=gid),
                'session/'+key+'/timing',rt,data_path,resume,output_result=d/'timing_complete.json')
    result=io.read(d/'analysis.json');io.atomic(root/'analysis.json',dict(session=str(d.relative_to(root)),results=result))
    (root/'READOUT.md').write_text((d/'READOUT.md').read_text());io.atomic(root/'scheduler_status.json',dict(state='finished',session=str(d.relative_to(root)),time=io.now()))
    return dict(state='finished',session=str(d.relative_to(root)))

def coordinate(project,study,lane,resume):
    global STOP,CHILD,TARGET_UUID
    def stop(*args):
        global STOP
        STOP=True
        if CHILD is not None and CHILD.poll() is None:CHILD.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if target(study)!=lane:raise RuntimeError('Wrong lane')
    root=io.root(project,study);status=root/'lanes'/lane/'status.json'
    with io.lock(Path(project)/'logs'/('.'+study+'_'+lane+'.lock')):
        if status.exists() and not resume:raise RuntimeError('Previously started lane; use --resume only after inspection')
        try:
            prerequisite(project,study,lane)
            if live_work():raise RuntimeError('Other local training/diagnostic processes found; nothing interrupted: '+repr(live_work()))
            g=gpu();TARGET_UUID=g['uuid']
            with io.lock(Path(project)/'logs/.practical_gpu_locks'/(g['uuid']+'.lock')):
                rt=runtime_probe(project,study);register(project,study,rt)
                cache=io.cache(project,study);data_path=root/'lanes'/lane/'cache.json';io.atomic(data_path,cache)
                run_process(project,study,lane,'cpu_release_tests','cpu',dict(), 'cpu/'+lane,rt,data_path,resume,
                            output_result=root/'lanes'/lane/'cpu_tests.json')
                if study=='gct2':state=gct_coordinate(project,lane,g['uuid'],rt,data_path,resume)
                else:state=bco_coordinate(project,lane,g['uuid'],rt,data_path,resume)
                io.atomic(status,dict(state,time=io.now(),pid=os.getpid(),gpu_uuid=g['uuid']));say(study.upper(),'LANE COMPLETE',lane,'No automatic follow-up')
        except BaseException as e:
            destination=status if (root/'registration.json').exists() else Path(project)/'logs'/f'{study}_{lane}_admission_error.json'
            io.atomic(destination,dict(state='interrupted' if STOP or isinstance(e,(KeyboardInterrupt,InterruptedError)) else 'failed',error=str(e),time=io.now(),pid=os.getpid()));raise

def worker(project,study,request):
    req=io.read(request)
    if req['source']!=io.verify()[0] or req['study']!=study:raise RuntimeError('Worker request source/study changed')
    import torch
    tr,rt,mods=io.engine(project,study);rt.configure(4);action=req['action'];j=req['job'];data=io.read(req['data_path'])
    # CPU test process does not initiate a CUDA context.
    if action=='cpu':
        result=subprocess.run([sys.executable,str(io.HERE/'tests/run_cpu.py'),'--project',str(project),'--study',study],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        if result.returncode:raise RuntimeError('Release CPU suite failed')
        dest=io.root(project,study)/'lanes'/target(study)/'cpu_tests.json';io.atomic(dest,dict(state='passed',source=io.verify()[0],runtime=req['runtime'],time=io.now()));return
    if rt.runtime(torch.device('cuda:0'))!=req['runtime']:raise RuntimeError('Worker runtime changed')
    io.guard_cache(data)
    try:
        if action=='gct_job':
            import gct2;r=gct2.execute(project,j,data,req['resume'])
        elif action=='report_gate':
            import gct2;r=gct2.report_gate(project,j['directory'],data)
            io.atomic(j['output'],dict(r,source=io.verify()[0],runtime=req['runtime']))
        elif action=='report':
            import gct2;r=gct2.report_one(project,j['seed'],j['arm'],data)
        else:
            import bco
            if action=='bco_job':r=bco.gate_job(project,j,data,req['resume'])
            elif action=='profile':
                r=bco.profile_reference(project,j['directory'],data,torch.device('cuda:0'))
                io.atomic(Path(j['directory'])/'profile_result.json',dict(state='complete',source=io.verify()[0],runtime=req['runtime'],summary_sha=io.sha(Path(j['directory'])/'profile_summary.json'),evidence={n:io.sha(Path(j['directory'])/n) for n in ('profile_summary.json','trace.json')}))
            elif action=='bco_compare':
                r=bco.gate_compare(j['directory']);d=Path(j['directory']);root=io.root(project,study)
                r.update(source=io.verify()[0],runtime=req['runtime'],gpu_uuid=j['gpu_uuid'],evidence={str((d/n/'result.json').relative_to(root)):h for n,h in r['evidence'].items()})
                io.atomic(j['output'],r)
            elif action=='timing':
                r=bco.run_timing(project,j['directory'],data,j['gpu_uuid'],gpu)
                io.atomic(Path(j['directory'])/'timing_complete.json',dict(state='complete',source=io.verify()[0],runtime=req['runtime'],time=io.now(),evidence={str(f.relative_to(Path(j['directory']))):io.sha(f) for f in Path(j['directory']).glob('blocks/*/[0-9][0-9].json')}))
            else:raise RuntimeError('Unknown worker action')
        if r.get('state')=='interrupted':raise SystemExit(76)
    except torch.OutOfMemoryError:
        import traceback;traceback.print_exc();raise SystemExit(75)

def launch(project,study,resume):
    lane=target(study);io.verify();logs=Path(project)/'logs';logs.mkdir(exist_ok=True)
    with io.lock(logs/('.'+study+'_'+lane+'_launch.lock')):
        with io.lock(logs/('.'+study+'_'+lane+'.lock')):pass
        f=logs/(study+'_'+lane+'_'+str(time.time_ns())+'.log')
        cmd=[sys.executable,'-u',str(io.HERE/'study.py'),'_coordinate','--study',study,'--project',str(project),'--lane',lane]
        if resume:cmd.append('--resume')
        env=dict(os.environ,CUBLAS_WORKSPACE_CONFIG=':4096:8',CUDA_VISIBLE_DEVICES='0')
        with f.open('xb',buffering=0) as out:p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,cwd=project,env=env,start_new_session=True)
        (logs/f'{study}_{lane}_latest.txt').write_text(str(f)+'\n');time.sleep(.7)
        if p.poll() not in (None,0):raise RuntimeError(f.read_text())
        print(f'LANE={lane}\nCOORDINATOR_PID={p.pid}\nLOG={f}\nDetached. Validation must pass before scientific training/timing.')

def status(project,study):
    r=io.root(project,study);paths=sorted((r/'lanes').glob('*/status.json'))
    for p in paths:
        s=io.read(p);print(p.parent.name,json.dumps(s,indent=2))
        if s.get('child_log') and Path(s['child_log']).is_file():print('\n'.join(Path(s['child_log']).read_text(errors='replace').splitlines()[-6:]))
    if study=='gct2':
        print('Parents:',len(list(r.glob('seed_*/parent/result.json'))),'/6')
        print('Continuations:',len(list(r.glob('seed_*/arms/*/result.json'))),'/18')
        print('REPORT:',len(list(r.glob('reports/seed_*/*.json'))),'/18')
    else:
        for p in r.glob('sessions/*/progress.json'):print(str(p.parent.name),io.read(p))
        print('Timing blocks:',len(list(r.glob('sessions/*/blocks/*/[0-9][0-9].json'))),'/24 in one session')
    if not paths:print('No lane status yet; inspect the coordinator log for admission/prerequisite waiting.')

def export(project,study):
    r=io.root(project,study);d=Path(project)/'exports';d.mkdir(exist_ok=True);out=d/(study+'_'+str(time.time_ns())+'.zip');manifest={}
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        for base,prefix in [(r,'results'),(io.HERE,'source'),(Path(project)/'logs'/study,'logs')]:
            for f in base.rglob('*'):
                if not f.is_file() or f.is_symlink() or 'checkpoints' in f.parts or '__pycache__' in f.parts or f.suffix not in ('.json','.jsonl','.py','.md','.csv','.log','.sh'):continue
                if f.name=='trace.json':continue
                raw=f.read_bytes()
                if f.suffix in ('.jsonl','.log'):raw=b'\n'.join(raw.splitlines()[-500:])+b'\n'
                name=prefix+'/'+str(f.relative_to(base));z.writestr(name,raw);manifest[name]=__import__('hashlib').sha256(raw).hexdigest()
        z.writestr('MANIFEST.json',json.dumps(manifest,indent=2));z.writestr('SCOPE.txt','Per-file snapshot. No checkpoint tensors, token arrays or profiler traces.\n')
    print('EXPORT_ZIP='+str(out))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['launch','status','integrity','export','_coordinate','_worker','_runtime'])
    ap.add_argument('--study',choices=['gct2','bco1'],required=True);ap.add_argument('--project',type=Path,default=io.HERE.parent)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--lane');ap.add_argument('--request',type=Path);a=ap.parse_args();p=a.project.resolve()
    if a.action=='integrity':print('GCT2/BCO1 ALL BUNDLED SOURCE HASHES: PASS',io.verify()[0])
    elif a.action=='launch':launch(p,a.study,a.resume)
    elif a.action=='status':status(p,a.study)
    elif a.action=='export':export(p,a.study)
    elif a.action=='_coordinate':coordinate(p,a.study,a.lane,a.resume)
    elif a.action=='_worker':worker(p,a.study,a.request)
    else:
        import torch
        _,rt,_=io.engine(p,a.study);rt.configure(4)
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 required')
        print(json.dumps(rt.runtime(torch.device('cuda:0'))))
if __name__=='__main__':
    try:main()
    except Exception:
        import traceback;traceback.print_exc();sys.exit(1)
