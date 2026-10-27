"""Bounded single-GPU coordinator. No edits to historical experiments."""
from __future__ import annotations
import contextlib
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
import uuid
import cms_protocol as p
import cms_io as io

_ACTIVE = None
_STOP = False


def say(*args):print(io.now(),*args,flush=True)


def stop_handler(signum,frame):
    global _STOP
    _STOP=True
    if _ACTIVE is not None and _ACTIVE.poll() is None:
        _ACTIVE.terminate()


def gpu_info():
    z=subprocess.run(['nvidia-smi','-i','0','--query-gpu=name,uuid,memory.free,utilization.gpu',
                      '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
    if z.returncode or not z.stdout.strip():raise RuntimeError('Index-0 GPU unavailable: '+z.stdout+z.stderr)
    lines=z.stdout.strip().splitlines()
    if len(lines)!=1:raise RuntimeError('Expected a single exposed index-0 GPU')
    parts=[x.strip() for x in lines[0].split(',')]
    if len(parts)!=4 or not re.fullmatch(r'GPU-[A-Za-z0-9-]+',parts[1]):raise RuntimeError('Unexpected GPU identity')
    return dict(name=parts[0],uuid=parts[1],free_mib=int(parts[2]),utilization=int(parts[3]))


def check_account():
    if getpass.getuser()!='account_b' or 'good' not in socket.gethostname():
        raise RuntimeError('Run only in the account_b GOOD notebook session; do not mix S1-B and OWN-S1-R1')


def seed_audit(project):
    roots=Path(project)/'runs';own=p.root(project).resolve();conflicts=[]
    regex=re.compile(r'(?:^|[^0-9])seed[_-]?('+ '|'.join(map(str,p.SEEDS))+r')(?:[^0-9]|$)',re.I)
    for directory,subdirs,files in os.walk(roots,followlinks=False):
        here=Path(directory)
        subdirs[:]=[n for n in subdirs if not (here/n).is_symlink() and (here/n).resolve()!=own]
        if here.resolve()==own:subdirs[:]=[];continue
        for name in subdirs+files:
            if regex.search(name):conflicts.append(str(here/name))
        for name in files:
            if name not in ('result.json','contract.json','diverged.json'):continue
            path=here/name
            if path.is_symlink() or path.stat().st_size>10<<20:continue
            try:
                x=io.read_json(path);seed=x.get('seed',(x.get('contract') or {}).get('seed'))
                if seed in p.SEEDS:conflicts.append(str(path))
            except (ValueError,AttributeError,OSError):continue
    if conflicts:raise RuntimeError('Reserved confirmation seeds appear in existing evidence; do not replace them silently: '+repr(sorted(set(conflicts))))
    return dict(checked_at=io.now(),seeds=list(p.SEEDS),conflicts=[],scope='existing runs path names and terminal/contract seed fields')


def live_training_processes():
    """Best effort within this notebook PID namespace; UUID flock is additional protection."""
    out=[]
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            if proc.stat().st_uid!=os.getuid() or int(proc.name)==os.getpid():continue
            args=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            scripts=[s for s in args[1:4] if s.endswith('.py')]
            if any(s.endswith('/run.py') for s in scripts) and 'train' in args:out.append(args)
            elif any(s.endswith(('/replay.py','/execution_control_probe_v1.py')) for s in scripts):out.append(args)
            elif any(s.endswith('/study.py') for s in scripts) and '_worker' in args:out.append(args)
        except (OSError,ValueError):continue
    return out


def wait_gpu(initial_uuid,timeout=24*3600):
    start=time.monotonic();consecutive=0
    while consecutive<2:
        if _STOP:raise InterruptedError('Stop requested')
        g=gpu_info()
        if g['uuid']!=initial_uuid:raise RuntimeError('Assigned physical GPU changed during coordinator lifetime')
        consecutive=consecutive+1 if g['free_mib']>=32768 else 0
        if consecutive>=2:return g
        if time.monotonic()-start>timeout:raise RuntimeError('24-hour GPU-memory wait exhausted')
        if not consecutive:say('[memory wait]',g['free_mib'],'MiB; need 32768')
        time.sleep(15)


def child(project,command,label,status_path,gpu_uuid,*,gpu=True,resume=False):
    global _ACTIVE
    logdir=Path(project)/'logs/cms1';logdir.mkdir(parents=True,exist_ok=True)
    for attempt in range(1,4):
        io.verify_sources(project)
        if _STOP:raise InterruptedError('Stop requested')
        if gpu:wait_gpu(gpu_uuid)
        if shutil.disk_usage(project).free<8*1024**3:raise RuntimeError('Less than 8 GiB disk headroom; no child started')
        logfile=logdir/f'{label}_{time.strftime("%Y%m%dT%H%M%SZ",time.gmtime())}_{attempt}.log'
        env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['CUDA_VISIBLE_DEVICES']='0' if gpu else ''
        cmd=list(command)
        if attempt>1 and '--resume' not in cmd:cmd.append('--resume')
        say('[start]',label,'attempt',attempt,'log',logfile)
        (Path(project)/'logs/cms1_current_child_log.txt').write_text(str(logfile)+'\n')
        with logfile.open('xb',buffering=0) as f:
            _ACTIVE=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                     env=env,cwd=project)
            io.atomic_json(status_path,dict(state='running',label=label,pid=os.getpid(),child_pid=_ACTIVE.pid,
                child_log=str(logfile),gpu_uuid=gpu_uuid,time=io.now(),attempt=attempt))
            code=_ACTIVE.wait()
        _ACTIVE=None
        if code==0:say('[done]',label);return
        if code==76 or _STOP:raise InterruptedError('Stopped after a committed checkpoint')
        if code==75 and attempt<3:
            say('[OOM] preserving attempt, retrying SAME recipe from last checkpoint');time.sleep(45);continue
        raise RuntimeError(f'Child failed ({code}): {logfile}; no next scientific job launched')


def register(project,info):
    root=p.root(project);root.mkdir(parents=True,exist_ok=True)
    registration=root/'registration.json';sha=io.verify_sources(project)
    expected=dict(protocol_sha=p.PROTOCOL_SHA,source_sha=sha,seeds=list(p.SEEDS),
                  backend_sha=p.digest(io.read_json(p.HERE/'FROZEN_BACKEND.json')),
                  scientific_updates=p.SPEC['actual_scientific_optimizer_updates'])
    if registration.exists():
        if io.read_json(registration)['fixed']!=expected:raise RuntimeError('Frozen registration changed')
    else:
        if shutil.disk_usage(project).free<64*1024**3:
            raise RuntimeError('Need 64 GiB free disk for retained parents/final checkpoints, gate evidence and headroom')
        audit=seed_audit(project)
        io.atomic_json(root/'PROTOCOL.json',p.SPEC)
        io.atomic_json(registration,dict(fixed=expected,seed_audit=audit,registered_at=io.now(),initial_gpu=info))
    data=io.verify_data(project);io.atomic_json(root/'cache_identity.json',data)
    io.append_json(root/'execution_sessions.jsonl',dict(time=io.now(),gpu=info,hostname=socket.gethostname()))
    return root


def make_job(project,group,name,values):
    root=p.root(project);path=root/'jobs'/group/(name+'.json')
    job=dict(values,source_sha=io.verify_sources(project),protocol_sha=p.PROTOCOL_SHA)
    if path.exists() and io.read_json(path)!=job:raise RuntimeError('Registered job changed: '+str(path))
    if not path.exists():io.atomic_json(path,job)
    return path


def launch_job(project,group,name,values,status,gpu_uuid,resume):
    job=make_job(project,group,name,values)
    cmd=[sys.executable,'-u',str(p.HERE/'study.py'),'_worker','--project',str(project),'--job',str(job)]
    if resume:cmd+=['--resume']
    child(project,cmd,group+'_'+name,status,gpu_uuid,resume=resume)


def run_gate(project,gpu_uuid,status,resume):
    from cms_analysis import compare_jobs
    gate=p.root(project)/'gates'/gpu_uuid
    resultfile=gate/'gate.json'
    if resultfile.exists():
        result=io.read_json(resultfile)
        if result['state']!='passed' or result['source_sha']!=io.verify_sources(project):
            raise RuntimeError('Existing gate does not match current source')
        # Recheck all retained proof records, not just a PASS string.
        for rel,expected in result['evidence'].items():
            if io.sha(gate/rel)!=expected:raise RuntimeError('Gate evidence changed: '+rel)
        say('[gate already passed]',gpu_uuid);return result
    group='gate_'+gpu_uuid
    for name,mode in [('parent','adamw'),('reference_parent','reference_adamw')]:
        launch_job(project,group,name,dict(action='train',directory=str(gate/name),seed=127,
            mode=mode,end=200,parent=None,gate=True),status,gpu_uuid,resume)
    checks={'parent_vs_frozen_adam':compare_jobs(gate/'parent',gate/'reference_parent',reference=True)}
    io.atomic_json(gate/'progress_checks.json',checks)
    for arm in p.ARMS:
        for repeat in range(2):
            name=f'{arm}_{repeat}';directory=gate/name
            values=dict(action='train',directory=str(directory),seed=127,mode=arm,end=420,
                        parent=str(gate/'parent'),gate=True)
            if arm=='scrambled_matched' and repeat==1:
                # A different process resumes halfway through active tempering.
                first=dict(values,interrupt_at=310)
                if not (directory/'latest.json').exists() and not (directory/'result.json').exists():
                    launch_job(project,group,name+'_interrupt',first,status,gpu_uuid,resume)
                launch_job(project,group,name+'_resume',values,status,gpu_uuid,True)
            else:launch_job(project,group,name,values,status,gpu_uuid,resume)
        checks[arm+'_repeat_resume']=compare_jobs(gate/(arm+'_0'),gate/(arm+'_1'))
        io.atomic_json(gate/'progress_checks.json',checks)
    for arm,mode in [('adamw','reference_adamw'),('bdt','reference_bdt')]:
        launch_job(project,group,mode,dict(action='train',directory=str(gate/mode),seed=127,
            mode=mode,end=420,parent=str(gate/'parent'),gate=True),status,gpu_uuid,resume)
        checks[arm+'_vs_frozen']=compare_jobs(gate/(arm+'_0'),gate/mode,reference=True)
    # The first active step must have the same pre-treatment gradient/raw proposal in all arms.
    first=[]
    for arm in p.ARMS:
        x=io.read_json(gate/(arm+'_0')/'result.json')
        row=next(r for r in x['comparison_rows'] if r['step']==201)
        first.append({k:row[k] for k in ('batch_hash','rng_before','rng_after_backward','gradient_sha','raw_proposal_sha')})
    if not all(v==first[0] for v in first):raise AssertionError('First active-step inputs differ across methods')
    checks['matched_first_active_gradient']=True
    evidence={str(q.relative_to(gate)):io.sha(q) for q in sorted(gate.rglob('*'))
              if q.is_file() and q.name in ('result.json','samples.npz','initialization.json','contract.json')}
    report=dict(state='passed',time=io.now(),source_sha=io.verify_sources(project),gpu_uuid=gpu_uuid,
                checks=checks,evidence=evidence,max_optimizer_updates=2160,
                scope='One existing development seed; repeats and interrupted-resume through active update 420, not efficacy')
    io.atomic_json(resultfile,report);say('CMS1 FULL-SIZE ACTIVE-CONTROL GATE: PASS')
    return report


def coordinate(project,run=False,resume=False):
    global _STOP
    _STOP=False
    project=Path(project).resolve();check_account();io.verify_sources(project)
    root=p.root(project);root.mkdir(parents=True,exist_ok=True);status=root/'scheduler_status.json'
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    with contextlib.ExitStack() as stack:
        stack.enter_context(io.file_lock(project/'logs/.cms1_scheduler.lock'))
        info=gpu_info()
        if 'H200' not in info['name']:raise RuntimeError('Expected H200; do not silently change accelerator class')
        stack.enter_context(io.file_lock(project/'logs/.practical_gpu_locks'/(info['uuid']+'.lock')))
        live=live_training_processes()
        if live:raise RuntimeError('Another training/diagnostic process is present in this session: '+repr(live))
        try:
            register(project,info)
            cpu_out=root/'cpu_tests.json'
            if not cpu_out.exists():
                child(project,[sys.executable,'-u',str(p.HERE/'tests.py'),'--project',str(project),'--output',str(cpu_out)],
                      'cpu_gate',status,info['uuid'],gpu=False)
            elif not io.read_json(cpu_out).get('unit_passed'):raise RuntimeError('Stored CPU gate did not pass')
            run_gate(project,info['uuid'],status,resume)
            if not run:
                io.atomic_json(status,dict(state='validated_only',pid=os.getpid(),time=io.now(),gpu_uuid=info['uuid']))
                say('[validated only] no confirmation seeds trained');return
            for seed in p.SEEDS:
                sd=root/f'seed_{seed}';parent=sd/'parent'
                launch_job(project,'science',f'seed_{seed}_parent',dict(action='train',directory=str(parent),
                    seed=seed,mode='adamw',end=200,parent=None,gate=False),status,info['uuid'],resume)
                for arm in p.ORDER[seed]:
                    launch_job(project,'science',f'seed_{seed}_{arm}',dict(action='train',directory=str(sd/'arms'/arm),
                        seed=seed,mode=arm,end=3000,parent=str(parent),gate=False),status,info['uuid'],resume)
            from cms_analysis import freeze_cohort,analyze
            freeze_cohort(project);say('[all 18 frozen] beginning registered REPORT evaluations')
            for seed in p.SEEDS:
                for arm in p.ARMS:
                    launch_job(project,'report',f'seed_{seed}_{arm}',dict(action='report',
                        directory=str(root/f'seed_{seed}'/'arms'/arm)),status,info['uuid'],resume)
            analysis=analyze(project)
            io.atomic_json(status,dict(state='finished',pid=os.getpid(),gpu_uuid=info['uuid'],time=io.now(),
                                      analysis=str(root/'analysis.json')))
            say('CMS1 STUDY COMPLETE; no additional studies auto-launched')
        except BaseException as ex:
            state='interrupted' if isinstance(ex,(InterruptedError,KeyboardInterrupt)) else 'failed'
            io.atomic_json(status,dict(state=state,pid=os.getpid(),time=io.now(),error=str(ex),gpu_uuid=info['uuid']))
            raise
