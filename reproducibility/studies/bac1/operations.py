"""Fresh registration, dependency checks, and outcome-independent admissions."""
from __future__ import annotations
import re, json, time, os, shutil, subprocess, sys
from pathlib import Path
import common as io


from parallel_admission import prerequisites



def register(project):
    root=io.root(project);project=Path(project)
    with io.lock(project/'logs/.research_followup_registration.lock',blocking=True):
        records=[]
        if io.SPEC['kind']=='ancestry_confirmation':
            seeds=set(io.SPEC['seeds'])
            # Candidate numerical labels are not called reserved until this succeeds.
            # Inspect recorded study protocols and top-level seed-bearing run names.
            conflicts=[]
            for d in (project/'runs').iterdir():
                if d.resolve()==root.resolve() or not d.is_dir():
                    continue
                named={int(x) for x in re.findall(r'seed[_-]?(\d+)',d.name)}
                if seeds&named:conflicts.append(str(d))
                p=d/'PROTOCOL.json'
                if p.exists():
                    z=io.read(p)
                    declared=set(z.get('seeds',[]))|set(z.get('fresh_seeds',[]))|set(z.get('development_seeds',[]))
                    if seeds&declared:conflicts.append(str(p))
                for s in seeds:
                    if (d/f'seed_{s}').exists() or (d/'replication'/f'seed_{s}').exists():
                        conflicts.append(str(d))
            if conflicts:
                raise RuntimeError('Fresh seed labels conflict; do not auto-renumber: '+str(sorted(set(conflicts))))
        else:
            for seed in io.SPEC['seeds']:
                d=project/'runs'/io.SPEC['source_parent_study']/f'seed_{seed}'
                for stage in ('parent','grown'):
                    f=d/stage/'result.json';v=io.read(f)
                    if v['state']!='complete' or v['contract']['seed']!=seed:
                        raise RuntimeError('Missing window-pilot development parent')
                    for key in ('source','protocol'):
                        if v['contract'][key]!=io.SPEC['source_parent_'+key+'_sha256']:
                            raise RuntimeError('Development-parent identity mismatch')
                    records.append(dict(seed=seed,stage=stage,result_sha256=io.sha(f),checkpoint_sha256=v['checkpoint']['sha256']))
        value=dict(source=io.verify()[0],protocol=io.digest(io.SPEC),seeds=io.SPEC['seeds'],parents=records,
                   interpretation=io.SPEC['inference'])
        io.immutable(root/'registration.json',value)
        return value


def coordinate(project,resume,science=False):
    """Default is engineering only. Scientific release is an explicit later flag."""
    import followup_queue as q
    import experiment
    import signal
    q.session();root=io.root(project);q.STOP=False
    def stop(*_):
        q.STOP=True
        if q.CHILD is not None and q.CHILD.poll() is None:q.CHILD.terminate()
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    with io.lock(Path(project)/f"logs/.{io.SPEC['short']}_{q.lane()}.lock"):
        io.immutable(root/'PROTOCOL.json',io.SPEC)
        io.immutable(root/'source_registration.json',dict(source=io.verify()[0],protocol=io.digest(io.SPEC)))
        try:
            start=time.monotonic()
            while True:
                if q.STOP:raise InterruptedError('Stopped while waiting')
                ready,why=prerequisites(project)
                io.atomic(q.lane_dir(project)/'status.json',dict(state='waiting_predecessor',reason=why,gpu_lock_held=False,pid=os.getpid(),time=io.now()))
                if ready and not q.other_work():break
                if time.monotonic()-start>io.SPEC['wait_hours']*3600:raise RuntimeError('Bounded admission wait expired')
                time.sleep(60)
            register(project)
            cp=q.lane_dir(project)/'cpu_validation.json'
            if not cp.exists():
                log=Path(project)/'logs'/f"{io.SPEC['short']}_cpu_{q.lane()}_{time.time_ns()}.log"
                with log.open('xb') as f:
                    proc=subprocess.run([sys.executable,str(io.HERE/'tests/test_suite.py')],stdout=f,stderr=subprocess.STDOUT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',FOLLOWUP_TEST_OUTPUT=str(q.lane_dir(project)/'cpu_integration.json')))
                if proc.returncode:raise RuntimeError('Target CPU tests failed: '+str(log))
                io.atomic(cp,dict(state='passed',source=io.verify()[0],log=str(log),log_sha256=io.sha(log),time=io.now()))
            elif io.read(cp).get('source')!=io.verify()[0] or io.read(cp).get('state')!='passed':
                raise RuntimeError('CPU validation/source mismatch')
            g=q.gpu();q.memory_wait(g['uuid'])
            with io.lock(Path(project)/'logs/.practical_gpu_locks'/(g['uuid']+'.lock')):
                if q.other_work():raise RuntimeError('Another local worker appeared; nothing interrupted')
                if shutil.disk_usage(project).free<io.SPEC['disk_free_gib']*1024**3:
                    raise RuntimeError('Insufficient registered disk headroom')
                rt=q.runtime_probe(project);io.immutable(root/'runtime_registration.json',rt)
                data=io.cache(project);dp=q.lane_dir(project)/'cache.json';io.atomic(dp,data)
                q.gate(project,rt,dp,g['uuid'],resume)
                # New schedules need full-sized proposal boundary tests in addition
                # to the 420-update actual-model reference/repeat/resume gate.
                if io.SPEC['kind']=='window_pilot':
                    log=Path(project)/'logs'/f"{io.SPEC['short']}_boundaries_{q.lane()}_{time.time_ns()}.log"
                    out=q.lane_dir(project)/'cuda_boundaries.json'
                    with log.open('xb') as f:
                        p=subprocess.run([sys.executable,str(io.HERE/'boundary_tests.py'),'--device','cuda','--output',str(out)],stdout=f,stderr=subprocess.STDOUT)
                    if p.returncode:raise RuntimeError('Target boundary tests failed: '+str(log))
                    if io.read(out)['source']!=io.verify()[0]:raise RuntimeError('Boundary source mismatch')
                if not science:
                    io.atomic(q.lane_dir(project)/'status.json',dict(state='engineering_complete_not_released',pid=os.getpid(),gpu_lock_held=False,science_started=False,time=io.now()))
                    return
                for j in experiment.science_jobs(project,io.SPEC['lane_seeds'][q.lane()]):
                    q.run_job(project,j,dp,rt,g['uuid'],str(Path(j['directory']).relative_to(root)),resume)
                io.atomic(q.lane_dir(project)/'status.json',dict(state='training_complete',pid=os.getpid(),time=io.now(),automatic_extension=False))
            paths=[root/f'seed_{s}'/'arms'/m/'result.json' for s in io.SPEC['seeds'] for m in io.SPEC['arms']]
            if io.SPEC['kind']=='ancestry_confirmation' and q.lane()==io.SPEC['evaluation_lane']:
                while not all(f.is_file() for f in paths):
                    if q.STOP:raise InterruptedError('Stopped at cohort barrier')
                    if time.monotonic()-start>io.SPEC['wait_hours']*3600:raise RuntimeError('Cohort wait expired')
                    io.atomic(q.lane_dir(project)/'status.json',dict(state='waiting_full_cohort',gpu_lock_held=False,pid=os.getpid(),time=io.now()))
                    time.sleep(60)
                with io.lock(root/'.finalizer.lock',blocking=True):
                    experiment.analyze(project)  # validates all 24 and seals before scoring
                    q.memory_wait(g['uuid'])
                    with io.lock(Path(project)/'logs/.practical_gpu_locks'/(g['uuid']+'.lock')):
                        if q.other_work():raise RuntimeError('Other work before REPORT')
                        for s in io.SPEC['seeds']:
                            for m in io.SPEC['arms']:
                                cmd=[sys.executable,str(io.HERE/'report_entry.py'),str(project),str(s),m]
                                log=Path(project)/'logs'/f"{io.SPEC['short']}_report_{s}_{m}_{time.time_ns()}.log"
                                with log.open('xb') as f:
                                    result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800)
                                if result.returncode:raise RuntimeError('Locked REPORT evaluator failed: '+str(log))
                        result=experiment.analyze(project)
                        io.atomic(root/'status.json',dict(state='finished',candidate=result['candidate'],time=io.now(),automatic_extension=False))
            elif io.SPEC['kind']=='window_pilot':
                with io.lock(root/'.finalizer.lock',blocking=True):
                    if all(f.is_file() for f in paths):
                        result=experiment.analyze(project)
                        io.atomic(root/'status.json',dict(state='finished',candidate=False,time=io.now(),automatic_extension=False))
        except BaseException as ex:
            io.atomic(q.lane_dir(project)/'status.json',dict(state='interrupted' if isinstance(ex,InterruptedError) else 'failed',error=str(ex),pid=os.getpid(),time=io.now()))
            raise
