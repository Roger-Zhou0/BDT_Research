#!/usr/bin/env python3
"""Entrypoint: integrity, launch, status, export, and isolated internal jobs."""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
import cms_protocol as p
import cms_io as io


def show_status(project):
    root=p.root(project)
    status=io.read_json(root/'scheduler_status.json') if (root/'scheduler_status.json').exists() else {'state':'not_started'}
    parents=[];arms=[];reports=[]
    for seed in p.SEEDS:
        sd=root/f'seed_{seed}'
        pr=sd/'parent/result.json'
        if pr.exists() and io.read_json(pr).get('state')=='complete':parents.append(seed)
        for arm in p.ARMS:
            d=sd/'arms'/arm
            if (d/'result.json').exists():
                x=io.read_json(d/'result.json')
                if x.get('state')=='complete':arms.append((seed,arm,x['dev']['loss']))
            if (d/'report.json').exists():reports.append((seed,arm))
    print('CMS1 STATUS:',status.get('state'))
    print(f'Parents: {len(parents)}/6 | Continuations: {len(arms)}/18 | REPORT evaluations: {len(reports)}/18')
    if status.get('state')=='running':
        print('Active:',status.get('label'),'child PID:',status.get('child_pid'))
        proc=Path('/proc')/str(status.get('pid',''))/'cmdline'
        if Path('/proc').exists() and not proc.exists():
            print('WARNING: recorded coordinator PID is no longer present; status may be stale.')
    if status.get('error'):print('ERROR:',status['error'])
    print('State file:',root/'scheduler_status.json')
    if status.get('child_log'):
        f=Path(status['child_log'])
        if f.exists():
            with f.open('rb') as s:s.seek(max(0,f.stat().st_size-12000));tail=s.read().decode(errors='replace')
            print('\nLATEST CHILD LOG:',f)
            print('\n'.join(tail.splitlines()[-10:]))
    if (root/'analysis.json').exists():
        a=io.read_json(root/'analysis.json')
        print('\nFINAL PRIMARY:',json.dumps(a['primary_bdt_minus_scrambled']))
        print('FINAL CONSTRUCTIVE:',json.dumps(a['constructive_bdt_minus_adam']))
    return status


def export(project):
    import hashlib
    project=Path(project).resolve();root=p.root(project)
    if not root.exists():raise RuntimeError('No study outputs yet')
    output=project/'exports';output.mkdir(exist_ok=True)
    filename=output/(p.VERSION+'_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:6]+'.zip')
    manifest={};warnings=[]
    sources=[(root,'run'),(p.HERE,'source')]
    include={'.json','.jsonl','.md','.csv','.txt','.log','.py','.sh','.npz'}
    with zipfile.ZipFile(filename,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for base,label in sources:
            for directory,subdirs,names in os.walk(base,followlinks=False):
                here=Path(directory)
                subdirs[:]=[n for n in subdirs if not n.startswith('.') and n not in ('__pycache__','checkpoints') and not (here/n).is_symlink()]
                for name in names:
                    f=here/name
                    if f.is_symlink() or f.suffix not in include or name.startswith('.'):continue
                    f.resolve().relative_to(base.resolve())
                    if f.stat().st_size>32<<20:
                        warnings.append('Skipped oversized file: '+str(f));continue
                    data=f.read_bytes()
                    if f.suffix in ('.jsonl','.log'):
                        data=b'\n'.join(data.splitlines()[-250:])+b'\n'
                    arc=label+'/'+str(f.relative_to(base))
                    z.writestr(arc,data);manifest[arc]=hashlib.sha256(data).hexdigest()
        # Include actual frozen backend code/protocol, never token arrays/checkpoints.
        for rel,want in io.read_json(p.HERE/'FROZEN_BACKEND.json').items():
            f=project/'fair_depth_lr_bracketing_v2'/rel
            if io.sha(f)!=want:raise RuntimeError('Backend source changed during export')
            data=f.read_bytes();arc='frozen_backend/'+rel
            z.writestr(arc,data);manifest[arc]=want
        z.writestr('EXPORT_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
        z.writestr('SNAPSHOT_NOTE.json',json.dumps(dict(time=io.now(),warnings=warnings,
            checkpoint_bytes_included=False,token_arrays_included=False,
            note='Per-file snapshot while jobs may be running; bounded 250-line log/trace tails, not a global atomic training snapshot.'),indent=2)+'\n')
    print('EXPORT_ZIP='+str(filename));print('EXPORT_SHA256='+io.sha(filename))
    return filename


def launch(project,run,resume):
    project=Path(project).resolve()
    from cms_scheduler import check_account
    check_account();io.verify_sources(project)
    logs=project/'logs';logs.mkdir(exist_ok=True)
    with io.file_lock(logs/'.cms1_launcher.lock'):
        # Never starts a second scientific writer if the existing coordinator owns its lock.
        with io.file_lock(logs/'.cms1_scheduler.lock'):pass
        log=logs/('cms1_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'_'+uuid.uuid4().hex[:6]+'.log')
        cmd=[sys.executable,'-u',str(p.HERE/'study.py'),'_coordinate','--project',str(project)]
        cmd += ['--run' if run else '--validate-only']
        if resume:cmd+=['--resume']
        env=os.environ.copy();env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['CUDA_VISIBLE_DEVICES']='0'
        with log.open('xb',buffering=0) as f:
            proc=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                  cwd=project,env=env,start_new_session=True)
        (logs/'cms1_latest_log.txt').write_text(str(log)+'\n')
        time.sleep(.6)
        if proc.poll() not in (None,0):
            print(log.read_text(),file=sys.stderr);raise RuntimeError('Coordinator stopped during startup')
        print('COORDINATOR_PID='+str(proc.pid));print('LOG='+str(log))
        print('Mode: '+('validate, then SIX parents and EIGHTEEN continuations' if run else 'validation only'))
        print('Training is established only after the full-size active-control gate passes and scientific logs advance.')


def worker(project,jobpath,resume):
    project=Path(project).resolve();jobpath=Path(jobpath).resolve();jobpath.relative_to(p.root(project))
    job=io.read_json(jobpath)
    if job['protocol_sha']!=p.PROTOCOL_SHA or job['source_sha']!=io.verify_sources(project):
        raise RuntimeError('Job source/protocol changed')
    # Set the workspace before any import that could initialize CUDA.
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Missing pre-import cuBLAS workspace')
    import cms_runtime as rt
    torch=rt.configure(4)
    from cms_train import train_job,report_job
    try:
        if job['action']=='report':report_job(project,Path(job['directory']));return 0
        r=train_job(project,Path(job['directory']),job['seed'],job['mode'],job['end'],
                    parent=Path(job['parent']) if job.get('parent') else None,
                    gate=job['gate'],resume=resume,interrupt_at=job.get('interrupt_at'))
        return 76 if r['state']=='interrupted' else 0
    except torch.OutOfMemoryError:
        traceback.print_exc();return 75


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['integrity','launch','status','export','analyze','_coordinate','_worker'])
    ap.add_argument('--project',default=str(p.HERE.parent));ap.add_argument('--resume',action='store_true')
    modes=ap.add_mutually_exclusive_group();modes.add_argument('--run',action='store_true');modes.add_argument('--validate-only',action='store_true')
    ap.add_argument('--job');args=ap.parse_args();project=Path(args.project).resolve()
    if args.action=='integrity':print('CMS1 SOURCE + FROZEN BACKEND INTEGRITY: PASS',io.verify_sources(project));return 0
    if args.action=='launch':
        if not (args.run or args.validate_only):ap.error('Choose --run or --validate-only')
        launch(project,args.run,args.resume);return 0
    if args.action=='_coordinate':
        from cms_scheduler import coordinate
        coordinate(project,args.run,args.resume);return 0
    if args.action=='_worker':
        if not args.job:ap.error('--job required')
        return worker(project,args.job,args.resume)
    if args.action=='status':show_status(project);return 0
    if args.action=='export':export(project);return 0
    if args.action=='analyze':
        from cms_analysis import analyze
        print(json.dumps(analyze(project),indent=2));return 0

if __name__=='__main__':
    try:raise SystemExit(main())
    except Exception as ex:
        print('STOPPED:',type(ex).__name__,str(ex),file=sys.stderr);traceback.print_exc();raise SystemExit(1)
