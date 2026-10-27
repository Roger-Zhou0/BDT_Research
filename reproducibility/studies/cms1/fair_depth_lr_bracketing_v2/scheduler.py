#!/usr/bin/env python3
"""Paired workload split by seed, never by method/GPU. No framework import in waiter."""
from __future__ import annotations
import argparse,fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
import protocol as p
from select_recipes import choose
HERE=Path(__file__).resolve().parent

def say(*x):print(time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),*x,flush=True)

def wait_gpu(min_mib):
    stable=0
    while stable<2:
        r=subprocess.run(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits','-i','0'],text=True,capture_output=True)
        try:n=int(r.stdout.strip().splitlines()[0])
        except (ValueError,IndexError):n=-1
        if r.returncode==0 and n>=min_mib:stable+=1
        else:stable=0
        if stable<2:say('[GPU wait]',n,'MiB free; need',min_mib);time.sleep(15)
    say('[GPU ready]',n,'MiB')

def run_child(cmd,log,root,minimum):
    log.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(1,6):
        p.verify_sources();wait_gpu(minimum)
        # Per-run checkpoint + an atomic temporary save. Does not delete anything.
        if shutil.disk_usage(root).free<8*1024**3:raise RuntimeError('Less than 8 GiB disk headroom; stopped before launch')
        offset=log.stat().st_size if log.exists() else 0
        say('[start]', 'attempt',attempt, 'log='+str(log))
        with log.open('a') as f:
            f.write('\n[attempt] '+str(attempt)+' '+time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())+'\n');f.flush()
            ret=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT).returncode
        if ret==0:say('[done]',log);return
        with log.open('rb') as f:f.seek(offset);text=f.read().decode(errors='replace')
        if 'OutOfMemoryError' in text or 'CUDA out of memory' in text:
            say('[OOM retry: same recipe, latest checkpoint]',attempt);time.sleep(45);continue
        raise RuntimeError(f'Non-OOM child failure ({ret}); read {log}')
    raise RuntimeError(f'OOM retry limit reached; checkpoints retained: {log}')

def jobs(seed,lrs):
    # Cyclic/reversed ordering distributes method and LR order within each GPU.
    out=[]
    ordered=list(lrs) if seed%2 else list(reversed(lrs))
    for i,lr in enumerate(ordered):
        methods=list(p.METHODS);k=(seed+i)%len(methods);methods=methods[k:]+methods[:k]
        if seed%2==0:methods.reverse()
        out.extend((method,lr) for method in methods)
    return out

def train_job(project,stage,seed,method,lr,minimum):
    d=p.trial_dir(project,stage,seed,method,lr)
    if (d/'result.json').exists() or (d/'diverged.json').exists():say('[already recorded]',d);return
    run_child([sys.executable,'-u',str(HERE/'run.py'),'train','--project',str(project),
               '--stage',stage,'--seed',str(seed),'--method',method,'--lr',str(lr)],
               d/'train.log',project,minimum)

def main():
    a=argparse.ArgumentParser();a.add_argument('--project',required=True);a.add_argument('--side',choices=['A','B'],required=True)
    a.add_argument('--min-free-gib',type=float,default=22);x=a.parse_args()
    project=Path(x.project).resolve();root=p.campaign(project);root.mkdir(parents=True,exist_ok=True)
    from contextlib import ExitStack
    from parent_evidence import verify_parent
    with ExitStack() as stack:
        for path in (root/f'.side_{x.side}.lock',project/'runs'/p.PARENT_VERSION/f'.side_{x.side}.lock'):
            # Advisory lock files only; append mode also supports network-filesystem locking.
            # No contents are written, and original result/source/protocol files remain untouched.
            f=stack.enter_context(path.open('a'))
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Old or new worker already active for this side. Do not duplicate.')
        p.verify_sources();verify_parent(project);minimum=round(x.min_free_gib*1024)
        say('[AMENDMENT START]',x.side,'new LRs',p.NEW_LRS,'old 42 trials reused, not rerun')
        say('[disk free GiB]',round(shutil.disk_usage(project).free/1024**3,1),'Plan ~60 GiB ADDITIONAL shared storage for 36 new dev/report trials.')
        subprocess.run([sys.executable,'-u',str(HERE/'run.py'),'prepare','--project',str(project)],check=True,cwd=project)
        run_child([sys.executable,'-u',str(HERE/'tests.py'),'--cuda'],root/f'gpu_gate_{x.side}.log',project,minimum)
        seed=127 if x.side=='A' else 128
        for method,lr in jobs(seed,p.NEW_LRS):train_job(project,'development',seed,method,lr,minimum)
        while True:
            state=choose(project);say('[selection]',state['state'])
            if state['state']=='locked':break
            if state['state']=='unbracketed':
                raise RuntimeError('Amended ten-point search still has a boundary winner. STOP; reporting not opened. No automatic extra search.')
            time.sleep(45)
        say('[LOCKED RECIPES]',state['selection']['chosen_lr'],'report regardless of relative development winner')
        seeds=p.REPORT_SEEDS[0::2] if x.side=='A' else p.REPORT_SEEDS[1::2]
        for seed in seeds:
            methods=list(p.METHODS);k=seed%3;methods=methods[k:]+methods[:k]
            for method in methods:
                train_job(project,'report',seed,method,state['selection']['chosen_lr'][method],minimum)
        say('[SIDE COMPLETE]',x.side,'No further studies queued.')
if __name__=='__main__':
    try:main()
    except Exception as e:say('[STOPPED]',str(e));raise
