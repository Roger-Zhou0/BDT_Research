#!/usr/bin/env python3
"""CPU-only analysis. Never selects using reporting outcomes."""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
import protocol as p
HERE=Path(__file__).resolve().parent
TC=json.loads((HERE/'T_CRITICAL.json').read_text())
def contrast(values):
    n=len(values)
    if n==0:return dict(n=0)
    mean=statistics.mean(values);sd=statistics.stdev(values) if n>1 else None
    intervals={}
    for level in ('95','97.5'):
        half=TC[str(n-1)][level]*sd/math.sqrt(n) if n>1 else None
        intervals[level]=[mean-half,mean+half] if half is not None else None
    return dict(n=n,mean=mean,sd=sd,ci95=intervals['95'],ci97p5=intervals['97.5'],wins=sum(x<0 for x in values),per_seed=values,
                perplexity_ratio=math.exp(mean))
def main():
    a=argparse.ArgumentParser();a.add_argument('--project',default='.');x=a.parse_args();root=p.campaign(x.project)
    print('E15/E16 — DECLARED LR-BRACKETING AMENDMENT V2')
    print('Fixed recipes, equal new LR-search budgets; final metrics on disjoint within-campaign REPORT partition.')
    lock=root/'selection.lock.json'
    if not lock.exists():
        print('No locked recipes. 42 v1 results inherited; three new LRs per method/seed. Reporting remains unopened.')
        for seed in p.DEV_SEEDS:
            for method in p.METHODS:
                done=sum((p.trial_dir(x.project,'development',seed,method,lr)/'result.json').is_file() for lr in p.INITIAL_LRS+p.EDGE_LRS)
                failed=sum((p.trial_dir(x.project,'development',seed,method,lr)/'diverged.json').is_file() for lr in p.INITIAL_LRS+p.EDGE_LRS)
                print(seed,method,'NEW completed',done,'numerically divergent',failed)
        un=root/'selection.unbracketed.json'
        if un.exists():print('UNBRACKETED',un.read_text())
        return
    locked=json.loads(lock.read_text())
    if locked['protocol_sha']!=p.PROTOCOL_SHA:raise RuntimeError('Protocol mismatch')
    print('LOCKED LEARNING RATES',locked['chosen_lr'])
    print('BUDGET PER METHOD',locked['budget_per_method'])
    complete={};missing=[];control_errors=[]
    for seed in p.REPORT_SEEDS:
        rr={}
        for method in p.METHODS:
            path=p.trial_dir(x.project,'report',seed,method,locked['chosen_lr'][method])/'result.json'
            if not path.exists():missing.append(str(path));continue
            r=json.loads(path.read_text());ct=r['contract']
            if r['step']!=p.STEPS or r['state']!='complete' or r.get('report') is None:
                control_errors.append(f'Invalid reporting endpoint {path}');continue
            if ct['protocol_sha']!=p.PROTOCOL_SHA or ct['stage']!='report' or ct['lr']!=locked['chosen_lr'][method] or ct['method']!=method or ct['seed']!=seed:
                control_errors.append(f'Invalid contract {path}');continue
            if r['contract_sha256']!=p.digest(ct):control_errors.append(f'Contract digest invalid {path}')
            rr[method]=r
        if len(rr)==3:
            if len({p.digest(r['initialization']) for r in rr.values()})!=1:control_errors.append(f'Initialization mismatch seed {seed}')
            if len({p.digest(r['contract']['data_hashes']) for r in rr.values()})!=1:control_errors.append(f'Data mismatch seed {seed}')
            if len({p.digest(r['contract']['source_sha256']) for r in rr.values()})!=1:control_errors.append(f'Source mismatch seed {seed}')
            if len({p.digest(r['contract']['runtime']) for r in rr.values()})!=1:control_errors.append(f'Within-seed runtime mismatch {seed}')
            if any(r['report']['partition_start']!=p.REPORT_OFFSET or r['report']['tokens']!=p.EVAL_SEQUENCES*p.EVAL_CONTEXT for r in rr.values()):
                control_errors.append(f'Report partition mismatch seed {seed}')
            complete[seed]=rr
    print('AVAILABLE COMPLETE REPORTING SEEDS',list(complete));print('MISSING RESULTS',len(missing))
    for seed,rr in complete.items():
        vals={m:r['report']['loss'] for m,r in rr.items()}
        print('seed',seed,vals,'temper-adam',vals['temper_cap3']-vals['adamw'],'temper-GS',vals['temper_cap3']-vals['gs_window'])
    stats={}
    for name,a,b in [('temper_minus_adam','temper_cap3','adamw'),('temper_minus_gs','temper_cap3','gs_window'),('gs_minus_adam','gs_window','adamw')]:
        v=[r[a]['report']['loss']-r[b]['report']['loss'] for r in complete.values()];s=contrast(v);stats[name]=s
        print(name,json.dumps(s))
    print('PRIMARY CHECKS: 97.5% simultaneous two-sided intervals for the TWO primary comparisons')
    gates={}
    for name in ('temper_minus_adam','temper_minus_gs'):
        s=stats[name];gates[name]=bool(len(complete)==6 and not control_errors and s.get('mean',1)<0 and s.get('wins',0)>=5 and s.get('ci97p5') and s['ci97p5'][1]<0)
        print(name,'SUPPORTED=',gates[name])
    print('CONTROL ERRORS',control_errors)
    out=dict(protocol=p.SPEC,selection=locked,available=list(complete),missing=missing,control_errors=control_errors,statistics=stats,gates=gates,
             cautions=['Reporting cache was used in earlier research; this is not a new historically unseen corpus.',
                       'Fixed GS Standard-Proj alpha=.1 with LR tuning, not exhaustive GS family optimization.',
                       'No ordinary-Dense, mature-budget, or constant-GPU-time conclusion follows.'])
    p.atomic_json(Path(x.project)/'fair_bracketing_analysis.json',out)
    print('WROTE',Path(x.project)/'fair_bracketing_analysis.json')
if __name__=='__main__':main()
