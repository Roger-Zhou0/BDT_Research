"""No-GPU cohort validation, paired statistics, and engineering comparisons."""
from __future__ import annotations
import csv
import itertools
import math
from pathlib import Path
import statistics
import cms_io as io
import cms_protocol as p

T = {2:(12.706204736174694,25.451699579357168),
     3:(4.302652729749462,6.205346816570706),
     4:(3.1824463052837078,4.176534846104503),
     5:(2.7764451051977934,3.4954059325164417),
     6:(2.5705818356363146,3.163381449748609)}


def paired(values):
    if not values or not all(math.isfinite(x) for x in values): raise ValueError('Finite paired values required')
    n=len(values); mean=statistics.mean(values)
    out=dict(n=n,differences=values,mean=mean,favorable=sum(x<0 for x in values))
    if n>=2:
        if n not in T: raise ValueError('Unregistered replication count')
        sd=statistics.stdev(values);se=sd/math.sqrt(n)
        out.update(sd=sd,ci95=[mean-T[n][0]*se,mean+T[n][0]*se],
                   ci975_bonferroni_two=[mean-T[n][1]*se,mean+T[n][1]*se])
    return out


def compare_jobs(a,b,*,reference=False):
    import numpy as np
    a,b=Path(a),Path(b);x=io.read_json(a/'result.json');y=io.read_json(b/'result.json')
    if x['state']!='complete' or y['state']!='complete' or x['step']!=y['step']:
        raise RuntimeError('Cannot compare incomplete engineering jobs')
    fields=('model','adam','rng') if reference else ('model','adam','rng','canonical','scrambled','permutations')
    for field in fields:
        if x['initialization']['identity'][field]!=y['initialization']['identity'][field]:
            raise AssertionError('Engineering initial mismatch '+field)
        if x['identity_final'][field]!=y['identity_final'][field]:
            raise AssertionError('Engineering final mismatch '+field)
    if x['comparison_rows']!=y['comparison_rows']:
        for xx,yy in zip(x['comparison_rows'],y['comparison_rows']):
            if xx!=yy:
                bad={k:(xx.get(k),yy.get(k)) for k in set(xx)|set(yy) if xx.get(k)!=yy.get(k)}
                raise AssertionError(f'Engineering trace mismatch at step {xx["step"]}: {bad}')
        raise AssertionError('Engineering trace lengths differ')
    arrays=0
    with np.load(a/'samples.npz') as aa,np.load(b/'samples.npz') as bb:
        if set(aa.files)!=set(bb.files):raise AssertionError('Sample keys differ')
        for key in aa.files:
            if not np.array_equal(aa[key],bb[key]):raise AssertionError('Sample array differs '+key)
            arrays+=1
    return dict(exact=True,steps=len(x['comparison_rows']),arrays=arrays,
                first_result_sha=io.sha(a/'result.json'),second_result_sha=io.sha(b/'result.json'))


def trial_dir(project,seed,arm):
    return p.root(project)/f'seed_{seed}'/'arms'/arm


def validate_cohort(project):
    """Never silently exclude a failed/missing seed or replace its endpoint."""
    results={};runtime=None;source=None
    for seed in p.SEEDS:
        parent=p.root(project)/f'seed_{seed}'/'parent'
        pr=io.read_json(parent/'result.json')
        if pr['state']!='complete' or pr['step']!=p.PREFIX:raise RuntimeError('Incomplete parent')
        parent_sha=io.read_json(parent/'latest.json')['sha256']
        prior=None
        for arm in p.ARMS:
            d=trial_dir(project,seed,arm);x=io.read_json(d/'result.json');ct=x['contract']
            if x['state']!='complete' or x['step']!=p.END or x['report'] is not None:
                raise RuntimeError('Incomplete or prematurely reported training record')
            if (ct['seed'],ct['method'],ct['protocol_sha'],ct['start_step'],ct['end_step'])!=(seed,arm,p.PROTOCOL_SHA,200,3000):
                raise RuntimeError('Trial registration mismatch')
            if ct['engineering'] or ct['tiny'] or ct['data_hashes']!=p.DATA_HASHES:
                raise RuntimeError('Wrong scientific provenance')
            if x['contract_sha']!=p.digest(ct) or io.read_json(d/'contract.json')!=ct:
                raise RuntimeError('Trial contract hash mismatch')
            if ct['parent_checkpoint_sha']!=parent_sha or x['initialization']['parent_sha']!=parent_sha:
                raise RuntimeError('Parent file mismatch')
            if x['initialization']['identity']!=pr['identity_final']:
                raise RuntimeError('Fork did not match immutable parent state')
            if runtime is not None and ct['runtime']!=runtime:raise RuntimeError('Runtime changed across cohort')
            if source is not None and ct['source_sha']!=source:raise RuntimeError('Sources changed across cohort')
            runtime,source=ct['runtime'],ct['source_sha']
            if len(x['history'])!=2800 or [r['step'] for r in x['history']]!=list(range(201,3001)):
                raise RuntimeError('Wrong recorded update history')
            first=next(r for r in x['curve'] if r['step']==201)
            before={k:first[k] for k in ('batch_hash','rng_before','rng_after_backward','gradient_sha','raw_proposal_sha')}
            if prior is not None:
                if prior['history']!=x['history'] or prior['first']!=before:
                    raise RuntimeError('Data/RNG/first pre-treatment gradient differs across arms')
            prior=dict(history=x['history'],first=before)
            if io.read_json(d/'latest.json')!=x['checkpoint']:raise RuntimeError('Final checkpoint pointer changed')
            if not math.isfinite(x['dev']['loss']):raise RuntimeError('Nonfinite final DEV')
            results[f'{seed}/{arm}']=io.sha(d/'result.json')
    if source!=io.verify_sources(project):raise RuntimeError('Current source differs from cohort')
    return dict(protocol_sha=p.PROTOCOL_SHA,results=results,runtime=runtime,source_sha=source,
                scope='All 18 training endpoints frozen before the first REPORT evaluation')


def freeze_cohort(project):
    cohort=validate_cohort(project);path=p.root(project)/'cohort.lock.json'
    if path.exists() and io.read_json(path)!=cohort:raise RuntimeError('Frozen cohort changed')
    if not path.exists():io.atomic_json(path,cohort)
    return cohort


def analyze(project):
    cohort=validate_cohort(project)
    if io.read_json(p.root(project)/'cohort.lock.json')!=cohort:raise RuntimeError('Cohort lock mismatch')
    rows=[];table={}
    for seed in p.SEEDS:
        table[seed]={}
        for arm in p.ARMS:
            d=trial_dir(project,seed,arm);rr=io.read_json(d/'report.json')
            if rr['result_sha']!=cohort['results'][f'{seed}/{arm}'] or rr['protocol_sha']!=p.PROTOCOL_SHA:
                raise RuntimeError('Report provenance mismatch')
            v=rr['report']
            if (v['partition_start'],v['partition_stop'],v['tokens'])!=(2500000,3550624,1048576):
                raise RuntimeError('Wrong reporting partition')
            table[seed][arm]=float(v['loss'])
            rows.append(dict(seed=seed,method=arm,loss=v['loss'],source=str(d/'report.json'),sha256=io.sha(d/'report.json')))
    primary=paired([table[s]['bdt']-table[s]['scrambled_matched'] for s in p.SEEDS])
    constructive=paired([table[s]['bdt']-table[s]['adamw'] for s in p.SEEDS])
    out=dict(state='complete',protocol_sha=p.PROTOCOL_SHA,rows=rows,
             primary_bdt_minus_scrambled=primary,constructive_bdt_minus_adam=constructive,
             both_simultaneous_upper_below_zero=all(v['ci975_bonferroni_two'][1]<0 for v in (primary,constructive)),
             scope='Six training seeds. Same S1-B REPORT partition as earlier work. New strict-execution cohort; no equivalence claim.')
    root=p.root(project);io.atomic_json(root/'analysis.json',out)
    with (root/'endpoints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    text=['# Matched-strength S1: final six-seed readout','',
          '| Seed | AdamW | BDT | Matched scrambled |','|---|---:|---:|---:|']
    text += [f'| {s} | {table[s]["adamw"]:.9f} | {table[s]["bdt"]:.9f} | {table[s]["scrambled_matched"]:.9f} |' for s in p.SEEDS]
    for label,v in [('Primary: BDT minus matched scrambled',primary),('Constructive: BDT minus AdamW',constructive)]:
        text += ['',f'## {label}',f'Mean {v["mean"]:+.9f}; 95% paired CI {v["ci95"]}; '
                 f'simultaneous-family 97.5% CI {v["ci975_bonferroni_two"]}; favorable signs {v["favorable"]}/6.']
    text += ['',out['scope'],'','No statistical equivalence was preregistered; an interval crossing zero is unresolved.']
    (root/'READOUT.md').write_text('\n'.join(text)+'\n')
    return out
