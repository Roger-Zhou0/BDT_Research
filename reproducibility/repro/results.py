"""Offline arithmetic on recorded final results, not model evaluation."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
SCORE_DIGEST='af1eceb024cafd222dc355e93cb060c782691d8b19293c2a879ddb093adbd3ec'
SEEDS=('946201','946202','946203','946204')
DEFINITIONS={
 'agt':[('draw0_reassigned','draw0_preserved',.975),('draw0_preserved','adamw',.975)],
 'frt':[('fixed_true_double','adamw',.95)],
 'art':[('assembled','fixed',.975),('assembled','combined',.975),('difference','fixed',.95),
        ('attention','fixed',.95),('assembled','difference',.95),('assembled','attention',.95),('assembled','adamw',.95)],
 'bct':[('bdt_budget_gs','structured_budget_gs',.95),('bdt_budget_gs','combined',.95),
        ('structured_budget_gs','combined',.95),('bdt_budget_gs','assembled',.95),('structured_budget_gs','assembled',.95)]}


def t_df3(confidence):
    # Exact df=3 CDF inversion; same formula as the manuscript arithmetic script.
    if not 0<confidence<1:raise ValueError('Confidence must lie in (0,1)')
    lo,hi=0.,100.
    for _ in range(150):
        z=(lo+hi)/2;y=z/math.sqrt(3)
        cdf=.5+(math.atan(y)+y/(1+y*y))/math.pi
        if cdf<(1+confidence)/2:lo=z
        else:hi=z
    return (lo+hi)/2


def paired(values,confidence=.95):
    values=list(values)
    if len(values)<2 or not all(math.isfinite(v) for v in values):raise ValueError('Finite paired units required')
    if len(values)==4:critical=t_df3(confidence)
    else:
        from scipy.stats import t
        critical=float(t.ppf((1+confidence)/2,len(values)-1))
    mean=statistics.mean(values);se=statistics.stdev(values)/math.sqrt(len(values))
    return {'n':len(values),'df':len(values)-1,'confidence':confidence,'mean':mean,
            'lower':mean-critical*se,'upper':mean+critical*se,'se':se,
            'negative_signs':sum(x<0 for x in values),'per_parent':values}


def report_results():
    scores=json.loads((ROOT/'data/report_scores.json').read_text())
    d=hashlib.sha256(json.dumps(scores,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if d!=SCORE_DIGEST:raise RuntimeError('Final REPORT score payload changed')
    for stage in scores:
        if set(scores[stage])!=set(SEEDS):raise RuntimeError('Incomplete parent cohort')
    # Repeated linked scores refer to the SAME checkpoint cells, not independent replications.
    for s in SEEDS:
        if not scores['agt'][s]['adamw']==scores['art'][s]['adamw']==scores['frt'][s]['adamw']:raise RuntimeError('Linked AdamW reference differs')
        if scores['frt'][s]['fixed_true_double']!=scores['art'][s]['fixed']:raise RuntimeError('Linked fixed reference differs')
        for arm in ['assembled','combined']:
            if scores['bct'][s][arm]!=scores['art'][s][arm]:raise RuntimeError('Linked ART1 reference differs')
    out={}
    for stage,rows in DEFINITIONS.items():
        out[stage]={}
        for a,b,level in rows:
            values=[scores[stage][s][a]-scores[stage][s][b] for s in SEEDS]
            rec=paired(values,level);rec['per_parent']=dict(zip(SEEDS,values));out[stage][f'{a} - {b}']=rec
    primary=out['bct']['bdt_budget_gs - structured_budget_gs']
    art=[out['art'][key] for key in ['assembled - fixed','assembled - combined']]
    return {'score_payload_sha256':d,'parent_units':4,'fresh_parent_confirmation':False,
            'logical_report_cells':56,'unique_checkpoint_cells':40,'contrasts':out,
            'BCT1_primary_upper_below_zero':primary['upper']<0,
            'ART1_both_primary_uppers_below_zero':all(r['upper']<0 for r in art),
            'scope':'Repeated parents and linked references are retained; no pooling of studies or charts as independent samples.',
            'model_evaluations_performed':0}


def allocation_results():
    with (ROOT/'data/allocation_diagnostics.csv').open() as handle:
        rows=list(csv.DictReader(handle))
    expected={(s,b,p,q) for s in SEEDS for b in ('bdt_budget_gs','structured_budget_gs')
              for p in (201,210,250,300,310,400,420,1000,2000,3000) for q in ('gate','up','down')}
    keys={(r['seed'],r['branch_state'],int(r['phase']),r['projection']) for r in rows}
    if len(rows)!=240 or keys!=expected:raise RuntimeError('Retained-observation grid differs')
    early=[r for r in rows if r['phase']=='400'];late=[r for r in rows if r['phase']=='3000']
    index={(r['seed'],r['branch_state'],r['projection'],r['phase']):r for r in rows}
    rising=all(float(index[s,b,q,'3000']['qD_full'])>float(index[s,b,q,'400']['qD_full'])
               for s in SEEDS for b in ('bdt_budget_gs','structured_budget_gs') for q in ('gate','up','down'))
    return {'observations':240,'parent_units':4,'prospective':False,'additional_independent_replications':0,
            'full_minus_structured_positive_at_400':sum(float(r['qD_full'])>float(r['qD_structured']) for r in early),
            'full_minus_structured_negative_at_3000':sum(float(r['qD_full'])<float(r['qD_structured']) for r in late),
            'comparisons_each_phase':len(early),'full_qD_itself_increases_400_to_3000_every_cell':rising,
            'fixed_coefficient_or_schedule_ruled_out':False,
            'precision':'qD inputs retained at nine decimal display precision; derived raw fractions/effective rho are not fresh tensor observations.'}


def inherited_results():
    d=json.loads((ROOT/'data/display_tables.json').read_text())['tables'];out={}
    for table in ('cms1_per_seed','gct2_all_endpoints','fev1_all_endpoints'):
        rows=d[table]
        out[table]={}
        for col,label in [(1,'BDT - AdamW'),(3,'BDT - matched')]:
            out[table][label]=paired((float(r[2])-float(r[col]) for r in rows),.975)
    rows=d['bac1_all_endpoints']
    out['bac1_all_endpoints']={
        'preserved - AdamW':paired((float(r[3])-float(r[1]) for r in rows),.975),
        'reassigned - preserved':paired((float(r[4])-float(r[3]) for r in rows),.975)}
    return {'input_precision':'Retained manuscript-displayed values, not newly recovered unrounded endpoint JSONs.',
            'scope':'Rounding can change final printed digits. Document evaluation reuses the GCT2 models.',
            'contrasts':out,'stored_table_count':len(d)}


def write_results(out:Path,include_inherited=False):
    out=out.resolve()
    if out.exists():raise FileExistsError('Choose a new analysis output directory')
    if out.is_relative_to(ROOT/'data') or out.is_relative_to(ROOT/'native'):raise ValueError('Read-only inputs')
    out.mkdir(parents=True)
    result=report_results();result['allocation_diagnostics']=allocation_results()
    if include_inherited:result['inherited_display_arithmetic']=inherited_results()
    (out/'recomputed_results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Recomputed REPORT contrasts','',
           'Loss is nats/token. Four reused parent units; lower favors the first policy. No new model evaluation.','',
           '| Study | Contrast | Mean | Interval level | Lower | Upper | Negative signs |',
           '|---|---|---:|---:|---:|---:|---:|']
    for stage,contrasts in result['contrasts'].items():
        for key,r in contrasts.items():lines.append(f'| {stage.upper()}1 | {key} | {r["mean"]:+.9f} | {100*r["confidence"]:g}% | {r["lower"]:+.9f} | {r["upper"]:+.9f} | {r["negative_signs"]}/4 |')
    (out/'REPORT_TABLE.md').write_text('\n'.join(lines)+'\n')
    p=result['contrasts']['bct'];rows=[]
    for key in ('bdt_budget_gs - structured_budget_gs','bdt_budget_gs - combined'):
        r=p[key];rows.append(f'{key} & {r["mean"]:+.6f} & [{r["lower"]:+.6f}, {r["upper"]:+.6f}] '+r'\\')
    (out/'bct1_table_rows.tex').write_text('\n'.join(rows)+'\n')
    return result
