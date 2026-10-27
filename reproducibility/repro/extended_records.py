"""Offline checks of recovered GCT2/FEV1 and BAC1 evidence.

Original raw rows, verified projections and printed table fragments are distinct.
This module never evaluates a model or reads checkpoints or token caches.
"""
from pathlib import Path
import hashlib,json,math
from repro.results import paired
ROOT=Path(__file__).resolve().parents[1]

def gct2_fev1(root=None):
    root=ROOT/'reference/recovered_records/gct2' if root is None else Path(root)
    read=lambda n:json.loads((root/n).read_text())
    v=read('VERIFIED_GCT2_FEV1.json');lock=read('GCT2_LOCK.json')
    expected={(s,a) for s in range(161,167) for a in ('adamw','bdt','scrambled_matched')}
    if len(v['rows'])!=18 or {(r['seed'],r['arm']) for r in v['rows']}!=expected:raise RuntimeError('Incomplete/duplicated GCT2 derived cohort')
    if len(lock['endpoints'])!=18 or {(r['seed'],r['arm']) for r in lock['endpoints']}!=expected:raise RuntimeError('Incomplete/duplicated original cohort seal')
    if hashlib.sha256((root/'GCT2_LOCK.json').read_bytes()).hexdigest()!=v['cohort_lock_sha256']:raise RuntimeError('Original cohort bytes changed')
    source='08ecced9d27c4527661c1e8ee46c398c05bd0b573a9d227bf2db092ef66f2c7a'
    if lock['source']!=source or lock['state']!='sealed':raise RuntimeError('Original scientific source/seal mismatch')
    by={(r['seed'],r['arm']):r for r in lock['endpoints']}
    for row in v['rows']:
        end=by[row['seed'],row['arm']]
        if (row['training_result_sha256'],row['checkpoint_link_sha256'])!=(end['training_result_sha256'],end['checkpoint_sha256']):raise RuntimeError('Derived record lost endpoint linkage')
        if (end['step'],end['adam_age'],end['parent_steps'],end['prefix_steps'])!=(3000,10630,7630,200):raise RuntimeError('Endpoint clocks differ')
    out={}
    for partition in ('REPORT','FEV1'):
        scores=v['summaries'][partition]['rows'];res={}
        if set(scores)!=set(map(str,range(161,167))):raise RuntimeError('Missing training parent')
        for row in v['rows']:
            if scores[str(row['seed'])][row['arm']]!=row[partition]:raise RuntimeError('Score projections disagree')
        for other in ('adamw','scrambled_matched'):
            diffs=[scores[str(s)]['bdt']-scores[str(s)][other] for s in range(161,167)]
            a=paired(diffs,.95);b=paired(diffs,.975);saved=v['summaries'][partition]['contrasts']['bdt_minus_'+other]
            if not math.isclose(a['mean'],saved['mean'],abs_tol=1e-14):raise RuntimeError('Recorded arithmetic mismatch')
            for calc,key in ((a,'ci95'),(b,'ci975')):
                if any(abs(x-y)>1e-13 for x,y in zip([calc['lower'],calc['upper']],saved[key])):raise RuntimeError('Recorded interval mismatch')
            res['bdt - '+other]={'differences':diffs,'paired_95':a,'paired_97_5':b}
        out[partition]={'scores':scores,'contrasts':res}
    return {'study':'GCT2 + FEV1','training_parents':6,'model_endpoint_cells':18,'evaluation_partitions':2,'record_hash_links':36,'checkpoint_tensor_reads':0,'model_evaluations_performed':0,'results':out,'evidence_level':'Unrounded audit-verified score projections with original record/checkpoint hashes and complete original GCT2 cohort seal. Not36 redistributed raw report files. FEV1 reuses the same18 checkpoints, not six additional trained parents.'}

BAC1_ANALYSIS_SHA256='5fad4314323a48a1ebe3d4799568594ac2aa451a3c4063c8b656d10af1070183'
BAC1_LOCK_SHA256='68b73d3b31d2cfdc80275289eacb482955b1c3a8e06e5f148c59313d05a0129a'
BAC1_SOURCE='1f119bbb7be46a323e8d08bbbf1041bbb6c29f296d8cebb673c9fb6d2a058fba'
BAC1_PROTOCOL='1fd9fc1ad636dcf0da1361f1184115e8cbd2cbcbd96485a31aa1cc2b6b0ca63e'
BAC1_ARMS=('adamw','bdt','draw0_preserved','draw0_reassigned')

def bac1_records(root=None,display_path=None):
    root=ROOT/'reference/recovered_records/bac1' if root is None else Path(root)
    display_path=ROOT/'data/display_tables.json' if display_path is None else Path(display_path)
    ap=root/'BAC1_FINAL_ANALYSIS.json';lp=root/'BAC1_COHORT_LOCK.json'
    if hashlib.sha256(ap.read_bytes()).hexdigest()!=BAC1_ANALYSIS_SHA256:raise RuntimeError('Original BAC1 final-analysis bytes changed')
    if hashlib.sha256(lp.read_bytes()).hexdigest()!=BAC1_LOCK_SHA256:raise RuntimeError('Original BAC1 cohort-lock bytes changed')
    analysis=json.loads(ap.read_text());lock=json.loads(lp.read_text())
    seeds=set(range(175,181));expected={f'seed_{s}/arms/{a}/result.json' for s in seeds for a in BAC1_ARMS}
    if analysis.get('state')!='complete' or analysis.get('source')!=BAC1_SOURCE or analysis.get('protocol')!=BAC1_PROTOCOL:raise RuntimeError('BAC1 analysis identity/state mismatch')
    if lock.get('source')!=BAC1_SOURCE or lock.get('protocol')!=BAC1_PROTOCOL or set(lock.get('seeds',[]))!=seeds or tuple(lock.get('arms',[]))!=BAC1_ARMS:raise RuntimeError('BAC1 cohort-lock identity mismatch')
    endpoints=lock.get('endpoints',{})
    if set(endpoints)!=expected or len(set(endpoints.values()))!=24 or not all(len(h)==64 for h in endpoints.values()):raise RuntimeError('BAC1 cohort lock is incomplete/duplicated')
    rows=analysis.get('rows',[]);by={r['seed']:r for r in rows}
    if len(rows)!=6 or set(by)!=seeds or any(set(r)!=set(BAC1_ARMS)|{'seed'} for r in rows):raise RuntimeError('BAC1 final rows incomplete/duplicated')
    diffs=[by[s]['draw0_reassigned']-by[s]['draw0_preserved'] for s in sorted(seeds)]
    calc=paired(diffs,.95);saved=analysis.get('primary',{})
    if saved.get('n')!=6 or saved.get('positive_families')!=6 or any(abs(x-y)>1e-14 for x,y in zip(diffs,saved.get('values',[]))):raise RuntimeError('BAC1 saved primary values disagree')
    if abs(calc['mean']-saved.get('mean',math.inf))>1e-14 or abs(math.sqrt(6)*calc['se']-saved.get('sd',math.inf))>1e-14:raise RuntimeError('BAC1 saved primary mean/SD disagree')
    if any(abs(x-y)>1e-13 for x,y in zip([calc['lower'],calc['upper']],saved.get('ci95',[]))):raise RuntimeError('BAC1 saved primary CI disagrees')
    displayed=json.loads(display_path.read_text())['tables']['bac1_all_endpoints']
    if len(displayed)!=6 or {int(r[0]) for r in displayed}!=seeds:raise RuntimeError('Incomplete BAC1 display table')
    for r in displayed:
        s=int(r[0]);src=by[s]
        wanted=[str(s),*(f'{src[a]:.9f}' for a in BAC1_ARMS),f'{src["draw0_reassigned"]-src["draw0_preserved"]:+.9f}']
        if list(r)!=wanted:raise RuntimeError(f'BAC1 display row no longer rounds from original analysis for seed {s}')
    anchors={}
    for a,b in [('draw0_preserved','adamw'),('draw0_reassigned','adamw'),('draw0_preserved','bdt')]:
        anchors[a+' - '+b]=paired([by[s][a]-by[s][b] for s in sorted(seeds)],.95)
    return {'study':'BAC1','training_parents':6,'model_endpoint_cells':24,'endpoint_result_hashes':24,
            'source':BAC1_SOURCE,'protocol':BAC1_PROTOCOL,'original_analysis_sha256':BAC1_ANALYSIS_SHA256,
            'original_cohort_lock_sha256':BAC1_LOCK_SHA256,'scores_unrounded':{str(s):{a:by[s][a] for a in BAC1_ARMS} for s in sorted(seeds)},
            'primary':{'comparison':'draw0_reassigned - draw0_preserved','saved':saved,'recomputed_paired_95':calc},
            'descriptive_anchor_contrasts':anchors,'model_evaluations_performed':0,
            'evidence_level':'Unrounded completed BAC1 final analysis plus exact24-endpoint original cohort lock and endpoint-result hashes, cross-checked against every nine-decimal manuscript row. The24 raw per-arm REPORT files are still not redistributed; no new model evaluation was performed.'}

def bac1_display(root=None):
    path=ROOT/'data/display_tables.json' if root is None else Path(root)
    tab=json.loads(path.read_text())['tables']['bac1_all_endpoints']
    if len(tab)!=6 or {int(r[0]) for r in tab}!=set(range(175,181)):raise RuntimeError('Incomplete BAC1 displayed cohort')
    scores={int(r[0]):dict(zip(['adamw','bdt','draw0_preserved','draw0_reassigned'],map(float,r[1:5]))) for r in tab}
    pairs=[('draw0_reassigned','draw0_preserved'),('draw0_preserved','adamw'),('draw0_reassigned','adamw'),('draw0_preserved','bdt')]
    contrasts={a+' - '+b:paired([scores[s][a]-scores[s][b] for s in sorted(scores)],.95) for a,b in pairs}
    mean=contrasts['draw0_reassigned - draw0_preserved']['mean']
    if abs(mean-.003300288692)>1e-9:raise RuntimeError('BAC1 display no longer matches reported primary to printed precision')
    return {'study':'BAC1','training_parents':6,'scores_at_displayed_precision':scores,'contrasts_from_displayed_rows':contrasts,'primary_reported_unrounded_mean':.003300288692,'primary_reported_unrounded_ci95':[.002971574612,.003629002772],'primary_only':'draw0_reassigned - draw0_preserved','anchor_intervals':'The three other intervals are derived descriptive summaries, not additional preregistered primary tests.','model_evaluations_performed':0,'evidence_level':'All24 nine-decimal displayed endpoints and separately retained reported primary summary; full original raw BAC1 REPORT files are not redistributed. No extra digits are invented.'}
