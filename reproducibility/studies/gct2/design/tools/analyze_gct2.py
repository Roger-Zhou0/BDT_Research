"""Prospective, strict six-parent paired endpoint analyzer.

Input format is specified in INPUT_SCHEMA.md. These are validated normalized
records from a future collector, not unverified hand-entered final losses.
Metadata validation cannot replace source/tensor audits by that collector.
"""
from __future__ import annotations
import argparse, datetime, json, math, statistics
from pathlib import Path
from design import GCT2, SEEDS, ARMS, CACHE, digest

def timestamp(value):
    if not isinstance(value,str):raise ValueError('Missing timestamp')
    dt=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    if dt.tzinfo is None:raise ValueError('Timestamp must include timezone')
    return dt

def hash_string(value):
    if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError('Expected SHA-256 digest')
    return value

def paired(values):
    if len(values)!=6 or not all(type(v) in (int,float) and math.isfinite(v) for v in values):
        raise ValueError('Require six finite paired differences')
    mean=statistics.mean(values);sd=statistics.stdev(values);se=sd/math.sqrt(6)
    if sd==0:
        return dict(n=6,values=values,mean=mean,sd=sd,ci95=None,ci975=None,
                    status='degenerate_variance_requires_review',favorable=sum(v<0 for v in values))
    t95=GCT2['analysis']['t975_df5'];tadj=GCT2['analysis']['t9875_df5']
    return dict(n=6,values=values,mean=mean,sd=sd,ci95=[mean-t95*se,mean+t95*se],
                ci975=[mean-tadj*se,mean+tadj*se],status='analyzed',favorable=sum(v<0 for v in values))

def analyze(bundle):
    if bundle.get('schema')!='gct2_endpoint_bundle_v1':raise ValueError('Wrong schema')
    if bundle.get('protocol_digest')!=digest(GCT2):raise ValueError('Protocol mismatch')
    if bundle.get('seeds')!=list(SEEDS):raise ValueError('Unregistered seed cohort or pilot pooling')
    if bundle.get('cache_sha256')!=CACHE:raise ValueError('Mixed or incorrect cache')
    if bundle.get('report_partition')!=GCT2['report']:raise ValueError('Endpoint/partition changed')
    if bundle.get('parent_cost_counted_per_path') is not True:raise ValueError('Parent cost cannot be omitted')
    inventory=bundle.get('inventory',{})
    if inventory.get('complete') is not True or inventory.get('collisions')!=[]:raise ValueError('Incomplete or conflicting seed inventory')
    hash_string(inventory.get('manifest_sha256'))
    checks=bundle.get('checks',{})
    required=('source','cache','growth_mapping','optimizer_age','all_parents_distinct','lineage',
              'first_active_inputs','full_batch_rng_histories','matching','final_checkpoints','target_gates')
    if any(checks.get(k) is not True for k in required):raise ValueError('Required audit has not passed')
    release=hash_string(bundle.get('training_source_sha256'))
    collector=hash_string(bundle.get('collector_source_sha256'))
    lock=bundle.get('cohort_lock',{})
    if lock.get('state')!='sealed':raise ValueError('Cohort must be sealed')
    sealed=timestamp(lock.get('sealed_at'))
    locked=lock.get('endpoints',[])
    if len(locked)!=18:raise ValueError('Need all eighteen locked final training records')
    expected={(s,a) for s in SEEDS for a in ARMS};indexed={}
    for row in locked:
        key=(row.get('seed'),row.get('arm'))
        if key not in expected or key in indexed:raise ValueError('Duplicate/unregistered locked endpoint')
        if row.get('step')!=3000 or row.get('adam_age')!=10630:raise ValueError('Wrong phase/global endpoint')
        if row.get('parent_steps')!=7630 or row.get('prefix_steps')!=200:raise ValueError('Wrong parent/prefix budget')
        if timestamp(row.get('completed_at'))>sealed:raise ValueError('Training ended after cohort seal')
        for k in ('training_result_sha256','checkpoint_sha256','prefix_sha256','small_parent_sha256'):hash_string(row.get(k))
        indexed[key]=row
    parents=[]
    for seed in SEEDS:
        group=[indexed[(seed,a)] for a in ARMS]
        for k in ('prefix_sha256','small_parent_sha256'):
            if len({x[k] for x in group})!=1:raise ValueError('Branches do not share their registered parent')
        parents.append(group[0]['small_parent_sha256'])
    if len(set(parents))!=6:raise ValueError('Independent parents were reused')
    records=bundle.get('reports',[])
    if len(records)!=18:raise ValueError('Require all eighteen REPORT evaluations')
    losses={}
    for row in records:
        key=(row.get('seed'),row.get('arm'))
        if key not in expected or key in losses:raise ValueError('Duplicate or unregistered report')
        if row.get('partition')!='REPORT' or row.get('tokens')!=1048576 or row.get('offset')!=2500000:
            raise ValueError('Unexpected scored tokens or partition')
        if timestamp(row.get('started_at'))<sealed:raise ValueError('REPORT was accessed before the cohort seal')
        if timestamp(row.get('completed_at'))<timestamp(row.get('started_at')):raise ValueError('Reversed report timestamps')
        for k in ('training_result_sha256','checkpoint_sha256'):
            if row.get(k)!=indexed[key][k]:raise ValueError('Report/checkpoint linkage mismatch')
        if row.get('training_source_sha256')!=release or row.get('collector_source_sha256')!=collector:
            raise ValueError('Source mismatch')
        value=row.get('loss')
        if type(value) not in (float,int) or not math.isfinite(value) or value<=0:raise ValueError('Nonfinite or invalid loss')
        losses[key]=value
    comparisons={}
    for first,second in GCT2['analysis']['contrasts']:
        comparisons[first+'_minus_'+second]=paired([losses[(s,first)]-losses[(s,second)] for s in SEEDS])
    favorable={key:(value['ci975'] is not None and value['ci975'][1]<0) for key,value in comparisons.items()}
    return dict(state='complete',protocol_digest=digest(GCT2),input_digest=digest(bundle),seeds=list(SEEDS),
        comparisons=comparisons,adjusted_contrast_support=favorable,joint_confirmation=all(favorable.values()),
        rows=[dict(seed=s,**{a:losses[(s,a)] for a in ARMS}) for s in SEEDS],
        scope='Fixed-recipe growth-transfer confirmation over six independent training seeds; established REPORT text. No GS or scratch superiority/time-to-quality claim.')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('bundle',type=Path);args=ap.parse_args()
    result=analyze(json.loads(args.bundle.read_text()));print(json.dumps(result,indent=2,allow_nan=False))
