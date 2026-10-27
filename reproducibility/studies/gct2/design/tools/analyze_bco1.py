"""Fixed-block paired cost analysis; no throughput or time-to-quality extrapolation."""
from __future__ import annotations
import argparse,json,math,statistics
from pathlib import Path
from design import BCO1,CONDITIONS,digest
from analyze_gct2 import hash_string

def describe(values):
    q=statistics.quantiles(values,n=4,method='inclusive')
    return dict(n=len(values),values=values,median=statistics.median(values),q25=q[0],q75=q[2])

def analyze(blocks,family='full_step'):
    spec=BCO1['timing']
    if family not in spec['families']:raise ValueError('Unknown measurement family')
    if len(blocks)!=12:raise ValueError('Require all twelve registered blocks')
    seen=set();uuids=set();runtimes=set();sources=set();states=set();rows=[]
    steps=20 if family=='full_step' else 100
    for b in blocks:
        index=b.get('block')
        if type(index) is not int or index not in range(12) or index in seen:raise ValueError('Unknown or duplicate block')
        seen.add(index)
        if b.get('protocol_digest')!=digest(BCO1) or b.get('state')!='complete' or b.get('family')!=family:
            raise ValueError('Wrong or incomplete block')
        if b.get('exact_equivalence_gate') is not True:raise ValueError('Candidate lacks exact equivalence')
        uuid=b.get('gpu_uuid')
        if not isinstance(uuid,str) or not uuid.startswith('GPU-'):raise ValueError('Missing actual GPU UUID')
        uuids.add(uuid);runtimes.add(hash_string(b.get('runtime_sha256')));sources.add(hash_string(b.get('source_sha256')))
        ms=b.get('measurements',[])
        if [r.get('condition') for r in ms]!=spec['orders'][index]:raise ValueError('Order/incomplete condition set changed')
        all_ms=ms+[b.get('sentinel_before',{}),b.get('sentinel_after',{})]
        for r in all_ms:
            if r.get('updates')!=steps or r.get('start')!=501 or r.get('end')!=500+steps:
                raise ValueError('Wrong timed window')
            if r.get('restored_after_warmup') is not True or r.get('synchronized') is not True:
                raise ValueError('Warmup or synchronization mismatch')
            if r.get('profiler') is not False or r.get('shadow_matching') is not False:
                raise ValueError('Primary timer contains profiler or matched-control shadows')
            if r.get('audit_hashes_inside_timer') is not False:raise ValueError('Hashing contaminates primary timer')
            value=r.get('wall_ms')
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:raise ValueError('Invalid timing')
            expected=steps//10 if r.get('condition')!='split_adam' else 0
            if r.get('observations_per_projection')!=expected:raise ValueError('Covariance cadence changed')
            states.add(hash_string(r.get('common_state_sha256')))
        if any(b[k].get('condition')!='split_adam' for k in ('sentinel_before','sentinel_after')):raise ValueError('Wrong timing sentinel')
        indexed={r['condition']:r for r in ms}
        for key in ('final_model_sha256','final_adam_sha256','final_covariance_sha256'):
            if hash_string(indexed['reference_bdt'].get(key))!=hash_string(indexed['candidate_bdt'].get(key)):
                raise ValueError('Timed candidate no longer agrees with reference')
        times={k:v['wall_ms']/steps for k,v in indexed.items()}
        before=b['sentinel_before']['wall_ms'];after=b['sentinel_after']['wall_ms']
        drift=abs(after-before)/before
        rows.append(dict(block=index,stable=drift<=.1,drift=drift,
                         candidate_over_reference=times['candidate_bdt']/times['reference_bdt'],
                         reference_over_split=times['reference_bdt']/times['split_adam'],
                         candidate_over_split=times['candidate_bdt']/times['split_adam']))
    if any(len(x)!=1 for x in (uuids,runtimes,sources,states)):raise ValueError('Mixed hardware/runtime/source/common state')
    rows.sort(key=lambda r:r['block']);stable=[r for r in rows if r['stable']]
    metrics={k:dict(all_blocks=describe([r[k] for r in rows]),
                    stable_subset=describe([r[k] for r in stable]) if len(stable)>=2 else None)
             for k in ('candidate_over_reference','reference_over_split','candidate_over_split')}
    qualified=len(stable)>=8
    fast=sum(r['candidate_over_reference']<1 for r in rows)
    meaningful=(family=='full_step' and qualified and fast>=10 and metrics['candidate_over_reference']['all_blocks']['median']<=.98)
    ref_overhead=metrics['reference_over_split']['all_blocks']['median']-1
    cand_overhead=metrics['candidate_over_split']['all_blocks']['median']-1
    return dict(state='complete',family=family,blocks=12,stable_blocks=len(stable),candidate_faster_blocks=fast,
                quality_control='sentinel_adequate' if qualified else 'timing_inconclusive',
                metrics=metrics,meaningful_engineering_reduction=meaningful,
                half_overhead_goal_met=(ref_overhead>0 and cand_overhead<=.5*ref_overhead),
                scope='Contemporaneous paired implementation measurement, not scientific replications or end-to-end speedup.')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('input',type=Path)
    ap.add_argument('--family',choices=BCO1['timing']['families'],default='full_step');args=ap.parse_args()
    print(json.dumps(analyze(json.loads(args.input.read_text()),args.family),indent=2))
