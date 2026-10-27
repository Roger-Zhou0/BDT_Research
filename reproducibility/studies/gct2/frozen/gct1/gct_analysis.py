"""GCT1: complete, provenance-checked DEV pilot; no confirmatory inference."""
from pathlib import Path
import csv,json,math,statistics
import gct_protocol as p
import gct_io as io

def compare(a,b):
 a=Path(a);b=Path(b);x=io.read(a/'result.json');y=io.read(b/'result.json')
 if x['state']!='complete' or y['state']!='complete':raise RuntimeError('Incomplete control')
 if x['initial']!=y['initial']:raise RuntimeError('Control parents differ')
 if len(x['proof'])!=len(y['proof']) or not x['proof']:raise RuntimeError('Missing exact-control trace')
 for r,s in zip(x['proof'],y['proof']):
  if r!=s:
   differing=[k for k in set(r)|set(s) if r.get(k)!=s.get(k)]
   raise RuntimeError(f"Repeated/reference execution differs at {r.get('step')}: {differing}")
 for k in ('model','adam'):
  if x['final'][k]!=y['final'][k]:raise RuntimeError('Final full state differs '+k)
 if x['dev']!=y['dev']:raise RuntimeError('Control final DEV differs')
 return dict(exact_recorded=True,steps=len(x['proof']),first_result_sha=io.sha(a/'result.json'),second_result_sha=io.sha(b/'result.json'),full_final_model_adam_equal=True)

def analyze(project,write=True):
 root=p.root(project);rows=[];lineage=[];source=io.verify(project)[0]
 for seed in p.SEEDS:
  d=root/f'seed_{seed}'
  for f in [d/'parent/result.json',d/'grown/result.json',d/'prefix/result.json']+[d/'arms'/m/'result.json' for m in p.ARMS]:
   if not f.is_file():return dict(state='incomplete',missing=str(f))
  parent,grown,prefix=[io.read(d/s/'result.json') for s in ('parent','grown','prefix')]
  for r in (parent,grown,prefix):
   if r['state']!='complete' or r['contract']['source']!=source or r['contract']['seed']!=seed or r['contract']['gate'] or r['contract']['tiny']:raise RuntimeError('Invalid scientific parent record')
   if r['contract'].get('protocol')!=p.PROTOCOL_SHA or r['contract'].get('cache')!=p.CACHE or r.get('contract_sha')!=p.digest(r['contract']):raise RuntimeError('Parent protocol/cache/digest mismatch')
  if parent['step']!=7630 or grown['step']!=0 or prefix['step']!=200:raise RuntimeError('Wrong parent/phase budget')
  if grown['contract']['parent_sha']!=parent['checkpoint']['sha256'] or prefix['contract']['parent_sha']!=grown['checkpoint']['sha256']:raise RuntimeError('Broken growth/prefix link')
  if prefix['initial']!=grown['final'] or prefix['final']['offset']!=7630 or prefix['final']['phase']!=200:raise RuntimeError('Prefix history or age did not preserve growth state')
  rs={m:io.read(d/'arms'/m/'result.json') for m in p.ARMS}
  base=rs['adamw'];headers=[];metrics=[]
  for m,r in rs.items():
   if r['state']!='complete' or r['step']!=3000 or r['contract']['mode']!=m or r['contract_sha']!=p.digest(r['contract']):raise RuntimeError('Incomplete or mislabeled final arm')
   for k in ('seed','source','protocol','config','runtime','cache','offset','gate','tiny'):
    if r['contract'][k]!=prefix['contract'][k]:raise RuntimeError('Mixed contract '+k)
   if r['initial']!=prefix['final'] or r['contract']['parent_sha']!=prefix['checkpoint']['sha256']:raise RuntimeError('Arms do not share exact post-growth prefix')
   if r['final']['phase']!=3000 or r['final']['offset']!=7630:raise RuntimeError('Wrong final optimizer phase')
   if len(r['history'])!=2800 or [z['step'] for z in r['history']]!=list(range(201,3001)):raise RuntimeError('Training history incomplete')
   if any(z['global_step']!=7630+z['step'] for z in r['history']):raise RuntimeError('Global data index was reset')
   if r['history']!=base['history']:raise RuntimeError('Training batches or RNG differ')
   if r['dev'].get('offset')!=0 or r['dev'].get('tokens')!=1048576 or r['dev'].get('stop')!=1050624:raise RuntimeError('Incorrect DEV partition')
   if not math.isfinite(r['dev']['loss']):raise RuntimeError('Nonfinite endpoint')
   first=next(z for z in r['curve'] if z['step']==201);headers.append({k:first[k] for k in ('gradient','raw','batch','rng_before','rng_after')})
   if r['matching']['active_groups']!=8400:raise RuntimeError('Matching history incomplete')
   metrics.append(r['matching'])
   # Verify every linked current checkpoint metadata record, without reopening huge tensors here.
   meta=io.read(d/'arms'/m/'latest.json')
   if meta!=r['checkpoint']:raise RuntimeError('Final checkpoint pointer changed')
  if any(h!=headers[0] for h in headers):raise RuntimeError('First active inputs differ')
  row=dict(seed=seed,small_parent_loss=parent['dev']['loss'],immediate_grown_loss=grown['grown_dev']['loss'],growth_loss_jump=grown['growth_loss_jump'],prefix_loss=prefix['dev']['loss'])
  row.update({m:r['dev']['loss'] for m,r in rs.items()});row['bdt_minus_adam']=row['bdt']-row['adamw'];row['bdt_minus_matched']=row['bdt']-row['scrambled_matched'];rows.append(row)
  lineage.append(dict(seed=seed,parent=parent['checkpoint'],grown=grown['checkpoint'],prefix=prefix['checkpoint'],growth_checks=grown['checks'],
    parent_seconds=parent['timing'],prefix_seconds=prefix['timing'],arms={m:dict(timing=r['timing'],memory=r['memory'],matching=r['matching'],result_sha=io.sha(d/'arms'/m/'result.json')) for m,r in rs.items()}))
 means={k:statistics.mean(r[k] for r in rows) for k in ('bdt_minus_adam','bdt_minus_matched')}
 if means['bdt_minus_adam']>=0:decision='Do not automatically expand this application.'
 elif means['bdt_minus_matched']>=0:decision='Incremental pilot signal only; directional transfer not supported by the mean comparison.'
 else:decision='Both pilot means favorable; consider a separately registered fresh confirmation. None launched automatically.'
 result=dict(state='complete',protocol=p.PROTOCOL_SHA,source=source,seeds=list(p.SEEDS),rows=rows,mean_contrasts=means,
   favorable_counts={k:sum(r[k]<0 for r in rows) for k in means},decision=decision,provenance=lineage,
   accounting=dict(parent_updates_per_path=7630,post_updates_per_path=3000,parent_tokens_per_path=250019840,post_tokens_per_path=98304000,campaign_optimizer_updates=p.SPEC['scientific_optimizer_updates']),
   scope='Two fresh parent-training seeds on S1-B. DEV pilot, no confirmatory tests. No scratch baseline, no GS superiority or end-to-end efficiency claim.',time=io.now())
 if write:
  io.atomic(root/'analysis.json',result)
  with (root/'endpoints.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  text=['# GCT1 complete — two-seed application pilot','',result['scope'],'','| Seed | AdamW | BDT | Matched scrambled | BDT − Adam | BDT − matched |','|---|---:|---:|---:|---:|---:|']
  text += [f"| {r['seed']} | {r['adamw']:.9f} | {r['bdt']:.9f} | {r['scrambled_matched']:.9f} | {r['bdt_minus_adam']:+.9f} | {r['bdt_minus_matched']:+.9f} |" for r in rows]
  text += ['',f"Mean BDT − Adam: {means['bdt_minus_adam']:+.9f}; mean BDT − matched: {means['bdt_minus_matched']:+.9f}.",'',decision,'','Parent cost is counted in full for each recommended path, not amortized over counterfactual arms. Timings include instrumentation and shared-device effects, not isolated optimizer overhead.','', 'All six endpoints, growth jumps, matching summaries and recorded time/memory are in analysis.json.']
  (root/'READOUT.md').write_text('\n'.join(text)+'\n')
 return result
