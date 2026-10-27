"""Locked parent import, outcome-blind gates, and fixed two-parent DEV analysis."""
from __future__ import annotations
import csv,gc,json,math,statistics
from dataclasses import asdict
from pathlib import Path
import common as io

def old_identity(ck,rt):
 o=ck['opt']
 return dict(model=rt.treehash(ck['model']),rng=rt.treehash(ck['rng']),adam=rt.treehash(o['adam']),canonical=rt.treehash(o['canonical']),scrambled=rt.treehash(o['scrambled']),permutations=rt.treehash(o['permutations']),phase=o['phase'],offset=o['offset'],**({'attention':rt.treehash(o['attention'])} if 'attention' in o else {}),**({'bank':rt.treehash(o['bank'])} if 'bank' in o else {}))

def import_grown(project,seed,directory,data,tiny=False,gate=False,original=None):
 import torch
 tr,rt,mods=io.engine(project);native=mods['native'];device=torch.device('cpu' if tiny else 'cuda:0');rt.configure(4)
 if gate:
  if seed!=127 or original is None:raise RuntimeError('Invalid gate parent import')
 else:
  if tiny or seed not in io.SPEC['seeds'] or original is not None:raise RuntimeError('Unregistered scientific parent import')
  original=Path(project)/'runs'/io.SPEC['source_parent_study']/f'seed_{seed}'/'grown'
 d=Path(directory).resolve();d.relative_to(io.root(project));orig=Path(original).resolve();r=io.read(orig/'result.json');ck,meta=io.load_ck(orig,torch)
 if r['state']!='complete' or r['step']!=0 or r['checkpoint']!=meta or r['final']!=meta['identity'] or r['contract']!=ck['contract'] or r['contract_sha']!=io.digest(r['contract']):raise RuntimeError('Source grown checkpoint linkage differs')
 if old_identity(ck,rt)!=meta['identity']:raise RuntimeError('Source model/optimizer/RNG/histories do not rehash')
 c=r['contract'];age=200 if gate else 7630;cfg=tr.config(native,data['path'],True,tiny)
 expect={'seed':seed,'post':True,'offset':age,'tiny':tiny,'gate':gate,'runtime':rt.runtime(device),'cache':data['hashes'],'config':asdict(cfg)}
 if not gate:expect.update(source=io.SPEC['source_parent_source_sha256'],protocol=io.SPEC['source_parent_protocol_sha256'])
 if any(c.get(k)!=v for k,v in expect.items()):raise RuntimeError('Source grown context differs: '+str([k for k,v in expect.items() if c.get(k)!=v]))
 if ck['step']!=0 or ck['opt']['adam']['age']!=age:raise RuntimeError('Wrong grown phase/global age')
 if not gate:
  pr=io.read(orig.parent/'parent/result.json')
  if pr['step']!=7630 or pr['contract']['seed']!=seed or c['parent_sha']!=pr['checkpoint']['sha256']:raise RuntimeError('Original small-parent chain differs')
  pck,pm=io.load_ck(orig.parent/'parent',torch)
  if old_identity(pck,rt)!=pm['identity'] or pr['checkpoint']!=pm:raise RuntimeError('Original small-parent tensor identity differs')
  del pck;gc.collect()
 ct=dict(version=io.SPEC['version'],protocol=io.digest(io.SPEC),source=io.verify()[0],dependencies=__import__('gct_protocol').DEPENDENCIES,
  seed=seed,stage='grown',mode='adamw',end=0,start=0,post=True,offset=age,tiny=tiny,gate=gate,config=asdict(cfg),runtime=rt.runtime(device),cache=data['hashes'],parent_sha=meta['sha256'])
 with io.lock(d/'.writer.lock'):
  io.immutable(d/'contract.json',ct)
  if (d/'result.json').exists():return io.completed(d,ct)
  from scope_policy import Policy
  model=tr.build(native,cfg,seed).to(device);model.load_state_dict(ck['model'],strict=True);opt=Policy(model,cfg,mods,'adamw',device,seed,True,age);opt.load(ck['opt']) if 'attention' in ck['opt'] else opt.load_legacy(ck['opt']);rt.restore_rng(ck['rng'])
  ident=tr.identity(model,opt)
  if {k:ident[k] for k in meta['identity']}!=meta['identity']:raise RuntimeError('Parent conversion altered reference state')
  if any(t.last_step or t.observations or bool(torch.count_nonzero(t.covariance)) for t in getattr(opt,'attention',{}).values()):raise RuntimeError('Attention histories not initially zero')
  dev=rt.evaluate(model,Path(data['path'])/'val.bin',cfg,native,device,tiny)
  if dev!=r['grown_dev']:raise RuntimeError('Original phase-zero evaluation not reproduced')
  timing=dict(training_step_seconds=0.,evaluation_seconds=0.,checkpoint_seconds=0.,updates=0)
  payload=dict(step=0,contract=ct,model=rt.cpu(model.state_dict()),opt=opt.state(),rng=rt.cpu(rt.rng()),initial=ident,curve=[],history=[],proof=[],timing=timing)
  new=io.save_ck(d,payload,torch,ident,keep=True);io.immutable(d/'initialization.json',ident)
  out=dict(state='complete',contract=ct,contract_sha=io.digest(ct),step=0,initial=ident,final=ident,checkpoint=new,grown_dev=dev,
   imported=dict(directory=str(orig),result_sha256=io.sha(orig/'result.json'),checkpoint_sha256=meta['sha256'],original_source=c['source'],original_protocol=c['protocol']),
   checks={'original_state_exact':True,'new_attention_histories_zero':True,'original_DEV_exact':True,'no_historical_write':True})
  io.atomic(d/'result.json',out);print('BAA1 IMPORT VERIFIED',seed,meta['sha256'],flush=True);return out

def gate_modes():
 return tuple(dict.fromkeys(('adamw',('bdt' if io.SPEC['kind']=='ancestry' else 'ffn_bdt'),*io.SPEC['arms'])))
def gate_plan(d,tiny=False):
 d=Path(d)
 def j(name,stage,mode,end,parent=None,reference=False,interrupt=None):
  return dict(directory=str(d/name),seed=127,stage=stage,mode=mode,end=end,parent=str(d/parent) if parent else None,reference=reference,gate=True,tiny=tiny,interrupt=interrupt)
 jobs=[j('ref_parent','parent','adamw',200,reference=True),j('ref_grown','grown','adamw',0,'ref_parent',reference=True),j('ref_prefix','prefix','adamw',200,'ref_grown',reference=True),j('imported','import','adamw',0,'ref_grown'),j('prefix','prefix','adamw',200,'imported'),j('prefix_repeat','prefix','adamw',200,'imported')]
 modes=gate_modes()
 for m in modes:
  jobs.append(j(m+'_0','arm',m,420,'prefix'))
  if m==modes[-1]:jobs.append(j(m+'_1','arm',m,420,'prefix',interrupt=310))
  jobs.append(j(m+'_1','arm',m,420,'prefix'))
 jobs.extend([j('ref_adam','arm','adamw',420,'ref_prefix',reference=True),j('ref_ffn','arm','bdt',420,'ref_prefix',reference=True)])
 return jobs

def execute(project,j,data,resume=False):
 if j['stage']=='import':return import_grown(project,j['seed'],j['directory'],data,j.get('tiny',False),j['gate'],j.get('parent') if j['gate'] else None)
 tr,rt,_=io.engine(project,reference=j.get('reference',False));rt.configure(4)
 kw=dict(gate=j['gate'],tiny=j.get('tiny',False),data=data,resume=resume)
 if j['stage']=='grown':return tr.grow(project,j['directory'],j['seed'],j['parent'],**kw)
 return tr.train(project,j['directory'],j['seed'],j['stage'],j['mode'],j['end'],j.get('parent'),interrupt=j.get('interrupt'),**kw)

def compare_results(a,b,reference=False):
 import torch
 _,rt,_=io.engine(Path(a).parents[2]);a=Path(a);b=Path(b);x=io.read(a/'result.json');y=io.read(b/'result.json')
 if x['state']!='complete' or y['state']!='complete' or not x['proof'] or len(x['proof'])!=len(y['proof']):raise RuntimeError('Missing gate traces')
 for k in ('model','adam','rng','phase','offset'):
  if x['initial'][k]!=y['initial'][k] or x['final'][k]!=y['final'][k]:raise RuntimeError('Gate exact state mismatch '+k)
 for u,v in zip(x['proof'],y['proof']):
  keys=('step','global_step','loss','grad_norm','lr','batch','rng_before','rng_after','gradient_sample','parameter_sample','gradient','model','adam','dev') if reference else tuple(set(u)|set(v))
  # PCT adds full-vector diagnostics at every step of its registered windows.
  # Reference's sparse proof does not contain full hashes at these extra steps;
  # compare its full hashes wherever present, plus mandatory samples every step.
  if reference:keys=tuple(k for k in keys if k not in ('gradient','model','adam') or v.get(k) is not None)
  if any(u.get(k)!=v.get(k) for k in keys):raise RuntimeError('Gate trace mismatch step '+str(u.get('step'))+' fields '+str([k for k in keys if u.get(k)!=v.get(k)]))
 # Full final tensors, not only sampled hash traces.
 ca,ma=io.load_ck(a,torch);cb,mb=io.load_ck(b,torch)
 for k in ('model','rng'):
  if rt.treehash(ca[k])!=rt.treehash(cb[k]):raise RuntimeError('Full final payload differs '+k)
 if rt.treehash(ca['opt']['adam'])!=rt.treehash(cb['opt']['adam']):raise RuntimeError('Full final moments/counters differ')
 for k in ('canonical','scrambled','permutations'):
  if rt.treehash(ca['opt'][k])!=rt.treehash(cb['opt'][k]):raise RuntimeError('Full FFN history differs '+k)
 if not reference:
  for key in ('attention','bank'):
   if rt.treehash(ca['opt'].get(key))!=rt.treehash(cb['opt'].get(key)):raise RuntimeError('Extra history repeat/resume differs '+key)
 return dict(exact=True,steps=len(x['proof']),full_payload_equal=True,source_result_hashes=[io.sha(a/'result.json'),io.sha(b/'result.json')])

def compare_gate(d):
 d=Path(d);anchor='bdt' if io.SPEC['kind']=='ancestry' else 'ffn_bdt'
 pairs=[('prefix','prefix_repeat',False),('prefix','ref_prefix',True),('adamw_0','ref_adam',True),(anchor+'_0','ref_ffn',True)]
 pairs += [(m+'_0',m+'_1',False) for m in gate_modes()]
 checks={a+'__'+b:compare_results(d/a,d/b,ref) for a,b,ref in pairs}
 rows=[next(z for z in io.read(d/(m+'_0')/'result.json')['curve'] if z['step']==201) for m in gate_modes()]
 for key in ('gradient','raw','batch','rng_before','rng_after'):
  if len({z[key] for z in rows})!=1:raise RuntimeError('First active input/proposals differ '+key)
 checks['first_active_inputs_equal']=True
 for mode in io.SPEC['arms']:
  r=io.read(d/(mode+'_0')/'result.json')
  if io.SPEC['kind']=='localization' and mode!='ffn_bdt':
   if not r.get('attention_summary',{}).get('applied_nonidentity_steps',0):raise RuntimeError('Gate attention intervention inactive')
  if io.SPEC['kind']=='ancestry' and mode.startswith('draw'):
   if not r.get('chart_summary',{}).get('matched_projection_steps',0):raise RuntimeError('Gate chart intervention inactive')
 checks['active_interventions_verified']=True
 return checks

def science_jobs(project,seeds=None):
 root=io.root(project);out=[]
 for seed in (io.SPEC['seeds'] if seeds is None else seeds):
  d=root/f'seed_{seed}'
  out.append(dict(directory=str(d/'grown'),seed=seed,stage='import',mode='adamw',end=0,parent=None,gate=False))
  out.append(dict(directory=str(d/'prefix'),seed=seed,stage='prefix',mode='adamw',end=200,parent=str(d/'grown'),gate=False))
  for m in io.SPEC['arm_order'][str(seed)]:out.append(dict(directory=str(d/'arms'/m),seed=seed,stage='arm',mode=m,end=3000,parent=str(d/'prefix'),gate=False))
 return out

def check_original_prefix(project,seed,directory):
 r=io.read(Path(directory)/'result.json');old=Path(project)/'runs'/io.SPEC['source_parent_study']/f'seed_{seed}'/'prefix';ref=io.read(old/'result.json')
 for k in ref['final']:
  if r['final'].get(k)!=ref['final'][k]:raise RuntimeError('New shared prefix differs from original '+k)
 if r['dev']!=ref['dev'] or r['history']!=ref['history']:raise RuntimeError('New prefix DEV/history differs from source')
 io.atomic(Path(directory)/'original_prefix_check.json',dict(state='passed',original_result_sha256=io.sha(old/'result.json'),new_result_sha256=io.sha(Path(directory)/'result.json'),time=io.now()))

def analyze(project):
 root=io.root(project);rows=[];seed_contrasts=[];endpoint_hashes={}
 for seed in io.SPEC['seeds']:
  d=root/f'seed_{seed}';prefix=io.completed(d/'prefix',{'seed':seed,'stage':'prefix','end':200,'gate':False,'tiny':False,'protocol':io.digest(io.SPEC)})
  if prefix is None:raise RuntimeError('Missing prefix')
  rs={m:io.completed(d/'arms'/m,{'seed':seed,'stage':'arm','mode':m,'end':3000,'gate':False,'tiny':False,'protocol':io.digest(io.SPEC)}) for m in io.SPEC['arms']}
  if any(r is None for r in rs.values()):raise RuntimeError('Missing registered endpoint')
  base=next(iter(rs.values()));inputs=[]
  for mode,r in rs.items():
   for k in ('config','source','cache','runtime','offset','seed'):
    if r['contract'][k]!=prefix['contract'][k]:raise RuntimeError('Mixed context '+k)
   if r['initial']!=prefix['final'] or r['contract']['parent_sha']!=prefix['checkpoint']['sha256']:raise RuntimeError('Not common prefix')
   if r['history']!=base['history'] or [x['step'] for x in r['history']]!=list(range(201,3001)) or any(x['global_step']!=7630+x['step'] for x in r['history']):raise RuntimeError('Different data/RNG history')
   if r['dev']['offset']!=0 or r['dev']['tokens']!=1048576 or not math.isfinite(r['dev']['loss']):raise RuntimeError('Wrong DEV endpoint')
   first=next(x for x in r['curve'] if x['step']==201);inputs.append({k:first[k] for k in ('gradient','raw','batch','rng_before','rng_after')})
   endpoint_hashes[str((d/'arms'/mode/'result.json').relative_to(root))]=io.sha(d/'arms'/mode/'result.json')
   for draw in (0,1):
    a=r['chart_design'][f'draw{draw}_preserved'];b=r['chart_design'][f'draw{draw}_reassigned']
    if a['preserved_ancestry_pairs']!=6 or b['preserved_ancestry_pairs']!=0 or b['false_shared_chart_pairs']!=6:raise RuntimeError('Wrong ancestry assignment')
    if a['chart_multiset_sha256']!=b['chart_multiset_sha256'] or a['chart_multiplicities']!=[2]*6 or b['chart_multiplicities']!=[2]*6:raise RuntimeError('Unbalanced charts')
  if any(x!=inputs[0] for x in inputs[1:]):raise RuntimeError('First active inputs differ')
  z=dict(seed=seed,**{m:r['dev']['loss'] for m,r in rs.items()});rows.append(z)
  preserved=statistics.mean(z[f'draw{k}_preserved'] for k in (0,1));reassigned=statistics.mean(z[f'draw{k}_reassigned'] for k in (0,1))
  seed_contrasts.append(dict(seed=seed,reassigned_minus_preserved=reassigned-preserved,preserved_minus_canonical=preserved-z['bdt'],canonical_minus_adam=z['bdt']-z['adamw'],draw0_primary=z['draw0_reassigned']-z['draw0_preserved'],draw1_primary=z['draw1_reassigned']-z['draw1_preserved']))
 io.immutable(root/'COHORT_LOCK.json',dict(source=io.verify()[0],protocol=io.digest(io.SPEC),seeds=io.SPEC['seeds'],arms=io.SPEC['arms'],endpoint_sha256=endpoint_hashes))
 keys=[k for k in seed_contrasts[0] if k!='seed'];contrasts={k:dict(values=[r[k] for r in seed_contrasts],mean=statistics.mean(r[k] for r in seed_contrasts)) for k in keys}
 candidate=all(z['reassigned_minus_preserved']>0 for z in seed_contrasts)
 result=dict(state='complete',candidate=candidate,automatic_extension=False,rows=rows,seed_contrasts=seed_contrasts,contrasts=contrasts,source=io.verify()[0],protocol=io.digest(io.SPEC),scope=io.SPEC['inference'],time=io.now(),cohort_lock_sha256=io.sha(root/'COHORT_LOCK.json'))
 io.atomic(root/'analysis.json',result)
 with (root/'endpoints.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (root/'READOUT.md').write_text('# BAA1 balanced-assignment development results\n\n'+io.SPEC['inference']+'\n\n'+json.dumps(result,indent=2)+'\n')
 return result
