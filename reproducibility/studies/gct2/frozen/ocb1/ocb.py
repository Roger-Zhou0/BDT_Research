"""Fixed-block CUDA cost benchmark. No hashes/profiler/checkpoints inside timers."""
from __future__ import annotations
import gc,json,math,os,time,uuid
from pathlib import Path
import torch
import common as io
from modeling import *
SPEC=io.benchmark_design()
KINDS={'dense12_split_ffn_adam':'split_ffn','dense6_split_ffn_adam':'split_ffn','dense12_bdt_canonical':'bdt','dense12_split_allproj_adam':'split_all','dense12_gs_allproj':'gs_all'}
AUDIT={'dense12_audit_adam':'adamw','dense12_audit_bdt':'bdt','dense12_audit_scrambled':'scrambled_matched'}

def release(m,opt,device):
 del m,opt;gc.collect()
 if device.type=='cuda':torch.cuda.empty_cache()

def snapshot(model,opt,cfg,seed,data):
 return dict(model=cpu(model.state_dict()),adam=cpu(normalized(model,opt)),rng=cpu(rng()),phase=opt.phase,
  canonical=cpu({q:t.state_dict() for q,t in opt.canonical.items()}),scrambled=cpu({q:t.state_dict() for q,t in opt.scrambled.items()}),
  permutations=cpu(opt.permutations),summary=copy.deepcopy(opt.summary),layers=12 if len(model.blocks) in (4,12) else 6,
  cfg=asdict_cfg(cfg),seed=seed,normalized_identity=dict(model=treehash(model.state_dict()),adam=normalized_hash(model,opt),rng=treehash(rng())),
  cache=data['hashes'],source=io.verify_source_only(),runtime=runtime(next(model.parameters()).device))

def asdict_cfg(c):
 from dataclasses import asdict
 return asdict(c)

def factory(mods,snap,device,condition,tiny=False):
 cfg=mods['native'].ExperimentConfig(**snap['cfg']);m=build(mods,cfg,127,device)
 if condition=='dense12_native_fused_adam':opt=native_optimizer(m,cfg,device)
 elif condition in AUDIT:opt=AuditPolicy(m,cfg,mods,AUDIT[condition],device,127,post=True)
 else:opt=LeanPolicy(m,cfg,mods,KINDS[condition],device,127)
 restore(m,opt,snap)
 return m,opt,cfg

@torch.no_grad()
def restore(model,opt,snap):
 model.load_state_dict(snap['model'],strict=True);restore_normalized(model,opt,snap['adam'])
 if not isinstance(opt,torch.optim.AdamW):
  opt.phase=snap['phase']
  for key in ('canonical','scrambled'):
   for q,t in getattr(opt,key).items():
    if q not in snap[key]:raise RuntimeError('Missing benchmark history '+key+'/'+q)
    t.load_state_dict(snap[key][q])
    if t.last_step!=snap['phase']:raise RuntimeError('Wrong restored covariance age')
  if opt.permutations is not None:
   if not torch.equal(opt.permutations.cpu(),snap['permutations']):raise RuntimeError('Permutation snapshot changed')
  opt.summary=copy.deepcopy(snap['summary'])
 model.zero_grad(set_to_none=True);restore_rng(snap['rng'])
 actual=dict(model=treehash(model.state_dict()),adam=normalized_hash(model,opt),rng=treehash(rng()))
 if actual!=snap['normalized_identity']:raise RuntimeError('Named state mapping/restoration is not exact')

@torch.no_grad()
def step(opt,phase,lr=.0006,full=False):
 if isinstance(opt,torch.optim.AdamW):
  for group in opt.param_groups:group['lr']=lr
  opt.step();return {}
 return opt.step(lr,phase,full=full,sampled=False)

def build_state(project,data,layers,device,tiny=False):
 mods=io.backend(project);cfg=config(mods['native'],data['path'],layers,3000,tiny=tiny)
 m=build(mods,cfg,127,device)
 opt=AuditPolicy(m,cfg,mods,'adamw',device,127,post=True) if layers==12 else LeanPolicy(m,cfg,mods,'split_ffn',device)
 store=mods['native'].TokenStore(Path(data['path'])/'train.bin')
 for s in range(1,501):
  xy=batch(store,cfg,127,s);backward(m,cfg,mods,device,xy)
  norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
  if not bool(torch.isfinite(norm)):raise FloatingPointError('State-builder gradient nonfinite')
  opt.step(mods['native'].learning_rate(cfg,s),s)
  if s%50==0:print('OCB1 ENGINEERING STATE',layers,s,'/500',flush=True);io.guard_cache(data)
 snap=snapshot(m,opt,cfg,127,data);snap['layers']=layers
 snap['common_state_sha']=treehash(snap)
 # CPU batch ring and gradients are frozen AFTER state selection; no validation selection.
 ring=[cpu(batch(store,cfg,127,s)) for s in range(501,521)]
 backward(m,cfg,mods,device,ring[0]);norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
 if not bool(torch.isfinite(norm)):raise FloatingPointError('Stored gradients nonfinite')
 snap['fixed_gradients']={n:cpu(p.grad) for n,p in m.named_parameters()}
 snap['batch_ring']=ring;snap['input_sha']=treehash(ring);snap['gradient_sha']=treehash(snap['fixed_gradients'])
 snap['parameters']=sum(p.numel() for p in m.parameters())
 if not tiny and snap['parameters']!={6:81074688,12:123551232}[layers]:raise RuntimeError('OCB architecture mismatch')
 del m,opt;gc.collect()
 if device.type=='cuda':torch.cuda.empty_cache()
 return snap

def save_state(path,snap):
 path=Path(path);tmp=path.with_suffix('.'+uuid.uuid4().hex+'.tmp');path.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open('xb') as f:torch.save(snap,f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,path);io.atomic(path.with_suffix('.json'),dict(path=path.name,sha256=io.sha(path),common_state_sha=snap['common_state_sha'],source=snap['source'],runtime=snap['runtime'],phase=snap['phase'],input_sha=snap['input_sha'],gradient_sha=snap['gradient_sha']))

def read_state(path):
 path=Path(path);meta=io.read(path.with_suffix('.json'))
 if io.sha(path)!=meta['sha256']:raise RuntimeError('Benchmark snapshot corrupted')
 snap=torch.load(path,map_location='cpu',weights_only=False)
 if snap['phase']!=500 or snap['common_state_sha']!=meta['common_state_sha']:raise RuntimeError('Wrong engineering state')
 original={k:v for k,v in snap.items() if k not in ('common_state_sha','fixed_gradients','batch_ring','input_sha','gradient_sha','parameters')}
 if treehash(original)!=snap['common_state_sha']:raise RuntimeError('Engineering state digest changed')
 if snap['input_sha']!=treehash(snap['batch_ring']) or snap['gradient_sha']!=treehash(snap['fixed_gradients']):raise RuntimeError('Benchmark inputs changed')
 return snap

def evidence_window(mods,snap,device,name,*,tiny=False,count=20,resume_half=False):
 m,opt,cfg=factory(mods,snap,device,name,tiny);rows=[]
 for i in range(count):
  phase=snap['phase']+i+1;xy=snap['batch_ring'][i%len(snap['batch_ring'])]
  backward(m,cfg,mods,device,xy);norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
  if not bool(torch.isfinite(norm)):raise FloatingPointError('Gate nonfinite')
  rec=step(opt,phase,full=True)
  rows.append(dict(step=phase,gradient=treehash({n:p.grad for n,p in m.named_parameters()}),raw=rec.get('raw'),applied=rec.get('applied'),model=treehash(m.state_dict()),adam=normalized_hash(m,opt)))
  if resume_half and i==count//2-1:
   midway=snapshot(m,opt,cfg,127,dict(hashes=snap['cache']));midway['batch_ring']=snap['batch_ring']
   del opt,m;gc.collect()
   if device.type=='cuda':torch.cuda.empty_cache()
   m,opt,cfg=factory(mods,midway,device,name,tiny);del midway
 out=dict(rows=rows,model=treehash(m.state_dict()),adam=normalized_hash(m,opt),rng=treehash(rng()))
 del opt,m;gc.collect()
 if device.type=='cuda':torch.cuda.empty_cache()
 return out

def correctness(project,snaps,device,tiny=False):
 mods=io.backend(project);s=snaps[12];results={};checks={}
 for name in ('dense12_split_ffn_adam','dense12_audit_adam','dense12_bdt_canonical','dense12_audit_bdt','dense12_audit_scrambled','dense12_gs_allproj'):
  results[name]=evidence_window(mods,s,device,name,tiny=tiny)
 for lean,audit in [('dense12_split_ffn_adam','dense12_audit_adam'),('dense12_bdt_canonical','dense12_audit_bdt')]:
  if results[lean]!=results[audit]:raise RuntimeError('Lean vs audit output/final states differ: '+lean)
  checks[lean+'_vs_audit']='exact'
 repeat=evidence_window(mods,s,device,'dense12_bdt_canonical',tiny=tiny)
 if repeat!=results['dense12_bdt_canonical']:raise RuntimeError('BDT execution not repeatable')
 resumed=evidence_window(mods,s,device,'dense12_audit_scrambled',tiny=tiny,resume_half=True)
 if resumed!=results['dense12_audit_scrambled']:raise RuntimeError('Interrupted audit restore differs')
 gs_repeat=evidence_window(mods,s,device,'dense12_gs_allproj',tiny=tiny)
 if gs_repeat!=results['dense12_gs_allproj']:raise RuntimeError('GS execution not repeatable')
 if not any(r['raw']!=r['applied'] for r in results['dense12_bdt_canonical']['rows']):raise RuntimeError('BDT benchmark is inactive')
 checks.update(bdt_repeat='exact',scrambled_interrupted_restore='exact',gs_repeat='exact',bdt_active=True)
 # Test GS boundaries using a separate dense-matrix formula, not its own implementation.
 u=torch.arange(36,dtype=torch.float32,device=device).reshape(4,3,3)/19
 A=torch.eye(4,device=device)*.9
 for j in range(3):A[j,j+1]=.05;A[j+1,j]=.05
 A[0,0]+=.05;A[-1,-1]+=.05
 if not torch.allclose(mods['smooth'](u),(A@u.reshape(4,-1)).reshape_as(u),atol=1e-7,rtol=2e-6):raise RuntimeError('GS boundary equation failed')
 checks['gs_boundary_equation']=True
 # Native fused and partition changes must map the SAME named moments before timing.
 for name in ('dense12_native_fused_adam','dense12_split_allproj_adam'):
  m,opt,cfg=factory(mods,s,device,name,tiny)
  if normalized_hash(m,opt)!=s['normalized_identity']['adam']:raise RuntimeError('Named state map failed')
  checks[name+'_mapping']=True;del m,opt;gc.collect()
 return dict(state='passed',checks=checks,records=results,scope='CPU geometry only' if tiny else 'Full-size active-policy exact gate',time=io.now(),source=io.verify(project)[0],runtime=runtime(device),states={str(k):v['common_state_sha'] for k,v in snaps.items()})

def observations(opt):
 if isinstance(opt,torch.optim.AdamW):return {},{}
 return ({q:t.observations for q,t in opt.canonical.items()},{q:t.observations for q,t in opt.scrambled.items()})

def apply_fixed_gradients(model,snap,device):
 for n,p in model.named_parameters():p.grad=snap['fixed_gradients'][n].to(device).clone()

def measure(project,snaps,device,condition,mode,tiny=False):
 if device.type!='cuda':raise RuntimeError('No fabricated CPU timing records')
 mods=io.backend(project);snap=snaps[6 if condition.startswith('dense6_') else 12]
 torch.cuda.reset_peak_memory_stats(device)
 m,opt,cfg=factory(mods,snap,device,condition,tiny)
 updates=20 if mode=='full_step' else 100
 def work(n):
  for i in range(n):
   if mode=='full_step':
    backward(m,cfg,mods,device,snap['batch_ring'][i%20],check=False)
    torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
   step(opt,501+i)
 if mode=='optimizer_only':apply_fixed_gradients(m,snap,device)
 # Initialization, warmup and post-warmup restoration are EXCLUDED.
 ev0=torch.cuda.Event(enable_timing=True);ev1=torch.cuda.Event(enable_timing=True)
 ev0.record();ev1.record();torch.cuda.synchronize(device)
 work(20);torch.cuda.synchronize(device);cold_peak=torch.cuda.max_memory_allocated(device)
 restore(m,opt,snap)
 if mode=='optimizer_only':apply_fixed_gradients(m,snap,device)
 before=observations(opt);torch.cuda.synchronize(device)
 baseline=torch.cuda.memory_allocated(device);reserved_baseline=torch.cuda.memory_reserved(device);torch.cuda.reset_peak_memory_stats(device)
 at=time.perf_counter();ev0.record();work(updates);ev1.record();torch.cuda.synchronize(device);wall=(time.perf_counter()-at)*1000
 event=float(ev0.elapsed_time(ev1));peak=torch.cuda.max_memory_allocated(device);reserved=torch.cuda.max_memory_reserved(device)
 # All correctness, hashing, state/telemetry recording happens AFTER stopping timers.
 after=observations(opt);counts=[];shadows=[]
 for which,arr in ((0,counts),(1,shadows)):
  for q,num in after[which].items():arr.append(num-before[which][q])
 expected=updates//10 if 'bdt' in condition or 'audit_' in condition else 0
 if counts and any(n!=expected for n in counts) or expected and not counts:raise RuntimeError('Covariance cadence mismatch')
 if shadows and any(n!=updates//10 for n in shadows):raise RuntimeError('Shadow cadence mismatch')
 if any(not bool(torch.isfinite(t).all()) for t in m.parameters()):raise FloatingPointError('Nonfinite benchmark model')
 state=normalized(m,opt)
 if any(z['step']!=500+updates or not bool(torch.isfinite(z['m']).all()) or not bool(torch.isfinite(z['v']).all()) for z in state.values()):raise RuntimeError('Benchmark final Adam state invalid')
 r=dict(condition=condition,updates=updates,first_step=501,last_step=500+updates,restored_after_warmup=True,
  common_state_sha=snap['common_state_sha'],input_sha=snap['input_sha'],gradient_sha=snap['gradient_sha'],wall_ms=wall,cuda_ms=event,
  baseline_allocated_bytes=baseline,baseline_reserved_bytes=reserved_baseline,peak_allocated_bytes=peak,peak_reserved_bytes=reserved,incremental_peak_bytes=peak-baseline,incremental_reserved_peak_bytes=reserved-reserved_baseline,
  cold_start_peak_allocated_bytes=cold_peak,profiler_enabled=False,audit_hashes_inside_timer=False,synchronized_boundaries=True,
  covariance_observations_per_projection=expected,observed_refresh_deltas=counts,shadow_refresh_deltas=shadows,
  shadow_matching_enabled=condition in AUDIT,final_model_sha=treehash(m.state_dict()),final_adam_sha=treehash(state),
  tokens_per_second=cfg.tokens_per_update*updates/(wall/1000) if mode=='full_step' else None,parameters=snap['parameters'])
 if not all(math.isfinite(x) and x>0 for x in (wall,event)):raise RuntimeError('Invalid timer')
 del state,opt,m;gc.collect();torch.cuda.empty_cache();return r

def profile_once(project,snap,device):
 # Separate untimed diagnostic. A profiler failure is documented, never a speed sample.
 try:
  mods=io.backend(project);m,opt,cfg=factory(mods,snap,device,'dense12_split_ffn_adam')
  with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
   backward(m,cfg,mods,device,snap['batch_ring'][0],check=False);torch.nn.utils.clip_grad_norm_(m.parameters(),1.);step(opt,501);torch.cuda.synchronize()
  result=dict(state='complete',attention_ops=sorted({e.key for e in prof.key_averages() if 'attention' in e.key.lower()}),not_a_timing_sample=True)
  del m,opt,prof;gc.collect();torch.cuda.empty_cache();return result
 except Exception as e:return dict(state='unavailable',reason=str(e),not_a_timing_sample=True)

def run(project,data,gpu_uuid,telemetry,*,tiny=False,resume=False):
 if tiny:raise RuntimeError('Timing campaign requires actual CUDA')
 device=torch.device('cuda:0');root=io.root(project,'ocb1');src=io.verify(project)[0];snaps={}
 for depth in (12,6):
  p=root/'states'/f'dense{depth}.pt'
  if p.exists():snap=read_state(p)
  else:snap=build_state(project,data,depth,device);save_state(p,snap)
  if snap['source']!=src or snap['runtime']!=runtime(device) or snap['cache']!=data['hashes']:raise RuntimeError('Snapshot context changed')
  snaps[depth]=snap
 gate=root/'gate.json'
 if gate.exists():
  g=io.read(gate)
  if g['state']!='passed' or g['source']!=src or g['gpu_uuid']!=gpu_uuid or g['runtime']!=runtime(device):raise RuntimeError('Gate provenance changed')
  for rel,h in g['evidence'].items():
   if io.sha(io.safe_file(root,rel))!=h:raise RuntimeError('Gate evidence changed')
 else:
  g=correctness(project,snaps,device);g['gpu_uuid']=gpu_uuid
  g['evidence']={str(p.relative_to(root)):io.sha(p) for p in (root/'states').glob('*.json')}
  io.atomic(gate,g)
 print('OCB1 FULL-SIZE ACTIVE-CONTROL GATE: PASS',flush=True)
 if not (root/'profile.json').exists():io.atomic(root/'profile.json',profile_once(project,snaps[12],device))
 rt=io.digest(runtime(device))
 for panel in ('main','instrumentation'):
  for mode in SPEC['measurement_families']:
   for block,order in enumerate(SPEC[panel+'_orders']):
    io.guard_cache(data);io.verify(project);p=root/'blocks'/panel/mode/f'{block:02d}.json'
    if p.exists():
     b=io.read(p)
     if b['gpu_uuid']!=gpu_uuid or b['runtime_sha']!=rt or b['implementation_sha']!=src:raise RuntimeError('Cannot mix timing identities')
     continue
    partial=p.with_suffix('.partial.json')
    if partial.exists():raise RuntimeError('A partial timing block exists. Preserve it; no automatic replacement block.')
    t=telemetry()
    if t['uuid']!=gpu_uuid:raise RuntimeError('Device changed')
    b=dict(state='running',panel=panel,mode=mode,block=block,protocol_sha=io.digest(SPEC),implementation_sha=src,
      runtime_sha=rt,runtime=runtime(device),gpu_uuid=gpu_uuid,cuda_gate_passed=True,measurements=[],telemetry_before=t,time=io.now())
    io.atomic(partial,b)
    for name in ['sentinel_before']+order+['sentinel_after']:
     cond='dense12_split_ffn_adam' if name.startswith('sentinel_') else name
     print('OCB1 MEASURE',panel,mode,block,name,flush=True)
     io.atomic(root/'progress.json',dict(panel=panel,mode=mode,block=block,condition=name,time=io.now()))
     record=measure(project,snaps,device,cond,mode)
     if name.startswith('sentinel_'):b[name]=record
     else:b['measurements'].append(record)
     io.atomic(partial,b)
    b['telemetry_after']=telemetry()
    if b['telemetry_after']['uuid']!=gpu_uuid:raise RuntimeError('Device changed within timing block')
    b['state']='complete';b['finished']=io.now();io.atomic(p,b);partial.unlink()
    print('OCB1 BLOCK COMPLETE',panel,mode,block,flush=True)
 from collect import ocb
 return ocb(project)
