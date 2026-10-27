"""Full-size pretrained growth, a shared post-growth prefix, and six continuations."""
from __future__ import annotations
import copy,json,os,signal,time
from dataclasses import asdict
from pathlib import Path
import torch
import torch.nn.functional as F
import gct_protocol as p
import gct_io as io
from gct_runtime import cpu,treehash,samplehash,rng,restore_rng,runtime,evaluate
from gct_policy import Policy,PROJS
STOP=False

def config(native,data,post,tiny=False):
 return native.ExperimentConfig(d_model=48 if tiny else 768,n_heads=4 if tiny else 12,
  n_layers_or_rounds=(4 if post else 2) if tiny else (12 if post else 6),max_recurrent_rounds=12,
  ffn_hidden=64 if tiny else 2048,vocab_size=97 if tiny else 50257,context_length=16 if tiny else 512,
  micro_batch=2 if tiny else 16,grad_accum=1 if tiny else 4,train_steps=p.POST_STEPS if post else p.PARENT_STEPS,
  learning_rate=.0006 if post else .0008,warmup_steps=200,min_lr_ratio=.1,weight_decay=.1,beta1=.9,beta2=.95,
  grad_clip=1.,precision='fp32' if tiny else 'bf16',data_dir=str(data),output_root='unused',checkpoint_every=250)

def build(native,cfg,seed):
 native.seed_all(seed);m=native.DenseStackLM(cfg);native.initialize_model(m);return m

def identity(model,opt):return dict(model=treehash(model.state_dict()),rng=treehash(rng()),**opt.identity())

def validate_scientific_job(seed,stage,mode,end,gate,tiny,parent,interrupt):
 if gate:
  if seed!=p.GATE_SEED:raise RuntimeError('Engineering gate must use seed127')
  return
 if tiny or seed not in p.SEEDS or interrupt is not None:raise RuntimeError('Unregistered scientific job')
 if stage=='parent':ok=mode=='adamw' and end==p.PARENT_STEPS and parent is None
 elif stage=='prefix':ok=mode=='adamw' and end==p.PREFIX and parent is not None
 elif stage=='arm':ok=mode in p.ARMS and end==p.POST_STEPS and parent is not None
 else:ok=False
 if not ok:raise RuntimeError('Scientific stage/method/budget does not match protocol')

def train(project,directory,seed,stage,mode,end,parent=None,*,gate=False,tiny=False,data=None,resume=False,interrupt=None):
 global STOP
 STOP=False
 def stopped(*args):
  global STOP
  STOP=True
 signal.signal(signal.SIGINT,stopped);signal.signal(signal.SIGTERM,stopped)
 validate_scientific_job(seed,stage,mode,end,gate,tiny,parent,interrupt)
 project=Path(project).resolve();directory=Path(directory).resolve();directory.relative_to(p.root(project))
 mods=io.backend(project);native=mods['native'];device=torch.device('cpu' if tiny else 'cuda:0')
 if not tiny and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):raise RuntimeError('CUDA BF16 unavailable')
 io.guard_cache(data);post=stage!='parent';cfg=config(native,data['path'],post,tiny);source,_=io.verify(project)
 pm=io.read(Path(parent)/'latest.json') if parent else None
 offset=pm['identity']['offset'] if parent else 0
 ct=dict(version=p.VERSION,protocol=p.PROTOCOL_SHA,source=source,dependencies=p.DEPENDENCIES,seed=seed,stage=stage,mode=mode,end=end,
  start=pm['step'] if pm else 0,post=post,offset=offset,tiny=tiny,gate=gate,config=asdict(cfg),runtime=runtime(device),cache=data['hashes'],
  parent_sha=pm['sha256'] if pm else None)
 with io.lock(directory/'.writer.lock'):
  if (directory/'fatal.json').exists():raise RuntimeError('Fatal error requires investigation; not auto-resumed: '+str(directory))
  cp=directory/'contract.json'
  if cp.exists() and io.read(cp)!=ct:raise RuntimeError('Existing contract changed: '+str(directory))
  if not cp.exists():io.atomic(cp,ct)
  if (directory/'result.json').exists():
   r=io.read(directory/'result.json')
   if r['contract']!=ct or r['state']!='complete' or r['step']!=end:raise RuntimeError('Completed result mismatch')
   return r
  own=(directory/'latest.json').exists()
  if (own or (directory/'attempts').exists()) and not resume:raise RuntimeError('Previous attempt exists; inspect then explicit --resume')
  attempt=directory/'attempts'/str(time.time_ns());attempt.mkdir(parents=True)
  model=build(native,cfg,seed).to(device)
  if not tiny and sum(t.numel() for t in model.parameters())!=p.PARAMETERS['post' if post else 'parent']:raise RuntimeError('Parameter count mismatch')
  opt=Policy(model,cfg,mods,mode,device,seed,post,offset)
  start=0;curve=[];history=[];proof=[];timing=dict(training_step_seconds=0.,evaluation_seconds=0.,checkpoint_seconds=0.,updates=0)
  if own or parent:
   ck,meta=io.load_ck(directory if own else parent,torch)
   if own and ck['contract']!=ct:raise RuntimeError('Resume contract changed')
   if not own:
    for key in ('config','runtime','source','cache','seed','post','offset','tiny','gate'):
     if ck['contract'][key]!=ct[key]:raise RuntimeError('Parent context differs: '+key)
    expected=0 if stage=='prefix' else 200
    if ck['step']!=expected:raise RuntimeError('Wrong shared parent phase')
   model.load_state_dict(ck['model']);opt.load(ck['opt']);restore_rng(ck['rng'])
   actual=identity(model,opt)
   if actual!=meta['identity']:raise RuntimeError('Checkpoint tensors/state do not reproduce saved identity')
   initial=ck['initial'] if own else actual;start=ck['step']
   if own:curve,history,proof,timing=ck['curve'],ck['history'],ck['proof'],ck['timing']
   del ck
  else:initial=identity(model,opt)
  ip=directory/'initialization.json'
  if ip.exists() and io.read(ip)!=initial:raise RuntimeError('Initial identity changed')
  if not ip.exists():io.atomic(ip,initial)
  store=native.TokenStore(Path(data['path'])/'train.bin')
  if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
  def checkpoint(step):
   at=time.monotonic();ident=identity(model,opt)
   payload=dict(step=step,contract=ct,model=cpu(model.state_dict()),opt=opt.state(),rng=cpu(rng()),initial=initial,
                curve=curve,history=history,proof=proof,timing=timing)
   meta=io.save_ck(directory,payload,torch,ident,keep=(step==end or (stage=='arm' and step in (1000,2000))))
   timing['checkpoint_seconds']+=time.monotonic()-at
   return meta
  print('[TRAIN]',stage,seed,mode,'phase',start,'to',end,'Adam offset',offset,'params',sum(t.numel() for t in model.parameters()),flush=True)
  try:
   for step in range(start+1,end+1):
    at=time.monotonic();model.train();model.zero_grad(set_to_none=True);rb=treehash(rng())
    global_step=offset+step
    x,y=store.batch(cfg.micro_batch*cfg.grad_accum,cfg.context_length,seed=seed,step=global_step,microstep=0,device=torch.device('cpu'))
    bh=treehash({'x':x,'y':y});lossval=0.
    for i in range(0,len(x),cfg.micro_batch):
     xx=x[i:i+cfg.micro_batch].to(device);yy=y[i:i+cfg.micro_batch].to(device)
     with native.autocast_context(device,cfg.precision):
      z,_=model(xx);loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),yy.reshape(-1))
     if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite training loss')
     (loss/cfg.grad_accum).backward();lossval+=float(loss.detach())/cfg.grad_accum
    ra=treehash(rng());full=step in (1,2,10,50,100,200,201,210,250,300,310,400,420,1000,2000,end)
    gh=treehash({n:t.grad for n,t in model.named_parameters()}) if full else None
    gs=samplehash((n,t.grad) for n,t in model.named_parameters()) if gate else None
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient')
    lr=native.learning_rate(cfg,step);report=opt.step(lr,step,full=full,sampled=gate)
    if treehash(rng())!=ra:raise RuntimeError('Optimizer consumed training RNG')
    row=dict(step=step,global_step=global_step,loss=lossval,grad_norm=float(norm),lr=lr,batch=bh,rng_before=rb,rng_after=ra)
    if gate:row.update(gradient_sample=gs,raw_sample=report['raw_sample'],applied_sample=report['applied_sample'],parameter_sample=samplehash(model.named_parameters()))
    if full:row.update(gradient=gh,raw=report['raw'],applied=report['applied'],model=treehash(model.state_dict()),adam=treehash(opt.named_adam()))
    dt=time.monotonic()-at;timing['training_step_seconds']+=dt;timing['updates']+=1
    eval_step=step in (1,50,100,200,201,400) or step%500==0 or step==end
    if eval_step:
     at=time.monotonic();row['dev']=evaluate(model,Path(data['path'])/'val.bin',cfg,native,device,tiny);timing['evaluation_seconds']+=time.monotonic()-at
    history.append({k:row[k] for k in ('step','global_step','batch','rng_before','rng_after')})
    if gate:proof.append(row.copy())
    if full or step%25==0 or eval_step:
     if full and post and not mode.startswith('reference_'):row['operators']=report['operators']
     curve.append(row.copy())
     io.atomic(directory/'status.json',dict(state='running',stage=stage,seed=seed,method=mode,step=step,global_step=global_step,end=end,pid=os.getpid(),time=io.now()))
     print('GCT1',seed,stage,mode,f'{step}/{end}','age',global_step,'train',round(lossval,6),'dev',row.get('dev',{}).get('loss'),'sec',round(dt,3),flush=True)
    with (attempt/'trace.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n')
    if step%250==0 or step==end or step==interrupt or STOP:checkpoint(step)
    if step==interrupt:return dict(state='test_paused',step=step)
    if STOP:return dict(state='interrupted',step=step)
    if step%100==0:io.guard_cache(data)
   meta=io.read(directory/'latest.json');final=next(r['dev'] for r in reversed(curve) if r['step']==end)
   result=dict(state='complete',contract=ct,contract_sha=p.digest(ct),step=end,initial=initial,final=meta['identity'],checkpoint=meta,
     dev=final,curve=curve,history=history,proof=proof,matching=opt.summary,timing=timing,
     memory=dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device)) if device.type=='cuda' else {},
     scope='engineering only' if gate else 'two-seed DEV transfer pilot; no REPORT and no end-to-end efficiency comparison')
   io.atomic(directory/'result.json',result);io.atomic(directory/'status.json',dict(state='complete',step=end,time=io.now()))
   print('[COMPLETE]',seed,stage,mode,end,flush=True);return result
  except torch.OutOfMemoryError:raise
  except Exception as ex:
   io.atomic(directory/'fatal.json',dict(type=type(ex).__name__,error=str(ex),step=locals().get('step',start),time=io.now()));raise

def source_name(name,parent_layers):
 if name.startswith('blocks.'):
  a=name.split('.');a[1]=str(int(a[1])%parent_layers);return '.'.join(a)
 return name

@torch.no_grad()
def copy_growth(parent_model,parent_opt,target_model,target_opt):
 """Map every weight and Adam moment, verify equality AND distinct storage."""
 layers=len(parent_model.blocks)
 if len(target_model.blocks)!=2*layers:raise RuntimeError('Growth must double depth')
 src=parent_model.state_dict();dst=target_model.state_dict()
 mapped={n:src[source_name(n,layers)].clone() for n in dst}
 target_model.load_state_dict(mapped,strict=True)
 del mapped
 old=parent_opt.named_adam();new=target_opt.named_adam();new['age']=old['age'];new['rest_groups']=copy.deepcopy(old['rest_groups'])
 for n in new['states']:new['states'][n]=cpu(old['states'][source_name(n,layers)])
 target_opt.load_adam(new);del new
 if target_opt.base.step_num!=target_opt.offset or target_opt.phase!=0:raise RuntimeError('Growth failed to preserve global age/reset phase')
 param_ptrs=[];moment_ptrs=[];checked=[];old_named=old['states'];new_named=target_opt.named_adam()['states']
 for n,t in target_model.named_parameters():
  sn=source_name(n,layers)
  if not torch.equal(t.cpu(),src[sn].cpu()):raise RuntimeError('Copied weight mismatch '+n)
  param_ptrs.append(t.data_ptr());a=old_named[sn];b=new_named[n]
  if treehash(a)!=treehash(b):raise RuntimeError('Copied named Adam mismatch '+n)
  if b['kind']=='projection':mt=[b['m'],b['v']]
  else:mt=[b['state'][k] for k in ('exp_avg','exp_avg_sq') if k in b['state']]
  moment_ptrs.extend(x.data_ptr() for x in mt);checked.append(dict(target=n,source=sn))
 if len(set(param_ptrs))!=len(param_ptrs) or len(set(moment_ptrs))!=len(moment_ptrs):raise RuntimeError('Duplicated storage is aliased')
 # Check no target moments/parameters alias the source tensors on the same device.
 old_p={v.data_ptr() for v in parent_model.parameters()}
 old_m={t.data_ptr() for row in old_named.values() for t in ([row['m'],row['v']] if row['kind']=='projection' else [row['state'][k] for k in ('exp_avg','exp_avg_sq') if k in row['state']])}
 if old_p.intersection(param_ptrs) or old_m.intersection(moment_ptrs):raise RuntimeError('Parent and grown storage alias')
 for group in (target_opt.canonical,target_opt.scrambled):
  for t in group.values():
   if t.last_step or t.observations or bool(torch.count_nonzero(t.covariance)) or t.basis.shape!=(2*layers,2*layers-1):raise RuntimeError('Covariance did not reset at growth')
 return dict(mapped_parameters=len(checked),mapping=checked,weights_exact=True,named_moments_exact=True,
   independent_parameter_storage=True,independent_moment_storage=True,parent_age=old['age'],grown_age=target_opt.base.step_num,
   phase=0,next_global_data_step=old['age']+1,covariance_depth_dimension=2*layers-1,both_covariance_histories_zero=True)

def grow(project,directory,seed,parent,*,data,gate=False,tiny=False,resume=False):
 project=Path(project).resolve();directory=Path(directory).resolve();directory.relative_to(p.root(project))
 if (gate and seed!=127) or (not gate and (seed not in p.SEEDS or tiny)):raise RuntimeError('Unregistered growth seed')
 io.guard_cache(data);mods=io.backend(project);native=mods['native'];device=torch.device('cpu' if tiny else 'cuda:0');source,_=io.verify(project)
 ck,pm=io.load_ck(parent,torch);pr=io.read(Path(parent)/'result.json')
 if pr['state']!='complete' or ck['contract']!=pr['contract']:raise RuntimeError('Incomplete small parent')
 if pr['checkpoint']['sha256']!=pm['sha256'] or pr['final']!=pm['identity']:raise RuntimeError('Small parent final link changed')
 if ck['step']!=(p.GATE_PARENT if gate else p.PARENT_STEPS):raise RuntimeError('Wrong parent endpoint')
 cfg=config(native,data['path'],True,tiny);scfg=config(native,data['path'],False,tiny);age=ck['step']
 for k,want in [('config',asdict(scfg)),('source',source),('runtime',runtime(device)),('cache',data['hashes']),('seed',seed),('post',False),('offset',0)]:
  if ck['contract'][k]!=want:raise RuntimeError('Wrong parent provenance '+k)
 ct=dict(version=p.VERSION,protocol=p.PROTOCOL_SHA,source=source,dependencies=p.DEPENDENCIES,seed=seed,stage='grown',mode='adamw',end=0,start=0,
         post=True,offset=age,tiny=tiny,gate=gate,config=asdict(cfg),runtime=runtime(device),cache=data['hashes'],parent_sha=pm['sha256'])
 with io.lock(directory/'.writer.lock'):
  if (directory/'contract.json').exists() and io.read(directory/'contract.json')!=ct:raise RuntimeError('Growth contract changed')
  if (directory/'result.json').exists():
   out=io.read(directory/'result.json');loaded,meta=io.load_ck(directory,torch)
   if loaded['contract']!=ct or out['checkpoint']['sha256']!=meta['sha256']:raise RuntimeError('Grown artifact changed')
   return out
  io.atomic(directory/'contract.json',ct)
  parent_model=build(native,scfg,seed).to(device);parent_opt=Policy(parent_model,scfg,mods,'adamw',device,seed,False)
  parent_model.load_state_dict(ck['model']);parent_opt.load(ck['opt']);restore_rng(ck['rng'])
  if identity(parent_model,parent_opt)!=pm['identity']:raise RuntimeError('Parent does not restore exactly before growth')
  parent_rng=cpu(rng());parent_dev=evaluate(parent_model,Path(data['path'])/'val.bin',scfg,native,device,tiny)
  if parent_dev!=pr['dev']:raise RuntimeError('Parent final DEV not reproducible at growth')
  before=time.monotonic();grown=build(native,cfg,seed).to(device);opt=Policy(grown,cfg,mods,'adamw',device,seed,True,age)
  checks=copy_growth(parent_model,parent_opt,grown,opt);mapping_seconds=time.monotonic()-before
  if not tiny and sum(t.numel() for t in grown.parameters())!=p.PARAMETERS['post']:raise RuntimeError('Wrong grown model size')
  restore_rng(parent_rng);eval_at=time.monotonic();grown_dev=evaluate(grown,Path(data['path'])/'val.bin',cfg,native,device,tiny)
  eval_seconds=time.monotonic()-eval_at
  if treehash(rng())!=treehash(parent_rng):raise RuntimeError('Growth/evaluation changed inherited RNG')
  checks['rng_preserved']=True
  initial=identity(grown,opt)
  meta=io.save_ck(directory,dict(step=0,contract=ct,model=cpu(grown.state_dict()),opt=opt.state(),rng=cpu(rng()),initial=initial,
                 curve=[],history=[],proof=[],timing=dict(training_step_seconds=0.,evaluation_seconds=0.,checkpoint_seconds=0.,updates=0)),torch,initial,keep=True)
  io.atomic(directory/'initialization.json',initial)
  out=dict(state='complete',contract=ct,contract_sha=p.digest(ct),step=0,initial=initial,final=initial,checkpoint=meta,
   checks=checks,parent_dev=parent_dev,grown_dev=grown_dev,growth_loss_jump=grown_dev['loss']-parent_dev['loss'],mapping_seconds=mapping_seconds,evaluation_seconds=eval_seconds,
   note='Stacking is not function preserving. Loss jump is recorded, never used to select/reject parents.')
  io.atomic(directory/'result.json',out);print('[GROWTH VERIFIED]',seed,'age',age,'jump',out['growth_loss_jump'],flush=True)
  return out
