"""Ordinary Dense, exact data indexing, GCT-compatible Adam and timing adapters."""
from __future__ import annotations
import copy
import torch
import torch.nn.functional as F
import common as io
from gct_runtime import cpu, treehash, samplehash, rng, restore_rng, runtime, evaluate, configure
from gct_policy import Policy as AuditPolicy

def config(native,data,layers=12,horizon=3000,peak=.0008,tiny=False):
 return native.ExperimentConfig(d_model=48 if tiny else 768,n_heads=4 if tiny else 12,
  n_layers_or_rounds=(4 if layers==12 else 2) if tiny else layers,max_recurrent_rounds=12,
  ffn_hidden=64 if tiny else 2048,vocab_size=97 if tiny else 50257,context_length=16 if tiny else 512,
  micro_batch=2 if tiny else 16,grad_accum=1 if tiny else 4,train_steps=horizon,
  learning_rate=peak,warmup_steps=200,min_lr_ratio=.1,weight_decay=.1,beta1=.9,beta2=.95,
  grad_clip=1.,precision='fp32' if tiny else 'bf16',data_dir=str(data),output_root='unused',checkpoint_every=250)

def build(mods,cfg,seed,device):
 n=mods['native'];n.seed_all(seed);m=n.DenseStackLM(cfg);n.initialize_model(m);return m.to(device)

def batch(store,cfg,seed,step):
 return store.batch(cfg.micro_batch*cfg.grad_accum,cfg.context_length,seed=seed,step=step,microstep=0,device=torch.device('cpu'))

def backward(model,cfg,mods,device,xy,check=True):
 model.train();model.zero_grad(set_to_none=True);x,y=xy;total=0.
 for i in range(0,len(x),cfg.micro_batch):
  xx=x[i:i+cfg.micro_batch].to(device);yy=y[i:i+cfg.micro_batch].to(device)
  with mods['native'].autocast_context(device,cfg.precision):
   z,_=model(xx);loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),yy.reshape(-1))
  if check and not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite loss')
  (loss/cfg.grad_accum).backward()
  if check:total+=float(loss.detach())/cfg.grad_accum
 return total

def uniform_adam(opt):
 """Partition-neutral, per-parameter Adam state for mapping/verification only."""
 if isinstance(opt,torch.optim.AdamW):raise TypeError('Use normalized(model,opt)')
 s=opt.named_adam();out={}
 for n,r in s['states'].items():
  if r['kind']=='projection':out[n]=dict(m=r['m'],v=r['v'],step=s['age'])
  elif r['state']:out[n]=dict(m=r['state']['exp_avg'],v=r['state']['exp_avg_sq'],step=int(r['state']['step']))
  else:out[n]=dict(m=torch.zeros_like(opt.byname[n]),v=torch.zeros_like(opt.byname[n]),step=0)
 return out

def normalized(model,opt):
 if not isinstance(opt,torch.optim.AdamW):return uniform_adam(opt)
 out={}
 for n,p in model.named_parameters():
  s=opt.state.get(p,{})
  out[n]=dict(m=s.get('exp_avg',torch.zeros_like(p)),v=s.get('exp_avg_sq',torch.zeros_like(p)),step=int(s.get('step',0)))
 return out

def native_optimizer(model,cfg,device):
 return torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused=device.type=='cuda')

@torch.no_grad()
def restore_normalized(model,opt,state):
 names=dict(model.named_parameters())
 if set(state)!=set(names):raise RuntimeError('Normalized state coverage changed')
 ages={v['step'] for v in state.values()}
 if len(ages)!=1:raise RuntimeError('Unequal per-parameter Adam ages')
 age=next(iter(ages))
 for n,s in state.items():
  if type(s['step']) is not int or s['step']<0:raise RuntimeError('Invalid age')
  for k in ('m','v'):
   if s[k].shape!=names[n].shape or s[k].dtype!=names[n].dtype or not bool(torch.isfinite(s[k]).all()):raise RuntimeError('Bad named moment '+n)
  if bool((s['v']<0).any()):raise RuntimeError('Negative second moment')
 if isinstance(opt,torch.optim.AdamW):
  for n,p in names.items():
   s=state[n];d=p.device if opt.defaults.get('fused') else 'cpu'
   opt.state[p]={'step':torch.tensor(float(age),device=d),'exp_avg':s['m'].to(p.device).clone(),'exp_avg_sq':s['v'].to(p.device).clone()}
 else:
  old=opt.named_adam();old['age']=age
  for n,r in old['states'].items():
   s=state[n]
   if r['kind']=='projection':r['m']=s['m'];r['v']=s['v']
   else:r['state']={'step':torch.tensor(float(age)),'exp_avg':s['m'],'exp_avg_sq':s['v']}
  opt.load_adam(old);opt.phase=age

class LeanPolicy(AuditPolicy):
 """Removes only unneeded shadows from GCT1; unchanged canonical arithmetic."""
 def __init__(self,model,cfg,mods,kind,device,seed=127):
  if kind not in ('split_ffn','bdt','split_all','gs_all','observe'):raise ValueError(kind)
  super().__init__(model,cfg,mods,'adamw',device,seed,post=False,age_offset=0)
  self.kind=kind
  if kind in ('split_all','gs_all'):
   for q in ('qkv','out'):self.groups['attention_'+q]=[getattr(b.attn,q).weight for b in model.blocks]
   grouped={id(v) for g in self.groups.values() for v in g}
   self.rest_names=[n for n,v in self.byname.items() if id(v) not in grouped]
   self.base=mods['role'].RoleAdamW(self.groups,betas=(.9,.95),eps=1e-8,weight_decay=.1)
   self.rest=torch.optim.AdamW([self.byname[n] for n in self.rest_names],lr=cfg.learning_rate,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused=device.type=='cuda')
  if kind in ('bdt','observe'):
   tc=mods['temper'].TemperConfig(200,200,10,.9,3.)
   self.canonical={q:mods['temper'].OnlineDepthTemper(len(model.blocks),device,torch.float32,tc) for q in ('gate','up','down')}
  self.nonidentity=False
 @torch.no_grad()
 def step(self,lr,phase,full=False,sampled=False):
  if phase!=self.phase+1 or self.base.step_num!=phase-1:raise RuntimeError('Lean optimizer age mismatch')
  raw=self.base.propose(lr);out={};reports={}
  for q,u in raw.items():
   v=u
   if q in self.canonical:
    v,rep=self.canonical[q].transform(u,phase,active=self.kind=='bdt')
    if abs(rep['centered_norm_ratio']-1)>5e-5 or rep['operator_condition']>3.00005:raise RuntimeError('Canonical invariant')
    if full:reports[q]=rep
    if self.kind=='observe':v=u
   if self.kind=='gs_all':v=self.mods['smooth'](u)
   out[q]=v
  record={}
  if full:record=dict(raw=treehash(raw),applied=treehash(out),operators=reports)
  if sampled:record.update(raw_sample=samplehash(sorted(raw.items())),applied_sample=samplehash(sorted(out.items())))
  self.base.apply(lr,out)
  for g in self.rest.param_groups:g['lr']=lr
  self.rest.step();self.phase=phase
  return record
 def state(self):
  return cpu(dict(adam=self.named_adam(),phase=self.phase,canonical={q:t.state_dict() for q,t in self.canonical.items()},kind=self.kind))
 def load(self,s):
  if s['kind']!=self.kind:raise RuntimeError('Lean policy changed')
  self.load_adam(s['adam']);self.phase=s['phase']
  if self.phase!=self.base.step_num or set(s['canonical'])!=set(self.canonical):raise RuntimeError('Lean state invalid')
  for q,t in self.canonical.items():
   t.load_state_dict(s['canonical'][q])
   if t.last_step!=self.phase:raise RuntimeError('Covariance age changed')

def normalized_hash(model,opt):return treehash(normalized(model,opt))
