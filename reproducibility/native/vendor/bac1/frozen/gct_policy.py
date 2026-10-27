"""Dense adapter for unchanged frozen Adam/BDT and CMS1 matching kernels.

Adam uses global age. LR and covariance use a separate post-growth phase age.
No parameter, optimizer, or scientific source in prior studies is modified.
"""
from __future__ import annotations
import copy
import torch
from gct_runtime import cpu,treehash,samplehash
PROJS=('gate','up','down')

class Policy:
 def __init__(self,model,cfg,mods,mode,device,seed,post=False,age_offset=0):
  if mode not in ('adamw','bdt','scrambled_matched','reference_adamw','reference_bdt'):raise ValueError(mode)
  if not post and mode not in ('adamw','reference_adamw'):raise ValueError('Small parent is untreated AdamW')
  self.mode,self.mods,self.post,self.offset,self.phase=mode,mods,post,int(age_offset),0
  self.groups={q:[getattr(b.ffn,q).weight for b in model.blocks] for q in PROJS}
  self.byname=dict(model.named_parameters());self.names={id(v):n for n,v in self.byname.items()}
  grouped={id(v) for g in self.groups.values() for v in g}
  self.rest_names=[n for n,v in self.byname.items() if id(v) not in grouped]
  self.base=mods['role'].RoleAdamW(self.groups,betas=(.9,.95),eps=1e-8,weight_decay=.1)
  self.rest=torch.optim.AdamW([self.byname[n] for n in self.rest_names],lr=cfg.learning_rate,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused=device.type=='cuda')
  self.canonical={};self.scrambled={};self.permutations=None
  if post:
   r=len(model.blocks);tc=mods['temper'].TemperConfig(200,200,10,.9,3.)
   for d in (self.canonical,self.scrambled):
    for q in PROJS:d[q]=mods['temper'].OnlineDepthTemper(r,device,torch.float32,tc)
   self.permutations=mods['geometry'].make_permutations(r,self.groups['gate'][0].shape[0],seed).to(device)
   mods['geometry'].validate_permutations(self.permutations)
  self.summary=dict(active_groups=0,matched_groups=0,identity_groups=0,numerical_zero_groups=0,max_angle_error=0.,max_norm_error=0.,max_common_error=0.,max_displacement_error=0.,max_amplification=0.)
 def named_adam(self):
  states={}
  for ps in self.groups.values():
   for v in ps:states[self.names[id(v)]]=dict(kind='projection',**self.base.state[id(v)])
  for n in self.rest_names:states[n]=dict(kind='rest',state=self.rest.state.get(self.byname[n],{}))
  return dict(age=self.base.step_num,states=states,group_names={q:[self.names[id(v)] for v in g] for q,g in self.groups.items()},rest_names=self.rest_names,
    hyper={k:getattr(self.base,k) for k in ('beta1','beta2','eps','weight_decay')},rest_groups=[{k:copy.deepcopy(v) for k,v in g.items() if k!='params'} for g in self.rest.param_groups])
 @torch.no_grad()
 def load_adam(self,state):
  expected=self.named_adam()
  for k in ('hyper','group_names','rest_names'):
   if state[k]!=expected[k]:raise RuntimeError('Named optimizer setting/partition mismatch: '+k)
  if set(state['states'])!=set(self.byname):raise RuntimeError('Optimizer does not cover all named parameters')
  age=state['age']
  if not isinstance(age,int) or age<0:raise RuntimeError('Invalid Adam age')
  self.base.step_num=age
  for q,ps in self.groups.items():
   for v in ps:
    n=self.names[id(v)];s=state['states'][n]
    if s['kind']!='projection' or set(s)!={'kind','m','v'}:raise RuntimeError('Wrong projection state '+n)
    for k in ('m','v'):
     t=s[k]
     if t.shape!=v.shape or t.dtype!=v.dtype or not bool(torch.isfinite(t).all()):raise RuntimeError('Invalid moment '+n+'/'+k)
     if k=='v' and bool((t<0).any()):raise RuntimeError('Negative second moment')
     self.base.state[id(v)][k].copy_(t.to(v.device))
  for n in self.rest_names:
   v=self.byname[n];s=state['states'][n]
   if s['kind']!='rest':raise RuntimeError('Wrong non-FFN state '+n)
   st=s['state']
   if age and v.requires_grad and (set(st)!={'step','exp_avg','exp_avg_sq'} or int(st['step'])!=age):raise RuntimeError('Missing trainable state or age mismatch '+n)
   self.rest.state[v]={}
   for k,t in st.items():
    if not torch.is_tensor(t) or not bool(torch.isfinite(t).all()):raise RuntimeError('Invalid rest state tensor')
    if k in ('exp_avg','exp_avg_sq') and (t.shape!=v.shape or t.dtype!=v.dtype):raise RuntimeError('Wrong rest moment shape/dtype')
    if k=='exp_avg_sq' and bool((t<0).any()):raise RuntimeError('Negative second moment')
    target=v.device if k!='step' or self.rest.defaults.get('fused',False) else 'cpu'
    self.rest.state[v][k]=t.to(target).clone()
  if len(state['rest_groups'])!=len(self.rest.param_groups):raise RuntimeError('Rest groups changed')
  for old,new in zip(state['rest_groups'],self.rest.param_groups):
   if {k:v for k,v in old.items() if k!='lr'}!={k:v for k,v in new.items() if k not in ('params','lr')}:raise RuntimeError('Adam implementation/settings changed')
   new['lr']=old['lr']
 def state(self):
  return cpu(dict(adam=self.named_adam(),offset=self.offset,phase=self.phase,post=self.post,canonical={q:v.state_dict() for q,v in self.canonical.items()},scrambled={q:v.state_dict() for q,v in self.scrambled.items()},permutations=self.permutations,summary=self.summary))
 def load(self,state):
  if state['post']!=self.post or state['offset']!=self.offset:raise RuntimeError('Global/phase optimizer context changed')
  self.load_adam(state['adam']);self.phase=int(state['phase'])
  if self.base.step_num!=self.offset+self.phase:raise RuntimeError('Adam age != offset + phase')
  if self.post:
   if not torch.equal(state['permutations'].cpu(),self.permutations.cpu()):raise RuntimeError('Permutation identity changed')
   for key in ('canonical','scrambled'):
    if set(state[key])!=set(PROJS):raise RuntimeError('Missing covariance group')
    for q,t in getattr(self,key).items():
     t.load_state_dict(state[key][q])
     # Reference paths deliberately do not advance their unused shadow histories.
     required=self.phase if not self.mode.startswith('reference_') or (self.mode=='reference_bdt' and key=='canonical') else None
     if required is not None and t.last_step!=required:raise RuntimeError('Covariance phase age mismatch')
  elif state['canonical'] or state['scrambled'] or state['permutations'] is not None:raise RuntimeError('Small parent must not have growth covariance')
  self.summary=copy.deepcopy(state['summary'])
 def identity(self):
  return dict(adam=treehash(self.named_adam()),canonical=treehash({q:t.state_dict() for q,t in self.canonical.items()}),scrambled=treehash({q:t.state_dict() for q,t in self.scrambled.items()}),permutations=treehash(self.permutations),phase=self.phase,offset=self.offset)
 @torch.no_grad()
 def step(self,lr,phase,full=False,sampled=False):
  if phase!=self.phase+1 or self.base.step_num!=self.offset+phase-1:raise RuntimeError('Optimizer phase/global age mismatch')
  raw=self.base.propose(lr);out={};reports={}
  for q,u in raw.items():
   if not self.post or self.mode=='reference_adamw':v=u
   elif self.mode=='reference_bdt':v,_=self.canonical[q].transform(u,phase,active=True)
   else:
    can,cr=self.canonical[q].transform(u,phase,active=True)
    transported=self.mods['geometry'].transport(u,self.permutations,q)
    ps,sr=self.scrambled[q].transform(transported,phase,active=True)
    mapped=self.mods['geometry'].transport(ps,self.permutations,q,inverse=True)
    matched,metrics=self.mods['geometry'].match_direction(u,can,mapped)
    if abs(cr['centered_norm_ratio']-1)>5e-5 or cr['operator_condition']>3.00005:raise RuntimeError('Frozen BDT invariant violated')
    v=u if self.mode=='adamw' else can if self.mode=='bdt' else matched
    reports[q]=dict(canonical=cr,scrambled=sr,matching=metrics)
    if phase>200:
     self.summary['active_groups']+=1
     k=metrics['kind'];self.summary['matched_groups' if k=='matched' else 'numerical_zero_groups' if k=='numerical_zero' else 'identity_groups']+=1
     for n in ('angle_error','norm_error','common_error','displacement_error','amplification'):
      if metrics.get(n) is not None:self.summary['max_'+n]=max(self.summary['max_'+n],float(metrics[n]))
   if self.post and phase<=200 and not torch.equal(v,u):raise RuntimeError('Applied treatment during shared Adam prefix')
   out[q]=v
  rec=dict(raw=treehash(raw) if full else None,applied=treehash(out) if full else None,
           raw_sample=samplehash(sorted(raw.items())) if sampled else None,applied_sample=samplehash(sorted(out.items())) if sampled else None,operators=reports)
  self.base.apply(lr,out)
  for g in self.rest.param_groups:g['lr']=lr
  self.rest.step();self.phase=phase
  return rec
