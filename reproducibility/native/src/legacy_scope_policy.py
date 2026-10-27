"""BAA1 named-Adam adapter. Only the FFN applied direction changes."""
from __future__ import annotations
import copy
import torch
from gct_policy import Policy as ReferencePolicy, PROJS
from gct_runtime import cpu,treehash,samplehash
from ancestry import CorrespondenceBank
import common as io

class Policy(ReferencePolicy):
 def __init__(self,model,cfg,mods,mode,device,seed,post=False,age_offset=0):
  if mode not in ('adamw','bdt',*io.SPEC['arms']):raise ValueError('Unregistered BAA1 policy')
  super().__init__(model,cfg,mods,'bdt' if mode=='bdt' else 'adamw',device,seed,post,age_offset)
  self.applied_arm=mode;self.bank=None
  if post:
   self.bank=CorrespondenceBank(len(model.blocks)//2,self.groups['gate'][0].shape[0],seed,device=device)
   self.canonical=self.bank.canonical
  self.chart_summary={'matched_projection_steps':0,'identity_projection_steps':0,'numerical_zero_projection_steps':0,'max_amplification':0.,'temporal_restores':0}
 def state(self):
  s=super().state();s['bank']=self.bank.state() if self.bank else None;s['chart_summary']=copy.deepcopy(self.chart_summary)
  return cpu(s)
 def load(self,s):
  super().load(s)
  if bool(s.get('bank'))!=bool(self.bank):raise RuntimeError('Missing or extra chart histories')
  if self.bank:self.bank.load(s['bank'])
  self.chart_summary=copy.deepcopy(s['chart_summary'])
 def load_legacy(self,s):
  if s['phase']!=0 or s['offset']!=self.offset or not self.bank:raise RuntimeError('New chart histories require a phase-zero grown checkpoint')
  ReferencePolicy.load(self,s)
  if self.bank.phase or any(x.observations or x.last_step for h in self.bank.charts.values() for x in h.values()):raise RuntimeError('Chart history not initially zero')
 def identity(self):
  return dict(super().identity(),bank=treehash(self.bank.state()) if self.bank else treehash(None))
 @torch.no_grad()
 def step(self,lr,phase,full=False,sampled=False):
  if not self.post:return super().step(lr,phase,full=full,sampled=sampled)
  if phase!=self.phase+1 or self.base.step_num!=self.offset+phase-1:raise RuntimeError('Phase/global age mismatch')
  raw=self.base.propose(lr)
  temporal=any(lo<=phase<=hi for lo,hi in io.SPEC['temporal_windows'])
  out,metrics=self.bank.step(raw,phase,self.applied_arm,diagnostics=full,temporal=temporal)
  # Retain the unchanged original shadow chart as well, so baseline state checks
  # include the original GCT1 FFN histories, not only model/Adam output equality.
  for q,u in raw.items():
   can=self.bank.last_canonical[q]
   z=self.mods['geometry'].transport(u,self.permutations,q)
   ps,sr=self.scrambled[q].transform(z,phase,active=True)
   mapped=self.mods['geometry'].transport(ps,self.permutations,q,inverse=True)
   _,old=self.mods['geometry'].match_direction(u,can,mapped)
   metrics[q]['original_shadow']=dict(operator=sr,matching=old)
   if phase>200:
    self.summary['active_groups']+=1;kind=old['kind']
    self.summary['matched_groups' if kind=='matched' else 'numerical_zero_groups' if kind=='numerical_zero' else 'identity_groups']+=1
    for name in ('angle_error','norm_error','common_error','displacement_error','amplification'):
     if old.get(name) is not None:self.summary['max_'+name]=max(self.summary['max_'+name],float(old[name]))
    for report in metrics[q]['charts'].values():
     kind=report['kind'];key='matched_projection_steps' if kind=='matched' else 'numerical_zero_projection_steps' if kind=='numerical_zero' else 'identity_projection_steps'
     self.chart_summary[key]+=1
     if report.get('amplification') is not None:self.chart_summary['max_amplification']=max(self.chart_summary['max_amplification'],report['amplification'])
  rec=dict(raw=treehash(raw) if full else None,applied=treehash(out) if full else None,
    raw_sample=samplehash(sorted(raw.items())) if sampled else None,applied_sample=samplehash(sorted(out.items())) if sampled else None,operators=metrics)
  self.base.apply(lr,out)
  for group in self.rest.param_groups:group['lr']=lr
  self.rest.step();self.phase=phase
  return rec
