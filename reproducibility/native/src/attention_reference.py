"""Scope factorial preserving the original native attention Adam moment kernel.

All arms keep the identical optimizer partition. Attention data proposals are
exposed from native, already-updated Adam moments; observations do not change
native untreated outputs. Active attention replaces only its data proposal.
"""
from __future__ import annotations
import copy
import torch
from gct_policy import Policy as ReferencePolicy,PROJS
from gct_runtime import cpu,treehash,samplehash
SCOPES={'adamw':(False,False),'ffn_bdt':(True,False),'attention_bdt':(False,True),'combined_bdt':(True,True)}

class Policy(ReferencePolicy):
 def __init__(self,model,cfg,mods,mode,device,seed,post=False,age_offset=0):
  if mode not in SCOPES:raise ValueError('Unregistered scope '+mode)
  ffn,attn=SCOPES[mode]
  super().__init__(model,cfg,mods,'bdt' if ffn else 'adamw',device,seed,post,age_offset)
  self.scope=mode;self.attn_active=attn
  self.attention_groups={q:[getattr(b.attn,k).weight for b in model.blocks] for q,k in [('attention_qkv','qkv'),('attention_out','out')]}
  # Attention remains in the original native/fused rest optimizer in all arms.
  if any(self.names[id(v)] not in self.rest_names for ps in self.attention_groups.values() for v in ps):raise RuntimeError('Attention partition changed')
  self.attention={}
  if post:
   tc=mods['temper'].TemperConfig(200,200,10,.9,3.)
   self.attention={q:mods['temper'].OnlineDepthTemper(len(model.blocks),device,torch.float32,tc) for q in self.attention_groups}
  self.attention_summary={'active_projection_steps':0,'applied_nonidentity_steps':0,'max_norm_error':0.,'max_condition':1.}
 def state(self):
  s=super().state();s.update(attention={q:t.state_dict() for q,t in self.attention.items()},attention_summary=copy.deepcopy(self.attention_summary),scope_format='native_moments_v1');return cpu(s)
 def load(self,s):
  super().load(s)
  if s.get('scope_format')!='native_moments_v1' or set(s.get('attention',{}))!=set(self.attention):raise RuntimeError('Attention state/format missing or changed')
  for q,t in self.attention.items():
   t.load_state_dict(s['attention'][q])
   if t.last_step!=self.phase:raise RuntimeError('Attention history phase mismatch')
  self.attention_summary=copy.deepcopy(s['attention_summary'])
 def load_legacy(self,s):
  if s['phase']!=0 or s['offset']!=self.offset or not self.post:raise RuntimeError('Only phase-zero grown states may gain new observation histories')
  ReferencePolicy.load(self,s)
  if any(t.last_step or t.observations or bool(torch.count_nonzero(t.covariance)) for t in self.attention.values()):raise RuntimeError('New attention histories must start at zero')
 def identity(self):
  return dict(super().identity(),attention=treehash({q:t.state_dict() for q,t in self.attention.items()}))
 @torch.no_grad()
 def attention_proposals(self,lr):
  age=self.base.step_num;bc1=1-self.base.beta1**age;bc2=1-self.base.beta2**age;out={}
  for q,ps in self.attention_groups.items():
   us=[]
   for v in ps:
    s=self.rest.state[v]
    if set(s)!={'step','exp_avg','exp_avg_sq'} or int(s['step'])!=age:raise RuntimeError('Attention native Adam age/state differs')
    m=s['exp_avg'];v2=s['exp_avg_sq']
    if m.shape!=v.shape or v2.shape!=v.shape or not bool(torch.isfinite(m).all()) or not bool(torch.isfinite(v2).all()) or bool((v2<0).any()):raise RuntimeError('Invalid attention native moments')
    # Same signed, data-only Adam formula as the FFN reference, using NATIVE moments.
    us.append((-float(lr))*(m/bc1)/((v2/bc2).sqrt()+self.base.eps))
   out[q]=torch.stack(us)
  return out
 @torch.no_grad()
 def step(self,lr,phase,full=False,sampled=False):
  before={q:[v.detach().clone() for v in ps] for q,ps in self.attention_groups.items()} if self.post else {}
  # Original FFN transformed application and the original native rest update.
  rec=super().step(lr,phase,full=full,sampled=sampled)
  if not self.post:return rec
  raw=self.attention_proposals(lr);applied={};reports={}
  for q,u in raw.items():
   v,report=self.attention[q].transform(u,phase,active=True)
   err=abs(report['centered_norm_ratio']-1)
   if err>5e-5 or report['operator_condition']>3.00005:raise RuntimeError('Attention canonical BDT invariant failed')
   if phase<=200 and not torch.equal(v,u):raise RuntimeError('Attention treatment active during shared prefix')
   selected=v if self.attn_active else u;applied[q]=selected
   changed=self.attn_active and phase>200 and not torch.equal(v,u)
   if changed:
    for i,param in enumerate(self.attention_groups[q]):
     param.copy_(before[q][i]).mul_(1-float(lr)*self.base.weight_decay).add_(selected[i])
   # Untreated/identity paths retain native rounded output, not a reimplementation.
   if phase>200:
    self.attention_summary['active_projection_steps']+=1
    self.attention_summary['applied_nonidentity_steps']+=int(changed)
    self.attention_summary['max_norm_error']=max(self.attention_summary['max_norm_error'],err)
    self.attention_summary['max_condition']=max(self.attention_summary['max_condition'],float(report['operator_condition']))
   reports[q]=dict(canonical=report,applied=bool(changed))
   if full:
    dims=tuple(range(1,u.ndim));rn=torch.linalg.vector_norm(u.flatten(1),dim=1);vn=torch.linalg.vector_norm(selected.flatten(1),dim=1)
    reports[q].update(raw_layer_norms=rn.cpu().tolist(),applied_layer_norms=vn.cpu().tolist(),layer_displacements=torch.linalg.vector_norm((selected-u).flatten(1),dim=1).cpu().tolist())
  if full:
   rec['ffn_raw']=rec['raw'];rec['ffn_applied']=rec['applied'];rec['attention_raw']=treehash(raw);rec['attention_applied']=treehash(applied)
   rec['raw']=treehash({'ffn':rec['ffn_raw'],'attention':rec['attention_raw']});rec['applied']=treehash({'ffn':rec['ffn_applied'],'attention':rec['attention_applied']})
   rec['operators']['attention']=reports
  if sampled:
   rec['raw_sample']=treehash({'ffn':rec['raw_sample'],'attention':samplehash(sorted(raw.items()))})
   rec['applied_sample']=treehash({'ffn':rec['applied_sample'],'attention':samplehash(sorted(applied.items()))})
  return rec
