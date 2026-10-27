"""AdamW for explicit role weights with exposed pre-decay data proposals."""
from __future__ import annotations
import math,torch

class RoleAdamW:
 def __init__(self,role_params_by_proj,betas=(0.9,0.95),eps=1e-8,weight_decay=0.1):
  self.params=role_params_by_proj;self.beta1,self.beta2=betas;self.eps=float(eps);self.weight_decay=float(weight_decay);self.step_num=0
  self.state={id(p):{'m':torch.zeros_like(p),'v':torch.zeros_like(p)} for ps in self.params.values() for p in ps}
 def state_dict(self):
  return dict(step_num=self.step_num,beta1=self.beta1,beta2=self.beta2,eps=self.eps,weight_decay=self.weight_decay,
              state={str(id(p)):{'m':self.state[id(p)]['m'].cpu(),'v':self.state[id(p)]['v'].cpu()} for ps in self.params.values() for p in ps},
              order={q:[str(id(p)) for p in ps] for q,ps in self.params.items()})
 def load_state_dict(self,d):
  self.step_num=int(d['step_num'])
  for q,ps in self.params.items():
   keys=d['order'][q]
   if len(keys)!=len(ps):raise RuntimeError('role optimizer state shape/order mismatch')
   for p,k in zip(ps,keys):
    self.state[id(p)]['m'].copy_(d['state'][k]['m'].to(p.device));self.state[id(p)]['v'].copy_(d['state'][k]['v'].to(p.device))
 @torch.no_grad()
 def propose(self,lr):
  self.step_num+=1;b1,b2=self.beta1,self.beta2;bc1=1-b1**self.step_num;bc2=1-b2**self.step_num;out={}
  for q,ps in self.params.items():
   us=[]
   for p in ps:
    if p.grad is None:raise RuntimeError(f'missing role gradient in {q}')
    st=self.state[id(p)];st['m'].mul_(b1).add_(p.grad,alpha=1-b1);st['v'].mul_(b2).addcmul_(p.grad,p.grad,value=1-b2)
    mhat=st['m']/bc1;vhat=st['v']/bc2;us.append((-float(lr))*mhat/(vhat.sqrt()+self.eps))
   out[q]=torch.stack(us)
  return out
 @torch.no_grad()
 def apply(self,lr,updates):
  decay=1-float(lr)*self.weight_decay
  for q,ps in self.params.items():
   u=updates[q]
   if len(ps)!=u.shape[0]:raise RuntimeError('role update shape mismatch')
   for r,p in enumerate(ps):p.mul_(decay).add_(u[r])
