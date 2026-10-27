"""Exact ASQ1 rho=2 kernel applied to adjacent descendants; no BDT consultation."""
import math,torch
from gct_runtime import treehash,samplehash
from asq1_sector_reference import transform as original_transform
GATE_ORACLE=False
ORACLE_CALLS=0
@torch.no_grad()
def transform(raw,phase,diagnostics=False):
 r=raw.shape[0]
 if r!=12 or raw.ndim<2 or raw.dtype!=torch.float32 or not bool(torch.isfinite(raw).all()):raise ValueError('Twelve finite FP32 rows required')
 order=torch.tensor(list(range(0,r,2))+list(range(1,r,2)),device=raw.device);inv=torch.argsort(order)
 if phase<=200:return raw,dict(identity=True,relative_scale=1.,spectral_calls=0)
 out,rec=original_transform(raw.index_select(0,order),phase,2.,False,diagnostics)
 return out.index_select(0,inv),dict(rec,spectral_calls=0,paired_indices='adjacent',rho_target=2.)
@torch.no_grad()
def oracle(raw,phase,out):
 global ORACLE_CALLS
 x=raw.reshape(12,-1).double();H=torch.eye(12,device=x.device,dtype=x.dtype)-torch.ones((12,12),device=x.device,dtype=x.dtype)/12;J=torch.zeros_like(H)
 for j in range(0,12,2):J[j,j+1]=J[j+1,j]=1
 u=H@x;D=(torch.eye(12,device=x.device,dtype=x.dtype)-J)@u/2;M=u-D
 rho=math.exp(max(0.,min(1.,(phase-200)/200))*math.log(2.));v=M+rho*D;v-=v.mean(0,keepdim=True)
 expected=x if phase<=200 or float(u.norm())==0 else x.mean(0,keepdim=True)+v*(u.norm()/v.norm())
 torch.testing.assert_close(out.reshape_as(x).double(),expected,atol=2e-7,rtol=8e-5)
 y=out.reshape_as(x).double();yc=y-y.mean(0,keepdim=True)
 if float(u.norm()) and abs(float(yc.norm()/u.norm())-1)>5e-5:raise RuntimeError('Fixed pooled norm mismatch')
 if float(x.norm()) and float((y.mean(0)-x.mean(0)).norm())*math.sqrt(12)/float(x.norm())>5e-6:raise RuntimeError('Common component mismatch')
 ORACLE_CALLS+=1

def make_policy(Original,model,cfg,mods,device,seed,post,age_offset):
 class FixedPolicy(Original):
  def __init__(self):
   if not post:raise RuntimeError('No fixed-rule parent pretraining')
   super().__init__(model,cfg,mods,'adamw',device,seed,post,age_offset)
   self.mode='reference_adamw';self.applied_arm='fixed_true_double'
  @torch.no_grad()
  def step(self,lr,phase,full=False,sampled=False):
   if phase!=self.phase+1 or self.base.step_num!=self.offset+phase-1:raise RuntimeError('Fixed phase/global age mismatch')
   old=[]
   def forbidden(*a,**k):raise RuntimeError('FRT1 attempted a spectral/oracle call')
   if GATE_ORACLE:
    st=self.state();before=treehash({k:st[k] for k in ('canonical','scrambled','bank')});del st
    for obj,key in [(self.bank,'step'),(mods['temper'].OnlineDepthTemper,'transform'),(mods['geometry'],'match_direction')]:old.append((obj,key,getattr(obj,key)));setattr(obj,key,forbidden)
   try:
    raw=self.base.propose(lr);out={};metrics={}
    for q,u in raw.items():
     v,rec=transform(u,phase,diagnostics=full or GATE_ORACLE)
     if GATE_ORACLE:oracle(u,phase,v)
     out[q]=v;metrics[q]=rec
    rec=dict(raw=treehash(raw) if full else None,applied=treehash(out) if full else None,raw_sample=samplehash(sorted(raw.items())) if sampled else None,applied_sample=samplehash(sorted(out.items())) if sampled else None,operators=metrics)
    self.base.apply(lr,out)
    for g in self.rest.param_groups:g['lr']=lr
    self.rest.step();self.phase=phase
    if GATE_ORACLE:
     st=self.state()
     if treehash({k:st[k] for k in ('canonical','scrambled','bank')})!=before:raise RuntimeError('Fixed inherited observers changed')
    return rec
   finally:
    for obj,key,value in reversed(old):setattr(obj,key,value)
 return FixedPolicy()
