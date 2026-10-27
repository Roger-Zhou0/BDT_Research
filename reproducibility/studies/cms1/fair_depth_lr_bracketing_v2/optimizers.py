"""Three policies over the same split-AdamW implementation.
GS is an Eq.2 reimplementation of arXiv:2606.30813v1, not author software.
No shadow tempering or covariance computation in Adam/GS production arms.
"""
from __future__ import annotations
import torch
from role_adamw import RoleAdamW
from tempering_reference import OnlineDepthTemper,TemperConfig
from protocol import GS_ALPHA

@torch.no_grad()
def window_smooth(u:torch.Tensor,alpha:float=GS_ALPHA)->torch.Tensor:
    if not 0 <= alpha < 1:raise ValueError('alpha must be in [0,1)')
    if u.ndim<2 or u.shape[0]<2:raise ValueError('depth stack required')
    if not bool(torch.isfinite(u).all()):raise FloatingPointError('Non-finite GS proposal')
    if alpha==0:return u.clone()
    # All RHS terms come from unmodified original proposals, not in-place neighbors.
    v=u*(1-alpha)
    v[1:]+=u[:-1]*(alpha/2)
    v[:-1]+=u[1:]*(alpha/2)
    v[0]+=u[0]*(alpha/2)
    v[-1]+=u[-1]*(alpha/2)
    return v

def depth_groups(model):
    groups={q:[getattr(ff,q).weight for ff in model.block.private_ffn.roles]
            for q in ('gate','up','down')}
    for q in ('qkv','out'):
        groups['attention_'+q]=[getattr(a,q).weight for a in model.block.attn_rounds]
    all_ids=[id(p) for ps in groups.values() for p in ps]
    if len(all_ids)!=len(set(all_ids)):raise RuntimeError('Projection aliasing across groups')
    depth=len(model.block.private_ffn.roles)
    for name,ps in groups.items():
        if len(ps)!=depth or len({tuple(p.shape) for p in ps})!=1:
            raise RuntimeError(f'Nonhomologous group {name}')
    ids=set(all_ids)
    rest=[p for p in model.parameters() if id(p) not in ids]
    if sum(p.numel() for p in rest)+sum(p.numel() for ps in groups.values() for p in ps)!=sum(p.numel() for p in model.parameters()):
        raise RuntimeError('Optimizer partition does not cover model exactly once')
    return groups,rest

class PolicyOptimizer:
    def __init__(self,model,cfg,method,device):
        if method not in ('adamw','temper_cap3','gs_window'):raise ValueError(method)
        self.method=method; self.groups,rest=depth_groups(model)
        self.base=RoleAdamW(self.groups,betas=(cfg.beta1,cfg.beta2),eps=1e-8,weight_decay=cfg.weight_decay)
        self.rest=torch.optim.AdamW(rest,lr=cfg.learning_rate,betas=(cfg.beta1,cfg.beta2),eps=1e-8,
                                    weight_decay=cfg.weight_decay,fused=device.type=='cuda')
        self.tempers={}
        if method=='temper_cap3':
            tc=TemperConfig(warmup_steps=200,ramp_steps=200,observation_every=10,
                            covariance_beta=0.9,max_operator_condition=3.)
            self.tempers={q:OnlineDepthTemper(len(ps),device,torch.float32,tc)
                          for q,ps in self.groups.items() if q in ('gate','up','down')}
    @torch.no_grad()
    def step(self,lr,step,diagnostic=False):
        if self.base.step_num!=step-1:raise RuntimeError('Adam step/order mismatch')
        raw=self.base.propose(lr)
        applied={}; reports={}
        for q,u in raw.items():
            if self.method=='temper_cap3' and q in self.tempers:
                v,rep=self.tempers[q].transform(u.float(),step,active=True)
                if abs(rep.get('centered_norm_ratio',1.)-1)>5e-5 or rep.get('operator_condition',1)>3.00005:
                    raise RuntimeError('Frozen cap-3 invariant violation')
                if diagnostic:reports[q]=rep
            elif self.method=='gs_window':
                v=window_smooth(u)
            else:v=u
            applied[q]=v
            if diagnostic:
                uf=u.reshape(u.shape[0],-1);vf=v.reshape(v.shape[0],-1)
                den=float(uf.norm()*vf.norm())
                reports.setdefault(q,{}).update(raw_norm=float(uf.norm()),applied_norm=float(vf.norm()),
                    cosine=float((uf*vf).sum())/den if den>0 else None,
                    scope='FFN' if q in ('gate','up','down') else 'attention')
        self.base.apply(lr,applied)
        for g in self.rest.param_groups:g['lr']=lr
        self.rest.step()
        return reports
    def state_dict(self):
        return dict(method=self.method,base=self.base.state_dict(),rest=self.rest.state_dict(),
                    tempers={q:t.state_dict() for q,t in self.tempers.items()})
    def load_state_dict(self,s):
        if s['method']!=self.method or set(s['tempers'])!=set(self.tempers):raise RuntimeError('Policy state mismatch')
        for k,v in [('beta1',self.base.beta1),('beta2',self.base.beta2),('eps',self.base.eps),('weight_decay',self.base.weight_decay)]:
            if s['base'][k]!=v:raise RuntimeError('Adam configuration mismatch')
        self.base.load_state_dict(s['base']);self.rest.load_state_dict(s['rest'])
        for q,t in self.tempers.items():t.load_state_dict(s['tempers'][q])
