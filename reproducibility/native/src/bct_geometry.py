"""Own-state sector-budget/orientation exchange. No post-mix renormalization."""
import torch

def norm(x):
    return torch.linalg.vector_norm(x, dtype=torch.float64)

def sectors(x):
    if x.shape[0]!=12:raise ValueError('Registered adjacent-copy depth12 required')
    pairs=.5*(x[0::2]+x[1::2])
    pairs=pairs-pairs.mean(dim=0,keepdim=True)
    m=pairs.repeat_interleave(2,dim=0)
    d=.5*(x[0::2]-x[1::2])
    d=torch.stack((d,-d),dim=1).flatten(0,1)
    return m,d

@torch.no_grad()
def discrepancy(b,s):
    rows={}
    for name,x,y in zip(('family_mean','descendant_difference'),sectors(b),sectors(s)):
        nb,ns=norm(x),norm(y);dot=(x.double()*y.double()).sum()
        cosine=float((dot/(nb*ns)).clamp(-1,1)) if float(nb)>0 and float(ns)>0 else None
        magnitude=(nb-ns).square();orientation=2*(nb*ns-dot)
        observed=norm(x.double()-y.double()).square()
        residual=abs(float(observed-magnitude-orientation))
        rows[name]=dict(bdt_norm=float(nb),structured_norm=float(ns),cosine=cosine,
          magnitude_discrepancy_sq=float(magnitude),orientation_discrepancy_sq=float(orientation),
          observed_discrepancy_sq=float(observed),decomposition_abs_error=residual)
    return rows

@torch.no_grad()
def exchange(raw,budget,direction,capture=False):
    if raw.ndim<2 or raw.shape[0]!=12 or raw.dtype!=torch.float32 or raw.shape!=budget.shape or raw.shape!=direction.shape:raise RuntimeError('Geometry shape/dtype contract')
    if not all(bool(torch.isfinite(x).all()) for x in (raw,budget,direction)):raise RuntimeError('Nonfinite candidate')
    common=raw.mean(0,keepdim=True);centered=raw-common;original_norm=norm(centered)
    if float(original_norm)==0:
        if not torch.equal(budget,raw) or not torch.equal(direction,raw):raise RuntimeError('Nonidentity zero-centered candidate')
        return raw.clone(),dict(zero_centered=True,fallback_sectors=[],total_norm_error=0.,sector_budget_error=0.,common_error=0.)
    bs=sectors(budget-common);ds=sectors(direction-common)
    components=[];fallback=[];target_norms=[]
    for name,x,y in zip(('family_mean','descendant_difference'),bs,ds):
        nx,ny=norm(x),norm(y);target_norms.append(nx)
        if float(nx)==0:z=torch.zeros_like(y)
        elif float(ny)==0:
            raise RuntimeError('Positive sector budget with undefined GS carrier: '+name)
        else:z=(y/ny.to(y.dtype))*nx.to(y.dtype)
        components.append(z)
    # The budget source has the prescribed total norm; no final normalization may alter its allocation.
    out=common+components[0]+components[1]
    oc=out-common;out_sectors=sectors(oc)
    scale=max(float(original_norm),torch.finfo(torch.float64).tiny)
    total_error=abs(float(norm(oc)-original_norm))/scale
    budget_error=max(abs(float(norm(x)-n))/scale for x,n in zip(out_sectors,target_norms))
    common_error=float(norm(out.mean(0,keepdim=True)-common))/max(float(norm(raw)),scale)
    if max(total_error,budget_error,common_error)>5e-5:raise RuntimeError('Crossover invariant violated')
    row=dict(zero_centered=False,fallback_sectors=fallback,total_norm_error=total_error,sector_budget_error=budget_error,common_error=common_error)
    if capture:
        nb=norm(budget-common);nd=norm(direction-common)
        row.update(target_sector_norms=[float(x) for x in target_norms],budget_centered_norm=float(nb),direction_centered_norm=float(nd),
          centered_cosine_from_Adam=float(((oc.double()*centered.double()).sum()/(norm(oc)*original_norm)).clamp(-1,1)))
    return out,row
