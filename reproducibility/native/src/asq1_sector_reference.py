"""Fixed true/false ancestry projectors; no learned or outcome-selected pairing."""
import math
import torch
@torch.no_grad()
def transform(u,phase,rho,false_pair=False,diagnostics=False):
    if u.ndim<2 or u.shape[0]%2 or u.shape[0]<4:raise ValueError('Even depth >=4 required')
    if rho<=0 or not math.isfinite(rho):raise ValueError('Positive finite rho required')
    if phase<=200 or rho==1:return u,{'identity':True,'relative_scale':1.}
    if u.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(u).all()):raise ValueError('Invalid proposal')
    r=u.shape[0];n=r//2;x=u.reshape(r,-1);mean=x.mean(0,keepdim=True);c=x-mean
    right=n+(torch.arange(n,device=u.device)+(1 if false_pair else 0))%n
    pairmean=(c[:n]+c[right])*.5;a=torch.empty_like(c);a[:n]=pairmean;a[right]=pairmean
    d=c-a;strength=max(0.,min(1.,(phase-200)/200));scale=math.exp(strength*math.log(rho))
    z=a+scale*d;z=z-z.mean(0,keepdim=True);nc=torch.linalg.vector_norm(c);nz=torch.linalg.vector_norm(z)
    if float(nc)==0:return u,{'identity':True,'relative_scale':scale}
    if not bool(torch.isfinite(nz)) or float(nz)<=0:raise FloatingPointError('Invalid sector norm')
    out=(mean+z*(nc/nz)).reshape_as(u);rec={'identity':False,'relative_scale':scale}
    if diagnostics:
        oc=out.reshape_as(c)-out.reshape_as(c).mean(0,keepdim=True);on=torch.linalg.vector_norm(oc)
        rec.update(centered_norm_relative_error=abs(float(on/nc)-1),common_max_abs_error=float((out.reshape_as(c).mean(0)-mean[0]).abs().max()),
                   raw_cosine=float((oc*c).sum()/(on*nc)),pair_mean_energy=float((a*a).sum()/(nc*nc)),pair_difference_energy=float((d*d).sum()/(nc*nc)))
        if rec['centered_norm_relative_error']>5e-5:raise FloatingPointError('Sector norm preservation failed')
    return out,rec
@torch.no_grad()
def raw_structure(u):
    x=u.flatten(1);r=len(x);n=r//2;norm=torch.linalg.vector_norm(x,dim=1)
    if bool((norm==0).any()):return {'defined':False}
    gram=(x/norm[:,None])@(x/norm[:,None]).T
    pairs=[float(gram[i,i+n]) for i in range(n)];unrelated=[float(gram[i,j]) for i in range(r) for j in range(i+1,r) if j-i!=n]
    return {'defined':True,'true_pair_mean_cosine':sum(pairs)/n,'unrelated_mean_cosine':sum(unrelated)/len(unrelated),'difference':sum(pairs)/n-sum(unrelated)/len(unrelated)}
