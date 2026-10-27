"""Registered ancestry geometry. Reductions FP64; applied proposals FP32."""
import math
import torch

class UndefinedSector(RuntimeError):pass

def pairs_from_map(ancestor_map):
 labels=sorted(set(ancestor_map))
 if labels!=list(range(len(labels))):raise ValueError('Ancestor labels must be contiguous')
 pairs=[tuple(i for i,x in enumerate(ancestor_map) if x==a) for a in labels]
 if any(len(p)!=2 for p in pairs):raise ValueError('Each ancestor must appear exactly twice')
 return pairs

@torch.no_grad()
def sector_pair(raw,canonical,ancestor_map):
 if raw.shape!=canonical.shape or raw.ndim<2 or raw.shape[0]!=len(ancestor_map):raise ValueError('Sector shape')
 if raw.dtype!=torch.float32 or canonical.dtype!=raw.dtype or raw.device!=canonical.device:raise ValueError('FP32 proposal application required')
 if not all(bool(torch.isfinite(x).all()) for x in (raw,canonical)):raise FloatingPointError('Nonfinite sector proposal')
 pairs=pairs_from_map(ancestor_map)
 x=raw.reshape(len(ancestor_map),-1).double();v=canonical.reshape_as(x).double()
 common=x.mean(0,keepdim=True);u=x-common;vc=v-v.mean(0,keepdim=True)
 swap=list(range(len(ancestor_map)))
 for a,b in pairs:swap[a],swap[b]=b,a
 D=(u-u[swap])/2;A=u-D
 identity_diagnostics=dict(energy_A=float(A.square().sum()),energy_D=float(D.square().sum()),plus=dict(coefficient_A=1.,coefficient_D=1.,canonical_departure_cosine=None,norm_error=0.,common_error=0.,angle_error=0.,displacement_error=0.),minus=dict(coefficient_A=1.,coefficient_D=1.,canonical_departure_cosine=None,norm_error=0.,common_error=0.,angle_error=0.,displacement_error=0.))
 if torch.equal(raw,canonical):return raw.clone(),raw.clone(),dict(kind='exact_identity',target_angle=0.,**identity_diagnostics)
 n=torch.linalg.vector_norm(u);nv=torch.linalg.vector_norm(vc)
 if float(n)==0:
  if float(nv)!=0:raise UndefinedSector('Zero raw with nonzero canonical')
  return raw.clone(),raw.clone(),dict(kind='zero_centered',target_angle=0.,**identity_diagnostics)
 if float(nv)==0 or abs(float(nv/n)-1)>1e-5:raise UndefinedSector('Canonical centered norm invalid')
 along=(u*vc).sum()/n;perp=vc-along/n*u;pn=torch.linalg.vector_norm(perp)
 theta=math.atan2(float(pn),float(along))
 if float(along)<=0:raise UndefinedSector('Canonical beyond 90 degrees')
 if float(pn/nv)<=1e-7:return raw.clone(),raw.clone(),dict(kind='numerical_zero',target_angle=theta,**identity_diagnostics)
 swap=list(range(len(ancestor_map)))
 for a,b in pairs:swap[a],swap[b]=b,a
 D=(u-u[swap])/2;A=u-D;alpha=(D*u).sum()/n.square();q=D-alpha*u;qn=torch.linalg.vector_norm(q)
 if float(qn)<=1e-7*float(n):raise UndefinedSector('Nonzero canonical angle but undefined ancestry-sector tangent; no fallback')
 d=q/qn;coef=float(n)*math.sin(theta)/float(qn);out=[];reports={}
 target_disp=float(torch.linalg.vector_norm(vc-u)/n)
 for label,sign in [('plus',1),('minus',-1)]:
  y=(common+math.cos(theta)*u+sign*n*math.sin(theta)*d).reshape_as(raw).float()
  yd=y.reshape_as(x).double();ym=yd.mean(0,keepdim=True);yc=yd-ym
  ya=(u*yc).sum()/n;yp=yc-ya/n*u
  angle=math.atan2(float(torch.linalg.vector_norm(yp)),float(ya))
  norm_error=abs(float(torch.linalg.vector_norm(yc)/n)-1)
  common_error=float(torch.linalg.vector_norm(ym-common))*math.sqrt(len(ancestor_map))/max(float(torch.linalg.vector_norm(x)),1e-300)
  angle_error=abs(angle-theta);disp=float(torch.linalg.vector_norm(yc-u)/n);de=abs(disp-target_disp)
  if norm_error>1e-5 or common_error>1e-5 or angle_error>5e-6 or de>2e-6+2e-4*target_disp:raise RuntimeError('Applied-cast sector matching failed')
  reports[label]=dict(norm_error=norm_error,common_error=common_error,angle_error=angle_error,displacement_error=de,actual_angle=angle,coefficient_A=math.cos(theta)-sign*coef*float(alpha),coefficient_D=math.cos(theta)+sign*coef*(1-float(alpha)),canonical_departure_cosine=sign*float((d*perp).sum()/pn))
  out.append(y)
 return *out,dict(kind='matched',target_angle=theta,energy_A=float(A.square().sum()),energy_D=float(D.square().sum()),sector_fraction=float(alpha),residual_relative_norm=float(qn/n),**reports)
