"""Block-diagonal depth tempering in whole ancestry sectors, not individual pairs."""
from __future__ import annotations
import math
import torch


def helmert(n,device,dtype):
    q=torch.zeros(n,n-1,device=device,dtype=dtype)
    for j in range(n-1):
        x=math.sqrt((j+1)*(j+2));q[:j+1,j]=1/x;q[j+1,j]=-(j+1)/x
    return q

def sector_bases(roles,device,dtype,false_pair=False):
    if roles<4 or roles%2:raise ValueError('Even depth >=4 required')
    n=roles//2;h=helmert(n,device,dtype);eye=torch.eye(n,device=device,dtype=dtype)
    right=n+(torch.arange(n,device=device)+(1 if false_pair else 0))%n
    m=torch.zeros(roles,n-1,device=device,dtype=dtype);d=torch.zeros(roles,n,device=device,dtype=dtype)
    m[:n]=h/math.sqrt(2);m[right]=h/math.sqrt(2)
    d[:n]=eye/math.sqrt(2);d[right]=-eye/math.sqrt(2)
    return m,d

class SectorConditionedBank:
    """EMA restrictions use the full centered norm; each sector retains its energy."""
    FORMAT='SCB1_projected_prefix_globalnorm_ema_v1'
    def __init__(self,canonical,false_pair,rho):
        if rho not in (1.,2.):raise ValueError('Unregistered rho')
        self.roles=canonical.roles;self.config=canonical.config
        self.false_pair=bool(false_pair);self.rho=float(rho);self.capture=False;self.report={}
        self.last_step=canonical.last_step;self.observations=canonical.observations
        self.bases=sector_bases(self.roles,canonical.basis.device,canonical.basis.dtype,self.false_pair)
        self.covs=[];self.evals=[];self.evecs=[]
        for b in self.bases:
            t=b.T@canonical.basis;c=t@canonical.covariance@t.T
            c=(c+c.T)*.5;ev,q=self._eigh(c)
            self.covs.append(c);self.evals.append(ev);self.evecs.append(q)
    @staticmethod
    def _eigh(c):
        ev,q=torch.linalg.eigh(c.double());scale=max(float(ev.abs().max()),1e-30)
        if float(ev.min()) < -1e-5*scale:raise FloatingPointError('Non-PSD sector covariance')
        return ev.clamp_min(0).to(c.dtype),q.to(c.dtype)
    @torch.no_grad()
    def transform(self,u,phase,active=True):
        if phase!=self.last_step+1:raise RuntimeError('Sector covariance age mismatch')
        b0=self.bases[0]
        if u.ndim<2 or len(u)!=self.roles or u.dtype!=b0.dtype or u.device!=b0.device or not bool(torch.isfinite(u).all()):
            raise ValueError('Invalid proposal shape/device/dtype/values')
        flat=u.reshape(self.roles,-1);common=flat.mean(0,keepdim=True);uc=flat-common;norm=torch.linalg.vector_norm(uc)
        zs=[b.T@uc for b in self.bases]
        if phase%self.config.observation_every==0 and float(norm)>0:
            beta=self.config.covariance_beta
            for k,z in enumerate(zs):
                zn=z/norm;self.covs[k].mul_(beta).add_(zn@zn.T,alpha=1-beta)
                self.covs[k].copy_((self.covs[k]+self.covs[k].T)*.5)
                self.evals[k],self.evecs[k]=self._eigh(self.covs[k])
            self.observations+=1
        self.last_step=phase
        a=max(0.,min(1.,(phase-self.config.warmup_steps)/self.config.ramp_steps)) if active else 0.
        if a==0 or float(norm)==0:
            self.report={'identity':True,'phase':phase,'observations':self.observations}
            return u.clone(),self.report
        outputs=[];raw_norms=[];conditions=[];cosines=[]
        for b,z,ev,q in zip(self.bases,zs,self.evals,self.evecs):
            raw=b@z;nr=torch.linalg.vector_norm(raw);raw_norms.append(nr)
            lmax=float(ev.max())
            if float(nr)==0 or lmax==0:
                outputs.append(raw);conditions.append(1.);cosines.append(None if float(nr)==0 else 1.)
                continue
            gains=(1+(self.config.max_operator_condition**2-1)*a*ev/lmax).rsqrt()
            rotated=b@(q@(gains[:,None]*(q.T@z)));nz=torch.linalg.vector_norm(rotated)
            if not bool(torch.isfinite(nz)) or float(nz)<=0:raise FloatingPointError('Invalid nonzero sector transform')
            rotated*=nr/nz;outputs.append(rotated);conditions.append(float(gains.max()/gains.min()))
            cosines.append(float((raw*rotated).sum()/(nr*nr)))
        rho=math.exp(a*math.log(self.rho));mixed=outputs[0]+rho*outputs[1]
        mixed=mixed-mixed.mean(0,keepdim=True);nz=torch.linalg.vector_norm(mixed)
        if not bool(torch.isfinite(nz)) or float(nz)<=0:raise FloatingPointError('Invalid pooled norm')
        out=(common+mixed*(norm/nz)).reshape_as(u)
        oc=out.reshape_as(flat)-out.reshape_as(flat).mean(0,keepdim=True)
        err=abs(float(torch.linalg.vector_norm(oc)/norm)-1)
        if err>5e-5 or max(conditions)>3.00005:raise FloatingPointError('SCB1 norm/condition failure')
        self.report={'identity':False,'phase':phase,'rho':rho,'observations':self.observations,
                     'centered_norm_relative_error':err,'sector_conditions':conditions}
        if self.capture:
            energies=[float((b.T@oc).square().sum()/norm.square()) for b in self.bases]
            denom=raw_norms[0].square()+rho*rho*raw_norms[1].square()
            wanted=[float(raw_norms[0].square()/denom),float(rho*rho*raw_norms[1].square()/denom)]
            energy_error=max(abs(x-y) for x,y in zip(energies,wanted))
            if energy_error>5e-5:raise FloatingPointError('Within-sector orientation changed intended allocation')
            self.report.update(sector_energy_fractions=energies,target_energy_fractions=wanted,
                               allocation_max_abs_error=energy_error,sector_cosines=cosines,
                               common_max_abs_error=float((out.reshape_as(flat).mean(0)-common[0]).abs().max()))
        return out,self.report
    def state_dict(self):
        return dict(kind=self.FORMAT,roles=self.roles,config=self.config.__dict__.copy(),rho=self.rho,false_pair=self.false_pair,
                    last_step=self.last_step,observations=self.observations,
                    covariances=[x.detach().cpu().clone() for x in self.covs],
                    eigenvalues=[x.detach().cpu().clone() for x in self.evals],
                    eigenvectors=[x.detach().cpu().clone() for x in self.evecs])
    def load_state_dict(self,state):
        expected={'kind','roles','config','rho','false_pair','last_step','observations','covariances','eigenvalues','eigenvectors'}
        if set(state)!=expected:raise RuntimeError('SCB1 state schema mismatch')
        for k in ('kind','roles','config','rho','false_pair'):
            if state[k]!=self.state_dict()[k]:raise RuntimeError('SCB1 resume setting changed: '+k)
        for name,target in [('covariances',self.covs),('eigenvalues',self.evals),('eigenvectors',self.evecs)]:
            if len(state[name])!=2:raise RuntimeError('Missing sector history')
            for src,dst in zip(state[name],target):
                if src.shape!=dst.shape or not bool(torch.isfinite(src).all()):raise RuntimeError('Invalid sector state')
                dst.copy_(src.to(dst))
        if not isinstance(state['last_step'],int) or state['last_step']<0 or not isinstance(state['observations'],int) or state['observations']<0:
            raise RuntimeError('Invalid saved sector age')
        self.last_step=state['last_step'];self.observations=state['observations']
