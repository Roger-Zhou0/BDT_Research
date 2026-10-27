import math,torch
@torch.no_grad()
def selective_output(bank,u,phase,mask):
    """Read already-updated covariance; original SCB1 operations except selected bypass."""
    flat=u.reshape(bank.roles,-1);common=flat.mean(0,keepdim=True);uc=flat-common
    norm=torch.linalg.vector_norm(uc)
    a=max(0.,min(1.,(phase-bank.config.warmup_steps)/bank.config.ramp_steps))
    if a==0 or float(norm)==0:return u.clone(),{'identity':True,'phase':phase}
    zs=[b.T@uc for b in bank.bases];outputs=[];raw_norms=[];conditions=[]
    for enabled,b,z,ev,q in zip(mask,bank.bases,zs,bank.evals,bank.evecs):
        raw=b@z;nr=torch.linalg.vector_norm(raw);raw_norms.append(nr)
        lmax=float(ev.max())
        if not enabled or float(nr)==0 or lmax==0:
            outputs.append(raw);conditions.append(1.);continue
        gains=(1+(bank.config.max_operator_condition**2-1)*a*ev/lmax).rsqrt()
        rotated=b@(q@(gains[:,None]*(q.T@z)));nz=torch.linalg.vector_norm(rotated)
        if not bool(torch.isfinite(nz)) or float(nz)<=0:raise FloatingPointError('Invalid sector direction')
        rotated*=nr/nz;outputs.append(rotated);conditions.append(float(gains.max()/gains.min()))
    rho=math.exp(a*math.log(2.));mixed=outputs[0]+rho*outputs[1]
    mixed=mixed-mixed.mean(0,keepdim=True);nz=torch.linalg.vector_norm(mixed)
    if not bool(torch.isfinite(nz)) or float(nz)<=0:raise FloatingPointError('Invalid full-stack norm')
    out=(common+mixed*(norm/nz)).reshape_as(u)
    oc=out.reshape_as(flat)-out.reshape_as(flat).mean(0,keepdim=True)
    err=abs(float(torch.linalg.vector_norm(oc)/norm)-1)
    if err>5e-5 or max(conditions)>3.00005:raise FloatingPointError('SFL1 norm/condition')
    report=dict(identity=False,phase=phase,conditioned_sectors=list(mask),rho=rho,
                centered_norm_relative_error=err,sector_conditions=conditions)
    if bank.capture:
        energies=[float((b.T@oc).square().sum()/norm.square()) for b in bank.bases]
        den=raw_norms[0].square()+rho*rho*raw_norms[1].square()
        wanted=[float(raw_norms[0].square()/den),float(rho*rho*raw_norms[1].square()/den)]
        error=max(abs(x-y) for x,y in zip(energies,wanted))
        if error>5e-5:raise FloatingPointError('Selective orientation changed allocation')
        report.update(sector_energy_fractions=energies,target_energy_fractions=wanted,
                      allocation_max_abs_error=error,common_max_abs_error=float((out.reshape_as(flat).mean(0)-common[0]).abs().max()))
    return out,report

