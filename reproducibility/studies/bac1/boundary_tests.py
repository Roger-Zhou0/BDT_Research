"""Engineering-only proposal tests at the real GWP1 phase boundaries.

Synthetic state/age fixtures are never written as scientific checkpoints.
The CUDA variant uses full 12x2048x768 projection shapes, not a language-model
training result. Actual model repeat/resume checks are a separate required gate.
"""
from pathlib import Path
import argparse,json,copy
import torch
import common as io

def run(device='cpu'):
 from window_schedule import MODES,strength,collect_history,last_history_step
 from window_temper import OnlineDepthTemper
 tr,rt,mods=io.engine(io.HERE.parent);rt.configure(1)
 ref=mods['temper'];dev=torch.device(device)
 rows=12 if dev.type=='cuda' else 4;hidden=2048 if dev.type=='cuda' else 8;width=768 if dev.type=='cuda' else 3
 gen=torch.Generator(device=dev).manual_seed(81537)
 phases=[200,201,399,400,3000,3001,6200,6201,6399,6400,9000,9001,15000]
 proofs=[]
 for q in ('gate','up','down'):
  shape=(rows,width,hidden) if q=='down' else (rows,hidden,width)
  raw=torch.randn(shape,generator=gen,device=dev,dtype=torch.float32)*torch.arange(1,rows+1,device=dev)[:,None,None]*1e-3
  for mode in MODES:
   for phase in phases:
    cfg=ref.TemperConfig(200,200,10,.9,3.)
    op=OnlineDepthTemper(rows,dev,torch.float32,cfg)
    # Deterministic positive covariance, explicitly synthetic for boundary tests.
    vals=torch.linspace(.01,1.,rows-1,device=dev);op.covariance.copy_(torch.diag(vals));op.eigenvalues.copy_(vals);op.last_step=phase-1
    before=op.covariance.clone();a=strength(mode,phase);observe=collect_history(mode,phase)
    if observe:
     actual,stats=op.transform(raw,phase,active=a>0,strength_override=a)
    else:actual=raw;stats={'strength':0.}
    if a==0:
     if not torch.equal(actual,raw):raise AssertionError('Inactive boundary is not exact Adam proposal')
    else:
     # Independent reference using the unchanged transform with an equivalent
     # ramp origin. History update is unchanged; only the declared ramp differs.
     warm=6200 if mode=='late_bdt' else 200
     reference=ref.OnlineDepthTemper(rows,dev,torch.float32,ref.TemperConfig(warm,200,10,.9,3.))
     reference.covariance.copy_(before);reference.eigenvalues.copy_(vals);reference.last_step=phase-1
     expected,_=reference.transform(raw,phase,active=True)
     if not torch.equal(actual,expected):raise AssertionError('Window transform differs from unchanged reference')
    if not observe and (op.last_step!=phase-1 or not torch.equal(op.covariance,before)):
     raise AssertionError('Stopped history advanced')
    if observe and op.last_step!=phase:raise AssertionError('Observed history missed phase')
    saved=op.state_dict();clone=OnlineDepthTemper(rows,dev,torch.float32,cfg);clone.load_state_dict(saved)
    for k in ('covariance','eigenvalues','eigenvectors'):
     if not torch.equal(getattr(op,k),getattr(clone,k)):raise AssertionError('History round-trip differs')
    proofs.append(dict(projection=q,mode=mode,phase=phase,strength=a,history_observed=observe,reference_exact=True))
 result=dict(state='passed',source=io.verify()[0],device=str(dev),shape_gate=[rows,hidden,width],tests=len(proofs),scope='Synthetic proposal/covariance boundary fixtures, not scientific training or timing',proofs=proofs)
 return result
if __name__=='__main__':
 a=argparse.ArgumentParser();a.add_argument('--device',default='cpu');a.add_argument('--output',type=Path,required=True);x=a.parse_args()
 if io.SPEC['kind']!='window_pilot':raise SystemExit('Window study only')
 v=run(x.device);io.atomic(x.output,v);print('BOUNDARY_TESTS_PASS',v['tests'],v['device'])
