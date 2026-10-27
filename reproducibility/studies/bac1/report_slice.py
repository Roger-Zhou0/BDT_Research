"""Deterministic indexed-token evaluation; no online training-data feedback."""
import math
import numpy as np
import torch
import torch.nn.functional as F

def evaluate_slice(model,path,cfg,native,device,offset,nseq):
 raw=np.memmap(path,dtype=np.uint16,mode='r');ctx=cfg.context_length
 if offset<0 or nseq<=0 or offset+nseq*(ctx+1)>len(raw):raise ValueError('Evaluation slice unavailable')
 total=0.;tokens=0;was=model.training;model.eval()
 try:
  with torch.no_grad():
   for i in range(0,nseq,cfg.micro_batch):
    n=min(cfg.micro_batch,nseq-i);start=offset+i*(ctx+1)
    a=np.array(raw[start:start+n*(ctx+1)],dtype=np.int64).reshape(n,ctx+1)
    x=torch.from_numpy(a[:,:-1].copy()).to(device);y=torch.from_numpy(a[:,1:].copy()).to(device)
    with native.autocast_context(device,cfg.precision):z,_=model(x);loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1))
    total+=float(loss)*y.numel();tokens+=y.numel()
  if tokens!=nseq*ctx or not math.isfinite(total):raise RuntimeError('Invalid evaluation reduction')
  return dict(loss=total/tokens,tokens=tokens,offset=offset,stop=offset+nseq*(ctx+1))
 finally:model.train(was)
