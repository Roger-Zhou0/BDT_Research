"""Execution helpers retained from the CPU-tested ODC runner; no training policy."""
import copy,hashlib,math,os,random
import numpy as np
import torch
import torch.nn.functional as F
def configure(threads=4):
 if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise RuntimeError('Workspace must be set before torch import')
 torch.set_num_threads(threads);torch.use_deterministic_algorithms(True,warn_only=False)
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.backends.cuda.matmul.allow_tf32=False

def rng():
 return dict(cpu=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,numpy=copy.deepcopy(np.random.get_state()),python=random.getstate())

def restore_rng(x):
 torch.set_rng_state(x['cpu'].cpu());np.random.set_state(x['numpy']);random.setstate(x['python'])
 if x['cuda'] is not None:torch.cuda.set_rng_state_all([s.cpu() for s in x['cuda']])

def cpu(x):
 if torch.is_tensor(x):return x.detach().cpu().clone()
 if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
 if isinstance(x,list):return [cpu(v) for v in x]
 if isinstance(x,tuple):return tuple(cpu(v) for v in x)
 return copy.deepcopy(x)

def treehash(x):
 h=hashlib.sha256()
 def add(z):
  if torch.is_tensor(z):
   t=z.detach().cpu().contiguous();h.update(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes())
  elif isinstance(z,np.ndarray):h.update(str((str(z.dtype),z.shape)).encode());h.update(z.tobytes())
  elif isinstance(z,dict):
   for k in sorted(z,key=lambda v:(type(v).__name__,str(v))):add(k);add(z[k])
  elif isinstance(z,(tuple,list)):
   h.update(type(z).__name__.encode())
   for v in z:add(v)
  else:h.update((type(z).__name__+repr(z)).encode())
  h.update(b'\0')
 add(x);return h.hexdigest()

def samplehash(items):
 parts=[]
 for name,value in items:
  if value is None or not value.numel():continue
  v=value.detach().reshape(-1);n=v.numel();k=min(32,n)
  idx=torch.zeros(1,device=v.device,dtype=torch.int64) if k==1 else torch.arange(k,device=v.device,dtype=torch.int64)*(n-1)//(k-1)
  parts.append((name,v[idx].float().cpu()))
 return treehash(parts)

def runtime(device):
 out=dict(torch=torch.__version__,numpy=np.__version__,device=device.type,threads=torch.get_num_threads(),deterministic=torch.are_deterministic_algorithms_enabled(),warn_only=torch.is_deterministic_algorithms_warn_only_enabled(),workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_tf32=torch.backends.cuda.matmul.allow_tf32)
 if device.type=='cuda':
  out.update(cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=torch.cuda.get_device_name(device),capability=list(torch.cuda.get_device_capability(device)))
 return out

@torch.no_grad()
def evaluate(model,path,cfg,native,device,tiny):
 state=rng();was=model.training;model.eval();total=0.;tokens=0
 nseq=4 if tiny else 2048;ctx=cfg.context_length
 data=np.memmap(path,dtype=np.uint16,mode='r')
 if nseq*(ctx+1)>len(data):raise RuntimeError('DEV partition too short')
 try:
  for i in range(0,nseq,cfg.micro_batch):
   n=min(cfg.micro_batch,nseq-i);start=i*(ctx+1)
   a=np.array(data[start:start+n*(ctx+1)],dtype=np.int64).reshape(n,ctx+1)
   x=torch.from_numpy(a[:,:-1].copy()).to(device);y=torch.from_numpy(a[:,1:].copy()).to(device)
   with native.autocast_context(device,cfg.precision):z,_=model(x);loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1))
   total+=float(loss)*y.numel();tokens+=y.numel()
  value=total/tokens
  if not math.isfinite(value):raise FloatingPointError('Nonfinite DEV')
  return dict(loss=value,tokens=tokens,offset=0,stop=nseq*(ctx+1))
 finally:model.train(was);restore_rng(state)
