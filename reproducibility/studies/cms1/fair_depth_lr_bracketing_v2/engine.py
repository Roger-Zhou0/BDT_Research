from __future__ import annotations
import copy,json,math,os,time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import model_support as ms
import protocol as p
from optimizers import PolicyOptimizer

class EvaluationPartition:
    """Deterministic nonoverlapping 513-token records in a fixed cache partition."""
    def __init__(self,path,offset,sequences=2048,context=512):
        self.a=np.memmap(path,dtype=np.uint16,mode='r')
        self.offset,self.sequences,self.context=int(offset),int(sequences),int(context)
        self.stop=self.offset+self.sequences*(self.context+1)
        if self.stop>len(self.a):raise ValueError('Evaluation partition outside cache')
    def batches(self,micro,device):
        for i in range(0,self.sequences,micro):
            n=min(micro,self.sequences-i)
            start=self.offset+i*(self.context+1)
            a=np.array(self.a[start:start+n*(self.context+1)],dtype=np.int64).reshape(n,self.context+1)
            yield torch.from_numpy(a[:,:-1].copy()).to(device),torch.from_numpy(a[:,1:].copy()).to(device)

@torch.no_grad()
def evaluate(model,partition,cfg,device):
    was=model.training; model.eval();total=0.;tokens=0
    try:
        with ms.preserved_rng():
            for x,y in partition.batches(cfg.micro_batch,device):
                with ms.native.autocast_context(device,cfg.precision):
                    z,_=model(x,stochastic=False,sparse_eval=False)
                    loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1))
                total+=float(loss)*y.numel();tokens+=y.numel()
        value=total/tokens
        if not math.isfinite(value):raise FloatingPointError('non-finite validation loss')
        return dict(loss=value,tokens=tokens,partition_start=partition.offset,partition_stop=partition.stop)
    finally:model.train(was)

def backward(model,store,cfg,seed,step,device):
    model.train()
    x,y=store.batch(cfg.micro_batch*cfg.grad_accum,cfg.context_length,seed=seed,step=step,microstep=0,device=torch.device('cpu'))
    tasktotal=auxtotal=0.
    for i in range(0,len(x),cfg.micro_batch):
        xx=x[i:i+cfg.micro_batch].to(device);yy=y[i:i+cfg.micro_batch].to(device)
        with ms.native.autocast_context(device,cfg.precision):
            z,d=model(xx,tau=ms.native.router_tau(cfg,step),stochastic=True,sparse_eval=False)
            task=F.cross_entropy(z.reshape(-1,z.shape[-1]),yy.reshape(-1));aux=d['auxiliary_loss']
            loss=task+aux
        if not bool(torch.isfinite(loss)):raise FloatingPointError('non-finite training loss')
        (loss/cfg.grad_accum).backward()
        tasktotal+=float(task.detach())/cfg.grad_accum;auxtotal+=float(aux.detach())/cfg.grad_accum
    return dict(train_task=tasktotal,train_aux=auxtotal)

def atomic_torch(path,obj):
    path=Path(path);tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as f:
            torch.save(obj,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if tmp.exists():tmp.unlink()

def save_checkpoint(path,model,opt,contract,step,curve,initial,timing):
    atomic_torch(path,dict(version=p.VERSION,contract=contract,step=step,
                          model={k:v.detach().cpu() for k,v in model.state_dict().items()},
                          optimizer=opt.state_dict(),rng=ms.rng_capture(),curve=curve,
                          initial=initial,timing=timing))

def train(project,stage,seed,method,lr,data,device,*,test_cfg=None,test_root=None,test_steps=None,interrupt_at=None):
    size='124m' if test_cfg is None else '_tiny'
    cfg=ms.make_config(size,data['path'],lr=lr) if test_cfg is None else copy.deepcopy(test_cfg)
    cfg.learning_rate=lr
    steps=p.STEPS if test_steps is None else int(test_steps)
    cfg.train_steps=steps
    if test_cfg is None:
        if cfg.tokens_per_update!=32768 or cfg.precision!='bf16':raise RuntimeError('Production configuration drift')
    rdir=p.trial_dir(project,stage,seed,method,lr) if test_root is None else Path(test_root)
    rdir.mkdir(parents=True,exist_ok=True)
    contract=dict(version=p.VERSION,protocol_sha=p.PROTOCOL_SHA,stage=stage,seed=seed,method=method,
                  lr=lr,config=asdict(cfg),data_hashes=data['hashes'],source_sha256=ms.source_hashes(),
                  runtime=ms.runtime_identity(device),grouping='FFN gate/up/down plus attention qkv/out split identically in all methods')
    cp=rdir/'contract.json';latest=rdir/'latest.pt';result=rdir/'result.json'
    if cp.exists() and json.loads(cp.read_text())!=contract:raise RuntimeError(f'Contract changed: {cp}')
    if result.exists():
        d=json.loads(result.read_text())
        if d['contract_sha256']!=p.digest(contract):raise RuntimeError('Result contract mismatch')
        print('[already complete]',result,flush=True);return d
    if (rdir/'diverged.json').exists():
        d=json.loads((rdir/'diverged.json').read_text())
        if d['contract_sha256']!=p.digest(contract):raise RuntimeError('Divergence contract mismatch')
        print('[numerically divergent recipe; no replacement]',rdir,flush=True);return d
    p.atomic_json(cp,contract)
    model=ms.build(size,cfg,seed)
    initial=ms.fingerprint(model)
    if size=='124m' and initial['parameters']!=123550560:raise RuntimeError(f'Unexpected parameter count {initial}')
    model.to(device)
    opt=PolicyOptimizer(model,cfg,method,device)
    # Complete initial model and RNG identities are compared across methods/LRs at selection.
    initial['pretraining_rng']=ms.state_tree_hash(ms.rng_capture())
    p.atomic_json(rdir/'initialization.json',initial)
    store=ms.native.TokenStore(Path(data['path'])/'train.bin')
    neval=p.EVAL_SEQUENCES if test_cfg is None else 4
    dev=EvaluationPartition(Path(data['path'])/'val.bin',p.DEV_OFFSET,neval,cfg.context_length)
    start=0;curve=[];timing=dict(train_seconds=0.,evaluation_seconds=0.,checkpoint_seconds=0.)
    if latest.exists():
        ck=torch.load(latest,map_location='cpu',weights_only=False)
        if ck['contract']!=contract or ck['initial']!=initial:raise RuntimeError('Resume contract/initialization mismatch')
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);ms.rng_restore(ck['rng'])
        start=int(ck['step']);curve=ck['curve'];timing=ck['timing'];del ck
        print(f'[resume] {stage} seed={seed} {method} lr={lr:g} step={start}',flush=True)
    else:print(f'[start fresh] {stage} seed={seed} {method} lr={lr:g} params={initial["parameters"]:,}',flush=True)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    try:
        for step in range(start+1,steps+1):
            begun=time.monotonic();model.zero_grad(set_to_none=True)
            train_values=backward(model,store,cfg,seed,step,device)
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
            if not bool(torch.isfinite(gn)):raise FloatingPointError('non-finite clipped gradient norm')
            actual_lr=ms.native.learning_rate(cfg,step)
            diagnostic=step in (200,400,1000,steps)
            try:
                rep=opt.step(actual_lr,step,diagnostic=diagnostic)
            except FloatingPointError as ex:
                raise RuntimeError('Optimizer-transform integrity failure; audit before classifying as numerical model divergence: '+str(ex)) from ex
            if device.type=='cuda':torch.cuda.synchronize(device)
            elapsed=time.monotonic()-begun;timing['train_seconds']+=elapsed
            row=dict(step=step,lr=actual_lr,grad_norm=float(gn),seconds=elapsed,**train_values)
            isval=step in p.VAL_STEPS or step==steps
            if isval:
                begun=time.monotonic();row['dev']=evaluate(model,dev,cfg,device);timing['evaluation_seconds']+=time.monotonic()-begun
            if diagnostic:row['updates']=rep
            if isval or step%25==0:
                curve.append(row)
                suffix=f' dev={row["dev"]["loss"]:.7f}' if isval else ''
                print(f'{stage} seed={seed} {method:<11} lr_peak={lr:g} step={step:4}/{steps} task={train_values["train_task"]:.5f}{suffix}',flush=True)
                p.atomic_json(rdir/'status.json',dict(state='running',step=step,method=method,seed=seed))
            if step%250==0 or step==steps or step==interrupt_at:
                begun=time.monotonic();save_checkpoint(latest,model,opt,contract,step,curve,initial,timing)
                timing['checkpoint_seconds']+=time.monotonic()-begun
            if step==interrupt_at:raise RuntimeError('EXPECTED_TEST_INTERRUPTION')
            if step%100==0 and test_cfg is None:ms.quick_cache_guard(data)
        final_dev=next(r['dev'] for r in reversed(curve) if r['step']==steps)
        report=None
        if stage=='report':
            if test_cfg is None:
                lock=json.loads((p.campaign(project)/'selection.lock.json').read_text())
                if lock['protocol_sha']!=p.PROTOCOL_SHA or lock['chosen_lr'][method]!=lr:raise RuntimeError('Reporting recipe is not locked')
                offset=p.REPORT_OFFSET
            else:offset=500
            rp=EvaluationPartition(Path(data['path'])/'val.bin',offset,neval,cfg.context_length)
            begun=time.monotonic();report=evaluate(model,rp,cfg,device);timing['evaluation_seconds']+=time.monotonic()-begun
        memory=dict(peak_allocated=torch.cuda.max_memory_allocated(device),peak_reserved=torch.cuda.max_memory_reserved(device)) if device.type=='cuda' else {}
        out=dict(state='complete',step=steps,contract=contract,contract_sha256=p.digest(contract),initialization=initial,
                 dev=final_dev,report=report,training=dict(curve=curve),timing=timing,memory=memory,
                 scope_note='No shared trained prefix. GS Standard-Proj alpha .1 applies from step 1. No spectral shadow calculation in AdamW. Timing is observational, not an uncontended overhead benchmark.')
        p.atomic_json(result,out);p.atomic_json(rdir/'status.json',dict(state='complete',step=steps,result=str(result)))
        print('[complete]',result,flush=True);return out
    except FloatingPointError as ex:
        out=dict(state='diverged',contract_sha256=p.digest(contract),contract=contract,step=locals().get('step',start),reason=str(ex))
        p.atomic_json(rdir/'diverged.json',out);print('[NUMERICAL DIVERGENCE]',out,flush=True);return out
