"""Versioned training jobs, exact shared-parent forks, and crash-safe resume."""
from __future__ import annotations
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import uuid
import numpy as np
import torch
import cms_protocol as p
import cms_io as io
import cms_runtime as rt
from cms_policy import Policy

_STOP = False


def _stop(signum, frame):
    global _STOP
    _STOP = True


def samples(items):
    names, values = [], []
    for name,value in items:
        if value is None or not value.numel(): continue
        flat = value.detach().reshape(-1); n = flat.numel(); k = min(32,n)
        # Integer-only indexing: floating linspace is unsafe for large embeddings.
        idx = torch.zeros(1,device=flat.device,dtype=torch.int64) if k == 1 else (
            torch.arange(k,device=flat.device,dtype=torch.int64)*(n-1))//(k-1)
        names.append(name); values.append(flat[idx].float().cpu())
    return names, torch.cat(values).numpy() if values else np.empty(0,dtype=np.float32)


class BatchTrace:
    def __init__(self,store,ms): self.store,self.ms,self.last,self.calls = store,ms,None,0
    def batch(self,*args,**kwargs):
        x,y = self.store.batch(*args,**kwargs)
        self.last = self.ms.state_tree_hash({'x':x,'y':y}); self.calls += 1
        return x,y


def instrument(policy,ms,enabled):
    trace = {}; propose,apply = policy.base.propose,policy.base.apply
    def traced_propose(lr):
        result = propose(lr); trace['raw'] = result
        return result
    def traced_apply(lr,result):
        trace['raw_applied_exact'] = all(torch.equal(trace['raw'][q],v) for q,v in result.items())
        if enabled:
            _,value = samples(sorted(result.items())); trace['proposal_sample'] = value
            trace['proposal_sample_sha'] = hashlib.sha256(value.tobytes()).hexdigest()
        if trace.get('full'):
            trace['raw_proposal_sha'] = ms.state_tree_hash(trace['raw'])
            trace['proposal_sha'] = ms.state_tree_hash(result)
        apply(lr,result)
    policy.base.propose = traced_propose; policy.base.apply = traced_apply
    return trace


def identity(model,policy,ms):
    return dict(model=ms.state_tree_hash(model.state_dict()), rng=ms.state_tree_hash(ms.rng_capture()),
                **policy.identity(ms))


def config(ms,data,tiny=False):
    c = ms.make_config('_tiny' if tiny else '124m',str(data),
        micro=2 if tiny else 16, accum=1 if tiny else 4,
        lr=p.LR, precision='fp32' if tiny else 'bf16')
    c.train_steps = p.END
    return c


def train_job(project,directory,seed,mode,end,*,parent=None,tiny=False,gate=False,
              resume=False,interrupt_at=None,data_identity=None):
    """Production jobs are restricted to the registered seed/arm/budget.

    `interrupt_at` is only available to engineering tests; it does not change
    the LR horizon, planned endpoint, or checkpoint contract.
    """
    global _STOP
    _STOP = False
    if not (gate or tiny):
        if seed not in p.SEEDS or mode not in p.ARMS:
            raise RuntimeError('Unregistered production seed/policy')
        if end != (p.PREFIX if parent is None else p.END): raise RuntimeError('Unregistered budget')
        if interrupt_at is not None: raise RuntimeError('Test interruption in production')
    if gate and seed != p.GATE_SEED: raise RuntimeError('Engineering tests must use the existing DEV seed')
    directory, project = Path(directory).resolve(),Path(project).resolve()
    directory.resolve().relative_to(project)
    with io.file_lock(directory/'.writer.lock'):
        if (directory/'fatal.json').exists():
            raise RuntimeError('Prior fatal failure requires inspection: '+str(directory/'fatal.json'))
        mods = rt.backend(project); ms,e = mods['model_support'],mods['engine']
        device = torch.device('cpu' if tiny else 'cuda:0')
        if not tiny and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            raise RuntimeError('Production/gate requires CUDA BF16')
        data_identity = data_identity or io.read_json(p.root(project)/'cache_identity.json')
        io.cache_guard(data_identity)
        data = Path(data_identity['path'])
        cfg = config(ms,data,tiny)
        signature = rt.signature(torch,ms,device)
        source_sha = io.verify_sources(project)
        parent_info = io.read_json(Path(parent)/'latest.json') if parent else None
        start_planned = p.PREFIX if parent else 0
        contract = dict(version=p.VERSION,protocol_sha=p.PROTOCOL_SHA,source_sha=source_sha,
            backend_sha=p.digest(io.read_json(p.HERE/'FROZEN_BACKEND.json')),
            seed=seed,method=mode,kind='continuation' if parent else 'parent',
            start_step=start_planned,end_step=end,config=asdict(cfg),
            data_hashes=data_identity['hashes'],runtime=signature,tiny=tiny,engineering=gate,
            parent_checkpoint_sha=parent_info['sha256'] if parent_info else None,
            endpoint_policy='REPORT only after all complete' if not gate else 'No REPORT in engineering')
        io.write_contract(directory,contract)
        if (directory/'result.json').exists():
            result = io.read_json(directory/'result.json')
            if result['contract'] != contract or result['contract_sha'] != p.digest(contract):
                raise RuntimeError('Completed result contract mismatch')
            print('[already complete]',directory,flush=True); return result
        own_pointer = directory/'latest.json'
        own_exists = own_pointer.exists()
        if own_exists and not resume:
            raise RuntimeError('Committed work exists; use --resume, do not overwrite')
        if not own_exists and (directory/'attempts').exists() and not resume:
            raise RuntimeError('Interrupted attempt exists; use --resume')
        directory.mkdir(parents=True,exist_ok=True)
        attempt = uuid.uuid4().hex[:12]
        attempt_dir = directory/'attempts'/attempt; attempt_dir.mkdir(parents=True)
        io.atomic_json(attempt_dir/'attempt.json',dict(time=io.now(),pid=os.getpid(),
                         resume=own_exists,host=os.uname().nodename))
        signal.signal(signal.SIGTERM,_stop); signal.signal(signal.SIGINT,_stop)
        model = ms.build('_tiny' if tiny else '124m',cfg,seed)
        init_fp = ms.fingerprint(model)
        if not tiny and init_fp['parameters'] != 123550560: raise RuntimeError('Wrong model size')
        model.to(device)
        policy = Policy(model,cfg,mode,device,mods,seed)
        start = 0; curve=[]; history=[]; comparison_rows=[]; arrays={}
        timing = dict(training=0.,evaluation=0.,checkpoint=0.)
        if own_exists or parent:
            source = directory if own_exists else Path(parent)
            ck,ckinfo = io.load_checkpoint(source,torch)
            if own_exists and ck['contract'] != contract:
                raise RuntimeError('Resume contract mismatch')
            if not own_exists:
                if ck['step'] != p.PREFIX or ck['contract']['seed'] != seed:
                    raise RuntimeError('Parent seed/step mismatch')
                if ck['contract']['config'] != contract['config'] or ck['contract']['runtime'] != signature:
                    raise RuntimeError('Parent config/runtime mismatch')
                if ck['contract']['source_sha'] != source_sha or ck['contract']['data_hashes'] != contract['data_hashes']:
                    raise RuntimeError('Parent source/cache mismatch')
            model.load_state_dict(ck['model']); policy.load_state_dict(ck['policy']); ms.rng_restore(ck['rng'])
            actual = identity(model,policy,ms)
            if actual != ckinfo['identity']:
                raise RuntimeError('Checkpoint did not restore exact model/Adam/RNG/operator identity')
            start = int(ck['step'])
            if own_exists:
                curve,history,comparison_rows = ck['curve'],ck['history'],ck['comparison_rows']
                timing = ck['timing']
                if gate or tiny: arrays=ck.get('arrays',{})
                initial = ck['initial']
            else:
                initial = dict(identity=actual,parent_sha=ckinfo['sha256'],parameters=init_fp['parameters'])
            del ck
        else:
            initial = dict(identity=identity(model,policy,ms),parent_sha=None,parameters=init_fp['parameters'])
        ipath=directory/'initialization.json'
        if ipath.exists() and io.read_json(ipath)!=initial: raise RuntimeError('Initial fork identity changed')
        if not ipath.exists(): io.atomic_json(ipath,initial)
        if start > end: raise RuntimeError('Checkpoint beyond registered endpoint')
        trace = instrument(policy,ms,gate or tiny)
        store = BatchTrace(ms.native.TokenStore(data/'train.bin'),ms)
        dev = e.EvaluationPartition(data/'val.bin',p.DEV_OFFSET,4 if tiny else p.EVAL_SEQUENCES,cfg.context_length)
        marks = {1,2,10,50,100,200,201,210,250,300,400,410,end}
        checkpoint_marks = {p.PREFIX,end}
        final_info = None
        def save(step):
            started=time.monotonic(); ident=identity(model,policy,ms)
            payload=dict(contract=contract,step=step,model=rt.cpu_tree(model.state_dict(),torch),
                policy=policy.state_dict(),rng=rt.cpu_tree(ms.rng_capture(),torch),
                initial=initial,curve=curve,history=history,comparison_rows=comparison_rows,
                timing=timing,arrays=arrays if gate or tiny else {})
            info=io.commit_checkpoint(directory,payload,torch,ident)
            timing['checkpoint']+=time.monotonic()-started
            return info
        print('[job]',mode,'seed',seed,'start',start,'end',end,'params',init_fp['parameters'],flush=True)
        try:
            for step in range(start+1,end+1):
                t0=time.monotonic(); model.zero_grad(set_to_none=True)
                before=ms.state_tree_hash(ms.rng_capture()); calls=store.calls
                train_values=e.backward(model,store,cfg,seed,step,device)
                if store.calls!=calls+1: raise RuntimeError('Unexpected batch-drawing pattern')
                after_bw=ms.state_tree_hash(ms.rng_capture())
                full=step in marks
                gh=ms.state_tree_hash({n:t.grad for n,t in model.named_parameters()}) if full else None
                if gate or tiny:
                    _,gpre=samples((n,t.grad) for n,t in model.named_parameters())
                gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip)
                if not bool(torch.isfinite(gn)): raise FloatingPointError('Nonfinite gradient norm')
                if gate or tiny:
                    _,gpost=samples((n,t.grad) for n,t in model.named_parameters())
                trace.clear();trace['full']=full
                lr=ms.native.learning_rate(cfg,step)
                updates=policy.step(lr,step,diagnostic=step in p.DIAG_STEPS or step==end)
                after_step=ms.state_tree_hash(ms.rng_capture())
                if after_step!=after_bw: raise RuntimeError('Optimizer consumed training RNG')
                if step<=p.PREFIX and not trace['raw_applied_exact']:
                    raise RuntimeError('Non-identity warmup')
                if device.type=='cuda': torch.cuda.synchronize()
                seconds=time.monotonic()-t0; timing['training']+=seconds
                row=dict(step=step,lr=lr,grad_norm=float(gn),
                    clip_factor=min(1.,cfg.grad_clip/(float(gn)+1e-6)),
                    batch_hash=store.last,rng_before=before,rng_after_backward=after_bw,
                    rng_after_step=after_step,raw_applied_exact=trace['raw_applied_exact'],**train_values)
                if gate or tiny:
                    _,param=samples(model.named_parameters())
                    for key,value in [('gradient_preclip',gpre),('gradient_postclip',gpost),
                                      ('proposal',trace['proposal_sample']),('parameter',param)]:
                        arrays[f'{step}_{key}']=value
                        row[key+'_sample_sha']=hashlib.sha256(value.tobytes()).hexdigest()
                if full:
                    row.update(gradient_sha=gh,raw_proposal_sha=trace['raw_proposal_sha'],
                               proposal_sha=trace['proposal_sha'],model_sha=ms.state_tree_hash(model.state_dict()),
                               adam_sha=ms.state_tree_hash(policy.adam_state()))
                isval=step in p.VAL_STEPS or step==end
                if isval:
                    tv=time.monotonic();row['dev']=e.evaluate(model,dev,cfg,device)
                    timing['evaluation']+=time.monotonic()-tv
                if updates: row['updates']=updates
                history.append({k:row[k] for k in ('step','batch_hash','rng_before','rng_after_backward','rng_after_step')})
                if gate or tiny: comparison_rows.append({k:v for k,v in row.items() if k!='updates'})
                if isval or step%25==0:
                    curve.append(row)
                    io.atomic_json(directory/'status.json',dict(state='running',step=step,end=end,
                                      method=mode,seed=seed,pid=os.getpid(),time=io.now(),attempt=attempt))
                    print(f'{p.VERSION} seed={seed} {mode} step={step}/{end} train={row["train_task"]:.6f}'
                          f' dev={row.get("dev",{}).get("loss")} sec={seconds:.3f}',flush=True)
                io.append_json(attempt_dir/'trace.jsonl',dict(row,seconds=seconds))
                trace.pop('raw',None)
                if step%p.CHECKPOINT_EVERY==0 or step in checkpoint_marks or step==interrupt_at or _STOP:
                    final_info=save(step)
                if step==interrupt_at:
                    io.atomic_json(directory/'status.json',dict(state='test_interrupted',step=step,end=end))
                    return dict(state='test_interrupted',step=step)
                if _STOP:
                    io.atomic_json(directory/'status.json',dict(state='interrupted',step=step,end=end))
                    return dict(state='interrupted',step=step)
                if step%100==0: io.cache_guard(data_identity)
            if final_info is None:
                final_info=io.read_json(directory/'latest.json')
            final_dev=next(row['dev'] for row in reversed(curve) if row['step']==end)
            if gate or tiny:
                np.savez_compressed(directory/'samples.npz',**arrays)
            result=dict(state='complete',step=end,contract=contract,contract_sha=p.digest(contract),
                initialization=initial,identity_final=final_info['identity'],checkpoint=final_info,
                dev=final_dev,report=None,curve=curve,history=history,
                comparison_rows=comparison_rows if gate or tiny else [],
                matching_summary=policy.summary,timing=timing,
                scope='engineering, not efficacy' if gate or tiny else 'new strict-execution seed-paired cohort')
            io.atomic_json(directory/'result.json',result)
            io.atomic_json(directory/'status.json',dict(state='complete',step=end,method=mode,seed=seed,time=io.now()))
            print('[complete]',mode,seed,end,flush=True)
            return result
        except torch.OutOfMemoryError as ex:
            io.atomic_json(attempt_dir/'failure.json',dict(kind='oom',step=locals().get('step',start),error=str(ex)))
            raise
        except Exception as ex:
            failure=dict(time=io.now(),kind=type(ex).__name__,step=locals().get('step',start),error=str(ex),attempt=attempt)
            io.atomic_json(directory/'fatal.json',failure)
            io.atomic_json(directory/'status.json',dict(state='failed',**failure))
            raise


def report_job(project,directory):
    """Evaluate the frozen final checkpoint; this function never trains."""
    project,directory=Path(project).resolve(),Path(directory).resolve()
    directory.relative_to(p.root(project))
    with io.file_lock(directory/'.writer.lock'):
        run=io.read_json(directory/'result.json');ct=run['contract']
        if ct['engineering'] or ct['tiny'] or ct['end_step']!=p.END: raise RuntimeError('Not a reporting trajectory')
        if ct['seed'] not in p.SEEDS or ct['method'] not in p.ARMS: raise RuntimeError('Unregistered report')
        cohort=io.read_json(p.root(project)/'cohort.lock.json')
        if cohort['protocol_sha']!=p.PROTOCOL_SHA: raise RuntimeError('Wrong locked cohort')
        key=f'{ct["seed"]}/{ct["method"]}'
        if cohort['results'].get(key)!=io.sha(directory/'result.json'): raise RuntimeError('Locked result changed')
        rp=directory/'report.json'
        if rp.exists():
            r=io.read_json(rp)
            if r['result_sha']!=cohort['results'][key]:raise RuntimeError('Existing report changed')
            return r
        mods=rt.backend(project);ms,e=mods['model_support'],mods['engine'];device=torch.device('cuda:0')
        if rt.signature(torch,ms,device)!=ct['runtime']:raise RuntimeError('Reporting runtime differs from training')
        data=io.read_json(p.root(project)/'cache_identity.json');io.cache_guard(data)
        ck,info=io.load_checkpoint(directory,torch)
        if info!=run['checkpoint']:raise RuntimeError('Final checkpoint changed')
        cfg=config(ms,Path(data['path']),False);model=ms.build('124m',cfg,ct['seed']).to(device)
        model.load_state_dict(ck['model']);del ck
        if ms.state_tree_hash(model.state_dict())!=info['identity']['model']:raise RuntimeError('Final model did not reload')
        partition=e.EvaluationPartition(Path(data['path'])/'val.bin',p.REPORT_OFFSET,p.EVAL_SEQUENCES,512)
        value=e.evaluate(model,partition,cfg,device)
        out=dict(state='complete',seed=ct['seed'],method=ct['method'],report=value,
                 result_sha=cohort['results'][key],checkpoint_sha=info['sha256'],protocol_sha=p.PROTOCOL_SHA,
                 runtime=ct['runtime'],source_sha=ct['source_sha'],time=io.now())
        io.atomic_json(rp,out);print('[report]',ct['seed'],ct['method'],value['loss'],flush=True)
        return out
