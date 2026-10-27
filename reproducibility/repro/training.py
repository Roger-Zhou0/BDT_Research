"""Portable BCT1 arithmetic driver; not the archived operational coordinator.

Uses unchanged native model, proposal policies, seed-indexed TokenStore, and
exact config/build/copy_growth function excerpts. New output/checkpoint/loop
infrastructure is labelled separately. GPU end-to-end parity is not certified.
"""
from __future__ import annotations
from dataclasses import asdict, replace
import copy
import hashlib
import json
from pathlib import Path
import gc
import os
import random
import time
import numpy as np
from .bootstrap import MODS, ROOT, gct_runtime, gct_policy, bct_policy, configure_cpu
from . import native_training_helpers as helpers
import torch
import torch.nn.functional as F

ARMS = ('bdt_budget_gs', 'structured_budget_gs')
FORMAT = 'portable_bct1_v1_NOT_native_operational_checkpoint'


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def code_identity() -> dict:
    return {'kernel_manifest': digest_file(ROOT/'native/NATIVE_KERNEL_MANIFEST.json'),
            'portable_training': digest_file(ROOT/'repro/training.py'),
            'import_adapter': digest_file(ROOT/'native/src/common.py'),
            'training_helpers': digest_file(ROOT/'repro/native_training_helpers.py'),
            'protocol': digest_file(ROOT/'configs/bct1_historical_protocol.json')}


def save(path: Path, payload: dict) -> None:
    """Only caller-owned output; sidecar binds bytes. Refuse overwrites."""
    if path.exists() or path.with_suffix('.sha256').exists():
        raise FileExistsError(path)
    with path.open('xb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    path.with_suffix('.sha256').write_text(digest_file(path) + '\n')


def load(path: Path) -> dict:
    if digest_file(path) != path.with_suffix('.sha256').read_text().strip():
        raise RuntimeError('Checkpoint checksum mismatch')
    value = torch.load(path, map_location='cpu', weights_only=True)
    if value.get('format') != FORMAT:
        raise RuntimeError('Not a checkpoint produced by this portable driver')
    return value


def safe_rng() -> dict:
    # Tensor/primitives only: permits restricted weights_only=True checkpoint loading.
    n = np.random.get_state()
    return {'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy': [n[0], n[1].tolist(), int(n[2]), int(n[3]), float(n[4])], 'python': random.getstate()}


def restore_rng(s: dict) -> None:
    torch.set_rng_state(s['torch'].cpu())
    if s['cuda']:
        torch.cuda.set_rng_state_all(s['cuda'])
    n = s['numpy']; np.random.set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    random.setstate(s['python'])


def snapshot(model, opt, cfg, seed: int) -> dict:
    return {'format': FORMAT, 'model': gct_runtime.cpu(model.state_dict()), 'opt': opt.state(),
            'rng': gct_runtime.cpu(safe_rng()), 'config': asdict(cfg), 'seed': seed,
            'identity': gct_runtime.treehash({'model': model.state_dict(), 'opt': opt.state()}), 'code_identity': code_identity()}


def restore(model, opt, cfg, seed: int, state: dict) -> None:
    if state['format'] != FORMAT or state['config'] != asdict(cfg) or state['seed'] != seed or state['code_identity'] != code_identity():
        raise ValueError('Checkpoint context differs')
    model.load_state_dict(state['model'], strict=True); opt.load(state['opt']); restore_rng(state['rng'])
    if gct_runtime.treehash({'model': model.state_dict(), 'opt': opt.state()}) != state['identity']:
        raise RuntimeError('Reloaded model/optimizer identity differs')


def run_steps(model, opt, cfg, store, seed, start, stop, trace, *, checkpoint_at=None, out=None):
    if opt.phase != start:
        raise ValueError('Start phase does not match optimizer state')
    device = next(model.parameters()).device
    for step in range(start + 1, stop + 1):
        if step % 100 == 0 and getattr(store,'package_cache_stats',None):
            for name, expected in store.package_cache_stats.items():
                stat=(store.path.parent/name).stat()
                if (stat.st_size,stat.st_mtime_ns,stat.st_ino)!=expected:
                    raise RuntimeError('Token cache changed during reproduction')
        model.train(); model.zero_grad(set_to_none=True)
        global_step = opt.offset + step
        x, y = store.batch(cfg.micro_batch * cfg.grad_accum, cfg.context_length, seed, global_step, 0, torch.device('cpu'))
        task = 0.
        for i in range(0, len(x), cfg.micro_batch):
            xx, yy = x[i:i+cfg.micro_batch].to(device), y[i:i+cfg.micro_batch].to(device)
            with MODS['native'].autocast_context(device, cfg.precision):
                logits, _ = model(xx)
                loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), yy.reshape(-1))
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Nonfinite training loss')
            (loss / cfg.grad_accum).backward(); task += float(loss.detach()) / cfg.grad_accum
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError('Nonfinite gradient norm')
        before_rng = gct_runtime.treehash(gct_runtime.rng())
        lr = MODS['native'].learning_rate(cfg, step)
        opt.step(lr, step, full=False, sampled=False)
        if before_rng != gct_runtime.treehash(gct_runtime.rng()):
            raise RuntimeError('Optimizer consumed training RNG')
        row = {'phase': step, 'global_step': global_step, 'lr': lr, 'training_loss': task,
               'batch_sha256': gct_runtime.treehash({'x': x, 'y': y})}
        trace.append(row)
        if checkpoint_at == step and out is not None:
            save(out, snapshot(model, opt, cfg, seed))
    return trace


def make_config(data, post, tiny):
    cfg = helpers.config(MODS['native'], data, post, tiny=tiny)
    # Native BCT adapter keeps twelve child / six parent rows even in engineering mode.
    return replace(cfg, n_layers_or_rounds=12 if post else 6)


def check_exact_cache(path: Path) -> dict:
    spec = json.loads((ROOT / 'configs/bct1_historical_protocol.json').read_text())
    for name, size in [('train.bin', 6_000_000_000), ('val.bin', 10_000_000)]:
        if (path / name).stat().st_size != size:
            raise ValueError(f'{name}: wrong byte count')
    hashes = {n: digest_file(path / n) for n in spec['cache_sha256']}
    if hashes != spec['cache_sha256']:
        raise ValueError('Not the recorded S1-B cache; a generic FineWeb-Edu sample is not interchangeable')
    return hashes


@torch.no_grad()
def report_evaluate(model, cfg, path: Path, offset=2_500_000) -> dict:
    """Fixed contiguous REPORT; separate from synthetic smoke and training data."""
    data = np.memmap(path, dtype=np.uint16, mode='r')
    nseq, ctx = 2048, cfg.context_length
    if offset + nseq * (ctx+1) > len(data):
        raise ValueError('REPORT slice unavailable')
    saved = safe_rng(); was = model.training; model.eval(); total=0.; tokens=0
    device=next(model.parameters()).device
    try:
        for i in range(0, nseq, cfg.micro_batch):
            n=min(cfg.micro_batch,nseq-i); begin=offset+i*(ctx+1)
            a=np.array(data[begin:begin+n*(ctx+1)],dtype=np.int64).reshape(n,ctx+1)
            x=torch.from_numpy(a[:,:-1].copy()).to(device); y=torch.from_numpy(a[:,1:].copy()).to(device)
            with MODS['native'].autocast_context(device,cfg.precision):
                z,_=model(x); loss=F.cross_entropy(z.reshape(-1,z.shape[-1]),y.reshape(-1))
            total += float(loss)*y.numel(); tokens += y.numel()
        return {'loss':total/tokens,'targets':tokens,'offset':offset,'sequences':nseq}
    finally:
        model.train(was);restore_rng(saved)


def pipeline(out: Path, *, data: Path | None = None, seed=127, smoke=True, resume_check=True) -> dict:
    """Explicit local reproduction, never attaches to archived experiment workers.

    In full mode, trains ONE historical family only; no automatic REPORT is
    evaluated. Cohort analysis must wait for all four families/eight endpoints.
    """
    out = out.resolve()
    if out.exists():
        raise FileExistsError('Choose a new output directory; no automatic overwrite/resume')
    if out.is_relative_to(ROOT/'native') or out.is_relative_to(ROOT/'data'):
        raise ValueError('Output must not replace packaged sources or evidence')
    if not smoke:
        if seed not in (946201,946202,946203,946204):
            raise ValueError('Use a BCT1 family identifier for this recipe')
        if data is None or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('Full recipe requires supplied exact data and CUDA BF16')
        cache=check_exact_cache(data)
    else:
        if seed != 127:raise ValueError('Smoke uses engineering seed127 only')
        cache={'kind':'synthetic token IDs, not paper evidence'}
    configure_cpu(1 if smoke else 4);out.mkdir(parents=True)
    if smoke:
        data=out/'synthetic';data.mkdir()
        np.random.default_rng(1729).integers(0,97,16384,dtype=np.uint16).tofile(data/'train.bin')
    device=torch.device('cpu' if smoke else 'cuda:0')
    cfgp=make_config(data,False,smoke);cfg=make_config(data,True,smoke)
    parent_steps=4 if smoke else 7630; end=420 if smoke else 3000
    # Smoke retains the numerical warmup/ramp, but not scientific size or parent age.
    start_time=time.monotonic()
    parent=helpers.build(MODS['native'],cfgp,seed).to(device)
    parent_opt=gct_policy.Policy(parent,cfgp,MODS,'adamw',device,seed,post=False,age_offset=0)
    store=MODS['native'].TokenStore(data/'train.bin');parent_trace=[]
    if not smoke:
        store.package_cache_stats={name:((data/name).stat().st_size,(data/name).stat().st_mtime_ns,(data/name).stat().st_ino) for name in cache}
    run_steps(parent,parent_opt,cfgp,store,seed,0,parent_steps,parent_trace)
    inherited_rng=safe_rng()
    model=helpers.build(MODS['native'],cfg,seed).to(device)
    warm=bct_policy.Policy(model,cfg,MODS,'warmup',device,seed,post=True,age_offset=parent_steps)
    growth=helpers.copy_growth(parent,parent_opt,model,warm)
    restore_rng(inherited_rng)
    if gct_runtime.treehash(safe_rng())!=gct_runtime.treehash(inherited_rng):
        raise RuntimeError('Growth changed inherited random state')
    growth['rng_preserved']=True
    del parent,parent_opt;gc.collect()
    prefix_trace=[];run_steps(model,warm,cfg,store,seed,0,200,prefix_trace)
    prefix=snapshot(model,warm,cfg,seed);save(out/'prefix.pt',prefix)
    parameter_count=sum(p.numel() for p in model.parameters())
    if not smoke and parameter_count!=123551232:raise RuntimeError('Model parameter count mismatch')
    del model,warm;gc.collect()
    spec=json.loads((ROOT/'configs/bct1_historical_protocol.json').read_text())
    order=ARMS if smoke else spec['arm_order'][str(seed)]
    endpoints={};resume={};first_batches={}
    for arm in order:
        model=helpers.build(MODS['native'],cfg,seed).to(device)
        opt=bct_policy.Policy(model,cfg,MODS,arm,device,seed,post=True,age_offset=parent_steps)
        restore(model,opt,cfg,seed,prefix)
        rows=[]
        run_steps(model,opt,cfg,store,seed,200,end,rows,checkpoint_at=310 if resume_check else None,out=out/(arm+'_310.pt'))
        endpoint=snapshot(model,opt,cfg,seed);save(out/(arm+'_endpoint.pt'),endpoint)
        endpoints[arm]={'phase':end,'identity':endpoint['identity'],'checkpoint_sha256':digest_file(out/(arm+'_endpoint.pt'))}
        first_batches[arm]=rows[0]['batch_sha256']
        (out/(arm+'_trace.json')).write_text(json.dumps(rows,indent=2)+'\n')
        if smoke and resume_check:
            # A mid-treatment resume must recover BOTH own-state candidate histories.
            state=load(out/(arm+'_310.pt'))
            restore(model,opt,cfg,seed,state);replay=[]
            run_steps(model,opt,cfg,store,seed,310,end,replay)
            actual=snapshot(model,opt,cfg,seed)
            resume[arm]={'from_phase':310,'to_phase':end,'exact_model_optimizer_match':actual['identity']==endpoint['identity'],
                         'exact_trace_match':replay==[r for r in rows if r['phase']>310]}
            if not all(resume[arm][k] for k in ('exact_model_optimizer_match','exact_trace_match')):
                raise RuntimeError('Resume differs from uninterrupted CPU execution')
        del model,opt;gc.collect()
    if len(set(first_batches.values()))!=1:raise RuntimeError('Arm fork data differ')
    result={'format':FORMAT,'mode':'CPU_SYNTHETIC_SMOKE' if smoke else 'UNVALIDATED_GPU_PORTABLE_REPRODUCTION',
            'seed':seed,'parameters':parameter_count,'parent_updates':parent_steps,'prefix_updates':200,
            'continuation_end_phase':end,'growth':{k:v for k,v in growth.items() if k!='mapping'},
            'endpoints':endpoints,'resume_checks':resume,'same_first_fork_batch':True,'cache':cache,
            'seconds':time.monotonic()-start_time,'runtime':gct_runtime.runtime(device),
            'paper_efficacy_evaluations':0,'global_REPORT_barrier_released':False,'code_identity':code_identity()}
    (out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');return result
