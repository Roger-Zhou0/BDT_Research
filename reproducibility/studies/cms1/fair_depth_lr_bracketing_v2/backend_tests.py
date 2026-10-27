#!/usr/bin/env python3
"""Math, model partition, real-engine resume, and selection tests. No production writes."""
from __future__ import annotations
import argparse,copy,json,tempfile
from pathlib import Path
import numpy as np
import torch
import protocol as p
import model_support as ms
from optimizers import PolicyOptimizer,window_smooth,depth_groups
from tempering_reference import OnlineDepthTemper,TemperConfig

def math_gate(device):
    torch.manual_seed(191)
    for R in (2,4,12):
        u=torch.randn(R,23,device=device)
        for alpha in (0.,.1,.2,.7):
            S=torch.eye(R,device=device)*(1-alpha)
            for i in range(R-1):S[i,i+1]=S[i+1,i]=alpha/2
            S[0,0]=S[-1,-1]=1-alpha/2
            v=window_smooth(u,alpha)
            assert torch.allclose(v,S@u,atol=6e-7,rtol=3e-6)
            assert torch.allclose(v.sum(0),u.sum(0),atol=3e-6,rtol=2e-6)
            if alpha==0:assert torch.equal(v,u)
    t=OnlineDepthTemper(12,device,torch.float32,TemperConfig(max_operator_condition=3))
    for step in range(1,410):
        u=torch.randn(12,31,device=device)+torch.arange(12,device=device)[:,None]*torch.randn(1,31,device=device)
        v,r=t.transform(u,step)
        assert r['operator_condition']<=3.00005
        assert abs(r['centered_norm_ratio']-1)<5e-5
        assert (v.mean(0)-u.mean(0)).abs().max()<2e-5
        if step<=200:assert torch.equal(u,v)
    print('[PASS] GS equation, reflecting boundaries, mean preservation; unchanged cap-three invariants')

def identity_gate(device):
    cfg=ms.make_config('_tiny','unused',micro=2,accum=1,precision='bf16' if device.type=='cuda' else 'fp32')
    m=ms.build('_tiny',cfg,17).to(device);ref=copy.deepcopy(m)
    ours=PolicyOptimizer(m,cfg,'adamw',device)
    stock=torch.optim.AdamW(ref.parameters(),lr=.0004,betas=(.9,.95),eps=1e-8,weight_decay=.1,fused=device.type=='cuda')
    for step in range(1,8):
        for a,b in zip(m.parameters(),ref.parameters()):
            g=torch.randn_like(a);a.grad=g.clone();b.grad=g.clone()
        ours.step(.0004,step);stock.step()
    err=max(float((a-b).detach().abs().max()) for a,b in zip(m.parameters(),ref.parameters()))
    assert err<3e-6,err
    # Window alpha=0 plus the same split Adam state must be exactly the identity.
    from role_adamw import RoleAdamW
    m=ms.build('_tiny',cfg,21).to(device);copy_m=copy.deepcopy(m)
    ga,_=depth_groups(m);gb,_=depth_groups(copy_m)
    oa=RoleAdamW(ga);ob=RoleAdamW(gb)
    for ps,qs in zip(ga.values(),gb.values()):
        for a,b in zip(ps,qs):g=torch.randn_like(a);a.grad=g.clone();b.grad=g.clone()
    ua=oa.propose(.0004);ub=ob.propose(.0004);oa.apply(.0004,ua);ob.apply(.0004,{q:window_smooth(v,0) for q,v in ub.items()})
    assert all(torch.equal(a,b) for a,b in zip(m.parameters(),copy_m.parameters()))
    print('[PASS] split Adam vs stock AdamW max error',err,'; GS alpha-zero exact identity')

def workflow_gate(device):
    import engine as e
    cfg=ms.make_config('_tiny','unused',micro=2,accum=1,precision='bf16' if device.type=='cuda' else 'fp32')
    with tempfile.TemporaryDirectory(prefix='fair_optimizer_engine_test_') as tmp:
        root=Path(tmp);data_dir=root/'data';data_dir.mkdir();rng=np.random.default_rng(77)
        for name in ('train.bin','val.bin'):rng.integers(0,97,size=12000,dtype=np.uint16).tofile(data_dir/name)
        data=dict(path=str(data_dir),hashes={'synthetic':'test-only'},stat={})
        cfg.data_dir=str(data_dir)
        # Exercise actual optimizer updates beyond onset, same model inputs and evaluation code.
        initial=[]
        for method in p.METHODS:
            d=root/method
            e.train(root,'development',127,method,.0004,data,device,test_cfg=cfg,test_root=d,test_steps=206)
            r=json.loads((d/'result.json').read_text());initial.append(r['initialization'])
            assert r['step']==206 and r['report'] is None
        assert initial[0]==initial[1]==initial[2]
        # Interrupt after tempering activates; saved moments/covariance and data sequence resume.
        d=root/'resumed'
        try:e.train(root,'development',127,'temper_cap3',.0004,data,device,test_cfg=cfg,test_root=d,test_steps=206,interrupt_at=203)
        except RuntimeError as ex:assert str(ex)=='EXPECTED_TEST_INTERRUPTION'
        else:raise AssertionError('Interruption not exercised')
        e.train(root,'development',127,'temper_cap3',.0004,data,device,test_cfg=cfg,test_root=d,test_steps=206)
        original=torch.load(root/'temper_cap3/latest.pt',map_location='cpu',weights_only=False)
        resumed=torch.load(d/'latest.pt',map_location='cpu',weights_only=False)
        # GPU kernels need not replay bitwise; CPU path must replay exactly.
        maxerr=max(float((original['model'][k]-resumed['model'][k]).abs().max()) for k in original['model'])
        assert maxerr<3e-6 if device.type=='cuda' else maxerr==0.0
        assert ms.state_tree_hash(original['rng'])==ms.state_tree_hash(resumed['rng'])
        for name in original['optimizer']['tempers']:
            assert torch.allclose(original['optimizer']['tempers'][name]['covariance'],resumed['optimizer']['tempers'][name]['covariance'],atol=2e-6,rtol=2e-5)
        # A different training contract must not overwrite an existing result.
        try:e.train(root,'development',127,'temper_cap3',.0008,data,device,test_cfg=cfg,test_root=d,test_steps=206)
        except RuntimeError as ex:assert 'Contract changed' in str(ex)
        else:raise AssertionError('Contract change accepted')
        dev=e.EvaluationPartition(data_dir/'val.bin',0,4,16)
        report=e.EvaluationPartition(data_dir/'val.bin',500,4,16)
        assert dev.stop<report.offset
    print('[PASS] real tiny-engine three policies; identical initialization; interrupted/resumed state; changed-contract refusal; disjoint evaluation windows')

def selector_gate():
    from select_recipes import choose
    with tempfile.TemporaryDirectory(prefix='fair_selector_test_') as tmp:
        for seed in p.DEV_SEEDS:
            for method in p.METHODS:
                for lr in p.INITIAL_LRS:
                    d=p.trial_dir(tmp,'development',seed,method,lr);d.mkdir(parents=True)
                    ct=dict(protocol_sha=p.PROTOCOL_SHA,method=method,seed=seed,lr=lr)
                    p.atomic_json(d/'initialization.json',dict(seed=seed))
                    p.atomic_json(d/'result.json',dict(contract=ct,step=p.STEPS,state='complete',report=None,dev=dict(loss=4+(lr-.0004)**2)))
        s=choose(tmp);assert s['state']=='locked' and all(v==.0004 for v in s['selection']['chosen_lr'].values())
        assert choose(tmp)['selection']==s['selection']
    with tempfile.TemporaryDirectory(prefix='fair_edge_test_') as tmp:
        for seed in p.DEV_SEEDS:
            for method in p.METHODS:
                for lr in p.INITIAL_LRS:
                    d=p.trial_dir(tmp,'development',seed,method,lr);d.mkdir(parents=True)
                    ct=dict(protocol_sha=p.PROTOCOL_SHA,method=method,seed=seed,lr=lr)
                    p.atomic_json(d/'initialization.json',dict(seed=seed))
                    p.atomic_json(d/'result.json',dict(contract=ct,step=p.STEPS,state='complete',report=None,dev=dict(loss=4-lr)))
        s=choose(tmp);assert s['state']=='extension_required'
        assert not (p.campaign(tmp)/'selection.lock.json').exists()
        for seed in p.DEV_SEEDS:
            for method in p.METHODS:
                for lr in p.EDGE_LRS:
                    d=p.trial_dir(tmp,'development',seed,method,lr);d.mkdir(parents=True)
                    p.atomic_json(d/'initialization.json',dict(seed=seed))
                    p.atomic_json(d/'result.json',dict(contract=dict(protocol_sha=p.PROTOCOL_SHA,method=method,seed=seed,lr=lr),step=p.STEPS,state='complete',report=None,dev=dict(loss=4-lr)))
        assert choose(tmp)['state']=='unbracketed'
    print('[PASS] per-method DEV-only selection, immutable lock, symmetric boundary extension, no reporting on an unbracketed grid')

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--cuda',action='store_true');x=a.parse_args()
    torch.set_num_threads(2 if not x.cuda else 4)
    if x.cuda and not torch.cuda.is_available():raise RuntimeError('CUDA gate requires GPU')
    if x.cuda and not torch.cuda.is_bf16_supported():raise RuntimeError('CUDA BF16 unsupported')
    dev=torch.device('cuda:0' if x.cuda else 'cpu')
    import queue
    assert hasattr(queue,'Queue')
    math_gate(dev);identity_gate(dev);workflow_gate(dev);selector_gate()
    print('FAIR OPTIMIZER CUDA GATE: PASS' if x.cuda else 'FAIR OPTIMIZER CPU TESTS: PASS')
